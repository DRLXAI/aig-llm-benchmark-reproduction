"""Adapter integration against a fake Ollama API, runner behaviour, sampler and host capture."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from aiglab.adapters.base import ChatRequest
from aiglab.adapters.fixture import FixtureAdapter
from aiglab.adapters.ollama import OllamaAdapter, parse_runtime_memory, split_ref
from aiglab.artifacts import RAW_REQUIRED, SEAL, read_json, verify_seal
from aiglab.host import capture_host
from aiglab.normalize import normalize_run
from aiglab.runner import execute, host_lock, preflight, rep_metrics
from aiglab.sampler import MemorySampler
from aiglab.workloads import get_workload
from aiglab.workloads.base import TaskOutcome
from tests.conftest import make_manifest


class FakeOllama:
    """Speaks enough of the Ollama API for the adapter. A mock runtime: it must never yield aig_tested."""

    def __init__(self, models_dir: Path) -> None:
        blob = b"GGUF-fake-weights"
        blob_digest = "sha256:" + hashlib.sha256(blob).hexdigest()
        (models_dir / "blobs").mkdir(parents=True)
        (models_dir / "blobs" / blob_digest.replace(":", "-")).write_bytes(blob)
        manifest = json.dumps(
            {"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": blob_digest, "size": len(blob)}]}
        ).encode()
        mpath = models_dir / "manifests/registry.ollama.ai/library/qwen2.5/7b-instruct"
        mpath.parent.mkdir(parents=True)
        mpath.write_bytes(manifest)
        digest = hashlib.sha256(manifest).hexdigest()
        self.loaded = False
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, data: Any) -> None:
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/api/version":
                    self._json({"version": "0.35.0"})
                elif self.path == "/api/tags":
                    self._json({"models": [{"name": "qwen2.5:7b-instruct", "digest": digest, "size": 4683087332}]})
                elif self.path == "/api/ps":
                    models = (
                        [
                            {
                                "name": "qwen2.5:7b-instruct",
                                "digest": digest,
                                "size": 5119263374,
                                "size_vram": 5119263374,
                                "context_length": 8192,
                            }
                        ]
                        if fake.loaded
                        else []
                    )
                    self._json({"models": models})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length))
                if self.path == "/api/show":
                    self._json(
                        {
                            "details": {
                                "format": "gguf",
                                "family": "qwen2",
                                "parameter_size": "7.6B",
                                "quantization_level": "Q4_K_M",
                            },
                            "model_info": {"general.parameter_count": 7615616512, "qwen2.context_length": 32768},
                            "capabilities": ["completion", "tools"],
                        }
                    )
                    return
                fake.loaded = True
                self.send_response(200)
                self.end_headers()
                chunks: list[dict[str, Any]]
                if req.get("tools"):
                    chunks = [
                        {
                            "message": {
                                "tool_calls": [{"function": {"name": "fetch_page", "arguments": {"url": "x"}}}]
                            },
                            "done": False,
                        }
                    ]
                else:
                    chunks = [
                        {"message": {"content": "Hello"}, "done": False},
                        {"message": {"content": " world"}, "done": False},
                    ]
                chunks.append(
                    {
                        "message": {"content": ""},
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 30,
                        "prompt_eval_cached_count": 4,
                        "prompt_eval_duration": 50_000_000,
                        "eval_count": 20,
                        "eval_duration": 500_000_000,
                        "load_duration": 1_000_000,
                        "total_duration": 560_000_000,
                    }
                )
                for c in chunks:
                    self.wfile.write(json.dumps(c).encode() + b"\n")

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def fake_ollama(tmp_path: Path) -> Iterator[FakeOllama]:
    fake = FakeOllama(tmp_path / "models")
    yield fake
    fake.server.shutdown()


def adapter_for(fake: FakeOllama, tmp_path: Path) -> OllamaAdapter:
    return OllamaAdapter(
        endpoint=fake.endpoint, models_dir=tmp_path / "models", log_path=tmp_path / "no.log", process_lister=lambda: []
    )


def test_adapter_parses_stream_and_timings(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    adapter = adapter_for(fake_ollama, tmp_path)
    m = make_manifest(runtime__id="ollama")
    r = adapter.chat(m, ChatRequest(id="x", messages=[{"role": "user", "content": "hi"}]))
    assert r.ok and r.content == "Hello world" and r.first_chunk_kind == "content"
    assert (r.prompt_tokens, r.prompt_cached_tokens, r.completion_tokens) == (30, 4, 20)
    assert r.decode_tokens_per_s == pytest.approx(40.0)
    assert r.ttft_ms is not None and r.wall_ms is not None and r.ttft_ms <= r.wall_ms
    t = adapter.chat(m, ChatRequest(id="y", messages=[{"role": "user", "content": "hi"}], tools=[{"type": "function"}]))
    assert t.first_chunk_kind == "tool_calls" and t.tool_calls


def test_adapter_identity_chain_from_disk(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    adapter = adapter_for(fake_ollama, tmp_path)
    ident = adapter.model_identity(make_manifest(runtime__id="ollama"))
    assert ident["api_digest"] == ident["manifest_file_sha256"]
    assert adapter.verify_artifact(ident)["ok"] is True


def test_mock_runtime_never_yields_aig_tested(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    """A real adapter pointed at a fake API completes a run, but without a serving process that
    loads the verified artifact the identity checks fail and the run is unknown."""
    adapter = adapter_for(fake_ollama, tmp_path)
    m = make_manifest(
        runtime__id="ollama",
        runtime__endpoint=fake_ollama.endpoint,
        workload__id="research",
        repetitions__warmup=0,
        repetitions__measured=1,
    )
    outcome = execute(
        m,
        tmp_path / "runs",
        adapter=adapter,
        lock_path=tmp_path / "lock",
        host_capture=lambda: {"hardware_profile_key": "test"},
        pressure_reader=lambda: {"pressure_level": {"value": 1}},
    )
    assert outcome.status == "completed"
    run = read_json(outcome.run_dir / "run.json")
    assert run["origin"] == "executed"
    failed = {k for k, v in run["identity_checks"].items() if not v["ok"]}
    assert {"serving_requested_artifact", "memory_sampling_target"} <= failed
    assert {r["evidence_type"] for r in normalize_run(outcome.run_dir)} == {"unknown"}


def test_fixture_run_writes_sealed_layout(run_fixture: Any) -> None:
    outcome = run_fixture("rag")
    names = {p.name for p in outcome.run_dir.iterdir()}
    assert set(RAW_REQUIRED) | {SEAL, "aggregate.json", "memory-samples.json"} <= names
    assert verify_seal(outcome.run_dir) == (True, [])
    run = read_json(outcome.run_dir / "run.json")
    assert run["origin"] == "fixture" and run["status"] == "completed"
    assert len([r for r in run["repetitions"] if r["phase"] == "measured"]) == 3
    assert outcome.aggregate["task_success"]["tasks_succeeded"] == 9


def test_session_prefix_isolates_repetitions(run_fixture: Any) -> None:
    seen: list[str] = []

    def respond(req: ChatRequest) -> Any:
        from tests.conftest import reference_responder

        if req.messages:  # the load request has no messages
            seen.append(req.messages[0]["content"].splitlines()[0])
        return reference_responder(req)

    run_fixture("research", responder=respond)
    prefixes = set(seen)
    assert len(prefixes) == 4  # 1 warmup + 3 measured repetitions, one lane each
    assert all(p.startswith("[session ") for p in prefixes)


def test_blocked_runs_execute_nothing(tmp_path: Path) -> None:
    adapter = FixtureAdapter()
    cloud = make_manifest(target__kind="cloud", target__provider="runpod")
    out = execute(cloud, tmp_path / "runs", adapter=adapter, lock_path=tmp_path / "lock")
    assert out.status == "blocked" and "paid infrastructure" in out.error
    assert adapter.requests == [] and not (tmp_path / "runs").exists()
    unimplemented = make_manifest(runtime__id="llama_cpp")
    assert execute(unimplemented, tmp_path / "runs", lock_path=tmp_path / "lock").status == "blocked"


def test_host_lock_blocks_a_second_run(tmp_path: Path) -> None:
    lock = tmp_path / "lock"
    with host_lock(lock) as held:
        assert held
        out = execute(make_manifest(), tmp_path / "runs", adapter=FixtureAdapter(), lock_path=lock)
        assert out.status == "blocked" and "another aiglab run" in (out.error or "")


def test_contention_blocks_unless_allowed(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    adapter = adapter_for(fake_ollama, tmp_path)
    adapter.resident_models = lambda: ["gemma4:12b"]  # type: ignore[method-assign]
    m = make_manifest(runtime__id="ollama", workload__id="research")
    pressure = lambda: {"pressure_level": {"value": 1}}  # noqa: E731
    assert preflight(m, adapter, pressure_reader=pressure).blocked
    assert not preflight(m, adapter, allow_resident=True, pressure_reader=pressure).blocked


def test_critical_memory_pressure_blocks(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    pf = preflight(
        make_manifest(runtime__id="ollama"),
        adapter_for(fake_ollama, tmp_path),
        pressure_reader=lambda: {"pressure_level": {"value": 4}},
    )
    assert pf.blocked and any("memory_pressure" in r for r in pf.block_reasons)


def test_quantization_mismatch_blocks(fake_ollama: FakeOllama, tmp_path: Path) -> None:
    pf = preflight(
        make_manifest(runtime__id="ollama", model__quantization="Q8_0"),
        adapter_for(fake_ollama, tmp_path),
        pressure_reader=lambda: {"pressure_level": {"value": 1}},
    )
    assert any(r.startswith("quantization") for r in pf.block_reasons)


def test_sampler_peaks_and_unavailable() -> None:
    values = iter([100, 300, 200] + [200] * 100)
    with MemorySampler(1, rss_reader=lambda _pid: next(values), footprint_reader=lambda _pid: 50) as s:
        pass
    summary = s.summary()
    assert summary["peak_rss"]["status"] == "measured" and summary["peak_rss"]["value_bytes"] >= 100
    none = MemorySampler(None)
    with none:
        pass
    assert none.summary()["peak_rss"] == {
        "value_bytes": None,
        "status": "unavailable",
        "reason": "no sampling target process",
        "method": "ps -o rss= (resident set incl. mapped weights)",
    }


def test_ttft_ignores_tool_call_first_responses() -> None:
    outcome = TaskOutcome(
        task_id="t",
        lane=0,
        success=True,
        failure_reason=None,
        wall_ms=1000.0,
        requests=[
            {"ok": True, "ttft_ms": 900.0, "first_chunk_kind": "tool_calls", "decode_tokens_per_s": 40.0},
            {"ok": True, "ttft_ms": 120.0, "first_chunk_kind": "content", "decode_tokens_per_s": 42.0},
        ],
    )
    mem = {"peak_rss": {"value_bytes": 1}, "peak_phys_footprint": {"value_bytes": 1}}
    m = rep_metrics([outcome], 1.0, mem, get_workload("research"))
    assert m["ttft_ms"] == 120.0 and m["ttft_samples"] == 1


def test_parse_runtime_memory() -> None:
    log = (
        "llama_kv_cache:       MTL0 KV buffer size =   448.00 MiB\n"
        "load_tensors:  MTL0_Mapped model buffer size =  4168.09 MiB\n"
        "sched_reserve:       MTL0 compute buffer size =   266.02 MiB\n"
    )
    parsed = parse_runtime_memory(log)
    assert parsed["kv_cache_mib"] == 448.0 and parsed["model_buffers_mib"] == {"MTL0_Mapped": 4168.09}
    assert parse_runtime_memory(None) == {}


def test_split_ref() -> None:
    assert split_ref("qwen2.5:7b-instruct") == ("library", "qwen2.5", "7b-instruct")
    assert split_ref("me/model") == ("me", "model", "latest")


def test_host_capture_is_deterministic_and_private() -> None:
    answers = {
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4 Pro\n",
        ("sysctl", "-n", "hw.memsize"): "25769803776\n",
        ("sysctl", "-n", "kern.memorystatus_vm_pressure_level"): "1\n",
        ("pmset", "-g", "batt"): "Now drawing from 'AC Power'\n",
    }

    def run(argv: Sequence[str]) -> str | None:
        return answers.get(tuple(argv))

    host = capture_host(None, run)
    if host["os"]["system"] != "Darwin":
        pytest.skip("macOS capture path")
    assert host["hardware_profile_key"] == "apple-m4-pro-24gb"
    assert host["power_draw"]["value"] is None and "root" in host["power_draw"]["reason"]
    assert host["power_source"]["value"] == "AC Power"
    text = json.dumps(host).lower()
    import socket

    assert socket.gethostname().lower() not in text
