"""Shared test helpers.

`synthetic_executed_run` hand-builds a run directory that *looks* executed so the payload,
dedupe and index code paths that only accept aig_tested input can be exercised. It exists only in
pytest tmp dirs, its run ids start with TESTONLY-, and test_evidence proves that nothing the
fixture adapter produces can reach this shape through the harness.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from aiglab.adapters.base import ChatRequest, ChatResult
from aiglab.adapters.fixture import FixtureAdapter
from aiglab.aggregate import aggregate_run
from aiglab.artifacts import seal, write_json
from aiglab.manifest import Manifest, parse_manifest
from aiglab.runner import RunOutcome, execute
from aiglab.workloads import WORKLOADS

BASE_MANIFEST: dict[str, Any] = {
    "manifest_version": 1,
    "id": "test-manifest",
    "methodology_version": "1.0.0",
    "model": {
        "family": "Qwen2.5",
        "name": "Qwen2.5 7B Instruct",
        "runtime_ref": "qwen2.5:7b-instruct",
        "quantization": "Q4_K_M",
    },
    "runtime": {"id": "fixture"},
    "target": {
        "kind": "local",
        "hardware_profile": "apple-m4-pro-24gb",
        "usable_memory_gb": 24,
        "memory_kind": "unified",
    },
    "config": {"context": 8192, "concurrency": 1, "max_output_tokens": 256, "temperature": 0, "seed": 42},
    "workload": {"id": "research", "version": "1.0.0"},
    "repetitions": {"warmup": 1, "measured": 3},
    "limits": {"request_timeout_s": 60},
    "notes": "test",
}


def manifest_dict(**overrides: Any) -> dict[str, Any]:
    raw = copy.deepcopy(BASE_MANIFEST)
    for dotted, value in overrides.items():
        node = raw
        parts = dotted.split("__")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return raw


def make_manifest(**overrides: Any) -> Manifest:
    return parse_manifest(manifest_dict(**overrides))


def reference_responder(request: ChatRequest) -> ChatResult:
    """Answers each task with its reference reply (correct output) — fixture origin."""
    replies = {t.id: t.fixture_reply for w in WORKLOADS.values() for t in w().tasks()}
    task_id = request.id.split("#")[0]
    return ChatResult(
        id=request.id,
        origin="fixture",
        ok=True,
        content=replies.get(task_id, "fixture"),
        done_reason="stop",
        ttft_ms=100.0,
        first_chunk_kind="content",
        wall_ms=400.0,
        prompt_tokens=80,
        completion_tokens=40,
        prompt_eval_ms=50.0,
        decode_ms=350.0,
    )


@pytest.fixture
def run_fixture(tmp_path: Path) -> Callable[..., RunOutcome]:
    def _run(
        workload: str = "research",
        responder: Callable[[ChatRequest], ChatResult] = reference_responder,
        **overrides: Any,
    ) -> RunOutcome:
        m = make_manifest(workload__id=workload, **overrides)
        return execute(
            m,
            tmp_path / "runs",
            adapter=FixtureAdapter(responder),
            lock_path=tmp_path / "run.lock",
            host_capture=lambda: {"hardware_profile_key": "fixture-host"},
            pressure_reader=lambda: {"pressure_level": {"value": 1}, "free_percent": {"value": 60}},
        )

    return _run


def _rep(index: int, phase: str, decode: float, rss: int, ttft: float = 250.0) -> dict[str, Any]:
    req = {
        "id": f"t#{index}",
        "origin": "executed",
        "ok": True,
        "ttft_ms": ttft,
        "first_chunk_kind": "content",
        "decode_tokens_per_s": decode,
        "completion_tokens": 100,
        "prompt_tokens": 300,
        "prompt_eval_ms": 90.0,
        "error": None,
    }
    return {
        "phase": phase,
        "index": index,
        "wall_s": 3.0,
        "memory": {},
        "metrics": {
            "ttft_ms": ttft,
            "prompt_eval_ms": 90.0,
            "decode_tokens_per_s": decode,
            "aggregate_output_tokens_per_s": decode,
            "task_completion_s": 2.5,
            "rep_wall_s": 3.0,
            "peak_rss_bytes": rss,
            "peak_phys_footprint_bytes": rss // 5,
            "tool_latency_ms": None,
            "tool_latency_ms_reason": "not applicable: workload has no tool or retrieval step",
            "prompt_tokens_max": 300,
            "success_rate": 1.0,
        },
        "outcomes": [
            {
                "task_id": "t",
                "lane": 0,
                "success": True,
                "failure_reason": None,
                "wall_ms": 2500.0,
                "requests": [req],
                "tool_calls": [],
            }
        ],
    }


def synthetic_executed_run(
    root: Path,
    *,
    run_id: str = "TESTONLY-20261002T000000Z-synthetic",
    context: int = 8192,
    concurrency: int = 1,
    workload: str = "coding",
    decodes: tuple[float, ...] = (40.0, 41.0, 40.5, 39.8, 40.2),
    rss: int = 5_000_000_000,
    started_at: str = "2026-10-02T12:00:00+00:00",
    runtime_ref: str = "qwen2.5:7b-instruct",
    hardware_key: str = "apple-m4-pro-24gb",
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    wl = WORKLOADS[workload].describe()
    run: dict[str, Any] = {
        "schema": "aiglab.run/1",
        "run_id": run_id,
        "origin": "executed",
        "status": "completed",
        "methodology_version": "1.0.0",
        "harness_version": "1.0.0",
        "adapter": {"id": "ollama", "version": "1.0.0", "executes_real_runtime": True},
        "manifest_id": "synthetic",
        "model_declared": {"family": "Qwen2.5", "name": "Qwen2.5 7B Instruct", "runtime_ref": runtime_ref},
        "workload": wl,
        "config": {"context": context, "concurrency": concurrency, "max_output_tokens": 256},
        "started_at": started_at,
        "preflight": {
            "runtime": {"version": "0.35.0"},
            "identity": {
                "ref": runtime_ref,
                "artifact_digest": "sha256:" + "ab" * 32,
                "artifact_size_bytes": 4_683_073_952,
                "api_digest": "sha256:" + "cd" * 32,
                "quantization_level": "Q4_K_M",
                "parameter_size": "7.6B",
            },
        },
        "host_at_start": {"pressure_level": {"value": 1}},
        "hardware_profile_key": hardware_key,
        "loaded_state_start": {"server_parallel_slots": 1},
        "identity_checks": {
            name: {"ok": True, "detail": "synthetic"}
            for name in (
                "artifact_hash",
                "manifest_digest",
                "serving_requested_artifact",
                "effective_context",
                "model_resident",
                "memory_sampling_target",
                "no_reload_during_run",
                "no_contention_during_run",
            )
        },
        "repetitions": [_rep(-1, "warmup", 30.0, rss)] + [_rep(i, "measured", d, rss) for i, d in enumerate(decodes)],
    }
    if mutate:
        mutate(run)
    write_json(run_dir / "run.json", run)
    write_json(run_dir / "manifest.json", {"id": "synthetic"})
    write_json(
        run_dir / "host.json",
        {
            "hardware_profile_key": hardware_key,
            "cpu": {"brand": {"value": "Apple M4 Pro"}},
            "memory": {"total_bytes": {"value": 25769803776}, "kind": "unified"},
            "os": {"product": "macOS", "version": "15.5"},
        },
    )
    digest = seal(run_dir)
    write_json(run_dir / "aggregate.json", aggregate_run(run, digest))
    return run_dir


def read(path: Path) -> Any:
    return json.loads(path.read_text())
