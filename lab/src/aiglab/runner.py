"""Run orchestration: preflight → load → identity → warmups → measured repetitions → seal.

The runner never decides an evidence class. It records what happened (origin, adapter,
identity checks, every request) and leaves classification to ``aiglab.evidence``.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import statistics
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiglab import __version__
from aiglab.adapters.base import ChatRequest, ChatResult, RuntimeAdapter
from aiglab.adapters.ollama import parse_runtime_memory
from aiglab.adapters.registry import RUNTIMES, get_adapter
from aiglab.aggregate import aggregate_run
from aiglab.artifacts import seal, write_json
from aiglab.host import capture_host, memory_pressure
from aiglab.manifest import Manifest
from aiglab.methodology import PRESSURE_BLOCK_LEVEL
from aiglab.sampler import MemorySampler
from aiglab.workloads import get_workload
from aiglab.workloads.base import TaskOutcome, Workload
from aiglab.workloads.coding import sandbox_available

ProgressFn = Callable[[dict[str, Any]], None]
#: One lock per host for every entry point (CLI run, queue, Estate), so benchmarks never overlap.
DEFAULT_LOCK = Path.home() / ".aiglab" / "run.lock"


@dataclass
class Check:
    name: str
    ok: bool
    blocking: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "blocking": self.blocking, "detail": self.detail}


@dataclass
class Preflight:
    checks: list[Check] = field(default_factory=list)
    identity: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str, blocking: bool = True) -> None:
        self.checks.append(Check(name, ok, blocking, detail))

    @property
    def blocked(self) -> bool:
        return any(c.blocking and not c.ok for c in self.checks)

    @property
    def block_reasons(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if c.blocking and not c.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "block_reasons": self.block_reasons,
            "checks": [c.as_dict() for c in self.checks],
            "runtime": self.runtime,
            "identity": self.identity,
        }


def make_adapter(manifest: Manifest) -> RuntimeAdapter:
    kwargs: dict[str, Any] = {}
    if manifest.runtime.id == "ollama" and manifest.runtime.endpoint:
        kwargs["endpoint"] = manifest.runtime.endpoint
    return get_adapter(manifest.runtime.id, **kwargs)


def preflight(
    manifest: Manifest,
    adapter: RuntimeAdapter | None,
    *,
    allow_resident: bool = False,
    pressure_reader: Callable[[], dict[str, Any]] = memory_pressure,
) -> Preflight:
    pf = Preflight()
    if manifest.target.kind == "cloud":
        pf.add(
            "target",
            False,
            "cloud targets are not executed by this harness: no cloud adapter exists and provisioning paid "
            "infrastructure needs explicit authority",
        )
        return pf
    pf.add("target", True, "local host")
    info = RUNTIMES.get(manifest.runtime.id)
    if info is None or not info.implemented or adapter is None:
        pf.add("runtime", False, f"no adapter implemented for runtime {manifest.runtime.id!r}")
        return pf
    status = adapter.detect()
    pf.runtime = status.as_dict()
    pf.add("runtime", status.usable, status.reason or f"{status.id} {status.version} reachable")
    if not status.usable:
        return pf

    identity = adapter.model_identity(manifest)
    pf.identity = identity
    if not identity.get("present"):
        pf.add("model_present", False, str(identity.get("reason") or "model not available"))
        return pf
    pf.add("model_present", True, f"{manifest.model.runtime_ref} present ({identity.get('api_digest')})")

    if info.test_only:
        pf.add(
            "test_only_runtime", True, "fixture runtime: output is test data and can never be evidence", blocking=False
        )
        return pf

    declared = manifest.model.quantization
    reported = identity.get("quantization_level")
    if declared:
        pf.add(
            "quantization",
            bool(reported) and str(reported).lower() == declared.lower(),
            f"declared {declared!r}, runtime reports {reported!r}",
        )
    else:
        pf.add("quantization", True, f"not declared; runtime reports {reported!r}", blocking=False)

    trained = identity.get("trained_context_length")
    if isinstance(trained, int):
        pf.add(
            "context_supported",
            manifest.context <= trained,
            f"requested {manifest.context}, model trained for {trained}",
        )
    else:
        pf.add("context_supported", True, "trained context length not reported by runtime", blocking=False)

    pin = manifest.model.expected_artifact_digest
    actual = identity.get("artifact_digest")
    if pin:
        pf.add("artifact_pin", pin == actual, f"manifest pins {pin}, runtime artifact is {actual}")
    pf.add(
        "artifact_on_disk",
        bool(identity.get("artifact_path")) and Path(str(identity["artifact_path"])).is_file(),
        f"{identity.get('artifact_path')}",
    )

    caps = identity.get("capabilities") or []
    workload = get_workload(manifest.workload_id)
    for cap in workload.requires_capabilities():
        pf.add(
            f"capability_{cap}",
            cap in caps,
            f"model advertises {caps}; a missing capability is recorded as a measured failure, not skipped",
            blocking=False,
        )
    if manifest.workload_id == "coding":
        pf.add("coding_sandbox", sandbox_available(), "sandbox-exec needed to execute model code", blocking=True)

    others = [
        m
        for m in adapter.resident_models()
        if m not in (manifest.model.runtime_ref, f"{manifest.model.runtime_ref}:latest")
    ]
    if others:
        pf.add(
            "no_contention",
            allow_resident,
            f"other models resident in the runtime: {others}",
            blocking=not allow_resident,
        )
    else:
        pf.add("no_contention", True, "no other models resident")

    pressure = pressure_reader()
    level = (pressure.get("pressure_level") or {}).get("value")
    if isinstance(level, int):
        pf.add(
            "memory_pressure",
            level < PRESSURE_BLOCK_LEVEL,
            f"kern.memorystatus_vm_pressure_level={level} (1 normal, 2 warn, 4 critical); free "
            f"{(pressure.get('free_percent') or {}).get('value')}%",
            blocking=level >= PRESSURE_BLOCK_LEVEL,
        )
    return pf


@contextlib.contextmanager
def host_lock(path: Path) -> Iterator[bool]:
    """One benchmark per host at a time: concurrent runs contaminate each other's memory and speed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        fh.write(str(os.getpid()))
        fh.flush()
        yield True
    finally:
        fh.close()


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def rep_metrics(
    outcomes: list[TaskOutcome], wall_s: float, memory: dict[str, Any], workload: Workload
) -> dict[str, Any]:
    reqs = [q for o in outcomes for q in o.requests]
    ok = [q for q in reqs if q.get("ok")]
    # TTFT only from responses whose first chunk was streamed content: tool calls may arrive complete.
    ttft = [
        float(q["ttft_ms"])
        for q in ok
        if isinstance(q.get("ttft_ms"), (int, float)) and q.get("first_chunk_kind") == "content"
    ]
    prompt_eval = [float(q["prompt_eval_ms"]) for q in ok if isinstance(q.get("prompt_eval_ms"), (int, float))]
    decode = [float(q["decode_tokens_per_s"]) for q in ok if isinstance(q.get("decode_tokens_per_s"), (int, float))]
    tokens = sum(int(q.get("completion_tokens") or 0) for q in ok)
    prompt_max = max((int(q.get("prompt_tokens") or 0) for q in ok), default=None)
    tools = [t.latency_ms for o in outcomes for t in o.tool_calls]
    m: dict[str, Any] = {
        "ttft_ms": _median(ttft),
        "ttft_samples": len(ttft),
        "prompt_eval_ms": _median(prompt_eval),
        "decode_tokens_per_s": _median(decode),
        "aggregate_output_tokens_per_s": (tokens / wall_s) if wall_s > 0 and tokens else None,
        "task_completion_s": _median([o.wall_ms / 1000 for o in outcomes]),
        "rep_wall_s": wall_s,
        "prompt_tokens_max": prompt_max,
        "prompt_cached_tokens_total": sum(int(q.get("prompt_cached_tokens") or 0) for q in ok),
        "success_rate": (sum(1 for o in outcomes if o.success) / len(outcomes)) if outcomes else None,
        "tasks_total": len(outcomes),
        "tasks_succeeded": sum(1 for o in outcomes if o.success),
        "requests_total": len(reqs),
        "requests_failed": len(reqs) - len(ok),
        "tool_calls": len(tools),
        "tool_latency_ms": _median(tools),
        "peak_rss_bytes": memory["peak_rss"]["value_bytes"],
        "peak_phys_footprint_bytes": memory["peak_phys_footprint"]["value_bytes"],
    }
    if not ttft:
        m["ttft_ms_reason"] = (
            "no response streamed content first (tool-call responses arrive complete, so first-token "
            "latency is not observable for them)"
        )
    if not prompt_eval:
        m["prompt_eval_ms_reason"] = "runtime reported no prompt-processing timing"
    if not decode:
        m["decode_tokens_per_s_reason"] = "runtime reported no decode timing"
    if not tools:
        m["tool_latency_ms_reason"] = (
            "not applicable: workload has no tool or retrieval step"
            if workload.tool_latency_meaning is None
            else "no tool call was made"
        )
    for key in ("peak_rss", "peak_phys_footprint"):
        if memory[key]["value_bytes"] is None:
            m[f"{key}_bytes_reason"] = memory[key].get("reason")
    return m


@dataclass
class RunOutcome:
    status: str  # completed | aborted | blocked
    run_dir: Path | None
    run_id: str
    preflight: Preflight
    error: str | None = None
    aggregate: dict[str, Any] | None = None


def new_run_id(manifest: Manifest, now: datetime | None = None) -> str:
    ts = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{manifest.id[:48]}-{manifest.short_hash[:8]}"


def execute(
    manifest: Manifest,
    runs_root: Path,
    *,
    adapter: RuntimeAdapter | None = None,
    repo_root: Path | None = None,
    allow_resident: bool = False,
    lock_path: Path | None = None,
    progress: ProgressFn | None = None,
    pressure_reader: Callable[[], dict[str, Any]] = memory_pressure,
    host_capture: Callable[[], dict[str, Any]] | None = None,
) -> RunOutcome:
    say = progress or (lambda _e: None)
    if adapter is None:
        try:
            adapter = make_adapter(manifest)
        except NotImplementedError:
            adapter = None
    run_id = new_run_id(manifest)
    pf = preflight(manifest, adapter, allow_resident=allow_resident, pressure_reader=pressure_reader)
    if pf.blocked or adapter is None:
        return RunOutcome("blocked", None, run_id, pf, error="; ".join(pf.block_reasons))

    lock = lock_path or DEFAULT_LOCK
    with host_lock(lock) as acquired:
        if not acquired:
            pf.add("host_lock", False, f"another aiglab run holds {lock}")
            return RunOutcome("blocked", None, run_id, pf, error="another aiglab run is in progress")
        return _execute_locked(manifest, runs_root, adapter, run_id, pf, repo_root, say, pressure_reader, host_capture)


def _execute_locked(
    manifest: Manifest,
    runs_root: Path,
    adapter: RuntimeAdapter,
    run_id: str,
    pf: Preflight,
    repo_root: Path | None,
    say: ProgressFn,
    pressure_reader: Callable[[], dict[str, Any]],
    host_capture: Callable[[], dict[str, Any]] | None,
) -> RunOutcome:
    runs_root.mkdir(parents=True, exist_ok=True)
    partial = runs_root / f".{run_id}.partial"
    partial.mkdir()
    workload = get_workload(manifest.workload_id)
    host = host_capture() if host_capture else capture_host(repo_root)
    record: dict[str, Any] = {
        "schema": "aiglab.run/1",
        "run_id": run_id,
        "origin": "executed" if adapter.executes_real_runtime else "fixture",
        "status": "running",
        "methodology_version": manifest.methodology_version,
        "harness_version": __version__,
        "adapter": {
            "id": adapter.id,
            "version": adapter.adapter_version,
            "executes_real_runtime": adapter.executes_real_runtime,
        },
        "manifest_id": manifest.id,
        "manifest_sha256": manifest.sha256,
        "model_declared": dict(manifest.raw.get("model") or {}),
        "target_declared": dict(manifest.raw.get("target") or {}),
        "workload": workload.describe(),
        "config": {
            "context": manifest.context,
            "concurrency": manifest.concurrency,
            "max_output_tokens": manifest.max_output_tokens,
            "temperature": manifest.temperature,
            "seed": manifest.seed,
            "runtime_options": dict(manifest.runtime.options),
        },
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "preflight": pf.as_dict(),
        "host_at_start": pressure_reader(),
        "hardware_profile_key": host.get("hardware_profile_key"),
        "repetitions": [],
        "identity_checks": {},
        "warnings": [c.as_dict() for c in pf.checks if not c.ok and not c.blocking],
    }
    write_json(partial / "manifest.json", dict(manifest.raw))
    write_json(partial / "host.json", host)
    sampler_samples: dict[str, Any] = {}
    marker = adapter.log_marker()
    error: str | None = None
    was_resident = True  # until known, assume someone else's model: never unload it
    try:
        identity = pf.identity
        if adapter.executes_real_runtime:
            say({"kind": "aiglab_progress", "step": "verify_artifact"})
            verify = getattr(adapter, "verify_artifact", None)
            record["identity_checks"]["artifact_hash"] = (
                verify(identity) if verify else {"ok": False, "detail": "adapter cannot verify its artifact"}
            )
        was_resident = manifest.model.runtime_ref in adapter.resident_models()
        record["model_was_resident_before_run"] = was_resident
        workload.setup()
        say({"kind": "aiglab_progress", "step": "load"})
        load = adapter.load(manifest)
        record["load"] = {**load.as_dict(), "cold": not was_resident}
        if not load.ok:
            raise RuntimeError(f"model load failed: {load.error}")
        state = adapter.loaded_state(manifest)
        record["loaded_state_start"] = state.__dict__
        if adapter.executes_real_runtime:
            record["identity_checks"].update(_identity_checks(manifest, identity, state))
        pid = state.sample_pid

        run_tag = manifest_hash_tag(run_id)

        def run_lane(n: int, lane: int) -> list[TaskOutcome]:
            chat = session_chat(adapter, manifest, f"{run_tag}-r{n}-l{lane}")
            return [workload.execute(t, chat, lane) for t in workload.tasks()]

        phases = ["warmup"] * manifest.warmup + ["measured"] * manifest.measured
        measured_index = 0
        for n, phase in enumerate(phases):
            say({"kind": "aiglab_progress", "step": "repetition", "n": n + 1, "of": len(phases), "phase": phase})
            sampler = MemorySampler(pid)
            t0 = time.perf_counter()
            with sampler, ThreadPoolExecutor(max_workers=manifest.concurrency) as pool:
                lanes = [pool.submit(run_lane, n, lane) for lane in range(manifest.concurrency)]
                outcomes = [o for f in lanes for o in f.result()]
            wall_s = time.perf_counter() - t0
            memory = sampler.summary()
            rep = {
                "phase": phase,
                "index": measured_index if phase == "measured" else -(n + 1),
                "wall_s": wall_s,
                "memory": memory,
                "metrics": rep_metrics(outcomes, wall_s, memory, workload),
                "outcomes": [o.as_dict() for o in outcomes],
            }
            if phase == "measured":
                measured_index += 1
            record["repetitions"].append(rep)
            sampler_samples[f"{phase}-{rep['index']}"] = sampler.samples()
        end_state = adapter.loaded_state(manifest)
        record["loaded_state_end"] = end_state.__dict__
        if adapter.executes_real_runtime:
            record["identity_checks"]["no_reload_during_run"] = {
                "ok": end_state.sample_pid is not None and end_state.sample_pid == pid,
                "detail": f"serving pid at start {pid}, at end {end_state.sample_pid}",
            }
            record["identity_checks"]["no_contention_during_run"] = {
                "ok": not end_state.other_resident_models,
                "detail": f"other resident models at end: {end_state.other_resident_models}",
            }
        record["status"] = "completed"
    except Exception as exc:  # recorded, not swallowed: the run becomes an aborted (unknown) record
        error = f"{type(exc).__name__}: {exc}"
        record["status"] = "aborted"
        record["error"] = error
    finally:
        workload.teardown()
        record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        # Restore the host: release the model only if this run loaded it. A model that was already
        # resident belongs to someone else and is left alone.
        record["unloaded_after_run"] = adapter.unload(manifest) if not was_resident else False
        excerpt = adapter.log_excerpt(marker)
        if excerpt:
            (partial / "runtime-log-excerpt.txt").write_text(excerpt + "\n", encoding="utf-8")
        record["runtime_reported_memory"] = {
            "source": "runtime log lines written during the run (runtime-reported, not OS-measured)",
            **parse_runtime_memory(excerpt),
        }
        write_json(partial / "memory-samples.json", sampler_samples)
        write_json(partial / "run.json", record)
        digest = seal(partial)
        final = runs_root / run_id
        os.replace(partial, final)
    agg = aggregate_run(record, digest)
    write_json(final / "aggregate.json", agg)
    say({"kind": "aiglab_progress", "step": "done", "status": record["status"], "run_id": run_id})
    return RunOutcome(record["status"], final, run_id, pf, error=error, aggregate=agg)


def manifest_hash_tag(run_id: str) -> str:
    return hashlib.sha256(run_id.encode()).hexdigest()[:8]


def session_chat(adapter: RuntimeAdapter, manifest: Manifest, session: str) -> Callable[[ChatRequest], ChatResult]:
    """A chat function whose prompts start with a per-(run, repetition, lane) session line.

    Runtimes reuse the KV cache for a shared prompt prefix. Without this, every repetition after the
    first would replay an identical prompt and measure cached-prefix TTFT instead of prompt
    processing. The session line sits at the very start, so nothing is reused *across*
    repetitions or lanes, while turns *within* one conversation still share their prefix exactly as
    a real agent's would. Cached-token counts reported by the runtime are recorded per request.
    """

    def chat(req: ChatRequest) -> ChatResult:
        messages = list(req.messages)
        if messages:
            first = dict(messages[0])
            first["content"] = f"[session {session}]\n{first.get('content') or ''}"
            messages[0] = first
        return adapter.chat(manifest, ChatRequest(id=req.id, messages=messages, tools=req.tools, format=req.format))

    return chat


def _identity_checks(manifest: Manifest, identity: dict[str, Any], state: Any) -> dict[str, Any]:
    return {
        "manifest_digest": {
            "ok": bool(identity.get("manifest_file_sha256"))
            and identity.get("manifest_file_sha256") == identity.get("api_digest"),
            "detail": f"api {identity.get('api_digest')} vs manifest file {identity.get('manifest_file_sha256')}",
        },
        "serving_requested_artifact": {
            "ok": bool(state.artifact_sha256_from_runtime)
            and state.artifact_sha256_from_runtime == identity.get("artifact_digest"),
            "detail": f"serving process loads {state.artifact_sha256_from_runtime}; model layer {identity.get('artifact_digest')}",
        },
        "effective_context": {
            "ok": state.effective_context == manifest.context,
            "detail": f"runtime reports {state.effective_context}, manifest requests {manifest.context}",
        },
        "model_resident": {"ok": bool(state.model_resident), "detail": "model listed by the runtime as loaded"},
        "memory_sampling_target": {
            "ok": state.sample_pid is not None,
            "detail": f"{state.sample_process} pid {state.sample_pid}",
        },
    }
