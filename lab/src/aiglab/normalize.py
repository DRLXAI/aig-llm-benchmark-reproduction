"""Run directory → normalized observation records.

Each completed run yields one CompatibilityObservation and one BenchmarkObservation, shaped for
the Planner's Base44 entities (field names and enums from the live schema snapshot in
lab/base44/schema-snapshot/). The lab record keeps everything the Base44 entity cannot hold:
units, sample counts, ranges, unavailable-metric reasons and the evidence classification.

Units used in the Base44 fields (the entities do not declare most units; see
docs/measurement-lab/BASE44-SYNC.md):
  observed_peak_memory / peak_memory  decimal GB (1e9 bytes), max of per-repetition peak RSS
  observed_tokens_sec / tokens_sec    median per-request decode tokens/s (runtime-timed)
  ttft                                median ms, client-measured, streaming
  completion_time                     median seconds per task (model + agent tool steps)
  tool_latency                        median ms per tool/retrieval step
  failures                            count of failed tasks across measured repetitions
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from aiglab.aggregate import aggregate_run
from aiglab.artifacts import read_json, seal_digest, verify_seal
from aiglab.dedupe import config_identity, config_key, observation_key
from aiglab.evidence import Classification, assert_publishable, classify_run

SOURCE_URL_TEMPLATE = "https://github.com/DRLXAI/agent-infra-guide/tree/main/evidence/measurements/runs/{run_id}"
SOURCE_NAME = "AIG Measurement Lab (methodology {methodology}) run {run_id}"


def _gb(value: Any) -> float | None:
    return round(float(value) / 1e9, 2) if isinstance(value, (int, float)) else None


def _round(value: Any, digits: int) -> float | None:
    return round(float(value), digits) if isinstance(value, (int, float)) else None


def _metric(agg: Mapping[str, Any], name: str, stat: str = "median") -> float | None:
    m = (agg.get("metrics") or {}).get(name) or {}
    if m.get("status") != "measured":
        return None
    v = (m.get("included") or {}).get(stat)
    return float(v) if isinstance(v, (int, float)) else None


def model_lab_key(run: Mapping[str, Any]) -> str:
    identity = (run.get("preflight") or {}).get("identity") or {}
    manifest_ref = identity.get("ref") or run.get("manifest_id")
    digest = str(identity.get("artifact_digest") or identity.get("api_digest") or "unknown")
    return f"{run['adapter']['id']}/{manifest_ref}/{digest[:19]}"


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Classification, list[str]]:
    run = read_json(run_dir / "run.json")
    seal_ok, seal_problems = verify_seal(run_dir)
    classification = classify_run(run, seal_ok=seal_ok)
    agg_path = run_dir / "aggregate.json"
    digest = seal_digest(run_dir) if (run_dir / "SHA256SUMS").is_file() else None
    agg = read_json(agg_path) if agg_path.is_file() else aggregate_run(run, digest)
    problems = list(seal_problems)
    if agg.get("seal_digest") != digest:
        problems.append("aggregate.json was computed from different raw data; re-run `aiglab aggregate`")
        agg = aggregate_run(run, digest, agg.get("exclusions") or [])
    return run, agg, classification, problems


def unavailable_notes(agg: Mapping[str, Any], workload: Mapping[str, Any]) -> list[str]:
    notes = ["power: unavailable (power draw requires root via powermetrics; the harness never escalates)"]
    notes.append("cost_per_task: unavailable (local hardware has no observed per-task price; never estimated here)")
    for name in ("tool_latency_ms", "peak_rss_bytes", "ttft_ms", "decode_tokens_per_s"):
        m = (agg.get("metrics") or {}).get(name) or {}
        if m.get("status") == "unavailable":
            notes.append(f"{name}: unavailable ({'; '.join(m.get('reasons') or ['not observed'])})")
    if workload.get("tool_latency_meaning"):
        notes.append(f"tool_latency = {workload['tool_latency_meaning']}")
    return notes


def normalize_run(run_dir: Path) -> list[dict[str, Any]]:
    run, agg, classification, problems = load_run(run_dir)
    if problems:
        classification = Classification("unknown", classification.origin, (*classification.reasons, *problems))
    pf = run.get("preflight") or {}
    identity = pf.get("identity") or {}
    runtime = pf.get("runtime") or {}
    host = read_json(run_dir / "host.json") if (run_dir / "host.json").is_file() else {}
    start_state = run.get("loaded_state_start") or {}
    run_id = str(run["run_id"])
    tested_at = str(run.get("started_at") or "")[:10] or None
    methodology = str(run.get("methodology_version"))
    hw_key = str(run.get("hardware_profile_key") or host.get("hardware_profile_key") or "unknown-host")
    m_key = model_lab_key(run)
    workload = run.get("workload") or {}
    cfg = run.get("config") or {}
    digest = str(agg.get("seal_digest") or "unsealed")
    runtime_label = f"{run['adapter']['id']} {runtime.get('version') or 'unknown'}"
    peak_rss_max = _metric(agg, "peak_rss_bytes", "max")
    slots = start_state.get("server_parallel_slots")

    common_notes = [
        f"origin={classification.origin}",
        f"evidence={classification.evidence_type}"
        + (f" ({'; '.join(classification.reasons)})" if classification.reasons else ""),
        f"measured repetitions={agg.get('measured_repetitions')} warmups={agg.get('warmup_repetitions')}"
        + (f" exclusions={agg.get('exclusions')}" if agg.get("exclusions") else ""),
        f"server parallel slots={slots}"
        + (
            " (requests above this queue inside the runtime)"
            if isinstance(slots, int) and slots < int(cfg.get("concurrency") or 1)
            else ""
        ),
        f"prompt tokens max={_metric(agg, 'prompt_tokens_max', 'max')} of context {cfg.get('context')} (context = allocated window)",
        f"host memory pressure at start={((run.get('host_at_start') or {}).get('pressure_level') or {}).get('value')}",
        "units: memory decimal GB (max of per-repetition peak RSS of the serving process, sampled), "
        "tokens/s median decode rate, ttft ms, completion_time s",
    ]
    base = {
        "run_id": run_id,
        "seal_digest": digest,
        "origin": classification.origin,
        "evidence_type": classification.evidence_type,
        "evidence_reasons": list(classification.reasons),
        "methodology_version": methodology,
        "tested_at": tested_at,
        "confidence": agg.get("confidence", "unknown"),
        "confidence_reasons": agg.get("confidence_reasons", []),
        "model": {
            "lab_key": m_key,
            "family": (run.get("model_declared") or {}).get("family"),
            "name": (run.get("model_declared") or {}).get("name"),
            "variant": (run.get("model_declared") or {}).get("variant"),
            "runtime_ref": identity.get("ref"),
            "artifact_digest": identity.get("artifact_digest"),
            "artifact_size_bytes": identity.get("artifact_size_bytes"),
            "quantization": identity.get("quantization_level"),
            "parameter_size": identity.get("parameter_size"),
            "parameter_count": identity.get("parameter_count"),
            "trained_context_length": identity.get("trained_context_length"),
        },
        "hardware": {
            "lab_key": hw_key,
            "chip": ((host.get("cpu") or {}).get("brand") or {}).get("value"),
            "gpu_cores": ((host.get("gpu") or {}).get("cores") or {}).get("value"),
            "memory_total_bytes": ((host.get("memory") or {}).get("total_bytes") or {}).get("value"),
            "memory_kind": (host.get("memory") or {}).get("kind"),
            "os": " ".join(
                str(x) for x in ((host.get("os") or {}).get("product"), (host.get("os") or {}).get("version")) if x
            ),
        },
        "runtime": {"id": run["adapter"]["id"], "version": runtime.get("version"), "server_parallel_slots": slots},
        "context": cfg.get("context"),
        "concurrency": cfg.get("concurrency"),
        "metrics": agg.get("metrics"),
        "runtime_completion": agg.get("runtime_completion"),
        "task_success": agg.get("task_success"),
        "runtime_reported_memory": run.get("runtime_reported_memory"),
        "source_url": SOURCE_URL_TEMPLATE.format(run_id=run_id),
        "source_name": SOURCE_NAME.format(methodology=methodology, run_id=run_id),
    }
    records: list[dict[str, Any]] = []
    for entity in ("CompatibilityObservation", "BenchmarkObservation"):
        ident = config_identity(
            entity=entity,
            methodology_version=methodology,
            runtime_id=run["adapter"]["id"],
            runtime_version=runtime.get("version"),
            model_artifact=str(identity.get("artifact_digest") or identity.get("api_digest") or "unknown"),
            hardware_profile=hw_key,
            context=int(cfg.get("context") or 0),
            concurrency=int(cfg.get("concurrency") or 0),
            workload=f"{workload.get('id')}@{workload.get('version')}",
        )
        cfg_k = config_key(ident)
        obs_k = observation_key(cfg_k, run_id, digest)
        notes = [f"[aig-lab {obs_k}] cfg={cfg_k} run={run_id}", *common_notes, *unavailable_notes(agg, workload)]
        fields: dict[str, Any]
        if entity == "CompatibilityObservation":
            fields = {
                "model_variant_id": m_key,
                "hardware_profile_id": hw_key,
                "runtime": runtime_label,
                "context": cfg.get("context"),
                "concurrency": cfg.get("concurrency"),
                "observed_peak_memory": _gb(peak_rss_max),
                "observed_tokens_sec": _round(_metric(agg, "decode_tokens_per_s"), 1),
                "completion_status": (agg.get("runtime_completion") or {}).get("status", "unknown"),
            }
            notes.append(f"workload exercised={workload.get('id')}@{workload.get('version')}")
        else:
            ts = agg.get("task_success") or {}
            fields = {
                "workload_id": workload.get("planner_key") or workload.get("id"),
                "model_variant_id": m_key,
                "hardware_or_host_id": hw_key,
                "context": cfg.get("context"),
                "concurrency": cfg.get("concurrency"),
                "completion_time": _round(_metric(agg, "task_completion_s"), 2),
                "ttft": _round(_metric(agg, "ttft_ms"), 1),
                "tokens_sec": _round(_metric(agg, "decode_tokens_per_s"), 1),
                "peak_memory": _gb(peak_rss_max),
                "failures": (ts.get("tasks_total") or 0) - (ts.get("tasks_succeeded") or 0)
                if ts.get("tasks_total")
                else None,
                "tool_latency": _round(_metric(agg, "tool_latency_ms"), 1),
                "power": None,
                "cost_per_task": None,
                "methodology_version": methodology,
            }
            notes.append(
                f"workload={workload.get('id')}@{workload.get('version')} definition={str(workload.get('definition_sha256'))[:16]} "
                f"tasks succeeded {ts.get('tasks_succeeded')}/{ts.get('tasks_total')}"
            )
        fields.update(
            {
                "source_name": base["source_name"],
                "source_url": base["source_url"],
                "evidence_type": classification.evidence_type,
                "tested_at": tested_at,
                "confidence": base["confidence"],
                "production_ready": False,
                "notes": " | ".join(notes),
            }
        )
        assert_publishable(fields["evidence_type"], classification)
        records.append(
            {
                "entity": entity,
                "observation_key": obs_k,
                "config_key": cfg_k,
                "config_identity": ident,
                **base,
                "workload": workload
                if entity == "BenchmarkObservation"
                else {"id": workload.get("id"), "version": workload.get("version")},
                "base44_fields": fields,
            }
        )
    return records
