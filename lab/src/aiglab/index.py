"""Evidence index and query layer for the Planner.

Answers, from normalized records only:
  * closest AIG-tested configuration to a plan (with every difference spelled out),
  * measured memory for an exact model/hardware/context/concurrency,
  * measured performance for a workload,
  * freshness and methodology version of each answer.

It never interpolates or extrapolates a number. A nearest neighbour is returned as a *different*
configuration with its differences listed, and ``exact`` is True only when model artifact,
hardware profile, context, concurrency (and workload, if asked) all match.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any

from aiglab.methodology import METHODOLOGY_VERSION, comparable
from aiglab.quality import freshness

INDEX_SCHEMA = "aiglab.evidence-index/1"


def _m(record: Mapping[str, Any], name: str, stat: str) -> float | None:
    m = (record.get("metrics") or {}).get(name) or {}
    if m.get("status") != "measured":
        return None
    v = (m.get("included") or {}).get(stat)
    return float(v) if isinstance(v, (int, float)) else None


def _gb(v: float | None) -> float | None:
    return round(v / 1e9, 2) if v is not None else None


def entry(record: Mapping[str, Any], today: date | None = None) -> dict[str, Any]:
    model = record.get("model") or {}
    hw = record.get("hardware") or {}
    size = model.get("artifact_size_bytes")
    mem = hw.get("memory_total_bytes")
    workload = record.get("workload") or {}
    return {
        "observation_key": record["observation_key"],
        "config_key": record["config_key"],
        "entity": record["entity"],
        "evidence_type": record["evidence_type"],
        "tested_at": record.get("tested_at"),
        "freshness": freshness(record.get("tested_at"), today),
        "methodology_version": record.get("methodology_version"),
        "confidence": record.get("confidence"),
        "model": {
            "lab_key": model.get("lab_key"),
            "family": model.get("family"),
            "name": model.get("name"),
            "runtime_ref": model.get("runtime_ref"),
            "quantization": model.get("quantization"),
            "parameter_size": model.get("parameter_size"),
            "artifact_size_gb": round(size / 1e9, 2) if isinstance(size, (int, float)) else None,
        },
        "hardware": {
            "lab_key": hw.get("lab_key"),
            "chip": hw.get("chip"),
            "memory_gib": round(mem / 2**30) if isinstance(mem, (int, float)) else None,
            "memory_kind": hw.get("memory_kind"),
            "os": hw.get("os"),
        },
        "runtime": record.get("runtime"),
        "context": record.get("context"),
        "concurrency": record.get("concurrency"),
        "workload": {"id": workload.get("id"), "version": workload.get("version")} if workload else None,
        "measured": {
            "peak_memory_gb": _gb(_m(record, "peak_rss_bytes", "max")),
            "peak_memory_gb_median": _gb(_m(record, "peak_rss_bytes", "median")),
            "kv_and_buffers_gb": _gb(_m(record, "peak_phys_footprint_bytes", "max")),
            "decode_tokens_per_s": _m(record, "decode_tokens_per_s", "median"),
            "decode_tokens_per_s_range": [
                _m(record, "decode_tokens_per_s", "min"),
                _m(record, "decode_tokens_per_s", "max"),
            ],
            "ttft_ms": _m(record, "ttft_ms", "median"),
            "task_completion_s": _m(record, "task_completion_s", "median"),
            "tool_latency_ms": _m(record, "tool_latency_ms", "median"),
            "task_success_rate": (record.get("task_success") or {}).get("rate"),
            "runtime_completion": (record.get("runtime_completion") or {}).get("status"),
            "prompt_tokens_max": _m(record, "prompt_tokens_max", "max"),
        },
        "run_id": record.get("run_id"),
        "source_url": record.get("source_url"),
    }


def build_index(records: Sequence[Mapping[str, Any]], today: date | None = None) -> dict[str, Any]:
    entries = sorted((entry(r, today) for r in records), key=lambda e: (e["entity"], e["observation_key"]))
    return {
        "schema": INDEX_SCHEMA,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "methodology_version": METHODOLOGY_VERSION,
        "counts": {
            "entries": len(entries),
            "aig_tested": sum(1 for e in entries if e["evidence_type"] == "aig_tested"),
        },
        "entries": entries,
    }


def _matches_model(e: Mapping[str, Any], model: str) -> int | None:
    """0 exact (runtime ref or lab key), 1 same family + quantization text, 2 same family, None no match."""
    q = model.lower().strip()
    m = e["model"]
    if q in {
        str(m.get("runtime_ref") or "").lower(),
        str(m.get("lab_key") or "").lower(),
        str(m.get("name") or "").lower(),
    }:
        return 0
    family = str(m.get("family") or "").lower()
    if family and family in q:
        quant = str(m.get("quantization") or "").lower()
        return 1 if quant and quant in q else 2
    return None


def _log2_gap(a: float | None, b: float | None) -> float:
    if not a or not b:
        return 1.0
    return abs(math.log2(a / b))


def closest(
    index: Mapping[str, Any],
    *,
    model: str,
    context: int,
    concurrency: int = 1,
    hardware_key: str | None = None,
    memory_gib: float | None = None,
    workload: str | None = None,
    entity: str | None = None,
    include_untested: bool = False,
    limit: int = 3,
) -> dict[str, Any]:
    candidates = []
    for e in index["entries"]:
        if not include_untested and e["evidence_type"] != "aig_tested":
            continue
        if entity and e["entity"] != entity:
            continue
        if not comparable(str(e.get("methodology_version")), METHODOLOGY_VERSION):
            continue
        model_gap = _matches_model(e, model)
        if model_gap is None:
            continue
        hw = e["hardware"]
        hw_gap = (
            0.0 if hardware_key and hw.get("lab_key") == hardware_key else _log2_gap(memory_gib, hw.get("memory_gib"))
        )
        ctx_gap = _log2_gap(context, e.get("context"))
        conc_gap = _log2_gap(concurrency, e.get("concurrency"))
        wl = (e.get("workload") or {}).get("id")
        wl_gap = 0.0 if not workload or wl == workload else 1.0
        score = 3 * model_gap + 2 * hw_gap + ctx_gap + conc_gap + wl_gap
        diffs = []
        if model_gap:
            diffs.append(
                f"model differs: measured {e['model'].get('runtime_ref')} ({e['model'].get('quantization')}) vs plan {model!r}"
            )
        if hw_gap:
            diffs.append(
                f"hardware differs: measured {hw.get('chip')} {hw.get('memory_gib')} GiB vs plan "
                f"{hardware_key or (str(memory_gib) + ' GiB' if memory_gib else 'unspecified')}"
            )
        if ctx_gap:
            direction = "smaller" if (e.get("context") or 0) < context else "larger"
            diffs.append(
                f"context differs: measured {e.get('context')} vs plan {context} ({direction}; KV cache scales with "
                "context, so measured memory does not transfer)"
            )
        if conc_gap:
            diffs.append(f"concurrency differs: measured {e.get('concurrency')} vs plan {concurrency}")
        if wl_gap:
            diffs.append(f"workload differs: measured {wl} vs plan {workload}")
        candidates.append({"score": round(score, 3), "exact": not diffs, "differences": diffs, "entry": e})
    candidates.sort(key=lambda c: (c["score"], c["entry"]["freshness"].get("age_days") or 0))
    best = candidates[:limit]
    return {
        "query": {
            "model": model,
            "context": context,
            "concurrency": concurrency,
            "hardware_key": hardware_key,
            "memory_gib": memory_gib,
            "workload": workload,
        },
        "exact_match": bool(best) and best[0]["exact"],
        "answer": (
            "exact AIG-tested configuration found"
            if best and best[0]["exact"]
            else "no exact AIG-tested configuration; nearest measured configurations are listed with their differences"
            if best
            else "no AIG-tested measurement for this model family; planning values remain calculated"
        ),
        "results": best,
    }


def measured_memory(
    index: Mapping[str, Any], *, model: str, hardware_key: str, context: int, concurrency: int
) -> dict[str, Any]:
    res = closest(
        index,
        model=model,
        context=context,
        concurrency=concurrency,
        hardware_key=hardware_key,
        entity="CompatibilityObservation",
        limit=10,
    )
    exact = [r["entry"] for r in res["results"] if r["exact"]]
    return {
        "query": res["query"],
        "measurements": [
            {
                "peak_memory_gb": e["measured"]["peak_memory_gb"],
                "tested_at": e["tested_at"],
                "freshness": e["freshness"],
                "methodology_version": e["methodology_version"],
                "confidence": e["confidence"],
                "run_id": e["run_id"],
            }
            for e in exact
        ],
        "status": "measured" if exact else "not_measured",
        "nearest": [] if exact else res["results"],
    }


def measured_performance(
    index: Mapping[str, Any], *, workload: str, model: str | None = None, hardware_key: str | None = None
) -> dict[str, Any]:
    rows = []
    for e in index["entries"]:
        if e["entity"] != "BenchmarkObservation" or e["evidence_type"] != "aig_tested":
            continue
        if (e.get("workload") or {}).get("id") != workload:
            continue
        if model and _matches_model(e, model) is None:
            continue
        if hardware_key and e["hardware"].get("lab_key") != hardware_key:
            continue
        rows.append(e)
    rows.sort(key=lambda e: (e["model"].get("runtime_ref") or "", e.get("context") or 0, e.get("concurrency") or 0))
    return {"workload": workload, "status": "measured" if rows else "not_measured", "results": rows}
