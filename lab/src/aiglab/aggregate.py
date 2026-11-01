"""Repeat-run aggregation.

Every measured repetition is retained. Outliers are *flagged* (robust z-score on the median
absolute deviation) and stay in the statistics. A repetition leaves the headline statistics only
through an explicit operator exclusion with a written reason, and the statistics over all
repetitions are always reported next to the post-exclusion ones so the effect is visible.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from aiglab.methodology import (
    CONFIDENCE_HIGH_MAX_CV,
    CONFIDENCE_HIGH_MIN_RUNS,
    CONFIDENCE_MEDIUM_MIN_RUNS,
    OUTLIER_MODIFIED_Z,
)

AGGREGATOR_VERSION = "1.0.0"

#: metric -> (unit, outlier-checked). Discrete counts are not outlier-checked.
METRICS: dict[str, tuple[str, bool]] = {
    "ttft_ms": ("ms", True),
    "prompt_eval_ms": ("ms", True),
    "decode_tokens_per_s": ("tokens/s", True),
    "aggregate_output_tokens_per_s": ("tokens/s", True),
    "task_completion_s": ("s", True),
    "rep_wall_s": ("s", True),
    "peak_rss_bytes": ("bytes", True),
    "peak_phys_footprint_bytes": ("bytes", True),
    "tool_latency_ms": ("ms", True),
    "prompt_tokens_max": ("tokens", False),
    "prompt_cached_tokens_total": ("tokens", False),
    "success_rate": ("fraction", False),
}
PRIMARY = ("decode_tokens_per_s", "peak_rss_bytes", "ttft_ms")


def median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def outlier_indices(values: Sequence[float]) -> list[int]:
    """Indices whose modified z-score exceeds the threshold. With MAD == 0 (identical majority),
    any value more than 10% away from the median is flagged instead."""
    if len(values) < 3:
        return []
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values)
    if mad == 0:
        return [i for i, v in enumerate(values) if abs(v - med) > 0.10 * abs(med) and v != med]
    return [i for i, v in enumerate(values) if abs(0.6745 * (v - med) / mad) > OUTLIER_MODIFIED_Z]


def describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None, "mean": None, "stdev": None, "cv": None}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "stdev": stdev,
        "cv": (stdev / mean) if mean else None,
    }


def aggregate_run(
    run: Mapping[str, Any],
    seal_digest: str | None,
    exclusions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    measured = [r for r in run.get("repetitions") or [] if r.get("phase") == "measured"]
    excluded = {int(e["repetition"]): str(e["reason"]) for e in exclusions}
    for idx, reason in excluded.items():
        if not reason.strip():
            raise ValueError(f"exclusion of repetition {idx} has no reason")
        if idx not in {int(r["index"]) for r in measured}:
            raise ValueError(f"repetition {idx} is not a measured repetition of this run")

    metrics: dict[str, Any] = {}
    for name, (unit, check_outliers) in METRICS.items():
        series = [(int(r["index"]), r["metrics"].get(name)) for r in measured]
        present = [(i, float(v)) for i, v in series if isinstance(v, (int, float))]
        if not present:
            reasons = sorted(
                {str(r["metrics"].get(f"{name}_reason")) for r in measured if r["metrics"].get(f"{name}_reason")}
            )
            metrics[name] = {"unit": unit, "status": "unavailable", "reasons": reasons}
            continue
        values = [v for _, v in present]
        outliers = [present[i][0] for i in outlier_indices(values)] if check_outliers else []
        included = [v for i, v in present if i not in excluded]
        metrics[name] = {
            "unit": unit,
            "status": "measured",
            "values": [
                {"repetition": i, "value": v, "excluded": i in excluded, "outlier": i in outliers} for i, v in present
            ],
            "outliers": outliers,
            "all": describe(values),
            "included": describe(included),
        }

    requests = [q for r in measured for o in r.get("outcomes") or [] for q in o.get("requests") or []]
    failed = [q for q in requests if not q.get("ok")]
    status = (
        "completed"
        if requests and not failed
        else ("failed" if requests and len(failed) == len(requests) else "partial")
    )
    if not requests:
        status = "unknown"
    outcomes = [o for r in measured for o in r.get("outcomes") or []]
    failure_counts = Counter(str(o.get("failure_reason")) for o in outcomes if not o.get("success"))

    agg = {
        "aggregator_version": AGGREGATOR_VERSION,
        "computed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run.get("run_id"),
        "seal_digest": seal_digest,
        "methodology_version": run.get("methodology_version"),
        "measured_repetitions": len(measured),
        "warmup_repetitions": len([r for r in run.get("repetitions") or [] if r.get("phase") == "warmup"]),
        "exclusions": [{"repetition": i, "reason": r} for i, r in sorted(excluded.items())],
        "metrics": metrics,
        "runtime_completion": {
            "status": status,
            "requests_total": len(requests),
            "requests_failed": len(failed),
            "errors": sorted({str(q.get("error")) for q in failed})[:10],
        },
        "task_success": {
            "tasks_total": len(outcomes),
            "tasks_succeeded": sum(1 for o in outcomes if o.get("success")),
            "rate": (sum(1 for o in outcomes if o.get("success")) / len(outcomes)) if outcomes else None,
            "failure_reasons": dict(failure_counts.most_common(10)),
        },
    }
    agg["confidence"], agg["confidence_reasons"] = confidence(agg, run)
    return agg


def confidence(agg: Mapping[str, Any], run: Mapping[str, Any]) -> tuple[str, list[str]]:
    n = agg["measured_repetitions"] - len(agg["exclusions"])
    reasons: list[str] = []
    if agg["runtime_completion"]["status"] == "unknown" or n == 0:
        return "unknown", ["no measured requests"]
    level = "high" if n >= CONFIDENCE_HIGH_MIN_RUNS else "medium" if n >= CONFIDENCE_MEDIUM_MIN_RUNS else "low"
    reasons.append(f"{n} included measured repetitions")
    if level == "high":
        for name in PRIMARY:
            m = agg["metrics"].get(name) or {}
            if m.get("outliers"):
                level = "medium"
                reasons.append(f"{name} has flagged outliers {m['outliers']}")
        cv = ((agg["metrics"].get("decode_tokens_per_s") or {}).get("included") or {}).get("cv")
        if cv is None or cv > CONFIDENCE_HIGH_MAX_CV:
            level = "medium"
            reasons.append(f"decode tokens/s CV {cv} exceeds {CONFIDENCE_HIGH_MAX_CV}")
        if agg["exclusions"]:
            level = "medium"
            reasons.append("operator exclusions present")
    pressure = ((run.get("host_at_start") or {}).get("pressure_level") or {}).get("value")
    if isinstance(pressure, int) and pressure >= 2 and level == "high":
        level = "medium"
        reasons.append(f"host memory pressure level {pressure} (warn) at run start")
    if agg["runtime_completion"]["status"] != "completed":
        reasons.append(f"runtime completion {agg['runtime_completion']['status']}")
    return level, reasons
