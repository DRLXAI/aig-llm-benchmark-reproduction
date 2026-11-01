"""Evidence quality validation over an evidence store (a directory of run directories).

Errors break evidence and fail the check (exit 2): a tampered or unsealed run, non-executed data
inside the evidence store, an aig_tested normalized record whose run is not aig_tested, or a
duplicated observation key. Warnings are recorded for a reader to weigh (low confidence, stale,
unavailable core metrics, host conditions).
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from aiglab.artifacts import iter_run_dirs, read_json
from aiglab.methodology import FRESH_CURRENT_DAYS, FRESH_REVALIDATE_DAYS
from aiglab.normalize import load_run


def freshness(tested_at: str | None, today: date | None = None) -> dict[str, Any]:
    if not tested_at:
        return {"state": "unknown", "age_days": None}
    try:
        d = date.fromisoformat(tested_at[:10])
    except ValueError:
        return {"state": "unknown", "age_days": None}
    age = ((today or datetime.now(UTC).date()) - d).days
    state = "current" if age <= FRESH_CURRENT_DAYS else "revalidate" if age <= FRESH_REVALIDATE_DAYS else "stale"
    return {"state": state, "age_days": age}


def check_run(run_dir: Path, *, in_store: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        run, agg, classification, problems = load_run(run_dir)
    except (OSError, ValueError, KeyError) as exc:
        return {
            "run_id": run_dir.name,
            "evidence_type": "unknown",
            "errors": [f"unreadable run: {exc}"],
            "warnings": [],
        }
    errors += problems
    if in_store and classification.origin != "executed":
        errors.append(f"origin {classification.origin!r} does not belong in the evidence store")
    if classification.origin == "executed" and not classification.aig_tested:
        warnings.append("executed run is NOT aig_tested: " + "; ".join(classification.reasons))
    if agg.get("confidence") in ("low", "unknown"):
        warnings.append(f"confidence {agg.get('confidence')}: {agg.get('confidence_reasons')}")
    for name in ("peak_rss_bytes", "decode_tokens_per_s", "ttft_ms"):
        if ((agg.get("metrics") or {}).get(name) or {}).get("status") != "measured":
            warnings.append(f"core metric {name} unavailable")
    if (agg.get("runtime_completion") or {}).get("status") != "completed":
        warnings.append(f"runtime completion {(agg.get('runtime_completion') or {}).get('status')}")
    for w in run.get("warnings") or []:
        warnings.append(f"preflight: {w.get('name')}: {w.get('detail')}")
    fresh = freshness(str(run.get("started_at") or ""))
    if fresh["state"] != "current":
        warnings.append(f"freshness {fresh['state']} ({fresh['age_days']} days)")
    return {
        "run_id": run.get("run_id", run_dir.name),
        "evidence_type": classification.evidence_type,
        "origin": classification.origin,
        "confidence": agg.get("confidence"),
        "freshness": fresh,
        "errors": errors,
        "warnings": warnings,
    }


def check_store(runs_root: Path, normalized_dir: Path | None = None) -> dict[str, Any]:
    reports = [check_run(d) for d in iter_run_dirs(runs_root)]
    by_run = {r["run_id"]: r for r in reports}
    store_errors: list[str] = []
    if normalized_dir is not None:
        for name in ("compatibility-observations.json", "benchmark-observations.json"):
            path = normalized_dir / name
            if not path.is_file():
                continue
            records = read_json(path)
            keys = Counter(r["observation_key"] for r in records)
            store_errors += [f"{name}: duplicate observation key {k}" for k, n in keys.items() if n > 1]
            for r in records:
                if r.get("evidence_type") == "aig_tested":
                    source = by_run.get(r.get("run_id"))
                    if source is None or source["evidence_type"] != "aig_tested":
                        store_errors.append(
                            f"{name}: {r['observation_key']} claims aig_tested but run {r.get('run_id')} is "
                            f"{'missing' if source is None else source['evidence_type']}"
                        )
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs": reports,
        "store_errors": store_errors,
        "summary": {
            "runs": len(reports),
            "aig_tested": sum(1 for r in reports if r["evidence_type"] == "aig_tested"),
            "with_errors": sum(1 for r in reports if r["errors"]) + (1 if store_errors else 0),
            "with_warnings": sum(1 for r in reports if r["warnings"]),
        },
        "ok": not store_errors and not any(r["errors"] for r in reports),
    }
