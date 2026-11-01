from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiglab.aggregate import aggregate_run, outlier_indices
from aiglab.artifacts import read_json
from tests.conftest import synthetic_executed_run


def agg_for(
    tmp_path: Path, decodes: tuple[float, ...], exclusions: list[dict[str, Any]] | None = None, pressure: int = 1
) -> dict[str, Any]:
    def mutate(run: dict[str, Any]) -> None:
        run["host_at_start"] = {"pressure_level": {"value": pressure}}

    run_dir = synthetic_executed_run(tmp_path, decodes=decodes, mutate=mutate)
    return aggregate_run(read_json(run_dir / "run.json"), "sha256:x", exclusions or [])


def test_outliers_flagged_by_robust_z() -> None:
    assert outlier_indices([40, 41, 40.5, 39.8, 12.0]) == [4]
    assert outlier_indices([40, 41, 40.5]) == []
    assert outlier_indices([1, 2]) == []  # too few to judge


def test_mad_zero_fallback() -> None:
    assert outlier_indices([10, 10, 10, 10, 12]) == [4]
    assert outlier_indices([10, 10, 10, 10, 10.5]) == []


def test_every_run_retained_and_outlier_not_dropped(tmp_path: Path) -> None:
    agg = agg_for(tmp_path, (40.0, 41.0, 40.5, 39.8, 12.0))
    m = agg["metrics"]["decode_tokens_per_s"]
    assert len(m["values"]) == 5
    assert m["outliers"] == [4]
    assert m["included"]["n"] == 5  # flagged, still included
    assert m["included"]["min"] == 12.0
    assert agg["confidence"] == "medium"
    assert any("outliers" in r for r in agg["confidence_reasons"])


def test_explicit_exclusion_is_recorded_and_visible(tmp_path: Path) -> None:
    agg = agg_for(tmp_path, (40.0, 41.0, 40.5, 39.8, 12.0), [{"repetition": 4, "reason": "fan curve change"}])
    m = agg["metrics"]["decode_tokens_per_s"]
    assert agg["exclusions"] == [{"repetition": 4, "reason": "fan curve change"}]
    assert m["included"]["n"] == 4 and m["all"]["n"] == 5
    assert m["all"]["min"] == 12.0 and m["included"]["min"] == 39.8
    assert [v["excluded"] for v in m["values"]] == [False, False, False, False, True]


def test_exclusion_requires_reason_and_existing_rep(tmp_path: Path) -> None:
    run = read_json(synthetic_executed_run(tmp_path) / "run.json")
    with pytest.raises(ValueError, match="no reason"):
        aggregate_run(run, None, [{"repetition": 1, "reason": "  "}])
    with pytest.raises(ValueError, match="not a measured repetition"):
        aggregate_run(run, None, [{"repetition": 99, "reason": "x"}])


def test_confidence_levels(tmp_path: Path) -> None:
    assert agg_for(tmp_path / "a", (40.0, 41.0, 40.5, 39.8, 40.2))["confidence"] == "high"
    assert agg_for(tmp_path / "b", (40.0, 41.0, 40.5, 39.8, 40.2), pressure=2)["confidence"] == "medium"
    assert agg_for(tmp_path / "c", (40.0, 41.0, 40.5))["confidence"] == "medium"
    assert agg_for(tmp_path / "d", (40.0,))["confidence"] == "low"
    assert agg_for(tmp_path / "e", (20.0, 40.0, 30.0, 45.0, 25.0))["confidence"] == "medium"  # CV too high


def test_unavailable_metric_keeps_reason(tmp_path: Path) -> None:
    agg = agg_for(tmp_path, (40.0, 41.0, 40.5))
    tool = agg["metrics"]["tool_latency_ms"]
    assert tool["status"] == "unavailable"
    assert tool["reasons"] == ["not applicable: workload has no tool or retrieval step"]


def test_runtime_completion_statuses(tmp_path: Path) -> None:
    def fail_one(run: dict[str, Any]) -> None:
        run["repetitions"][1]["outcomes"][0]["requests"][0]["ok"] = False
        run["repetitions"][1]["outcomes"][0]["requests"][0]["error"] = "HTTP 500: out of memory"

    run = read_json(synthetic_executed_run(tmp_path, mutate=fail_one) / "run.json")
    agg = aggregate_run(run, None)
    assert agg["runtime_completion"]["status"] == "partial"
    assert agg["runtime_completion"]["errors"] == ["HTTP 500: out of memory"]
