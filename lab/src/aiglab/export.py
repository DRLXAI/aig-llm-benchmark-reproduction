"""Exports of normalized records: JSON (complete lab records) and CSV (flat, for analysis)."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aiglab.artifacts import iter_run_dirs
from aiglab.normalize import normalize_run

FILES = {
    "CompatibilityObservation": "compatibility-observations",
    "BenchmarkObservation": "benchmark-observations",
}
KEY_COLUMNS = ["observation_key", "config_key", "run_id", "origin", "evidence_type", "evidence_reasons", "seal_digest"]


def collect(runs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in iter_run_dirs(runs_root):
        records.extend(normalize_run(run_dir))
    return sorted(records, key=lambda r: (r["entity"], r["observation_key"]))


def write_json(records: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for entity, stem in FILES.items():
        rows = [r for r in records if r["entity"] == entity]
        path = out_dir / f"{stem}.json"
        path.write_text(json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def write_csv(records: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for entity, stem in FILES.items():
        rows = [r for r in records if r["entity"] == entity]
        field_names: list[str] = []
        for r in rows:
            for k in r["base44_fields"]:
                if k not in field_names:
                    field_names.append(k)
        path = out_dir / f"{stem}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(KEY_COLUMNS + field_names)
            for r in rows:
                key_values = [
                    "; ".join(r[k]) if isinstance(r.get(k), list) else ("" if r.get(k) is None else r.get(k))
                    for k in KEY_COLUMNS
                ]
                values = ["" if r["base44_fields"].get(k) is None else r["base44_fields"].get(k) for k in field_names]
                writer.writerow(key_values + values)
        paths.append(path)
    return paths
