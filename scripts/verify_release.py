#!/usr/bin/env python3
"""Verify this archived, already-public AIG benchmark snapshot without a network connection."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "public-measurements"
sys.path.insert(0, str(ROOT / "lab" / "src"))
from aiglab.workloads import WORKLOADS  # noqa: E402


def verify() -> dict[str, object]:
    seal = (DATA / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    for line in seal:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if not match:
            raise ValueError("Malformed checksum line")
        expected, name = match.groups()
        if name in seen or not (DATA / name).is_file():
            raise ValueError(f"Duplicate/missing checksum target: {name}")
        seen.add(name)
        actual = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Incorrect SHA-256: {name}")
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    if index.get("generated_at") != "2026-10-30T17:22:38+00:00":
        raise ValueError("Unexpected release snapshot date")
    entries = index["entries"]
    if len(entries) != 15:
        raise ValueError(f"Expected 15 published configurations, found {len(entries)}")
    required = {"index.json", *(f"{e['run_id']}.json" for e in entries)}
    if seen != required:
        raise ValueError(f"Checksum manifest does not cover the exact index: {seen ^ required}")
    if set(x.name for x in DATA.iterdir()) != required | {"SHA256SUMS"}:
        raise ValueError("Unexpected file in public measurements")
    measured = 0
    for entry in entries:
        name = entry["run_id"] + ".json"
        run = json.loads((DATA / name).read_text(encoding="utf-8"))
        if run["run_id"] != entry["run_id"]:
            raise ValueError("Run id mismatch")
        workload = run["workload"]
        impl = WORKLOADS[workload["id"]]
        if workload["definition_sha256"] != impl.definition_sha256():
            raise ValueError(f"Historic workload source differs from published run {name}")
        if workload["version"] != impl.version:
            raise ValueError(f"Workload version differs: {name}")
        measured_count = sum(rep.get("phase") == "measured" for rep in run["repetitions"])
        if measured_count != 5:
            raise ValueError(f"Expected 5 measured repetitions: {name}")
        measured += measured_count
    if measured != 75:
        raise ValueError(f"Repetition count differs from release: {measured}")
    return {"snapshot": index["generated_at"], "runs": len(entries), "measured_repetitions": measured,
            "verified_files": len(seen), "verified_workload_digests": len(entries)}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
