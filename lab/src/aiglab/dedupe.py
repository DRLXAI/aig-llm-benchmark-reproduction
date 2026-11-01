"""Deterministic identity keys.

``config_key``       what was measured: entity + methodology + runtime(+version) + model artifact
                     digest + hardware profile + context + concurrency (+ workload@version).
                     Repeat batches of the same configuration share it.
``observation_key``  one published observation: config_key + the tested run identity (run_id and
                     the digest of its sealed raw data). Re-exporting or re-syncing the same run
                     always yields the same key, so it can never be inserted twice.

Neither key includes timestamps of export, aggregation exclusions or free text, so recomputing
them is stable. An operator exclusion changes a record's *content*, not its identity; the sync
plan reports that as an update that needs explicit approval.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config_identity(
    *,
    entity: str,
    methodology_version: str,
    runtime_id: str,
    runtime_version: str | None,
    model_artifact: str,
    hardware_profile: str,
    context: int,
    concurrency: int,
    workload: str | None,
) -> dict[str, Any]:
    if entity not in ("CompatibilityObservation", "BenchmarkObservation"):
        raise ValueError(f"unknown entity {entity!r}")
    if entity == "BenchmarkObservation" and not workload:
        raise ValueError("a BenchmarkObservation identity needs a workload")
    return {
        "entity": entity,
        "methodology_version": methodology_version,
        "runtime": f"{runtime_id}@{runtime_version or 'unknown'}",
        "model_artifact": model_artifact,
        "hardware_profile": hardware_profile,
        "context": int(context),
        "concurrency": int(concurrency),
        "workload": workload if entity == "BenchmarkObservation" else None,
    }


def config_key(identity: Mapping[str, Any]) -> str:
    return "cfg_" + _digest(identity)[:32]


def observation_key(cfg_key: str, run_id: str, seal_digest: str) -> str:
    return "obs_" + _digest({"config_key": cfg_key, "run_id": run_id, "seal_digest": seal_digest})[:32]


def content_digest(payload: Mapping[str, Any]) -> str:
    return "sha256:" + _digest(payload)
