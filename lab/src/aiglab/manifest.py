"""Run manifest: the machine-readable statement of exactly what a run will measure.

A manifest is JSON (stdlib only, byte-stable for hashing). The JSON Schema in
``lab/schemas/run-manifest.v1.schema.json`` documents the same contract; this module is the
enforcing validator and returns every problem at once rather than stopping at the first.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiglab.methodology import SUPPORTED_METHODOLOGY_VERSIONS

MANIFEST_VERSION = 1
TARGET_KINDS = ("local", "cloud")
MEMORY_KINDS = ("unified", "ddr", "vram_gddr", "vram_hbm", "unknown")
MIN_CONTEXT, MAX_CONTEXT = 512, 1_048_576
MAX_CONCURRENCY = 32
MAX_WARMUP, MAX_MEASURED = 10, 50


@dataclass(frozen=True)
class ModelSpec:
    family: str
    name: str
    runtime_ref: str
    variant: str | None = None
    quantization: str | None = None
    precision: str | None = None
    parameter_count: str | None = None
    expected_artifact_digest: str | None = None


@dataclass(frozen=True)
class RuntimeSpec:
    id: str
    endpoint: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    kind: str
    hardware_profile: str | None = None
    usable_memory_gb: float | None = None
    memory_kind: str | None = None
    provider: str | None = None
    region: str | None = None
    instance: str | None = None
    hourly_usd: float | None = None
    price_source_url: str | None = None


@dataclass(frozen=True)
class Manifest:
    id: str
    methodology_version: str
    model: ModelSpec
    runtime: RuntimeSpec
    target: TargetSpec
    context: int
    concurrency: int
    workload_id: str
    workload_version: str
    warmup: int
    measured: int
    max_output_tokens: int
    temperature: float
    seed: int
    request_timeout_s: int
    notes: str
    raw: Mapping[str, Any]
    sha256: str

    @property
    def short_hash(self) -> str:
        return self.sha256[:12]


class ManifestError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def manifest_hash(raw: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(raw)).hexdigest()


def _str(d: Mapping[str, Any], key: str, errors: list[str], where: str, required: bool = True) -> str | None:
    value = d.get(key)
    if value is None:
        if required:
            errors.append(f"{where}.{key} is required")
        return None
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}.{key} must be a non-empty string")
        return None
    return value.strip()


def _int(
    d: Mapping[str, Any], key: str, errors: list[str], where: str, lo: int, hi: int, default: int | None = None
) -> int:
    value = d.get(key, default)
    if value is None:
        errors.append(f"{where}.{key} is required")
        return lo
    if not isinstance(value, int) or isinstance(value, bool) or not lo <= value <= hi:
        errors.append(f"{where}.{key} must be an integer in {lo}..{hi}; got {value!r}")
        return lo
    return value


def _num(d: Mapping[str, Any], key: str, errors: list[str], where: str) -> float | None:
    value = d.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        errors.append(f"{where}.{key} must be a non-negative number or null")
        return None
    return float(value)


def _mapping(raw: Mapping[str, Any], key: str, errors: list[str]) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be an object")
        return {}
    return value


KNOWN_TOP = {
    "manifest_version",
    "id",
    "methodology_version",
    "model",
    "runtime",
    "target",
    "config",
    "workload",
    "repetitions",
    "limits",
    "notes",
}


def parse_manifest(
    raw: Any,
    *,
    known_runtimes: frozenset[str] | None = None,
    known_workloads: Mapping[str, str] | None = None,
) -> Manifest:
    """Validate and parse. Raises ManifestError listing every problem found."""
    if known_runtimes is None or known_workloads is None:
        from aiglab.adapters.registry import runtime_ids
        from aiglab.workloads import workload_versions

        known_runtimes = known_runtimes if known_runtimes is not None else runtime_ids()
        known_workloads = known_workloads if known_workloads is not None else workload_versions()

    errors: list[str] = []
    if not isinstance(raw, Mapping):
        raise ManifestError(["manifest must be a JSON object"])
    unknown = sorted(set(raw) - KNOWN_TOP)
    if unknown:
        errors.append(f"unknown top-level keys: {unknown}")
    if raw.get("manifest_version") != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION}")
    mid = _str(raw, "id", errors, "$") or ""
    if mid and not all(c.isalnum() or c in "-_." for c in mid):
        errors.append("id may contain only letters, digits, '-', '_' and '.'")
    methodology = _str(raw, "methodology_version", errors, "$") or ""
    if methodology and methodology not in SUPPORTED_METHODOLOGY_VERSIONS:
        errors.append(f"methodology_version {methodology!r} is not supported {sorted(SUPPORTED_METHODOLOGY_VERSIONS)}")

    m = _mapping(raw, "model", errors)
    model = ModelSpec(
        family=_str(m, "family", errors, "model") or "",
        name=_str(m, "name", errors, "model") or "",
        runtime_ref=_str(m, "runtime_ref", errors, "model") or "",
        variant=_str(m, "variant", errors, "model", required=False),
        quantization=_str(m, "quantization", errors, "model", required=False),
        precision=_str(m, "precision", errors, "model", required=False),
        parameter_count=_str(m, "parameter_count", errors, "model", required=False),
        expected_artifact_digest=_str(m, "expected_artifact_digest", errors, "model", required=False),
    )
    if model.expected_artifact_digest and not model.expected_artifact_digest.startswith("sha256:"):
        errors.append("model.expected_artifact_digest must look like 'sha256:<hex>'")

    r = _mapping(raw, "runtime", errors)
    runtime_id = _str(r, "id", errors, "runtime") or ""
    if runtime_id and runtime_id not in known_runtimes:
        errors.append(f"runtime.id {runtime_id!r} is not a known runtime {sorted(known_runtimes)}")
    options = r.get("options") or {}
    if not isinstance(options, Mapping):
        errors.append("runtime.options must be an object")
        options = {}
    endpoint = _str(r, "endpoint", errors, "runtime", required=False)
    if endpoint and not endpoint.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
        errors.append("runtime.endpoint must be a loopback URL; remote endpoints are not measured by this harness")
    runtime = RuntimeSpec(id=runtime_id, endpoint=endpoint, options=dict(options))

    t = _mapping(raw, "target", errors)
    kind = _str(t, "kind", errors, "target") or ""
    if kind and kind not in TARGET_KINDS:
        errors.append(f"target.kind must be one of {TARGET_KINDS}")
    memory_kind = _str(t, "memory_kind", errors, "target", required=False)
    if memory_kind and memory_kind not in MEMORY_KINDS:
        errors.append(f"target.memory_kind must be one of {MEMORY_KINDS}")
    target = TargetSpec(
        kind=kind,
        hardware_profile=_str(t, "hardware_profile", errors, "target", required=False),
        usable_memory_gb=_num(t, "usable_memory_gb", errors, "target"),
        memory_kind=memory_kind,
        provider=_str(t, "provider", errors, "target", required=False),
        region=_str(t, "region", errors, "target", required=False),
        instance=_str(t, "instance", errors, "target", required=False),
        hourly_usd=_num(t, "hourly_usd", errors, "target"),
        price_source_url=_str(t, "price_source_url", errors, "target", required=False),
    )
    if kind == "cloud" and not target.provider:
        errors.append("target.provider is required for a cloud target")
    if target.hourly_usd is not None and not target.price_source_url:
        errors.append("target.hourly_usd requires target.price_source_url (a price without a source is not evidence)")

    c = _mapping(raw, "config", errors)
    context = _int(c, "context", errors, "config", MIN_CONTEXT, MAX_CONTEXT)
    concurrency = _int(c, "concurrency", errors, "config", 1, MAX_CONCURRENCY)
    max_output = _int(c, "max_output_tokens", errors, "config", 16, 32_768, default=1024)
    seed = _int(c, "seed", errors, "config", 0, 2**31 - 1, default=42)
    temperature = c.get("temperature", 0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not 0 <= temperature <= 2:
        errors.append("config.temperature must be a number in 0..2")
        temperature = 0

    w = _mapping(raw, "workload", errors)
    workload_id = _str(w, "id", errors, "workload") or ""
    workload_version = _str(w, "version", errors, "workload") or ""
    if workload_id and workload_id not in known_workloads:
        errors.append(f"workload.id {workload_id!r} is not a known workload {sorted(known_workloads)}")
    elif workload_id and workload_version and known_workloads[workload_id] != workload_version:
        errors.append(
            f"workload {workload_id!r} is at version {known_workloads[workload_id]!r}, manifest pins "
            f"{workload_version!r}; reproduce older versions from the git commit that defined them"
        )

    reps = _mapping(raw, "repetitions", errors)
    warmup = _int(reps, "warmup", errors, "repetitions", 0, MAX_WARMUP)
    measured = _int(reps, "measured", errors, "repetitions", 1, MAX_MEASURED)

    limits = raw.get("limits") or {}
    if not isinstance(limits, Mapping):
        errors.append("limits must be an object")
        limits = {}
    request_timeout = _int(limits, "request_timeout_s", errors, "limits", 5, 3600, default=600)

    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        errors.append("notes must be a string")
        notes = ""

    if errors:
        raise ManifestError(errors)
    return Manifest(
        id=mid,
        methodology_version=methodology,
        model=model,
        runtime=runtime,
        target=target,
        context=context,
        concurrency=concurrency,
        workload_id=workload_id,
        workload_version=workload_version,
        warmup=warmup,
        measured=measured,
        max_output_tokens=max_output,
        temperature=float(temperature),
        seed=seed,
        request_timeout_s=request_timeout,
        notes=notes,
        raw=dict(raw),
        sha256=manifest_hash(raw),
    )


def load_manifest(path: Path) -> Manifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError([f"{path}: cannot read JSON ({exc})"]) from exc
    return parse_manifest(raw)
