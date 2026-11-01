"""Evidence classes and the one function allowed to decide `aig_tested`.

The five evidence classes are exactly the Planner's (Base44 entity enum). The lab adds an
*origin* to every run, which records how the numbers came to exist. Only one origin —
``executed``, produced by an adapter that genuinely drives a real runtime — can ever lead to
``aig_tested``, and even then only when every integrity check passes. Everything else is
downgraded with the reasons attached. There is no override flag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, get_args

from aiglab.methodology import SUPPORTED_METHODOLOGY_VERSIONS

EvidenceType = Literal["aig_tested", "primary_documented", "calculated", "vendor_claim", "unknown"]
EVIDENCE_TYPES: Final[tuple[str, ...]] = get_args(EvidenceType)

Origin = Literal["executed", "fixture", "simulated", "imported", "estimated"]
ORIGINS: Final[tuple[str, ...]] = get_args(Origin)

#: Adapters whose code path executes a real runtime on real hardware. Adding an id here is a
#: methodology decision, reviewed like code: the adapter must measure, not compute.
EXECUTING_ADAPTERS: Final[frozenset[str]] = frozenset({"ollama"})

#: Evidence types an imported (third-party) record may carry. Never aig_tested.
IMPORT_ALLOWED: Final[frozenset[str]] = frozenset({"primary_documented", "vendor_claim", "calculated", "unknown"})


class EvidenceIntegrityError(ValueError):
    """Raised when something tries to publish an evidence class it has not earned."""


@dataclass(frozen=True)
class Classification:
    evidence_type: EvidenceType
    origin: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def aig_tested(self) -> bool:
        return self.evidence_type == "aig_tested"


def classify_run(run: Mapping[str, Any], *, seal_ok: bool) -> Classification:
    """Decide the evidence class of a run record (the parsed ``run.json``).

    ``seal_ok`` is the result of re-hashing the run directory against its SHA256SUMS; the caller
    must compute it from disk, never take it from the record.
    """
    origin = str(run.get("origin") or "")
    if origin not in ORIGINS:
        return Classification("unknown", origin or "missing", (f"unrecognised origin {origin!r}",))

    if origin in ("fixture", "simulated"):
        return Classification("unknown", origin, (f"{origin} data is test-only and is never evidence",))
    if origin == "estimated":
        return Classification("calculated", origin, ("estimates are calculations, not measurements",))
    if origin == "imported":
        declared = str(run.get("declared_evidence_type") or "unknown")
        if declared not in IMPORT_ALLOWED:
            return Classification(
                "unknown",
                origin,
                (f"imported data may not claim {declared!r}; only this harness can produce aig_tested",),
            )
        return Classification(declared, origin, ("imported third-party data keeps its declared class",))  # type: ignore[arg-type]

    reasons = list(_executed_failures(run, seal_ok=seal_ok))
    if reasons:
        return Classification("unknown", origin, tuple(reasons))
    return Classification("aig_tested", origin, ())


def _executed_failures(run: Mapping[str, Any], *, seal_ok: bool) -> list[str]:
    failures: list[str] = []
    adapter = run.get("adapter") or {}
    adapter_id = str(adapter.get("id") or "")
    if adapter_id not in EXECUTING_ADAPTERS:
        failures.append(f"adapter {adapter_id!r} is not a registered executing adapter")
    if adapter.get("executes_real_runtime") is not True:
        failures.append("adapter did not declare that it executes a real runtime")
    if not seal_ok:
        failures.append("artifact seal (SHA256SUMS) is missing or does not match the files on disk")
    if run.get("status") != "completed":
        failures.append(f"harness status is {run.get('status')!r}, not 'completed'")
    if str(run.get("methodology_version")) not in SUPPORTED_METHODOLOGY_VERSIONS:
        failures.append(f"methodology version {run.get('methodology_version')!r} is not supported")

    reps = run.get("repetitions") or []
    measured = [r for r in reps if r.get("phase") == "measured"]
    if not measured:
        failures.append("no measured repetition was executed")
    for rep in reps:
        for req in rep.get("requests") or []:
            if req.get("origin", "executed") != "executed":
                failures.append(f"request {req.get('id')} has origin {req.get('origin')!r}")
                break

    identity = run.get("identity_checks") or {}
    if not identity:
        failures.append("runtime identity was not verified")
    for name, check in sorted(identity.items()):
        if check.get("ok") is not True:
            failures.append(f"identity check {name!r} failed: {check.get('detail')}")
    return failures


def assert_publishable(claimed: str, classification: Classification) -> None:
    """Refuse any record whose claimed class is stronger than its classification allows."""
    if claimed not in EVIDENCE_TYPES:
        raise EvidenceIntegrityError(f"{claimed!r} is not one of {EVIDENCE_TYPES}")
    if claimed == "aig_tested" and not classification.aig_tested:
        raise EvidenceIntegrityError(
            "refusing to publish aig_tested for a run classified "
            f"{classification.evidence_type!r}: {'; '.join(classification.reasons)}"
        )
