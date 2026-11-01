"""Versioned methodology constants. Changing what a metric means requires a new version.

The prose methodology lives in docs/measurement-lab/METHODOLOGY.md and must be kept in step with
METHODOLOGY_VERSION. A record is only comparable with another record of the same major version.
"""

from __future__ import annotations

METHODOLOGY_VERSION = "1.0.0"
SUPPORTED_METHODOLOGY_VERSIONS = frozenset({"1.0.0"})
METHODOLOGY_DOC = "docs/measurement-lab/METHODOLOGY.md"

#: Memory sampling cadence. Peak memory is the maximum of samples, so it is a lower bound on the
#: true peak by construction; the interval is recorded with every measurement.
RSS_SAMPLE_INTERVAL_S = 0.25
FOOTPRINT_SAMPLE_INTERVAL_S = 2.0

#: Outlier rule (robust z-score on the median absolute deviation). Flagged, never dropped.
OUTLIER_MODIFIED_Z = 3.5

#: Confidence rule for a published aggregate (see METHODOLOGY.md "Confidence").
CONFIDENCE_HIGH_MIN_RUNS = 5
CONFIDENCE_MEDIUM_MIN_RUNS = 3
CONFIDENCE_HIGH_MAX_CV = 0.10

#: Freshness, in days since tested_at. 90 matches the Planner Data Studio "stale" flag.
FRESH_CURRENT_DAYS = 90
FRESH_REVALIDATE_DAYS = 180

#: Host gate: kern.memorystatus_vm_pressure_level. 1 normal, 2 warn, 4 critical.
PRESSURE_BLOCK_LEVEL = 4


def major(version: str) -> str:
    return version.split(".", 1)[0]


def comparable(a: str, b: str) -> bool:
    return major(a) == major(b)
