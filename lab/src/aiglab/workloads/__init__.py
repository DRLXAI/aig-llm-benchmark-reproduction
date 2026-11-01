"""Workload registry. A workload id is also its Planner workload key (Base44 WORKLOAD_TYPES)."""

from __future__ import annotations

from aiglab.workloads.automation import AutomationWorkload
from aiglab.workloads.base import Workload
from aiglab.workloads.browser_tool import BrowserToolWorkload
from aiglab.workloads.coding import CodingWorkload
from aiglab.workloads.rag import RagWorkload
from aiglab.workloads.research import ResearchWorkload

WORKLOADS: dict[str, type[Workload]] = {
    w.id: w for w in (CodingWorkload, ResearchWorkload, BrowserToolWorkload, RagWorkload, AutomationWorkload)
}


def workload_versions() -> dict[str, str]:
    return {k: v.version for k, v in WORKLOADS.items()}


def get_workload(workload_id: str) -> Workload:
    return WORKLOADS[workload_id]()
