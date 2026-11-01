"""Workload contract.

A workload is a fixed, versioned set of tasks with deterministic success checks. Prompts,
fixture data and checkers are part of the workload version: changing any of them requires a
version bump, and ``definition_sha256`` (hash of the module source plus fixture files) makes an
unannounced change visible in every run record.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from aiglab.adapters.base import ChatRequest, ChatResult

Chat = Callable[[ChatRequest], ChatResult]


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    #: what the fixture adapter answers; used only by tests and pipeline dry-runs.
    fixture_reply: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    name: str
    ok: bool
    latency_ms: float
    detail: str = ""


@dataclass
class TaskOutcome:
    task_id: str
    lane: int
    success: bool
    failure_reason: str | None
    #: task wall clock: model requests plus the agent's own tool/retrieval steps. Excludes grading.
    wall_ms: float
    requests: list[dict[str, Any]] = field(default_factory=list)
    #: time the harness spent grading the output (never part of completion time).
    check_ms: float = 0.0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    output_excerpt: str = ""
    check_detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["tool_calls"] = [asdict(t) for t in self.tool_calls]
        return out


class ToolTimer:
    def __init__(self) -> None:
        self.records: list[ToolCallRecord] = []

    @contextmanager
    def time(self, name: str) -> Iterator[ToolCallRecord]:
        rec = ToolCallRecord(name=name, ok=True, latency_ms=0.0)
        t0 = time.perf_counter()
        try:
            yield rec
        except Exception as exc:
            rec.ok = False
            rec.detail = f"{type(exc).__name__}: {exc}"[:300]
            raise
        finally:
            rec.latency_ms = (time.perf_counter() - t0) * 1000
            self.records.append(rec)


class Workload:
    id: ClassVar[str]
    version: ClassVar[str]
    planner_key: ClassVar[str]
    title: ClassVar[str]
    description: ClassVar[str]
    uses_tools: ClassVar[bool] = False
    #: services outside the harness process the workload depends on, if any.
    external_dependencies: ClassVar[tuple[str, ...]] = ()
    #: services the harness starts itself to isolate what would otherwise be external.
    isolated_services: ClassVar[tuple[str, ...]] = ()
    #: what tool_latency means for this workload, or None when not applicable.
    tool_latency_meaning: ClassVar[str | None] = None
    fixture_dir: ClassVar[Path | None] = None

    def tasks(self) -> list[Task]:
        raise NotImplementedError

    def setup(self) -> None:
        """Start isolated services. Called once per run, before any request."""

    def teardown(self) -> None:
        """Stop isolated services."""

    def execute(self, task: Task, chat: Chat, lane: int) -> TaskOutcome:
        raise NotImplementedError

    def requires_capabilities(self) -> tuple[str, ...]:
        return ("tools",) if self.uses_tools else ()

    @classmethod
    def definition_sha256(cls) -> str:
        h = hashlib.sha256()
        h.update(inspect.getsource(inspect.getmodule(cls) or cls).encode())
        h.update(inspect.getsource(Workload).encode())
        if cls.fixture_dir is not None and cls.fixture_dir.is_dir():
            for path in sorted(cls.fixture_dir.rglob("*")):
                if path.is_file():
                    h.update(path.relative_to(cls.fixture_dir).as_posix().encode())
                    h.update(path.read_bytes())
        return h.hexdigest()

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "id": cls.id,
            "version": cls.version,
            "planner_key": cls.planner_key,
            "title": cls.title,
            "description": cls.description,
            "uses_tools": cls.uses_tools,
            "external_dependencies": list(cls.external_dependencies),
            "isolated_services": list(cls.isolated_services),
            "tool_latency_meaning": cls.tool_latency_meaning,
            "definition_sha256": cls.definition_sha256(),
        }


def single_turn(task: Task, chat: Chat, lane: int, request: ChatRequest) -> tuple[TaskOutcome, ChatResult]:
    t0 = time.perf_counter()
    result = chat(request)
    outcome = TaskOutcome(
        task_id=task.id,
        lane=lane,
        success=False,
        failure_reason=None,
        wall_ms=(time.perf_counter() - t0) * 1000,
        requests=[result.as_dict()],
        output_excerpt=result.content[:4000],
    )
    if not result.ok:
        outcome.failure_reason = f"runtime: {result.error}"
    elif result.done_reason == "length":
        outcome.failure_reason = "output truncated at max_output_tokens"
    return outcome, result


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace(",", "").split())
