"""automation@1.0.0 — turn unstructured requests into a fixed JSON record.

Each task is a fictional support email that must become one JSON object matching a schema. The
schema is passed to the runtime as a structured-output constraint (Ollama ``format``), as an
automation pipeline would. Success = valid JSON and every field equal to the expected value
(strings compared case-insensitively after trimming; dates exactly).
"""

from __future__ import annotations

import json
from typing import Any

from aiglab.adapters.base import ChatRequest
from aiglab.workloads.base import Chat, Task, TaskOutcome, Workload, single_turn

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
        "action": {"type": "string", "enum": ["refund", "replace", "cancel", "escalate"]},
        "order_id": {"type": "string"},
        "due_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
    },
    "required": ["customer", "priority", "action", "order_id", "due_date"],
}

SYSTEM = (
    "Extract the request into JSON with fields customer, priority (low|normal|urgent), action "
    "(refund|replace|cancel|escalate), order_id and due_date (YYYY-MM-DD). The email date is given; resolve "
    "relative dates against it. Output only the JSON object."
)

TASKS: dict[str, dict[str, Any]] = {
    "refund": {
        "email": "Date: 2026-03-02\nFrom: Priya Raman, Larchwood Bakery\nHi, order KX-4471 arrived crushed. "
        "We'd like our money back please, no rush, but by 2026-03-20 would be ideal.",
        "expected": {
            "customer": "Larchwood Bakery",
            "priority": "normal",
            "action": "refund",
            "order_id": "KX-4471",
            "due_date": "2026-03-20",
        },
    },
    "replace": {
        "email": "Date: 2026-05-11\nFrom: Tomas Hale (Northpine Clinic)\nURGENT: the label printer from order "
        "KX-5102 is dead and we can't print patient labels. Please send a replacement unit within 2 days.",
        "expected": {
            "customer": "Northpine Clinic",
            "priority": "urgent",
            "action": "replace",
            "order_id": "KX-5102",
            "due_date": "2026-05-13",
        },
    },
    "cancel": {
        "email": "Date: 2026-07-30\nFrom: Ada Ferris, Ferris & Sons Hardware\nPlease cancel order KX-6020 "
        "before it ships. Whenever you get to it this week is fine; it ships on 2026-08-04.",
        "expected": {
            "customer": "Ferris & Sons Hardware",
            "priority": "low",
            "action": "cancel",
            "order_id": "KX-6020",
            "due_date": "2026-08-04",
        },
    },
}

LENIENT_PRIORITY = {"cancel": {"low", "normal"}}


def check_record(task_id: str, content: str) -> tuple[bool, str]:
    try:
        record = json.loads(content)
    except json.JSONDecodeError as exc:
        return False, f"output is not valid JSON: {exc}"
    if not isinstance(record, dict):
        return False, "output is not a JSON object"
    expected = TASKS[task_id]["expected"]
    wrong = []
    for key, want in expected.items():
        got = str(record.get(key, "")).strip()
        if key == "priority" and got.lower() in LENIENT_PRIORITY.get(task_id, set()):
            continue
        if key == "due_date":
            if got != want:
                wrong.append(f"{key}={got!r} (want {want!r})")
        elif got.lower() != want.lower():
            wrong.append(f"{key}={got!r} (want {want!r})")
    return (not wrong), ("all fields correct" if not wrong else "wrong fields: " + ", ".join(wrong))


class AutomationWorkload(Workload):
    id = "automation"
    version = "1.0.0"
    planner_key = "automation"
    title = "Automation: structured extraction to a JSON schema"
    description = (
        "Three fictional support emails to JSON records (customer, priority, action, order_id, due_date) "
        "using the runtime's schema-constrained output. Success = every field matches, including resolved "
        "relative dates."
    )

    def tasks(self) -> list[Task]:
        return [Task(id=k, prompt=str(v["email"]), fixture_reply=json.dumps(v["expected"])) for k, v in TASKS.items()]

    def execute(self, task: Task, chat: Chat, lane: int) -> TaskOutcome:
        request = ChatRequest(
            id=f"{task.id}#lane{lane}",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.prompt}],
            format=SCHEMA,
        )
        outcome, result = single_turn(task, chat, lane, request)
        if outcome.failure_reason is None:
            ok, detail = check_record(task.id, result.content)
            outcome.success, outcome.check_detail = ok, detail
            outcome.failure_reason = None if ok else detail
        return outcome
