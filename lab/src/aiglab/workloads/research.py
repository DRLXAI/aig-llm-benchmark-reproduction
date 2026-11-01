"""research@1.0.0 — synthesise an answer from supplied sources, including a superseded figure.

The sources describe fictional organisations, so a model cannot answer from memorised facts;
success requires reading. The checker is deterministic: every required fact must appear in the
answer (case/comma-insensitive regexes) and at least two sources must be cited as [S1]..[S3].
"""

from __future__ import annotations

import re

from aiglab.adapters.base import ChatRequest
from aiglab.workloads.base import Chat, Task, TaskOutcome, Workload, normalize_text, single_turn

SYSTEM = (
    "You are a research analyst. Answer only from the numbered sources provided. Be concise, cite every "
    "claim with its source tag such as [S1], and when sources disagree prefer the most recent one."
)

TASKS = {
    "halvard": {
        "sources": [
            "[S1] Press release, March 2023: Halvard Data Systems opened its second data center, in Porto. "
            "The Porto site has 2,400 racks.",
            "[S2] Annual report excerpt: In fiscal 2024, Halvard's revenue grew 18% to EUR 412 million. The "
            "company's first data center, in Tallinn, opened in 2019.",
            "[S3] Interview, 2024: CTO Mirela Ostrova said the Porto facility runs on contracted wind power, "
            "while Tallinn uses a grid mix.",
        ],
        "question": (
            "In what year and in which city did Halvard open its first data center? How many racks does the "
            "second site have? By what percentage did revenue grow in fiscal 2024? Who is the CTO?"
        ),
        "required": [r"\b2019\b", r"tallinn", r"\b2400\b", r"\b18 ?%", r"ostrova"],
        "reference": (
            "Halvard opened its first data center in Tallinn in 2019 [S2]. The second site, Porto, has 2,400 "
            "racks [S1]. Fiscal 2024 revenue grew 18% [S2]. The CTO is Mirela Ostrova [S3]."
        ),
    },
    "brightwater": {
        "sources": [
            "[S1] 2022 project plan: The Brightwater water main replacement has an estimated cost of $58 "
            "million, with completion expected in 2025.",
            "[S2] Council minutes, May 2024: Staff revised the cost estimate to $71 million after pipe material "
            "costs rose, and moved expected completion to 2027.",
            "[S3] Local news, June 2024: The council approved the revised Brightwater plan by a vote of 7 to 2.",
        ],
        "question": (
            "What is the most recent cost estimate and expected completion year for the Brightwater project, "
            "why did the estimate change, and what was the council vote on the revised plan?"
        ),
        "required": [r"\$? ?71", r"\b2027\b", r"pipe|material", r"\b7 ?(?:to|-|–) ?2\b"],
        "reference": (
            "The latest estimate is $71 million with completion in 2027 [S2], revised because pipe material "
            "costs rose [S2]. The council approved the revised plan 7 to 2 [S3]."
        ),
    },
}

CITATION = re.compile(r"\[S([123])\]")


def check_answer(answer: str, required: list[str]) -> tuple[bool, str]:
    text = normalize_text(answer)
    missing = [pattern for pattern in required if not re.search(pattern, text)]
    cited = sorted(set(CITATION.findall(answer)))
    problems = []
    if missing:
        problems.append(f"missing required facts {missing}")
    if len(cited) < 2:
        problems.append(f"cited {len(cited)} source(s), need at least 2")
    return (not problems), ("; ".join(problems) if problems else f"all facts present; cited S{','.join(cited)}")


class ResearchWorkload(Workload):
    id = "research"
    version = "1.0.0"
    planner_key = "research"
    title = "Research: cited synthesis from supplied sources"
    description = (
        "Two questions over three short fictional sources each (one with a superseded figure). Success = "
        "every required fact present and at least two sources cited."
    )

    def tasks(self) -> list[Task]:
        return [
            Task(id=name, prompt=self.prompt(name), fixture_reply=str(spec["reference"]))
            for name, spec in TASKS.items()
        ]

    @staticmethod
    def prompt(name: str) -> str:
        spec = TASKS[name]
        sources = "\n".join(spec["sources"])
        return f"Sources:\n{sources}\n\nQuestion: {spec['question']}"

    def execute(self, task: Task, chat: Chat, lane: int) -> TaskOutcome:
        request = ChatRequest(
            id=f"{task.id}#lane{lane}",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.prompt}],
        )
        outcome, result = single_turn(task, chat, lane, request)
        if outcome.failure_reason is None:
            ok, detail = check_answer(result.content, list(TASKS[task.id]["required"]))
            outcome.success, outcome.check_detail = ok, detail
            outcome.failure_reason = None if ok else detail
        return outcome
