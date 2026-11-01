"""rag@1.0.0 — retrieve from a local knowledge base, then answer from the retrieved passages.

Retrieval is a deterministic BM25 implementation over a fixed fictional corpus (no embedding
model, no external service), so the only model-dependent step is answering. The corpus contains
a deliberate distractor (a retired VPN port). tool_latency for this workload is the retrieval
step's wall time; retrieval recall against the gold documents is recorded per task.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

from aiglab.adapters.base import ChatRequest
from aiglab.workloads.base import Chat, Task, TaskOutcome, ToolTimer, Workload, normalize_text, single_turn

CORPUS: dict[str, str] = {
    "vpn-runbook": "Kestrel VPN runbook. The current Kestrel VPN gateway listens on port 8443. Rotate the "
    "gateway certificate every 90 days.",
    "vpn-legacy": "Legacy VPN notes. The legacy VPN, retired in 2023, used port 1194 and is no longer reachable.",
    "incident-policy": "Incident policy. Severity 1 incidents must be acknowledged within 15 minutes and a "
    "postmortem is due within 72 hours.",
    "oncall": "On-call guide. Escalations go to ops-oncall@kestrel.example. The secondary on-call is paged after "
    "10 minutes without acknowledgement.",
    "backup-policy": "Backup policy. Nightly database backups run at 02:30 UTC and are retained for 35 days. "
    "Weekly backups are retained for 1 year.",
    "access-policy": "Access policy. Production access requires a hardware security key. Access reviews happen "
    "quarterly.",
    "deploy-guide": "Deploy guide. Deploy freezes start every Friday at 16:00 local time. Rollbacks use the kctl "
    "rollback command.",
    "vendor-policy": "Vendor policy. New vendors require a security review that takes up to 10 business days.",
    "laptop-policy": "Laptop policy. Laptops must enable full-disk encryption. The replacement cycle is 4 years.",
    "dr-plan": "Disaster recovery plan. Tier-1 services target an RPO of 1 hour and an RTO of 4 hours.",
    "travel-policy": "Travel policy. Hotel stays are reimbursed up to 180 dollars per night.",
}

QUESTIONS = {
    "vpn_port": {
        "question": "What port does the current Kestrel VPN gateway listen on, and how often is its certificate "
        "rotated?",
        "gold": ["vpn-runbook"],
        "must": [r"\b8443\b", r"\b90 days?\b"],
        "must_not": [r"\b1194\b"],
        "reference": "The current gateway listens on port 8443 and its certificate is rotated every 90 days.",
    },
    "backups": {
        "question": "How long are nightly database backups retained, and at what time do they run?",
        "gold": ["backup-policy"],
        "must": [r"\b35 days?\b", r"02:30"],
        "must_not": [],
        "reference": "Nightly backups run at 02:30 UTC and are retained for 35 days.",
    },
    "sev1": {
        "question": "Within how long is a postmortem due for a Severity 1 incident, and which address receives "
        "escalations?",
        "gold": ["incident-policy", "oncall"],
        "must": [r"\b72 hours?\b", r"ops-oncall@kestrel\.example"],
        "must_not": [],
        "reference": "A postmortem is due within 72 hours; escalations go to ops-oncall@kestrel.example.",
    },
}

TOP_K = 3
SYSTEM = (
    "Answer the question using only the provided context passages. Be brief and exact. If the context does "
    "not contain the answer, say so."
)
TOKEN = re.compile(r"[a-z0-9@.:-]+")


def tokenize(text: str) -> list[str]:
    return [t.strip(".:") for t in TOKEN.findall(text.lower()) if t.strip(".:")]


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = sorted(docs)
        self.tokens = {d: tokenize(docs[d]) for d in self.ids}
        self.k1, self.b = k1, b
        self.avgdl = sum(len(t) for t in self.tokens.values()) / len(self.ids)
        df: Counter[str] = Counter()
        for toks in self.tokens.values():
            df.update(set(toks))
        n = len(self.ids)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = TOP_K) -> list[str]:
        q = tokenize(query)
        scores: list[tuple[float, str]] = []
        for d in self.ids:
            tf = Counter(self.tokens[d])
            dl = len(self.tokens[d])
            s = 0.0
            for term in q:
                if term in tf:
                    f = tf[term]
                    s += (
                        self.idf.get(term, 0.0)
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    )
            scores.append((s, d))
        scores.sort(key=lambda x: (-x[0], x[1]))  # ties broken by id: deterministic
        return [d for _, d in scores[:k]]


class RagWorkload(Workload):
    id = "rag"
    version = "1.0.0"
    planner_key = "rag"
    title = "RAG: retrieve then answer"
    description = (
        "Three questions over an 11-document fictional knowledge base (one distractor). Deterministic BM25 "
        "top-3 retrieval, then a grounded answer. Success = required values present, distractor absent."
    )
    tool_latency_meaning = "median wall time of the retrieval step (in-process BM25 top-3)"

    def __init__(self) -> None:
        self.index = BM25(CORPUS)

    def tasks(self) -> list[Task]:
        return [Task(id=k, prompt=str(v["question"]), fixture_reply=str(v["reference"])) for k, v in QUESTIONS.items()]

    def execute(self, task: Task, chat: Chat, lane: int) -> TaskOutcome:
        spec = QUESTIONS[task.id]
        timer = ToolTimer()
        t0 = time.perf_counter()
        with timer.time("retrieve"):
            hits = self.index.search(task.prompt)
        context = "\n".join(f"[{d}] {CORPUS[d]}" for d in hits)
        request = ChatRequest(
            id=f"{task.id}#lane{lane}",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context passages:\n{context}\n\nQuestion: {task.prompt}"},
            ],
        )
        outcome, result = single_turn(task, chat, lane, request)
        outcome.wall_ms = (time.perf_counter() - t0) * 1000
        outcome.tool_calls = timer.records
        gold = list(spec["gold"])
        recall = sum(1 for g in gold if g in hits) / len(gold)
        outcome.check_detail = f"retrieved {hits}; gold recall {recall:.2f}"
        if outcome.failure_reason is None:
            answer = normalize_text(result.content)
            problems = [f"missing {p}" for p in spec["must"] if not re.search(p, answer)]
            problems += [f"contains distractor {p}" for p in spec["must_not"] if re.search(p, answer)]
            outcome.success = not problems
            if problems:
                outcome.failure_reason = "; ".join(problems) + f" ({outcome.check_detail})"
        return outcome
