"""coding@1.0.0 — implement three small, fully specified Python functions; hidden unit tests decide.

Generated code is executed only inside a macOS seatbelt profile (``sandbox-exec``) that denies
all network access and all writes outside a fresh temporary directory, with a wall-clock limit.
If the sandbox is unavailable the checker refuses to execute model output and the task is
recorded as a failure with that reason — never as a pass.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aiglab.adapters.base import ChatRequest
from aiglab.workloads.base import Chat, Task, TaskOutcome, Workload, single_turn

SYSTEM = (
    "You are a careful Python engineer. Reply with exactly one ```python code block containing the "
    "complete function and any imports it needs. No explanations, no tests, no example usage."
)

SLUGIFY_REF = """import re

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
"""

MERGE_REF = """def merge_intervals(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out
"""

DURATION_REF = """import re

def parse_duration(text: str) -> int:
    m = re.fullmatch(r"(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?", text)
    if not text or m is None:
        raise ValueError(text)
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + s
"""

TESTS = {
    "slugify": """
from solution import slugify
assert slugify("Hello, World!") == "hello-world", slugify("Hello, World!")
assert slugify("  --A  B--  ") == "a-b"
assert slugify("Ünïcode 2024") == "n-code-2024", slugify("Ünïcode 2024")
assert slugify("") == ""
assert slugify("already-a-slug") == "already-a-slug"
assert slugify("Ops__Runbook   v2") == "ops-runbook-v2"
""",
    "merge_intervals": """
from solution import merge_intervals
def norm(r): return [tuple(x) for x in r]
assert norm(merge_intervals([])) == []
assert norm(merge_intervals([(5, 6), (1, 3), (2, 4)])) == [(1, 4), (5, 6)]
assert norm(merge_intervals([(1, 3), (3, 5)])) == [(1, 5)]
assert norm(merge_intervals([(1, 2), (3, 4)])) == [(1, 2), (3, 4)]
assert norm(merge_intervals([(1, 10), (2, 3), (4, 12)])) == [(1, 12)]
assert norm(merge_intervals([(7, 7)])) == [(7, 7)]
""",
    "parse_duration": """
from solution import parse_duration
cases = {"1h30m": 5400, "45s": 45, "2h": 7200, "10m5s": 605, "1h0m1s": 3601, "0s": 0}
for text, want in cases.items():
    got = parse_duration(text)
    assert got == want, (text, got, want)
for bad in ["", "5x", "1m1h", "h", "1.5h", "10 m"]:
    try:
        parse_duration(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"no ValueError for {bad!r}")
""",
}

PROMPTS = {
    "slugify": (
        "Write a Python function `slugify(text: str) -> str` that lowercases the text, replaces every run "
        "of characters that are not ASCII letters a-z or digits 0-9 with a single hyphen, and strips leading "
        'and trailing hyphens. Example: slugify("  Hello, World!  ") == "hello-world".'
    ),
    "merge_intervals": (
        "Write a Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` "
        "that merges overlapping or touching closed intervals and returns them as tuples sorted by start. "
        "Touching means (1, 3) and (3, 5) merge into (1, 5); (1, 2) and (3, 4) stay separate. "
        "The input may be unsorted and may be empty."
    ),
    "parse_duration": (
        "Write a Python function `parse_duration(text: str) -> int` that converts strings like '1h30m', '45s', "
        "'2h', '10m5s' and '1h0m1s' to a number of seconds. The units are h, m and s; each appears at most "
        "once, in that order, preceded by a non-negative integer with no spaces. Raise ValueError for an "
        "empty string or for anything else that does not match this format."
    ),
}

REFERENCES = {"slugify": SLUGIFY_REF, "merge_intervals": MERGE_REF, "parse_duration": DURATION_REF}
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)
EXEC_TIMEOUT_S = 20


def extract_code(text: str) -> str:
    m = CODE_BLOCK.search(text)
    return m.group(1) if m else text


def sandbox_available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


def run_sandboxed(code: str, test: str) -> tuple[bool, str]:
    """Execute model code + hidden tests with no network and writes confined to a temp dir."""
    if not sandbox_available():
        return False, "checker refused to execute model output: sandbox-exec is not available on this host"
    with tempfile.TemporaryDirectory(prefix="aiglab-coding-") as tmp:
        root = Path(tmp).resolve()
        (root / "solution.py").write_text(code, encoding="utf-8")
        (root / "test_hidden.py").write_text(test, encoding="utf-8")
        profile = (
            "(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* (subpath "{root}") (literal "/dev/null"))'
        )
        try:
            proc = subprocess.run(
                # -E -s: ignore PYTHON* env vars and user site-packages. Not -I: it implies -P, which
                # would drop the script directory that `from solution import ...` relies on.
                ["sandbox-exec", "-p", profile, sys.executable, "-E", "-s", "-B", "test_hidden.py"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT_S,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8"},
            )
        except subprocess.TimeoutExpired:
            return False, f"hidden tests exceeded {EXEC_TIMEOUT_S}s"
        tail = (proc.stdout + proc.stderr).strip()[-600:]
        return proc.returncode == 0, ("hidden tests passed" if proc.returncode == 0 else f"hidden tests failed: {tail}")


class CodingWorkload(Workload):
    id = "coding"
    version = "1.0.0"
    planner_key = "coding"
    title = "Coding: implement specified functions"
    description = (
        "Three fully specified Python functions (slugify, merge_intervals, parse_duration). Success = the "
        "hidden unit tests pass when the model's code runs in a no-network, write-confined sandbox."
    )
    isolated_services = ("sandbox-exec seatbelt for executing model output (no network, temp-dir writes only)",)

    def tasks(self) -> list[Task]:
        return [
            Task(id=name, prompt=PROMPTS[name], fixture_reply=f"```python\n{REFERENCES[name]}```")
            for name in ("slugify", "merge_intervals", "parse_duration")
        ]

    def execute(self, task: Task, chat: Chat, lane: int) -> TaskOutcome:
        request = ChatRequest(
            id=f"{task.id}#lane{lane}",
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.prompt}],
        )
        outcome, result = single_turn(task, chat, lane, request)
        if outcome.failure_reason is None:
            t0 = time.perf_counter()
            ok, detail = run_sandboxed(extract_code(result.content), TESTS[task.id])
            outcome.check_ms = (time.perf_counter() - t0) * 1000
            outcome.success = ok
            outcome.check_detail = detail
            outcome.failure_reason = None if ok else detail
        return outcome
