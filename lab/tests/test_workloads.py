from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aiglab.adapters.base import ChatRequest, ChatResult
from aiglab.workloads import WORKLOADS, get_workload
from aiglab.workloads.automation import check_record
from aiglab.workloads.browser_tool import BrowserToolWorkload
from aiglab.workloads.coding import REFERENCES, TESTS, extract_code, run_sandboxed, sandbox_available
from aiglab.workloads.rag import BM25, CORPUS, QUESTIONS
from aiglab.workloads.research import TASKS as RESEARCH
from aiglab.workloads.research import check_answer

needs_sandbox = pytest.mark.skipif(not sandbox_available(), reason="sandbox-exec not available")


def test_registry_ids_are_unique_and_versioned() -> None:
    assert set(WORKLOADS) == {"coding", "research", "browser_tool", "rag", "automation"}
    for w in WORKLOADS.values():
        assert w.version == "1.0.0"
        assert len(w.definition_sha256()) == 64
        assert w.definition_sha256() == w.definition_sha256()


@needs_sandbox
@pytest.mark.parametrize("name", list(REFERENCES))
def test_coding_references_pass(name: str) -> None:
    ok, detail = run_sandboxed(REFERENCES[name], TESTS[name])
    assert ok, detail


@needs_sandbox
def test_coding_wrong_code_fails() -> None:
    ok, detail = run_sandboxed("def slugify(text):\n    return text.lower()\n", TESTS["slugify"])
    assert not ok and "hidden tests failed" in detail


@needs_sandbox
def test_coding_sandbox_blocks_network_and_writes(tmp_path: Path) -> None:
    target = tmp_path / "escaped.txt"
    ok, detail = run_sandboxed(f"open({str(target)!r}, 'w').write('x')\n", "import solution\n")
    assert not ok and "Operation not permitted" in detail and not target.exists()
    ok, detail = run_sandboxed(
        "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=3)\n", "import solution\n"
    )
    assert not ok and "Operation not permitted" in detail


def test_extract_code() -> None:
    assert extract_code("Here:\n```python\nx = 1\n```\nbye") == "x = 1\n"
    assert extract_code("x = 2") == "x = 2"


def test_research_checker() -> None:
    assert check_answer(str(RESEARCH["halvard"]["reference"]), list(RESEARCH["halvard"]["required"]))[0]
    ok, detail = check_answer("Tallinn in 2019 [S2].", list(RESEARCH["halvard"]["required"]))
    assert not ok and "missing required facts" in detail and "cited 1" in detail


def test_bm25_is_deterministic_and_finds_gold() -> None:
    index = BM25(CORPUS)
    for spec in QUESTIONS.values():
        hits = index.search(str(spec["question"]))
        assert hits == index.search(str(spec["question"]))
        assert set(spec["gold"]) <= set(hits)


def test_automation_checker() -> None:
    from aiglab.workloads.automation import TASKS

    good = json.dumps(TASKS["replace"]["expected"])
    assert check_record("replace", good)[0]
    bad = json.dumps(TASKS["replace"]["expected"] | {"due_date": "2026-05-12"})
    assert "due_date" in check_record("replace", bad)[1]
    assert not check_record("replace", "not json")[0]


def scripted_browser(base_holder: dict[str, str]) -> Any:
    """A scripted 'model' that browses correctly: index -> pricing -> atlas-8 -> save -> answer."""
    steps: dict[str, int] = {}

    def respond(req: ChatRequest) -> ChatResult:
        task = req.id.split("#")[0]
        n = steps.get(req.id.split("#turn")[0], 0)
        steps[req.id.split("#turn")[0]] = n + 1
        base = base_holder["base"]
        plan = {
            "atlas8_price": [
                ("fetch_page", {"url": f"{base}/index.html"}),
                ("fetch_page", {"url": f"{base}/pricing.html"}),
                ("fetch_page", {"url": f"{base}/plans/atlas-8.html"}),
                ("save_finding", {"key": "atlas8_monthly", "value": "$37.40"}),
                None,
            ],
            "atlas16_regions": [("fetch_page", {"url": "https://evil.example/"}), None],
        }[task]
        step = plan[min(n, len(plan) - 1)]
        if step is None:
            return ChatResult(id=req.id, origin="fixture", ok=True, content="The price is $37.40.")
        name, args = step
        return ChatResult(
            id=req.id, origin="fixture", ok=True, tool_calls=[{"function": {"name": name, "arguments": args}}]
        )

    return respond


def test_browser_tool_success_path_measures_tool_latency() -> None:
    wl = BrowserToolWorkload()
    wl.setup()
    try:
        assert wl.site is not None
        respond = scripted_browser({"base": wl.site.base})
        tasks = {t.id: t for t in wl.tasks()}
        outcome = wl.execute(tasks["atlas8_price"], respond, lane=0)
        assert outcome.success, outcome.failure_reason
        assert [t.name for t in outcome.tool_calls] == ["fetch_page"] * 3 + ["save_finding"]
        assert all(t.latency_ms > 0 for t in outcome.tool_calls[:3])
        off_site = wl.execute(tasks["atlas16_regions"], respond, lane=0)
        assert not off_site.success
        assert off_site.tool_calls[0].ok is False  # off-site URL refused: the workload stays isolated
    finally:
        wl.teardown()


def test_every_workload_declares_external_dependencies() -> None:
    for w in WORKLOADS.values():
        assert w.external_dependencies == ()  # all external services are isolated in v1.0.0
    assert get_workload("browser_tool").isolated_services
