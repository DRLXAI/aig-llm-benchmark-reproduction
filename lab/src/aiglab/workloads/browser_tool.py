"""browser_tool@1.0.0 — multi-step tool use against a local fixture website.

The "web" is a fixture site served by the harness itself on 127.0.0.1 (an ephemeral port), so
the workload has no external dependency and is reproducible. ``fetch_page`` performs a real HTTP
GET against that server; any other URL is refused, which keeps the run isolated even if the model
invents links. Tool latency is the measured duration of each tool execution.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from aiglab.adapters.base import ChatRequest
from aiglab.workloads.base import Chat, Task, TaskOutcome, ToolTimer, Workload, normalize_text

MAX_TURNS = 8

PAGES: dict[str, str] = {
    "/index.html": "Nimbus Compute home. Pages: {base}/pricing.html (pricing), {base}/regions.html (regions), "
    "{base}/blog.html (blog).",
    "/blog.html": "Nimbus blog. Latest post: 'Why we moved to fanless racks'. Back to {base}/index.html.",
    "/pricing.html": "Plans overview. Detailed plan pages: {base}/plans/atlas-4.html, {base}/plans/atlas-8.html, "
    "{base}/plans/atlas-16.html.",
    "/plans/atlas-4.html": "Atlas 4: 4 vCPU, 16 GB RAM. Monthly price: $19.20. Hourly price: $0.029.",
    "/plans/atlas-8.html": "Atlas 8: 8 vCPU, 32 GB RAM. Monthly price: $37.40. Hourly price: $0.056.",
    "/plans/atlas-16.html": "Atlas 16: 16 vCPU, 64 GB RAM. Monthly price: $74.80. Hourly price: $0.112.",
    "/regions.html": "Regions by continent: {base}/regions/eu.html and {base}/regions/us.html.",
    "/regions/eu.html": "EU regions: Frankfurt offers Atlas 4 and Atlas 8. Dublin offers Atlas 4 only.",
    "/regions/us.html": "US regions: Ashburn offers all plans. Portland offers Atlas 16 only.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": "Fetch a web page by absolute URL and return its text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "absolute http URL"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_finding",
            "description": "Save a named finding for the report.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                "required": ["key", "value"],
            },
        },
    },
]

SYSTEM = (
    "You are a browsing agent. Use fetch_page to read pages, follow links you find, and use save_finding to "
    "record the requested result before giving your final answer. Do not guess values you have not read."
)

TASKS = {
    "atlas8_price": {
        "prompt": "Starting at {base}/index.html, find the monthly price of the Atlas 8 plan. Save it with "
        "save_finding using key 'atlas8_monthly' and then reply with the price.",
        "key": "atlas8_monthly",
        "must": [r"37\.40"],
        "must_not": [r"19\.20", r"74\.80"],
    },
    "atlas16_regions": {
        "prompt": "Starting at {base}/index.html, find every region where the Atlas 16 plan is available. Save "
        "them with save_finding using key 'atlas16_regions' (comma-separated region names) and reply with "
        "the list.",
        "key": "atlas16_regions",
        "must": [r"ashburn", r"portland"],
        "must_not": [r"frankfurt", r"dublin"],
    },
}


class _Handler(BaseHTTPRequestHandler):
    base = ""

    def do_GET(self) -> None:
        body = PAGES.get(self.path.split("?", 1)[0])
        if body is None:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return
        data = body.format(base=self.base).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


class FixtureSite:
    def __init__(self) -> None:
        handler: type[_Handler] = type("SiteHandler", (_Handler,), {})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        host, port = self.server.server_address[:2]
        self.base = f"http://{host!s}:{port}"
        handler.base = self.base
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _args(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments") or {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return dict(raw)


class BrowserToolWorkload(Workload):
    id = "browser_tool"
    version = "1.0.0"
    planner_key = "browser_tool"
    title = "Browser / tool-heavy: multi-hop tool use"
    description = (
        "Two multi-hop browsing tasks over a harness-served fixture site using fetch_page and save_finding "
        "tool calls (max 8 model turns). Success = the saved finding and the final answer contain the "
        "correct values and no distractor values."
    )
    uses_tools = True
    isolated_services = ("fixture website served by the harness on 127.0.0.1 (ephemeral port)",)
    tool_latency_meaning = "median wall time of one tool execution (fetch_page HTTP GET to the local fixture site)"

    def __init__(self) -> None:
        self.site: FixtureSite | None = None

    def setup(self) -> None:
        self.site = FixtureSite()
        self.site.start()

    def teardown(self) -> None:
        if self.site is not None:
            self.site.stop()
            self.site = None

    def tasks(self) -> list[Task]:
        return [Task(id=name, prompt=str(spec["prompt"])) for name, spec in TASKS.items()]

    def _fetch(self, url: str) -> str:
        assert self.site is not None
        if not url.startswith(self.site.base + "/"):
            return f"error: only pages under {self.site.base}/ are reachable in this workload"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return str(resp.read().decode("utf-8"))
        except OSError as exc:
            return f"error: {exc}"

    def execute(self, task: Task, chat: Chat, lane: int) -> TaskOutcome:
        if self.site is None:
            raise RuntimeError("browser_tool workload used without setup()")
        spec = TASKS[task.id]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": task.prompt.format(base=self.site.base)},
        ]
        timer = ToolTimer()
        saved: dict[str, str] = {}
        fetches = 0
        outcome = TaskOutcome(task_id=task.id, lane=lane, success=False, failure_reason=None, wall_ms=0.0)
        t0 = time.perf_counter()
        final = ""
        for turn in range(MAX_TURNS):
            result = chat(ChatRequest(id=f"{task.id}#lane{lane}#turn{turn}", messages=list(messages), tools=TOOLS))
            outcome.requests.append(result.as_dict())
            if not result.ok:
                outcome.failure_reason = f"runtime: {result.error}"
                break
            if not result.tool_calls:
                final = result.content
                break
            messages.append({"role": "assistant", "content": result.content, "tool_calls": result.tool_calls})
            for call in result.tool_calls:
                name = str((call.get("function") or {}).get("name"))
                args = _args(call)
                with timer.time(name) as rec:
                    if name == "fetch_page":
                        fetches += 1
                        reply = self._fetch(str(args.get("url", "")))
                        rec.ok = not reply.startswith("error:")
                    elif name == "save_finding":
                        saved[str(args.get("key", ""))] = str(args.get("value", ""))
                        reply = "saved"
                    else:
                        rec.ok = False
                        reply = f"error: unknown tool {name!r}"
                messages.append({"role": "tool", "content": reply, "tool_name": name})
        else:
            outcome.failure_reason = f"no final answer within {MAX_TURNS} model turns"
        outcome.wall_ms = (time.perf_counter() - t0) * 1000
        outcome.tool_calls = timer.records
        outcome.output_excerpt = final[:4000]
        if outcome.failure_reason is None:
            c0 = time.perf_counter()
            value = normalize_text(saved.get(str(spec["key"]), ""))
            answer = normalize_text(final)
            problems = []
            if not value:
                problems.append(f"save_finding({spec['key']!r}) was not called")
            for pattern in spec["must"]:
                if not re.search(pattern, value):
                    problems.append(f"saved value lacks {pattern}")
                if not re.search(pattern, answer):
                    problems.append(f"final answer lacks {pattern}")
            for pattern in spec["must_not"]:
                if re.search(pattern, value):
                    problems.append(f"saved value contains distractor {pattern}")
            if fetches < 2:
                problems.append(f"only {fetches} fetch_page call(s); the answer needs at least 2 hops")
            outcome.check_ms = (time.perf_counter() - c0) * 1000
            outcome.success = not problems
            outcome.check_detail = "; ".join(problems) or f"correct after {fetches} fetches"
            outcome.failure_reason = None if outcome.success else outcome.check_detail
        return outcome
