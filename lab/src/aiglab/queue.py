"""File-based benchmark queue, built for Estate's deterministic-job contract.

Estate starts ``aiglab queue run-next`` with no model and no user text in argv; the queue holds
the manifests (added by a person or a reviewed change), so *what* runs is decided before Estate
is involved. One item runs per invocation. Exit codes follow Estate: 0 healthy, 2 findings,
3 incomplete/blocked.

    <state>/queue/<item>.json   item state + full history (never deleted)
    <state>/runs/<run_id>/      sealed run directories (import into the repo with `aiglab import-runs`)
    <state>/queue.lock          claim lock; the host benchmark lock is runner.DEFAULT_LOCK, shared with
                                manual `aiglab run`, so a queued and a manual benchmark never overlap

Retry policy: an aborted run returns to ``pending`` until ``max_attempts``; a run blocked by a
transient condition (another benchmark running) returns to pending; any other blocked preflight
stays ``blocked`` with its reasons until someone runs ``aiglab queue retry <id>``.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiglab.artifacts import read_json, write_json
from aiglab.manifest import Manifest, parse_manifest
from aiglab.normalize import load_run
from aiglab.runner import DEFAULT_LOCK, RunOutcome, execute

STATES = ("pending", "running", "succeeded", "failed", "blocked")
TRANSIENT_BLOCKS = ("another aiglab run",)
Executor = Callable[..., RunOutcome]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextlib.contextmanager
def _locked(state_dir: Path) -> Iterator[None]:
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "queue.lock").open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def _items(state_dir: Path) -> list[dict[str, Any]]:
    qdir = state_dir / "queue"
    if not qdir.is_dir():
        return []
    return sorted((read_json(p) for p in qdir.glob("*.json")), key=lambda i: (i["created_at"], i["id"]))


def _save(state_dir: Path, item: dict[str, Any]) -> None:
    item["updated_at"] = _now()
    (state_dir / "queue").mkdir(parents=True, exist_ok=True)
    write_json(state_dir / "queue" / f"{item['id']}.json", item)


def add(
    state_dir: Path, manifest_raw: dict[str, Any], *, max_attempts: int = 2, allow_duplicate: bool = False
) -> dict[str, Any]:
    manifest = parse_manifest(manifest_raw)
    with _locked(state_dir):
        if not allow_duplicate:
            for item in _items(state_dir):
                if item["manifest_sha256"] == manifest.sha256 and item["state"] in ("pending", "running"):
                    raise ValueError(f"identical manifest already queued as {item['id']} ({item['state']})")
        item = {
            "id": f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{manifest.id[:48]}",
            "manifest": dict(manifest.raw),
            "manifest_sha256": manifest.sha256,
            "state": "pending",
            "attempts": 0,
            "max_attempts": max_attempts,
            "created_at": _now(),
            "history": [],
        }
        _save(state_dir, item)
    return item


def status(state_dir: Path) -> dict[str, Any]:
    items = _items(state_dir)
    counts = {s: sum(1 for i in items if i["state"] == s) for s in STATES}
    return {
        "counts": counts,
        "items": [
            {k: i.get(k) for k in ("id", "state", "attempts", "max_attempts", "created_at", "updated_at")}
            | {"last": (i["history"] or [None])[-1]}
            for i in items
        ],
    }


def retry(state_dir: Path, item_id: str) -> dict[str, Any]:
    with _locked(state_dir):
        path = state_dir / "queue" / f"{item_id}.json"
        item: dict[str, Any] = read_json(path)
        if item["state"] not in ("failed", "blocked"):
            raise ValueError(f"{item_id} is {item['state']}; only failed or blocked items can be retried")
        item["state"] = "pending"
        item["max_attempts"] = max(item["max_attempts"], item["attempts"] + 1)
        item["history"].append({"at": _now(), "event": "retry requested"})
        _save(state_dir, item)
        return item


def _recover_interrupted(state_dir: Path, host_lock_free: bool) -> list[str]:
    recovered: list[str] = []
    if not host_lock_free:
        return recovered
    for item in _items(state_dir):
        if item["state"] == "running":
            item["history"].append(
                {"at": _now(), "event": "interrupted", "detail": "found running with no live benchmark"}
            )
            item["state"] = "pending" if item["attempts"] < item["max_attempts"] else "failed"
            _save(state_dir, item)
            recovered.append(item["id"])
    return recovered


def run_next(
    state_dir: Path,
    *,
    repo_root: Path | None = None,
    executor: Executor = execute,
    lock_path: Path | None = None,
    **execute_kwargs: Any,
) -> tuple[int, dict[str, Any]]:
    lock_path = lock_path or DEFAULT_LOCK
    with _locked(state_dir):
        free = _host_lock_free(lock_path)
        recovered = _recover_interrupted(state_dir, free)
        pending = [i for i in _items(state_dir) if i["state"] == "pending"]
        if not pending:
            return 0, {"state": "idle", "recovered": recovered, "queue": status(state_dir)["counts"]}
        item = pending[0]
        item["state"] = "running"
        item["attempts"] += 1
        item["history"].append({"at": _now(), "event": "started", "attempt": item["attempts"]})
        _save(state_dir, item)

    manifest: Manifest = parse_manifest(item["manifest"])
    try:
        outcome = executor(manifest, state_dir / "runs", repo_root=repo_root, lock_path=lock_path, **execute_kwargs)
        error = outcome.error
    except Exception as exc:  # the queue must always record an outcome
        outcome, error = None, f"{type(exc).__name__}: {exc}"

    entry: dict[str, Any] = {"at": _now(), "event": "finished", "attempt": item["attempts"]}
    code: int
    if outcome is not None and outcome.status == "completed" and outcome.run_dir is not None:
        _run, agg, classification, problems = load_run(outcome.run_dir)
        entry.update(
            outcome="completed",
            run_id=outcome.run_id,
            run_dir=str(outcome.run_dir),
            evidence_type=classification.evidence_type,
            evidence_reasons=list(classification.reasons) + problems,
            confidence=agg.get("confidence"),
            task_success=agg.get("task_success"),
        )
        item["state"] = "succeeded"
        code = 0 if classification.aig_tested else 2
    elif outcome is not None and outcome.status == "blocked":
        transient = any(t in (error or "") for t in TRANSIENT_BLOCKS)
        entry.update(
            outcome="blocked", reasons=outcome.preflight.block_reasons or [error or "blocked"], transient=transient
        )
        item["state"] = "pending" if transient else "blocked"
        code = 3
    else:
        entry.update(outcome="aborted", error=error, run_id=getattr(outcome, "run_id", None))
        item["state"] = "pending" if item["attempts"] < item["max_attempts"] else "failed"
        code = 2
    with _locked(state_dir):
        item["history"].append(entry)
        _save(state_dir, item)
    return code, {"state": item["state"], "item": item["id"], "result": entry, "recovered": recovered}


def _host_lock_free(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
        return True


def dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=str)
