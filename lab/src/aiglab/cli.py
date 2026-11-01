"""`aiglab` command line. Exit codes everywhere: 0 healthy, 2 findings, 3 incomplete/blocked."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from aiglab import __version__
from aiglab.artifacts import iter_run_dirs, read_json, seal_digest, verify_seal, write_json
from aiglab.manifest import ManifestError, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
STORE = REPO_ROOT / "evidence" / "measurements"
RUNS = STORE / "runs"
NORMALIZED = STORE / "normalized"
INDEX = STORE / "index.json"
BLOCKED = STORE / "blocked"
DEFAULT_STATE = Path.home() / ".aiglab" / "state"


def out(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def progress(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------- commands
def cmd_list(a: argparse.Namespace) -> int:
    from aiglab.adapters.registry import detect_all
    from aiglab.workloads import WORKLOADS

    data: dict[str, Any] = {
        "harness_version": __version__,
        "runtimes": detect_all(),
        "workloads": [w.describe() for w in WORKLOADS.values()],
    }
    if not a.no_models:
        try:
            from aiglab.adapters.ollama import OllamaAdapter

            data["ollama_models"] = [
                {
                    "name": m.get("name"),
                    "size_bytes": m.get("size"),
                    "digest": m.get("digest"),
                    "quantization": (m.get("details") or {}).get("quantization_level"),
                }
                for m in OllamaAdapter().available_models()
            ]
        except OSError as exc:
            data["ollama_models"] = {"unavailable": str(exc)}
    out(data)
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    results, bad = [], 0
    for p in a.manifests:
        try:
            m = load_manifest(Path(p))
            results.append({"manifest": p, "valid": True, "id": m.id, "sha256": m.sha256})
        except ManifestError as exc:
            bad += 1
            results.append({"manifest": p, "valid": False, "errors": exc.errors})
    out(results)
    return 2 if bad else 0


def _load(path: str) -> Any:
    try:
        return load_manifest(Path(path))
    except ManifestError as exc:
        out({"manifest": path, "valid": False, "errors": exc.errors})
        raise SystemExit(2) from exc


def cmd_dry_run(a: argparse.Namespace) -> int:
    from aiglab.runner import make_adapter, preflight
    from aiglab.workloads import get_workload

    m = _load(a.manifest)
    try:
        adapter = make_adapter(m)
    except NotImplementedError:
        adapter = None
    pf = preflight(m, adapter, allow_resident=a.allow_resident)
    wl = get_workload(m.workload_id)
    tasks = len(wl.tasks())
    out(
        {
            "manifest": m.id,
            "would_execute": not pf.blocked,
            "plan": {
                "repetitions": {"warmup": m.warmup, "measured": m.measured},
                "lanes": m.concurrency,
                "tasks_per_lane": tasks,
                "task_executions": (m.warmup + m.measured) * m.concurrency * tasks,
                "model_requests": "at least the task count; tool workloads make up to 8 per task",
            },
            "evidence_if_executed": (
                "aig_tested only if every identity and seal check passes"
                if adapter and adapter.executes_real_runtime
                else "never evidence (non-executing or unavailable runtime)"
            ),
            "preflight": pf.as_dict(),
        }
    )
    return 3 if pf.blocked else 0


def _record_blocked(m: Any, outcome: Any, blocked_dir: Path) -> Path:
    blocked_dir.mkdir(parents=True, exist_ok=True)
    path = blocked_dir / f"{outcome.run_id}.json"
    write_json(
        path,
        {
            "schema": "aiglab.blocked-run/1",
            "run_id": outcome.run_id,
            "manifest_id": m.id,
            "manifest_sha256": m.sha256,
            "manifest": dict(m.raw),
            "preflight": outcome.preflight.as_dict(),
        },
    )
    return path


def cmd_run(a: argparse.Namespace) -> int:
    from aiglab.adapters.base import RuntimeAdapter
    from aiglab.adapters.ollama import OllamaAdapter
    from aiglab.normalize import load_run
    from aiglab.runner import execute, make_adapter

    m = _load(a.manifest)
    adapter: RuntimeAdapter | None = None
    if m.runtime.id == "ollama":
        adapter = OllamaAdapter(endpoint=m.runtime.endpoint, verify_artifact_hash=not a.skip_artifact_hash)
    else:
        try:
            adapter = make_adapter(m)
        except NotImplementedError:
            adapter = None
    outcome = execute(
        m, Path(a.runs_dir), adapter=adapter, repo_root=REPO_ROOT, allow_resident=a.allow_resident, progress=progress
    )
    if outcome.status == "blocked":
        path = _record_blocked(m, outcome, Path(a.blocked_dir)) if not a.no_record_blocked else None
        out({"status": "blocked", "reasons": outcome.preflight.block_reasons, "recorded": str(path) if path else None})
        return 3
    assert outcome.run_dir is not None
    _run, agg, classification, problems = load_run(outcome.run_dir)
    out(
        {
            "status": outcome.status,
            "error": outcome.error,
            "run_dir": str(outcome.run_dir),
            "evidence_type": classification.evidence_type,
            "evidence_reasons": list(classification.reasons) + problems,
            "confidence": agg.get("confidence"),
            "task_success": agg.get("task_success"),
            "runtime_completion": agg.get("runtime_completion"),
            "headline": {
                k: ((agg["metrics"].get(k) or {}).get("included") or {}).get("median")
                for k in ("decode_tokens_per_s", "ttft_ms", "task_completion_s", "peak_rss_bytes")
            },
        }
    )
    return 0 if classification.aig_tested else 2


def cmd_aggregate(a: argparse.Namespace) -> int:
    from aiglab.aggregate import aggregate_run

    run_dir = Path(a.run_dir)
    ok, problems = verify_seal(run_dir)
    if not ok:
        out({"error": "seal verification failed; refusing to aggregate", "problems": problems})
        return 2
    exclusions = []
    for spec in a.exclude or []:
        idx, _, reason = spec.partition(":")
        if not reason.strip():
            out({"error": f"exclusion {spec!r} needs a reason: --exclude <repetition>:<reason>"})
            return 3
        exclusions.append({"repetition": int(idx), "reason": reason.strip()})
    previous = read_json(run_dir / "aggregate.json") if (run_dir / "aggregate.json").is_file() else {}
    if not exclusions and not a.clear_exclusions:
        exclusions = previous.get("exclusions") or []
    agg = aggregate_run(read_json(run_dir / "run.json"), seal_digest(run_dir), exclusions)
    write_json(run_dir / "aggregate.json", agg)
    out(
        {k: agg[k] for k in ("run_id", "measured_repetitions", "exclusions", "confidence", "confidence_reasons")}
        | {"outliers": {k: v.get("outliers") for k, v in agg["metrics"].items() if v.get("outliers")}}
    )
    return 0


def _records(runs_dir: str) -> list[dict[str, Any]]:
    from aiglab.export import collect

    return collect(Path(runs_dir))


def cmd_export_json(a: argparse.Namespace) -> int:
    from aiglab.export import write_json as export_json

    paths = export_json(_records(a.runs_dir), Path(a.out))
    out({"written": [str(p) for p in paths]})
    return 0


def cmd_export_csv(a: argparse.Namespace) -> int:
    from aiglab.export import write_csv

    paths = write_csv(_records(a.runs_dir), Path(a.out))
    out({"written": [str(p) for p in paths]})
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    from aiglab.quality import check_store

    report = check_store(Path(a.runs_dir), Path(a.normalized_dir) if a.normalized_dir else None)
    out(report)
    return 0 if report["ok"] else 2


def cmd_base44_plan(a: argparse.Namespace) -> int:
    from aiglab.base44 import build_plan, existing_keys_from_records, load_id_map, write_import_files

    existing: dict[str, Any] = {}
    for path in a.existing or []:
        data = read_json(Path(path))
        existing.update(existing_keys_from_records(data if isinstance(data, list) else data.get("items", [])))
    plan = build_plan(
        _records(a.runs_dir), id_map=load_id_map(Path(a.id_map)), existing=existing, include_unknown=a.include_unknown
    )
    write_json(Path(a.out), plan)
    files = write_import_files(plan, Path(a.import_dir)) if a.import_dir else {}
    out(
        {
            "plan": a.out,
            "plan_digest": plan["plan_digest"],
            "summary": plan["summary"],
            "excluded": len(plan["excluded"]),
            "blocked_reasons": sorted({r for i in plan["items"] for r in i["reasons"]}),
            "import_files": {k: str(v) for k, v in files.items()},
            "next": "review the plan; apply with: aiglab base44-sync --plan <plan> --apply --data-env <dev|prod> "
            f"--confirm {plan['plan_digest']}",
        }
    )
    return 3 if plan["summary"].get("blocked") else 0


def cmd_base44_sync(a: argparse.Namespace) -> int:
    from aiglab.base44 import SyncError, apply_plan

    plan = read_json(Path(a.plan))
    if not a.apply:
        out(
            {
                "mode": "dry_run",
                "plan_digest": plan["plan_digest"],
                "summary": plan["summary"],
                "would_create": [i["observation_key"] for i in plan["items"] if i["action"] == "create"],
                "note": "nothing written; pass --apply --data-env <dev|prod> --confirm <plan_digest> to write",
            }
        )
        return 0
    try:
        result = apply_plan(
            plan, confirm_digest=a.confirm or "", data_env=a.data_env or "", allow_updates=a.allow_updates
        )
    except SyncError as exc:
        out({"mode": "apply", "error": str(exc)})
        return 3
    out({"mode": "apply", "data_env": a.data_env, "result": result})
    return 2 if result.get("errors") else 0


def cmd_index(a: argparse.Namespace) -> int:
    from aiglab.index import build_index

    index = build_index(_records(a.runs_dir))
    write_json(Path(a.out), index)
    out({"written": a.out, "counts": index["counts"]})
    return 0


def cmd_query(a: argparse.Namespace) -> int:
    from aiglab.index import closest, measured_memory, measured_performance

    index = read_json(Path(a.index))
    if a.kind == "closest":
        out(
            closest(
                index,
                model=a.model,
                context=a.context,
                concurrency=a.concurrency,
                hardware_key=a.hardware,
                memory_gib=a.memory_gib,
                workload=a.workload,
                include_untested=a.include_untested,
            )
        )
    elif a.kind == "memory":
        out(
            measured_memory(
                index, model=a.model, hardware_key=a.hardware or "", context=a.context, concurrency=a.concurrency
            )
        )
    else:
        out(measured_performance(index, workload=a.workload or "", model=a.model, hardware_key=a.hardware))
    return 0


def cmd_host(_a: argparse.Namespace) -> int:
    from aiglab.host import capture_host

    out(capture_host(REPO_ROOT))
    return 0


def cmd_reproduce(a: argparse.Namespace) -> int:
    run_dir = Path(a.runs_dir) / a.run_id
    ok, problems = verify_seal(run_dir)
    run = read_json(run_dir / "run.json")
    manifest = read_json(run_dir / "manifest.json")
    target = Path(a.out)
    write_json(target, manifest)
    identity = (run.get("preflight") or {}).get("identity") or {}
    out(
        {
            "source_run": a.run_id,
            "seal_ok": ok,
            "seal_problems": problems,
            "manifest_written": str(target),
            "pin_to_match": {
                "methodology_version": run.get("methodology_version"),
                "harness_version": run.get("harness_version"),
                "harness_commit": read_json(run_dir / "host.json")
                .get("harness", {})
                .get("git_commit", {})
                .get("value"),
                "workload": f"{run['workload']['id']}@{run['workload']['version']} definition {run['workload']['definition_sha256']}",
                "runtime": f"{run['adapter']['id']} {(run.get('preflight') or {}).get('runtime', {}).get('version')}",
                "model_artifact": identity.get("artifact_digest"),
                "hardware_profile": run.get("hardware_profile_key"),
            },
            "command": f"cd lab && uv run aiglab run {target}",
            "note": "for a strict reproduction set model.expected_artifact_digest in the manifest to the artifact above",
        }
    )
    return 0 if ok else 2


def cmd_import_runs(a: argparse.Namespace) -> int:
    from aiglab.evidence import classify_run

    dest = Path(a.runs_dir)
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    for run_dir in iter_run_dirs(Path(a.source)):
        ok, problems = verify_seal(run_dir)
        run = read_json(run_dir / "run.json")
        cls = classify_run(run, seal_ok=ok)
        target = dest / run_dir.name
        if not ok or cls.origin != "executed":
            report.append({"run_id": run_dir.name, "imported": False, "reason": problems or f"origin {cls.origin}"})
        elif target.exists():
            report.append({"run_id": run_dir.name, "imported": False, "reason": "already in the evidence store"})
        else:
            shutil.copytree(run_dir, target)
            report.append({"run_id": run_dir.name, "imported": True, "evidence_type": cls.evidence_type})
    out(report)
    return 0


def cmd_queue(a: argparse.Namespace) -> int:
    from aiglab import queue

    state = Path(a.state_dir)
    if a.queue_cmd == "add":
        try:
            item = queue.add(
                state, read_json(Path(a.manifest)), max_attempts=a.max_attempts, allow_duplicate=a.allow_duplicate
            )
        except (ValueError, ManifestError) as exc:
            out({"error": str(exc)})
            return 3
        out({"queued": item["id"]})
        return 0
    if a.queue_cmd == "list":
        out(queue.status(state))
        return 0
    if a.queue_cmd == "retry":
        try:
            out({"pending": queue.retry(state, a.item)["id"]})
        except (ValueError, OSError) as exc:
            out({"error": str(exc)})
            return 3
        return 0
    code, result = queue.run_next(state, repo_root=REPO_ROOT, progress=progress)
    out(result)
    return code


# ---------------------------------------------------------------------------- parser
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aiglab", description="AIG Measurement Lab")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("list", help="runtimes (detected), workloads, and locally available models")
    s.add_argument("--no-models", action="store_true")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("validate", help="validate manifests")
    s.add_argument("manifests", nargs="+")
    s.set_defaults(fn=cmd_validate)

    for name, fn, help_text in (
        ("dry-run", cmd_dry_run, "preflight only; executes nothing"),
        ("run", cmd_run, "execute a manifest on this host"),
    ):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("manifest")
        s.add_argument("--allow-resident", action="store_true", help="run even if other models are loaded (recorded)")
        if name == "run":
            s.add_argument("--runs-dir", default=str(RUNS))
            s.add_argument("--blocked-dir", default=str(BLOCKED))
            s.add_argument("--no-record-blocked", action="store_true")
            s.add_argument("--skip-artifact-hash", action="store_true")
        s.set_defaults(fn=fn)

    s = sub.add_parser("aggregate", help="recompute aggregate.json; optional explicit exclusions")
    s.add_argument("run_dir")
    s.add_argument("--exclude", action="append", metavar="REP:REASON")
    s.add_argument("--clear-exclusions", action="store_true")
    s.set_defaults(fn=cmd_aggregate)

    for name, fn, default in (("export-json", cmd_export_json, NORMALIZED), ("export-csv", cmd_export_csv, NORMALIZED)):
        s = sub.add_parser(name, help=f"write normalized records ({name.split('-')[1]})")
        s.add_argument("--runs-dir", default=str(RUNS))
        s.add_argument("--out", default=str(default))
        s.set_defaults(fn=fn)

    s = sub.add_parser("check-evidence", help="validate evidence quality and integrity")
    s.add_argument("--runs-dir", default=str(RUNS))
    s.add_argument("--normalized-dir", default=str(NORMALIZED))
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("base44-plan", help="prepare idempotent Base44 sync payloads (dry-run)")
    s.add_argument("--runs-dir", default=str(RUNS))
    s.add_argument("--id-map", default=str(REPO_ROOT / "lab" / "base44" / "id-map.json"))
    s.add_argument("--existing", action="append", help="Data Studio export(s) of existing observation records")
    s.add_argument("--include-unknown", action="store_true")
    s.add_argument("--out", default=str(STORE / "base44-sync-plan.json"))
    s.add_argument("--import-dir", help="also write Data Studio import files here")
    s.set_defaults(fn=cmd_base44_plan)

    s = sub.add_parser("base44-sync", help="dry-run (default) or apply a reviewed plan")
    s.add_argument("--plan", default=str(STORE / "base44-sync-plan.json"))
    s.add_argument("--apply", action="store_true")
    s.add_argument("--data-env", choices=["dev", "prod"])
    s.add_argument("--confirm", help="plan_digest from the reviewed dry-run")
    s.add_argument("--allow-updates", action="store_true")
    s.set_defaults(fn=cmd_base44_sync)

    s = sub.add_parser("index", help="build the evidence index for the Planner")
    s.add_argument("--runs-dir", default=str(RUNS))
    s.add_argument("--out", default=str(INDEX))
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("query", help="query the evidence index")
    s.add_argument("kind", choices=["closest", "memory", "performance"])
    s.add_argument("--index", default=str(INDEX))
    s.add_argument("--model")
    s.add_argument("--context", type=int, default=4096)
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--hardware")
    s.add_argument("--memory-gib", type=float)
    s.add_argument("--workload")
    s.add_argument("--include-untested", action="store_true")
    s.set_defaults(fn=cmd_query)

    s = sub.add_parser("host", help="print captured host facts")
    s.set_defaults(fn=cmd_host)

    s = sub.add_parser("reproduce", help="write a manifest to reproduce a prior run, with what to pin")
    s.add_argument("run_id")
    s.add_argument("--runs-dir", default=str(RUNS))
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_reproduce)

    s = sub.add_parser("import-runs", help="copy sealed executed runs (e.g. from Estate) into the evidence store")
    s.add_argument("--from", dest="source", default=str(DEFAULT_STATE / "runs"))
    s.add_argument("--runs-dir", default=str(RUNS))
    s.set_defaults(fn=cmd_import_runs)

    s = sub.add_parser("queue", help="benchmark queue (Estate integration)")
    s.add_argument("--state-dir", default=str(DEFAULT_STATE))
    qs = s.add_subparsers(dest="queue_cmd", required=True)
    q = qs.add_parser("add")
    q.add_argument("manifest")
    q.add_argument("--max-attempts", type=int, default=2)
    q.add_argument("--allow-duplicate", action="store_true")
    qs.add_parser("list")
    q = qs.add_parser("retry")
    q.add_argument("item")
    qs.add_parser("run-next")
    s.set_defaults(fn=cmd_queue)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "query" and args.kind in ("closest", "memory") and not args.model:
        print("--model is required for this query", file=sys.stderr)
        return 3
    code: int = args.fn(args)
    return code
