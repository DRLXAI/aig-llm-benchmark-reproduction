"""Base44 sync: idempotent payloads for the Planner's observation entities.

Dry-run by default. Nothing here writes unless ``apply_plan`` is called with an explicit data
environment *and* the digest of the exact plan being applied, which the operator copies from the
dry-run output. Two write paths:

1. Data Studio import (no credentials in the harness): ``write_import_files`` emits one JSON array
   per entity, already de-duplicated against the Data Studio export you pass with ``--existing``.
   Data Studio's own import adds a DataImportAudit row.
2. Base44 CLI (``base44 exec``): the generated script re-reads existing records and skips any
   observation key already present *at write time*, then writes one DataImportAudit row.

Identity: the payload's ``notes`` start with ``[aig-lab obs_<key>]`` because neither entity has an
identity field (see schema-snapshot/SNAPSHOT.json "hazards").
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiglab.dedupe import content_digest

APP_ID = "6aaae7a7711572d416979dd6"
SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "base44" / "schema-snapshot"
DEFAULT_ID_MAP = Path(__file__).resolve().parents[2] / "base44" / "id-map.json"
ENTITIES = ("CompatibilityObservation", "BenchmarkObservation")
KEY_IN_NOTES = re.compile(r"\[aig-lab (obs_[0-9a-f]{32})\]")
REFERENCE_FIELDS = {
    "CompatibilityObservation": {"model_variant_id": "model_variant", "hardware_profile_id": "hardware_profile"},
    "BenchmarkObservation": {"model_variant_id": "model_variant", "hardware_or_host_id": "hardware_profile"},
}
DATA_ENVS = ("dev", "prod")

Runner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class SyncError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- schema
def load_schema(entity: str, snapshot_dir: Path = SNAPSHOT_DIR) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads((snapshot_dir / f"{entity}.jsonc").read_text(encoding="utf-8"))
    return schema


def validate_payload(entity: str, payload: Mapping[str, Any], snapshot_dir: Path = SNAPSHOT_DIR) -> list[str]:
    schema = load_schema(entity, snapshot_dir)
    props: dict[str, Any] = schema["properties"]
    errors = [f"unknown field {k!r}" for k in payload if k not in props]
    for name in schema.get("required", []):
        if payload.get(name) in (None, ""):
            errors.append(f"required field {name!r} missing")
    for k, v in payload.items():
        spec = props.get(k)
        if spec is None or v is None:
            if v is None:
                errors.append(f"{k} is null; omit unavailable fields instead")
            continue
        t = spec.get("type")
        ok = (
            (t == "string" and isinstance(v, str))
            or (t == "number" and isinstance(v, (int, float)) and not isinstance(v, bool))
            or (t == "boolean" and isinstance(v, bool))
        )
        if not ok:
            errors.append(f"{k} must be {t}, got {type(v).__name__}")
        if "enum" in spec and v not in spec["enum"]:
            errors.append(f"{k}={v!r} not in {spec['enum']}")
        if spec.get("format") == "date" and isinstance(v, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            errors.append(f"{k} must be YYYY-MM-DD")
    return errors


# ---------------------------------------------------------------------------- id map
def load_id_map(path: Path = DEFAULT_ID_MAP) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {"model_variant": {}, "hardware_profile": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: dict(raw.get(k) or {}) for k in ("model_variant", "hardware_profile")}


def existing_keys_from_records(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Observation keys already present (from a Data Studio export or a live read)."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        m = KEY_IN_NOTES.search(str(rec.get("notes") or ""))
        if m:
            out[m.group(1)] = dict(rec)
    return out


# ---------------------------------------------------------------------------- plan
def _payload(record: Mapping[str, Any], id_map: Mapping[str, Mapping[str, str]]) -> tuple[dict[str, Any], list[str]]:
    entity = str(record["entity"])
    fields = {k: v for k, v in dict(record["base44_fields"]).items() if v is not None}
    unresolved = []
    for field, kind in REFERENCE_FIELDS[entity].items():
        lab_key = str(fields.get(field))
        resolved = id_map.get(kind, {}).get(lab_key)
        if resolved:
            fields[field] = resolved
        else:
            unresolved.append(
                f"{field}: lab key {lab_key!r} has no Base44 {kind} id in lab/base44/id-map.json "
                f"(create or match the record in Data Studio, then map it)"
            )
    return fields, unresolved


def _comparable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in ("notes",)}


def build_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    id_map: Mapping[str, Mapping[str, str]],
    existing: Mapping[str, Mapping[str, Any]] | None = None,
    include_unknown: bool = False,
    snapshot_dir: Path = SNAPSHOT_DIR,
) -> dict[str, Any]:
    existing = existing or {}
    items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for rec in records:
        key = str(rec["observation_key"])
        if rec.get("origin") != "executed":
            excluded.append({"observation_key": key, "reason": f"origin {rec.get('origin')!r} is never synced"})
            continue
        if rec.get("evidence_type") == "unknown" and not include_unknown:
            excluded.append(
                {
                    "observation_key": key,
                    "reason": "executed run failed integrity checks: " + "; ".join(rec.get("evidence_reasons") or []),
                }
            )
            continue
        payload, unresolved = _payload(rec, id_map)
        digest = content_digest(payload)
        if key in seen:
            if seen[key] != digest:
                raise SyncError(f"{key} appears twice with different content; refusing to plan")
            continue
        seen[key] = digest
        errors = validate_payload(str(rec["entity"]), payload, snapshot_dir)
        if rec.get("evidence_type") == "aig_tested" and payload.get("evidence_type") != "aig_tested":
            errors.append("evidence_type changed during payload construction")
        if key in existing:
            same = content_digest(_comparable(payload)) == content_digest(
                _comparable({k: existing[key].get(k) for k in payload})
            )
            action = "skip_existing" if same else "update_requires_approval"
            target_id = existing[key].get("id")
        else:
            action, target_id = "create", None
        if unresolved or errors:
            action = "blocked"
        items.append(
            {
                "entity": rec["entity"],
                "observation_key": key,
                "config_key": rec.get("config_key"),
                "run_id": rec.get("run_id"),
                "evidence_type": payload.get("evidence_type"),
                "action": action,
                "existing_id": target_id,
                "reasons": unresolved + errors,
                "content_digest": digest,
                "payload": payload,
            }
        )
    items.sort(key=lambda i: (i["entity"], i["observation_key"]))
    summary: dict[str, int] = {}
    for i in items:
        summary[i["action"]] = summary.get(i["action"], 0) + 1
    plan = {
        "schema": "aiglab.base44-sync-plan/1",
        "app_id": APP_ID,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "dry_run",
        "existing_source": "provided" if existing else "none (dedupe happens again at write time)",
        "summary": summary,
        "excluded": excluded,
        "items": items,
    }
    plan["plan_digest"] = content_digest(
        {"items": [(i["observation_key"], i["action"], i["content_digest"]) for i in items]}
    )
    return plan


def write_import_files(plan: Mapping[str, Any], out_dir: Path) -> dict[str, Path]:
    """Data Studio import files: only `create` items, one JSON array per entity."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for entity in ENTITIES:
        rows = [i["payload"] for i in plan["items"] if i["entity"] == entity and i["action"] == "create"]
        path = out_dir / f"{entity}.import.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written[entity] = path
    return written


# ---------------------------------------------------------------------------- CLI path
def find_base44_cli() -> list[str] | None:
    configured = os.environ.get("AIGLAB_BASE44_CLI")
    if configured:
        return ["node", configured] if configured.endswith(".js") else [configured]
    binary = shutil.which("base44")
    if binary:
        return [binary]
    for candidate in sorted(Path.home().glob(".npm/_npx/*/node_modules/base44/bin/run.js")):
        return ["node", str(candidate)]
    return None


def render_apply_script(plan: Mapping[str, Any], allow_updates: bool) -> str:
    creates = [i for i in plan["items"] if i["action"] == "create"]
    updates = [i for i in plan["items"] if i["action"] == "update_requires_approval"] if allow_updates else []
    data = json.dumps(
        {
            "creates": [{"entity": i["entity"], "key": i["observation_key"], "payload": i["payload"]} for i in creates],
            "updates": [
                {"entity": i["entity"], "key": i["observation_key"], "id": i["existing_id"], "payload": i["payload"]}
                for i in updates
            ],
            "plan_digest": plan["plan_digest"],
        }
    )
    return f"""
const plan = {data};
const keyRe = /\\[aig-lab (obs_[0-9a-f]{{32}})\\]/;
const result = {{ created: [], skipped: [], updated: [], errors: [] }};
const existing = {{}};
for (const entity of ["CompatibilityObservation", "BenchmarkObservation"]) {{
  const page = await base44.entities[entity].filter({{}}, {{ limit: 5000 }});
  for (const r of (page.items || page || [])) {{
    const m = keyRe.exec(r.notes || "");
    if (m) existing[m[1]] = r.id;
  }}
}}
for (const c of plan.creates) {{
  if (existing[c.key]) {{ result.skipped.push(c.key); continue; }}
  if (c.payload.evidence_type === undefined) {{ result.errors.push(c.key + ": evidence_type missing"); continue; }}
  try {{ const r = await base44.entities[c.entity].create(c.payload); result.created.push([c.key, r.id]); existing[c.key] = r.id; }}
  catch (e) {{ result.errors.push(c.key + ": " + (e.message || e)); }}
}}
for (const u of plan.updates) {{
  try {{ await base44.entities[u.entity].update(u.id, u.payload); result.updated.push(u.key); }}
  catch (e) {{ result.errors.push(u.key + ": " + (e.message || e)); }}
}}
await base44.entities.DataImportAudit.create({{
  entity_name: "CompatibilityObservation+BenchmarkObservation",
  import_format: "json",
  row_count: plan.creates.length + plan.updates.length,
  accepted_count: result.created.length + result.updated.length,
  rejected_count: result.errors.length,
  source_label: "aiglab " + plan.plan_digest,
  notes: JSON.stringify({{ skipped: result.skipped.length, errors: result.errors.slice(0, 10) }})
}});
console.log(JSON.stringify(result));
"""


def _default_runner(argv: Sequence[str], stdin: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), input=stdin, capture_output=True, text=True, timeout=600, check=False)


def apply_plan(
    plan: Mapping[str, Any],
    *,
    confirm_digest: str,
    data_env: str,
    allow_updates: bool = False,
    runner: Runner = _default_runner,
    cli: list[str] | None = None,
) -> dict[str, Any]:
    if confirm_digest != plan.get("plan_digest"):
        raise SyncError(
            "confirmation digest does not match this plan; re-run the dry-run and confirm what you reviewed"
        )
    if data_env not in DATA_ENVS:
        raise SyncError(f"--data-env must be one of {DATA_ENVS}; it is never defaulted")
    if any(i["action"] == "blocked" for i in plan["items"]):
        raise SyncError("plan contains blocked items; resolve them (see reasons) before applying")
    if not allow_updates and any(i["action"] == "update_requires_approval" for i in plan["items"]):
        raise SyncError("plan would change existing records; pass --allow-updates only after reviewing each change")
    if not any(i["action"] in ("create", "update_requires_approval") for i in plan["items"]):
        return {"created": [], "skipped": [i["observation_key"] for i in plan["items"]], "updated": [], "errors": []}
    base = cli or find_base44_cli()
    if base is None:
        raise SyncError("Base44 CLI not found: install it (`npm i -g base44`) or set AIGLAB_BASE44_CLI")
    argv = [*base, "exec", "--app-id", str(plan.get("app_id") or APP_ID), "--data-env", data_env, "--json"]
    proc = runner(argv, render_apply_script(plan, allow_updates))
    if proc.returncode != 0:
        raise SyncError(f"base44 exec failed (exit {proc.returncode}): {(proc.stdout + proc.stderr)[-800:]}")
    lines = [line for line in proc.stdout.splitlines() if line.strip().startswith("{")]
    result: dict[str, Any] = json.loads(lines[-1]) if lines else {"raw": proc.stdout[-800:]}
    return result
