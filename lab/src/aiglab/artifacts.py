"""Run directory layout and the artifact seal.

    <runs-root>/<run_id>/
        manifest.json            the manifest exactly as run (raw)
        run.json                 identity, preflight, every repetition and request (raw)
        host.json                host facts at run start (raw)
        memory-samples.json      every memory sample (raw)
        runtime-log-excerpt.txt  memory-relevant runtime log lines, if readable (raw)
        SHA256SUMS               seal over every raw file above, written last
        aggregate.json           derived: statistics + any operator exclusions (outside the seal)

Raw files are written once and never edited. ``aggregate.json`` records the SHA256SUMS digest it
was computed from, so a derived file can never silently drift from the raw data.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

SEAL = "SHA256SUMS"
DERIVED = frozenset({"aggregate.json"})
RAW_REQUIRED = ("manifest.json", "run.json", "host.json")


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_files(run_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in run_dir.iterdir()
        if p.is_file() and p.name != SEAL and p.name not in DERIVED and not p.name.endswith(".tmp")
    )


def seal(run_dir: Path) -> str:
    lines = [f"{file_sha256(p)}  {p.name}" for p in _raw_files(run_dir)]
    (run_dir / SEAL).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return seal_digest(run_dir)


def seal_digest(run_dir: Path) -> str:
    return "sha256:" + file_sha256(run_dir / SEAL)


def verify_seal(run_dir: Path) -> tuple[bool, list[str]]:
    seal_path = run_dir / SEAL
    if not seal_path.is_file():
        return False, [f"{SEAL} missing"]
    problems: list[str] = []
    listed: dict[str, str] = {}
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        listed[name] = digest
    for required in RAW_REQUIRED:
        if required not in listed:
            problems.append(f"{required} is not covered by the seal")
    present = {p.name for p in _raw_files(run_dir)}
    for name in sorted(present - set(listed)):
        problems.append(f"unsealed raw file {name}")
    for name, digest in sorted(listed.items()):
        path = run_dir / name
        if not path.is_file():
            problems.append(f"sealed file {name} is missing")
        elif file_sha256(path) != digest:
            problems.append(f"{name} was modified after sealing")
    return (not problems), problems


def iter_run_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "run.json").is_file())
