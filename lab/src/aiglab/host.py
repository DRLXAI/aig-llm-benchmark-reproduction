"""Host facts, captured read-only and without privileges.

Nothing here identifies the person or the machine instance (no hostname, serial, user name or
network addresses). Every fact records how it was obtained so a reader can judge it; a fact that
cannot be read is recorded as null with the reason.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiglab

Runner = Callable[[Sequence[str]], str | None]


def run_cmd(argv: Sequence[str], timeout: float = 15.0) -> str | None:
    """Run a fixed argv; return stdout or None. Never raises, never uses a shell."""
    try:
        proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _fact(value: Any, method: str, reason: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"value": value, "method": method}
    if value is None:
        out["reason"] = reason or "not available on this host"
    return out


def _sysctl(run: Runner, name: str) -> str | None:
    out = run(["sysctl", "-n", name])
    return out.strip() if out else None


def _int_or_none(text: str | None) -> int | None:
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def memory_pressure(run: Runner = run_cmd) -> dict[str, Any]:
    level = _int_or_none(_sysctl(run, "kern.memorystatus_vm_pressure_level"))
    free_pct = None
    out = run(["memory_pressure", "-Q"])
    if out:
        m = re.search(r"free percentage:\s*(\d+)%", out)
        free_pct = int(m.group(1)) if m else None
    return {
        "pressure_level": _fact(level, "sysctl kern.memorystatus_vm_pressure_level (1 normal, 2 warn, 4 critical)"),
        "free_percent": _fact(free_pct, "memory_pressure -Q"),
    }


def power_source(run: Runner = run_cmd) -> dict[str, Any]:
    out = run(["pmset", "-g", "batt"])
    if not out:
        return _fact(None, "pmset -g batt")
    m = re.search(r"drawing from '([^']+)'", out)
    return _fact(m.group(1) if m else None, "pmset -g batt")


def _apple_gpu_cores(run: Runner) -> int | None:
    out = run(["system_profiler", "SPDisplaysDataType", "-json"])
    if not out:
        return None
    try:
        displays = json.loads(out).get("SPDisplaysDataType") or []
    except json.JSONDecodeError:
        return None
    for d in displays:
        cores = d.get("sppci_cores")
        if cores is not None:
            return _int_or_none(str(cores))
    return None


def harness_identity(repo_root: Path | None, run: Runner = run_cmd) -> dict[str, Any]:
    commit = dirty = None
    if repo_root is not None:
        out = run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
        commit = out.strip() if out else None
        status = run(["git", "-C", str(repo_root), "status", "--porcelain", "--", "lab/src", "lab/pyproject.toml"])
        dirty = bool(status.strip()) if status is not None else None
    return {
        "name": "aiglab",
        "version": aiglab.__version__,
        "git_commit": _fact(commit, "git rev-parse HEAD", "not a git checkout"),
        "lab_tree_dirty": _fact(
            dirty, "git status --porcelain -- lab/src lab/pyproject.toml (the measuring code)", "not a git checkout"
        ),
    }


def capture_host(repo_root: Path | None = None, run: Runner = run_cmd) -> dict[str, Any]:
    system = platform.system()
    host: dict[str, Any] = {
        "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "os": {
            "system": system,
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "harness": harness_identity(repo_root, run),
        "python": platform.python_version(),
    }
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    host["load_average_1m"] = _fact(round(load[0], 2) if load else None, "os.getloadavg")

    if system == "Darwin":
        prod = _sysctl(run, "kern.osproductversion")
        build = _sysctl(run, "kern.osversion")
        host["os"].update({"product": "macOS", "version": prod, "build": build})
        chip = _sysctl(run, "machdep.cpu.brand_string")
        memsize = _int_or_none(_sysctl(run, "hw.memsize"))
        host["cpu"] = {
            "brand": _fact(chip, "sysctl machdep.cpu.brand_string"),
            "logical_cores": _fact(_int_or_none(_sysctl(run, "hw.logicalcpu")), "sysctl hw.logicalcpu"),
            "performance_cores": _fact(
                _int_or_none(_sysctl(run, "hw.perflevel0.physicalcpu")), "sysctl hw.perflevel0.physicalcpu"
            ),
            "efficiency_cores": _fact(
                _int_or_none(_sysctl(run, "hw.perflevel1.physicalcpu")), "sysctl hw.perflevel1.physicalcpu"
            ),
        }
        host["gpu"] = {
            "name": _fact(chip, "Apple silicon: the GPU is part of the SoC named by the CPU brand"),
            "cores": _fact(_apple_gpu_cores(run), "system_profiler SPDisplaysDataType sppci_cores"),
        }
        host["memory"] = {
            "total_bytes": _fact(memsize, "sysctl hw.memsize"),
            "kind": "unified" if chip and chip.startswith("Apple") else "ddr",
            **memory_pressure(run),
        }
        host["power_source"] = power_source(run)
        gib = round(memsize / 2**30) if memsize else None
        host["hardware_profile_key"] = f"{slug(chip)}-{gib}gb" if chip and gib else None
    else:
        meminfo = Path("/proc/meminfo")
        total = None
        if meminfo.exists():
            m = re.search(r"MemTotal:\s+(\d+) kB", meminfo.read_text())
            total = int(m.group(1)) * 1024 if m else None
        cpu = None
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            m = re.search(r"model name\s*:\s*(.+)", cpuinfo.read_text())
            cpu = m.group(1).strip() if m else None
        host["cpu"] = {"brand": _fact(cpu, "/proc/cpuinfo"), "logical_cores": _fact(os.cpu_count(), "os.cpu_count")}
        host["gpu"] = {"name": _fact(None, "not captured on this OS by harness 1.0.0")}
        host["memory"] = {"total_bytes": _fact(total, "/proc/meminfo"), "kind": "ddr"}
        host["power_source"] = _fact(None, "not captured on this OS by harness 1.0.0")
        gib = round(total / 2**30) if total else None
        host["hardware_profile_key"] = f"{slug(cpu)}-{gib}gb" if cpu and gib else None
    host["power_draw"] = _fact(
        None,
        "powermetrics",
        "power draw requires root (powermetrics); the harness never escalates privileges",
    )
    return host
