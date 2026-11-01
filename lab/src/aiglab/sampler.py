"""Process memory sampler for the runtime's model-serving process.

Two OS-level signals, both readable without privileges for a same-user process:

* RSS (``ps -o rss=``), sampled every RSS_SAMPLE_INTERVAL_S. On Apple silicon this includes the
  resident, memory-mapped model weights that Metal uses in place, so it is the closest
  unprivileged proxy for "memory this model configuration occupies".
* phys_footprint (``footprint -p``; macOS), sampled every FOOTPRINT_SAMPLE_INTERVAL_S. This is
  dirty + compressed memory and excludes clean mapped weights; it isolates KV cache and buffers.

Peaks are maxima over samples and therefore lower bounds on the true peak. Sample counts and
intervals are recorded with every value.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

from aiglab.methodology import FOOTPRINT_SAMPLE_INTERVAL_S, RSS_SAMPLE_INTERVAL_S

UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}


def read_rss_bytes(pid: int) -> int | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = out.strip()
    return int(out) * 1024 if out.isdigit() else None


def read_footprint_bytes(pid: int) -> int | None:
    if shutil.which("footprint") is None:
        return None
    try:
        out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"Footprint:\s*([\d.]+)\s*(B|KB|MB|GB)", out)
    return int(float(m.group(1)) * UNITS[m.group(2)]) if m else None


class MemorySampler:
    def __init__(
        self,
        pid: int | None,
        rss_reader: Callable[[int], int | None] = read_rss_bytes,
        footprint_reader: Callable[[int], int | None] = read_footprint_bytes,
    ) -> None:
        self.pid = pid
        self.rss_reader = rss_reader
        self.footprint_reader = footprint_reader
        self.rss: list[tuple[float, int]] = []
        self.footprint: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._t0 = 0.0

    def _loop(self, reader: Callable[[int], int | None], sink: list[tuple[float, int]], interval: float) -> None:
        assert self.pid is not None
        while not self._stop.is_set():
            value = reader(self.pid)
            if value is not None:
                sink.append((round(time.perf_counter() - self._t0, 3), value))
            self._stop.wait(interval)

    def __enter__(self) -> MemorySampler:
        self._t0 = time.perf_counter()
        if self.pid is not None:
            for reader, sink, interval in (
                (self.rss_reader, self.rss, RSS_SAMPLE_INTERVAL_S),
                (self.footprint_reader, self.footprint, FOOTPRINT_SAMPLE_INTERVAL_S),
            ):
                t = threading.Thread(target=self._loop, args=(reader, sink, interval), daemon=True)
                t.start()
                self._threads.append(t)
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=15)
        if self.pid is not None:  # one closing sample so short phases still get a reading
            for reader, sink in ((self.rss_reader, self.rss), (self.footprint_reader, self.footprint)):
                value = reader(self.pid)
                if value is not None:
                    sink.append((round(time.perf_counter() - self._t0, 3), value))

    def summary(self) -> dict[str, Any]:
        def peak(samples: list[tuple[float, int]], interval: float, method: str) -> dict[str, Any]:
            if not samples:
                reason = "no sampling target process" if self.pid is None else "no samples could be read"
                return {"value_bytes": None, "status": "unavailable", "reason": reason, "method": method}
            return {
                "value_bytes": max(v for _, v in samples),
                "status": "measured",
                "method": method,
                "samples": len(samples),
                "interval_s": interval,
                "bound": "lower bound (maximum of samples)",
            }

        return {
            "pid": self.pid,
            "peak_rss": peak(self.rss, RSS_SAMPLE_INTERVAL_S, "ps -o rss= (resident set incl. mapped weights)"),
            "peak_phys_footprint": peak(
                self.footprint,
                FOOTPRINT_SAMPLE_INTERVAL_S,
                "footprint -p (dirty+compressed; excludes clean mapped weights)",
            ),
        }

    def samples(self) -> dict[str, list[tuple[float, int]]]:
        return {"rss": self.rss, "phys_footprint": self.footprint}
