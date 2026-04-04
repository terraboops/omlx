# SPDX-License-Identifier: Apache-2.0
"""Per-run profiler: CPU + GPU memory + wall time.

Captures metrics in a background thread while a benchmark runs:
  - Metal active + peak memory (mx.get_active_memory)
  - Process RSS (psutil or /proc fallback)
  - CPU utilization % (psutil)
  - System swap delta (sysctl vm.swapusage)
  - Time-series samples at configurable interval

Emits structured profile data for scoring & comparison.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class ProfileSample:
    t: float                    # Seconds since start
    metal_active_gb: float
    metal_peak_gb: float
    rss_gb: float
    cpu_pct: float              # CPU% since last sample
    swap_gb: float              # Swap delta from baseline


@dataclass
class ProfileResult:
    samples: List[ProfileSample] = field(default_factory=list)
    total_seconds: float = 0.0
    # Aggregates
    metal_peak_gb: float = 0.0
    metal_avg_gb: float = 0.0
    rss_peak_gb: float = 0.0
    rss_avg_gb: float = 0.0
    cpu_avg_pct: float = 0.0
    cpu_peak_pct: float = 0.0
    swap_peak_gb: float = 0.0

    def summary(self) -> dict:
        return {
            "total_seconds": round(self.total_seconds, 2),
            "metal_peak_gb": round(self.metal_peak_gb, 2),
            "metal_avg_gb": round(self.metal_avg_gb, 2),
            "rss_peak_gb": round(self.rss_peak_gb, 2),
            "cpu_avg_pct": round(self.cpu_avg_pct, 1),
            "cpu_peak_pct": round(self.cpu_peak_pct, 1),
            "swap_peak_gb": round(self.swap_peak_gb, 2),
            "num_samples": len(self.samples),
        }


def _get_swap_gb() -> float:
    """Read swap used in GB from sysctl."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], timeout=5, text=True,
        )
        parts = out.split()
        for i, p in enumerate(parts):
            if p == "used" and i + 2 < len(parts):
                return float(parts[i + 2].rstrip("M")) / 1024.0
    except Exception:
        pass
    return 0.0


def _get_process_stats():
    """Return (rss_gb, cpu_pct) via psutil, 0s on failure."""
    try:
        import psutil
        p = psutil.Process()
        return (
            p.memory_info().rss / 1e9,
            p.cpu_percent(interval=None),
        )
    except Exception:
        return (0.0, 0.0)


class Profiler:
    """Background thread profiler — samples at fixed interval."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._swap_baseline: float = 0.0
        self.result: ProfileResult = ProfileResult()

    def start(self):
        self.result = ProfileResult()
        self._stop.clear()
        self._start_time = time.perf_counter()
        self._swap_baseline = _get_swap_gb()

        # Initialize psutil cpu_percent measurement
        try:
            import psutil
            psutil.Process().cpu_percent(interval=None)
        except Exception:
            pass

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> ProfileResult:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.result.total_seconds = time.perf_counter() - self._start_time
        self._compute_aggregates()
        return self.result

    def _loop(self):
        while not self._stop.is_set():
            try:
                t = time.perf_counter() - self._start_time
                rss, cpu = _get_process_stats()
                swap = _get_swap_gb() - self._swap_baseline
                sample = ProfileSample(
                    t=t,
                    metal_active_gb=mx.get_active_memory() / 1e9,
                    metal_peak_gb=mx.get_peak_memory() / 1e9,
                    rss_gb=rss,
                    cpu_pct=cpu,
                    swap_gb=swap,
                )
                self.result.samples.append(sample)
            except Exception as e:
                logger.warning("profiler sample failed: %s", e)
            self._stop.wait(self.sample_interval)

    def _compute_aggregates(self):
        if not self.result.samples:
            return
        s = self.result.samples
        self.result.metal_peak_gb = max(x.metal_peak_gb for x in s)
        self.result.metal_avg_gb = sum(x.metal_active_gb for x in s) / len(s)
        self.result.rss_peak_gb = max(x.rss_gb for x in s)
        self.result.rss_avg_gb = sum(x.rss_gb for x in s) / len(s)
        self.result.cpu_avg_pct = sum(x.cpu_pct for x in s) / len(s)
        self.result.cpu_peak_pct = max(x.cpu_pct for x in s)
        self.result.swap_peak_gb = max(x.swap_gb for x in s)
