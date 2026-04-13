# SPDX-License-Identifier: Apache-2.0
"""Per-run profiler: CPU + GPU memory + wall time.

Captures metrics in a background thread while a benchmark runs:
  - Metal active + peak memory (mx.get_active_memory)
  - Process physical footprint (phys_footprint via proc_pid_rusage)
  - CPU utilization % (psutil, reused Process object)
  - System swap delta (sysctl vm.swapusage)
  - Swap I/O throughput (vm_stat page counter deltas)
  - Time-series samples at configurable interval

Emits structured profile data for scoring & comparison.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
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
    phys_footprint_gb: float    # macOS physical footprint (includes Metal + compressed)
    cpu_pct: float              # CPU% since last sample
    swap_gb: float              # Swap delta from baseline
    swap_io_mb_per_s: float     # Swap I/O throughput (pageins+swapins+swapouts)


@dataclass
class ProfileResult:
    samples: List[ProfileSample] = field(default_factory=list)
    total_seconds: float = 0.0
    # Aggregates
    metal_peak_gb: float = 0.0
    metal_avg_gb: float = 0.0
    rss_peak_gb: float = 0.0
    rss_avg_gb: float = 0.0
    phys_footprint_peak_gb: float = 0.0
    cpu_avg_pct: float = 0.0
    cpu_peak_pct: float = 0.0
    swap_peak_gb: float = 0.0
    swap_io_peak_mb_per_s: float = 0.0

    def summary(self) -> dict:
        return {
            "total_seconds": round(self.total_seconds, 2),
            "metal_peak_gb": round(self.metal_peak_gb, 2),
            "metal_avg_gb": round(self.metal_avg_gb, 2),
            "rss_peak_gb": round(self.rss_peak_gb, 2),
            "phys_footprint_peak_gb": round(self.phys_footprint_peak_gb, 2),
            "cpu_avg_pct": round(self.cpu_avg_pct, 1),
            "cpu_peak_pct": round(self.cpu_peak_pct, 1),
            "swap_peak_gb": round(self.swap_peak_gb, 2),
            "swap_io_peak_mb_per_s": round(self.swap_io_peak_mb_per_s, 1),
            "num_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# macOS physical footprint via proc_pid_rusage (ctypes)
# ---------------------------------------------------------------------------

def _get_phys_footprint_gb() -> float:
    """Get process physical footprint in GB via macOS proc_pid_rusage().

    phys_footprint includes: wired + compressed + GPU-shared + purgeable.
    This is the correct metric on Apple Silicon where Metal allocations
    use unified memory but don't appear in RSS.
    """
    try:
        libproc_path = ctypes.util.find_library("proc")
        if libproc_path is None:
            libproc_path = "/usr/lib/libproc.dylib"
        libproc = ctypes.CDLL(libproc_path)

        RUSAGE_INFO_V4 = 4
        # rusage_info_v4 is ~240 bytes; allocate generous buffer
        buf = (ctypes.c_uint8 * 512)()
        ret = libproc.proc_pid_rusage(os.getpid(), RUSAGE_INFO_V4, ctypes.byref(buf))
        if ret != 0:
            return 0.0

        # ri_phys_footprint is a uint64 at byte offset 72 in rusage_info_v4
        # (after uuid[16] + 7 uint64 fields = 16 + 56 = 72)
        phys_footprint = int.from_bytes(bytes(buf[72:80]), byteorder='little')
        return phys_footprint / 1e9
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Swap depth via ctypes (NO subprocess — fork under memory pressure is fatal)
# ---------------------------------------------------------------------------

def _init_sysctl():
    """Load libc and prepare sysctlbyname for swap queries."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libc.dylib")
        return libc
    except Exception:
        return None

_LIBC = _init_sysctl()


def _get_swap_gb() -> float:
    """Read swap used in GB via sysctlbyname, psutil fallback (no subprocess)."""
    # Try ctypes sysctlbyname first (fastest, no fork)
    if _LIBC is not None:
        try:
            # struct xsw_usage { uint64_t total, avail, used; }
            buf = (ctypes.c_uint64 * 3)()
            size = ctypes.c_size_t(ctypes.sizeof(buf))
            ret = _LIBC.sysctlbyname(
                b"vm.swapusage", ctypes.byref(buf), ctypes.byref(size), None, 0,
            )
            if ret == 0:
                return buf[2] / 1e9  # used bytes → GB
        except Exception:
            pass
    # Fallback: psutil (no fork, works under sandbox)
    try:
        import psutil
        return psutil.swap_memory().used / 1e9
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Swap I/O throughput via psutil swap_memory() deltas (no subprocess)
# ---------------------------------------------------------------------------

def _get_swap_used_bytes() -> int:
    """Get current swap used bytes via psutil (no fork)."""
    try:
        import psutil
        return psutil.swap_memory().used
    except Exception:
        return 0


def _compute_swap_io_mb_per_s(
    prev_swap_bytes: int,
    curr_swap_bytes: int,
    dt: float,
) -> float:
    """Estimate swap I/O throughput from swap-used deltas.

    This is a lower bound — it only captures net growth, not churn
    (pages swapped in+out that net to zero). But it catches the
    runaway pressure case where swap is growing fast.
    """
    if dt <= 0:
        return 0.0
    delta = abs(curr_swap_bytes - prev_swap_bytes)
    return delta / (1024 * 1024) / dt


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class Profiler:
    """Background thread profiler — samples at fixed interval."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._swap_baseline: float = 0.0
        self._psutil_proc = None  # Reused across samples (fix: CPU always 0.0)
        self._prev_swap_bytes: int = 0
        self._prev_swap_time: float = 0.0
        self.result: ProfileResult = ProfileResult()

    def start(self):
        self.result = ProfileResult()
        self._stop.clear()
        self._start_time = time.perf_counter()
        self._swap_baseline = _get_swap_gb()

        # Reuse ONE psutil.Process() across all samples so cpu_percent()
        # returns meaningful deltas (fresh objects always return 0.0).
        try:
            import psutil
            self._psutil_proc = psutil.Process()
            self._psutil_proc.cpu_percent(interval=None)  # Prime first reading
        except Exception:
            self._psutil_proc = None

        # Prime swap I/O throughput tracking
        self._prev_swap_bytes = _get_swap_used_bytes()
        self._prev_swap_time = time.perf_counter()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> ProfileResult:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.result.total_seconds = time.perf_counter() - self._start_time
        self._compute_aggregates()
        return self.result

    def _sample_process_stats(self):
        """Return (rss_gb, cpu_pct) using the persistent Process object."""
        if self._psutil_proc is None:
            return (0.0, 0.0)
        try:
            return (
                self._psutil_proc.memory_info().rss / 1e9,
                self._psutil_proc.cpu_percent(interval=None),
            )
        except Exception:
            return (0.0, 0.0)

    def _loop(self):
        while not self._stop.is_set():
            try:
                now = time.perf_counter()
                t = now - self._start_time
                rss, cpu = self._sample_process_stats()
                phys_footprint = _get_phys_footprint_gb()
                swap = _get_swap_gb() - self._swap_baseline

                # Swap I/O throughput (from swap-used deltas, no fork)
                curr_swap_bytes = _get_swap_used_bytes()
                dt = now - self._prev_swap_time
                swap_io = _compute_swap_io_mb_per_s(
                    self._prev_swap_bytes, curr_swap_bytes, dt,
                )
                self._prev_swap_bytes = curr_swap_bytes
                self._prev_swap_time = now

                sample = ProfileSample(
                    t=t,
                    metal_active_gb=mx.get_active_memory() / 1e9,
                    metal_peak_gb=mx.get_peak_memory() / 1e9,
                    rss_gb=rss,
                    phys_footprint_gb=phys_footprint,
                    cpu_pct=cpu,
                    swap_gb=swap,
                    swap_io_mb_per_s=round(swap_io, 1),
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
        self.result.phys_footprint_peak_gb = max(x.phys_footprint_gb for x in s)
        self.result.cpu_avg_pct = sum(x.cpu_pct for x in s) / len(s)
        self.result.cpu_peak_pct = max(x.cpu_pct for x in s)
        self.result.swap_peak_gb = max(x.swap_gb for x in s)
        self.result.swap_io_peak_mb_per_s = max(x.swap_io_mb_per_s for x in s)
