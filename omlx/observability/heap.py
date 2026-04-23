# SPDX-License-Identifier: Apache-2.0
"""Heap snapshots for leak hunting.

Captures, at a single moment:
- Metal active / peak / cache sizes (mx.get_*_memory + device_info)
- Process RSS + macOS physical footprint (phys_footprint — the correct
  metric on Apple Silicon where Metal uses unified memory and does NOT
  appear in RSS)
- System swap (used GB)
- Top-N Python allocations (via `tracemalloc`, if started)

Diffing two snapshots is the primary analyst workflow:

    before = snapshot("before-prefill")
    run_prefill(...)
    after = snapshot("after-prefill")
    print(snapshot_diff(before, after).format())

A steady positive delta across many iterations of the same loop body is
the canonical leak signature. Take snapshots every N iterations and plot
the Metal peak — flat = clean, sawtooth = healthy GC, monotone = leak.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Reuse the phys_footprint ctypes helper from bench.profiler without a
# hard import dependency (the profiler module imports mlx lazily too).
def _phys_footprint_gb() -> float:
    try:
        libproc_path = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
        libproc = ctypes.CDLL(libproc_path)
        buf = (ctypes.c_uint8 * 512)()
        if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:
            return 0.0
        return int.from_bytes(bytes(buf[72:80]), "little") / 1e9
    except Exception:
        return 0.0


def _swap_used_gb() -> float:
    try:
        import psutil
        return psutil.swap_memory().used / 1e9
    except Exception:
        return 0.0


def _rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


def _metal_snapshot() -> Dict[str, float]:
    """Capture Metal allocator state.

    Returns dict with keys: active_gb, peak_gb, cache_gb (pool reservation).
    Keys may be missing if the MLX build does not expose them.
    """
    try:
        import mlx.core as mx
    except Exception:
        return {}
    out: Dict[str, float] = {}
    try:
        out["active_gb"] = mx.get_active_memory() / 1e9
    except Exception:
        pass
    try:
        out["peak_gb"] = mx.get_peak_memory() / 1e9
    except Exception:
        pass
    # Buffer cache pool — exposed on newer MLX, omit if missing
    for attr in ("get_cache_memory", "metal.get_cache_memory"):
        try:
            target = mx
            for part in attr.split("."):
                target = getattr(target, part)
            out["cache_gb"] = target() / 1e9
            break
        except Exception:
            continue
    return out


def _tracemalloc_top(n: int) -> List[Tuple[str, int]]:
    """Top-N Python allocations by size, or [] if tracemalloc isn't started."""
    if not tracemalloc.is_tracing():
        return []
    try:
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics("filename")[:n]
        return [(str(s.traceback), s.size) for s in stats]
    except Exception:
        return []


@dataclass
class HeapSnapshot:
    label: str
    t: float  # perf_counter at capture
    metal: Dict[str, float] = field(default_factory=dict)
    rss_gb: float = 0.0
    phys_footprint_gb: float = 0.0
    swap_used_gb: float = 0.0
    py_top: List[Tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "t": round(self.t, 4),
            "metal": {k: round(v, 4) for k, v in self.metal.items()},
            "rss_gb": round(self.rss_gb, 4),
            "phys_footprint_gb": round(self.phys_footprint_gb, 4),
            "swap_used_gb": round(self.swap_used_gb, 4),
            "py_top_count": len(self.py_top),
        }


def snapshot(label: str = "", py_top_n: int = 10) -> HeapSnapshot:
    """Capture a heap snapshot. Cheap (~1 ms); safe to call often."""
    return HeapSnapshot(
        label=label,
        t=time.perf_counter(),
        metal=_metal_snapshot(),
        rss_gb=_rss_gb(),
        phys_footprint_gb=_phys_footprint_gb(),
        swap_used_gb=_swap_used_gb(),
        py_top=_tracemalloc_top(py_top_n),
    )


@dataclass
class HeapDiff:
    before: HeapSnapshot
    after: HeapSnapshot

    @property
    def dt_s(self) -> float:
        return self.after.t - self.before.t

    @property
    def delta_metal_active_gb(self) -> float:
        return self.after.metal.get("active_gb", 0.0) - self.before.metal.get("active_gb", 0.0)

    @property
    def delta_metal_peak_gb(self) -> float:
        return self.after.metal.get("peak_gb", 0.0) - self.before.metal.get("peak_gb", 0.0)

    @property
    def delta_rss_gb(self) -> float:
        return self.after.rss_gb - self.before.rss_gb

    @property
    def delta_phys_footprint_gb(self) -> float:
        return self.after.phys_footprint_gb - self.before.phys_footprint_gb

    @property
    def delta_swap_gb(self) -> float:
        return self.after.swap_used_gb - self.before.swap_used_gb

    def to_dict(self) -> dict:
        return {
            "dt_s": round(self.dt_s, 4),
            "delta_metal_active_gb": round(self.delta_metal_active_gb, 4),
            "delta_metal_peak_gb": round(self.delta_metal_peak_gb, 4),
            "delta_rss_gb": round(self.delta_rss_gb, 4),
            "delta_phys_footprint_gb": round(self.delta_phys_footprint_gb, 4),
            "delta_swap_gb": round(self.delta_swap_gb, 4),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }

    def format(self) -> str:
        lines = [
            f"heap diff [{self.before.label} → {self.after.label}] dt={self.dt_s:.2f}s",
            f"  metal active: {self.delta_metal_active_gb:+.3f} GB"
            f" ({self.before.metal.get('active_gb', 0):.2f} → {self.after.metal.get('active_gb', 0):.2f})",
            f"  metal peak  : {self.delta_metal_peak_gb:+.3f} GB"
            f" ({self.before.metal.get('peak_gb', 0):.2f} → {self.after.metal.get('peak_gb', 0):.2f})",
            f"  rss         : {self.delta_rss_gb:+.3f} GB",
            f"  phys fp     : {self.delta_phys_footprint_gb:+.3f} GB",
            f"  swap        : {self.delta_swap_gb:+.3f} GB",
        ]
        return "\n".join(lines)


def snapshot_diff(before: HeapSnapshot, after: Optional[HeapSnapshot] = None) -> HeapDiff:
    """Diff two snapshots. If `after` is omitted, captures now."""
    if after is None:
        after = snapshot(label="now")
    return HeapDiff(before=before, after=after)
