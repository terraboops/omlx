# SPDX-License-Identifier: Apache-2.0
"""Central registry for timers and counters.

Thread-safe singleton. Holds named `TimerStats` and `Counter` objects
and knows how to render a summary or dump JSON. Designed so the analyst
can sprinkle instrumentation across the codebase and pull a single
aggregated report at run end.

The registry can be globally disabled via `OMLX_OBSERVABILITY=0` to make
`timer`/`counter` calls effectively no-ops (cheap `if disabled: return`
short-circuit). Useful when shipping instrumentation into hot paths:
leave it in place, turn it on only during analysis.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class TimerStats:
    """Aggregated stats for a named timer.

    Stores raw samples so we can compute percentiles at flush time.
    Sample retention is bounded: after `sample_cap` samples, we switch
    to reservoir sampling (Algorithm R) to keep a uniform random sample
    without unbounded memory.
    """

    name: str
    count: int = 0
    total_s: float = 0.0
    min_s: float = float("inf")
    max_s: float = 0.0
    samples: List[float] = field(default_factory=list)
    sample_cap: int = 4096
    _reservoir_seen: int = 0

    def record(self, dt: float) -> None:
        self.count += 1
        self.total_s += dt
        if dt < self.min_s:
            self.min_s = dt
        if dt > self.max_s:
            self.max_s = dt
        self._reservoir_add(dt)

    def _reservoir_add(self, dt: float) -> None:
        self._reservoir_seen += 1
        if len(self.samples) < self.sample_cap:
            self.samples.append(dt)
            return
        # Reservoir sampling: keep uniform random subset
        import random
        j = random.randint(0, self._reservoir_seen - 1)
        if j < self.sample_cap:
            self.samples[j] = dt

    def percentile(self, p: float) -> float:
        if not self.samples:
            return 0.0
        s = sorted(self.samples)
        k = max(0, min(len(s) - 1, int(p * (len(s) - 1))))
        return s[k]

    def summary(self) -> dict:
        mean = self.total_s / self.count if self.count else 0.0
        return {
            "name": self.name,
            "count": self.count,
            "total_s": round(self.total_s, 6),
            "mean_ms": round(mean * 1000, 4),
            "min_ms": round(self.min_s * 1000, 4) if self.count else 0.0,
            "p50_ms": round(self.percentile(0.50) * 1000, 4),
            "p95_ms": round(self.percentile(0.95) * 1000, 4),
            "p99_ms": round(self.percentile(0.99) * 1000, 4),
            "max_ms": round(self.max_s * 1000, 4),
        }


@dataclass
class CounterStats:
    """Monotonic counter or gauge."""
    name: str
    value: float = 0.0
    count: int = 0  # Number of add() calls; distinct from value sum

    def add(self, delta: float = 1.0) -> None:
        self.value += delta
        self.count += 1

    def set(self, v: float) -> None:
        self.value = v
        self.count += 1

    def summary(self) -> dict:
        return {"name": self.name, "value": self.value, "updates": self.count}


class Registry:
    """Thread-safe holder for timers and counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timers: Dict[str, TimerStats] = {}
        self._counters: Dict[str, CounterStats] = {}
        self._started_at: float = time.perf_counter()
        # Opt-out via env. Defaults on so dropping `timer()` into code works.
        self.enabled: bool = os.environ.get("OMLX_OBSERVABILITY", "1") != "0"

    def timer(self, name: str) -> TimerStats:
        with self._lock:
            t = self._timers.get(name)
            if t is None:
                t = TimerStats(name=name)
                self._timers[name] = t
            return t

    def counter(self, name: str) -> CounterStats:
        with self._lock:
            c = self._counters.get(name)
            if c is None:
                c = CounterStats(name=name)
                self._counters[name] = c
            return c

    def reset(self) -> None:
        with self._lock:
            self._timers.clear()
            self._counters.clear()
            self._started_at = time.perf_counter()

    def snapshot(self) -> dict:
        with self._lock:
            elapsed = time.perf_counter() - self._started_at
            timers = [t.summary() for t in self._timers.values()]
            counters = [c.summary() for c in self._counters.values()]
        timers.sort(key=lambda t: t["total_s"], reverse=True)
        counters.sort(key=lambda c: c["name"])
        return {
            "elapsed_s": round(elapsed, 4),
            "timers": timers,
            "counters": counters,
        }

    def dump(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.snapshot(), indent=2))
        return p

    def print_summary(self, top_n: Optional[int] = 20) -> None:
        snap = self.snapshot()
        print(f"observability run: {snap['elapsed_s']:.2f}s elapsed")
        timers = snap["timers"]
        if timers:
            print(f"\n  top timers by total_s (n={len(timers)}):")
            print(f"  {'name':<40} {'count':>8} {'total_s':>10} {'mean_ms':>10} {'p95_ms':>10}")
            for t in timers[:top_n]:
                print(
                    f"  {t['name']:<40} {t['count']:>8} "
                    f"{t['total_s']:>10.4f} {t['mean_ms']:>10.3f} {t['p95_ms']:>10.3f}"
                )
        if snap["counters"]:
            print(f"\n  counters (n={len(snap['counters'])}):")
            for c in snap["counters"]:
                print(f"  {c['name']:<40} value={c['value']:<15} updates={c['updates']}")


registry = Registry()


def reset() -> None:
    """Reset all timers and counters in the global registry."""
    registry.reset()
