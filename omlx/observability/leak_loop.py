# SPDX-License-Identifier: Apache-2.0
"""Leak detection harness.

Runs a user-supplied iteration body repeatedly and records heap state.
Emits a per-iteration CSV and a terminal verdict based on linear-fit
slope of the Metal peak.

A clean implementation shows a flat Metal peak after warmup. A leak
shows a monotone climb — the slope (GB/iter) is the canonical measure.

Usage from Python
-----------------

    from omlx.observability.leak_loop import leak_loop

    def body(i):
        # one iteration of whatever you want to stress
        out = generate_tokens(prompt, n=64)
        return out

    result = leak_loop(body, iterations=200, warmup=10, csv_path="/tmp/leak.csv")
    print(result.verdict())

Usage from CLI
--------------

    python -m omlx.observability.leak_loop --iterations 200 \\
        --target "omlx.bench.snapkv_bench:quick_decode"
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from omlx.observability.heap import HeapSnapshot, snapshot

logger = logging.getLogger("omlx.observability.leak")


@dataclass
class LeakResult:
    iterations: int
    warmup: int
    snapshots: List[HeapSnapshot] = field(default_factory=list)
    dt_per_iter_s: List[float] = field(default_factory=list)
    csv_path: Optional[Path] = None

    def _post_warmup(self) -> List[HeapSnapshot]:
        return self.snapshots[self.warmup:]

    def slope_metal_peak_gb_per_iter(self) -> float:
        """Least-squares slope of Metal peak GB vs iteration, post-warmup."""
        pts = self._post_warmup()
        if len(pts) < 2:
            return 0.0
        xs = list(range(len(pts)))
        ys = [s.metal.get("peak_gb", 0.0) for s in pts]
        n = len(xs)
        mx_ = sum(xs) / n
        my_ = sum(ys) / n
        num = sum((xs[i] - mx_) * (ys[i] - my_) for i in range(n))
        den = sum((xs[i] - mx_) ** 2 for i in range(n))
        if den == 0:
            return 0.0
        return num / den

    def slope_phys_footprint_gb_per_iter(self) -> float:
        pts = self._post_warmup()
        if len(pts) < 2:
            return 0.0
        xs = list(range(len(pts)))
        ys = [s.phys_footprint_gb for s in pts]
        n = len(xs)
        mx_ = sum(xs) / n
        my_ = sum(ys) / n
        num = sum((xs[i] - mx_) * (ys[i] - my_) for i in range(n))
        den = sum((xs[i] - mx_) ** 2 for i in range(n))
        return num / den if den else 0.0

    def verdict(self, threshold_mb_per_iter: float = 1.0) -> str:
        slope_gb = self.slope_metal_peak_gb_per_iter()
        fp_gb = self.slope_phys_footprint_gb_per_iter()
        slope_mb = slope_gb * 1024
        fp_mb = fp_gb * 1024
        if not self._post_warmup():
            return "INCONCLUSIVE: no post-warmup samples"
        tag = (
            "LEAK" if (slope_mb > threshold_mb_per_iter or fp_mb > threshold_mb_per_iter)
            else "CLEAN"
        )
        return (
            f"{tag}: metal_peak={slope_mb:+.3f} MB/iter  "
            f"phys_fp={fp_mb:+.3f} MB/iter  "
            f"(threshold={threshold_mb_per_iter} MB/iter, "
            f"n={len(self._post_warmup())} post-warmup iters)"
        )


def leak_loop(
    body: Callable[[int], object],
    iterations: int = 200,
    warmup: int = 10,
    csv_path: Optional[str] = None,
    gc_every: int = 0,
    clear_cache_every: int = 0,
    log_every: int = 10,
) -> LeakResult:
    """Run `body(i)` for `iterations` and record heap state per iter.

    Parameters
    ----------
    body : callable
        Function called with the iteration index. Its return value is
        discarded. Keep it fast — the whole loop is bounded by this.
    iterations : int
        Total iterations including warmup.
    warmup : int
        Iterations excluded from slope analysis. Let JIT + caches settle.
    csv_path : str, optional
        Path for per-iteration CSV (iter, t, metal_active, metal_peak, rss, phys_fp, swap).
    gc_every : int
        If > 0, call gc.collect() every N iters. Use to attribute leaks to
        Python vs Metal (if gc eliminates the slope, it's a Python ref leak).
    clear_cache_every : int
        If > 0, call mx.clear_cache() every N iters. Same attribution idea
        for Metal pool fragmentation vs a true leak.
    log_every : int
        Stdout progress cadence.

    Returns
    -------
    LeakResult with slopes, verdict(), and optional CSV on disk.
    """
    result = LeakResult(iterations=iterations, warmup=warmup)
    if csv_path:
        result.csv_path = Path(csv_path)
        result.csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_f = result.csv_path.open("w", newline="")
        writer = csv.writer(csv_f)
        writer.writerow([
            "iter", "t_s", "metal_active_gb", "metal_peak_gb",
            "rss_gb", "phys_footprint_gb", "swap_gb", "body_dt_s",
        ])
    else:
        csv_f = None
        writer = None

    t_start = time.perf_counter()
    try:
        for i in range(iterations):
            iter_t0 = time.perf_counter()
            try:
                body(i)
            except Exception as e:
                logger.error("iteration %d raised: %s", i, e)
                raise
            iter_dt = time.perf_counter() - iter_t0
            result.dt_per_iter_s.append(iter_dt)

            if gc_every and (i + 1) % gc_every == 0:
                gc.collect()
            if clear_cache_every and (i + 1) % clear_cache_every == 0:
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except Exception:
                    pass

            snap = snapshot(label=f"iter{i}")
            result.snapshots.append(snap)
            if writer is not None:
                writer.writerow([
                    i,
                    round(snap.t - t_start, 4),
                    snap.metal.get("active_gb", 0.0),
                    snap.metal.get("peak_gb", 0.0),
                    snap.rss_gb,
                    snap.phys_footprint_gb,
                    snap.swap_used_gb,
                    round(iter_dt, 4),
                ])
                csv_f.flush()  # Survive a crash mid-loop

            if log_every and (i + 1) % log_every == 0:
                peak = snap.metal.get("peak_gb", 0.0)
                active = snap.metal.get("active_gb", 0.0)
                logger.info(
                    "iter %d/%d  metal active=%.4f peak=%.4f  phys_fp=%.4f  body_dt=%.3fs",
                    i + 1, iterations, active, peak,
                    snap.phys_footprint_gb, iter_dt,
                )
    finally:
        if csv_f is not None:
            csv_f.close()

    return result


def _import_callable(spec: str) -> Callable:
    """Parse a 'module.path:attr' spec into the callable."""
    if ":" not in spec:
        raise ValueError(f"target must be 'module.path:attr', got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", required=True,
                    help="'module.path:callable' invoked as body(i)")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--gc-every", type=int, default=0)
    ap.add_argument("--clear-cache-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--threshold-mb", type=float, default=1.0,
                    help="Leak verdict threshold in MB/iter")
    args = ap.parse_args()

    body = _import_callable(args.target)
    result = leak_loop(
        body=body,
        iterations=args.iterations,
        warmup=args.warmup,
        csv_path=args.csv,
        gc_every=args.gc_every,
        clear_cache_every=args.clear_cache_every,
        log_every=args.log_every,
    )
    verdict = result.verdict(threshold_mb_per_iter=args.threshold_mb)
    print(verdict)
    if result.csv_path:
        print(f"csv: {result.csv_path}")
    return 0 if verdict.startswith("CLEAN") else 2


if __name__ == "__main__":
    raise SystemExit(main())
