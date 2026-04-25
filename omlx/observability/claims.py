# SPDX-License-Identifier: Apache-2.0
"""Stale-claims verification harness.

CLAUDE.md and the task log accumulate claimed multipliers ("248×",
"1.33× at 8K", "16% improvement"). The analyst's job is to verify them.

This module provides a tiny declarative DSL for stale-claim checks:

    claim(
        name="duokv_pre_alloc_slab_64k",
        description="DuoKV pre-alloc slab: 248x faster decode at 64K",
        baseline=lambda: ...,
        optimized=lambda: ...,
        expected_ratio=248.0,
        tolerance=0.5,   # within 50% of claim
    )

Run all claims with `verify_all()` — outputs a JSON report with claimed
ratio, observed ratio, verdict (PASS / SUSPECT / FAIL), and the raw
timings. Each claim is run N times for stability.

The harness does NOT load the model — it expects the analyst to pass
fixtures (cache instances, tensors) for whichever claim they care about.
That keeps each verification cheap and isolated.
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union


@dataclass
class ClaimResult:
    name: str
    description: str
    expected_ratio: float
    tolerance: float
    baseline_ms: List[float] = field(default_factory=list)
    optimized_ms: List[float] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def baseline_median_ms(self) -> float:
        return statistics.median(self.baseline_ms) if self.baseline_ms else 0.0

    @property
    def optimized_median_ms(self) -> float:
        return statistics.median(self.optimized_ms) if self.optimized_ms else 0.0

    @property
    def observed_ratio(self) -> float:
        opt = self.optimized_median_ms
        if opt <= 0:
            return float("inf")
        return self.baseline_median_ms / opt

    def verdict(self) -> str:
        if self.error:
            return "ERROR"
        if not self.baseline_ms or not self.optimized_ms:
            return "INCONCLUSIVE"
        if self.expected_ratio <= 0:
            return "BAD_CLAIM"
        ratio = self.observed_ratio
        # Pass band: within +/- tolerance fraction of expected.
        low = self.expected_ratio * (1.0 - self.tolerance)
        high = self.expected_ratio * (1.0 + self.tolerance)
        if low <= ratio <= high:
            return "PASS"
        # Beats the claim — flag because it could mean the baseline is
        # broken or the test is not isolating the work.
        if ratio > high:
            return "EXCEEDS"
        # Misses by > 4x: the claim is dead, not just suspect.
        if ratio < self.expected_ratio * 0.25:
            return "FAIL"
        return "SUSPECT"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "expected_ratio": self.expected_ratio,
            "tolerance": self.tolerance,
            "baseline_median_ms": round(self.baseline_median_ms, 4),
            "optimized_median_ms": round(self.optimized_median_ms, 4),
            "observed_ratio": round(self.observed_ratio, 3),
            "verdict": self.verdict(),
            "samples_n": len(self.baseline_ms),
            "baseline_ms": [round(x, 4) for x in self.baseline_ms],
            "optimized_ms": [round(x, 4) for x in self.optimized_ms],
            "error": self.error,
        }


@dataclass
class Claim:
    name: str
    description: str
    baseline: Callable[[], object]
    optimized: Callable[[], object]
    expected_ratio: float
    tolerance: float = 0.5
    repeats: int = 5
    warmup: int = 2
    setup: Optional[Callable[[], object]] = None
    teardown: Optional[Callable[[], None]] = None


def _time_callable(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    # Force MLX synchronization so async GPU work counts.
    try:
        import mlx.core as mx
        mx.synchronize()
    except Exception:
        pass
    return (time.perf_counter() - t0) * 1000


def run_claim(claim: Claim) -> ClaimResult:
    """Run a single claim, returning timings and verdict."""
    result = ClaimResult(
        name=claim.name,
        description=claim.description,
        expected_ratio=claim.expected_ratio,
        tolerance=claim.tolerance,
    )
    try:
        if claim.setup is not None:
            claim.setup()

        # Warmup
        for _ in range(claim.warmup):
            claim.baseline()
            claim.optimized()
        gc.collect()

        # Alternate runs to fight thermal/memory drift
        for _ in range(claim.repeats):
            result.baseline_ms.append(_time_callable(claim.baseline))
            result.optimized_ms.append(_time_callable(claim.optimized))

        if claim.teardown is not None:
            claim.teardown()
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


def verify_all(claims: List[Claim], dump_path: Optional[Union[str, Path]] = None) -> List[ClaimResult]:
    """Run a list of claims, optionally dumping results to JSON."""
    results = [run_claim(c) for c in claims]
    if dump_path is not None:
        p = Path(dump_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([r.to_dict() for r in results], indent=2))
    return results


def format_summary(results: List[ClaimResult]) -> str:
    lines = [
        f"{'name':<40} {'verdict':<10} {'expected':>10} {'observed':>10} {'b_ms':>10} {'o_ms':>10}",
        "-" * 100,
    ]
    for r in results:
        verdict = r.verdict()
        ratio = r.observed_ratio
        ratio_str = f"{ratio:>10.2f}" if ratio != float("inf") else f"{'inf':>10}"
        lines.append(
            f"{r.name:<40} {verdict:<10} "
            f"{r.expected_ratio:>10.2f} {ratio_str} "
            f"{r.baseline_median_ms:>10.3f} {r.optimized_median_ms:>10.3f}"
        )
    return "\n".join(lines)
