# SPDX-License-Identifier: Apache-2.0
"""Scoring function for hypercar feature combinations.

Weighted scoring:
  - Decode speed (primary target: 25+ tok/s)
  - Prefill speed (buffer time — minor weight)
  - Memory efficiency (peak below budget)
  - Stability (no swap, no leak, coherence pass)

Score is a single number 0-100. Higher = better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class RunMetrics:
    """Metrics from a single feature-combination run."""
    # Identification
    run_id: str
    features: Dict[str, object]  # flag name → value

    # Timing
    prefill_toks: float = 0.0
    decode_toks: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0

    # Memory
    metal_peak_gb: float = 0.0
    metal_avg_gb: float = 0.0
    rss_peak_gb: float = 0.0
    swap_peak_gb: float = 0.0

    # CPU
    cpu_avg_pct: float = 0.0
    cpu_peak_pct: float = 0.0

    # Correctness
    coherence_pass: bool = False
    fidelity_cosine: float = 0.0
    niah_pass_rate: float = 0.0

    # Status
    status: str = "pending"  # pending, pass, fail_memory, fail_decode, fail_prefill, fail_coherence, error
    error_msg: str = ""

    # Computed
    score: float = 0.0


def score_run(m: RunMetrics, target_decode_toks: float = 25.0, memory_budget_gb: float = 38.0) -> float:
    """Compute 0-100 score for a single run.

    Weights:
      50% — decode speed (main target)
      20% — prefill speed (buffer time)
      15% — memory efficiency
      10% — stability (no swap, no leak)
       5% — quality (coherence, fidelity, NIAH)
    """
    if m.status != "pass":
        return 0.0

    # Decode score: 50 points, saturates at target × 2
    decode_score = min(m.decode_toks / target_decode_toks, 2.0) * 25.0

    # Prefill score: 20 points, target 100 tok/s
    prefill_score = min(m.prefill_toks / 100.0, 1.0) * 20.0

    # Memory score: 15 points, full marks if below 50% of budget
    mem_ratio = m.metal_peak_gb / memory_budget_gb
    if mem_ratio <= 0.5:
        memory_score = 15.0
    elif mem_ratio <= 1.0:
        memory_score = 15.0 * (1.0 - (mem_ratio - 0.5) / 0.5)  # linear decay
    else:
        memory_score = 0.0  # over budget

    # Stability score: 10 points
    stability_score = 10.0
    if m.swap_peak_gb > 0.5:
        stability_score *= 0.5  # penalize swap
    if m.swap_peak_gb > 2.0:
        stability_score *= 0.25  # heavy penalty

    # Quality score: 5 points
    quality_score = 0.0
    if m.coherence_pass:
        quality_score += 2.0
    if m.fidelity_cosine >= 0.95:
        quality_score += 1.5
    elif m.fidelity_cosine >= 0.90:
        quality_score += 0.75
    if m.niah_pass_rate >= 0.6:
        quality_score += 1.5
    elif m.niah_pass_rate >= 0.3:
        quality_score += 0.75

    total = decode_score + prefill_score + memory_score + stability_score + quality_score
    return round(total, 2)


def describe_run(m: RunMetrics) -> str:
    """Short one-line description."""
    features_str = ",".join(f"{k}={v}" for k, v in m.features.items() if v)
    return (
        f"[{m.score:>5.1f}pts] {m.run_id:>20s} | "
        f"dec={m.decode_toks:>5.1f}t/s pre={m.prefill_toks:>5.0f}t/s "
        f"mem={m.metal_peak_gb:>4.1f}GB swap={m.swap_peak_gb:>3.1f}GB "
        f"[{features_str}]"
    )
