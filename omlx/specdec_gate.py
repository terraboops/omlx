# SPDX-License-Identifier: Apache-2.0
"""MagicDec-style cost-model gate for speculative decoding decisions.

Derived from MagicDec (arXiv:2408.11049) Eq. 2-4, re-fit for single-user
(batch=1) interactive serving on Apple Silicon unified memory.

The gate predicts whether speculative decoding (EAGLE-2, TriForce, or
Lookahead) would speed up decode at a given context length, based on
measured per-token KV-load vs model-compute latencies.

Key insight: speculative decoding amortizes the model compute cost across
N draft tokens, but pays a verification overhead. The crossover point
depends on:
  - context_len: longer context → more KV-load → spec-decode more beneficial
  - accept_rate: higher acceptance → more tokens per verification → better
  - draft_overhead: cost of generating draft tokens (cheap draft model or
    n-gram table vs full model forward pass)

Usage:
    from omlx.specdec_gate import SpecDecGate, load_constants
    gate = SpecDecGate(load_constants())
    should, info = gate.should_speculate(context_len=16384)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONSTANTS_PATH = Path(__file__).parent / "specdec_constants.json"

# Default constants (conservative estimates for M4 Pro 48GB with
# Qwen3-Coder-30B-A3B-8bit in duo KV mode). These get replaced by
# calibration data from specdec_calibrate.py.
DEFAULT_CONSTANTS = {
    # Per-token KV load latency (µs) as a function of context length.
    # Model: kv_load_us = kv_base_us + kv_per_token_us * context_len
    "kv_base_us": 50.0,         # Fixed overhead per decode step
    "kv_per_token_us": 0.005,   # Per-token KV load cost (µs/token)

    # Model compute latency per token (µs) — dominated by MoE FFN.
    # Relatively constant across context lengths on M4 Pro.
    "compute_us": 18500.0,      # ~18.5ms per token ≈ 54 tok/s

    # Speculative decoding parameters
    "draft_overhead_factor": 0.15,  # Draft cost as fraction of verify cost
    "verify_overhead_factor": 1.2,  # Verification is ~1.2x a single forward

    # Minimum context length to even consider spec-decode
    "min_context": 4096,

    # Calibration metadata
    "model": "default",
    "calibrated": False,
}


@dataclass
class SpecDecDecision:
    """Result of the speculative decoding gate."""
    should_speculate: bool
    projected_speedup: float  # e.g., 1.5 means 50% faster
    reason: str
    context_len: int
    accept_rate: float
    draft_tokens: int


class SpecDecGate:
    """Cost-model gate for speculative decoding decisions."""

    def __init__(self, constants: dict | None = None):
        self.c = constants or DEFAULT_CONSTANTS

    def _autoregressive_latency_us(self, context_len: int) -> float:
        """Latency of one autoregressive decode step (µs)."""
        kv_load = self.c["kv_base_us"] + self.c["kv_per_token_us"] * context_len
        compute = self.c["compute_us"]
        return kv_load + compute

    def _speculative_latency_us(self, context_len: int, draft_tokens: int,
                                 accept_rate: float) -> float:
        """Latency of one speculative decoding cycle (µs).

        A cycle: generate `draft_tokens` drafts, then verify all at once.
        Expected accepted tokens: draft_tokens * accept_rate.
        Total tokens produced per cycle: 1 + draft_tokens * accept_rate
        (the +1 is the guaranteed token from verification).
        """
        # Draft cost: draft_tokens × (draft_overhead × auto_latency)
        auto_lat = self._autoregressive_latency_us(context_len)
        draft_cost = draft_tokens * self.c["draft_overhead_factor"] * auto_lat

        # Verification cost: one forward pass over draft_tokens + 1 tokens
        # This is approximately verify_overhead_factor × auto_latency
        # (the extra tokens are processed in parallel, so marginal cost is small)
        verify_cost = self.c["verify_overhead_factor"] * auto_lat

        return draft_cost + verify_cost

    def should_speculate(self, context_len: int,
                          accept_rate: float = 0.7,
                          draft_tokens: int = 5) -> tuple[bool, SpecDecDecision]:
        """Decide whether speculative decoding would be faster.

        Args:
            context_len: Current context length in tokens
            accept_rate: Expected draft token acceptance rate (0-1)
            draft_tokens: Number of draft tokens per cycle

        Returns:
            (should_speculate, decision_details)
        """
        # Below minimum context, never speculate
        if context_len < self.c["min_context"]:
            return False, SpecDecDecision(
                should_speculate=False,
                projected_speedup=1.0,
                reason=f"context {context_len} < min {self.c['min_context']}",
                context_len=context_len,
                accept_rate=accept_rate,
                draft_tokens=draft_tokens,
            )

        # Autoregressive: 1 token per forward pass
        auto_lat = self._autoregressive_latency_us(context_len)

        # Speculative: expected_tokens per cycle
        spec_lat = self._speculative_latency_us(context_len, draft_tokens, accept_rate)
        expected_tokens = 1 + draft_tokens * accept_rate

        # Per-token latency comparison
        auto_per_token = auto_lat
        spec_per_token = spec_lat / expected_tokens

        speedup = auto_per_token / spec_per_token if spec_per_token > 0 else 1.0

        should = speedup > 1.1  # Need at least 10% speedup to justify complexity

        reason = (f"auto={auto_per_token:.0f}µs/tok, "
                  f"spec={spec_per_token:.0f}µs/tok ({expected_tokens:.1f} tok/cycle), "
                  f"speedup={speedup:.2f}x")

        return should, SpecDecDecision(
            should_speculate=should,
            projected_speedup=round(speedup, 3),
            reason=reason,
            context_len=context_len,
            accept_rate=accept_rate,
            draft_tokens=draft_tokens,
        )


def load_constants(path: Path | str | None = None) -> dict:
    """Load calibration constants from JSON file."""
    p = Path(path) if path else CONSTANTS_PATH
    if p.exists():
        try:
            data = json.loads(p.read_text())
            logger.info(f"Loaded specdec constants from {p} "
                        f"(calibrated={data.get('calibrated', False)})")
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {p}: {e}, using defaults")
    return DEFAULT_CONSTANTS.copy()


def save_constants(constants: dict, path: Path | str | None = None):
    """Save calibration constants to JSON file."""
    p = Path(path) if path else CONSTANTS_PATH
    p.write_text(json.dumps(constants, indent=2))
    logger.info(f"Saved specdec constants to {p}")
