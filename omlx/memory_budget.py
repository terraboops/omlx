# SPDX-License-Identifier: Apache-2.0
"""Adaptive memory budget for prefill/decode workloads.

The hypercar tradeoff: use ALL available memory headroom for prefill
to maximize throughput, BUT reserve enough for decode working set and
active KV caches.

On a 48GB M4 Pro with Qwen3-30B loaded (17.2GB):
  - Total: 48GB
  - Model: 17.2GB (fixed)
  - Decode working set: ~1GB (activations + attention)
  - Safety margin: 4GB (OS + apps + fragmentation)
  - Available for prefill + KV: ~25GB

When user CLEARS context (no active KV), budget expands:
  - Can use 25GB for large dequant chunks (16K-32K)
  - Fast prefill via big batch sizes

When user is DECODING (KV cache active at 65K = 1.4GB):
  - Reserve decode working set (1GB)
  - Budget shrinks to 23GB for background prefill

When user has LARGE active KV (256K = 5GB):
  - Budget shrinks to 19GB for dequant/prefill
  - Smaller chunks to avoid eviction
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from omlx.utils.hardware import get_max_working_set_bytes, get_total_memory_bytes

logger = logging.getLogger(__name__)

# Reserve for OS, GUI apps, filesystem cache, etc.
DEFAULT_OS_RESERVE_GB = 4.0
# Minimum free space for decode working set during generation
DECODE_WORKING_SET_GB = 1.0
# Minimum safety margin never touched
SAFETY_MARGIN_GB = 2.0


@dataclass
class MemoryBudget:
    """Snapshot of memory budget at a point in time."""
    total_gb: float              # Total unified memory
    max_working_set_gb: float    # MLX's recommended max
    model_gb: float              # Estimated model weights (baseline active)
    active_kv_gb: float          # Currently-used KV cache (fp16 + compressed)
    current_active_gb: float     # mx.get_active_memory() right now
    available_gb: float          # Remaining headroom for new allocations
    prefill_budget_gb: float     # How much to spend on prefill intermediates

    def describe(self) -> str:
        return (
            f"MemoryBudget: total={self.total_gb:.1f}GB "
            f"max_ws={self.max_working_set_gb:.1f}GB "
            f"active={self.current_active_gb:.1f}GB "
            f"available={self.available_gb:.1f}GB "
            f"prefill_budget={self.prefill_budget_gb:.1f}GB"
        )


def compute_budget(
    model_gb: float = 17.2,
    active_kv_gb: float = 0.0,
    decoding: bool = False,
    os_reserve_gb: float = DEFAULT_OS_RESERVE_GB,
) -> MemoryBudget:
    """Compute memory budget for new allocations.

    Args:
        model_gb: Estimated model weight memory (constant, ~17GB for Qwen3-30B).
        active_kv_gb: Currently-allocated KV cache (use 0 for fresh prefill).
        decoding: If True, reserve working set for ongoing decode.
        os_reserve_gb: Memory reserved for OS/apps (default 4GB).

    Returns:
        MemoryBudget describing how much to spend on prefill intermediates.
    """
    total_bytes = get_total_memory_bytes()
    max_ws_bytes = get_max_working_set_bytes()

    total_gb = total_bytes / 1e9
    max_ws_gb = max_ws_bytes / 1e9
    current_active_gb = mx.get_active_memory() / 1e9

    # Use the smaller of total and max working set (MLX's safety limit)
    usable_gb = min(total_gb, max_ws_gb)

    # Reserve for OS, decode, safety margin
    reserved_gb = os_reserve_gb + SAFETY_MARGIN_GB
    if decoding:
        reserved_gb += DECODE_WORKING_SET_GB

    # What's already allocated (model + active KV cache)
    allocated_gb = model_gb + active_kv_gb

    # Available = total - reserved - allocated
    available_gb = max(0.0, usable_gb - reserved_gb - allocated_gb)

    # Prefill budget is the full available amount (can be reduced if decoding)
    prefill_budget_gb = available_gb

    return MemoryBudget(
        total_gb=total_gb,
        max_working_set_gb=max_ws_gb,
        model_gb=model_gb,
        active_kv_gb=active_kv_gb,
        current_active_gb=current_active_gb,
        available_gb=available_gb,
        prefill_budget_gb=prefill_budget_gb,
    )


def compute_dequant_chunk_size(
    query_len: int,
    num_query_heads: int,
    head_dim: int,
    num_layers: int,
    target_bytes: int = 12 * 1024**3,  # 12GB default (will be capped by budget)
    budget: Optional[MemoryBudget] = None,
) -> int:
    """Compute optimal dequant chunk size given memory budget.

    Memory per streaming_tq_attention call:
      scores tensor: query_len × chunk_size × num_query_heads × 4 bytes
      K/V chunks:    2 × chunk_size × num_kv_heads × head_dim × 2 bytes (fp16)
      Total:         chunk_size × (query_len × H_q × 4 + 2 × H_kv × D × 2)

    We want this to fit within target_bytes per layer. With vertical_eval
    patch, only eval_every layers accumulate simultaneously.

    Args:
        query_len: L (number of queries in this prefill chunk)
        num_query_heads: H_q
        head_dim: D
        num_layers: total model layers (for cross-layer estimation)
        target_bytes: memory budget for streaming attention intermediates
        budget: precomputed MemoryBudget (computed fresh if None)

    Returns:
        Chunk size (power of 2, between 1024 and 65536)
    """
    if budget is None:
        budget = compute_budget()

    # Empirical peak formula: peak_bytes ≈ 2.5 × L × chunk × H_q × 4 bytes
    # (accounts for scores tensor + softmax + attention outputs)
    # With vertical_eval, only ONE layer's streaming_tq_attention runs at
    # peak at a time (mx.eval between chunks collapses the graph).
    #
    # So: peak = 2.5 × query_len × chunk_size × num_query_heads × 4
    # Solve for chunk_size: chunk_size = peak_budget / (10 × query_len × H_q)

    # Use 70% of prefill budget for attention, 30% reserve
    budget_bytes = int(budget.prefill_budget_gb * 1e9 * 0.7)
    effective_target = min(target_bytes, budget_bytes)

    # Max chunk that fits within peak budget.
    # peak = 2.5 × L × chunk × H_q × 4 bytes (empirical)
    # Apply safety factor tau=1.2 for MLX version variance
    peak_multiplier = 10  # 2.5 × 4 bytes
    tau = 1.2  # Safety factor for cross-version / cross-kernel variance
    max_chunk = int(effective_target / (peak_multiplier * query_len * num_query_heads * tau))

    # Round down to power of 2
    if max_chunk >= 65536:
        chunk = 65536
    elif max_chunk >= 32768:
        chunk = 32768
    elif max_chunk >= 16384:
        chunk = 16384
    elif max_chunk >= 8192:
        chunk = 8192
    elif max_chunk >= 4096:
        chunk = 4096
    elif max_chunk >= 2048:
        chunk = 2048
    elif max_chunk >= 1024:
        chunk = 1024
    else:
        chunk = 512  # Minimum

    logger.debug(
        "compute_dequant_chunk_size: L=%d budget=%.1fGB target=%.1fGB → chunk=%d",
        query_len, budget.prefill_budget_gb, effective_target / 1e9, chunk,
    )

    return chunk


def estimate_kv_cache_gb(
    num_tokens: int,
    num_layers: int = 47,  # layers 1-47 are TQ3
    num_kv_heads: int = 4,
    head_dim: int = 128,
    bits: int = 3,
    layer_0_fp16: bool = True,
) -> float:
    """Estimate compressed KV cache size for given context length.

    TQ3 storage per token: 2 × num_kv_heads × (head_dim × bits / 8 + 4 bytes norms)
                           (factor 2 for K and V)

    fp16 layer 0 adds: num_tokens × num_kv_heads × head_dim × 2 × 2 bytes
    """
    # TQ3 compressed layers
    bits_per_token = 2 * num_kv_heads * (head_dim * bits / 8 + 4)  # K+V with norms
    tq_bytes = num_tokens * num_layers * bits_per_token

    # fp16 layer 0
    layer0_bytes = 0
    if layer_0_fp16:
        layer0_bytes = num_tokens * num_kv_heads * head_dim * 2 * 2  # fp16 K+V

    return (tq_bytes + layer0_bytes) / 1e9


def log_budget(label: str = "", **kwargs) -> MemoryBudget:
    """Compute and log current budget with optional label."""
    budget = compute_budget(**kwargs)
    logger.info("[%s] %s", label or "memory", budget.describe())
    return budget
