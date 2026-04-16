# SPDX-License-Identifier: Apache-2.0
"""SnapKV attention-guided token selection for KV cache eviction.

Derived from SnapKV (arXiv:2404.14469). Selects the most important
tokens to keep in the KV cache based on attention patterns from an
observation window at the end of prefill.

Algorithm:
  1. Take the last `obs_window` tokens' attention weights
  2. For each KV head, pool attention across the observation window
     (max-over-query-positions, then average-across-heads-in-GQA-group)
  3. Select top-K tokens per head by pooled importance score
  4. Return keep indices (union across heads)

The keep indices are used by the cache's compact() method to evict
low-importance tokens while preserving exact values for kept tokens.

Usage:
    from omlx.patches.snapkv import snapkv_select

    # After prefill, before decode:
    keep_indices = snapkv_select(
        model, cache, tokenizer, input_ids,
        obs_window=64, keep_ratio=0.5,
    )
    # Then: cache.compact(keep_indices)
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def compute_attention_importance(
    query: mx.array,
    key: mx.array,
    scale: float,
    obs_window: int = 64,
) -> mx.array:
    """Compute per-token importance scores from attention weights.

    Uses the last `obs_window` query positions to score all key positions.
    Pools via max-over-query-positions to capture any position where
    a token was highly attended to.

    Args:
        query: (B, H_q, T, D) — query projections
        key:   (B, H_kv, T, D) — key projections
        scale: attention scale (1/sqrt(d))
        obs_window: number of trailing query positions to use

    Returns:
        importance: (B, H_kv, T) — per-token importance score per KV head
    """
    B, H_q, T, D = query.shape
    H_kv = key.shape[1]
    gqa_ratio = H_q // H_kv  # GQA group size

    # Use only the observation window queries
    obs_start = max(0, T - obs_window)
    Q_obs = query[:, :, obs_start:, :]  # (B, H_q, obs_len, D)

    # Compute attention scores: Q_obs @ K^T
    # For GQA: expand K to match Q heads
    if gqa_ratio > 1:
        K_expanded = mx.repeat(key, gqa_ratio, axis=1)  # (B, H_q, T, D)
    else:
        K_expanded = key

    # Scores: (B, H_q, obs_len, T)
    scores = (Q_obs @ K_expanded.swapaxes(-1, -2)) * scale

    # Causal mask: observation window queries can only attend to positions <= their index
    obs_len = Q_obs.shape[2]
    # Position indices for obs queries: [obs_start, obs_start+1, ..., T-1]
    # Position indices for keys: [0, 1, ..., T-1]
    # Mask: key_pos <= query_pos
    q_pos = mx.arange(obs_start, T).reshape(1, 1, obs_len, 1)
    k_pos = mx.arange(T).reshape(1, 1, 1, T)
    causal_mask = k_pos <= q_pos  # (1, 1, obs_len, T)

    scores = mx.where(causal_mask, scores, mx.array(float('-inf')))

    # Softmax attention weights
    weights = mx.softmax(scores, axis=-1)  # (B, H_q, obs_len, T)

    # Pool across observation window: max over query positions
    # This captures tokens that ANY observation query attended to highly
    max_weights = mx.max(weights, axis=2)  # (B, H_q, T)

    # Average across GQA group to get per-KV-head importance
    if gqa_ratio > 1:
        # Reshape: (B, H_kv, gqa_ratio, T) → mean over gqa_ratio
        max_weights = max_weights.reshape(B, H_kv, gqa_ratio, T)
        importance = mx.mean(max_weights, axis=2)  # (B, H_kv, T)
    else:
        importance = max_weights

    return importance


def snapkv_select(
    importance: mx.array,
    keep_count: int,
    always_keep_last: int = 64,
) -> mx.array:
    """Select top-K tokens to keep based on importance scores.

    Args:
        importance: (B, H_kv, T) — per-token importance per head
        keep_count: total tokens to keep (including always_keep_last)
        always_keep_last: always keep this many recent tokens (sink/window)

    Returns:
        keep_mask: (B, T) — boolean mask of tokens to keep (union across heads)
    """
    B, H_kv, T = importance.shape

    if keep_count >= T:
        return mx.ones((B, T), dtype=mx.bool_)

    # Always keep the last `always_keep_last` tokens
    selectable = max(0, T - always_keep_last)
    k_selectable = max(1, keep_count - always_keep_last)

    if selectable == 0:
        return mx.ones((B, T), dtype=mx.bool_)

    # Pool importance across heads (union strategy: max across heads)
    pooled = mx.max(importance[:, :, :selectable], axis=1)  # (B, selectable)

    # Top-k selection
    top_k_indices = mx.argpartition(-pooled, kth=k_selectable, axis=-1)[:, :k_selectable]

    # Build keep indices (sorted for cache compaction)
    # Union top-k across batch (B=1 for inference)
    all_indices = set()
    top_k_np = top_k_indices.tolist() if hasattr(top_k_indices, 'tolist') else [[]]
    for b_indices in top_k_np:
        if isinstance(b_indices, list):
            all_indices.update(b_indices)
        else:
            all_indices.add(int(b_indices))

    # Add always-keep-last positions
    for pos in range(selectable, T):
        all_indices.add(pos)

    # Build boolean mask
    keep_mask = mx.zeros((B, T), dtype=mx.bool_)
    sorted_indices = sorted(all_indices)
    for pos in sorted_indices:
        keep_mask = keep_mask.at[:, pos].add(mx.ones((B,), dtype=mx.bool_))

    return keep_mask


def get_keep_indices(keep_mask: mx.array) -> list[int]:
    """Extract sorted keep indices from a mask (for cache compaction)."""
    mask_np = keep_mask[0].tolist() if keep_mask.shape[0] > 0 else []
    return [i for i, v in enumerate(mask_np) if v]


def compact_cache(cache: list, keep_indices: list[int]) -> None:
    """Compact KV cache in-place, keeping only selected token positions.

    This is the core operation for SnapKV: after computing importance
    and selecting which tokens to keep, rewrite the cache to contain
    only those positions. The kept tokens retain their exact values
    (zero approximation error).

    Args:
        cache: List of KVCache objects (one per layer)
        keep_indices: Sorted list of token positions to keep
    """
    idx = mx.array(keep_indices)
    for c in cache:
        keys = c.state[0]    # (B, H_kv, T, D)
        values = c.state[1]  # (B, H_kv, T, D)

        # Gather selected positions: (B, H_kv, len(idx), D)
        keys_compact = keys[:, :, idx, :]
        values_compact = values[:, :, idx, :]

        c.state = (keys_compact, values_compact)

    mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])


def count_kept(keep_mask: mx.array) -> int:
    """Count number of kept tokens."""
    return int(mx.sum(keep_mask).item())
