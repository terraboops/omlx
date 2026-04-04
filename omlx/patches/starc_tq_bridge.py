# SPDX-License-Identifier: Apache-2.0
"""STARC + TurboQuant integration — sparse decode over compressed KV.

The bridge between STARC (cluster-based sparse attention) and TurboQuant
(3-bit compressed KV). At long context:
  - TQ3 decode scans ALL 65K+ compressed entries per token → ~5 tok/s
  - STARC selects 15% of tokens via K-means → ~10K effective context
  - Combined: dequantize only the 10K selected → ~25-40 tok/s target

Integration pattern:
  1. Prefill end: dequantize compressed KV → cluster with K-means
  2. Store centroids + cluster assignments (tiny, ~1MB per layer)
  3. Decode: query × centroids → top-k clusters → gather indices
  4. Dequantize ONLY subset → standard Flash Attention → fast output
"""

from __future__ import annotations

import logging
from typing import List, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def extract_keys_from_tq_cache(cache) -> Optional[mx.array]:
    """Dequantize TQ cache to extract fp16 keys for K-means clustering.

    This is a one-time cost at the end of prefill, O(context × D).
    Returns: (H_kv, T, D) or None if cache empty / not quantized.
    """
    if not hasattr(cache, '_quantized') or not cache._quantized:
        return None
    if cache._k_norms is None or cache.offset == 0:
        return None

    T = cache.offset
    # Dequantize entire compressed history (once)
    keys_fp16 = cache._codec.dequantize(
        cache._k_norms[:, :, :T],
        cache._k_packed[:, :, :T],
    )
    # Shape: (B, H_kv, T, D) → drop batch → (H_kv, T, D)
    return keys_fp16[0]


def initialize_starc_for_tq_caches(starc_manager, cache_list):
    """Run initial K-means clustering using dequantized TQ caches.

    Args:
        starc_manager: omlx.patches.starc_attention.StarcManager
        cache_list: List of cache objects (may be mix of KVCache and TQ)
    """
    from omlx.turboquant_kv import TurboQuantKVCache
    from omlx.patches.starc_attention import _kmeans_cosine, _build_csr_index

    for layer_idx, cache_obj in enumerate(cache_list):
        starc_manager._cache_id_to_layer[id(cache_obj)] = layer_idx

        # Extract keys — different path for TQ vs standard cache
        if isinstance(cache_obj, TurboQuantKVCache):
            keys = extract_keys_from_tq_cache(cache_obj)
        else:
            keys = starc_manager._extract_keys(cache_obj)

        if keys is None:
            logger.debug("STARC: layer %d has no keys, skipping", layer_idx)
            continue

        T = keys.shape[1]
        if T < starc_manager.min_seq_len:
            logger.debug("STARC: layer %d seq_len=%d < min %d",
                         layer_idx, T, starc_manager.min_seq_len)
            continue

        state = starc_manager.layers[layer_idx]
        n_clusters = max(1, int(T * state.cluster_ratio))

        # Run K-means on dequantized keys
        state.centroids, state.assignments = _kmeans_cosine(
            keys, n_clusters, max_iters=15
        )
        # Build CSR index for fast subset gathering
        (state.cluster_sizes, state.sorted_indices, state.cluster_offsets) = \
            _build_csr_index(state.assignments, n_clusters)

        state.num_tokens_clustered = T
        state.decode_steps_since_recluster = 0
        # Free the temporary dequantized keys
        del keys

    mx.synchronize()
    mx.clear_cache()
    logger.info("STARC: clustered %d layers", sum(1 for s in starc_manager.layers if s.initialized))


def starc_tq_decode_attention(
    cache,              # TurboQuantKVCache
    starc_manager,      # StarcManager
    layer_idx: int,
    queries: mx.array,  # (B, H_q, 1, D)
    scale: float,
    mask=None,
) -> mx.array:
    """Decode attention using STARC subset selection + TQ dequantize.

    Replaces the full-context decode_attention with:
      1. STARC picks ~15% of tokens via cluster selection
      2. Dequantize only those tokens from compressed storage
      3. Run standard Flash Attention on the subset

    This is O(budget) instead of O(context) per decode step.
    """
    state = starc_manager.layers[layer_idx]
    if not state.initialized or cache.offset < starc_manager.min_seq_len:
        # Fall back to full decode
        return cache.decode_attention(
            queries,
            keys_state=(cache._k_norms[:, :, :cache.offset],
                        cache._k_packed[:, :, :cache.offset]),
            values_state=(cache._v_norms[:, :, :cache.offset],
                          cache._v_packed[:, :, :cache.offset]),
            scale=scale, mask=mask,
        )

    # STARC subset selection
    subset = starc_manager.select_subset(layer_idx, queries, cache.offset)
    if subset is None or subset.shape[0] == 0:
        # STARC unavailable — fall back
        return cache.decode_attention(
            queries,
            keys_state=(cache._k_norms[:, :, :cache.offset],
                        cache._k_packed[:, :, :cache.offset]),
            values_state=(cache._v_norms[:, :, :cache.offset],
                          cache._v_packed[:, :, :cache.offset]),
            scale=scale, mask=mask,
        )

    # Gather compressed KV at subset indices
    # subset: (budget,) int32 — sorted, deduplicated
    k_norms_sub = cache._k_norms[:, :, subset]  # (B, H_kv, budget)
    k_packed_sub = cache._k_packed[:, :, subset]  # (B, H_kv, budget, pw)
    v_norms_sub = cache._v_norms[:, :, subset]
    v_packed_sub = cache._v_packed[:, :, subset]

    # Dequantize only the subset — much smaller than full context
    k_sub = cache._codec.dequantize(k_norms_sub, k_packed_sub)  # (B, H_kv, budget, D)
    v_sub = cache._codec.dequantize(v_norms_sub, v_packed_sub)

    # GQA expansion for standard SDPA
    B, H_q, L, D = queries.shape
    H_kv = k_sub.shape[1]
    if H_q > H_kv:
        n_groups = H_q // H_kv
        k_sub = mx.repeat(k_sub, n_groups, axis=1)
        v_sub = mx.repeat(v_sub, n_groups, axis=1)

    # Standard Flash Attention on the subset (fast!)
    output = mx.fast.scaled_dot_product_attention(
        queries, k_sub.astype(queries.dtype), v_sub.astype(queries.dtype),
        scale=scale, mask=None,  # No mask — STARC only selects past tokens
    )
    return output
