# SPDX-License-Identifier: Apache-2.0
"""STARC: Clustered sparsity attention for long-context decode.

Based on STARC (ASPLOS 2026, arXiv 2505.05772).
Clusters KV pairs by semantic similarity, selects top clusters per query
during decode to reduce attention from O(T) to O(B) where B << T.

Composes with TurboQuant: STARC selects token subset, then standard SDPA
computes attention on the smaller tensor.

Algorithm:
  1. Prefill end: K-means cluster keys (cosine distance, K-means++ init)
  2. Decode: query × centroids → top-k clusters → gather KV subset → SDPA
  3. Recluster every 128 decode steps
  4. Budget: ~15% of context (configurable)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_PATCHED = False


# ---------------------------------------------------------------------------
# K-Means Clustering (cosine similarity)
# ---------------------------------------------------------------------------


def _kmeans_init_pp(
    keys: mx.array, n_clusters: int, seed: int = 42
) -> mx.array:
    """K-means++ initialization using numpy (one-time cost).

    Args:
        keys: (H_kv, T, D) key vectors
        n_clusters: K

    Returns:
        centroids: (H_kv, K, D) initial centroids (L2-normalized)
    """
    H, T, D = keys.shape
    keys_np = np.array(keys)

    # L2-normalize
    norms = np.linalg.norm(keys_np, axis=-1, keepdims=True)
    keys_norm = keys_np / np.maximum(norms, 1e-10)

    rng = np.random.default_rng(seed)
    centroids = np.zeros((H, n_clusters, D), dtype=np.float32)

    for h in range(H):
        # First centroid: random
        idx = rng.integers(0, T)
        centroids[h, 0] = keys_norm[h, idx]

        for k in range(1, n_clusters):
            # Distances to nearest centroid
            sims = keys_norm[h] @ centroids[h, :k].T  # (T, k)
            max_sim = sims.max(axis=-1)  # (T,)
            dist = 1.0 - max_sim  # cosine distance
            dist = np.maximum(dist, 0.0)

            # Sample proportional to distance squared
            probs = dist ** 2
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = np.ones(T) / T

            idx = rng.choice(T, p=probs)
            centroids[h, k] = keys_norm[h, idx]

    return mx.array(centroids)


def _kmeans_cosine(
    keys: mx.array,
    n_clusters: int,
    n_iters: int = 15,
    seed: int = 42,
) -> Tuple[mx.array, mx.array]:
    """K-means with cosine distance and K-means++ init.

    Args:
        keys: (H_kv, T, D) key vectors
        n_clusters: K
        n_iters: Lloyd iterations
        seed: for initialization

    Returns:
        centroids: (H_kv, K, D) L2-normalized cluster centroids
        assignments: (H_kv, T) cluster IDs
    """
    H, T, D = keys.shape

    # Clamp n_clusters to available tokens
    n_clusters = min(n_clusters, T)
    if n_clusters <= 1:
        centroids = mx.mean(keys, axis=1, keepdims=True)  # (H, 1, D)
        assignments = mx.zeros((H, T), dtype=mx.int32)
        return centroids, assignments

    # K-means++ init
    centroids = _kmeans_init_pp(keys, n_clusters, seed)

    # L2-normalize keys
    keys_norm = keys / mx.maximum(mx.linalg.norm(keys, axis=-1, keepdims=True), 1e-10)

    for _ in range(n_iters):
        # Normalize centroids
        centroids = centroids / mx.maximum(
            mx.linalg.norm(centroids, axis=-1, keepdims=True), 1e-10
        )

        # Assign: cosine similarity = keys_norm @ centroids^T
        sims = keys_norm @ centroids.swapaxes(-1, -2)  # (H, T, K)
        assignments = mx.argmax(sims, axis=-1)  # (H, T)

        # Update centroids: one-hot matmul approach
        one_hot = (mx.expand_dims(assignments, -1) == mx.arange(n_clusters)).astype(mx.float32)
        # new_centroids = one_hot^T @ keys_norm → (H, K, D)
        centroids = one_hot.swapaxes(-1, -2) @ keys_norm

        # Handle empty clusters by re-using farthest point
        cluster_sizes = one_hot.sum(axis=1)  # (H, K)
        # Normalize non-empty centroids
        centroids = centroids / mx.maximum(
            mx.linalg.norm(centroids, axis=-1, keepdims=True), 1e-10
        )

    mx.eval(centroids, assignments)
    return centroids, assignments


def _build_csr_index(
    assignments: mx.array, n_clusters: int
) -> Tuple[mx.array, mx.array, mx.array]:
    """Build CSR-style index from cluster assignments.

    Args:
        assignments: (H_kv, T) cluster IDs
        n_clusters: K

    Returns:
        sorted_indices: (H_kv, T) token indices sorted by cluster
        cluster_offsets: (H_kv, K+1) CSR offsets
        cluster_sizes: (H_kv, K) tokens per cluster
    """
    H, T = assignments.shape

    # Compute cluster sizes
    one_hot = (mx.expand_dims(assignments, -1) == mx.arange(n_clusters)).astype(mx.float32)
    cluster_sizes = one_hot.sum(axis=1).astype(mx.int32)  # (H, K)

    # Sort token indices by cluster assignment
    sorted_indices = mx.argsort(assignments, axis=-1)  # (H, T)

    # Compute CSR offsets via cumsum of sizes
    cluster_offsets = mx.concatenate(
        [mx.zeros((H, 1), dtype=mx.int32), mx.cumsum(cluster_sizes, axis=-1)],
        axis=-1,
    )  # (H, K+1)

    mx.eval(sorted_indices, cluster_offsets, cluster_sizes)
    return sorted_indices, cluster_offsets, cluster_sizes


def _select_clusters(
    query: mx.array,
    centroids: mx.array,
    cluster_sizes: mx.array,
    sorted_indices: mx.array,
    cluster_offsets: mx.array,
    budget: int,
) -> mx.array:
    """Select top clusters until budget is reached.

    Args:
        query: (H_kv, 1, D) query vector (averaged over GQA group)
        centroids: (H_kv, K, D) cluster centroids
        cluster_sizes: (H_kv, K) tokens per cluster
        sorted_indices: (H_kv, T) sorted token indices
        cluster_offsets: (H_kv, K+1) CSR offsets
        budget: maximum tokens to select

    Returns:
        selected_indices: (budget,) token indices (union across heads,
            padded with -1 if fewer than budget tokens selected)
    """
    H, K, D = centroids.shape

    # Score clusters: query @ centroids^T → (H, 1, K) → (H, K)
    scores = (query @ centroids.swapaxes(-1, -2)).squeeze(1)  # (H, K)

    # Sort clusters by score descending (per head)
    sorted_cluster_ids = mx.argsort(-scores, axis=-1)  # (H, K)

    # Greedy selection: accumulate clusters until budget reached
    # Process on CPU/numpy for the greedy loop (K is small, ~64-256)
    sorted_cluster_ids_np = np.array(sorted_cluster_ids)
    cluster_sizes_np = np.array(cluster_sizes)
    sorted_indices_np = np.array(sorted_indices)
    cluster_offsets_np = np.array(cluster_offsets)

    all_selected = set()
    for h in range(H):
        tokens_selected = 0
        for rank in range(K):
            c = sorted_cluster_ids_np[h, rank]
            c_size = cluster_sizes_np[h, c]
            if tokens_selected + c_size > budget and tokens_selected > 0:
                # Partial: take what fits
                remaining = budget - tokens_selected
                start = cluster_offsets_np[h, c]
                for j in range(remaining):
                    all_selected.add(int(sorted_indices_np[h, start + j]))
                break
            # Full cluster
            start = cluster_offsets_np[h, c]
            end = cluster_offsets_np[h, c + 1]
            for j in range(start, end):
                all_selected.add(int(sorted_indices_np[h, j]))
            tokens_selected += c_size
            if tokens_selected >= budget:
                break

    # Convert to sorted array, pad to budget
    selected = sorted(all_selected)
    if len(selected) < budget:
        selected.extend([-1] * (budget - len(selected)))
    elif len(selected) > budget:
        selected = selected[:budget]

    return mx.array(selected, dtype=mx.int32)


# ---------------------------------------------------------------------------
# Cluster State
# ---------------------------------------------------------------------------


class StarcClusterState:
    """Per-layer clustering state."""

    def __init__(self, n_kv_heads: int, head_dim: int, cluster_ratio: float = 1 / 32):
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.cluster_ratio = cluster_ratio

        self.centroids: Optional[mx.array] = None  # (H_kv, K, D)
        self.assignments: Optional[mx.array] = None  # (H_kv, T)
        self.cluster_sizes: Optional[mx.array] = None  # (H_kv, K)
        self.sorted_indices: Optional[mx.array] = None  # (H_kv, T)
        self.cluster_offsets: Optional[mx.array] = None  # (H_kv, K+1)

        self.num_tokens_clustered: int = 0
        self.decode_steps_since_recluster: int = 0

    @property
    def initialized(self) -> bool:
        return self.centroids is not None

    @property
    def n_clusters(self) -> int:
        return self.centroids.shape[1] if self.centroids is not None else 0


class StarcManager:
    """Manages STARC clustering state across all layers."""

    def __init__(
        self,
        num_layers: int,
        n_kv_heads: int,
        head_dim: int,
        cluster_ratio: float = 1 / 32,
        budget_pct: float = 0.15,
        recluster_interval: int = 128,
        min_seq_len: int = 512,
    ):
        self.layers = [
            StarcClusterState(n_kv_heads, head_dim, cluster_ratio)
            for _ in range(num_layers)
        ]
        self.budget_pct = budget_pct
        self.recluster_interval = recluster_interval
        self.min_seq_len = min_seq_len
        self._cache_id_to_layer: Dict[int, int] = {}

    def initialize(self, cache_list: List[Any]) -> None:
        """Run initial K-means clustering after prefill.

        Args:
            cache_list: List of cache objects (one per layer).
        """
        for layer_idx, cache_obj in enumerate(cache_list):
            self._cache_id_to_layer[id(cache_obj)] = layer_idx

            # Extract keys from cache
            keys = self._extract_keys(cache_obj)
            if keys is None:
                continue

            T = keys.shape[1]
            if T < self.min_seq_len:
                continue

            state = self.layers[layer_idx]
            n_clusters = max(1, int(T * state.cluster_ratio))

            state.centroids, state.assignments = _kmeans_cosine(
                keys, n_clusters, n_iters=15
            )
            state.sorted_indices, state.cluster_offsets, state.cluster_sizes = (
                _build_csr_index(state.assignments, n_clusters)
            )
            state.num_tokens_clustered = T

        logger.info(
            f"STARC: initialized clustering for {len(cache_list)} layers"
        )

    def get_layer_idx(self, cache_obj: Any) -> Optional[int]:
        """Get layer index from cache object identity."""
        return self._cache_id_to_layer.get(id(cache_obj))

    def select_subset(
        self,
        layer_idx: int,
        queries: mx.array,
        seq_len: int,
    ) -> Optional[mx.array]:
        """Select token subset for decode attention.

        Args:
            layer_idx: Which layer.
            queries: (B, H_q, 1, D) query vectors.
            seq_len: Current sequence length.

        Returns:
            Selected token indices (budget,) or None if STARC inactive.
        """
        state = self.layers[layer_idx]
        if not state.initialized or seq_len < self.min_seq_len:
            return None

        budget = max(1, int(seq_len * self.budget_pct))

        # Average query across GQA group → (H_kv, 1, D)
        H_q = queries.shape[1]
        H_kv = state.n_kv_heads
        gqa = H_q // H_kv
        # queries is (B, H_q, 1, D) → take first batch, reshape for GQA
        q = queries[0]  # (H_q, 1, D)
        q = q.reshape(H_kv, gqa, 1, -1).mean(axis=1)  # (H_kv, 1, D)

        indices = _select_clusters(
            q,
            state.centroids,
            state.cluster_sizes,
            state.sorted_indices,
            state.cluster_offsets,
            budget,
        )

        # Filter out padding (-1)
        valid = indices[indices >= 0]

        state.decode_steps_since_recluster += 1
        return valid

    def _extract_keys(self, cache_obj: Any) -> Optional[mx.array]:
        """Extract fp16 key vectors from a cache object.

        Returns: (H_kv, T, D) or None if not extractable.
        """
        # Standard KVCache stores keys as (B, H, T, D)
        if hasattr(cache_obj, "keys") and cache_obj.keys is not None:
            k = cache_obj.keys
            if k.ndim == 4:
                return k[0]  # Drop batch dim → (H, T, D)
        # BatchKVCache may store differently
        if hasattr(cache_obj, "_keys") and cache_obj._keys is not None:
            k = cache_obj._keys
            if k.ndim == 4:
                return k[0]
        return None


# ---------------------------------------------------------------------------
# SDPA Patch
# ---------------------------------------------------------------------------

# Global manager reference (set by apply_starc_attention_patch)
_manager: Optional[StarcManager] = None


def apply_starc_attention_patch(
    budget_pct: float = 0.15,
    cluster_ratio: float = 1 / 32,
    recluster_interval: int = 128,
    min_seq_len: int = 512,
) -> bool:
    """Monkey-patch SDPA to add STARC cluster selection before attention.

    This wraps whatever SDPA is currently installed (possibly TurboQuant-patched).
    During decode (L=1), STARC selects a subset of KV tokens before calling SDPA.

    Returns True if patch was applied.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import mlx_lm.models.base as mlx_base
    except ImportError:
        logger.warning("STARC: mlx_lm.models.base not available")
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def _starc_sdpa(queries, keys, values, cache=None, scale=None, mask=None, **kwargs):
        global _manager

        # Only intercept decode (L=1) when manager exists
        if _manager is not None and queries.shape[-2] == 1 and cache is not None:
            layer_idx = _manager.get_layer_idx(cache)
            if layer_idx is not None:
                seq_len = keys.shape[-2]
                subset = _manager.select_subset(layer_idx, queries, seq_len)
                if subset is not None and subset.shape[0] > 0:
                    # Gather KV subset
                    keys = keys[:, :, subset]
                    values = values[:, :, subset]
                    mask = None  # No causal mask needed for subset

        return original_sdpa(queries, keys, values, cache=cache, scale=scale, mask=mask, **kwargs)

    mlx_base.scaled_dot_product_attention = _starc_sdpa
    _PATCHED = True
    logger.info("STARC: attention patch applied")
    return True
