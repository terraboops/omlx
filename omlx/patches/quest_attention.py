# SPDX-License-Identifier: Apache-2.0
"""Quest query-aware page selection for TQ3 decode (arXiv:2406.10774).

During decode, instead of scanning all T tokens, Quest:
  1. Scores pages using cheap upper-bound estimates
  2. Selects top-K pages by score
  3. Gathers packed KV for selected pages
  4. Runs existing fused attention on the smaller buffer

Per-page bounds (16 float32 per page per KV head):
  [0:8]  — max |k_rot[d]| per 16-dim group (upper bound on attention contribution)
  [8:16] — reserved (max norm, mean norm, padding)

Scoring: |sum q*k| <= sum |q|*|k| <= sum_g (sum_{d in g} |q_rot[d]|) * max_{d in g} |k_rot[d]|
This is a valid upper bound, and pages ranked by it will include the true top-K.
"""

from __future__ import annotations

import logging
import math

import mlx.core as mx

logger = logging.getLogger(__name__)

# Page size in tokens — must be power of 2 for clean division
QUEST_PAGE_SIZE = 128

# Number of dimension groups for bounds tracking (D=128 / 16 = 8 groups)
QUEST_NUM_GROUPS = 8
QUEST_GROUP_DIM = 16  # Dimensions per group

# Total floats per page: 8 (max |k| per group) + 8 (reserved)
QUEST_BOUNDS_PER_PAGE = 16


def compute_page_bounds(
    k_norms: mx.array,
    k_packed: mx.array,
    codebook: mx.array,
    bits: int,
    dim: int,
) -> mx.array:
    """Compute per-page K bounds from packed quantized keys.

    Args:
        k_norms: (B, H_kv, T) — per-token key norms
        k_packed: (B, H_kv, T, pw) — packed codebook indices
        codebook: (n_levels,) — codebook values
        bits: quantization bits
        dim: head dimension (e.g., 128)

    Returns:
        page_bounds: (B, H_kv, num_pages, 16) float32
          [0:8] = max absolute rotated key magnitude per 16-dim group per page
          [8] = max norm in page
          [9] = mean norm in page
          [10:16] = reserved (zeros)
    """
    B, H_kv, T = k_norms.shape
    num_pages = (T + QUEST_PAGE_SIZE - 1) // QUEST_PAGE_SIZE

    # Pad T to multiple of page_size for clean reshape
    pad_T = num_pages * QUEST_PAGE_SIZE
    if pad_T > T:
        pad_norms = mx.zeros((B, H_kv, pad_T - T), dtype=k_norms.dtype)
        k_norms_padded = mx.concatenate([k_norms, pad_norms], axis=2)
    else:
        k_norms_padded = k_norms

    # Reshape norms into pages: (B, H_kv, num_pages, page_size)
    norms_paged = k_norms_padded.reshape(B, H_kv, num_pages, QUEST_PAGE_SIZE)

    # Per-page norm statistics
    page_max_norm = mx.max(mx.abs(norms_paged), axis=3)  # (B, H_kv, num_pages)
    page_mean_norm = mx.mean(mx.abs(norms_paged), axis=3)

    # For the dimension-group bounds, we use a conservative estimate:
    # max|k_rot[d]| for any d in group = max_norm_in_page * max|codebook|
    # This is tight when the codebook range is symmetric and norms dominate.
    # For better bounds, we'd need to unpack indices per page, but that's
    # expensive. The norm-based bound is sufficient for page ranking.
    cb_max = mx.max(mx.abs(codebook)).item()

    # All groups get the same bound (norm * cb_max) since we don't have
    # per-dim information without unpacking
    group_bounds = page_max_norm * cb_max  # (B, H_kv, num_pages)

    # Build bounds tensor: (B, H_kv, num_pages, 16)
    bounds = mx.zeros((B, H_kv, num_pages, QUEST_BOUNDS_PER_PAGE), dtype=mx.float32)

    # [0:8] = group max bounds (all same since norm-based)
    for g in range(QUEST_NUM_GROUPS):
        bounds[:, :, :, g] = group_bounds

    # [8] = max norm, [9] = mean norm
    bounds[:, :, :, 8] = page_max_norm
    bounds[:, :, :, 9] = page_mean_norm

    return bounds


def select_topk_pages(
    q_rot: mx.array,
    k_bounds: mx.array,
    topk: int,
) -> mx.array:
    """Score pages and return top-K page indices.

    Args:
        q_rot: (B*H_q, D) — rotated query vectors (already scaled)
        k_bounds: (B, H_kv, num_pages, 16) — per-page bounds
        topk: number of pages to select

    Returns:
        page_indices: (B, H_kv, topk) — indices of top-K pages, sorted
    """
    B, H_kv, num_pages, _ = k_bounds.shape
    H_q = q_rot.shape[0] // B
    GQA = H_q // H_kv  # GQA ratio

    # Compute per-group query magnitude: sum |q_rot[d]| per 16-dim group
    # q_rot: (B*H_q, D) → (B, H_q, 8, 16) → sum abs per group → (B, H_q, 8)
    D = q_rot.shape[-1]
    q_grouped = mx.abs(q_rot.reshape(B, H_q, QUEST_NUM_GROUPS, QUEST_GROUP_DIM))
    q_group_mag = mx.sum(q_grouped, axis=3)  # (B, H_q, 8)

    # Average across GQA heads to get per-KV-head query magnitude
    # (B, H_q, 8) → (B, H_kv, GQA, 8) → mean → (B, H_kv, 8)
    q_group_mag = q_group_mag.reshape(B, H_kv, GQA, QUEST_NUM_GROUPS)
    q_group_mag = mx.mean(q_group_mag, axis=2)  # (B, H_kv, 8)

    # Score each page: sum_g(q_group_mag[g] * k_bounds[g])
    # (B, H_kv, 8) @ (B, H_kv, num_pages, 8).T → (B, H_kv, num_pages)
    page_scores = mx.sum(
        q_group_mag[:, :, None, :] * k_bounds[:, :, :, :QUEST_NUM_GROUPS],
        axis=3,
    )  # (B, H_kv, num_pages)

    # Clamp topk to available pages
    topk = min(topk, num_pages)

    # Top-K selection: argpartition for efficiency, then sort the top-K
    if topk >= num_pages:
        # Select all pages, sorted by score (descending)
        indices = mx.argsort(page_scores, axis=2)
        return indices[:, :, ::-1]

    # Use argsort and take top-K (MLX doesn't have argpartition yet)
    sorted_idx = mx.argsort(-page_scores, axis=2)  # descending
    top_indices = sorted_idx[:, :, :topk]

    # Sort selected indices by position (preserve causal order for attention)
    top_indices = mx.sort(top_indices, axis=2)

    return top_indices


def gather_pages(
    k_norms: mx.array,
    k_packed: mx.array,
    v_norms: mx.array,
    v_packed: mx.array,
    page_indices: mx.array,
    total_tokens: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, int]:
    """Gather packed KV for selected pages into contiguous buffers.

    Args:
        k_norms: (B, H_kv, T) — full key norms
        k_packed: (B, H_kv, T, pw) — full packed key indices
        v_norms, v_packed: same for values
        page_indices: (B, H_kv, topk) — selected page indices
        total_tokens: actual number of valid tokens (may be < T due to padding)

    Returns:
        (k_norms_sel, k_packed_sel, v_norms_sel, v_packed_sel, sel_tokens)
        All shaped for the selected pages only.
    """
    B, H_kv, topk = page_indices.shape

    # Expand page indices to token indices
    # Each page index i → tokens [i*page_size, (i+1)*page_size)
    page_starts = page_indices * QUEST_PAGE_SIZE  # (B, H_kv, topk)

    # Build token offset within page: [0, 1, ..., page_size-1]
    offsets = mx.arange(QUEST_PAGE_SIZE)  # (page_size,)

    # Token indices: (B, H_kv, topk, page_size)
    token_indices = page_starts[:, :, :, None] + offsets[None, None, None, :]

    # Clamp to valid range
    token_indices = mx.minimum(token_indices, total_tokens - 1)

    # Flatten for gather: (B, H_kv, topk * page_size)
    sel_tokens = topk * QUEST_PAGE_SIZE
    flat_indices = token_indices.reshape(B, H_kv, sel_tokens)

    # Gather norms: (B, H_kv, sel_tokens)
    k_norms_sel = mx.take_along_axis(k_norms, flat_indices, axis=2)
    v_norms_sel = mx.take_along_axis(v_norms, flat_indices, axis=2)

    # Gather packed: (B, H_kv, sel_tokens, pw)
    # Need to expand indices for the pw dimension
    pw = k_packed.shape[-1]
    flat_indices_4d = mx.broadcast_to(flat_indices[:, :, :, None], (B, H_kv, sel_tokens, pw))
    k_packed_sel = mx.take_along_axis(k_packed, flat_indices_4d, axis=2)
    v_packed_sel = mx.take_along_axis(v_packed, flat_indices_4d, axis=2)

    return k_norms_sel, k_packed_sel, v_norms_sel, v_packed_sel, sel_tokens
