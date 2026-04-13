# SPDX-License-Identifier: Apache-2.0
"""MInference sparse prefill attention dispatch (arXiv:2407.02490).

Reads the per-head pattern table from Task 4 calibration and, during
prefill, applies per-head sparse masks instead of full O(n²) attention:
  - a_shape:        sink tokens (first 4) + causal band
  - vertical_slash: top-importance columns + causal band
  - block_sparse:   block-diagonal with overlap
  - dense:          standard full attention (no optimization)

Applied via monkey-patching scaled_dot_product_attention.
Composes with prefill_last_logit_patch (downstream, orthogonal) and
vertical_eval (forces mx.eval per layer, helps prevent graph bloat).

Flag: --prefill-sparse minference on hypercar_server. Default off.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False
_PATTERN_TABLE: Optional[dict] = None
_LAYER_COUNTER = [0]  # Tracks which layer is being computed

# Default pattern table path
DEFAULT_PATTERN_DIR = Path(__file__).parent / "minference_patterns"


def load_pattern_table(model_name: str = "qwen3_coder_30b_a3b_instruct_8bit") -> dict:
    """Load the calibration pattern table for a model.

    Args:
        model_name: filename stem (without .json) in minference_patterns/

    Returns:
        Pattern table dict with 'heads' list and 'num_layers'/'num_heads'.
    """
    path = DEFAULT_PATTERN_DIR / f"{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"MInference pattern table not found: {path}\n"
            f"Run: python scripts/minference_calibrate.py --model <your-model>"
        )

    table = json.loads(path.read_text())

    # Build lookup dict: (layer, head) → pattern entry
    lookup = {}
    for entry in table["heads"]:
        lookup[(entry["layer"], entry["head"])] = entry
    table["_lookup"] = lookup

    logger.info(
        f"Loaded MInference patterns: {table['num_layers']} layers × "
        f"{table['num_heads']} heads, "
        f"distribution: {table['summary']['pattern_counts']}"
    )
    return table


# ---------------------------------------------------------------------------
# Sparse mask builders
# ---------------------------------------------------------------------------

def _build_a_shape_mask(L: int, params: dict) -> mx.array:
    """Build A-shape sparse mask: sink columns + causal band.

    Returns (L, L) bool mask where True = attend.
    """
    num_sink = params.get("num_sink", 4)
    band_width = params.get("band_width", 64)

    # Start with causal mask
    rows = mx.arange(L)[:, None]  # (L, 1)
    cols = mx.arange(L)[None, :]  # (1, L)

    # Causal: cols <= rows
    causal = cols <= rows

    # Sink columns: cols < num_sink
    sink = cols < num_sink

    # Band: rows - cols < band_width (local window)
    band = (rows - cols) < band_width

    mask = causal & (sink | band)
    return mask


def _build_vertical_slash_mask(
    L: int,
    params: dict,
    keys: Optional[mx.array] = None,
) -> mx.array:
    """Build vertical-slash sparse mask: important columns + causal band.

    If keys are provided, selects columns by key norm (dynamic).
    Otherwise uses the band_width + num_vertical_cols from calibration.
    """
    band_width = params.get("band_width", 64)
    num_vert = params.get("num_vertical_cols", 16)

    rows = mx.arange(L)[:, None]
    cols = mx.arange(L)[None, :]

    causal = cols <= rows
    band = (rows - cols) < band_width

    if keys is not None and num_vert > 0:
        # Dynamic: pick columns by key norm (proxy for importance)
        # keys shape: (1, 1, L, D) for a single head slice
        k_norms = mx.linalg.norm(keys[0, 0], axis=-1)  # (L,)
        # Top-K column indices by norm
        top_indices = mx.argsort(-k_norms)[:num_vert]
        # Build vertical column mask
        vert_mask = mx.zeros((L,), dtype=mx.bool_)
        vert_mask = vert_mask.at[top_indices].add(mx.ones((num_vert,), dtype=mx.bool_))
        vert = vert_mask[None, :]  # (1, L) — broadcast to (L, L)
    else:
        # Static: use first num_vert positions as vertical columns
        vert = cols < num_vert

    mask = causal & (band | vert)
    return mask


def _build_block_sparse_mask(L: int, params: dict) -> mx.array:
    """Build block-sparse mask: block-diagonal with overlap to previous block."""
    block_size = params.get("block_size", 64)

    rows = mx.arange(L)[:, None]
    cols = mx.arange(L)[None, :]

    causal = cols <= rows

    # Block assignments
    row_block = rows // block_size
    col_block = cols // block_size

    # Same block or previous block
    same_or_prev = (row_block == col_block) | (row_block == col_block + 1)

    mask = causal & same_or_prev
    return mask


# ---------------------------------------------------------------------------
# Sparse SDPA
# ---------------------------------------------------------------------------

def _sparse_sdpa_single_head(
    q: mx.array,       # (1, 1, L, D)
    k: mx.array,       # (1, 1, L, D)
    v: mx.array,       # (1, 1, L, D)
    scale: float,
    pattern: str,
    params: dict,
    causal_mask: Optional[mx.array] = None,
) -> mx.array:
    """Apply sparse attention for a single head.

    Returns (1, 1, L, D) attention output.
    """
    L = q.shape[2]

    if pattern == "dense" or L <= 128:
        # Dense: use standard SDPA (no sparsity gain at short seq)
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=causal_mask,
        )

    # Build sparse mask for this pattern
    if pattern == "a_shape":
        sparse_mask = _build_a_shape_mask(L, params)
    elif pattern == "vertical_slash":
        sparse_mask = _build_vertical_slash_mask(L, params, keys=k)
    elif pattern == "block_sparse":
        sparse_mask = _build_block_sparse_mask(L, params)
    else:
        # Unknown pattern — fall back to dense
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=causal_mask,
        )

    # Apply sparse mask as additive mask (-inf for masked positions)
    # sparse_mask: (L, L) bool, True = attend
    additive_mask = mx.where(sparse_mask, 0.0, -1e9).astype(q.dtype)
    additive_mask = additive_mask[None, None, :, :]  # (1, 1, L, L)

    # Combine with existing causal mask if present
    if causal_mask is not None:
        additive_mask = additive_mask + causal_mask

    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=scale, mask=additive_mask,
    )


def sparse_prefill_sdpa(
    queries: mx.array,   # (B, H_q, L, D)
    keys: mx.array,      # (B, H_kv, L, D)
    values: mx.array,    # (B, H_kv, L, D)
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """MInference sparse prefill: per-head pattern dispatch.

    For each query head, looks up its pattern from the calibration table
    and applies the corresponding sparse mask. Dense heads use standard SDPA.
    """
    global _LAYER_COUNTER

    if _PATTERN_TABLE is None:
        # No pattern table loaded — fall back to dense
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
        )

    B, H_q, L, D = queries.shape
    H_kv = keys.shape[1]
    GQA = H_q // H_kv
    num_layers = _PATTERN_TABLE["num_layers"]

    layer_idx = _LAYER_COUNTER[0] % num_layers
    _LAYER_COUNTER[0] += 1

    lookup = _PATTERN_TABLE["_lookup"]

    # Check if ALL heads in this layer are dense — fast path
    all_dense = all(
        lookup.get((layer_idx, h), {}).get("pattern", "dense") == "dense"
        for h in range(H_q)
    )
    if all_dense or L <= 128:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
        )

    # Group heads by pattern type for batched dispatch
    # This avoids per-head kernel launches when many heads share a pattern
    pattern_groups: dict[str, list[int]] = {}
    for h in range(H_q):
        entry = lookup.get((layer_idx, h))
        if entry is None:
            pattern = "dense"
        else:
            pattern = entry["pattern"]
        pattern_groups.setdefault(pattern, []).append(h)

    # If majority is dense, just do full dense (overhead of slicing > savings)
    dense_count = len(pattern_groups.get("dense", []))
    if dense_count > H_q * 0.7:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
        )

    # Build outputs per head
    outputs = [None] * H_q

    for pattern_type, head_indices in pattern_groups.items():
        if pattern_type == "dense":
            # Batch all dense heads together
            q_dense = queries[:, head_indices, :, :]
            kv_heads = [h // GQA for h in head_indices]
            k_dense = keys[:, kv_heads, :, :]
            v_dense = values[:, kv_heads, :, :]
            out = mx.fast.scaled_dot_product_attention(
                q_dense, k_dense, v_dense, scale=scale, mask=mask,
            )
            for i, h in enumerate(head_indices):
                outputs[h] = out[:, i:i+1, :, :]
        else:
            # Sparse patterns: need per-head masks
            # Get representative params from first head in group
            first_entry = lookup.get((layer_idx, head_indices[0]), {})
            params = first_entry.get("params", {})

            # Build mask once for this pattern (shared across heads with same params)
            if pattern_type == "a_shape":
                sparse_mask = _build_a_shape_mask(L, params)
            elif pattern_type == "vertical_slash":
                # For vertical_slash with dynamic column selection,
                # we use the first KV head's keys as a proxy
                kv_h = head_indices[0] // GQA
                sparse_mask = _build_vertical_slash_mask(
                    L, params, keys=keys[:, kv_h:kv_h+1, :, :],
                )
            elif pattern_type == "block_sparse":
                sparse_mask = _build_block_sparse_mask(L, params)
            else:
                sparse_mask = None

            if sparse_mask is not None:
                additive_mask = mx.where(sparse_mask, 0.0, -1e9).astype(queries.dtype)
                additive_mask = additive_mask[None, None, :, :]
                if mask is not None:
                    combined_mask = additive_mask + mask
                else:
                    combined_mask = additive_mask
            else:
                combined_mask = mask

            # Batch these heads
            q_sparse = queries[:, head_indices, :, :]
            kv_heads = [h // GQA for h in head_indices]
            k_sparse = keys[:, kv_heads, :, :]
            v_sparse = values[:, kv_heads, :, :]

            # Broadcast mask across heads in this group
            out = mx.fast.scaled_dot_product_attention(
                q_sparse, k_sparse, v_sparse,
                scale=scale, mask=combined_mask,
            )
            for i, h in enumerate(head_indices):
                outputs[h] = out[:, i:i+1, :, :]

    # Concatenate all heads
    return mx.concatenate(outputs, axis=1)


# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

def apply_minference_prefill_patch(
    model_name: str = "qwen3_coder_30b_a3b_instruct_8bit",
) -> bool:
    """Monkey-patch SDPA for MInference sparse prefill.

    This wraps the EXISTING SDPA (whether original or already patched by
    turboquant_attention) to add sparse dispatch during prefill.
    """
    global _PATCHED, _PATTERN_TABLE, _LAYER_COUNTER

    if _PATCHED:
        return False

    try:
        _PATTERN_TABLE = load_pattern_table(model_name)
    except FileNotFoundError as e:
        logger.error(str(e))
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    existing_sdpa = mlx_base.scaled_dot_product_attention

    def minference_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask,
        sinks=None,
    ) -> mx.array:
        L = queries.shape[-2]

        # Only apply sparse dispatch during prefill (L > 1)
        # Decode (L=1) is already handled by TQ3 fused kernel or Quest
        if L > 1 and L > 128:
            return sparse_prefill_sdpa(queries, keys, values, scale, mask)

        # Short sequence or decode — pass through to existing SDPA
        return existing_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = minference_sdpa

    # Also patch any model modules that already imported it
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is existing_sdpa or func is not minference_sdpa:
                setattr(mod, "scaled_dot_product_attention", minference_sdpa)

    _PATCHED = True
    _LAYER_COUNTER[0] = 0
    logger.info(
        "MInference sparse prefill patch applied "
        f"({_PATTERN_TABLE['summary']['pattern_counts']})"
    )
    return True


def reset_layer_counter():
    """Reset layer counter at the start of each forward pass.

    Should be called before each new prefill to ensure correct layer mapping.
    """
    _LAYER_COUNTER[0] = 0
