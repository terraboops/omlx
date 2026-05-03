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

    Raises:
        FileNotFoundError: if the table file is missing.
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

    # Task 382: detect synthetic-placeholder tables and warn LOUDLY.
    # The pattern table at this path may be a synthetic placeholder
    # (per the `note` field) rather than real calibration output.
    # Dispatching with synthetic patterns produces:
    #   - Plausible per-head pattern types (vertical_slash / a_shape / ...)
    #   - Plausible param values (num_vertical_cols, band_width, ...)
    #   - But the values DO NOT reflect the head's actual attention map
    # → likely quality regression on retrieval/long-context gates,
    #   speedup numbers cited in CLAUDE.md become unsubstantiated.
    note = table.get("note", "")
    is_synthetic = "SYNTHETIC" in note or "placeholder" in note.lower()
    if is_synthetic:
        logger.warning(
            "MInference pattern table %s is a SYNTHETIC PLACEHOLDER "
            "(note: %r) — per-head patterns are programmatic defaults, "
            "NOT measured from actual attention maps. Quality regressions "
            "on long-context gates are LIKELY. Predicted speedups in "
            "CLAUDE.md are UNSUBSTANTIATED until calibration runs. To fix: "
            "python scripts/minference_calibrate.py --model <your-model>",
            path.name, note[:100],
        )

    logger.info(
        f"Loaded MInference patterns: {table['num_layers']} layers × "
        f"{table['num_heads']} heads, "
        f"distribution: {table['summary']['pattern_counts']}"
        f"{' [SYNTHETIC]' if is_synthetic else ''}"
    )
    return table


# ---------------------------------------------------------------------------
# Sparse mask builders
# ---------------------------------------------------------------------------

def _build_a_shape_mask(L_q: int, params: dict, L_kv: int = 0) -> mx.array:
    """Build A-shape sparse mask: sink columns + causal band.

    Returns (L_q, L_kv) bool mask where True = attend.
    For chunked prefill, L_kv > L_q (accumulated context).
    """
    if L_kv == 0:
        L_kv = L_q
    num_sink = params.get("num_sink", 4)
    band_width = params.get("band_width", 64)

    # Row indices map to global positions: [L_kv - L_q, L_kv)
    offset = L_kv - L_q
    rows = mx.arange(L_q)[:, None] + offset  # (L_q, 1) global row positions
    cols = mx.arange(L_kv)[None, :]  # (1, L_kv)

    # Causal: cols <= rows (global positions)
    causal = cols <= rows

    # Sink columns: cols < num_sink
    sink = cols < num_sink

    # Band: rows - cols < band_width (local window)
    band = (rows - cols) < band_width

    mask = causal & (sink | band)
    return mask


def _build_vertical_slash_mask(
    L_q: int,
    params: dict,
    keys: Optional[mx.array] = None,
    L_kv: int = 0,
) -> mx.array:
    """Build vertical-slash sparse mask: important columns + causal band.

    Returns (L_q, L_kv) bool mask. For chunked prefill, L_kv > L_q.
    """
    if L_kv == 0:
        L_kv = L_q
    band_width = params.get("band_width", 64)
    num_vert = params.get("num_vertical_cols", 16)

    offset = L_kv - L_q
    rows = mx.arange(L_q)[:, None] + offset  # global positions
    cols = mx.arange(L_kv)[None, :]

    causal = cols <= rows
    band = (rows - cols) < band_width

    # Always use the runtime K-norm heuristic to pick the top-K vertical
    # columns. The calibration table records `vertical_col_indices` as a
    # diagnostic, but those indices are POSITION-indexed (specific to the
    # calibration prompt's content); at runtime with a different prompt,
    # the high-attention positions are different. MInference's design
    # intent is: calibration determines the PATTERN TYPE + PARAM COUNTS
    # per head; the runtime selects the specific columns content-adaptively.
    # Empirically: D15 (2026-05-03) — using stored indices caused 4K NIAH
    # to retrieve `'postgresql://localhost:5432/app_db'` (config-block
    # token) instead of the needle `'ALPHA-7749'`.
    if keys is not None and num_vert > 0:
        k_norms = mx.linalg.norm(keys[0, 0], axis=-1)  # (L_kv,)
        top_indices = mx.argsort(-k_norms)[:num_vert]
        vert_mask = mx.zeros((L_kv,), dtype=mx.bool_)
        vert_mask = vert_mask.at[top_indices].add(mx.ones((num_vert,), dtype=mx.bool_))
        vert = vert_mask[None, :]  # (1, L_kv)
    else:
        vert = cols < num_vert

    mask = causal & (band | vert)
    return mask


def _build_block_sparse_mask(L_q: int, params: dict, L_kv: int = 0) -> mx.array:
    """Build block-sparse mask: block-diagonal with overlap to previous block.

    Returns (L_q, L_kv) bool mask. For chunked prefill, L_kv > L_q.
    """
    if L_kv == 0:
        L_kv = L_q
    block_size = params.get("block_size", 64)

    offset = L_kv - L_q
    rows = mx.arange(L_q)[:, None] + offset  # global positions
    cols = mx.arange(L_kv)[None, :]

    causal = cols <= rows

    row_block = rows // block_size
    col_block = cols // block_size

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
    # Use -3.4e4 (fp16-safe just-below-max-finite) instead of -1e9 which
    # clamps to -inf in fp16 and risks NaN propagation through softmax at
    # high sparsity. SparseKVCache Phase 1 (Tasks 384-386) converged on the
    # same constant for the same reason.
    additive_mask = mx.where(sparse_mask, 0.0, -3.4e4).astype(q.dtype)
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

    B, H_q, L_q, D = queries.shape
    L_kv = keys.shape[2]
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
    if all_dense or L_q <= 128:
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

            # Build mask for this pattern — rectangular (L_q × L_kv) for chunked prefill
            if pattern_type == "a_shape":
                sparse_mask = _build_a_shape_mask(L_q, params, L_kv=L_kv)
            elif pattern_type == "vertical_slash":
                kv_h = head_indices[0] // GQA
                sparse_mask = _build_vertical_slash_mask(
                    L_q, params, keys=keys[:, kv_h:kv_h+1, :, :], L_kv=L_kv,
                )
            elif pattern_type == "block_sparse":
                sparse_mask = _build_block_sparse_mask(L_q, params, L_kv=L_kv)
            else:
                sparse_mask = None

            if sparse_mask is not None:
                # See note at _sparse_sdpa_single_head: -3.4e4 is fp16-safe.
                additive_mask = mx.where(sparse_mask, 0.0, -3.4e4).astype(queries.dtype)
                additive_mask = additive_mask[None, None, :, :]
                if mask is not None and not isinstance(mask, str):
                    combined_mask = additive_mask + mask
                else:
                    # "causal" string or None — sparse mask already includes causality
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

def _model_id_to_pattern_name(model_id: str) -> str:
    """Derive the pattern-table filename stem from a HuggingFace-style model ID.

    Mirrors the convention used in `scripts/minference_calibrate.py` —
    the snake_case stem of the last path component.
    """
    last = model_id.rstrip("/").split("/")[-1]
    return last.lower().replace("-", "_")


def apply_minference_prefill_patch(
    model_name: str = "qwen3_coder_30b_a3b_instruct_8bit",
    model_id: str | None = None,
) -> bool:
    """Monkey-patch SDPA for MInference sparse prefill.

    Args:
        model_name: pattern-table filename stem (without ``.json``). Used
            directly if provided.
        model_id: full HuggingFace-style model ID (e.g.
            ``"mlx-community/Qwen3.6-35B-A3B-4bit"``). If given, overrides
            ``model_name`` by deriving the stem via the same convention as
            the calibration script. This is the path bench/server use to
            stay in sync with whatever model is loaded.

    This wraps the EXISTING SDPA (whether original or already patched by
    turboquant_attention) to add sparse dispatch during prefill.
    """
    global _PATCHED, _PATTERN_TABLE, _LAYER_COUNTER

    if _PATCHED:
        return False

    if model_id is not None:
        model_name = _model_id_to_pattern_name(model_id)

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

        # Only apply sparse dispatch during prefill with fp16 KV (L > 1)
        # Skip quantized KV (tuples from QuantizedKVCache) — dequant params
        # are cache-object-specific and can't be inferred from the tuple alone.
        # MInference works in fp16 mode (--kv-mode fp16) or with TQ3's
        # short-history dequant path.
        if L > 1 and L > 128 and isinstance(keys, mx.array):
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
