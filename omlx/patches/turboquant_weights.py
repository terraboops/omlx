# SPDX-License-Identifier: Apache-2.0
"""TurboQuant 3.5-bit weight patch: replace Linear layers with TurboQuantLinear.

Walks the model tree and converts nn.Linear / nn.QuantizedLinear layers to
TurboQuantLinear, which applies orthogonal rotation before 3-bit quantized matmul.

Skips:
  - Embedding layers (not weight-quantizable)
  - lm_head / output projection (keep full precision for logit quality)
  - Layers with index < fp16_layers (keep in fp16 for first-layer sensitivity)
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.nn as nn

logger = logging.getLogger(__name__)

_patch_applied = False

# Module name patterns to skip (embeddings, lm_head, normalization, etc.)
_SKIP_PATTERNS = frozenset({
    "embed_tokens", "wte", "wpe", "embedding",
    "lm_head", "output",
    "norm", "layernorm", "rmsnorm",
    "rotary", "rope",
})


def _should_skip(name: str) -> bool:
    """Check if a module name should be skipped for weight quantization."""
    lower = name.lower()
    return any(pattern in lower for pattern in _SKIP_PATTERNS)


def _get_layer_index(name: str) -> int:
    """Extract layer index from module path like 'model.layers.5.self_attn.q_proj'."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1  # Not a numbered layer


def apply_turboquant_weights_patch(
    model: Any,
    fp16_layers: int = 0,
    group_size: int = 64,
    bits: int = 3,
) -> int:
    """Replace Linear / QuantizedLinear layers with TurboQuantLinear.

    Args:
        model: The loaded MLX model.
        fp16_layers: Number of initial layers to keep in fp16.
        group_size: Quantization group size (default 64).
        bits: Quantization bits (default 3).

    Returns:
        Number of layers converted.
    """
    global _patch_applied
    if _patch_applied:
        logger.info("TurboQuant weights patch already applied")
        return 0

    from ..turboquant_linear import TurboQuantLinear

    converted = 0
    skipped = 0
    kept_fp16 = 0

    # Walk the model tree and collect (parent, attr_name, module) for replacement
    replacements = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
            continue

        # Skip special modules
        leaf_name = name.split(".")[-1] if name else ""
        if _should_skip(leaf_name) or _should_skip(name):
            skipped += 1
            continue

        # Skip layers below fp16_layers threshold
        layer_idx = _get_layer_index(name)
        if 0 <= layer_idx < fp16_layers:
            kept_fp16 += 1
            continue

        # Find the parent module and attribute name for replacement
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            elif isinstance(parent, (list, tuple)):
                parent = parent[int(part)]
            else:
                parent = parent[part]

        replacements.append((parent, parts[-1], module, name))

    # Apply replacements
    for parent, attr_name, module, full_name in replacements:
        seed = hash(full_name) & 0xFFFFFFFF  # Deterministic per-layer seed
        tq_linear = TurboQuantLinear.from_linear(
            module, group_size=group_size, bits=bits, seed=seed
        )

        if isinstance(parent, dict):
            parent[attr_name] = tq_linear
        else:
            setattr(parent, attr_name, tq_linear)

        converted += 1

    logger.info(
        f"TurboQuant weights: converted {converted} layers to {bits}-bit "
        f"(skipped {skipped} special, kept {kept_fp16} fp16)"
    )

    _patch_applied = True
    return converted
