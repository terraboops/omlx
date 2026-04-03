# SPDX-License-Identifier: Apache-2.0
"""TurboQuant runtime activation rotation for pre-converted TQ3.5 models.

When a model's weights were rotated offline (by turboquant_convert.py), the
runtime must also rotate activations: X_rotated = X @ R before each linear layer.

This patch wraps QuantizedLinear.__call__ to inject the rotation, using the
same deterministic rotation matrix (seed=input_dim) as the converter.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_patch_applied = False


@lru_cache(maxsize=16)
def _givens_angles(dim: int, seed: int = 0) -> Tuple[mx.array, mx.array]:
    """Generate random Givens rotation angles for D/2 coordinate pairs.

    Returns (cos_angles, sin_angles) each of shape (D/2,).
    """
    key = mx.random.key(seed)
    n_pairs = dim // 2
    angles = mx.random.uniform(shape=(n_pairs,), key=key) * 2.0 * 3.14159265
    cos_a = mx.cos(angles).astype(mx.float32)
    sin_a = mx.sin(angles).astype(mx.float32)
    mx.eval(cos_a, sin_a)
    return cos_a, sin_a


def _apply_givens_rotation(x: mx.array, cos_a: mx.array, sin_a: mx.array) -> mx.array:
    """Apply D/2 independent 2D Givens rotations: O(D) per vector.

    Each pair (x[..., 2i], x[..., 2i+1]) is rotated by angle_i:
        x_new[2i]   = cos(a_i) * x[2i] - sin(a_i) * x[2i+1]
        x_new[2i+1] = sin(a_i) * x[2i] + cos(a_i) * x[2i+1]

    Args:
        x: (..., D) input tensor, D must be even
        cos_a: (D/2,) cosines of rotation angles
        sin_a: (D/2,) sines of rotation angles

    Returns:
        (..., D) rotated tensor
    """
    # Split into even/odd pairs
    x_even = x[..., 0::2]  # (..., D/2)
    x_odd = x[..., 1::2]   # (..., D/2)

    # Apply 2D rotation to each pair
    y_even = cos_a * x_even - sin_a * x_odd
    y_odd = sin_a * x_even + cos_a * x_odd

    # Interleave back: stack and reshape
    # (..., D/2) + (..., D/2) → (..., D/2, 2) → (..., D)
    y = mx.stack([y_even, y_odd], axis=-1).reshape(*x.shape)
    return y


def _get_layer_index(name: str) -> int:
    """Extract layer index from module path."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


# Module name patterns to skip rotation (same as converter)
_SKIP_PATTERNS = frozenset({
    "embed_tokens", "wte", "wpe", "embedding",
    "lm_head", "output",
    "norm", "layernorm", "rmsnorm",
    "rotary", "rope", "conv1d",
    "router",  # MoE router gates stay unrotated
})


def _should_skip(name: str) -> bool:
    lower = name.lower()
    return any(pattern in lower for pattern in _SKIP_PATTERNS)


def apply_turboquant_runtime_patch(
    model: Any,
    fp16_layers: int = 0,
) -> int:
    """Wrap QuantizedLinear and QuantizedSwitchLinear layers with activation rotation.

    For each quantized layer whose weights were rotated by the converter,
    monkey-patches __call__ to do: x_rotated = x @ R before the quantized matmul.

    The rotation matrix R is deterministic from the input dimension (seed=input_dim),
    matching what the converter used.

    Args:
        model: The loaded MLX model (with QuantizedLinear layers from TQ3.5 safetensors).
        fp16_layers: Number of initial layers to skip (their weights weren't rotated).

    Returns:
        Number of layers patched.
    """
    global _patch_applied
    if _patch_applied:
        return 0

    patched = 0

    for name, module in model.named_modules():
        # Only patch quantized linear layers
        cls_name = type(module).__name__
        if cls_name not in ("QuantizedLinear", "QuantizedSwitchLinear"):
            continue

        # Skip special modules
        leaf_name = name.split(".")[-1] if name else ""
        if _should_skip(leaf_name) or _should_skip(name):
            continue

        # Skip fp16 layers
        layer_idx = _get_layer_index(name)
        if 0 <= layer_idx < fp16_layers:
            continue

        # Get input dimension for rotation matrix
        # QuantizedLinear/QuantizedSwitchLinear: in_dim = scales.shape[-1] * group_size
        in_dim = module.scales.shape[-1] * module.group_size

        # PlanarQuant: D/2 independent Givens rotations — O(D) per vector
        if in_dim % 2 == 0:
            cos_a, sin_a = _givens_angles(in_dim, seed=in_dim)
            module._tq_givens_cos = cos_a
            module._tq_givens_sin = sin_a
        else:
            # Odd dimension: pad to even, rotate, slice back (rare edge case)
            cos_a, sin_a = _givens_angles(in_dim + 1, seed=in_dim)
            module._tq_givens_cos = cos_a
            module._tq_givens_sin = sin_a
            module._tq_givens_odd = True

        patched += 1

    # Class-level monkey-patch for QuantizedLinear
    if patched > 0:
        _patch_quantized_linear_class()

    logger.info(f"TurboQuant runtime: patched {patched} layers with activation rotation")
    _patch_applied = True
    return patched


def _patch_quantized_linear_class():
    """Patch QuantizedLinear.__call__ to apply rotation if _tq_rotation is set."""
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    # Patch QuantizedLinear
    if not getattr(nn.QuantizedLinear, '_tq_patched', False):
        _orig_ql_call = nn.QuantizedLinear.__call__

        def _rotated_ql_call(self, x):
            if hasattr(self, '_tq_givens_cos'):
                x = _apply_givens_rotation(x, self._tq_givens_cos, self._tq_givens_sin)
            return _orig_ql_call(self, x)

        nn.QuantizedLinear.__call__ = _rotated_ql_call
        nn.QuantizedLinear._tq_patched = True

    # Patch QuantizedSwitchLinear
    if not getattr(QuantizedSwitchLinear, '_tq_patched', False):
        _orig_qsl_call = QuantizedSwitchLinear.__call__

        def _rotated_qsl_call(self, x, indices, sorted_indices=False):
            if hasattr(self, '_tq_givens_cos'):
                x = _apply_givens_rotation(x, self._tq_givens_cos, self._tq_givens_sin)
            return _orig_qsl_call(self, x, indices, sorted_indices=sorted_indices)

        QuantizedSwitchLinear.__call__ = _rotated_qsl_call
        QuantizedSwitchLinear._tq_patched = True
