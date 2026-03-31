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
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_patch_applied = False


@lru_cache(maxsize=16)
def _random_signs(dim: int, seed: int = 0) -> mx.array:
    """Random diagonal sign matrix D for randomized Hadamard: R = D @ H."""
    key = mx.random.key(seed)
    signs = mx.where(mx.random.uniform(shape=(dim,), key=key) > 0.5,
                     mx.ones(dim), -mx.ones(dim))
    mx.eval(signs)
    return signs.astype(mx.float32)


def _fast_hadamard_transform(x: mx.array, signs: mx.array) -> mx.array:
    """In-place fast Walsh-Hadamard transform: O(D log D) instead of O(D^2).

    Applies the randomized Hadamard: y = (x * signs) @ H / sqrt(D)
    where H is the Walsh-Hadamard matrix computed via butterfly operations.

    Args:
        x: (..., D) input tensor, D must be power of 2
        signs: (D,) random sign vector

    Returns:
        (..., D) transformed tensor
    """
    # Apply random signs
    y = x * signs

    D = y.shape[-1]
    h = 1
    while h < D:
        # Butterfly: swap pairs at stride h
        y_even = y[..., 0::2*h]  # won't work for in-place, need reshape
        # Use reshape-based butterfly instead
        y = y.reshape(*y.shape[:-1], D // (2 * h), 2, h)
        a = y[..., 0, :]  # first half
        b = y[..., 1, :]  # second half
        y = mx.concatenate([a + b, a - b], axis=-1).reshape(*x.shape[:-1], D // (2 * h), 2 * h)
        y = y.reshape(*x.shape[:-1], D)
        h *= 2

    # Normalize
    return y / (D ** 0.5)


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

        # Power-of-2 dims use fast WHT, others use dense rotation matrix
        is_pow2 = in_dim > 0 and (in_dim & (in_dim - 1)) == 0

        if is_pow2:
            signs = _random_signs(in_dim, seed=in_dim)
            module._tq_wht_signs = signs
        else:
            # Fallback: dense rotation (slower but correct for non-power-of-2)
            from omlx.turboquant_convert import _rotation_matrix
            module._tq_dense_rotation = _rotation_matrix(in_dim, seed=in_dim)

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
            if hasattr(self, '_tq_wht_signs'):
                x = _fast_hadamard_transform(x, self._tq_wht_signs)
            elif hasattr(self, '_tq_dense_rotation'):
                x = x @ self._tq_dense_rotation
            return _orig_ql_call(self, x)

        nn.QuantizedLinear.__call__ = _rotated_ql_call
        nn.QuantizedLinear._tq_patched = True

    # Patch QuantizedSwitchLinear
    if not getattr(QuantizedSwitchLinear, '_tq_patched', False):
        _orig_qsl_call = QuantizedSwitchLinear.__call__

        def _rotated_qsl_call(self, x, indices, sorted_indices=False):
            if hasattr(self, '_tq_wht_signs'):
                x = _fast_hadamard_transform(x, self._tq_wht_signs)
            elif hasattr(self, '_tq_dense_rotation'):
                x = x @ self._tq_dense_rotation
            return _orig_qsl_call(self, x, indices, sorted_indices=sorted_indices)

        QuantizedSwitchLinear.__call__ = _rotated_qsl_call
        QuantizedSwitchLinear._tq_patched = True
