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
def _rotation_matrix(dim: int, seed: int = 0) -> mx.array:
    """WHT rotation matrix: block-diagonal diag(signs) @ H.

    Must match turboquant_convert.py and turboquant_linear.py exactly.
    Uses Walsh-Hadamard Transform with random sign flips.
    For non-power-of-2 dims: block decomposition (e.g. 768 → 512+256).
    """
    import math

    def _wht(x):
        shape = x.shape
        D = shape[-1]
        flat = x.reshape(-1, D).astype(mx.float32)
        h = 1
        while h < D:
            flat_r = flat.reshape(-1, D // (2 * h), 2, h)
            a = flat_r[:, :, 0, :]
            b = flat_r[:, :, 1, :]
            flat_r = mx.stack([a + b, a - b], axis=2)
            flat = flat_r.reshape(-1, D)
            h *= 2
        flat = flat / math.sqrt(D)
        return flat.reshape(shape)

    if dim > 0 and (dim & (dim - 1)) == 0:
        key = mx.random.key(seed)
        uniform = mx.random.uniform(shape=(dim,), key=key)
        signs = mx.where(uniform > 0.5, mx.ones(dim), -mx.ones(dim)).astype(mx.float32)
        mx.eval(signs)
        I = mx.eye(dim, dtype=mx.float32)
        H = _wht(I)
        R = signs[:, None] * H
        mx.eval(R)
        return R

    import numpy as np
    R_np = np.zeros((dim, dim), dtype=np.float32)
    remaining = dim
    offset = 0
    block_seed = seed
    while remaining > 0:
        block_size = 1
        while block_size * 2 <= remaining:
            block_size *= 2
        key = mx.random.key(block_seed)
        uniform = mx.random.uniform(shape=(block_size,), key=key)
        signs = mx.where(uniform > 0.5, mx.ones(block_size), -mx.ones(block_size)).astype(mx.float32)
        mx.eval(signs)
        I = mx.eye(block_size, dtype=mx.float32)
        H = _wht(I)
        block = np.array(signs[:, None] * H)
        R_np[offset:offset+block_size, offset:offset+block_size] = block
        offset += block_size
        remaining -= block_size
        block_seed += 1
    R = mx.array(R_np, dtype=mx.float32)
    mx.eval(R)
    return R


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

        # WHT rotation: x_rotated = x @ R (matching converter's rotation)
        R = _rotation_matrix(in_dim, seed=in_dim)
        module._tq_rotation = R

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
            if hasattr(self, '_tq_rotation'):
                x = x @ self._tq_rotation
            return _orig_ql_call(self, x)

        nn.QuantizedLinear.__call__ = _rotated_ql_call
        nn.QuantizedLinear._tq_patched = True

    # Patch QuantizedSwitchLinear
    if not getattr(QuantizedSwitchLinear, '_tq_patched', False):
        _orig_qsl_call = QuantizedSwitchLinear.__call__

        def _rotated_qsl_call(self, x, indices, sorted_indices=False):
            if hasattr(self, '_tq_rotation'):
                x = x @ self._tq_rotation
            return _orig_qsl_call(self, x, indices, sorted_indices=sorted_indices)

        QuantizedSwitchLinear.__call__ = _rotated_qsl_call
        QuantizedSwitchLinear._tq_patched = True
