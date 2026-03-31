# SPDX-License-Identifier: Apache-2.0
"""TurboQuant 3.5-bit weight quantization: orthogonal rotation + 3-bit quantized matmul.

The "3.5-bit" scheme applies an orthogonal rotation matrix R to weights before
quantizing to 3-bit. At runtime, activations are also rotated so that the
mathematical result is unchanged:

    X_rotated @ W_rotated^T = X @ R @ (W @ R)^T = X @ R @ R^T @ W^T = X @ W^T

The rotation spreads weight outliers evenly across dimensions, making 3-bit
quantization near-optimal for every coordinate. The extra 0.5 bits of overhead
come from the rotation matrix and high-precision group scaling factors.
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


def _rotation_matrix(dim: int, seed: int = 0) -> mx.array:
    """Random orthogonal rotation via QR decomposition.

    Reuses the same algorithm as turboquant_kv.py for consistency.
    """
    key = mx.random.key(seed)
    Q, R = mx.linalg.qr(mx.random.normal(shape=(dim, dim), key=key), stream=mx.cpu)
    signs = mx.sign(mx.diag(R))
    signs = mx.where(signs == 0, mx.ones_like(signs), signs)
    Q = (Q * signs[None, :]).astype(mx.float32)
    mx.eval(Q)
    return Q


class TurboQuantLinear(nn.Module):
    """3.5-bit weight quantization: orthogonal rotation + 3-bit quantized matmul.

    Stores:
      - rotation_matrix: (input_dims, input_dims) orthogonal R
      - weight: packed 3-bit quantized W_rotated
      - scales: per-group scales for dequantization
      - biases: quantization biases (asymmetric)
      - bias: optional affine bias
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 3,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.group_size = group_size
        self.bits = bits

        # Placeholder rotation matrix (identity-like until from_linear is called)
        self.rotation_matrix = mx.eye(input_dims, dtype=mx.float32)

        # Placeholder quantized weights (will be set by from_linear)
        scale = 1.0 / (input_dims ** 0.5)
        placeholder = mx.random.uniform(
            low=-scale, high=scale, shape=(output_dims, input_dims)
        )
        self.weight, self.scales, *biases_list = mx.quantize(
            placeholder, group_size=group_size, bits=bits
        )
        self.biases = biases_list[0] if biases_list else None

        if bias:
            self.bias = mx.zeros((output_dims,))

        self.freeze()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        group_size: int = 64,
        bits: int = 3,
        seed: int = 0,
    ) -> "TurboQuantLinear":
        """Convert nn.Linear or nn.QuantizedLinear to TurboQuantLinear.

        Steps:
          1. Extract fp16 weights (dequantize if needed)
          2. Generate rotation matrix R
          3. Compute W_rotated = W @ R
          4. Quantize W_rotated to 3-bit
          5. Store packed weights + R
        """
        # Extract weight dimensions
        has_bias = hasattr(linear, "bias") and "bias" in linear
        if isinstance(linear, nn.QuantizedLinear):
            # Dequantize first
            w = mx.dequantize(
                linear.weight,
                linear.scales,
                linear.get("biases"),
                group_size=linear.group_size,
                bits=linear.bits,
            )
            input_dims = linear.input_dims
            output_dims = linear.output_dims
        else:
            w = linear.weight  # (output_dims, input_dims)
            output_dims, input_dims = w.shape

        # Create the layer
        layer = cls(input_dims, output_dims, bias=has_bias, group_size=group_size, bits=bits)

        # Generate orthogonal rotation matrix
        R = _rotation_matrix(input_dims, seed=seed)
        layer.rotation_matrix = R

        # Rotate weights: W_rotated = W @ R
        w_rotated = w.astype(mx.float32) @ R

        # Quantize the rotated weights
        layer.weight, layer.scales, *biases_list = mx.quantize(
            w_rotated, group_size=group_size, bits=bits
        )
        layer.biases = biases_list[0] if biases_list else None

        # Copy bias if present
        if has_bias:
            layer.bias = linear.bias

        layer.freeze()
        return layer

    def __call__(self, x: mx.array) -> mx.array:
        # Rotate activations: X_rotated = X @ R
        x_rotated = x @ self.rotation_matrix

        # Quantized matmul: X_rotated @ W_rotated^T
        out = mx.quantized_matmul(
            x_rotated,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

        if "bias" in self:
            out = out + self["bias"]

        return out
