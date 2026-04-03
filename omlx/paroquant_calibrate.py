# SPDX-License-Identifier: Apache-2.0
"""ParoQuant-lite: learn optimal Givens rotation angles from calibration data.

Instead of random Givens angles, we search for the angle per channel pair
that minimizes the quantization error of the weight matrix. This is a
simplified version of ParoQuant (ICLR 2026) that doesn't require PyTorch/CUDA.

Algorithm:
  For each pair (i, j) of weight columns:
    1. Try N candidate angles (0, pi/N, 2*pi/N, ...)
    2. For each angle: rotate columns i,j → quantize → measure MSE
    3. Pick the angle with lowest MSE

This is O(N * D * groups) per layer — expensive but one-time.
The resulting angles are stored alongside the quantized model.
"""

from __future__ import annotations

import logging
import time
from typing import Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def calibrate_givens_angles(
    weight: mx.array,
    group_size: int = 64,
    bits: int = 3,
    n_candidates: int = 16,
) -> Tuple[mx.array, mx.array]:
    """Find optimal Givens rotation angles for a weight matrix.

    For each pair of adjacent columns (2i, 2i+1), searches over n_candidates
    angles to find the one that minimizes post-quantization MSE.

    Args:
        weight: (out_dim, in_dim) weight matrix
        group_size: quantization group size
        bits: quantization bits
        n_candidates: angles to try per pair (more = better but slower)

    Returns:
        (cos_angles, sin_angles) — optimal angles for D/2 pairs
    """
    out_dim, in_dim = weight.shape
    n_pairs = in_dim // 2

    # Candidate angles: evenly spaced in [0, pi) — pi suffices due to symmetry
    candidate_angles = np.linspace(0, np.pi, n_candidates, endpoint=False)

    w_np = np.array(weight.astype(mx.float32))
    best_cos = np.zeros(n_pairs, dtype=np.float32)
    best_sin = np.zeros(n_pairs, dtype=np.float32)

    for p in range(n_pairs):
        i, j = 2 * p, 2 * p + 1
        col_i = w_np[:, i].copy()
        col_j = w_np[:, j].copy()

        best_mse = float('inf')
        best_angle = 0.0

        for angle in candidate_angles:
            c, s = np.cos(angle), np.sin(angle)

            # Rotate columns
            rot_i = c * col_i - s * col_j
            rot_j = s * col_i + c * col_j

            # Put rotated columns back, quantize, measure error
            w_test = w_np.copy()
            w_test[:, i] = rot_i
            w_test[:, j] = rot_j

            # Quick MSE estimate: quantize just the affected columns' groups
            # For speed, we quantize the full row and measure MSE on affected cols
            w_mx = mx.array(w_test)
            qw, scales, *rest = mx.quantize(w_mx, group_size=group_size, bits=bits)
            biases = rest[0] if rest else None
            w_deq = mx.dequantize(qw, scales, biases, group_size=group_size, bits=bits)
            mx.eval(w_deq)

            # MSE on the two affected columns
            w_deq_np = np.array(w_deq)
            mse = ((w_test[:, i] - w_deq_np[:, i]) ** 2).mean() + \
                  ((w_test[:, j] - w_deq_np[:, j]) ** 2).mean()

            if mse < best_mse:
                best_mse = mse
                best_angle = angle

        best_cos[p] = np.cos(best_angle)
        best_sin[p] = np.sin(best_angle)

    return mx.array(best_cos), mx.array(best_sin)


def calibrate_givens_angles_fast(
    weight: mx.array,
    group_size: int = 64,
    bits: int = 3,
    n_candidates: int = 16,
) -> Tuple[mx.array, mx.array]:
    """Fast vectorized version: processes all pairs in parallel.

    Instead of per-pair quantize/dequantize (expensive), uses the
    analytical MSE formula for uniform quantization after rotation.

    The key insight: rotation changes the column distribution's variance
    and kurtosis. The angle that makes the two columns most "uniform-like"
    (lowest kurtosis) will quantize best.

    Args:
        weight: (out_dim, in_dim) weight matrix
        group_size: quantization group size
        bits: quantization bits
        n_candidates: angles to try per pair

    Returns:
        (cos_angles, sin_angles) — optimal angles for D/2 pairs
    """
    out_dim, in_dim = weight.shape
    n_pairs = in_dim // 2

    w = weight.astype(mx.float32)
    col_even = w[:, 0::2]  # (out_dim, n_pairs)
    col_odd = w[:, 1::2]   # (out_dim, n_pairs)

    # Candidate angles
    angles = mx.linspace(0, 3.14159265, n_candidates)  # (n_candidates,)
    cos_a = mx.cos(angles)  # (n_candidates,)
    sin_a = mx.sin(angles)

    best_mse = mx.full((n_pairs,), float('inf'))
    best_idx = mx.zeros((n_pairs,), dtype=mx.int32)

    for k in range(n_candidates):
        c = cos_a[k]
        s = sin_a[k]

        # Rotate all pairs simultaneously
        rot_even = c * col_even - s * col_odd  # (out_dim, n_pairs)
        rot_odd = s * col_even + c * col_odd

        # Proxy for quantization error: variance of residuals after rounding
        # to nearest quantization level. Approximated by column-wise kurtosis
        # (lower kurtosis = more uniform = better quantization)
        # Simpler proxy: range / std ratio (lower = more uniform)
        for cols in [rot_even, rot_odd]:
            col_range = cols.max(axis=0) - cols.min(axis=0)
            col_std = mx.sqrt(mx.mean(cols ** 2, axis=0) - mx.mean(cols, axis=0) ** 2 + 1e-10)
            # A column with small range/std quantizes better
            mse_proxy = col_range / (col_std + 1e-10)

        # Combined proxy for the pair
        range_even = rot_even.max(axis=0) - rot_even.min(axis=0)
        range_odd = rot_odd.max(axis=0) - rot_odd.min(axis=0)
        std_even = mx.sqrt(mx.var(rot_even, axis=0) + 1e-10)
        std_odd = mx.sqrt(mx.var(rot_odd, axis=0) + 1e-10)
        pair_mse = (range_even / std_even) + (range_odd / std_odd)

        mx.eval(pair_mse)

        # Update best
        improved = pair_mse < best_mse
        best_mse = mx.where(improved, pair_mse, best_mse)
        best_idx = mx.where(improved, k, best_idx)

    mx.eval(best_idx)

    # Gather best angles
    best_angles = mx.take(angles, best_idx)
    return mx.cos(best_angles).astype(mx.float32), mx.sin(best_angles).astype(mx.float32)
