# SPDX-License-Identifier: Apache-2.0
"""Per-layer TTT learning rate schedules from reservoir computing theory.

From Echo State Transformer (arXiv:2507.02917): shallow layers act as
fixed-point reservoirs (low leak rate), deep layers as adaptive readout
(high leak rate). This matches the empirical observation that early
transformer layers capture word-level features while late layers handle
task-specific reasoning.

Usage:
    from omlx.ttt_schedules import compute_layer_lrs, spectral_radius

    lrs = compute_layer_lrs(base_lr=1e-4, n_layers=48, schedule="reservoir")
    # lrs[0] ≈ 0 (frozen), lrs[47] ≈ 1e-4 (full adaptation)
"""

from __future__ import annotations

import math
from typing import Literal


def compute_layer_lrs(
    base_lr: float,
    n_layers: int = 48,
    schedule: Literal["uniform", "reservoir", "cosine"] = "reservoir",
    gamma: float = 1.5,
) -> list[float]:
    """Compute per-layer learning rates.

    Args:
        base_lr: Maximum learning rate (applied to the most adaptive layer).
        n_layers: Total number of layers.
        schedule: Schedule type.
            "uniform": all layers get base_lr (current default).
            "reservoir": alpha_l = base_lr * (l/L)^gamma — shallow frozen, deep adaptive.
            "cosine": high at edges, low in middle (U-shaped importance).
        gamma: Exponent for reservoir schedule (1.5 = moderate, 2.0 = aggressive).

    Returns:
        List of n_layers learning rates.
    """
    if schedule == "uniform":
        return [base_lr] * n_layers

    elif schedule == "reservoir":
        # Power-law: shallow layers ≈ frozen, deep layers = full LR
        lrs = []
        for l in range(n_layers):
            ratio = l / max(n_layers - 1, 1)
            lrs.append(base_lr * (ratio ** gamma))
        return lrs

    elif schedule == "cosine":
        # U-shaped: high at edges, low in middle
        lrs = []
        for l in range(n_layers):
            # cos schedule: 0→π maps to 1→-1
            # We want high at 0 and n-1, low at n/2
            t = l / max(n_layers - 1, 1)  # 0 to 1
            weight = 0.5 * (1 + math.cos(2 * math.pi * t))  # U-shape
            # Ensure minimum 10% of base_lr
            weight = 0.1 + 0.9 * weight
            lrs.append(base_lr * weight)
        return lrs

    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def spectral_radius_approx(W, n_iter: int = 10) -> float:
    """Approximate spectral radius via power iteration.

    Returns the largest singular value (operator norm) of W,
    which bounds the spectral radius.

    Args:
        W: (m, n) matrix (can be mlx.core.array or numpy)
        n_iter: Number of power iteration steps

    Returns:
        Approximate spectral radius (float)
    """
    try:
        import mlx.core as mx
        m, n = W.shape
        # Random initial vector
        v = mx.random.normal((n, 1))
        v = v / mx.sqrt(mx.sum(v * v))

        for _ in range(n_iter):
            u = W @ v
            sigma = mx.sqrt(mx.sum(u * u))
            u = u / mx.maximum(sigma, mx.array(1e-8))
            v = W.T @ u
            sigma = mx.sqrt(mx.sum(v * v))
            v = v / mx.maximum(sigma, mx.array(1e-8))

        mx.eval(sigma)
        return float(sigma.item())
    except ImportError:
        import numpy as np
        m, n = W.shape
        v = np.random.randn(n, 1).astype(np.float32)
        v /= np.linalg.norm(v)
        for _ in range(n_iter):
            u = W @ v
            sigma = np.linalg.norm(u)
            u /= max(sigma, 1e-8)
            v = W.T @ u
            sigma = np.linalg.norm(v)
            v /= max(sigma, 1e-8)
        return float(sigma)


def format_schedule(lrs: list[float], name: str = "schedule") -> str:
    """Pretty-print a learning rate schedule."""
    n = len(lrs)
    lines = [f"TTT {name}: {n} layers"]
    lines.append(f"  Edge (0-3):   {[f'{lr:.2e}' for lr in lrs[:4]]}")
    mid = n // 2
    lines.append(f"  Middle ({mid-1}-{mid+1}): {[f'{lr:.2e}' for lr in lrs[mid-1:mid+2]]}")
    lines.append(f"  Edge ({n-4}-{n-1}): {[f'{lr:.2e}' for lr in lrs[-4:]]}")
    lines.append(f"  Min: {min(lrs):.2e}, Max: {max(lrs):.2e}, Ratio: {max(lrs)/max(min(lrs), 1e-12):.0f}x")
    return "\n".join(lines)
