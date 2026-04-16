# SPDX-License-Identifier: Apache-2.0
"""PyramidKV per-layer KV cache budget allocation.

Derived from PyramidKV (arXiv:2406.02069): allocate more KV budget to
early and late layers (high attention diversity / task-critical) and less
to middle layers (topologically redundant representations).

The budget vector is a 48-entry array that sums to `total_budget`.
Each entry is the maximum number of tokens that layer should keep.

Default schedule: exponential decay from edges to center.
  budget[i] = max(floor, base * beta^distance_from_nearest_edge)
  where distance = min(i, n_layers - 1 - i)

Usage:
    from omlx.pyramid_budget import compute_budget_vector, budget_for_layer
    budget = compute_budget_vector(total_budget=16384, n_layers=48)
    layer_budget = budget_for_layer(layer_idx=24, total_budget=16384)
"""

from __future__ import annotations

import math


def compute_budget_vector(
    total_budget: int,
    n_layers: int = 48,
    beta: float = 0.7,
    floor_pct: float = 0.3,
) -> list[int]:
    """Compute per-layer KV budget vector using pyramidal decay.

    The schedule gives highest budgets to edge layers (0, 1, ..., and
    n-1, n-2, ...) and lowest to middle layers. This matches the
    PyramidKV paper's finding that middle layers have redundant attention
    patterns and can survive with fewer KV tokens.

    Args:
        total_budget: Total tokens across all layers (sum of vector).
        n_layers: Number of transformer layers (48 for Qwen3-Coder).
        beta: Decay rate per layer from edge (0.7 = 30% decay per step).
            Lower = more aggressive compression of middle layers.
        floor_pct: Minimum budget as fraction of the uniform allocation
            (total_budget / n_layers). Prevents any layer from being
            starved below this floor.

    Returns:
        List of n_layers ints summing to total_budget.
    """
    uniform = total_budget / n_layers
    floor = max(1, int(uniform * floor_pct))

    # Raw weights: beta^distance from nearest edge
    raw = []
    for i in range(n_layers):
        dist = min(i, n_layers - 1 - i)
        raw.append(beta ** dist)

    # Two-pass normalization:
    # Pass 1: identify layers that would fall below floor
    raw_sum = sum(raw)
    floored = []
    non_floored_raw = []
    non_floored_idx = []
    floor_total = 0
    for i, w in enumerate(raw):
        scaled = total_budget * w / raw_sum
        if scaled < floor:
            floored.append(i)
            floor_total += floor
        else:
            non_floored_raw.append(w)
            non_floored_idx.append(i)

    # Pass 2: distribute remaining budget among non-floored layers
    remaining = total_budget - floor_total
    non_floored_sum = sum(non_floored_raw) if non_floored_raw else 1

    budgets = [floor] * n_layers
    for i, w in zip(non_floored_idx, non_floored_raw):
        budgets[i] = max(floor, int(remaining * w / non_floored_sum))

    # Fine-tune to hit exact total
    delta = total_budget - sum(budgets)
    if delta > 0:
        # Add remainder to edge layers
        for j in range(delta):
            budgets[j % n_layers] += 1
    elif delta < 0:
        # Remove from highest-budget layers
        sorted_idx = sorted(range(n_layers), key=lambda i: -budgets[i])
        for j in range(-delta):
            idx = sorted_idx[j % n_layers]
            if budgets[idx] > floor:
                budgets[idx] -= 1

    return budgets


def budget_for_layer(
    layer_idx: int,
    total_budget: int,
    n_layers: int = 48,
    beta: float = 0.7,
    floor_pct: float = 0.3,
) -> int:
    """Get the KV budget for a single layer.

    Convenience wrapper around compute_budget_vector for when you
    need just one layer's budget (e.g., in a per-layer eviction loop).
    """
    budgets = compute_budget_vector(total_budget, n_layers, beta, floor_pct)
    return budgets[layer_idx]


def format_budget_summary(budgets: list[int]) -> str:
    """Pretty-print the budget vector for logging."""
    n = len(budgets)
    total = sum(budgets)
    lines = [f"PyramidKV budget: {total} total across {n} layers"]
    lines.append(f"  Edge (0-3):   {budgets[:4]} (avg {sum(budgets[:4])//4})")
    mid = n // 2
    lines.append(f"  Middle ({mid-2}-{mid+1}): {budgets[mid-2:mid+2]} (avg {sum(budgets[mid-2:mid+2])//4})")
    lines.append(f"  Edge ({n-4}-{n-1}): {budgets[-4:]} (avg {sum(budgets[-4:])//4})")
    lines.append(f"  Min: {min(budgets)}, Max: {max(budgets)}, Ratio: {max(budgets)/max(min(budgets),1):.1f}x")
    return "\n".join(lines)
