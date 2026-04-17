# SPDX-License-Identifier: Apache-2.0
"""OPLoRA: Orthogonal Projection for LoRA gradient safety.

Prevents catastrophic forgetting during test-time training (TTT) by
projecting LoRA gradients onto the subspace orthogonal to the frozen
model's top-k singular vectors. This ensures TTT updates don't
interfere with the base model's learned representations.

From OPLoRA (arXiv:2510.13003): double-sided orthogonal projection
preserves the base model's capacity while allowing adaptation in
the complementary subspace.

Usage:
    from omlx.oplora import project_lora_grads, compute_svd_cache

    # Once at startup: compute truncated SVD for target layers
    svd_cache = compute_svd_cache(model, target_layers, k=8)

    # Each TTT step: project gradients before applying
    dA_safe, dB_safe = project_lora_grads(
        W, A, B, dA, dB, svd_cache[layer_name])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger("hypercar.oplora")

DEFAULT_SVD_CACHE_PATH = Path("omlx/ttt_svd_cache.npz")


def _truncated_svd(W: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Compute truncated SVD of W: top-k left and right singular vectors.

    Uses randomized SVD for efficiency: project to 2k dimensions,
    compute exact SVD there, then project back.

    Args:
        W: (m, n) weight matrix
        k: number of top singular vectors to keep

    Returns:
        U_k: (m, k) — top-k left singular vectors
        V_k: (n, k) — top-k right singular vectors
    """
    m, n = W.shape
    k = min(k, min(m, n))

    if max(m, n) <= 256:
        # Small matrix: use full SVD
        U, S, Vt = mx.linalg.svd(W.astype(mx.float32), stream=mx.cpu)
        mx.eval(U, S, Vt)
        return U[:, :k], Vt[:k, :].T
    else:
        # Randomized SVD for large matrices
        # Step 1: Random projection to 2k dimensions
        oversample = min(2 * k, n)
        Omega = mx.random.normal((n, oversample)).astype(mx.float32)
        Y = W.astype(mx.float32) @ Omega  # (m, oversample)
        mx.eval(Y)

        # Step 2: QR factorization
        Q, R = mx.linalg.qr(Y, stream=mx.cpu)
        mx.eval(Q, R)

        # Step 3: Project W into the low-rank subspace
        B_proj = Q.T @ W.astype(mx.float32)  # (oversample, n)
        mx.eval(B_proj)

        # Step 4: SVD of the small matrix
        U_small, S, Vt = mx.linalg.svd(B_proj, stream=mx.cpu)
        mx.eval(U_small, S, Vt)

        # Step 5: Recover full-size U
        U_k = Q @ U_small[:, :k]  # (m, k)
        V_k = Vt[:k, :].T  # (n, k)
        mx.eval(U_k, V_k)
        return U_k, V_k


def project_lora_grads(
    W: mx.array,
    A: mx.array,
    B: mx.array,
    dA: mx.array,
    dB: mx.array,
    svd_data: dict[str, mx.array],
) -> tuple[mx.array, mx.array]:
    """Project LoRA gradients onto the orthogonal complement of W's top-k SVD.

    Ensures that LoRA updates don't interfere with the base model's
    primary learned directions.

    Args:
        W: (m, n) frozen weight matrix
        A: (r, n) LoRA down-projection
        B: (m, r) LoRA up-projection
        dA: (r, n) gradient for A
        dB: (m, r) gradient for B
        svd_data: dict with "U_k" (m, k) and "V_k" (n, k)

    Returns:
        (dA_projected, dB_projected) — gradients in the safe subspace
    """
    U_k = svd_data["U_k"]  # (m, k) — top-k left singular vectors
    V_k = svd_data["V_k"]  # (n, k) — top-k right singular vectors

    # Project dA: remove component along V_k (right singular vectors)
    # dA lives in (r, n) space. Project out the V_k directions from columns.
    # dA_safe = dA - dA @ V_k @ V_k^T
    dA_proj = dA @ V_k  # (r, k)
    dA_safe = dA - dA_proj @ V_k.T  # (r, n)

    # Project dB: remove component along U_k (left singular vectors)
    # dB lives in (m, r) space. Project out the U_k directions from rows.
    # dB_safe = dB - U_k @ U_k^T @ dB
    dB_proj = U_k.T @ dB  # (k, r)
    dB_safe = dB - U_k @ dB_proj  # (m, r)

    return dA_safe, dB_safe


def compute_svd_cache(
    model,
    target_layers: list[str] | None = None,
    k: int = 8,
) -> dict[str, dict[str, mx.array]]:
    """Compute truncated SVD for frozen weight matrices.

    Caches the top-k singular vectors for each target weight matrix.
    These are used by project_lora_grads to ensure gradient safety.

    Args:
        model: The loaded model
        target_layers: List of layer paths to cache (default: down_proj in MoE)
        k: Number of top singular vectors to keep

    Returns:
        dict mapping layer_path → {"U_k": array, "V_k": array}
    """
    cache = {}

    if target_layers is None:
        # Default: cache SVD for layers the TTT engine adapts
        # TTT adapts down_proj in expert MoE layers
        target_layers = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                # MoE layer — TTT adapts shared_expert or gate
                if hasattr(layer.mlp, 'shared_expert'):
                    target_layers.append(f"layers.{i}.mlp.shared_expert.down_proj")

    for path in target_layers:
        # Navigate to the weight matrix
        parts = path.split(".")
        obj = model
        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part, None)
            if obj is None:
                logger.warning(f"SVD cache: path {path} not found")
                break

        if obj is None:
            continue

        W = obj.weight if hasattr(obj, 'weight') else obj
        if not isinstance(W, mx.array):
            continue

        logger.info(f"Computing SVD for {path}: {W.shape} (k={k})")
        U_k, V_k = _truncated_svd(W, k)
        cache[path] = {"U_k": U_k, "V_k": V_k}

    logger.info(f"SVD cache: {len(cache)} matrices cached (k={k})")
    return cache


def save_svd_cache(cache: dict, path: Path = DEFAULT_SVD_CACHE_PATH) -> None:
    """Save SVD cache to disk."""
    save_dict = {}
    for layer_path, data in cache.items():
        safe_key = layer_path.replace(".", "_")
        save_dict[f"{safe_key}_U_k"] = np.array(data["U_k"].astype(mx.float32))
        save_dict[f"{safe_key}_V_k"] = np.array(data["V_k"].astype(mx.float32))
    save_dict["_layer_paths"] = np.array(list(cache.keys()))
    np.savez(str(path), **save_dict)
    logger.info(f"Saved SVD cache to {path}")


def load_svd_cache(path: Path = DEFAULT_SVD_CACHE_PATH) -> dict:
    """Load SVD cache from disk."""
    if not path.exists():
        raise FileNotFoundError(f"SVD cache not found: {path}")
    data = np.load(str(path), allow_pickle=True)
    cache = {}
    for layer_path in data["_layer_paths"]:
        safe_key = str(layer_path).replace(".", "_")
        cache[str(layer_path)] = {
            "U_k": mx.array(data[f"{safe_key}_U_k"]),
            "V_k": mx.array(data[f"{safe_key}_V_k"]),
        }
    logger.info(f"Loaded SVD cache: {len(cache)} matrices")
    return cache
