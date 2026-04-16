# SPDX-License-Identifier: Apache-2.0
"""LatentKVCache — SVD-projected low-rank KV cache for MLA compression.

Stores KV cache in a compressed latent space using offline-computed SVD
projection matrices. Based on Task 53's MLA probe which found joint KV
rank of 241/1024 (24%) — 76% compression viable.

Architecture:
  Full KV: (B, H_kv, T, D) × 2 (K and V) = (B, H_kv, T, 2D)
  Latent:  (B, T, d_c) where d_c = rank@99% ≈ 241

  Compress: latent = [K; V].reshape(T, H_kv*2D) @ W_down  (full → latent)
  Expand:   [K; V]  = latent @ W_up                        (latent → full)

  W_down: (H_kv*2D, d_c) — left singular vectors from offline SVD
  W_up:   (d_c, H_kv*2D) — pseudo-inverse of W_down

Memory savings at 1M context (Qwen3-Coder, H_kv=4, D=128):
  Full fp16: 48 layers × 1M × 4 × 128 × 2 (K+V) × 2 bytes = 100.7 GB
  Full 3-bit: ~22.5 GB
  Latent:    48 layers × 1M × 241 × 2 bytes = ~22.1 GB (fp16 latent)
  Latent quantized to 3-bit: ~4.1 GB (82% savings vs 3-bit GQA)

The quality-memory trade-off is controlled by d_c (latent dim).
Higher d_c = better quality but more memory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LatentProjection:
    """SVD projection matrices for one layer's KV compression.

    W_down: (kv_dim, d_c) — projects full KV to latent space
    W_up:   (d_c, kv_dim) — projects latent back to full KV
    kv_dim: H_kv * 2 * D (concatenated K and V dimension)
    d_c:    latent dimension (rank cutoff)
    """
    W_down: any  # mx.array (kv_dim, d_c)
    W_up: any    # mx.array (d_c, kv_dim)
    kv_dim: int
    d_c: int
    layer_idx: int

    def compress(self, kv_flat):
        """Compress full KV to latent: (B, T, kv_dim) → (B, T, d_c)."""
        return kv_flat @ self.W_down

    def expand(self, latent):
        """Expand latent to full KV: (B, T, d_c) → (B, T, kv_dim)."""
        return latent @ self.W_up


def compute_projections_from_svd(
    rank_data_path: str | Path,
    model,
    tokenizer,
    d_c: int = 241,
    context_len: int = 4096,
) -> list[LatentProjection]:
    """Compute SVD projection matrices from a calibration prefill.

    This is an offline one-time computation. The resulting projections
    are saved and loaded at server startup.

    Args:
        rank_data_path: Path to MLA rank probe JSON (for reference)
        model: Loaded MLX model
        tokenizer: Model tokenizer
        d_c: Target latent dimension (default: 241 from Task 53 probe)
        context_len: Calibration context length

    Returns:
        List of LatentProjection, one per layer
    """
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import KVCache

    n_layers = len(model.layers)
    projections = []

    # Build calibration context
    code = "def fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n\n"
    tokens = tokenizer.encode(code)
    reps = (context_len // len(tokens)) + 1
    full_tokens = (tokens * reps)[:context_len]

    # Prefill
    cache = [KVCache() for _ in range(n_layers)]
    chunk_size = 2048
    for start in range(0, len(full_tokens), chunk_size):
        end = min(start + chunk_size, len(full_tokens))
        x = mx.array([full_tokens[start:end]])
        logits = model(x, cache=cache)
        mx.eval(logits)

    # Extract projections per layer
    for layer_idx in range(n_layers):
        c = cache[layer_idx]
        keys = c.state[0][0].astype(mx.float32)    # (H_kv, T, D)
        values = c.state[1][0].astype(mx.float32)   # (H_kv, T, D)
        mx.eval(keys, values)

        H_kv, T, D = keys.shape

        # Concatenate K and V: (T, H_kv*2D)
        K_flat = np.array(keys.transpose(1, 0, 2).reshape(T, H_kv * D))
        V_flat = np.array(values.transpose(1, 0, 2).reshape(T, H_kv * D))
        KV_flat = np.concatenate([K_flat, V_flat], axis=1)  # (T, H_kv*2D)

        kv_dim = KV_flat.shape[1]

        # SVD: KV_flat = U @ diag(S) @ Vt
        # W_down = Vt[:d_c].T  (right singular vectors, transposed)
        # W_up = Vt[:d_c]
        U, S, Vt = np.linalg.svd(KV_flat, full_matrices=False)

        # Truncate to d_c
        actual_d_c = min(d_c, len(S))
        W_down_np = Vt[:actual_d_c].T  # (kv_dim, d_c)
        W_up_np = Vt[:actual_d_c]       # (d_c, kv_dim)

        projections.append(LatentProjection(
            W_down=mx.array(W_down_np.astype(np.float16)),
            W_up=mx.array(W_up_np.astype(np.float16)),
            kv_dim=kv_dim,
            d_c=actual_d_c,
            layer_idx=layer_idx,
        ))

        if layer_idx % 16 == 0:
            # Reconstruction quality check
            latent = KV_flat @ W_down_np
            reconstructed = latent @ W_up_np
            mse = np.mean((KV_flat - reconstructed) ** 2)
            rel_err = mse / (np.mean(KV_flat ** 2) + 1e-10)
            logger.info(f"  Layer {layer_idx}: d_c={actual_d_c}, "
                        f"rel_error={rel_err:.6f}")

    return projections


def save_projections(projections: list[LatentProjection],
                     path: str | Path):
    """Save projection matrices to disk."""
    import mlx.core as mx

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "n_layers": len(projections),
        "d_c": projections[0].d_c if projections else 0,
        "kv_dim": projections[0].kv_dim if projections else 0,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))

    for proj in projections:
        mx.savez(str(path / f"layer_{proj.layer_idx}.npz"),
                 W_down=proj.W_down, W_up=proj.W_up)

    logger.info(f"Saved {len(projections)} projections to {path}")


def load_projections(path: str | Path) -> list[LatentProjection]:
    """Load projection matrices from disk."""
    import mlx.core as mx

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())

    projections = []
    for i in range(meta["n_layers"]):
        data = mx.load(str(path / f"layer_{i}.npz"))
        projections.append(LatentProjection(
            W_down=data["W_down"],
            W_up=data["W_up"],
            kv_dim=meta["kv_dim"],
            d_c=meta["d_c"],
            layer_idx=i,
        ))

    logger.info(f"Loaded {len(projections)} projections (d_c={meta['d_c']})")
    return projections


def memory_estimate_gb(n_layers: int, context_len: int, d_c: int,
                        bytes_per_elem: float = 2.0) -> float:
    """Estimate latent KV cache memory in GB.

    Args:
        n_layers: Number of transformer layers
        context_len: Context length in tokens
        d_c: Latent dimension
        bytes_per_elem: Bytes per element (2 for fp16, 0.375 for 3-bit)
    """
    return n_layers * context_len * d_c * bytes_per_elem / 1e9
