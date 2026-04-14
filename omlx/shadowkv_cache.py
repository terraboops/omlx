# SPDX-License-Identifier: Apache-2.0
"""ShadowKV: SVD-compressed K cache for long-context inference (arXiv:2410.21465).

Task 44 proved the K cache is low-rank (median 177/512 at 99% energy).
This module stores K as a low-rank approximation U @ S @ V^T per layer,
with per-layer rank chosen to capture 99% of the Frobenius norm.

Memory savings: ~65% of K cache at the cost of one-time SVD at prefill.

Design:
  - V cache: stored as-is (fp16 or quantized, unmodified)
  - K cache: stored as (U, S, Vt) per layer after prefill
  - Decode: new K tokens appended to a small fp16 overflow buffer;
    periodically re-SVD'd into the main low-rank representation

Usage:
    from omlx.shadowkv_cache import ShadowKVCache
    cache = [ShadowKVCache(target_rank=192) for _ in range(n_layers)]
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


class ShadowKVCache:
    """SVD-compressed K cache with fp16 V cache.

    K is stored as a low-rank approximation: K ≈ U @ diag(S) @ Vt
    where U: (B, H, T, rank), S: (B, H, rank), Vt: (B, H, rank, D).

    V is stored in full fp16: (B, H, T, D).

    During decode, new KV tokens are appended to an fp16 overflow buffer.
    When overflow exceeds `recompress_threshold`, the full K is reconstructed,
    re-SVD'd, and the overflow is cleared.
    """

    def __init__(
        self,
        target_rank: int = 192,
        energy_threshold: float = 0.99,
        recompress_threshold: int = 256,
    ):
        self.target_rank = target_rank
        self.energy_threshold = energy_threshold
        self.recompress_threshold = recompress_threshold
        self.offset = 0

        # Low-rank K components
        self._k_U: Optional[mx.array] = None     # (B, H, T, rank)
        self._k_S: Optional[mx.array] = None     # (B, H, rank)
        self._k_Vt: Optional[mx.array] = None    # (B, H, rank, D)

        # V cache (full, uncompressed)
        self._values: Optional[mx.array] = None   # (B, H, T, D)

        # Overflow buffer for decode tokens (fp16)
        self._k_overflow: Optional[mx.array] = None  # (B, H, overflow_len, D)
        self._v_overflow: Optional[mx.array] = None

        self._compressed = False

    def _compress_keys(self, keys: mx.array):
        """SVD-compress keys: (B, H, T, D) → (U, S, Vt) per head.

        Uses numpy SVD (MLX SVD is CPU-only) and selects rank to capture
        `energy_threshold` of the Frobenius norm, capped at `target_rank`.
        """
        B, H, T, D = keys.shape

        # Process per batch × head (SVD on 2D matrices)
        all_U = []
        all_S = []
        all_Vt = []

        for b in range(B):
            batch_U = []
            batch_S = []
            batch_Vt = []
            for h in range(H):
                K_np = np.array(keys[b, h].astype(mx.float32))  # (T, D)
                U, S, Vt = np.linalg.svd(K_np, full_matrices=False)

                # Select rank: min(target_rank, rank_for_energy_threshold)
                energy = np.cumsum(S ** 2)
                total_energy = energy[-1] if len(energy) > 0 else 1.0
                if total_energy > 0:
                    rank_for_energy = int(np.searchsorted(energy / total_energy,
                                                          self.energy_threshold)) + 1
                else:
                    rank_for_energy = 1
                rank = min(self.target_rank, rank_for_energy, len(S))

                batch_U.append(mx.array(U[:, :rank].astype(np.float16)))
                batch_S.append(mx.array(S[:rank].astype(np.float32)))
                batch_Vt.append(mx.array(Vt[:rank, :].astype(np.float16)))

            all_U.append(mx.stack(batch_U))    # (H, T, rank)
            all_S.append(mx.stack(batch_S))    # (H, rank)
            all_Vt.append(mx.stack(batch_Vt))  # (H, rank, D)

        self._k_U = mx.stack(all_U)    # (B, H, T, rank)
        self._k_S = mx.stack(all_S)    # (B, H, rank)
        self._k_Vt = mx.stack(all_Vt)  # (B, H, rank, D)
        self._compressed = True

        actual_rank = self._k_S.shape[-1]
        compression = 1.0 - (T * actual_rank + actual_rank + actual_rank * D) / (T * D)
        logger.debug(f"ShadowKV compressed: T={T} D={D} rank={actual_rank} "
                     f"({compression:.0%} savings)")

    def _reconstruct_keys(self) -> mx.array:
        """Reconstruct K from low-rank: K = U @ diag(S) @ Vt."""
        if not self._compressed:
            return self._k_overflow  # Not yet compressed

        # U: (B, H, T, rank), S: (B, H, rank), Vt: (B, H, rank, D)
        # K = U @ diag(S) @ Vt = (U * S[..., None, :]) @ Vt
        US = self._k_U * self._k_S[:, :, None, :]  # (B, H, T, rank)
        K = US @ self._k_Vt  # (B, H, T, D)

        # Append overflow if any
        if self._k_overflow is not None and self._k_overflow.shape[2] > 0:
            K = mx.concatenate([K, self._k_overflow], axis=2)

        return K

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new KV. Compress K via SVD after prefill.

        During prefill (T_new > 1): accumulate fp16, compress at end.
        During decode (T_new = 1): append to overflow buffer.
        """
        B, H, T_new, D = keys.shape
        self.offset += T_new

        if T_new > 1:
            # Prefill: accumulate then compress
            if self._k_overflow is None:
                self._k_overflow = keys
                self._values = values
            else:
                self._k_overflow = mx.concatenate([self._k_overflow, keys], axis=2)
                self._values = mx.concatenate([self._values, values], axis=2)

            # Compress after prefill accumulation
            if self._k_overflow.shape[2] >= 512:  # Min tokens for meaningful SVD
                self._compress_keys(self._k_overflow)
                self._k_overflow = None
        else:
            # Decode: append to overflow
            if self._k_overflow is None:
                self._k_overflow = keys
                self._v_overflow = values
            else:
                self._k_overflow = mx.concatenate([self._k_overflow, keys], axis=2)
                self._v_overflow = mx.concatenate([self._v_overflow, values], axis=2)

            # Append values
            if self._values is None:
                self._values = values
            else:
                self._values = mx.concatenate([self._values, values], axis=2)

            # Recompress if overflow is large
            if (self._compressed and self._k_overflow is not None
                    and self._k_overflow.shape[2] >= self.recompress_threshold):
                full_K = self._reconstruct_keys()
                self._compress_keys(full_K)
                self._k_overflow = None

        # Return reconstructed K + full V for attention
        K = self._reconstruct_keys() if self._compressed else self._k_overflow
        return K, self._values

    @property
    def state(self):
        K = self._reconstruct_keys() if self._compressed else self._k_overflow
        return K, self._values

    def memory_savings_gb(self) -> float:
        """Estimate memory savings vs full fp16 K cache."""
        if not self._compressed or self._k_U is None:
            return 0.0
        B, H, T, rank = self._k_U.shape
        D = self._k_Vt.shape[-1]
        full_bytes = B * H * T * D * 2  # fp16
        compressed_bytes = (B * H * T * rank * 2  # U
                           + B * H * rank * 4      # S (float32)
                           + B * H * rank * D * 2) # Vt
        return (full_bytes - compressed_bytes) / 1e9
