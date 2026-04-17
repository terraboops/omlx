# SPDX-License-Identifier: Apache-2.0
"""Trigonometric pre-RoPE importance scoring for KV cache eviction.

From TriAttention (arXiv:2604.04921): pre-RoPE Q and K concentrate
around stable per-head centres. The attention preference decomposes
into a trigonometric series, enabling O(d)-per-key importance scoring.

For a key at position p relative to query at position q:
  score(p) = sum_i (q_centre_i * k_centre_i * cos((q-p) * theta_i))

This is O(d_head) per key — independent of context length n.
Compared to attention-based scoring which is O(n*d_head).

Usage:
    from omlx.patches.trig_score import TrigScorer
    scorer = TrigScorer.load()  # loads calibrated centres
    importance = scorer.score(T, query_pos=T-1)  # (T,) scores
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger("hypercar.trig_score")

DEFAULT_CENTRES_PATH = Path(__file__).parent / "qk_centres" / "qwen3_coder_30b.npz"


class TrigScorer:
    """Trigonometric importance scorer using calibrated Q/K centres.

    Computes per-position importance scores analytically from the
    Q/K centre vectors and RoPE frequency components.
    """

    def __init__(
        self,
        q_centres: dict[int, np.ndarray],
        k_centres: dict[int, np.ndarray],
        theta: np.ndarray,
        rope_dims: int,
        target_layers: list[int] | None = None,
    ):
        self.q_centres = q_centres
        self.k_centres = k_centres
        self.theta = theta  # (half_d,) RoPE frequencies
        self.rope_dims = rope_dims
        self.n_layers = max(q_centres.keys()) + 1

        if target_layers is None:
            target_layers = list(range(max(0, self.n_layers - 4), self.n_layers))
        self.target_layers = target_layers

        # Precompute per-head dot products: q_centre * k_centre per freq
        # For non-traditional RoPE: pairs are (i, i+half_d)
        half_d = len(theta)
        self.head_freq_products = {}  # layer → (H_kv, half_d)
        for lid in target_layers:
            q = q_centres[lid]  # (H_q, D)
            k = k_centres[lid]  # (H_kv, D)
            H_kv = k.shape[0]
            H_q = q.shape[0]
            gqa = H_q // H_kv

            # Average Q centres across GQA group
            q_grouped = q.reshape(H_kv, gqa, -1).mean(axis=1)  # (H_kv, D)

            # Product of paired dims: q[i]*k[i] + q[i+half_d]*k[i+half_d]
            prod = (q_grouped[:, :half_d] * k[:, :half_d] +
                    q_grouped[:, half_d:2*half_d] * k[:, half_d:2*half_d])
            self.head_freq_products[lid] = prod  # (H_kv, half_d)

    @classmethod
    def load(cls, path: str | Path | None = None,
             target_layers: list[int] | None = None) -> TrigScorer:
        """Load calibrated centres from .npz file."""
        path = Path(path) if path else DEFAULT_CENTRES_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Q/K centres not found: {path}\n"
                f"Run: .venv/bin/python scripts/calibrate_qk_centres.py")

        data = np.load(str(path))
        n_layers = int(data["n_layers"][0])
        rope_dims = int(data["rope_dims"][0])
        theta = data["rope_theta"]

        q_centres = {}
        k_centres = {}
        for i in range(n_layers):
            if f"q_centre_{i}" in data:
                q_centres[i] = data[f"q_centre_{i}"]
                k_centres[i] = data[f"k_centre_{i}"]

        logger.info(f"Loaded Q/K centres: {len(q_centres)} layers, "
                    f"rope_dims={rope_dims}, theta shape={theta.shape}")
        return cls(q_centres, k_centres, theta, rope_dims, target_layers)

    def score(self, T: int, query_pos: int | None = None,
              obs_window: int = 64) -> mx.array:
        """Compute per-position importance scores via trigonometric series.

        Args:
            T: Total number of key positions to score
            query_pos: Position of the query (default: T-1, last position)
            obs_window: Number of query positions to average over

        Returns:
            importance: (T,) — per-position importance score
        """
        if query_pos is None:
            query_pos = T - 1

        # Score from multiple query positions (observation window)
        obs_start = max(0, query_pos - obs_window + 1)
        q_positions = mx.arange(obs_start, query_pos + 1)  # (obs_len,)
        k_positions = mx.arange(T)  # (T,)

        # Relative positions: (obs_len, T)
        rel_pos = q_positions[:, None] - k_positions[None, :]  # positive = key is earlier

        # Causal mask: only score positions before the query
        causal = k_positions[None, :] <= q_positions[:, None]  # (obs_len, T)

        theta_mx = mx.array(self.theta)  # (half_d,)

        # Compute trigonometric score per layer, aggregate
        all_scores = []
        for lid in self.target_layers:
            if lid not in self.head_freq_products:
                continue
            prod = mx.array(self.head_freq_products[lid])  # (H_kv, half_d)

            # cos((q-p) * theta_i) for each (obs_pos, key_pos, freq)
            # angles: (obs_len, T, half_d)
            angles = rel_pos[:, :, None].astype(mx.float32) * theta_mx[None, None, :]
            cos_angles = mx.cos(angles)  # (obs_len, T, half_d)

            # Score: sum over frequencies of prod * cos
            # prod: (H_kv, half_d) → broadcast: (1, 1, H_kv, half_d)
            # cos: (obs_len, T, half_d) → (obs_len, T, 1, half_d)
            scores = mx.sum(
                cos_angles[:, :, None, :] * prod[None, None, :, :],
                axis=-1)  # (obs_len, T, H_kv)

            # Apply causal mask
            scores = mx.where(causal[:, :, None], scores, mx.array(0.0))

            # Pool: max over obs window, max over KV heads
            scores = mx.max(scores, axis=0)  # (T, H_kv)
            scores = mx.max(scores, axis=-1)  # (T,)
            all_scores.append(scores)

        if not all_scores:
            return mx.ones(T)

        # Aggregate across layers: max
        stacked = mx.stack(all_scores, axis=0)
        result = mx.max(stacked, axis=0)  # (T,)
        mx.eval(result)
        return result

    def classify_heads(self, freq_threshold: float = 0.3) -> dict[int, list[str]]:
        """Classify heads as streaming vs retrieval by dominant frequency.

        Heads with high-frequency dominance → streaming (narrow window).
        Heads with low-frequency dominance → retrieval (wide window).

        Args:
            freq_threshold: fraction of energy in top half of frequencies
                above which a head is classified as streaming.

        Returns:
            dict mapping layer_idx → list of "streaming"/"retrieval" per head
        """
        classification = {}
        half_d = len(self.theta)
        mid = half_d // 2  # split frequencies into low/high

        for lid in self.q_centres:
            prod = self.head_freq_products.get(lid)
            if prod is None:
                continue
            H_kv = prod.shape[0]
            labels = []
            for h in range(H_kv):
                energy = np.abs(prod[h])
                high_energy = energy[mid:].sum()
                total_energy = energy.sum() + 1e-8
                if high_energy / total_energy > freq_threshold:
                    labels.append("streaming")
                else:
                    labels.append("retrieval")
            classification[lid] = labels

        return classification
