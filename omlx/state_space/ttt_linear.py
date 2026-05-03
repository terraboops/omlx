# SPDX-License-Identifier: Apache-2.0
"""TTT-Linear block (Sun et al., arXiv:2407.04620).

A linear-time attention replacement whose hidden state is itself a linear
layer ``W ∈ R^{D×D}``, updated per token via inner-loop gradient descent
on a reconstruction objective. Per-token compute is O(D²) regardless of
context length — the structural property that makes prefill cost flat
beyond the chunk-amortized constant.

Math reminder (paper Section 3.1, the core update rule)::

    Q_t = W_Q x_t,  K_t = W_K x_t,  V_t = W_V x_t        (token projections)
    l(W; x_t)  = || W K_t  -  V_t ||²                     (reconstruction loss)
    ∇_W l      = 2 (W K_t - V_t) K_t^T                    (closed-form gradient)
    W_t        = W_{t-1}  -  η ∇_W l(W_{t-1}; x_t)        (inner SGD step)
    o_t        = W_t Q_t                                  (token output)

Mini-batch TTT (paper Section 3.3) processes ``b`` tokens at once with a
single gradient step against the batch — this is what makes the inner loop
fast enough to be practical, since each gradient update on B tokens costs
only marginally more than on 1 token but advances the hidden state ``b``
positions. For Hypercar's 4096-token chunked prefill, the natural choice
is ``mini_batch_size = 4096`` matching ``PREFILL_CHUNK``.

This module is intentionally bench-machinery-free — it's a pure layer.
The integration story (replacing the streaming-tagged heads in Qwen3.6's
``self_attn`` with a TTT-Linear instance per (layer, head)) lives in a
patches module, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TTTLinearConfig:
    """Configuration for a single (layer, head) TTT-Linear block.

    Attributes:
        head_dim: ``D`` — dimension of the per-head representation. For
            Qwen3.6 with ``head_dim=256`` this is 256.
        eta: Inner-loop learning rate for the per-token SGD step. Paper
            uses ``0.1`` for TTT-Linear at start of training; for
            distillation we may need to learn this per (layer, head).
        mini_batch_size: How many tokens are processed per inner gradient
            step. Defaults to 1 (true online); set to 4096 to match
            ``PREFILL_CHUNK`` for prefill efficiency.
        use_layer_norm: Apply LayerNorm to ``W_t Q_t`` before returning
            outputs (paper found this stabilizes training).
    """

    head_dim: int
    eta: float = 0.1
    mini_batch_size: int = 1
    use_layer_norm: bool = True


class TTTLinear(nn.Module):
    """A single (layer, head) TTT-Linear block."""

    def __init__(self, config: TTTLinearConfig):
        super().__init__()
        self.config = config
        D = config.head_dim
        # Token projections (Q, K, V) — same shape and role as in standard
        # attention; the difference is purely in how outputs are computed
        # from them.
        self.W_Q = nn.Linear(D, D, bias=False)
        self.W_K = nn.Linear(D, D, bias=False)
        self.W_V = nn.Linear(D, D, bias=False)
        # Optional output norm (paper uses LN on the post-update output).
        self.ln = nn.LayerNorm(D) if config.use_layer_norm else None

    def init_state(self, batch_size: int, dtype=mx.float32) -> mx.array:
        """Return the initial hidden state ``W_0`` for ``batch_size`` items.

        Identity-scaled-by-1/sqrt(D) init: small but non-zero so the first
        few tokens' outputs are roughly proportional to ``Q`` before any
        K/V history accumulates. Subject to revision based on distillation
        results in Phase 0 step 3.
        """
        D = self.config.head_dim
        W = mx.broadcast_to(
            mx.eye(D, dtype=dtype) / (D ** 0.5),
            (batch_size, D, D),
        )
        return W

    def __call__(
        self,
        x: mx.array,                       # (B, L, D)
        state: Optional[mx.array] = None,  # (B, D, D)
    ) -> tuple[mx.array, mx.array]:
        """Forward pass over ``L`` tokens, threading hidden state.

        Returns
        -------
        outputs : (B, L, D)
            Per-token output ``o_t = LN(W_t Q_t)`` where ``W_t`` is the
            post-update hidden state for the mini-batch containing token t.
        new_state : (B, D, D)
            Final hidden state after consuming all ``L`` tokens. Pass this
            as ``state`` on the next chunk to resume the recurrence.
        """
        B, L, D = x.shape
        if D != self.config.head_dim:
            raise ValueError(
                f"input head_dim={D} does not match config "
                f"head_dim={self.config.head_dim}"
            )
        if state is None:
            W = self.init_state(B, dtype=x.dtype)
        else:
            W = state

        Q = self.W_Q(x)  # (B, L, D)
        K = self.W_K(x)  # (B, L, D)
        V = self.W_V(x)  # (B, L, D)

        b = self.config.mini_batch_size
        eta = self.config.eta

        outputs = []
        start = 0
        while start < L:
            end = min(start + b, L)
            Qb = Q[:, start:end, :]                      # (B, b', D)
            Kb = K[:, start:end, :]                      # (B, b', D)
            Vb = V[:, start:end, :]                      # (B, b', D)
            # Closed-form gradient against W_{t-b} (the pre-update state
            # for this mini-batch). Compute residual R = W K - V with K
            # transposed to (B, D, b'); then grad = 2 R K^T summed over b'.
            Kb_T = mx.transpose(Kb, (0, 2, 1))           # (B, D, b')
            Vb_T = mx.transpose(Vb, (0, 2, 1))           # (B, D, b')
            Rb = mx.matmul(W, Kb_T) - Vb_T               # (B, D, b')
            grad = 2.0 * mx.matmul(Rb, Kb)               # (B, D, D)
            bs_actual = end - start
            W = W - (eta / bs_actual) * grad
            # Outputs use the post-update W (paper: "outputs in a mini-batch
            # all use the same W_t"). o = W Q_i, vectorized over b'.
            Qb_T = mx.transpose(Qb, (0, 2, 1))           # (B, D, b')
            out_T = mx.matmul(W, Qb_T)                   # (B, D, b')
            out = mx.transpose(out_T, (0, 2, 1))         # (B, b', D)
            if self.ln is not None:
                out = self.ln(out)
            outputs.append(out)
            start = end

        return mx.concatenate(outputs, axis=1), W
