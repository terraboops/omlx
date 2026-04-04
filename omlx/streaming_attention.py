# SPDX-License-Identifier: Apache-2.0
"""Streaming TQ Attention: Online Softmax over chunked dequantized KV.

Implements FlashAttention's core algorithm at the Python/MLX level to avoid
ever materializing the full fp16 KV history. Dequantizes compressed KV in
small chunks (~16K tokens) and computes attention incrementally using
numerically stable online softmax.

Peak memory: model + compressed_KV + ONE dequant chunk.
Scales to 512K+ context without memory spikes.

The three layers of streaming:
  1. Prefill chunks (8K tokens via prefill_step_size)
  2. Streaming quantize (compress each chunk immediately)
  3. Streaming dequant (this module — online softmax, never hold full fp16)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def self_attention_with_lse(
    queries: mx.array,  # (B, H_q, L, D)
    keys: mx.array,     # (B, H_kv, L, D)
    values: mx.array,   # (B, H_kv, L, D)
    scale: float,
    mask,
) -> tuple:
    """Self-attention that returns BOTH output AND log-sum-exp.

    Avoids the wasteful recompute in merge_attention by computing LSE
    directly from the raw scores alongside the attention output.
    """
    B, H_q, L, D = queries.shape
    H_kv = keys.shape[1]
    n_groups = H_q // H_kv

    q_scaled = queries * scale

    # Compute scores with GQA handling
    if n_groups > 1:
        q_grouped = q_scaled.reshape(B, H_kv, n_groups, L, D)
        K_expanded = mx.expand_dims(keys, axis=2)
        scores = q_grouped @ K_expanded.swapaxes(-1, -2)
        scores = scores.reshape(B, H_q, L, -1)
    else:
        scores = q_scaled @ keys.swapaxes(-1, -2)

    # Apply causal mask
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            qL, kL = scores.shape[-2], scores.shape[-1]
            q_idx = mx.arange(kL - qL, kL)
            k_idx = mx.arange(kL)
            causal = q_idx[:, None] >= k_idx[None]
            scores = mx.where(causal, scores, -1e9)
        elif mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -1e9)
        else:
            scores = scores + mask

    # LSE directly from raw scores (no recompute needed)
    chunk_max = mx.max(scores, axis=-1, keepdims=True)
    lse = (chunk_max + mx.log(mx.sum(mx.exp(scores - chunk_max), axis=-1, keepdims=True))).squeeze(-1)

    # Attention output
    if n_groups > 1:
        probs = mx.softmax(scores.reshape(B, H_kv, n_groups, L, -1), axis=-1)
        V_expanded = mx.expand_dims(values, axis=2)
        output = (probs @ V_expanded).reshape(B, H_q, L, D)
    else:
        probs = mx.softmax(scores, axis=-1)
        output = probs @ values

    return output, lse


@mx.compile
def _merge_outputs(
    self_out: mx.array,     # (B, H_q, L, D)
    self_lse: mx.array,     # (B, H_q, L)
    history_out: mx.array,  # (B, H_q, L, D)
    history_lse: mx.array,  # (B, H_q, L)
) -> mx.array:
    """Merge self-attention and history-attention via LSE weighting.

    Fused with @mx.compile for a single Metal kernel launch.
    Numerically stable: uses max-subtraction before exp().
    """
    max_lse = mx.maximum(self_lse, history_lse)
    self_w = mx.exp(self_lse - max_lse)
    hist_w = mx.exp(history_lse - max_lse)
    total = self_w + hist_w

    self_w = (self_w / mx.maximum(total, 1e-10))[..., None]
    hist_w = (hist_w / mx.maximum(total, 1e-10))[..., None]

    return self_w * self_out + hist_w * history_out


@mx.compile
def _online_softmax_update(
    o_prev: mx.array,    # (B, H_q, L, D) running output
    l_prev: mx.array,    # (B, H_q, L) running sum
    m_prev: mx.array,    # (B, H_q, L) running max
    scores: mx.array,    # (B, H_q, L, chunk_len) attention scores for this chunk
    V_chunk: mx.array,   # (B, H_kv, chunk_len, D) values for this chunk
    n_groups: int,       # GQA factor: H_q // H_kv
) -> tuple:
    """Numerically stable online softmax update (fused via @mx.compile).

    This is the core FlashAttention accumulation step. Each call processes
    one chunk of KV history and updates the running statistics.
    """
    # chunk_max: (B, H_q, L)
    chunk_max = mx.max(scores, axis=-1)

    # Safe max update — prevents inf in exp()
    m_new = mx.maximum(m_prev, chunk_max)
    alpha = mx.exp(m_prev - m_new)    # rescale factor for previous accumulator
    beta = mx.exp(chunk_max - m_new)  # scale factor for new chunk

    # Softmax within chunk: exp(scores - chunk_max) for numerical stability
    # scores: (B, H_q, L, chunk_len), chunk_max: (B, H_q, L) → broadcast
    p = mx.exp(scores - chunk_max[..., None])  # (B, H_q, L, chunk_len)

    # p @ V_chunk: need to handle GQA (expand V from H_kv to H_q)
    if n_groups > 1:
        # V_chunk: (B, H_kv, chunk_len, D) → (B, H_kv, 1, chunk_len, D)
        V_expanded = mx.expand_dims(V_chunk, axis=2)
        # p: (B, H_q, L, chunk_len) → (B, H_kv, n_groups, L, chunk_len)
        B_size, H_q, L, C = p.shape
        H_kv = V_chunk.shape[1]
        p_grouped = p.reshape(B_size, H_kv, n_groups, L, C)
        # pV: (B, H_kv, n_groups, L, D)
        pV = p_grouped @ V_expanded  # matmul on last two dims
        pV = pV.reshape(B_size, H_q, L, -1)  # (B, H_q, L, D)
    else:
        pV = p @ V_chunk  # (B, H_q, L, D)

    # Update running sum: l_new = alpha * l_prev + beta * sum(p)
    p_sum = mx.sum(p, axis=-1)  # (B, H_q, L)
    l_new = alpha * l_prev + beta * p_sum

    # Update running output: o_new = alpha * o_prev + beta * (p @ V)
    o_new = alpha[..., None] * o_prev + beta[..., None] * pV

    return o_new, l_new, m_new


def streaming_tq_attention(
    queries: mx.array,       # (B, H_q, L, D) queries for new chunk
    codec,                   # TurboQuantMSECodec
    k_norms: mx.array,      # (B, H_kv, T_total) compressed key norms
    k_packed: mx.array,     # (B, H_kv, T_total, pw) compressed key indices
    v_norms: mx.array,      # (B, H_kv, T_total) compressed value norms
    v_packed: mx.array,     # (B, H_kv, T_total, pw) compressed value indices
    total_tokens: int,       # Total tokens in compressed history
    scale: float,            # Attention scale (1/sqrt(d))
    chunk_size: int = 16384, # Dequant chunk size
    mask: Optional[mx.array] = None,
    return_lse: bool = False,  # Return log-sum-exp for merge with self-attention
) -> mx.array:
    """Compute attention over compressed KV using streaming dequant + online softmax.

    NEVER materializes the full fp16 KV history. Dequantizes in chunks,
    computes partial attention, updates running statistics, frees chunk.

    Args:
        queries: (B, H_q, L, D) — queries from the new prefill chunk
        codec: TurboQuantMSECodec for dequantization
        k_norms, k_packed: compressed key storage
        v_norms, v_packed: compressed value storage
        total_tokens: how many tokens are in the compressed history
        scale: attention scale factor
        chunk_size: tokens per dequant chunk (controls peak memory)
        mask: optional causal mask for the current chunk's self-attention

    Returns:
        (B, H_q, L, D) attention output
    """
    B, H_q, L, D = queries.shape
    H_kv = k_norms.shape[1]
    n_groups = H_q // H_kv

    # Scale queries once
    queries = queries * scale

    # Initialize online softmax accumulators
    o = mx.zeros((B, H_q, L, D), dtype=queries.dtype)  # running output
    l = mx.zeros((B, H_q, L), dtype=mx.float32)         # running sum
    m = mx.full((B, H_q, L), float('-inf'), dtype=mx.float32)  # running max

    # Pre-compute GQA reshaping for queries (static, done once)
    if n_groups > 1:
        B_size = queries.shape[0]
        q_grouped = queries.reshape(B_size, H_kv, n_groups, L, D)

    # Adaptive chunk size: scale up when queries L is small.
    # Peak scores tensor = L × chunk_size × H_q × 4 bytes
    # Target 2GB peak → chunk_size = 2GB / (L × H_q × 4)
    target_bytes = 2 * 1024**3
    max_chunk = target_bytes // (L * H_q * 4)
    # Round to power of 2, cap at 32768
    if max_chunk >= 32768: adaptive_chunk = 32768
    elif max_chunk >= 16384: adaptive_chunk = 16384
    elif max_chunk >= 8192: adaptive_chunk = 8192
    elif max_chunk >= 4096: adaptive_chunk = 4096
    elif max_chunk >= 2048: adaptive_chunk = 2048
    else: adaptive_chunk = max(1024, chunk_size)
    chunk_size = min(adaptive_chunk, max(chunk_size, adaptive_chunk))

    n_chunks = (total_tokens + chunk_size - 1) // chunk_size
    logger.debug(
        "streaming_tq_attention: %d history tokens → %d chunks of %d "
        "(B=%d H_q=%d L=%d D=%d) adaptive_from_L=%d",
        total_tokens, n_chunks, chunk_size, B, H_q, L, D, L,
    )

    # Process compressed history in chunks
    for chunk_start in range(0, total_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_tokens)
        actual_len = chunk_end - chunk_start

        logger.debug(
            "  dequant chunk %d-%d (%d tokens)",
            chunk_start, chunk_end, actual_len,
        )

        # Dequantize ONE chunk — small, temporary
        K_chunk = codec.dequantize(
            k_norms[:, :, chunk_start:chunk_end],
            k_packed[:, :, chunk_start:chunk_end],
        )  # (B, H_kv, actual_len, D)

        V_chunk = codec.dequantize(
            v_norms[:, :, chunk_start:chunk_end],
            v_packed[:, :, chunk_start:chunk_end],
        )  # (B, H_kv, actual_len, D)

        # Pad last chunk to chunk_size for @mx.compile cache hits
        if actual_len < chunk_size:
            pad_len = chunk_size - actual_len
            K_chunk = mx.concatenate([
                K_chunk,
                mx.zeros((B, H_kv, pad_len, D), dtype=K_chunk.dtype)
            ], axis=2)
            V_chunk = mx.concatenate([
                V_chunk,
                mx.zeros((B, H_kv, pad_len, D), dtype=V_chunk.dtype)
            ], axis=2)

        # Compute attention scores: Q @ K^T
        if n_groups > 1:
            K_expanded = mx.expand_dims(K_chunk, axis=2)
            scores = q_grouped @ K_expanded.swapaxes(-1, -2)
            scores = scores.reshape(B_size, H_q, L, -1)
        else:
            scores = queries @ K_chunk.swapaxes(-1, -2)

        # Mask out padding in last chunk (set padded scores to -inf)
        if actual_len < chunk_size:
            pad_mask = mx.concatenate([
                mx.ones((1, 1, 1, actual_len)),
                mx.zeros((1, 1, 1, chunk_size - actual_len)),
            ], axis=-1)
            scores = mx.where(pad_mask, scores, -1e9)

        # Online softmax update (fused via @mx.compile)
        o, l, m = _online_softmax_update(o, l, m, scores, V_chunk, n_groups)

        # CRITICAL: Force MLX to evaluate NOW and free chunk tensors.
        # Without this, MLX hoards the entire computation graph and
        # allocates all intermediate tensors simultaneously at execution
        # time, causing peak memory to explode.
        mx.eval(o, l, m)

    # Final normalization: o = o / l
    output = o / mx.maximum(l[..., None], 1e-10)
    output = output.astype(queries.dtype)

    if return_lse:
        # Return log-sum-exp for merging with self-attention
        # LSE = log(l) + m (the log of the total softmax denominator)
        lse = mx.log(mx.maximum(l, 1e-10)) + m  # (B, H_q, L)
        return output, lse

    return output
