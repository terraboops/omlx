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

import math
from typing import Optional

import mlx.core as mx


def merge_attention(
    self_out: mx.array,      # (B, H_q, L, D) self-attention output (new chunk)
    history_out: mx.array,   # (B, H_q, L, D) history attention output
    history_lse: mx.array,   # (B, H_q, L) history log-sum-exp
    queries: mx.array,       # (B, H_q, L, D) original queries (for self-attn LSE)
    keys: mx.array,          # (B, H_kv, L, D) new chunk keys
    values: mx.array,        # (B, H_kv, L, D) new chunk values
    scale: float,
    mask,
) -> mx.array:
    """Merge self-attention (on new chunk) with streaming history attention.

    Uses the online softmax merge trick: rescale each output by the ratio
    of its partition's softmax denominator to the total denominator.

    self_out came from mx.fast.scaled_dot_product_attention (with causal mask).
    history_out came from streaming_tq_attention (no mask, all past visible).
    """
    B, H_q, L, D = self_out.shape
    H_kv = keys.shape[1]
    n_groups = H_q // H_kv

    # Compute self-attention LSE (log-sum-exp of the new chunk's scores)
    # We need to recompute scores for the self-attention to get its LSE
    q_scaled = queries * scale
    if n_groups > 1:
        q_grouped = q_scaled.reshape(B, H_kv, n_groups, L, D)
        K_expanded = mx.expand_dims(keys, axis=2)
        scores = q_grouped @ K_expanded.swapaxes(-1, -2)
        scores = scores.reshape(B, H_q, L, -1)
    else:
        scores = q_scaled @ keys.swapaxes(-1, -2)

    # Apply causal mask to self-attention scores
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            qL, kL = scores.shape[-2], scores.shape[-1]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            causal = q_indices[:, None] >= k_indices[None]
            scores = mx.where(causal, scores, -1e9)
        elif mask is not None:
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, -1e9)
            else:
                scores = scores + mask

    # Self-attention LSE: log(sum(exp(scores))) per query
    self_lse = mx.logsumexp(scores, axis=-1)  # (B, H_q, L)

    # Merge: weighted combination based on LSE
    # total_lse = log(exp(self_lse) + exp(history_lse))
    max_lse = mx.maximum(self_lse, history_lse)
    self_weight = mx.exp(self_lse - max_lse)      # (B, H_q, L)
    hist_weight = mx.exp(history_lse - max_lse)    # (B, H_q, L)
    total_weight = self_weight + hist_weight

    # Normalize weights
    self_w = (self_weight / mx.maximum(total_weight, 1e-10))[..., None]   # (B, H_q, L, 1)
    hist_w = (hist_weight / mx.maximum(total_weight, 1e-10))[..., None]

    merged = self_w * self_out + hist_w * history_out
    return merged.astype(self_out.dtype)


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

    # Process compressed history in chunks
    for chunk_start in range(0, total_tokens, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_tokens)

        # Dequantize ONE chunk — small, temporary
        K_chunk = codec.dequantize(
            k_norms[:, :, chunk_start:chunk_end],
            k_packed[:, :, chunk_start:chunk_end],
        )  # (B, H_kv, chunk_len, D)

        V_chunk = codec.dequantize(
            v_norms[:, :, chunk_start:chunk_end],
            v_packed[:, :, chunk_start:chunk_end],
        )  # (B, H_kv, chunk_len, D)

        # Compute attention scores: Q @ K^T
        # queries: (B, H_q, L, D), K_chunk: (B, H_kv, chunk_len, D)
        if n_groups > 1:
            # GQA: expand K from H_kv to H_q
            K_expanded = mx.expand_dims(K_chunk, axis=2)  # (B, H_kv, 1, chunk_len, D)
            B_size = queries.shape[0]
            q_grouped = queries.reshape(B_size, H_kv, n_groups, L, D)
            # scores: (B, H_kv, n_groups, L, chunk_len)
            scores = q_grouped @ K_expanded.swapaxes(-1, -2)
            scores = scores.reshape(B_size, H_q, L, -1)  # (B, H_q, L, chunk_len)
        else:
            scores = queries @ K_chunk.swapaxes(-1, -2)  # (B, H_q, L, chunk_len)

        # Causal masking: only needed if this chunk contains future tokens
        # For PAST chunks (chunk_end <= current_query_position): no mask needed
        # For CURRENT chunk: apply causal mask
        # For simplicity in streaming prefill: all history chunks are PAST
        # (the current chunk's self-attention is handled by the regular SDPA path)

        # Online softmax update (fused via @mx.compile)
        o, l, m = _online_softmax_update(o, l, m, scores, V_chunk, n_groups)

        # K_chunk, V_chunk go out of scope — MLX can free them
        # (lazy eval means they may persist until the next mx.eval, but
        #  the computation graph doesn't reference them after this iteration)

    # Final normalization: o = o / l
    output = o / mx.maximum(l[..., None], 1e-10)
    output = output.astype(queries.dtype)

    if return_lse:
        # Return log-sum-exp for merging with self-attention
        # LSE = log(l) + m (the log of the total softmax denominator)
        lse = mx.log(mx.maximum(l, 1e-10)) + m  # (B, H_q, L)
        return output, lse

    return output
