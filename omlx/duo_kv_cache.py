# SPDX-License-Identifier: Apache-2.0
"""DuoAttention two-storage-class KV cache (arXiv:2410.10819).

Each attention head is classified as either:
  - retrieval: needs full KV context → uses QuantizedKVCache (native 3-bit)
  - streaming: needs only sink + local window → uses fp16 ring buffer

The policy table from Task 12 calibration determines the classification.
Memory savings come from streaming heads using ~256-token ring buffers
instead of full-context caches.

Usage:
    from omlx.duo_kv_cache import DuoKVCache, load_duo_policy
    policy = load_duo_policy()
    cache = [DuoKVCache(policy, layer_idx=i) for i in range(n_layers)]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache, KVCache, QuantizedKVCache

# Lazy import to avoid circular dependency
_TurboQuantKVCache = None
def _get_tq_cache_class():
    global _TurboQuantKVCache
    if _TurboQuantKVCache is None:
        from omlx.turboquant_kv import TurboQuantKVCache
        _TurboQuantKVCache = TurboQuantKVCache
    return _TurboQuantKVCache

logger = logging.getLogger(__name__)

DEFAULT_POLICY_DIR = Path(__file__).parent / "patches" / "duoattention_policies"


def load_duo_policy(
    model_name: str = "qwen3_coder_30b_a3b_instruct_8bit",
) -> dict:
    """Load per-head DuoAttention policy from calibration JSON."""
    path = DEFAULT_POLICY_DIR / f"{model_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"DuoAttention policy not found: {path}\n"
            f"Run: python scripts/duoattention_calibrate.py"
        )
    policy = json.loads(path.read_text())

    # Build per-layer lookup: layer_idx → {head_idx: policy_type}
    lookup = {}
    for entry in policy["heads"]:
        layer = entry["layer"]
        head = entry["head"]
        lookup.setdefault(layer, {})[head] = entry["policy"]

    policy["_lookup"] = lookup
    streaming_pct = policy.get("streaming_fraction", 0) * 100
    if not hasattr(load_duo_policy, "_logged"):
        logger.info(f"DuoAttention policy: {streaming_pct:.0f}% streaming, "
                    f"window={policy.get('window', 256)}, sink={policy.get('sink', 4)}")
        load_duo_policy._logged = True
    return policy


class StreamingKVCache:
    """Ring-buffer KV cache for streaming heads.

    Keeps only the most recent `window` tokens plus `sink` initial tokens.
    Total capacity: sink + window tokens in fp16.
    """

    def __init__(self, window: int = 256, sink: int = 4):
        self.window = window
        self.sink = sink
        self.capacity = sink + window
        self.offset = 0
        self._keys: Optional[mx.array] = None   # (B, 1, capacity, D)
        self._values: Optional[mx.array] = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Add new KV and return the ring buffer contents.

        keys/values shape: (B, 1, T_new, D) — single head per cache.
        """
        B, H, T_new, D = keys.shape

        if self._keys is None:
            # First call — allocate buffer
            self._keys = mx.zeros((B, H, self.capacity, D), dtype=keys.dtype)
            self._values = mx.zeros((B, H, self.capacity, D), dtype=values.dtype)

        if self.offset < self.capacity:
            # Still filling — append directly
            end = min(self.offset + T_new, self.capacity)
            actual = end - self.offset
            self._keys[:, :, self.offset:end] = keys[:, :, :actual]
            self._values[:, :, self.offset:end] = values[:, :, :actual]
            self.offset = end

            if T_new > actual:
                # Overflow into ring region — vectorized scatter
                remaining = T_new - actual
                ring_start = self.sink
                ring_len = self.window
                positions = ring_start + (mx.arange(remaining) + (self.offset - self.capacity)) % ring_len
                self._keys[:, :, positions] = keys[:, :, actual:actual + remaining]
                self._values[:, :, positions] = values[:, :, actual:actual + remaining]
                self.offset += remaining
        else:
            # Ring mode — vectorized scatter instead of per-token loop
            ring_start = self.sink
            ring_len = self.window
            positions = ring_start + (mx.arange(T_new) + (self.offset - self.capacity)) % ring_len
            self._keys[:, :, positions] = keys[:, :, :T_new]
            self._values[:, :, positions] = values[:, :, :T_new]
            self.offset += T_new

        # Return valid portion
        valid = min(self.offset, self.capacity)
        return self._keys[:, :, :valid], self._values[:, :, :valid]

    @property
    def state(self):
        valid = min(self.offset, self.capacity)
        return (self._keys[:, :, :valid], self._values[:, :, :valid])


class DuoKVCache:
    """Two-storage-class KV cache per the DuoAttention policy.

    For each KV head, allocates either:
      - QuantizedKVCache (retrieval heads: full context, 3-bit)
      - StreamingKVCache (streaming heads: sink + window, fp16)

    The attention layer sees a single cache object and calls
    update_and_fetch normally. DuoKVCache splits the heads,
    dispatches to the appropriate sub-cache, and concatenates
    the results.
    """

    def __init__(
        self,
        policy: dict,
        layer_idx: int,
        n_kv_heads: int = 4,
        bits: int = 3,
        group_size: int = 64,
        window: int = 256,
        sink: int = 4,
        quantize_retrieval: bool = False,
    ):
        self.layer_idx = layer_idx
        self.n_kv_heads = n_kv_heads
        # NOTE: do NOT set self.bits — mlx-lm SDPA checks hasattr(cache, 'bits')
        # and routes to quantized_matmul which is incompatible with fp16 KV.
        self._quant_bits = bits  # stored for QuantizedKVCache retrieval heads
        self.group_size = group_size
        self._quantize_retrieval = quantize_retrieval
        self.window = window
        self.sink = sink
        self.offset = 0

        # Classify each KV head
        lookup = policy.get("_lookup", {})
        layer_policy = lookup.get(layer_idx, {})

        # Map Q heads to KV heads
        n_q_heads = policy.get("n_heads", 32)
        gqa = n_q_heads // n_kv_heads

        self.head_types = []  # "retrieval" or "streaming" per KV head
        for kv_h in range(n_kv_heads):
            # A KV head is streaming if ALL its Q heads are streaming
            q_heads = range(kv_h * gqa, (kv_h + 1) * gqa)
            all_streaming = all(
                layer_policy.get(qh, "retrieval") == "streaming"
                for qh in q_heads
            )
            self.head_types.append("streaming" if all_streaming else "retrieval")

        self.capacity = sink + window  # streaming heads' max KV length
        self._is_streaming = [t == "streaming" for t in self.head_types]
        self._n_streaming = sum(self._is_streaming)
        self._step = 256  # pre-allocation headroom

        if quantize_retrieval:
            # Split architecture: one shared QuantizedKVCache for all
            # retrieval heads, one shared StreamingKVCache for all streaming
            # heads. Two update_and_fetch calls instead of per-head loops.
            self._retrieval_head_indices = [h for h in range(n_kv_heads) if not self._is_streaming[h]]
            self._streaming_head_indices = [h for h in range(n_kv_heads) if self._is_streaming[h]]
            # One shared TurboQuantKVCache for ALL retrieval heads
            # TQ3: WHT decorrelation + fused decode_attention (no dequant needed)
            TQCache = _get_tq_cache_class()
            self._retrieval_cache = TQCache(
                bits=bits, seed=42,
                min_quant_tokens=256) if self._retrieval_head_indices else None
            # Per-head StreamingKVCache (ring buffer needs per-head offset)
            self._streaming_head_caches = {
                h: StreamingKVCache(window, sink) for h in self._streaming_head_indices
            }
            # Legacy compat
            self._retrieval_caches = True
            self._streaming_caches = True
            self._keys = None
            self._values = None
            self._kv_len = 0
        else:
            # Original: single fp16 buffer for all heads, streaming trim
            self._retrieval_caches = None
            self._streaming_caches = None
            self._keys: Optional[mx.array] = None
            self._values: Optional[mx.array] = None
            self._kv_len = 0

        n_streaming = sum(1 for t in self.head_types if t == "streaming")
        mode = "quantized-retrieval" if quantize_retrieval else "fp16-all"
        logger.debug(f"Layer {layer_idx}: {n_streaming}/{n_kv_heads} streaming KV heads ({mode})")

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new KV, trim streaming heads to sink + window.

        When quantize_retrieval=True: dispatches per-head to QuantizedKVCache
        (retrieval) or StreamingKVCache (streaming), then merges dequantized
        results. Memory savings: 8x for retrieval heads at rest.

        When quantize_retrieval=False (default): uses pre-allocated fp16 slab
        with headroom. Slice assignment is O(1) vs concat's O(context) copy.
        """
        B, H_kv, T_new, D = keys.shape
        self.offset += T_new

        # Quantized retrieval mode: per-head dispatch
        if self._quantize_retrieval and self._retrieval_caches is not None:
            return self._update_quantized(keys, values, B, H_kv, T_new, D)

        if self._keys is None:
            # First call — pre-allocate with headroom
            alloc = T_new + self._step
            self._keys = mx.zeros((B, H_kv, alloc, D), dtype=keys.dtype)
            self._values = mx.zeros((B, H_kv, alloc, D), dtype=values.dtype)
            self._keys[:, :, :T_new] = keys
            self._values[:, :, :T_new] = values
            self._kv_len = T_new
        elif self._kv_len + T_new <= self._keys.shape[2]:
            # Fits in pre-allocated buffer — O(1) slice write
            self._keys[:, :, self._kv_len:self._kv_len + T_new] = keys
            self._values[:, :, self._kv_len:self._kv_len + T_new] = values
            self._kv_len += T_new
        else:
            # Buffer full — grow with new headroom (rare, every _step tokens)
            new_alloc = self._kv_len + T_new + self._step
            new_k = mx.zeros((B, H_kv, new_alloc, D), dtype=keys.dtype)
            new_v = mx.zeros((B, H_kv, new_alloc, D), dtype=values.dtype)
            new_k[:, :, :self._kv_len] = self._keys[:, :, :self._kv_len]
            new_v[:, :, :self._kv_len] = self._values[:, :, :self._kv_len]
            new_k[:, :, self._kv_len:self._kv_len + T_new] = keys
            new_v[:, :, self._kv_len:self._kv_len + T_new] = values
            self._keys = new_k
            self._values = new_v
            self._kv_len += T_new

        T_total = self._kv_len

        # Trim streaming heads if context exceeds capacity
        if T_total > self.capacity and self._n_streaming > 0:
            # Single gather replaces per-head Python loop (12 intermediates → 1 op)
            # Streaming: [0..sink-1, T-window..T-1, 0-padded to T_total]
            # Retrieval: [0..T_total-1]
            sink_idx = list(range(self.sink))
            window_idx = list(range(T_total - self.window, T_total))
            stream_idx = sink_idx + window_idx  # length = capacity
            # Pad to T_total: gather a harmless index, then zero-mask
            stream_padded = stream_idx + [0] * (T_total - len(stream_idx))
            retrieval_idx = list(range(T_total))

            # Build (H_kv, T_total) gather index
            gather = [stream_padded if self._is_streaming[h] else retrieval_idx
                      for h in range(H_kv)]
            g = mx.array(gather)[None, :, :, None]  # (1, H_kv, T_total, 1)
            g = mx.broadcast_to(g, (B, H_kv, T_total, D))

            out_k = mx.take_along_axis(self._keys[:, :, :T_total, :], g, axis=2)
            out_v = mx.take_along_axis(self._values[:, :, :T_total, :], g, axis=2)

            # Zero-mask padded positions for streaming heads
            if len(stream_idx) < T_total:
                mask = mx.ones((H_kv, T_total), dtype=mx.bool_)
                for h in range(H_kv):
                    if self._is_streaming[h]:
                        mask[h, len(stream_idx):] = False
                out_k = mx.where(mask[None, :, :, None], out_k, mx.zeros_like(out_k))
                out_v = mx.where(mask[None, :, :, None], out_v, mx.zeros_like(out_v))

            return out_k, out_v

        return self._keys[:, :, :T_total], self._values[:, :, :T_total]

    def _update_quantized(self, keys, values, B, H_kv, T_new, D):
        """Dispatch to TQ3 (retrieval) and StreamingKV (streaming).

        TQ3 handles quantization internally. Returns fp16 K/V for the
        standard SDPA path during prefill. During decode (T_new=1),
        compute_attention uses TQ3's fused decode_attention instead.
        """
        ret_idx = self._retrieval_head_indices
        str_idx = self._streaming_head_indices

        # Update retrieval heads via TurboQuantKVCache
        ret_k, ret_v = None, None
        if ret_idx and self._retrieval_cache is not None:
            k_ret = keys[:, ret_idx, :, :]
            v_ret = values[:, ret_idx, :, :]
            ret_k, ret_v = self._retrieval_cache.update_and_fetch(k_ret, v_ret)
            # TQ3 returns fp16 during warmup, quantized state after.
            # During prefill we need fp16 for standard SDPA.
            if isinstance(ret_k, tuple):
                # Quantized — dequantize for prefill SDPA
                # (decode uses compute_attention which avoids this)
                _dq = self._retrieval_cache._codec.dequantize_fused if (
                    hasattr(self._retrieval_cache, '_codec') and
                    self._retrieval_cache._codec is not None and
                    hasattr(self._retrieval_cache._codec, 'dequantize_fused')
                ) else None
                if _dq:
                    ret_k = _dq(
                        self._retrieval_cache._k_norms[:, :, :self._retrieval_cache.offset],
                        self._retrieval_cache._k_packed[:, :, :self._retrieval_cache.offset])
                    ret_v = _dq(
                        self._retrieval_cache._v_norms[:, :, :self._retrieval_cache.offset],
                        self._retrieval_cache._v_packed[:, :, :self._retrieval_cache.offset])
                else:
                    # Fallback — should not normally hit this
                    ret_k = ret_k if not isinstance(ret_k, tuple) else keys[:, ret_idx, :, :]
                    ret_v = ret_v if not isinstance(ret_v, tuple) else values[:, ret_idx, :, :]

        # Update streaming heads (per-head ring buffers)
        str_k, str_v = None, None
        if str_idx and self._streaming_head_caches:
            str_k_heads = []
            str_v_heads = []
            for i, h in enumerate(str_idx):
                k_h = keys[:, h:h+1, :, :]
                v_h = values[:, h:h+1, :, :]
                sk, sv = self._streaming_head_caches[h].update_and_fetch(k_h, v_h)
                str_k_heads.append(sk)
                str_v_heads.append(sv)
            if str_k_heads:
                str_k = mx.concatenate(str_k_heads, axis=1)
                str_v = mx.concatenate(str_v_heads, axis=1)

        # Merge for standard SDPA (prefill path)
        if ret_k is not None and str_k is not None:
            max_len = max(ret_k.shape[2], str_k.shape[2])
            if ret_k.shape[2] < max_len:
                pad = max_len - ret_k.shape[2]
                ret_k = mx.concatenate([ret_k, mx.zeros((B, len(ret_idx), pad, D), dtype=ret_k.dtype)], axis=2)
                ret_v = mx.concatenate([ret_v, mx.zeros((B, len(ret_idx), pad, D), dtype=ret_v.dtype)], axis=2)
            if str_k.shape[2] < max_len:
                pad = max_len - str_k.shape[2]
                str_k = mx.concatenate([str_k, mx.zeros((B, len(str_idx), pad, D), dtype=str_k.dtype)], axis=2)
                str_v = mx.concatenate([str_v, mx.zeros((B, len(str_idx), pad, D), dtype=str_v.dtype)], axis=2)
            out_k = mx.zeros((B, H_kv, max_len, D), dtype=keys.dtype)
            out_v = mx.zeros((B, H_kv, max_len, D), dtype=values.dtype)
            out_k[:, ret_idx] = ret_k
            out_v[:, ret_idx] = ret_v
            out_k[:, str_idx] = str_k
            out_v[:, str_idx] = str_v
            return out_k, out_v
        elif ret_k is not None:
            return ret_k, ret_v
        elif str_k is not None:
            return str_k, str_v
        else:
            return keys, values

    def compute_attention(self, queries, keys_out, values_out, scale, mask):
        """Split-SDPA: quantized attention for retrieval, fp16 for streaming.

        When quantize_retrieval=True, runs two separate SDPA calls to avoid
        dequantizing retrieval heads on every decode step.

        Args:
            queries: (B, H_q, T_q, D)
            keys_out, values_out: merged K/V from update_and_fetch
            scale: attention scale factor
            mask: causal mask

        Returns:
            (B, H_q, T_q, D) attention output
        """
        if not self._quantize_retrieval or self._retrieval_cache is None:
            # Default: standard fp16 SDPA on merged K/V
            return mx.fast.scaled_dot_product_attention(
                queries, keys_out, values_out, scale=scale, mask=mask)

        B, H_q, T_q, D = queries.shape

        # Only use split-SDPA during decode (T_q=1). During prefill,
        # use standard SDPA on the merged fp16 K/V from update_and_fetch.
        if T_q > 1:
            return mx.fast.scaled_dot_product_attention(
                queries, keys_out, values_out, scale=scale, mask=mask)
        gqa = H_q // self.n_kv_heads

        # Split Q heads by type
        ret_idx = self._retrieval_head_indices
        str_idx = self._streaming_head_indices

        # Map KV head indices to Q head indices
        ret_q_idx = []
        for kv_h in ret_idx:
            ret_q_idx.extend(range(kv_h * gqa, (kv_h + 1) * gqa))
        str_q_idx = []
        for kv_h in str_idx:
            str_q_idx.extend(range(kv_h * gqa, (kv_h + 1) * gqa))

        out = mx.zeros_like(queries)

        # Retrieval: TQ3 fused decode attention (no dequant needed)
        if ret_idx and self._retrieval_cache is not None:
            rc = self._retrieval_cache
            Q_ret = queries[:, ret_q_idx, :, :]
            if (hasattr(rc, 'decode_attention') and rc._quantized
                    and rc._k_norms is not None):
                # Quantized — use TQ3's fused attention (no dequant)
                ret_out = rc.decode_attention(Q_ret, scale=scale, mask=mask)
                out[:, ret_q_idx] = ret_out
            elif rc._fp16_keys is not None:
                # Still in fp16 warmup — use standard SDPA
                ret_out = mx.fast.scaled_dot_product_attention(
                    Q_ret, rc._fp16_keys, rc._fp16_values,
                    scale=scale, mask=mask)
                out[:, ret_q_idx] = ret_out
            else:
                # Fallback: use merged K/V from update_and_fetch
                K_ret = keys_out[:, ret_idx, :, :]
                V_ret = values_out[:, ret_idx, :, :]
                ret_out = mx.fast.scaled_dot_product_attention(
                    Q_ret, K_ret, V_ret, scale=scale, mask=mask)
                out[:, ret_q_idx] = ret_out

        # Streaming: fp16 SDPA
        if str_idx and self._streaming_head_caches:
            Q_str = queries[:, str_q_idx, :, :]
            # Gather streaming K/V from the merged output (already fp16)
            K_str = keys_out[:, str_idx, :, :]
            V_str = values_out[:, str_idx, :, :]
            str_out = mx.fast.scaled_dot_product_attention(
                Q_str, K_str, V_str, scale=scale, mask=mask)
            out[:, str_q_idx] = str_out

        return out

    @property
    def state(self):
        if self._keys is None:
            return None, None
        return self._keys[:, :, :self._kv_len], self._values[:, :, :self._kv_len]

    @state.setter
    def state(self, v):
        """Set state — needed for SnapKV compact_cache compatibility."""
        keys, values = v
        if keys is not None:
            self._kv_len = keys.shape[2]
            self.offset = self._kv_len
            # Pre-allocate with headroom
            alloc = self._kv_len + self._step
            B, H, _, D = keys.shape
            self._keys = mx.zeros((B, H, alloc, D), dtype=keys.dtype)
            self._values = mx.zeros((B, H, alloc, D), dtype=values.dtype)
            self._keys[:, :, :self._kv_len] = keys
            self._values[:, :, :self._kv_len] = values
        else:
            self._keys = None
            self._values = None
            self._kv_len = 0
            self.offset = 0
