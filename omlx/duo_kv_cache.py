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
            # Split architecture: QuantizedKVCache per retrieval head,
            # StreamingKVCache per streaming head. Saves ~8x memory for
            # retrieval heads (3-bit vs fp16), enabling 1M context.
            self._retrieval_caches = {}  # head_idx → QuantizedKVCache
            self._streaming_caches = {}  # head_idx → StreamingKVCache
            for h in range(n_kv_heads):
                if self._is_streaming[h]:
                    self._streaming_caches[h] = StreamingKVCache(window, sink)
                else:
                    self._retrieval_caches[h] = QuantizedKVCache(
                        bits=bits, group_size=group_size)
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
        """Per-head dispatch for quantized retrieval mode.

        Retrieval heads → QuantizedKVCache (3-bit, full context)
        Streaming heads → StreamingKVCache (fp16, ring buffer)

        Returns merged fp16 K/V for SDPA. The memory savings come from
        storage (retrieval heads are 8x smaller at rest), not from avoiding
        dequantization during attention.
        """
        out_k_heads = []
        out_v_heads = []
        max_len = 0

        for h in range(H_kv):
            k_h = keys[:, h:h+1, :, :]   # (B, 1, T_new, D)
            v_h = values[:, h:h+1, :, :]

            if h in self._streaming_caches:
                k_out, v_out = self._streaming_caches[h].update_and_fetch(k_h, v_h)
            elif h in self._retrieval_caches:
                k_out, v_out = self._retrieval_caches[h].update_and_fetch(k_h, v_h)
                # QuantizedKVCache returns quantized tuples — dequantize for SDPA
                if isinstance(k_out, tuple):
                    k_out = mx.dequantize(
                        *k_out, group_size=self._retrieval_caches[h].group_size,
                        bits=self._retrieval_caches[h].bits)
                    v_out = mx.dequantize(
                        *v_out, group_size=self._retrieval_caches[h].group_size,
                        bits=self._retrieval_caches[h].bits)
            else:
                continue

            out_k_heads.append(k_out)
            out_v_heads.append(v_out)
            max_len = max(max_len, k_out.shape[2])

        # Pad all heads to same length and concatenate
        padded_k = []
        padded_v = []
        for k, v in zip(out_k_heads, out_v_heads):
            if k.shape[2] < max_len:
                pad = max_len - k.shape[2]
                k = mx.concatenate([k, mx.zeros((B, 1, pad, D), dtype=k.dtype)], axis=2)
                v = mx.concatenate([v, mx.zeros((B, 1, pad, D), dtype=v.dtype)], axis=2)
            padded_k.append(k)
            padded_v.append(v)

        return mx.concatenate(padded_k, axis=1), mx.concatenate(padded_v, axis=1)

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
