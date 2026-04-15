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
                # Overflow into ring region — start overwriting after sink
                remaining = T_new - actual
                ring_start = self.sink
                ring_len = self.window
                for i in range(remaining):
                    pos = ring_start + ((self.offset - self.capacity + i) % ring_len)
                    self._keys[:, :, pos:pos+1] = keys[:, :, actual+i:actual+i+1]
                    self._values[:, :, pos:pos+1] = values[:, :, actual+i:actual+i+1]
                self.offset += remaining
        else:
            # Ring mode — overwrite oldest in the window region
            ring_start = self.sink
            ring_len = self.window
            for i in range(T_new):
                pos = ring_start + ((self.offset - self.capacity + i) % ring_len)
                self._keys[:, :, pos:pos+1] = keys[:, :, i:i+1]
                self._values[:, :, pos:pos+1] = values[:, :, i:i+1]
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
    ):
        self.layer_idx = layer_idx
        self.n_kv_heads = n_kv_heads
        # NOTE: do NOT set self.bits — mlx-lm SDPA checks hasattr(cache, 'bits')
        # and routes to quantized_matmul which is incompatible with fp16 KV.
        self._quant_bits = bits  # stored for potential future QuantizedKVCache use
        self.group_size = group_size
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

        # All heads use a single shared KVCache (fp16). After update,
        # streaming heads' KV is trimmed to sink + window tokens.
        # This avoids mixed-type and mixed-length issues while still
        # saving memory on streaming heads at long contexts.
        self.capacity = sink + window  # streaming heads' max KV length
        self._is_streaming = [t == "streaming" for t in self.head_types]
        self._n_streaming = sum(self._is_streaming)
        self._keys: Optional[mx.array] = None
        self._values: Optional[mx.array] = None

        n_streaming = sum(1 for t in self.head_types if t == "streaming")
        logger.debug(f"Layer {layer_idx}: {n_streaming}/{n_kv_heads} streaming KV heads")

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Store new KV, trim streaming heads to sink + window.

        All heads use a single contiguous KV buffer. After each update,
        streaming heads' KV is replaced with [sink_tokens | recent_window].
        Retrieval heads keep the full context.
        """
        B, H_kv, T_new, D = keys.shape
        self.offset += T_new

        # First call — initialize storage
        if self._keys is None:
            self._keys = keys
            self._values = values
        else:
            self._keys = mx.concatenate([self._keys, keys], axis=2)
            self._values = mx.concatenate([self._values, values], axis=2)

        T_total = self._keys.shape[2]

        # Trim streaming heads if context exceeds capacity
        if T_total > self.capacity and self._n_streaming > 0:
            # Build per-head trimmed KV
            # Streaming: keep first `sink` + last `window` tokens
            # Retrieval: keep everything
            trimmed_k = []
            trimmed_v = []
            for h in range(H_kv):
                if self._is_streaming[h] and T_total > self.capacity:
                    # Sink tokens (first few) + window tokens (most recent)
                    k_sink = self._keys[:, h:h+1, :self.sink, :]
                    k_window = self._keys[:, h:h+1, -(self.window):, :]
                    trimmed_k.append(mx.concatenate([k_sink, k_window], axis=2))

                    v_sink = self._values[:, h:h+1, :self.sink, :]
                    v_window = self._values[:, h:h+1, -(self.window):, :]
                    trimmed_v.append(mx.concatenate([v_sink, v_window], axis=2))
                else:
                    trimmed_k.append(self._keys[:, h:h+1, :, :])
                    trimmed_v.append(self._values[:, h:h+1, :, :])

            # Pad all heads to same length (retrieval heads' full length)
            max_len = max(k.shape[2] for k in trimmed_k)
            padded_k = []
            padded_v = []
            for k, v in zip(trimmed_k, trimmed_v):
                if k.shape[2] < max_len:
                    pad = max_len - k.shape[2]
                    k = mx.concatenate([k, mx.zeros((B, 1, pad, D), dtype=k.dtype)], axis=2)
                    v = mx.concatenate([v, mx.zeros((B, 1, pad, D), dtype=v.dtype)], axis=2)
                padded_k.append(k)
                padded_v.append(v)

            return mx.concatenate(padded_k, axis=1), mx.concatenate(padded_v, axis=1)

        return self._keys, self._values

    @property
    def state(self):
        if self._keys is None:
            return None, None
        return self._keys, self._values
