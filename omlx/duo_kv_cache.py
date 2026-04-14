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
    logger.info(f"DuoAttention policy: {streaming_pct:.0f}% streaming, "
                f"window={policy.get('window', 256)}, sink={policy.get('sink', 4)}")
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


class DuoKVCache(_BaseCache):
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
        self.bits = bits
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

        # Create sub-caches
        self.sub_caches = []
        for ht in self.head_types:
            if ht == "streaming":
                self.sub_caches.append(StreamingKVCache(window=window, sink=sink))
            else:
                self.sub_caches.append(QuantizedKVCache(group_size=group_size, bits=bits))

        n_streaming = sum(1 for t in self.head_types if t == "streaming")
        logger.debug(f"Layer {layer_idx}: {n_streaming}/{n_kv_heads} streaming KV heads")

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Split keys/values by head, dispatch to sub-caches, recombine."""
        B, H_kv, T_new, D = keys.shape

        all_keys = []
        all_values = []

        for h in range(H_kv):
            k_h = keys[:, h:h+1, :, :]  # (B, 1, T_new, D)
            v_h = values[:, h:h+1, :, :]

            k_out, v_out = self.sub_caches[h].update_and_fetch(k_h, v_h)
            all_keys.append(k_out)
            all_values.append(v_out)

        self.offset += T_new

        # All sub-caches may return different sequence lengths
        # Pad to max length for concatenation
        max_len = max(k.shape[2] for k in all_keys)
        padded_keys = []
        padded_values = []
        for k, v in zip(all_keys, all_values):
            if k.shape[2] < max_len:
                pad = max_len - k.shape[2]
                k = mx.concatenate([k, mx.zeros((B, 1, pad, D), dtype=k.dtype)], axis=2)
                v = mx.concatenate([v, mx.zeros((B, 1, pad, D), dtype=v.dtype)], axis=2)
            padded_keys.append(k)
            padded_values.append(v)

        return mx.concatenate(padded_keys, axis=1), mx.concatenate(padded_values, axis=1)

    @property
    def state(self):
        """Return concatenated state from all sub-caches."""
        all_k = []
        all_v = []
        for sc in self.sub_caches:
            k, v = sc.state
            all_k.append(k)
            all_v.append(v)
        max_len = max(k.shape[2] for k in all_k)
        B = all_k[0].shape[0]
        D = all_k[0].shape[3]
        padded_k = []
        padded_v = []
        for k, v in zip(all_k, all_v):
            if k.shape[2] < max_len:
                pad = max_len - k.shape[2]
                k = mx.concatenate([k, mx.zeros((B, 1, pad, D), dtype=k.dtype)], axis=2)
                v = mx.concatenate([v, mx.zeros((B, 1, pad, D), dtype=v.dtype)], axis=2)
            padded_k.append(k)
            padded_v.append(v)
        return mx.concatenate(padded_k, axis=1), mx.concatenate(padded_v, axis=1)
