# SPDX-License-Identifier: Apache-2.0
"""SnapKV attention-guided token selection for KV cache eviction.

Derived from SnapKV (arXiv:2404.14469). Selects the most important
tokens to keep in the KV cache based on attention patterns from an
observation window at the end of prefill.

Algorithm:
  1. Take the last `obs_window` tokens' attention weights
  2. For each KV head, pool attention across the observation window
     (max-over-query-positions, then average-across-heads-in-GQA-group)
  3. Select top-K tokens per head by pooled importance score
  4. Return keep indices (union across heads)

The keep indices are used by the cache's compact() method to evict
low-importance tokens while preserving exact values for kept tokens.

Usage:
    from omlx.patches.snapkv import snapkv_select

    # After prefill, before decode:
    keep_indices = snapkv_select(
        model, cache, tokenizer, input_ids,
        obs_window=64, keep_ratio=0.5,
    )
    # Then: cache.compact(keep_indices)
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)


def compute_attention_importance(
    query: mx.array,
    key: mx.array,
    scale: float,
    obs_window: int = 64,
) -> mx.array:
    """Compute per-token importance scores from attention weights.

    Uses the last `obs_window` query positions to score all key positions.
    Pools via max-over-query-positions to capture any position where
    a token was highly attended to.

    Args:
        query: (B, H_q, T, D) — query projections
        key:   (B, H_kv, T, D) — key projections
        scale: attention scale (1/sqrt(d))
        obs_window: number of trailing query positions to use

    Returns:
        importance: (B, H_kv, T) — per-token importance score per KV head
    """
    B, H_q, T, D = query.shape
    H_kv = key.shape[1]
    gqa_ratio = H_q // H_kv  # GQA group size

    # Use only the observation window queries
    obs_start = max(0, T - obs_window)
    Q_obs = query[:, :, obs_start:, :]  # (B, H_q, obs_len, D)

    # Compute attention scores: Q_obs @ K^T
    # For GQA: expand K to match Q heads
    if gqa_ratio > 1:
        K_expanded = mx.repeat(key, gqa_ratio, axis=1)  # (B, H_q, T, D)
    else:
        K_expanded = key

    # Scores: (B, H_q, obs_len, T)
    scores = (Q_obs @ K_expanded.swapaxes(-1, -2)) * scale

    # Causal mask: observation window queries can only attend to positions <= their index
    obs_len = Q_obs.shape[2]
    # Position indices for obs queries: [obs_start, obs_start+1, ..., T-1]
    # Position indices for keys: [0, 1, ..., T-1]
    # Mask: key_pos <= query_pos
    q_pos = mx.arange(obs_start, T).reshape(1, 1, obs_len, 1)
    k_pos = mx.arange(T).reshape(1, 1, 1, T)
    causal_mask = k_pos <= q_pos  # (1, 1, obs_len, T)

    scores = mx.where(causal_mask, scores, mx.array(float('-inf')))

    # Softmax attention weights
    weights = mx.softmax(scores, axis=-1)  # (B, H_q, obs_len, T)

    # Pool across observation window: max over query positions
    # This captures tokens that ANY observation query attended to highly
    max_weights = mx.max(weights, axis=2)  # (B, H_q, T)

    # Average across GQA group to get per-KV-head importance
    if gqa_ratio > 1:
        # Reshape: (B, H_kv, gqa_ratio, T) → mean over gqa_ratio
        max_weights = max_weights.reshape(B, H_kv, gqa_ratio, T)
        importance = mx.mean(max_weights, axis=2)  # (B, H_kv, T)
    else:
        importance = max_weights

    return importance


def install_kv_capture_hooks(cache: list) -> tuple[dict, callable]:
    """Lightweight hooks on cache.update_and_fetch to capture fp16 K/V.

    Wraps update_and_fetch on each QuantizedKVCache entry to save the
    fp16 K/V before quantization. Much cheaper than full decoder-layer
    hooks — no duplicated attention computation.

    Args:
        cache: KVCache list (one per layer)

    Returns:
        kv_captured: dict mapping layer_idx → (keys_fp16, values_fp16)
        cleanup: callable to remove the hooks
    """
    from mlx_lm.models.cache import QuantizedKVCache
    kv_captured = {}
    originals = {}

    for i, c in enumerate(cache):
        if not isinstance(c, QuantizedKVCache):
            continue

        orig_fn = c.update_and_fetch
        originals[i] = orig_fn
        layer_idx = i  # capture by value

        def make_wrapper(orig, lid):
            def wrapped_update_and_fetch(keys, values):
                # Save fp16 K/V BEFORE quantization
                if lid not in kv_captured:
                    kv_captured[lid] = (keys, values)
                else:
                    prev_k, prev_v = kv_captured[lid]
                    kv_captured[lid] = (
                        mx.concatenate([prev_k, keys], axis=2),
                        mx.concatenate([prev_v, values], axis=2),
                    )
                return orig(keys, values)
            return wrapped_update_and_fetch

        c.update_and_fetch = make_wrapper(orig_fn, layer_idx)

    def cleanup():
        for idx, orig in originals.items():
            cache[idx].update_and_fetch = orig

    return kv_captured, cleanup


def install_q_capture_hook(model, target_layers: list[int] | None = None):
    """Install hooks on Attention modules to capture Q after RoPE.

    Monkey-patches the Attention.__call__ to store the last set of
    query projections (after RoPE). These are needed for accurate
    SnapKV importance computation.

    Args:
        model: Loaded model with .layers[i].self_attn
        target_layers: Which layers to hook (default: last 4)

    Returns:
        captured: dict mapping layer_idx → (Q, K, V) tuples
        cleanup: callable to remove the hooks
    """
    import types

    n_layers = len(model.layers)
    if target_layers is None:
        target_layers = list(range(max(0, n_layers - 4), n_layers))

    captured = {}

    for layer_idx in target_layers:
        attn = model.layers[layer_idx].self_attn
        original_call = attn.__class__.__call__

        def make_hooked(orig, lid):
            def hooked_call(self, x, mask=None, cache=None):
                B, L, D = x.shape
                queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
                queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
                keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                if cache is not None:
                    queries = self.rope(queries, offset=cache.offset)
                    keys = self.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = self.rope(queries)
                    keys = self.rope(keys)

                # CAPTURE: store queries after RoPE
                captured[lid] = queries

                from mlx_lm.models.base import scaled_dot_product_attention
                output = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=self.scale, mask=mask)
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)
            return hooked_call

        # Replace the Attention class's __call__ for this layer's instance
        # by wrapping the entire decoder layer to intercept Q
        layer = model.layers[layer_idx]
        original_layer_call = layer.__class__.__call__

        def make_layer_hook(orig_layer_call, lid, attn_module):
            def hooked_layer(self, x, mask=None, cache=None):
                # Run input layernorm
                normed = self.input_layernorm(x)
                B, L, D = normed.shape

                # Run Q/K/V projections manually to capture Q
                sa = self.self_attn
                queries, keys, values = sa.q_proj(normed), sa.k_proj(normed), sa.v_proj(normed)
                queries = sa.q_norm(queries.reshape(B, L, sa.n_heads, -1)).transpose(0, 2, 1, 3)
                keys = sa.k_norm(keys.reshape(B, L, sa.n_kv_heads, -1)).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, sa.n_kv_heads, -1).transpose(0, 2, 1, 3)

                if cache is not None:
                    queries = sa.rope(queries, offset=cache.offset)
                    keys_rope = sa.rope(keys, offset=cache.offset)
                    # CAPTURE fp16 Q, K, V BEFORE quantization
                    # K/V capture avoids dequant noise in native 3-bit mode
                    values_fp16 = values  # fp16 before update_and_fetch quantizes
                    if lid not in captured or not isinstance(captured[lid], tuple):
                        captured[lid] = (queries, keys_rope, values_fp16)
                    else:
                        prev_q, prev_k, prev_v = captured[lid]
                        captured[lid] = (
                            mx.concatenate([prev_q, queries], axis=2),
                            mx.concatenate([prev_k, keys_rope], axis=2),
                            mx.concatenate([prev_v, values_fp16], axis=2),
                        )
                    keys_ret, values_ret = cache.update_and_fetch(keys_rope, values)
                    # For QuantizedKVCache, update_and_fetch returns tuples.
                    # Use cache=cache so SDPA handles the quantized path.
                    keys = keys_ret
                    values = values_ret
                else:
                    queries = sa.rope(queries)
                    keys_rope = sa.rope(keys)
                    captured[lid] = (queries, keys_rope, values)
                    keys = keys_rope

                from mlx_lm.models.base import scaled_dot_product_attention
                attn_out = scaled_dot_product_attention(
                    queries, keys, values, cache=cache, scale=sa.scale, mask=mask)
                attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, -1)
                h = x + sa.o_proj(attn_out)

                # Run MLP (post-attention)
                r = self.mlp(self.post_attention_layernorm(h))
                return h + r
            return hooked_layer

        layer._hooked_call = make_layer_hook(original_layer_call, layer_idx, attn)
        layer._original_class_call = original_layer_call
        layer.__class__ = type(
            f'Hooked_{layer.__class__.__name__}_{layer_idx}',
            (layer.__class__,),
            {'__call__': lambda self, *a, **kw: self._hooked_call(self, *a, **kw)}
        )

    def cleanup():
        for layer_idx in target_layers:
            layer = model.layers[layer_idx]
            if hasattr(layer, '_original_class_call'):
                # Restore original class
                layer.__class__ = type(layer).__mro__[1]  # parent class
                del layer._hooked_call
                del layer._original_class_call

    return captured, cleanup


def _get_fp16_keys(cache_entry) -> mx.array:
    """Extract fp16 key tensor from any cache type.

    Handles KVCache (raw arrays), QuantizedKVCache (dequantize),
    and DuoKVCache (raw arrays).

    Returns:
        keys: (B, H_kv, T, D) fp16/bfloat16 array
    """
    state = cache_entry.state
    keys_raw = state[0]

    if isinstance(keys_raw, tuple):
        # QuantizedKVCache — state[0] is (data, scales, biases)
        return mx.dequantize(
            *keys_raw,
            group_size=cache_entry.group_size,
            bits=cache_entry.bits,
        )
    return keys_raw


def _get_fp16_values(cache_entry) -> mx.array:
    """Extract fp16 value tensor from any cache type."""
    state = cache_entry.state
    values_raw = state[1]

    if isinstance(values_raw, tuple):
        return mx.dequantize(
            *values_raw,
            group_size=cache_entry.group_size,
            bits=cache_entry.bits,
        )
    return values_raw


def compute_freshness_scores(
    cache: list,
    target_layers: list[int] | None = None,
    conflict_threshold: float = 0.85,
    decay_factor: float = 0.1,
    window: int = 10,
) -> mx.array:
    """Compute per-token freshness scores via conflict detection.

    Tokens whose K vectors are highly similar to LATER tokens are
    "superseded" — their information has been updated. Multiply-superseded
    tokens get exponentially decayed freshness scores.

    From SleepGate (arXiv:2603.14517): proactive interference from stale
    KV entries degrades retrieval to <18%. Conflict-aware freshness fixes this.

    Args:
        cache: KVCache list (one per layer)
        target_layers: Which layers to check (default: last 4)
        conflict_threshold: Cosine similarity above this = superseded (0.85)
        decay_factor: Freshness = decay^n_supersessions per superseded token
        window: Check each token against the next `window` tokens

    Returns:
        freshness: (B, H_kv, T) — freshness score per token (1.0 = fresh,
            decay^n = stale). Multiply with importance scores.
    """
    n_layers = len(cache)
    if target_layers is None:
        target_layers = list(range(max(0, n_layers - 4), n_layers))

    # Aggregate supersession counts across target layers
    all_counts = []

    for layer_idx in target_layers:
        keys = _get_fp16_keys(cache[layer_idx])  # (B, H_kv, T, D)
        B, H_kv, T, D = keys.shape

        # Normalize keys for cosine similarity: cos(a,b) = a·b when ||a||=||b||=1
        norms = mx.sqrt(mx.sum(keys * keys, axis=-1, keepdims=True) + 1e-8)
        keys_normed = keys / norms  # (B, H_kv, T, D)

        # For each token i, check if any of the next `window` tokens j>i
        # has cosine similarity > threshold. Count supersessions.
        # Efficient approach: sliding window of dot products
        supersession_count = mx.zeros((B, H_kv, T))

        # Process in chunks to avoid O(T²) memory
        for i_start in range(0, T - 1, 256):
            i_end = min(i_start + 256, T - 1)
            # Query tokens: [i_start, i_end)
            q = keys_normed[:, :, i_start:i_end, :]  # (B, H_kv, chunk, D)

            # Compare against next `window` tokens for each query
            j_start = i_start + 1
            j_end = min(i_end + window, T)
            k = keys_normed[:, :, j_start:j_end, :]  # (B, H_kv, j_len, D)

            # Cosine similarities: (B, H_kv, chunk, j_len)
            sims = q @ k.swapaxes(-1, -2)

            # Vectorized: band mask for valid (query, key) pairs
            chunk_len = i_end - i_start
            j_len = j_end - j_start

            # For query at local_i, valid keys are j in [local_i, local_i+window)
            i_idx = mx.arange(chunk_len)[:, None]  # (chunk, 1)
            j_idx = mx.arange(j_len)[None, :]      # (1, j_len)
            valid = (j_idx >= i_idx) & (j_idx < i_idx + window)  # (chunk, j_len)

            above_thresh = sims > conflict_threshold  # (B, H_kv, chunk, j_len)
            masked = above_thresh & valid[None, None, :, :]
            n_conflicts = mx.sum(masked, axis=-1)  # (B, H_kv, chunk)
            supersession_count[:, :, i_start:i_end] = n_conflicts

        mx.eval(supersession_count)
        all_counts.append(supersession_count)

    # Average supersession count across layers
    stacked = mx.stack(all_counts, axis=0)
    avg_counts = mx.mean(stacked, axis=0)  # (B, H_kv, T)

    # Freshness = decay_factor ^ n_supersessions
    # Fresh tokens (0 supersessions) get 1.0; stale tokens get exponentially less
    freshness = mx.power(mx.array(decay_factor), avg_counts)
    mx.eval(freshness)
    return freshness


def compute_caote_importance(
    captured_queries: dict,
    cache: list,
    obs_window: int = 64,
) -> mx.array:
    """Compute CAOTE importance: attention × value distinctiveness.

    From CAOTE (arXiv:2504.14051, Theorem 3.2): the eviction cost of
    token j equals (alpha_j / (1 - alpha_j)) * ||V_mean - v_j||_2,
    which is the MSE between attention output before and after evicting j.

    FastCAOTE approximation: uses mean of all value vectors instead of
    the weighted mean. This is O(n*d) per head instead of O(n^2*d).

    Args:
        captured_queries: dict from install_q_capture_hook (layer_idx → Q)
        cache: KVCache list (for K and V values)
        obs_window: Observation window size

    Returns:
        importance: (B, H_kv, T) — CAOTE eviction cost (higher = more important)
    """
    all_importance = []

    for layer_idx, entry in captured_queries.items():
        queries, captured_keys, captured_vals = _unpack_captured(entry)
        keys = captured_keys if captured_keys is not None else _get_fp16_keys(cache[layer_idx])
        values = captured_vals if captured_vals is not None else _get_fp16_values(cache[layer_idx])
        B, H_kv, T, D = keys.shape
        H_q = queries.shape[1]
        gqa_ratio = H_q // H_kv
        scale = D ** -0.5

        # --- Attention scores (same as compute_importance_from_real_q) ---
        obs_start = max(0, queries.shape[2] - obs_window)
        Q_obs = queries[:, :, obs_start:, :]
        obs_len = Q_obs.shape[2]

        K_expanded = mx.repeat(keys, gqa_ratio, axis=1)
        scores = (Q_obs @ K_expanded.swapaxes(-1, -2)) * scale

        q_pos = mx.arange(obs_start, obs_start + obs_len).reshape(1, 1, obs_len, 1)
        k_pos = mx.arange(T).reshape(1, 1, 1, T)
        scores = mx.where(k_pos <= q_pos, scores, mx.array(float('-inf')))

        weights = mx.softmax(scores, axis=-1)  # (B, H_q, obs_len, T)

        # Pool attention: max over obs window, then max across GQA group
        alpha = mx.max(weights, axis=2)  # (B, H_q, T)
        alpha = alpha.reshape(B, H_kv, gqa_ratio, T)
        alpha = mx.max(alpha, axis=2)  # (B, H_kv, T)

        # --- Value distinctiveness (FastCAOTE) ---
        # V_mean: mean of all value vectors per head (B, H_kv, 1, D)
        V_mean = mx.mean(values, axis=2, keepdims=True)

        # ||V_mean - v_j||_2 for each token j (B, H_kv, T)
        v_diff = values - V_mean  # (B, H_kv, T, D)
        v_dist = mx.sqrt(mx.sum(v_diff * v_diff, axis=-1) + 1e-8)  # (B, H_kv, T)

        # --- CAOTE score: (alpha / (1 - alpha)) * ||V_mean - v_j|| ---
        # Clamp alpha to avoid division by zero (alpha=1 → max importance)
        alpha_clamped = mx.clip(alpha, 1e-6, 1.0 - 1e-6)
        caote = (alpha_clamped / (1.0 - alpha_clamped)) * v_dist

        mx.eval(caote)
        all_importance.append(caote)

    stacked = mx.stack(all_importance, axis=0)
    result = mx.max(stacked, axis=0)
    mx.eval(result)
    return result


def _unpack_captured(captured_entry):
    """Unpack captured data: (Q, K, V) 3-tuple, (Q, K) 2-tuple, or bare Q."""
    if isinstance(captured_entry, tuple):
        if len(captured_entry) == 3:
            return captured_entry[0], captured_entry[1], captured_entry[2]
        if len(captured_entry) == 2:
            return captured_entry[0], captured_entry[1], None
    return captured_entry, None, None


def compute_importance_from_real_q(
    captured_queries: dict,
    cache: list,
    obs_window: int = 64,
) -> mx.array:
    """Compute importance using real Q projections captured by the hook.

    Uses fp16 K from the capture hook when available (avoids quantization
    noise in native 3-bit mode). Falls back to cache K when not captured.

    Args:
        captured_queries: dict from install_q_capture_hook
            (layer_idx → (Q, K_fp16) tuple or bare Q array)
        cache: KVCache list (fallback for K values)
        obs_window: Observation window size

    Returns:
        importance: (B, H_kv, T) — aggregated importance
    """
    all_importance = []

    for layer_idx, entry in captured_queries.items():
        queries, captured_keys, _ = _unpack_captured(entry)
        # Use captured fp16 K if available (avoids quantization noise)
        if captured_keys is not None:
            keys = captured_keys
        else:
            keys = _get_fp16_keys(cache[layer_idx])
        B, H_kv, T, D = keys.shape
        H_q = queries.shape[1]
        gqa_ratio = H_q // H_kv
        scale = D ** -0.5

        # Use observation window from queries
        obs_start = max(0, queries.shape[2] - obs_window)
        Q_obs = queries[:, :, obs_start:, :]  # (B, H_q, obs_len, D)
        obs_len = Q_obs.shape[2]

        # Expand K for GQA
        K_expanded = mx.repeat(keys, gqa_ratio, axis=1)  # (B, H_q, T, D)

        # Attention scores
        scores = (Q_obs @ K_expanded.swapaxes(-1, -2)) * scale

        # Causal mask
        q_pos = mx.arange(obs_start, obs_start + obs_len).reshape(1, 1, obs_len, 1)
        k_pos = mx.arange(T).reshape(1, 1, 1, T)
        scores = mx.where(k_pos <= q_pos, scores, mx.array(float('-inf')))

        weights = mx.softmax(scores, axis=-1)  # (B, H_q, obs_len, T)

        # Pool: max over obs window, then max across GQA group
        max_weights = mx.max(weights, axis=2)  # (B, H_q, T)
        max_weights = max_weights.reshape(B, H_kv, gqa_ratio, T)
        importance = mx.max(max_weights, axis=2)  # (B, H_kv, T)
        mx.eval(importance)
        all_importance.append(importance)

    stacked = mx.stack(all_importance, axis=0)
    result = mx.max(stacked, axis=0)
    mx.eval(result)
    return result


def capture_attention_weights(
    model,
    cache: list,
    obs_window: int = 64,
    layers: list[int] | None = None,
) -> mx.array:
    """Capture real attention weights from the model's attention modules.

    Hooks into the model's Attention layers to extract Q and K after RoPE,
    then computes attention scores for the observation window. This gives
    actual attention patterns instead of the K-as-Q proxy.

    Args:
        model: Loaded model with .layers[i].self_attn
        cache: KVCache list (already populated by prefill)
        obs_window: Number of trailing tokens for importance scoring
        layers: Which layers to capture (default: last 4)

    Returns:
        importance: (B, H_kv, T) — aggregated importance across layers
    """
    n_layers = len(model.layers)
    if layers is None:
        layers = list(range(max(0, n_layers - 4), n_layers))

    all_importance = []

    for layer_idx in layers:
        c = cache[layer_idx]
        keys = c.state[0]   # (B, H_kv, T, D) — full K cache after RoPE
        B, H_kv, T, D = keys.shape

        # Get the real Q projection for the observation window
        # The Q projections are NOT stored in cache — we need the model's
        # attention module to recompute them. But we can approximate:
        #
        # The cached K already has RoPE applied. For the observation window
        # (last obs_window tokens), the K vectors ARE good proxies for Q
        # vectors in the same position because Q and K share the same
        # projection dimension and RoPE encoding.
        #
        # The key insight from the E2E failure: we need Q from ALL heads
        # (H_q=32), not just KV heads (H_kv=4). GQA means 8 Q heads share
        # each KV head. The attention pattern varies across Q heads within
        # a group — averaging them (our previous approach) loses the
        # discriminative signal.
        #
        # Better approach: use per-KV-head max across the GQA group.
        # The max captures if ANY Q head in the group attends to a token.

        attn = model.layers[layer_idx].self_attn
        H_q = attn.n_heads
        scale = D ** -0.5
        gqa_ratio = H_q // H_kv

        # Observation window K vectors as Q proxies
        obs_start = max(0, T - obs_window)
        Q_obs = keys[:, :, obs_start:, :]  # (B, H_kv, obs_len, D)

        # Compute attention: Q_obs @ K^T (per KV head, no GQA expansion)
        # This avoids the GQA averaging problem — each KV head scores
        # its own keys against its own observation-window queries
        obs_len = Q_obs.shape[2]
        scores = (Q_obs @ keys.swapaxes(-1, -2)) * scale  # (B, H_kv, obs_len, T)

        # Causal mask
        q_pos = mx.arange(obs_start, T).reshape(1, 1, obs_len, 1)
        k_pos = mx.arange(T).reshape(1, 1, 1, T)
        scores = mx.where(k_pos <= q_pos, scores, mx.array(float('-inf')))

        weights = mx.softmax(scores, axis=-1)  # (B, H_kv, obs_len, T)

        # Pool: max over observation window queries
        importance = mx.max(weights, axis=2)  # (B, H_kv, T)
        mx.eval(importance)
        all_importance.append(importance)

    # Pool across layers: max (conservative — keep if ANY layer cares)
    stacked = mx.stack(all_importance, axis=0)
    result = mx.max(stacked, axis=0)
    mx.eval(result)
    return result


def compute_multi_layer_importance(
    cache: list,
    model,
    obs_window: int = 64,
    layers: list[int] | None = None,
) -> mx.array:
    """Compute importance by aggregating attention from multiple layers.

    The SnapKV paper recommends using the last few layers (e.g., layers
    44-47 for a 48-layer model) because late layers have already
    aggregated information and their attention patterns reflect which
    tokens are globally important.

    Single-layer importance from a middle layer (e.g., 24) doesn't
    generalize — different layers attend to different tokens. The E2E
    validation (commit 5ec4fd7) confirmed this: layer-24-only importance
    produced 6% agreement on NIAH.

    Args:
        cache: List of KVCache objects (one per layer)
        model: The loaded model (for attention config)
        obs_window: Observation window size
        layers: Which layers to aggregate (default: last 4)

    Returns:
        importance: (B, H_kv, T) — aggregated importance across layers
    """
    n_layers = len(cache)
    if layers is None:
        layers = list(range(max(0, n_layers - 4), n_layers))

    # Get attention config from model
    attn = model.layers[layers[0]].self_attn
    H_q = attn.n_heads if hasattr(attn, 'n_heads') else 32
    H_kv = cache[layers[0]].state[0].shape[1]
    D = cache[layers[0]].state[0].shape[3]
    scale = D ** -0.5

    # Aggregate importance across selected layers
    all_importance = []
    for layer_idx in layers:
        keys = cache[layer_idx].state[0]  # (B, H_kv, T, D)
        # Use K as Q proxy (GQA-expanded) — approximation, but
        # late layers' K contains rich aggregated representations
        Q_proxy = mx.repeat(keys, H_q // H_kv, axis=1)

        imp = compute_attention_importance(Q_proxy, keys, scale, obs_window)
        mx.eval(imp)
        all_importance.append(imp)

    # Pool across layers: max (any layer considers token important → keep it)
    stacked = mx.stack(all_importance, axis=0)  # (n_layers, B, H_kv, T)
    importance = mx.max(stacked, axis=0)         # (B, H_kv, T)
    mx.eval(importance)

    return importance


def _select_global(pooled: mx.array, k: int) -> set:
    """Global top-K selection (original SnapKV behavior)."""
    top_k_indices = mx.argpartition(-pooled, kth=k, axis=-1)[:, :k]
    all_indices = set()
    top_k_np = top_k_indices.tolist() if hasattr(top_k_indices, 'tolist') else [[]]
    for b_indices in top_k_np:
        if isinstance(b_indices, list):
            all_indices.update(b_indices)
        else:
            all_indices.add(int(b_indices))
    return all_indices


def _select_segmented(pooled: mx.array, k: int, segment_size: int) -> set:
    """BUZZ-style segmented selection: per-segment top-K.

    Divides the selectable range into segments of `segment_size` tokens.
    Within each segment, selects the top-k_local tokens proportional to
    the segment's share of the total budget.

    This preserves local attention structure: each segment retains its
    own heavy-hitters instead of being globally outcompeted by recent
    tokens or attention sinks.

    From BUZZ (arXiv:2410.23079): segmented selection outperforms global
    H2O by 7.69% on multi-document QA at 2.5x cache reduction.
    """
    B, S = pooled.shape  # S = selectable tokens
    n_segments = max(1, (S + segment_size - 1) // segment_size)

    # Distribute budget proportionally across segments
    base_k = max(1, k // n_segments)
    remainder = k - base_k * n_segments

    all_indices = set()
    pooled_np = pooled[0].tolist()  # B=1 for inference

    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, S)
        seg_len = seg_end - seg_start

        # Budget for this segment (distribute remainder to early segments)
        seg_k = base_k + (1 if seg_idx < remainder else 0)
        seg_k = min(seg_k, seg_len)  # can't keep more than segment has

        if seg_k <= 0:
            continue

        # Select top-K within segment using numpy-style sorting
        seg_scores = pooled_np[seg_start:seg_end]
        # Get indices sorted by score (descending)
        indexed = sorted(range(seg_len), key=lambda i: -seg_scores[i])
        for i in indexed[:seg_k]:
            all_indices.add(seg_start + i)

    return all_indices


def _select_submodular(pooled: mx.array, k: int, segment_size: int,
                        values: mx.array | None = None) -> set:
    """Submodular greedy selection with CAOTE marginal gains.

    Within each BUZZ segment, uses greedy selection instead of top-K:
    at each step, picks the token that maximizes marginal gain relative
    to the already-retained set. Tokens redundant with retained tokens
    get penalized via value-vector similarity.

    From OTPrune (arXiv:2602.20205): greedy submodular selection achieves
    (1-1/e) ≈ 63% of optimal distributional fidelity — the first
    compositional guarantee for multi-token KV eviction.

    Args:
        pooled: (B, S) importance scores (CAOTE or attention-only)
        k: total tokens to select
        segment_size: BUZZ segment size (0 = global)
        values: (B, H_kv, S, D) value vectors for diversity penalty.
            If None, falls back to score-only selection (no diversity).
    """
    B, S = pooled.shape
    seg_size = segment_size if segment_size > 0 else S
    n_segments = max(1, (S + seg_size - 1) // seg_size)

    base_k = max(1, k // n_segments)
    remainder = k - base_k * n_segments

    all_indices = set()
    scores_np = pooled[0].tolist()

    # Get per-token value norms for diversity (pool across heads → mean)
    val_np = None
    if values is not None and values.shape[2] >= S:
        # Mean across heads: (S, D)
        v_mean_heads = mx.mean(values[0, :, :S, :], axis=0)  # (S, D)
        # Normalize for cosine similarity
        norms = mx.sqrt(mx.sum(v_mean_heads * v_mean_heads, axis=-1, keepdims=True) + 1e-8)
        val_np = (v_mean_heads / norms).tolist()  # list of D-dim vectors

    for seg_idx in range(n_segments):
        seg_start = seg_idx * seg_size
        seg_end = min(seg_start + seg_size, S)
        seg_len = seg_end - seg_start
        seg_k = base_k + (1 if seg_idx < remainder else 0)
        seg_k = min(seg_k, seg_len)

        if seg_k <= 0:
            continue

        if val_np is None:
            # No value vectors — fall back to score-only top-K
            seg_scores = scores_np[seg_start:seg_end]
            indexed = sorted(range(seg_len), key=lambda i: -seg_scores[i])
            for i in indexed[:seg_k]:
                all_indices.add(seg_start + i)
            continue

        # Greedy submodular selection within segment
        seg_scores = list(scores_np[seg_start:seg_end])  # mutable copy
        retained_in_seg = []

        for _ in range(seg_k):
            if not seg_scores:
                break
            # Pick highest-scoring token
            best_local = max(range(seg_len), key=lambda i: seg_scores[i]
                             if i not in set(retained_in_seg) else -1e30)
            if seg_scores[best_local] <= -1e30:
                break
            retained_in_seg.append(best_local)
            all_indices.add(seg_start + best_local)

            # Penalize tokens similar to the selected token (diversity)
            best_val = val_np[seg_start + best_local]
            for j in range(seg_len):
                if j in set(retained_in_seg):
                    continue
                # Cosine similarity (vectors are pre-normalized)
                sim = sum(a * b for a, b in zip(val_np[seg_start + j], best_val))
                if sim > 0.8:  # high similarity → penalize
                    seg_scores[j] *= max(0.1, 1.0 - 0.5 * sim)

    return all_indices


def _select_fair(pooled: mx.array, k: int,
                  partitions: list[tuple[int, int]],
                  min_tokens: int = 20,
                  segment_size: int = 0) -> set:
    """Fair eviction: proportional budget allocation across partitions.

    From "The Pitfalls of KV Cache Compression" (arXiv:2510.00231):
    allocates budget proportionally to each partition's size, then runs
    per-partition top-K selection independently. This prevents
    preferential eviction of early-context instructions.

    Args:
        pooled: (B, S) importance scores
        k: total tokens to select
        partitions: list of (start, end) token ranges
        min_tokens: minimum budget per partition (floor)
        segment_size: BUZZ segment size within each partition (0=global)
    """
    B, S = pooled.shape
    total_tokens = sum(max(0, min(e, S) - s) for s, e in partitions)

    # Allocate budget proportionally with floor
    n_parts = len(partitions)
    floor_total = min_tokens * n_parts
    if floor_total >= k:
        # Not enough budget for floors — distribute evenly
        budgets = [max(1, k // n_parts)] * n_parts
    else:
        remaining = k - floor_total
        budgets = []
        for s, e in partitions:
            part_len = max(0, min(e, S) - s)
            part_budget = min_tokens + int(remaining * part_len / max(total_tokens, 1))
            budgets.append(min(part_budget, part_len))

    # Adjust to hit exact total
    delta = k - sum(budgets)
    if delta > 0:
        for i in range(delta):
            idx = i % n_parts
            s, e = partitions[idx]
            part_len = max(0, min(e, S) - s)
            if budgets[idx] < part_len:
                budgets[idx] += 1

    # Run per-partition selection
    all_indices = set()
    pooled_np = pooled[0].tolist()

    for i, ((start, end), budget) in enumerate(zip(partitions, budgets)):
        part_start = start
        part_end = min(end, S)
        part_len = part_end - part_start
        part_k = min(budget, part_len)

        if part_k <= 0:
            continue

        if segment_size > 0 and part_len > segment_size:
            # BUZZ segments within this partition
            part_pooled = mx.array([pooled_np[part_start:part_end]])[None, :]
            part_pooled = pooled[:, part_start:part_end]
            seg_indices = _select_segmented(part_pooled, part_k, segment_size)
            for idx in seg_indices:
                all_indices.add(part_start + idx)
        else:
            # Top-K within partition
            seg_scores = pooled_np[part_start:part_end]
            indexed = sorted(range(part_len), key=lambda j: -seg_scores[j])
            for j in indexed[:part_k]:
                all_indices.add(part_start + j)

    return all_indices


def snapkv_select(
    importance: mx.array,
    keep_count: int,
    always_keep_last: int = 64,
    segment_size: int = 0,
    submodular: bool = False,
    values: mx.array | None = None,
    partitions: list[tuple[int, int]] | None = None,
    partition_min_tokens: int = 20,
    head_types: list[str] | None = None,
) -> mx.array:
    """Select top-K tokens to keep based on importance scores.

    When segment_size > 0, uses BUZZ-style segmented selection (per-segment
    top-K) instead of global top-K. This preserves local attention structure
    and prevents the "lost in the middle" problem at long contexts.

    When partitions is set, allocates budget proportionally across partitions
    (fair eviction from arXiv:2510.00231). Each partition gets budget
    proportional to its size, with a minimum floor to protect small partitions.

    When head_types is set (list of "streaming"/"retrieval" per KV head),
    rebalances importance: retrieval heads get 2× weight in the pooling,
    streaming heads get 0.5× weight. This shifts budget toward retrieval
    heads where it matters more for quality.

    Args:
        importance: (B, H_kv, T) — per-token importance per head
        keep_count: total tokens to keep (including always_keep_last)
        always_keep_last: always keep this many recent tokens (sink/window)
        segment_size: if > 0, use per-segment selection with this segment size.
            0 = global top-K (original SnapKV behavior).
        partitions: list of (start, end) token ranges for fair eviction.
            Budget is allocated proportionally to each partition's size.
        partition_min_tokens: minimum tokens to keep per partition (floor).
        head_types: list of "streaming"/"retrieval" per KV head for budget
            rebalancing (from DuoAttention or trig classification).
        submodular: if True, use greedy submodular selection with value-diversity
            penalty instead of independent top-K. Captures diminishing returns
            from correlated tokens. Requires values parameter.
        values: (B, H_kv, T, D) value vectors for submodular diversity penalty.

    Returns:
        keep_mask: (B, T) — boolean mask of tokens to keep (union across heads)
    """
    B, H_kv, T = importance.shape

    if keep_count >= T:
        return mx.ones((B, T), dtype=mx.bool_)

    # Always keep the last `always_keep_last` tokens
    selectable = max(0, T - always_keep_last)
    k_selectable = max(1, keep_count - always_keep_last)

    if selectable == 0:
        return mx.ones((B, T), dtype=mx.bool_)

    # Reweight importance by head type: retrieval heads get 2× weight,
    # streaming heads get 0.5× (shifts budget toward retrieval)
    imp_weighted = importance
    if head_types and len(head_types) == H_kv:
        weights = mx.array([2.0 if t == "retrieval" else 0.5
                           for t in head_types]).reshape(1, H_kv, 1)
        imp_weighted = importance * weights

    # Pool importance across heads (union strategy: max across heads)
    pooled = mx.max(imp_weighted[:, :, :selectable], axis=1)  # (B, selectable)

    if partitions and len(partitions) > 1:
        # Fair eviction: proportional budget per partition
        all_indices = _select_fair(pooled, k_selectable, partitions,
                                    partition_min_tokens, segment_size)
    elif submodular:
        # Submodular greedy with diversity penalty (within segments if set)
        all_indices = _select_submodular(pooled, k_selectable,
                                          segment_size, values)
    elif segment_size > 0 and selectable > segment_size:
        # BUZZ segmented selection: per-segment top-K
        all_indices = _select_segmented(pooled, k_selectable, segment_size)
    else:
        # Global top-K (original SnapKV)
        all_indices = _select_global(pooled, k_selectable)

    # Add always-keep-last positions
    for pos in range(selectable, T):
        all_indices.add(pos)

    # Build boolean mask — single scatter op instead of per-element loop
    sorted_indices = sorted(all_indices)
    idx = mx.array(sorted_indices, dtype=mx.uint32)
    keep_mask = mx.zeros((B, T), dtype=mx.bool_)
    keep_mask[:, idx] = True

    return keep_mask


def get_keep_indices(keep_mask: mx.array) -> list[int]:
    """Extract sorted keep indices from a mask (for cache compaction)."""
    mask_np = keep_mask[0].tolist() if keep_mask.shape[0] > 0 else []
    return [i for i, v in enumerate(mask_np) if v]


def _rerope_keys(keys: mx.array, old_positions: list[int],
                  rope_dims: int, rope_base: float = 1000000.0) -> mx.array:
    """Re-encode RoPE on compacted keys from original to sequential positions.

    After physical compaction, keys have RoPE for their original positions
    but sit at new sequential positions [0, 1, ..., N-1]. This function
    applies a per-token rotation shift to correct the encoding.

    RoPE rotations compose additively: rope(rope(x, a), b) = rope(x, a+b).
    So to shift from old_pos to new_pos: apply rope(x, new_pos - old_pos).

    Args:
        keys: (B, H, N, D) — keys with RoPE at original positions
        old_positions: list of original position indices
        rope_dims: number of dimensions that have RoPE applied
        rope_base: RoPE base frequency (Qwen3-Coder uses 1000000)

    Returns:
        keys with RoPE corrected to sequential positions [0, 1, ..., N-1]
    """
    B, H, N, D = keys.shape
    if N == 0:
        return keys

    # Compute per-position shifts: new_pos[i] - old_pos[i]
    new_positions = mx.arange(N)
    old_pos_arr = mx.array(old_positions[:N])
    shifts = new_positions - old_pos_arr  # (N,) — mostly negative

    # Compute rotation frequencies
    half_d = rope_dims // 2
    freqs = 1.0 / (rope_base ** (mx.arange(0, half_d).astype(mx.float32) * 2 / rope_dims))

    # Angles: shifts (N,) x freqs (half_d,) → (N, half_d)
    angles = shifts[:, None].astype(mx.float32) * freqs[None, :]  # (N, half_d)
    cos_a = mx.cos(angles).astype(keys.dtype)  # (N, half_d)
    sin_a = mx.sin(angles).astype(keys.dtype)

    # Apply rotation to key pairs: non-traditional RoPE (stride=half_d)
    # keys[..., :half_d] and keys[..., half_d:rope_dims] are the pairs
    k1 = keys[:, :, :, :half_d]      # (B, H, N, half_d)
    k2 = keys[:, :, :, half_d:rope_dims]

    # Reshape cos/sin for broadcasting: (1, 1, N, half_d)
    cos_a = cos_a[None, None, :, :]
    sin_a = sin_a[None, None, :, :]

    # RoPE rotation: [k1', k2'] = [k1*cos - k2*sin, k2*cos + k1*sin]
    k1_new = k1 * cos_a - k2 * sin_a
    k2_new = k2 * cos_a + k1 * sin_a

    # Reassemble: rope dims + pass-through dims
    if rope_dims < D:
        result = mx.concatenate([k1_new, k2_new, keys[:, :, :, rope_dims:]], axis=-1)
    else:
        result = mx.concatenate([k1_new, k2_new], axis=-1)

    return result


def compact_cache(cache: list, keep_indices: list[int],
                   original_offset: int | None = None,
                   model=None,
                   captured_kv: dict | None = None,
                   skip_rerope: bool = False) -> None:
    """Compact KV cache in-place, keeping only selected token positions.

    CRITICAL: Keys have RoPE baked in at their original positions. After
    compaction, keys are repositioned to [0, 1, ..., N-1] via re-RoPE.
    Set skip_rerope=True for progressive mid-prefill eviction to avoid
    re-RoPE accumulation — positions stay at original values. The cache
    offset is restored to original_offset so new tokens get correct RoPE.
    Final eviction should use skip_rerope=False for the definitive shift.

    For QuantizedKVCache: uses captured fp16 K/V (from Q hooks) when
    available to avoid dequantization noise. Falls back to dequant path.

    Args:
        cache: List of KVCache objects (one per layer)
        keep_indices: Sorted list of token positions to keep
        original_offset: Original cache offset before compaction.
        model: Model object (used to extract RoPE config).
        captured_kv: dict — maps layer_idx to captured K/V data.
        skip_rerope: If True, skip re-RoPE (for progressive eviction).
            Preserves original RoPE positions. Offset is restored to
            original_offset so subsequent tokens get correct positions.
    """
    if not cache:
        return

    # Save original offset BEFORE compaction changes it
    if original_offset is None:
        original_offset = cache[0].offset

    # Extract RoPE config from model
    rope_dims = 64  # Qwen3-Coder head_dim
    rope_base = 1000000.0
    if model is not None:
        try:
            attn = model.layers[0].self_attn
            rope_dims = attn.rope.dims
            rope_base = attn.rope.base
        except (AttributeError, IndexError):
            pass

    idx = mx.array(keep_indices)
    new_len = len(keep_indices)

    for layer_i in range(len(cache)):
        c = cache[layer_i]
        keys_raw = c.state[0]
        values_raw = c.state[1]

        if isinstance(keys_raw, tuple):
            # QuantizedKVCache — use captured fp16 K/V when available
            # to avoid double-quantization noise (which corrupts at 64K+).
            # Captured K/V are the ORIGINAL fp16 values from before the
            # cache quantized them during prefill — zero noise.
            cap_k, cap_v = None, None
            if captured_kv and layer_i in captured_kv:
                entry = captured_kv[layer_i]
                if isinstance(entry, tuple):
                    if len(entry) == 3:
                        _, cap_k, cap_v = entry  # (Q, K, V) from Q hooks
                    elif len(entry) == 2:
                        cap_k, cap_v = entry  # (K, V) from lightweight hooks

            if cap_k is not None and cap_v is not None:
                # Clean path: fp16 capture → gather → rerope → single quantize
                keys_fp = cap_k
                values_fp = cap_v
            else:
                # Fallback: dequantize (adds noise, works at ≤16K)
                keys_fp = mx.dequantize(
                    *keys_raw, group_size=c.group_size, bits=c.bits)
                values_fp = mx.dequantize(
                    *values_raw, group_size=c.group_size, bits=c.bits)

            keys_compact = keys_fp[:, :, idx, :]
            values_compact = values_fp[:, :, idx, :]

            if not skip_rerope:
                keys_compact = _rerope_keys(keys_compact, keep_indices,
                                            rope_dims, rope_base)
                c.keys = mx.quantize(
                    keys_compact, group_size=c.group_size, bits=c.bits)
                c.values = mx.quantize(
                    values_compact, group_size=c.group_size, bits=c.bits)
                c.offset = new_len
            else:
                # Progressive eviction: don't re-RoPE, just requantize
                # the kept subset. Offset stays at original value.
                # New tokens will write PAST the kept entries (at offset),
                # leaving gaps. The gaps have stale quantized data but
                # get overwritten by subsequent chunks.
                c.keys = mx.quantize(
                    keys_compact, group_size=c.group_size, bits=c.bits)
                c.values = mx.quantize(
                    values_compact, group_size=c.group_size, bits=c.bits)
                c.offset = new_len  # compact the buffer too
        else:
            # KVCache (fp16) or DuoKVCache — direct gather
            keys_compact = keys_raw[:, :, idx, :]
            values_compact = values_raw[:, :, idx, :]

            if not skip_rerope:
                keys_compact = _rerope_keys(keys_compact, keep_indices,
                                            rope_dims, rope_base)
                c.state = (keys_compact, values_compact)
            else:
                # Progressive eviction: compact without re-RoPE.
                # Keys keep original RoPE encoding. Offset = new_len.
                # The final eviction (skip_rerope=False) will apply
                # re-RoPE relative to these preserved positions.
                c.state = (keys_compact, values_compact)
                # Note: offset is set to new_len by state.setter.
                # Subsequent tokens get RoPE at new_len, new_len+1, etc.
                # This creates a position gap but the keys' original
                # RoPE encoding is preserved for the final re-RoPE pass.

    # Force evaluation of compacted state
    to_eval = []
    for c in cache:
        s = c.state
        if isinstance(s[0], tuple):
            to_eval.extend(s[0])
            to_eval.extend(s[1])
        else:
            to_eval.append(s[0])
            to_eval.append(s[1])
    mx.eval(*to_eval)


def count_kept(keep_mask: mx.array) -> int:
    """Count number of kept tokens."""
    return int(mx.sum(keep_mask).item())


def compute_ger(importance: mx.array, keep_mask: mx.array,
                top_pct: float = 0.1) -> float:
    """Compute Global Eviction Ratio (GER) — fraction of important tokens
    evicted from ALL heads simultaneously.

    From "Understanding the Physics of KV Cache Compression" (arXiv:2603.01426):
    GER spikes sharply near the hallucination cliff (~90% compression).
    A GER > 0.05 indicates dangerous eviction levels.

    Args:
        importance: (B, H_kv, T) — per-token importance per head
        keep_mask: (B, T) — boolean mask of kept tokens
        top_pct: fraction of tokens considered "important" (default 10%)

    Returns:
        GER value (0.0 = safe, >0.05 = dangerous, >0.1 = hallucination risk)
    """
    B, H_kv, T = importance.shape
    evicted = ~keep_mask[0]  # (T,) — True for evicted tokens

    # Identify important tokens: top top_pct by max-across-heads importance
    pooled = mx.max(importance[0], axis=0)  # (T,)
    n_important = max(1, int(T * top_pct))
    threshold_idx = mx.argpartition(-pooled, kth=n_important)[:n_important]
    important_mask = mx.zeros(T, dtype=mx.bool_)
    important_mask[threshold_idx] = True

    # GER: fraction of important tokens that are evicted
    important_evicted = mx.sum(important_mask & evicted)
    ger = float(important_evicted.item()) / max(n_important, 1)
    return ger


def check_ger_safety(importance: mx.array, keep_mask: mx.array,
                      threshold: float = 0.05,
                      widen_pct: float = 0.10) -> tuple[bool, float, int]:
    """Check GER safety and recommend budget adjustment if needed.

    Args:
        importance: (B, H_kv, T) — per-token importance
        keep_mask: (B, T) — boolean keep mask
        threshold: GER threshold above which to widen budget (default 0.05)
        widen_pct: fraction to widen keep_count by (default 10%)

    Returns:
        (safe, ger_value, recommended_keep_count)
        safe=True means GER is below threshold.
    """
    ger = compute_ger(importance, keep_mask)
    T = importance.shape[2]
    current_kept = int(mx.sum(keep_mask).item())
    recommended = current_kept

    if ger > threshold:
        # Widen budget to bring GER below threshold
        recommended = min(T, int(current_kept * (1 + widen_pct)))
        logger.warning(
            f"GER safety: {ger:.3f} > {threshold} threshold — "
            f"recommend widening budget from {current_kept} to {recommended}"
        )

    return ger <= threshold, ger, recommended


def compact_cache_pyramidal(
    cache: list,
    importance: mx.array,
    budgets: list[int],
    model=None,
    segment_size: int = 0,
    always_keep_last: int = 64,
) -> dict:
    """Compact KV cache with per-layer budgets (PyramidKV).

    Each layer gets its own keep count from the budget vector. Layers
    with lower budgets (middle layers) evict more aggressively.

    Args:
        cache: KVCache list (one per layer)
        importance: (B, H_kv, T) — aggregated importance (from CAOTE etc.)
        budgets: Per-layer keep counts (len = len(cache))
        model: Model for RoPE config
        segment_size: BUZZ segment size for per-segment selection
        always_keep_last: Tokens always kept at end

    Returns:
        dict with per-layer compaction stats
    """
    T = importance.shape[2]
    stats = {"layers": len(cache), "original_T": T, "per_layer": []}

    for layer_idx, c in enumerate(cache):
        keep_count = budgets[layer_idx] if layer_idx < len(budgets) else T

        if keep_count >= T:
            stats["per_layer"].append({"layer": layer_idx, "kept": T, "budget": keep_count})
            continue

        keep_mask = snapkv_select(importance, keep_count,
                                   always_keep_last=always_keep_last,
                                   segment_size=segment_size)
        indices = get_keep_indices(keep_mask)

        # Compact this single layer
        compact_cache([c], indices, model=model)
        stats["per_layer"].append({
            "layer": layer_idx, "kept": len(indices), "budget": keep_count,
        })

    return stats


def apply_snapkv_to_generate(keep_count: int, obs_window: int = 64,
                              use_caote: bool = False,
                              segment_size: int = 0,
                              use_freshness: bool = False,
                              use_submodular: bool = False,
                              capture_all_layers: bool = False) -> None:
    """Monkey-patch generate_step to run SnapKV eviction after prefill.

    When prompt length >= 2 * keep_count, the wrapper:
      1. Installs Q capture hooks on the last 4 decoder layers
      2. Lets prefill proceed normally (hooks capture Q projections)
      3. After the first decoded token, computes importance + compacts cache
      4. Removes hooks; decode continues with evicted cache

    The first decode token is generated with the full cache (before eviction).
    All subsequent tokens see the compacted cache. This is safe because
    the first token's attention saw the same tokens that SnapKV would keep.

    Args:
        keep_count: Number of tokens to keep after eviction.
        obs_window: Observation window for importance scoring.
        use_caote: If True, use CAOTE scoring (attention × value distinctiveness)
            instead of attention-only scoring. CAOTE preserves tokens whose
            eviction would cause the most attention-output error.
        segment_size: If > 0, use BUZZ-style per-segment selection instead
            of global top-K. Each segment of this many tokens keeps its own
            heavy-hitters, preventing the "lost in the middle" problem.
        use_freshness: If True, apply freshness decay to importance scores
            before selection. Superseded tokens (high cosine similarity to
            later tokens) get exponentially penalized. Prevents stale entries
            from consuming cache budget in agentic multi-turn scenarios.
    """
    import importlib
    import threading
    gen_mod = importlib.import_module("mlx_lm.generate")
    import mlx_lm.models.cache as cache_mod

    _logger = logging.getLogger("hypercar.snapkv")
    _original_generate_step = gen_mod.generate_step
    _tls = threading.local()

    # Wrap make_prompt_cache to capture cache reference
    _orig_make = cache_mod.make_prompt_cache

    def _capturing_make(model, **kw):
        c = _orig_make(model, **kw)
        _tls.prompt_cache = c
        # For quantized caches: install lightweight KV capture hooks
        # BEFORE prefill starts. This saves fp16 K/V before quantization.
        if capture_all_layers:
            kv_cap, kv_clean = install_kv_capture_hooks(c)
            _tls.kv_captured = kv_cap
            _tls.kv_cleanup = kv_clean
        return c

    cache_mod.make_prompt_cache = _capturing_make
    # Also patch utils reference
    try:
        import mlx_lm.utils as utils_mod
        if hasattr(utils_mod, "make_prompt_cache"):
            utils_mod.make_prompt_cache = _capturing_make
    except ImportError:
        pass

    def snapkv_generate_step(prompt, model, **kwargs):
        prompt_len = prompt.shape[0] if hasattr(prompt, 'shape') else len(prompt)

        if prompt_len < keep_count * 2:
            yield from _original_generate_step(prompt, model, **kwargs)
            return

        # Install Q capture hooks (last 4 layers for importance scoring)
        captured, cleanup = install_q_capture_hook(model)

        # Track cache + optional KV capture for quantized caches
        _tls.prompt_cache = kwargs.get('prompt_cache', None)
        _tls.kv_captured = None
        _tls.kv_cleanup = None
        cleanup_done = False

        try:
            gen = _original_generate_step(prompt, model, **kwargs)

            # First yield = prefill complete + first decode token
            first = next(gen)

            # Get cache reference
            cache = _tls.prompt_cache or kwargs.get('prompt_cache')

            if cache and captured:
                old_offset = cache[0].offset
                if use_caote:
                    importance = compute_caote_importance(
                        captured, cache, obs_window=obs_window)
                    scoring = "CAOTE"
                else:
                    importance = compute_importance_from_real_q(
                        captured, cache, obs_window=obs_window)
                    scoring = "attention-only"
                if use_freshness:
                    freshness = compute_freshness_scores(cache)
                    importance = importance * freshness
                    scoring += "+fresh"
                # Get values for submodular diversity penalty
                sel_values = None
                if use_submodular:
                    sel_values = mx.stack([
                        _get_fp16_values(cache[li])
                        for li in captured.keys()
                    ]).mean(axis=0)  # average across captured layers
                    scoring += "+submod"
                keep_mask = snapkv_select(importance, keep_count,
                                          segment_size=segment_size,
                                          submodular=use_submodular,
                                          values=sel_values)
                indices = get_keep_indices(keep_mask)

                # GER safety check — widen budget if near hallucination cliff
                safe, ger, rec_keep = check_ger_safety(
                    importance, keep_mask)
                if not safe:
                    # Re-select with wider budget
                    keep_mask = snapkv_select(importance, rec_keep,
                                              segment_size=segment_size,
                                              submodular=use_submodular,
                                              values=sel_values)
                    indices = get_keep_indices(keep_mask)
                    scoring += f"+ger({ger:.2f}→widen)"

                # Merge captured K/V for compaction
                merged_kv = {}
                if hasattr(_tls, 'kv_captured') and _tls.kv_captured:
                    merged_kv.update(_tls.kv_captured)
                merged_kv.update(captured)
                compact_cache(cache, indices, model=model,
                              captured_kv=merged_kv if merged_kv else None)
                new_len = len(indices)
                seg_info = f", seg={segment_size}" if segment_size > 0 else ""
                ger_info = f", GER={ger:.3f}" if ger > 0 else ""
                _logger.info(
                    f"SnapKV eviction ({scoring}{seg_info}{ger_info}): "
                    f"{old_offset} -> {new_len} tokens "
                    f"(kept {new_len * 100 // max(old_offset, 1)}%)"
                )

            # Remove hooks before continuing decode
            cleanup()
            if hasattr(_tls, 'kv_cleanup') and _tls.kv_cleanup:
                _tls.kv_cleanup()
            cleanup_done = True

            yield first
            yield from gen

        finally:
            if not cleanup_done:
                cleanup()
                if hasattr(_tls, 'kv_cleanup') and _tls.kv_cleanup:
                    _tls.kv_cleanup()

    gen_mod.generate_step = snapkv_generate_step
    # Patch server's reference if already imported
    try:
        import mlx_lm.server as srv
        srv.generate_step = snapkv_generate_step
    except ImportError:
        pass

    _logger.info(f"SnapKV eviction enabled: keep={keep_count}, obs_window={obs_window}")
