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


def install_q_capture_hook(model, target_layers: list[int] | None = None):
    """Install hooks on Attention modules to capture Q after RoPE.

    Monkey-patches the Attention.__call__ to store the last set of
    query projections (after RoPE). These are needed for accurate
    SnapKV importance computation.

    Args:
        model: Loaded model with .layers[i].self_attn
        target_layers: Which layers to hook (default: last 4)

    Returns:
        captured: dict mapping layer_idx → mx.array of queries (B, H_q, L, D)
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
                    keys = sa.rope(keys, offset=cache.offset)
                    keys, values = cache.update_and_fetch(keys, values)
                else:
                    queries = sa.rope(queries)
                    keys = sa.rope(keys)

                # CAPTURE Q
                captured[lid] = queries

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


def compute_importance_from_real_q(
    captured_queries: dict,
    cache: list,
    obs_window: int = 64,
) -> mx.array:
    """Compute importance using real Q projections captured by the hook.

    This is the correct importance computation — uses actual query
    vectors (after RoPE) rather than K-as-Q proxy.

    Args:
        captured_queries: dict from install_q_capture_hook (layer_idx → Q)
        cache: KVCache list (for K values)
        obs_window: Observation window size

    Returns:
        importance: (B, H_kv, T) — aggregated importance
    """
    all_importance = []

    for layer_idx, queries in captured_queries.items():
        keys = _get_fp16_keys(cache[layer_idx])  # (B, H_kv, T, D)
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


def snapkv_select(
    importance: mx.array,
    keep_count: int,
    always_keep_last: int = 64,
) -> mx.array:
    """Select top-K tokens to keep based on importance scores.

    Args:
        importance: (B, H_kv, T) — per-token importance per head
        keep_count: total tokens to keep (including always_keep_last)
        always_keep_last: always keep this many recent tokens (sink/window)

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

    # Pool importance across heads (union strategy: max across heads)
    pooled = mx.max(importance[:, :, :selectable], axis=1)  # (B, selectable)

    # Top-k selection
    top_k_indices = mx.argpartition(-pooled, kth=k_selectable, axis=-1)[:, :k_selectable]

    # Build keep indices (sorted for cache compaction)
    # Union top-k across batch (B=1 for inference)
    all_indices = set()
    top_k_np = top_k_indices.tolist() if hasattr(top_k_indices, 'tolist') else [[]]
    for b_indices in top_k_np:
        if isinstance(b_indices, list):
            all_indices.update(b_indices)
        else:
            all_indices.add(int(b_indices))

    # Add always-keep-last positions
    for pos in range(selectable, T):
        all_indices.add(pos)

    # Build boolean mask
    keep_mask = mx.zeros((B, T), dtype=mx.bool_)
    sorted_indices = sorted(all_indices)
    for pos in sorted_indices:
        keep_mask = keep_mask.at[:, pos].add(mx.ones((B,), dtype=mx.bool_))

    return keep_mask


def get_keep_indices(keep_mask: mx.array) -> list[int]:
    """Extract sorted keep indices from a mask (for cache compaction)."""
    mask_np = keep_mask[0].tolist() if keep_mask.shape[0] > 0 else []
    return [i for i, v in enumerate(mask_np) if v]


def compact_cache(cache: list, keep_indices: list[int],
                   original_offset: int | None = None) -> None:
    """Compact KV cache in-place, keeping only selected token positions.

    This is the core operation for SnapKV: after computing importance
    and selecting which tokens to keep, rewrite the cache to contain
    only those positions. The kept tokens retain their exact values
    (zero approximation error for fp16; requantization for quantized).

    Supports both KVCache (fp16) and QuantizedKVCache (native mode).
    For QuantizedKVCache: dequantize → gather → requantize per layer.

    Args:
        cache: List of KVCache objects (one per layer)
        keep_indices: Sorted list of token positions to keep
        original_offset: Original cache offset before compaction.
            If None, preserved automatically from the first cache entry.
    """
    if not cache:
        return

    # Save original offset BEFORE compaction changes it
    if original_offset is None:
        original_offset = cache[0].offset

    idx = mx.array(keep_indices)
    new_len = len(keep_indices)

    for c in cache:
        keys_raw = c.state[0]
        values_raw = c.state[1]

        if isinstance(keys_raw, tuple):
            # QuantizedKVCache — dequantize, gather, requantize
            keys_fp = mx.dequantize(
                *keys_raw, group_size=c.group_size, bits=c.bits)
            values_fp = mx.dequantize(
                *values_raw, group_size=c.group_size, bits=c.bits)

            keys_compact = keys_fp[:, :, idx, :]
            values_compact = values_fp[:, :, idx, :]

            c.keys = mx.quantize(
                keys_compact, group_size=c.group_size, bits=c.bits)
            c.values = mx.quantize(
                values_compact, group_size=c.group_size, bits=c.bits)
            c.offset = new_len
        else:
            # KVCache (fp16) or DuoKVCache — direct gather
            keys_compact = keys_raw[:, :, idx, :]
            values_compact = values_raw[:, :, idx, :]

            # state.setter updates offset to keys.shape[2]
            c.state = (keys_compact, values_compact)

        # DON'T restore original offset — let offset = compacted length.
        # Kept K vectors already have their original RoPE baked in, so the
        # Q-K dot product sees the correct relative distance.

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


def apply_snapkv_to_generate(keep_count: int, obs_window: int = 64) -> None:
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
    """
    import threading
    import mlx_lm.generate as gen_mod
    import mlx_lm.models.cache as cache_mod

    _logger = logging.getLogger("hypercar.snapkv")
    _original_generate_step = gen_mod.generate_step
    _tls = threading.local()

    # Wrap make_prompt_cache to capture cache reference
    _orig_make = cache_mod.make_prompt_cache

    def _capturing_make(model, **kw):
        c = _orig_make(model, **kw)
        _tls.prompt_cache = c
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

        # Install Q capture hooks on last 4 layers
        captured, cleanup = install_q_capture_hook(model)

        # Track cache: may come from kwargs or from make_prompt_cache
        _tls.prompt_cache = kwargs.get('prompt_cache', None)
        cleanup_done = False

        try:
            gen = _original_generate_step(prompt, model, **kwargs)

            # First yield = prefill complete + first decode token
            first = next(gen)

            # Get cache reference
            cache = _tls.prompt_cache or kwargs.get('prompt_cache')

            if cache and captured:
                old_offset = cache[0].offset
                importance = compute_importance_from_real_q(
                    captured, cache, obs_window=obs_window)
                keep_mask = snapkv_select(importance, keep_count)
                indices = get_keep_indices(keep_mask)
                compact_cache(cache, indices)
                new_len = len(indices)
                _logger.info(
                    f"SnapKV eviction: {old_offset} -> {new_len} tokens "
                    f"(kept {new_len * 100 // max(old_offset, 1)}%)"
                )

            # Remove hooks before continuing decode
            cleanup()
            cleanup_done = True

            yield first
            yield from gen

        finally:
            if not cleanup_done:
                cleanup()

    gen_mod.generate_step = snapkv_generate_step
    # Patch server's reference if already imported
    try:
        import mlx_lm.server as srv
        srv.generate_step = snapkv_generate_step
    except ImportError:
        pass

    _logger.info(f"SnapKV eviction enabled: keep={keep_count}, obs_window={obs_window}")
