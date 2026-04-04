# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Prefill (L>1): cache.dequantize() + mx.fast.scaled_dot_product_attention
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)
_sdpa_call_count = 0

_PATCHED = False


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        from ..turboquant_kv import TurboQuantKVCache, BatchTurboQuantKVCache

        # Unwrap VLM _IntOffsetCacheProxy to detect underlying TQ cache
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (TurboQuantKVCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (TurboQuantKVCache, BatchTurboQuantKVCache)):
            if queries.shape[-2] == 1 and real_cache._quantized:
                # Fused decode attention — no dequantize, works for both
                # single and batch (kernel uses batch_idx from grid)
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            else:
                # Prefill path — check if streaming mode is active
                if getattr(real_cache, '_streaming_active', False):
                    # STREAMING MODE: keys/values are ONLY the new chunk
                    # History is in compressed storage — no ghost concat
                    global _sdpa_call_count
                    _sdpa_call_count += 1
                    logger.debug(
                        "SDPA streaming call=%d history=%d queries_L=%d",
                        _sdpa_call_count, real_cache._new_chunk_start, queries.shape[-2],
                    )
                    from ..streaming_attention import (
                        streaming_tq_attention, self_attention_with_lse, _merge_outputs,
                    )

                    history_end = real_cache._new_chunk_start

                    # 1. Self-attention on new chunk (returns output + LSE, no recompute)
                    self_out, self_lse = self_attention_with_lse(
                        queries, keys.astype(queries.dtype),
                        values.astype(queries.dtype), scale, mask,
                    )

                    if history_end == 0:
                        return self_out

                    # 2. Streaming history attention (online softmax, chunked dequant)
                    #    Each chunk: dequant → scores → update → mx.eval → free
                    history_out, history_lse = streaming_tq_attention(
                        queries=queries,
                        codec=real_cache._codec,
                        k_norms=real_cache._k_norms[:, :, :history_end],
                        k_packed=real_cache._k_packed[:, :, :history_end],
                        v_norms=real_cache._v_norms[:, :, :history_end],
                        v_packed=real_cache._v_packed[:, :, :history_end],
                        total_tokens=history_end,
                        scale=scale,
                        chunk_size=real_cache._dequant_chunk_size,
                        return_lse=True,
                    )

                    # 3. Merge via LSE weighting (@mx.compile fused)
                    merged = _merge_outputs(self_out, self_lse,
                                            history_out, history_lse)

                    # CRITICAL: Force evaluation of this layer's attention output.
                    # Without this, all 47 layers' streaming results + self-attention
                    # + MLP intermediates accumulate in the MLX graph simultaneously,
                    # causing 53GB peak at 32K context. Each layer's attention output
                    # is ~8K×128×32heads×2bytes = 64MB — nothing. But the graph
                    # holding all intermediates for all layers at once is the killer.
                    mx.eval(merged)
                    return merged
                else:
                    # Short history: fp16 from update_and_fetch (fast path)
                    return mx.fast.scaled_dot_product_attention(
                        queries,
                        keys.astype(queries.dtype),
                        values.astype(queries.dtype),
                        scale=scale,
                        mask=mask,
                    )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_sdpa or func is not patched_sdpa:
                setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base
        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
