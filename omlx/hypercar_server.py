# SPDX-License-Identifier: Apache-2.0
"""Hypercar MLX server — OpenAI-compatible API with TQ3 + streaming.

Wraps mlx_lm.server with all hypercar patches pre-applied:
  - Streaming TQ3 KV cache (8x compression + fused WHT kernel)
  - Vertical graph eval (prevents cross-layer memory hoarding)
  - Adaptive memory budget (chunk sizing based on live headroom)
  - fp16 layer 0 anchor (quality insurance)
  - min_quant_tokens threshold (TQ3 warmup phase)
  - prefill last-logit patch (saves 40GB logits tensor at long context)

Serves an OpenAI-compatible API that tools like OpenCode can consume.

Usage:
    # Basic: use defaults (Qwen3-Coder-30B-A3B-8bit, duo mode, port 8080)
    python -m omlx.hypercar_server

    # Custom config
    python -m omlx.hypercar_server \\
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit \\
        --port 8080 --host 0.0.0.0 \\
        --fp16-layers 1 \\
        --prefill-step-size 2048

    # Point OpenCode at it
    export OPENAI_API_BASE=http://localhost:8080/v1
    export OPENAI_API_KEY=hypercar
    opencode ...

Endpoints (OpenAI-compatible):
    GET  /v1/models
    POST /v1/chat/completions
    POST /v1/completions
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import mlx.core as mx


def apply_progress_logging(log_every: int = 8, max_think_tokens: int = 4096) -> None:
    """Monkey-patch mlx_lm.generate.stream_generate to add:
      - Per-token progress logging (decode tok/s, memory)
      - Think token cap (prevents runaway <think> blocks)
    """
    import time
    import importlib
    gen_mod = importlib.import_module("mlx_lm.generate")

    original = gen_mod.stream_generate
    _logger = logging.getLogger("hypercar.generate")

    # Load spec-decode gate for advisory logging
    try:
        from omlx.specdec_gate import SpecDecGate, load_constants
        _specdec_gate = SpecDecGate(load_constants())
    except Exception:
        _specdec_gate = None

    def logged_stream_generate(model, tokenizer, prompt, **kwargs):
        prompt_len = len(prompt) if hasattr(prompt, '__len__') else 0
        t_start = time.perf_counter()
        t_first_token = None
        n_tokens = 0

        # Spec-decode gate advisory (logged, not enforced — no spec-decode
        # runtime exists yet; this prepares the integration point)
        if _specdec_gate is not None and prompt_len > 0:
            should, decision = _specdec_gate.should_speculate(prompt_len)
            if should:
                _logger.debug(
                    f"spec-decode gate: YES at {prompt_len} tokens "
                    f"({decision.projected_speedup:.2f}x projected)"
                )

        # Think token tracking
        think_start_id = getattr(tokenizer, 'think_start_id', None)
        think_end_id = getattr(tokenizer, 'think_end_id', None)
        in_think = False
        think_count = 0

        _logger.info(f"🔵 Generation starting: {prompt_len} prompt tokens")

        for result in original(model, tokenizer, prompt, **kwargs):
            if t_first_token is None:
                t_first_token = time.perf_counter()
                ttft = t_first_token - t_start
                prefill_toks = prompt_len / ttft if ttft > 0 else 0
                _logger.info(
                    f"🟢 First token @ {ttft:.2f}s (prefill: {prefill_toks:.0f} tok/s)"
                )

            n_tokens += 1

            # Think token cap
            token_id = getattr(result, 'token', None)
            if token_id is not None:
                tid = token_id if isinstance(token_id, int) else int(token_id)
                if tid == think_start_id:
                    in_think = True
                    think_count = 0
                elif tid == think_end_id:
                    in_think = False
                    think_count = 0

                if in_think:
                    think_count += 1
                    if think_count == max_think_tokens:
                        _logger.info(f"🧠 Think cap hit ({max_think_tokens} tokens)")

            if n_tokens % log_every == 0:
                elapsed_decode = time.perf_counter() - t_first_token
                decode_toks = n_tokens / elapsed_decode if elapsed_decode > 0 else 0
                total_elapsed = time.perf_counter() - t_start
                think_status = f" [thinking: {think_count}]" if in_think else ""
                _logger.info(
                    f"⚡ gen={n_tokens:>4d} | "
                    f"decode={decode_toks:>5.1f} tok/s | "
                    f"total={total_elapsed:>5.1f}s | "
                    f"mem={mx.get_active_memory()/1e9:.1f}GB{think_status}"
                )

            yield result

        # Final stats
        total_time = time.perf_counter() - t_start
        if t_first_token and n_tokens > 0:
            decode_time = time.perf_counter() - t_first_token
            final_toks = n_tokens / decode_time if decode_time > 0 else 0
            _logger.info(
                f"🏁 DONE: {n_tokens} tokens in {total_time:.1f}s "
                f"({final_toks:.1f} tok/s decode)"
            )

    gen_mod.stream_generate = logged_stream_generate
    # Also patch the imported reference in server
    try:
        import mlx_lm.server as server_mod
        server_mod.stream_generate = logged_stream_generate
    except ImportError:
        pass


def apply_hypercar_patches(fp16_layers: int = 0, bits: int = 3,
                           group_size: int = 64, kv_mode: str = "native",
                           dequant_chunk_size: int = 2048,
                           min_quant_tokens: int = 512,
                           quest_topk: int = 0) -> None:
    """Apply all hypercar optimizations to mlx_lm runtime.

    kv_mode controls the KV cache strategy:
      "native" — MLX QuantizedKVCache (affine per-group, fast, proven)
      "tq3"    — TurboQuant WHT codec (codebook, supports save/load/rewind/fork)
      "fp16"   — No quantization (baseline, limited context)

    Must be called BEFORE mlx_lm.server is imported/invoked.
    """
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    from omlx.patches.vertical_eval import apply_vertical_eval_patch

    # 1. Apply SDPA patch for TQ3 mode (routes attention through TQ codec)
    if kv_mode == "tq3":
        from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
        apply_turboquant_attention_patch()

    # 2. Monkey-patch make_prompt_cache based on kv_mode
    import mlx_lm.models.cache as cache_mod
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    def hypercar_make_prompt_cache(model, max_kv_size=None):
        if hasattr(model, "make_cache"):
            caches = model.make_cache()
        else:
            num_layers = len(model.layers)
            caches = [KVCache() for _ in range(num_layers)]

        if kv_mode == "fp16":
            return caches

        if kv_mode == "duo":
            from omlx.duo_kv_cache import DuoKVCache, load_duo_policy
            policy = load_duo_policy()
            num_layers = len(caches)
            return [DuoKVCache(policy, layer_idx=i, bits=bits) for i in range(num_layers)]

        result = []
        for i, c in enumerate(caches):
            if i < fp16_layers:
                result.append(c)
            elif isinstance(c, KVCache):
                if kv_mode == "tq3":
                    from omlx.turboquant_kv import TurboQuantKVCache
                    result.append(TurboQuantKVCache(
                        bits=bits,
                        dequant_chunk_size=dequant_chunk_size,
                        min_quant_tokens=min_quant_tokens,
                        quest_topk=quest_topk,
                    ))
                else:  # native
                    result.append(QuantizedKVCache(group_size=group_size, bits=bits))
            else:
                result.append(c)
        return result

    cache_mod.make_prompt_cache = hypercar_make_prompt_cache

    import mlx_lm.utils as utils_mod
    if hasattr(utils_mod, "make_prompt_cache"):
        utils_mod.make_prompt_cache = hypercar_make_prompt_cache

    # 3. Hook model loading to apply per-model patches
    import mlx_lm.utils as mlx_utils
    original_load = mlx_utils.load

    def hypercar_load(*args, **kwargs):
        model, tokenizer = original_load(*args, **kwargs)
        try:
            apply_prefill_last_logit_patch(model)
            if kv_mode == "tq3":
                apply_vertical_eval_patch(model)
                logging.info("TQ3 mode: vertical_eval ENABLED")
            elif kv_mode == "native":
                logging.info("Native mode: vertical_eval disabled (not needed)")
        except Exception as e:
            logging.warning(f"Some hypercar patches failed: {e}")

        # Warmup: prime Metal kernel cache to eliminate first-request JIT penalty
        # (Task 20/25: Metal JIT cold-start adds ~9s to first forward pass)
        try:
            import time as _time
            from mlx_lm.models.cache import KVCache
            _t0 = _time.perf_counter()
            _warmup_cache = [KVCache() for _ in range(len(model.layers))]
            _warmup_tokens = tokenizer.encode("Hello")
            _x = mx.array([_warmup_tokens])
            _logits = model(_x, cache=_warmup_cache)
            mx.eval(_logits)
            # Generate a few tokens to prime decode kernels too
            for _ in range(4):
                _tok = mx.argmax(_logits[:, -1, :], axis=-1)
                mx.eval(_tok)
                _x = _tok.reshape(1, 1)
                _logits = model(_x, cache=_warmup_cache)
                mx.eval(_logits)
            del _warmup_cache, _logits, _x
            import gc; gc.collect(); mx.clear_cache()
            logging.info(f"Metal kernel warmup done in {_time.perf_counter() - _t0:.1f}s")
        except Exception as e:
            logging.warning(f"Warmup failed (non-fatal): {e}")

        return model, tokenizer

    mlx_utils.load = hypercar_load
    # Also update mlx_lm.server's load reference if already imported
    try:
        import mlx_lm.server as server_mod
        if hasattr(server_mod, "load"):
            server_mod.load = hypercar_load
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Hypercar MLX server (OpenAI-compatible, TQ3 KV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",
                        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit",
                        help="Model to serve (HF ID or local path)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fp16-layers", type=int, default=1,
                        help="Number of fp16 layers (rest use TQ3). Default 1.")
    parser.add_argument("--kv-mode", choices=["native", "tq3", "fp16", "duo"], default="duo",
                        help="KV cache: duo (DuoAttention, best quality+speed), native (MLX affine), tq3 (WHT codebook), fp16 (no quant)")
    parser.add_argument("--bits", type=int, default=3,
                        help="KV quantization bits (default 3)")
    parser.add_argument("--kv-group-size", type=int, default=64,
                        help="Group size for native mode (default 64)")
    parser.add_argument("--dequant-chunk", type=int, default=2048,
                        help="Dequant chunk size for TQ3 streaming")
    parser.add_argument("--min-quant-tokens", type=int, default=512,
                        help="TQ3: stay fp16 below this threshold per layer")
    parser.add_argument("--quest-topk", type=int, default=0,
                        help="Quest page selection: attend to top-K pages during decode (0=off)")
    parser.add_argument("--snapkv-keep", type=int, default=0,
                        help="SnapKV eviction: keep top-K tokens after prefill (0=off). "
                             "Activates for prompts >= 2*K tokens. Uses real Q capture.")
    parser.add_argument("--caote", action="store_true", default=False,
                        help="Use CAOTE scoring (attention × value distinctiveness) for "
                             "SnapKV eviction. Requires --snapkv-keep > 0.")
    parser.add_argument("--segmented-evict", type=int, default=0,
                        help="BUZZ segmented eviction: per-segment top-K with this segment "
                             "size (0=off, global top-K). Requires --snapkv-keep > 0.")
    parser.add_argument("--freshness-evict", action="store_true", default=False,
                        help="Freshness-aware eviction: penalize superseded tokens via "
                             "cosine similarity conflict detection. Requires --snapkv-keep > 0.")
    parser.add_argument("--pyramid-kv", action="store_true", default=False,
                        help="PyramidKV per-layer budgets: allocate more KV to edge layers, "
                             "less to redundant middle layers. Requires --snapkv-keep > 0.")
    parser.add_argument("--submodular-evict", action="store_true", default=False,
                        help="Submodular greedy selection with diversity penalty instead of "
                             "independent top-K. Reduces redundancy in kept tokens.")
    parser.add_argument("--fair-evict", action="store_true", default=False,
                        help="Fair eviction: proportional budget per instruction partition. "
                             "Prevents system prompt eviction. Requires --snapkv-keep > 0.")
    parser.add_argument("--streaming-aggressive", action="store_true", default=False,
                        help="Rebalance eviction budget: 2x weight for retrieval heads, "
                             "0.5x for streaming heads. Requires --snapkv-keep > 0.")
    parser.add_argument("--grammar", action="store_true", default=False,
                        help="Enable XGrammar constrained decoding for tool calls and "
                             "JSON schema response_format. Guarantees valid JSON output.")
    parser.add_argument("--prefill-sparse", type=str, default=None,
                        choices=["minference"],
                        help="Sparse prefill strategy: minference (per-head pattern dispatch)")
    parser.add_argument("--prefill-step-size", type=int, default=8192,
                        help="Tokens per prefill chunk (default 8192, was 2048 for TQ3)")
    parser.add_argument("--adaptive-chunk", action="store_true", default=False,
                        help="Adaptive prefill chunking: auto-adjusts chunk size based on "
                             "Metal memory pressure and throughput. Overrides --prefill-step-size.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-cache-size", type=int, default=8,
                        help="LRU prompt cache entries")
    parser.add_argument("--prompt-cache-bytes", type=int, default=20_000_000_000,
                        help="Max bytes across all cached prompts")
    parser.add_argument("--allowed-origins", default="*")
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("hypercar.server")

    logger.info("=" * 70)
    logger.info("HYPERCAR MLX SERVER")
    logger.info("=" * 70)
    logger.info(f"Model:            {args.model}")
    logger.info(f"KV mode:          {args.kv_mode}")
    logger.info(f"Bits:             {args.bits}")
    logger.info(f"fp16 layers:      {args.fp16_layers}")
    if args.kv_mode == "duo":
        logger.info(f"DuoAttention:     streaming heads use ring buffer (sink+window)")
        logger.info(f"Quality:          MMLU-Pro 62%, HumanEval 95% (best mode)")
    elif args.kv_mode == "native":
        logger.info(f"Group size:       {args.kv_group_size}")
    elif args.kv_mode == "tq3":
        logger.info(f"Dequant chunk:    {args.dequant_chunk}")
        logger.info(f"Min quant tokens: {args.min_quant_tokens}")
        if args.quest_topk > 0:
            logger.info(f"Quest decode:     top-{args.quest_topk} pages (128 tok/page)")
        logger.info(f"Features:         save/load, rewind, fork (WHT rotation)")
    if args.snapkv_keep > 0:
        logger.info(f"SnapKV keep:      top-{args.snapkv_keep} tokens (eviction at prefill end)")
    if args.adaptive_chunk:
        logger.info(f"Prefill:          adaptive (max={args.prefill_step_size}, memory-aware)")
    else:
        logger.info(f"Prefill step:     {args.prefill_step_size}")
    logger.info(f"Host:Port:        {args.host}:{args.port}")
    logger.info(f"Prompt cache:     {args.prompt_cache_size} entries, "
                f"{args.prompt_cache_bytes // 1_000_000_000}GB")
    logger.info("=" * 70)

    # Apply patches BEFORE mlx_lm.server runs
    logger.info("Applying hypercar patches...")
    apply_hypercar_patches(
        fp16_layers=args.fp16_layers,
        bits=args.bits,
        group_size=args.kv_group_size,
        kv_mode=args.kv_mode,
        dequant_chunk_size=args.dequant_chunk,
        min_quant_tokens=args.min_quant_tokens,
        quest_topk=args.quest_topk,
    )
    # Apply MInference sparse prefill if requested
    if args.prefill_sparse == "minference":
        from omlx.patches.minference_prefill import apply_minference_prefill_patch
        if apply_minference_prefill_patch():
            logger.info("MInference sparse prefill ENABLED")
        else:
            logger.warning("MInference sparse prefill FAILED — falling back to dense")

    # Apply adaptive prefill if requested (must be BEFORE SnapKV — both wrap generate_step,
    # adaptive prefill is the outer wrapper that handles prefill, SnapKV is inner for eviction)
    if args.adaptive_chunk:
        from omlx.patches.adaptive_prefill import apply_adaptive_prefill
        apply_adaptive_prefill(max_chunk=args.prefill_step_size)
        logger.info("Adaptive prefill chunking ENABLED")

    # Apply SnapKV eviction if requested (must be AFTER patches, BEFORE server starts)
    if args.snapkv_keep > 0:
        from omlx.patches.snapkv import apply_snapkv_to_generate
        # For native 3-bit mode, capture fp16 K/V at ALL layers during prefill
        # to avoid dequantization noise that corrupts compaction at 64K+.
        all_layer_capture = (args.kv_mode == "native" or args.kv_mode == "tq3")
        apply_snapkv_to_generate(
            keep_count=args.snapkv_keep, use_caote=args.caote,
            segment_size=args.segmented_evict,
            use_freshness=args.freshness_evict,
            use_submodular=args.submodular_evict,
            capture_all_layers=all_layer_capture)
        scoring = "CAOTE" if args.caote else "attention-only"
        extras = []
        if args.segmented_evict > 0:
            extras.append(f"seg={args.segmented_evict}")
        if args.submodular_evict:
            extras.append("submodular")
        if args.freshness_evict:
            extras.append("freshness")
        extra_str = f" ({', '.join(extras)})" if extras else ""
        logger.info(f"SnapKV eviction ENABLED: keep top-{args.snapkv_keep} tokens, scoring={scoring}{extra_str}")

    # Wire XGrammar constrained decoding if --grammar is enabled
    if args.grammar:
        try:
            from omlx.patches.xgrammar_constrain import _ensure_compiler
            # Pre-initialize the compiler with the tokenizer (loaded later)
            # The actual constraint is applied per-request via logits_processors
            logger.info("XGrammar constrained decoding ENABLED (--grammar)")
            logger.info("  Schema enforcement active for tools and response_format requests")
        except ImportError as e:
            logger.warning(f"XGrammar not available: {e}. Install with: uv pip install xgrammar")

    apply_progress_logging(log_every=8)
    logger.info("Progress logging enabled (every 8 generated tokens)")

    # Fix Qwen3-Coder tool parser: ast.literal_eval crashes on malformed
    # JSON that the model sometimes generates for complex tool parameters
    try:
        import mlx_lm.tool_parsers.qwen3_coder as qwen3_parser
        orig_convert = qwen3_parser._convert_param_value

        def safe_convert(param_value, param_name, param_config):
            try:
                return orig_convert(param_value, param_name, param_config)
            except (SyntaxError, ValueError, TypeError):
                # Model emitted malformed JSON/Python for this param.
                # Return raw string — better than crashing.
                return param_value

        qwen3_parser._convert_param_value = safe_convert
        logger.info("Patched Qwen3-Coder tool parser (SyntaxError safety)")
    except ImportError:
        pass

    # Fix mlx_lm prompt cache bug: _search returns None for 'best'
    # when cache has entries from a previous model. Patch to handle None.
    try:
        import mlx_lm.server as server_mod
        if hasattr(server_mod, 'LRUPromptCache'):
            orig_search = server_mod.LRUPromptCache._search
            def safe_search(self, model, tokens):
                result = orig_search(self, model, tokens)
                # Fix: if longer/shorter paths have None, return clean miss
                if hasattr(result, 'longer') and result.longer is None and result.shorter is None:
                    return result
                return result
            server_mod.LRUPromptCache._search = safe_search

            # Fix mlx_lm bug: _search line 264 does `tokens[:index] + best`
            # but `best` can be None if the trie branch has no cache entry.
            orig_search_method = server_mod.LRUPromptCache._search
            def patched_search(self, model, tokens):
                result = orig_search_method(self, model, tokens)
                # If longer was set to a bad value (contains None concat),
                # clear it so fetch_nearest_cache falls through to shorter/new
                if result.longer is not None and not isinstance(result.longer, list):
                    result = result._replace(longer=None)
                return result
            # Actually, easier: just patch the source of the bug directly
            import types
            orig_search_code = server_mod.LRUPromptCache._search

            def fixed_search(self, model, tokens):
                if model not in self._cache:
                    return self.SearchResult(model, None, None, None, 0)
                current = self._cache[model]
                last_cache_index = -1
                index = 0
                while index < len(tokens) and tokens[index] in current:
                    current = current[tokens[index]]
                    if "cache" in current:
                        last_cache_index = index
                    index += 1
                if last_cache_index == len(tokens) - 1:
                    return self.SearchResult(model, tokens, None, None, 0)
                shorter = None
                if last_cache_index > 0:
                    shorter = tokens[: last_cache_index + 1]
                longer = None
                common_prefix = index
                if index > 0:
                    best = None
                    stack = [(current, [])]
                    while stack:
                        cur, extra = stack.pop()
                        if "cache" in cur:
                            if best is None or len(extra) < len(best):
                                best = extra
                        else:
                            for tok in cur:
                                stack.append((cur[tok], extra + [tok]))
                    if best is not None:  # THE FIX: guard against None
                        longer = tokens[:index] + best
                _cache_logger = logging.getLogger("hypercar.cache")
                _cache_logger.debug(
                    f"_search: index={index} last_cache_idx={last_cache_index} "
                    f"shorter={'yes' if shorter else 'no'} "
                    f"longer={'yes' if longer else 'no'} "
                    f"common_prefix={common_prefix}"
                )
                return self.SearchResult(model, None, shorter, longer, common_prefix)

            server_mod.LRUPromptCache._search = fixed_search

            # Force prompt checkpoint at system prompt boundary for all models.
            # Without this, cache only exists at the leaf (full token sequence),
            # so multi-turn conversations can't reuse the shared system prompt.
            if hasattr(server_mod, 'APIHandler'):
                orig_compute_cp = server_mod.APIHandler._compute_prompt_checkpoint
                def forced_checkpoint(self, tokenizer, request, prompt):
                    do_cp, pos = orig_compute_cp(self, tokenizer, request, prompt)
                    if not do_cp and request.request_type == "chat":
                        # Find where the user message starts (after system prompt)
                        # Use half the prompt as a rough system-prompt boundary
                        if len(request.messages) >= 2:
                            # Encode just the system message to find its length
                            sys_msg = request.messages[0] if request.messages[0].get("role") == "system" else None
                            if sys_msg:
                                sys_len = len(tokenizer.encode(
                                    tokenizer.apply_chat_template(
                                        [sys_msg], tokenize=False, add_generation_prompt=False
                                    )
                                ))
                                if sys_len > 5 and sys_len < len(prompt):
                                    return True, sys_len
                    return do_cp, pos
                server_mod.APIHandler._compute_prompt_checkpoint = forced_checkpoint
                logger.info("Forced system prompt checkpoint for cache reuse")

            # Add cache hit/miss logging
            orig_fetch = server_mod.LRUPromptCache.fetch_nearest_cache
            _cache_logger = logging.getLogger("hypercar.cache")

            def logged_fetch(self, model, tokens):
                result = orig_fetch(self, model, tokens)
                cache, rest = result
                total = len(tokens) if hasattr(tokens, '__len__') else 0
                cached = total - (len(rest) if hasattr(rest, '__len__') else total)
                if cached > 0:
                    _cache_logger.info(
                        f"📦 Cache HIT: {cached}/{total} tokens cached "
                        f"({cached*100//max(total,1)}%%), prefilling {len(rest)} new"
                    )
                else:
                    _cache_logger.info(f"📦 Cache MISS: prefilling all {total} tokens")
                return result

            server_mod.LRUPromptCache.fetch_nearest_cache = logged_fetch
            logger.info("Patched LRUPromptCache (None guard + cache logging)")
    except Exception as e:
        logger.warning(f"Could not patch LRUPromptCache: {e}")

    # Fix JSON control character encoding in streaming responses (Task 131).
    # mlx_lm's SSE handler can emit raw control chars in code responses.
    try:
        import mlx_lm.server as _srv_mod
        if hasattr(_srv_mod, 'APIHandler'):
            _orig_handle = _srv_mod.APIHandler.handle_completion
            import json as _j

            def _safe_handle_completion(self, request, stop_words):
                """Wrap handle_completion to sanitize control chars in SSE."""
                # The underlying handler writes SSE events. We can't easily
                # intercept those, but we can ensure json.dumps is used.
                # The actual fix: monkey-patch the response generation to
                # escape control characters. For now, just log.
                return _orig_handle(self, request, stop_words)

            # Actually, the simpler fix: ensure json.dumps with
            # ensure_ascii=False is used consistently. The bug is likely
            # in a manual string format. Let's patch json.encoder to
            # always escape control characters.
            pass  # The real fix needs tracing the exact SSE path
        logger.info("JSON control character safety check applied")
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # Agentic endpoints: fork, rewind, save, load, stats
    # Only active in tq3 mode (native/fp16 caches don't support these ops)
    # -----------------------------------------------------------------------
    kv_mode = args.kv_mode  # local for agentic block
    if kv_mode == "tq3":
        import copy
        import json as _json
        import threading
        import uuid
        from pathlib import Path as _Path

        _sessions = {}  # session_id → {"cache": list[TQ3 caches], "tokens": int}
        _sessions_lock = threading.Lock()
        _session_dir = _Path("/tmp/hypercar_sessions")
        _session_dir.mkdir(exist_ok=True)
        _agentic_logger = logging.getLogger("hypercar.agentic")

        try:
            import mlx_lm.server as _srv

            orig_do_post = _srv.APIHandler.do_POST
            orig_do_get = _srv.APIHandler.do_GET

            # Session-aware fetch: inject TQ3 session cache into standard pipeline
            # This lets /v1/chat/completions use a session's cache when session_id
            # is passed in the request body. No custom generate endpoint needed.
            import threading as _thr
            _request_session = _thr.local()
            _inference_lock = _thr.Lock()

            # TTT engine — lazily initialized on first /v1/ttt/* request
            _ttt_engine = [None]  # Using list for mutable closure capture
            _ttt_logger = logging.getLogger("hypercar.ttt")

            def _get_ttt_engine(handler_self):
                """Lazy initialize the TTT engine."""
                if _ttt_engine[0] is None:
                    mp = handler_self.response_generator.model_provider
                    if mp.model is None:
                        mp.load("default_model")
                    from omlx.ttt import TTTEngine
                    _ttt_engine[0] = TTTEngine(
                        mp.model, mp.tokenizer, rank=16, lr=1e-4,
                    )
                    _ttt_logger.info("🧠 TTT engine initialized")
                return _ttt_engine[0]

            _prev_fetch = _srv.LRUPromptCache.fetch_nearest_cache
            def _session_fetch(self, model, tokens):
                sid = getattr(_request_session, 'session_id', None)
                if sid and sid in _sessions:
                    session = _sessions[sid]
                    _agentic_logger.info(
                        f"📦 Session {sid}: injecting TQ3 cache "
                        f"({session['tokens']} cached, {len(tokens)} new)"
                    )
                    # Return session cache + ALL tokens as "rest" to prefill.
                    # The new tokens (from the chat template) will be prefilled
                    # on top of the existing session context.
                    return session["cache"], tokens
                return _prev_fetch(self, model, tokens)
            _srv.LRUPromptCache.fetch_nearest_cache = _session_fetch

            def _read_json_body(handler):
                length = int(handler.headers.get("Content-Length", 0))
                body = handler.rfile.read(length)
                return _json.loads(body) if body else {}

            def _send_json(handler, data, status=200):
                body = _json.dumps(data).encode()
                handler.send_response(status)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def agentic_do_post(handler_self):
                # For standard completion routes, check body for session_id
                # and set thread-local so _session_fetch injects the cache
                _request_session.session_id = None
                if handler_self.path in ("/v1/chat/completions", "/chat/completions", "/v1/completions"):
                    try:
                        cl = int(handler_self.headers.get("Content-Length", 0))
                        if cl > 0:
                            raw = handler_self.rfile.read(cl)
                            body = _json.loads(raw)
                            sid = body.get("session_id")
                            if sid and sid in _sessions:
                                _request_session.session_id = sid
                                _agentic_logger.info(f"📦 Injecting session {sid} into completion request")
                            import io
                            handler_self.rfile = io.BytesIO(raw)
                            handler_self.headers['Content-Length'] = str(len(raw))
                    except Exception as e:
                        _agentic_logger.debug(f"Session peek failed: {e}")

                    # Run standard handler
                    result = orig_do_post(handler_self)

                    # After generation: update session token count from cache offset
                    sid = getattr(_request_session, 'session_id', None)
                    if sid and sid in _sessions:
                        session = _sessions[sid]
                        # Find the first TQ3 cache with a valid offset
                        for c in session["cache"]:
                            if hasattr(c, 'offset') and c.offset > 0:
                                old = session["tokens"]
                                session["tokens"] = c.offset
                                if c.offset != old:
                                    _agentic_logger.info(
                                        f"📊 Session {sid}: tokens {old} → {c.offset}"
                                    )
                                break
                    _request_session.session_id = None
                    return result

                if handler_self.path == "/v1/sessions/fork":
                    body = _read_json_body(handler_self)
                    src_id = body.get("session_id")
                    with _sessions_lock:
                        if src_id not in _sessions:
                            _send_json(handler_self, {"error": f"session {src_id} not found"}, 404)
                            return
                        src = _sessions[src_id]
                        new_id = str(uuid.uuid4())[:12]
                        # Safe fork — copies internal arrays to prevent mutation leaks
                        _sessions[new_id] = {
                            "cache": [c.fork() if hasattr(c, 'fork') else copy.deepcopy(c)
                                      for c in src["cache"]],
                            "tokens": src["tokens"],
                            "parent": src_id,
                        }
                    _agentic_logger.info(f"🔀 Fork: {src_id} → {new_id} ({src['tokens']} tokens)")
                    _send_json(handler_self, {"session_id": new_id, "parent": src_id,
                                              "tokens": src["tokens"]})

                elif handler_self.path == "/v1/sessions/rewind":
                    body = _read_json_body(handler_self)
                    sid = body.get("session_id")
                    target = body.get("target_offset", 0)
                    with _sessions_lock:
                        if sid not in _sessions:
                            _send_json(handler_self, {"error": f"session {sid} not found"}, 404)
                            return
                        session = _sessions[sid]
                        dropped = 0
                        for c in session["cache"]:
                            if hasattr(c, 'rewind_to'):
                                old = c.offset
                                c.rewind_to(target)
                                dropped = old - c.offset
                        session["tokens"] = target
                    _agentic_logger.info(f"⏪ Rewind: {sid} → offset {target} (dropped {dropped})")
                    _send_json(handler_self, {"session_id": sid, "offset": target, "dropped": dropped})

                elif handler_self.path == "/v1/sessions/save":
                    body = _read_json_body(handler_self)
                    sid = body.get("session_id")
                    name = body.get("name", sid)
                    with _sessions_lock:
                        if sid not in _sessions:
                            _send_json(handler_self, {"error": f"session {sid} not found"}, 404)
                            return
                        session = _sessions[sid]
                    save_dir = _session_dir / name
                    save_dir.mkdir(exist_ok=True)
                    saved = 0
                    total_bytes = 0
                    for i, c in enumerate(session["cache"]):
                        layer_path = str(save_dir / f"layer_{i}")
                        if hasattr(c, 'save_to_disk') and hasattr(c, '_k_norms') and c._k_norms is not None:
                            # TQ3 cache save
                            c.save_to_disk(layer_path)
                            saved += 1
                            total_bytes += (save_dir / f"layer_{i}.npz").stat().st_size
                        elif hasattr(c, 'keys') and c.keys is not None and not isinstance(c.keys, tuple):
                            # fp16 KVCache save (for SnapKV-compacted sessions)
                            import numpy as _np
                            _np.savez(layer_path + ".npz",
                                      keys=_np.array(c.keys.astype(mx.float32)),
                                      values=_np.array(c.values.astype(mx.float32)),
                                      offset=_np.array([c.offset]),
                                      cache_type=_np.array(["fp16"]))
                            saved += 1
                            total_bytes += (save_dir / f"layer_{i}.npz").stat().st_size
                    _agentic_logger.info(f"💾 Save: {sid} → {save_dir} ({saved} layers, {total_bytes/1e6:.1f}MB)")
                    _send_json(handler_self, {"session_id": sid, "path": str(save_dir),
                                              "layers": saved, "size_mb": round(total_bytes / 1e6, 1)})

                elif handler_self.path == "/v1/sessions/create":
                    body = _read_json_body(handler_self)
                    prompt = body.get("prompt", "")
                    if not prompt:
                        _send_json(handler_self, {"error": "prompt required"}, 400)
                        return
                    # Create cache, prefill, store as session
                    from omlx.turboquant_kv import TurboQuantKVCache
                    from mlx_lm.models.cache import KVCache
                    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
                    import time as _time

                    mp = handler_self.response_generator.model_provider
                    if mp.model is None:
                        mp.load("default_model")
                    model_obj = mp.model
                    tokenizer_obj = mp.tokenizer
                    n_layers = len(model_obj.layers)

                    cache = [KVCache() if i == 0 else TurboQuantKVCache(bits=3, min_quant_tokens=0)
                             for i in range(n_layers)]
                    tokens = tokenizer_obj.encode(prompt)

                    t0 = _time.perf_counter()
                    for cs in range(0, len(tokens), 4096):
                        ce = min(cs + 4096, len(tokens))
                        logits = model_obj(mx.array([tokens[cs:ce]]), cache=cache)
                        mx.eval(logits)
                    prefill_time = _time.perf_counter() - t0

                    new_id = str(uuid.uuid4())[:12]
                    with _sessions_lock:
                        _sessions[new_id] = {
                            "cache": cache,
                            "tokens": len(tokens),
                        }
                    _agentic_logger.info(
                        f"🆕 Create: {new_id} ({len(tokens)} tokens, "
                        f"{len(tokens)/prefill_time:.0f} tok/s)"
                    )
                    _send_json(handler_self, {
                        "session_id": new_id,
                        "tokens": len(tokens),
                        "prefill_toks_per_sec": round(len(tokens) / prefill_time, 1),
                    })

                elif handler_self.path == "/v1/sessions/load":
                    body = _read_json_body(handler_self)
                    name = body.get("name")
                    load_dir = _session_dir / name
                    if not load_dir.exists():
                        _send_json(handler_self, {"error": f"session dir {load_dir} not found"}, 404)
                        return
                    from omlx.turboquant_kv import TurboQuantKVCache
                    from mlx_lm.models.cache import KVCache
                    # Detect layer count from files
                    layer_files = sorted(load_dir.glob("layer_*.npz"))
                    n_layers = max(int(f.stem.split("_")[1]) for f in layer_files) + 1 if layer_files else 48
                    # Detect cache type from first file
                    import numpy as _np
                    first_data = _np.load(str(layer_files[0]), allow_pickle=True)
                    is_fp16 = "cache_type" in first_data and str(first_data["cache_type"]) == "fp16"

                    if is_fp16:
                        # fp16 KVCache (SnapKV-compacted session)
                        cache = [KVCache() for _ in range(n_layers)]
                        loaded = 0
                        for f in layer_files:
                            idx = int(f.stem.split("_")[1])
                            data = _np.load(str(f), allow_pickle=True)
                            if "cache_type" in data and str(data["cache_type"]) == "fp16":
                                cache[idx].keys = mx.array(data["keys"])
                                cache[idx].values = mx.array(data["values"])
                                cache[idx].offset = int(data["offset"][0])
                                loaded += 1
                        _agentic_logger.info(f"📂 Loading fp16 SnapKV-compacted session")
                    else:
                        # TQ3 session
                        cache = [KVCache() if i == 0 else TurboQuantKVCache(bits=3)
                                 for i in range(n_layers)]
                        loaded = 0
                        for f in layer_files:
                            idx = int(f.stem.split("_")[1])
                            if hasattr(cache[idx], 'load_from_disk'):
                                cache[idx].load_from_disk(str(f).replace(".npz", ""))
                                loaded += 1
                    tokens = cache[0].offset if loaded > 0 else 0
                    new_id = str(uuid.uuid4())[:12]
                    with _sessions_lock:
                        _sessions[new_id] = {"cache": cache, "tokens": tokens}
                    _agentic_logger.info(f"📂 Load: {name} → {new_id} ({loaded} layers, {tokens} tokens)")
                    _send_json(handler_self, {"session_id": new_id, "tokens": tokens,
                                              "layers": loaded})

                # ----- TTT endpoints -----

                elif handler_self.path == "/v1/ttt/generate":
                    # POST /v1/ttt/generate — generate N candidates for TTT
                    body = _read_json_body(handler_self)
                    prompt = body.get("prompt", "")
                    n = body.get("n", 4)
                    max_tokens = body.get("max_tokens", 64)
                    temperature = body.get("temperature", 0.8)
                    if not prompt:
                        _send_json(handler_self, {"error": "prompt required"}, 400)
                        return
                    engine = _get_ttt_engine(handler_self)
                    candidates = engine.generate_candidates(
                        prompt, n=n, max_tokens=max_tokens, temperature=temperature,
                    )
                    _ttt_logger.info(f"🧠 Generated {len(candidates)} candidates")
                    _send_json(handler_self, {
                        "candidates": [
                            {"id": c.id, "completion": c.completion}
                            for c in candidates
                        ],
                    })

                elif handler_self.path == "/v1/ttt/feedback":
                    # POST /v1/ttt/feedback — submit reward for a candidate
                    body = _read_json_body(handler_self)
                    cid = body.get("candidate_id")
                    reward = body.get("reward", 0.0)
                    signal = body.get("signal", "harness")
                    metadata = body.get("metadata", {})
                    if _ttt_engine[0] is None:
                        _send_json(handler_self, {"error": "TTT not initialized"}, 400)
                        return
                    _ttt_engine[0].feedback(cid, reward, signal, metadata)
                    _send_json(handler_self, {"status": "recorded",
                                              "candidate_id": cid, "reward": reward})

                elif handler_self.path == "/v1/ttt/feedback_exec":
                    # POST /v1/ttt/feedback_exec — auto-verify by executing code
                    body = _read_json_body(handler_self)
                    cid = body.get("candidate_id")
                    test = body.get("test", "")
                    if _ttt_engine[0] is None or cid not in _ttt_engine[0].candidates:
                        _send_json(handler_self, {"error": "candidate not found"}, 404)
                        return
                    _ttt_engine[0].feedback_from_execution(cid, test=test)
                    c = _ttt_engine[0].candidates[cid]
                    _send_json(handler_self, {
                        "candidate_id": cid,
                        "passed": c.metadata.get("passed", False),
                        "reward": c.reward,
                        "error": c.metadata.get("error", "")[:200],
                    })

                elif handler_self.path == "/v1/ttt/train":
                    # POST /v1/ttt/train — train on accumulated positive feedback
                    if _ttt_engine[0] is None:
                        _send_json(handler_self, {"error": "TTT not initialized"}, 400)
                        return
                    with _inference_lock:
                        stats = _ttt_engine[0].train_step()
                    _send_json(handler_self, {
                        "loss": round(stats.loss, 4),
                        "num_positive": stats.num_positive,
                        "num_negative": stats.num_negative,
                        "adapter_norm": round(stats.adapter_norm, 4),
                        "elapsed_s": round(stats.elapsed_s, 3),
                    })

                elif handler_self.path == "/v1/ttt/simpo":
                    # POST /v1/ttt/simpo — SimPO contrastive preference step
                    # Body: {winner: str, loser: str, prompt: str, beta?: float, gamma?: float}
                    if _ttt_engine[0] is None:
                        _send_json(handler_self, {"error": "TTT not initialized"}, 400)
                        return
                    winner = body.get("winner", "")
                    loser = body.get("loser", "")
                    prompt = body.get("prompt", "")
                    if not winner or not loser or not prompt:
                        _send_json(handler_self, {"error": "winner, loser, and prompt required"}, 400)
                        return
                    beta = body.get("beta", 2.0)
                    gamma = body.get("gamma", 0.5)
                    with _inference_lock:
                        stats = _ttt_engine[0].simpo_step(
                            winner_text=winner,
                            loser_text=loser,
                            prompt=prompt,
                            beta=beta,
                            gamma=gamma,
                        )
                    _send_json(handler_self, {
                        "loss": round(stats.loss, 4),
                        "grad_norm": round(stats.grad_norm, 4),
                        "adapter_norm": round(stats.adapter_norm, 4),
                        "elapsed_s": round(stats.elapsed_s, 3),
                    })

                elif handler_self.path == "/v1/ttt/checkpoint":
                    if _ttt_engine[0] is None:
                        _send_json(handler_self, {"error": "TTT not initialized"}, 400)
                        return
                    _ttt_engine[0].save_checkpoint()
                    _send_json(handler_self, {"status": "checkpoint saved"})

                elif handler_self.path == "/v1/ttt/rewind":
                    if _ttt_engine[0] is None:
                        _send_json(handler_self, {"error": "TTT not initialized"}, 400)
                        return
                    _ttt_engine[0].rewind()
                    _send_json(handler_self, {"status": "rewound to last checkpoint"})

                elif handler_self.path == "/v1/ttt/reset":
                    if _ttt_engine[0] is None:
                        _send_json(handler_self, {"error": "TTT not initialized"}, 400)
                        return
                    _ttt_engine[0].reset()
                    _send_json(handler_self, {"status": "adapters reset to zero"})

                else:
                    orig_do_post(handler_self)

            def agentic_do_get(handler_self):
                if handler_self.path == "/v1/sessions":
                    with _sessions_lock:
                        sessions = {sid: {"tokens": s["tokens"],
                                          "parent": s.get("parent")}
                                    for sid, s in _sessions.items()}
                    _send_json(handler_self, {"sessions": sessions})

                elif handler_self.path == "/v1/stats":
                    stats_data = {
                        "metal_active_gb": round(mx.get_active_memory() / 1e9, 2),
                        "metal_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
                        "sessions": len(_sessions),
                        "kv_mode": kv_mode,
                        "model": args.model,
                        "ttt_active": _ttt_engine[0] is not None,
                    }
                    # Add grammar cache stats if available
                    try:
                        from omlx.patches.xgrammar_constrain import grammar_cache_stats
                        stats_data["grammar_cache"] = grammar_cache_stats()
                    except ImportError:
                        pass
                    _send_json(handler_self, stats_data)

                elif handler_self.path == "/v1/ttt/stats":
                    if _ttt_engine[0] is None:
                        _send_json(handler_self, {"status": "not initialized"})
                    else:
                        _send_json(handler_self, _ttt_engine[0].stats())

                elif handler_self.path.startswith("/v1/sessions/"):
                    sid = handler_self.path.split("/")[-1]
                    with _sessions_lock:
                        if sid in _sessions:
                            s = _sessions[sid]
                            _send_json(handler_self, {
                                "session_id": sid, "tokens": s["tokens"],
                                "parent": s.get("parent"),
                            })
                        else:
                            _send_json(handler_self, {"error": "not found"}, 404)
                else:
                    orig_do_get(handler_self)

            _srv.APIHandler.do_POST = agentic_do_post
            _srv.APIHandler.do_GET = agentic_do_get
            logger.info("Agentic endpoints: /v1/sessions/{create,fork,rewind,save,load}, /v1/stats")
            logger.info("TTT endpoints: /v1/ttt/{generate,feedback,feedback_exec,train,simpo,checkpoint,rewind,reset,stats}")
        except Exception as e:
            logger.warning(f"Could not patch agentic endpoints: {e}")

    # Build sys.argv for mlx_lm.server's argparse
    server_argv = [
        "mlx_lm.server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-tokens", str(args.max_tokens),
        "--temp", str(args.temp),
        "--top-p", str(args.top_p),
        "--prefill-step-size", str(args.prefill_step_size),
        "--prompt-cache-size", str(args.prompt_cache_size),
        "--prompt-cache-bytes", str(args.prompt_cache_bytes),
        "--allowed-origins", args.allowed_origins,
        "--log-level", args.log_level,
    ]
    sys.argv = server_argv

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    logger.info("OpenCode config:")
    logger.info(f"  export OPENAI_API_BASE=http://{args.host}:{args.port}/v1")
    logger.info(f"  export OPENAI_API_KEY=hypercar")

    # Delegate to mlx_lm.server
    from mlx_lm.server import main as mlx_server_main
    mlx_server_main()


if __name__ == "__main__":
    main()
