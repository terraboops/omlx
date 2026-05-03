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

from omlx.model_constants import SERVER_DEFAULT_MODEL_ID

# Task 355 Cycle 1: extracted apply_progress_logging + validate_drafter_for_request
# into omlx/server/progress_logging.py. Re-export here for backward compat
# (existing call sites import these names from omlx.hypercar_server).
from omlx.server.progress_logging import (  # noqa: E402,F401
    apply_progress_logging,
    validate_drafter_for_request,
)
# Task 355 Cycle 2: extracted apply_hypercar_patches into omlx/server/patches.py.
# Re-export here for backward compat.
from omlx.server.patches import apply_hypercar_patches  # noqa: E402,F401


def main():
    # Task 355 Cycle 3: argparse declarations extracted to omlx/server/cli_args.py.
    from omlx.server.cli_args import build_parser
    parser = build_parser(epilog=__doc__)
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
        if args.duo_quantize:
            logger.info(f"DuoKV quantize:   retrieval heads use QuantizedKVCache({args.bits}-bit)")
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
        quantize_retrieval=args.duo_quantize,
    )
    # Apply MInference sparse prefill if requested. Pass model_id so the
    # patch loads the right per-model pattern table (D2 / D3 fixes).
    if args.prefill_sparse == "minference":
        from omlx.patches.minference_prefill import apply_minference_prefill_patch
        if apply_minference_prefill_patch(model_id=args.model):
            logger.info("MInference sparse prefill ENABLED")
        else:
            logger.warning("MInference sparse prefill FAILED — falling back to dense")

    # Apply TTT head-routing dispatcher if requested (Task 388 Phase 2).
    # --ttt-router-policy alone → bit-equivalence mode (no behavior change),
    # useful for confirming the patch installs cleanly before adding blocks.
    # --ttt-router-policy + --ttt-router-blocks-dir → routes streaming heads
    # with a loaded TTT block through the recurrence; retrieval heads keep
    # softmax attention.
    if args.ttt_router_blocks_dir is not None and args.ttt_router_policy is None:
        logger.error(
            "--ttt-router-blocks-dir requires --ttt-router-policy. "
            "The blocks dir alone has no head classification → "
            "router cannot decide which heads to route."
        )
    elif args.ttt_router_policy is not None:
        from pathlib import Path
        from omlx.patches.ttt_head_router import (
            TTTHeadRouter, apply_ttt_head_router_patch,
        )
        ttt_dir = (Path(args.ttt_router_blocks_dir)
                   if args.ttt_router_blocks_dir else None)
        router = TTTHeadRouter.from_policy(
            Path(args.ttt_router_policy), ttt_dir=ttt_dir,
        )
        if apply_ttt_head_router_patch(router):
            n_blocks = len(router.ttt_blocks)
            if n_blocks == 0:
                logger.info(
                    "TTT head router INSTALLED in bit-equivalence mode "
                    "(policy loaded; no TTT blocks → all heads route to "
                    "original SDPA, output bit-identical to baseline)"
                )
            else:
                logger.info(
                    f"TTT head router INSTALLED with {n_blocks} TTT blocks "
                    f"loaded across {router.n_layers} layers × "
                    f"{router.n_heads} heads"
                )
        else:
            logger.warning(
                "TTT head router install FAILED (already patched?) — "
                "falling back to original SDPA"
            )

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

    # Load speculative-decoding drafter if --draft-model is set (Task 341).
    # Loaded BEFORE the main model so a load failure surfaces early; main
    # model load happens later via the patched mlx_utils.load.
    drafter_model = None
    if args.draft_model:
        try:
            from mlx_lm import load as _mlx_load
            logger.info(f"Loading drafter for speculative decoding: {args.draft_model}")
            drafter_model, drafter_tokenizer = _mlx_load(args.draft_model)
            logger.info(
                f"Drafter loaded (vocab_size={len(drafter_tokenizer)}, "
                f"num_draft_tokens={args.num_draft_tokens})"
            )
            logger.info(
                "  Tokenizer-identity vs main model is checked per-request; "
                "mismatch disables drafter with a warning."
            )
        except Exception as e:
            logger.error(f"Failed to load drafter {args.draft_model!r}: {e}")
            logger.error("Continuing WITHOUT speculative decoding.")
            drafter_model = None

    apply_progress_logging(
        log_every=8,
        draft_model=drafter_model,
        num_draft_tokens=args.num_draft_tokens,
    )
    logger.info("Progress logging enabled (every 8 generated tokens)")
    if drafter_model is not None:
        logger.info(
            f"Speculative decoding ENABLED: drafter={args.draft_model}, "
            f"num_draft_tokens={args.num_draft_tokens}"
        )

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
    # Patch mlx_lm's json module to always use ensure_ascii=True, which
    # guarantees all control characters are \uXXXX escaped in SSE output.
    try:
        import mlx_lm.server as _srv_mod
        import json as _json_mod

        # Wrap json.dumps used by the server to force ensure_ascii=True
        _orig_json_dumps = _json_mod.dumps

        def _safe_json_dumps(*args, **kwargs):
            kwargs.setdefault("ensure_ascii", True)
            return _orig_json_dumps(*args, **kwargs)

        # Patch json.dumps in the server's module namespace
        _srv_mod.json.dumps = _safe_json_dumps
        logger.info("Patched JSON encoding for safe control character escaping")
    except Exception as e:
        logger.debug(f"JSON encoding patch skipped: {e}")

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
