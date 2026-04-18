#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SnapKV physical compaction benchmark — memory savings + quality validation.

Validates Task 46: after SnapKV eviction via compact_cache, Metal memory
drops AND NIAH retrieval still works. This proves the selection algorithm
is correct AND the physical compaction saves memory.

Usage:
    .venv/bin/python -m omlx.bench.snapkv_bench
    .venv/bin/python -m omlx.bench.snapkv_bench --context 16384
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"


def build_niah_prompt(context_tokens: int, tokenizer) -> tuple[list[int], str]:
    """Build a NIAH prompt with a secret code buried in filler text.

    Returns (token_ids, needle_answer).
    """
    needle = "SNAPKV-COMPACT-7743"
    # Place needle at ~30% depth (not too early, not too late)
    filler_unit = "The quick brown fox jumps over the lazy dog. "
    filler_tokens_per_unit = len(tokenizer.encode(filler_unit))

    pre_needle_units = max(1, (context_tokens * 30 // 100) // filler_tokens_per_unit)
    needle_text = f"The secret verification code is {needle}. Remember this code."

    # Build prompt
    pre_filler = filler_unit * pre_needle_units
    messages = [{"role": "user", "content": (
        pre_filler + needle_text + " " + filler_unit * 20
        + "\n\nWhat is the secret verification code? "
        "Reply with ONLY the code, nothing else."
    )}]

    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(chat_text)

    # Pad or trim to target length
    if len(input_ids) < context_tokens:
        # Add more filler
        extra_units = (context_tokens - len(input_ids)) // filler_tokens_per_unit + 1
        extra_filler = filler_unit * extra_units
        messages = [{"role": "user", "content": (
            pre_filler + needle_text + " " + extra_filler
            + "\n\nWhat is the secret verification code? "
            "Reply with ONLY the code, nothing else."
        )}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(chat_text)

    # Trim to exact target if overshot
    if len(input_ids) > context_tokens:
        input_ids = input_ids[:context_tokens]

    return input_ids, needle


def generate_tokens(model, tokenizer, input_ids, cache, n_tokens=32,
                     prefill_chunk=4096):
    """Generate n tokens from a prefilled cache.

    Prefills in chunks to avoid OOM on long contexts (O(n²) attention
    per chunk × accumulated KV length).
    """
    prompt = mx.array(input_ids)

    # Chunked prefill
    for start in range(0, len(input_ids), prefill_chunk):
        end = min(start + prefill_chunk, len(input_ids))
        logits = model(prompt[start:end][None], cache=cache)
        mx.eval([c.state for c in cache])
        mx.clear_cache()

    # Decode
    tokens = []
    for _ in range(n_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(token.item())
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)

    return tokens


def _make_bench_cache(n_layers: int, kv_mode: str = "fp16",
                       bits: int = 3, group_size: int = 64):
    """Create KV cache list for the given mode."""
    from mlx_lm.models.cache import KVCache, QuantizedKVCache
    if kv_mode == "native":
        return [QuantizedKVCache(group_size=group_size, bits=bits)
                for _ in range(n_layers)]
    return [KVCache() for _ in range(n_layers)]


def run_test(model, tokenizer, context_tokens, keep_ratio, obs_window=64,
             use_caote=False, segment_size=0, skip_baseline=False,
             kv_mode="fp16"):
    """Run one SnapKV compaction test at given context length and keep ratio.

    Args:
        skip_baseline: Skip the expensive baseline generation (for 64K+ contexts
            where O(n²) baseline prefill takes 30+ minutes). When True, only runs
            the SnapKV path and checks needle retrieval without token agreement.
        kv_mode: "fp16" or "native" (3-bit QuantizedKVCache).

    Returns dict with results.
    """
    from mlx_lm.models.cache import KVCache
    from omlx.patches.snapkv import (
        install_q_capture_hook, compute_importance_from_real_q,
        compute_caote_importance,
        snapkv_select, get_keep_indices, compact_cache,
    )

    n_layers = len(model.layers)
    input_ids, needle = build_niah_prompt(context_tokens, tokenizer)
    actual_tokens = len(input_ids)
    keep_count = max(64, int(actual_tokens * keep_ratio))

    scoring_label = "CAOTE" if use_caote else "attention-only"
    logger.info(f"\n{'='*60}")
    logger.info(f"Context: {actual_tokens} tokens, keep ratio: {keep_ratio:.0%} "
                f"({keep_count} tokens), scoring: {scoring_label}, kv_mode: {kv_mode}")

    # --- Baseline: generate without eviction ---
    base_tokens = None
    base_text = ""
    if skip_baseline:
        logger.info(f"  Baseline: SKIPPED (--skip-baseline, context too long for O(n²))")
    else:
        gc.collect(); mx.clear_cache()
        cache_base = _make_bench_cache(n_layers, kv_mode)
        base_tokens = generate_tokens(model, tokenizer, input_ids, cache_base, n_tokens=32)
        base_text = tokenizer.decode(base_tokens)
        metal_baseline = mx.get_active_memory() / 1e9
        del cache_base; gc.collect(); mx.clear_cache()

        logger.info(f"  Baseline Metal after gen: {metal_baseline:.2f} GB")
        logger.info(f"  Baseline output: {base_text[:80]!r}")

    # --- SnapKV: prefill with Q capture, compact, generate ---
    gc.collect(); mx.clear_cache()

    # Install Q capture hooks (last 4 layers for importance scoring)
    captured, cleanup = install_q_capture_hook(model)

    cache = _make_bench_cache(n_layers, kv_mode)

    # For native mode: install lightweight KV capture hooks on ALL
    # quantized cache entries BEFORE prefill to save fp16 K/V
    from omlx.patches.snapkv import install_kv_capture_hooks
    kv_captured, kv_cleanup = None, None
    if kv_mode == "native":
        kv_captured, kv_cleanup = install_kv_capture_hooks(cache)

    # Prefill in chunks
    chunk_size = 4096
    x_ids = input_ids
    for start in range(0, len(x_ids), chunk_size):
        end = min(start + chunk_size, len(x_ids))
        chunk = mx.array([x_ids[start:end]])
        logits = model(chunk, cache=cache)
        mx.eval(logits)

    metal_after_prefill = mx.get_active_memory() / 1e9
    kv_offset_before = cache[0].offset

    # Compute importance from captured Q (+ values if CAOTE)
    scoring = "CAOTE" if use_caote else "attention-only"
    if use_caote:
        importance = compute_caote_importance(captured, cache, obs_window=obs_window)
    else:
        importance = compute_importance_from_real_q(captured, cache, obs_window=obs_window)
    mx.eval(importance)
    cleanup()  # remove hooks

    # Select tokens to keep
    keep_mask = snapkv_select(importance, keep_count, segment_size=segment_size)
    indices = get_keep_indices(keep_mask)
    actual_kept = len(indices)

    # Measure memory BEFORE compaction
    mx.eval(*[c.state[0] for c in cache], *[c.state[1] for c in cache])
    metal_before_compact = mx.get_active_memory() / 1e9

    # Compact cache (re-RoPE keys to sequential positions)
    # Merge captured K/V for compaction: lightweight hooks (all quantized layers)
    # + Q hooks (last 4 layers with Q, K, V)
    merged_kv = {}
    if kv_captured:
        merged_kv.update(kv_captured)
    merged_kv.update(captured)  # Q-hook data overrides for scoring layers
    compact_cache(cache, indices, model=model,
                  captured_kv=merged_kv if merged_kv else None)
    if kv_cleanup:
        kv_cleanup()
    gc.collect(); mx.clear_cache()
    metal_after_compact = mx.get_active_memory() / 1e9
    kv_offset_after = cache[0].offset

    logger.info(f"  Metal before compact: {metal_before_compact:.2f} GB")
    logger.info(f"  Metal after compact:  {metal_after_compact:.2f} GB")
    logger.info(f"  Memory saved:         {metal_before_compact - metal_after_compact:.2f} GB "
                f"({(metal_before_compact - metal_after_compact) / max(metal_before_compact, 0.01) * 100:.0f}%)")
    logger.info(f"  KV offset:            {kv_offset_before} -> {kv_offset_after}")
    logger.info(f"  Tokens kept:          {actual_kept}/{actual_tokens} "
                f"({actual_kept * 100 // actual_tokens}%)")

    # Generate from compacted cache using the PREFILL logits.
    # The prefill logits were computed with full context (before eviction),
    # which gives the best first-token prediction. Subsequent tokens
    # use the compacted cache.
    skv_tokens = []
    for _ in range(32):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        skv_tokens.append(token.item())
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)

    skv_text = tokenizer.decode(skv_tokens)
    logger.info(f"  SnapKV output:        {skv_text[:80]!r}")

    # Check NIAH
    needle_found = needle.lower() in skv_text.lower()
    base_needle = needle.lower() in base_text.lower() if base_text else None

    # Token agreement (only if baseline was run)
    if base_tokens:
        agree = sum(1 for a, b in zip(base_tokens, skv_tokens) if a == b)
        agreement = agree / max(len(base_tokens), 1) * 100
    else:
        agreement = -1  # baseline skipped

    if base_needle is not None:
        logger.info(f"  Needle in baseline:   {base_needle}")
    logger.info(f"  Needle in SnapKV:     {needle_found}")
    if agreement >= 0:
        logger.info(f"  Token agreement:      {agreement:.0f}%")

    # Memory savings check
    memory_saved_pct = (metal_before_compact - metal_after_compact) / max(metal_before_compact, 0.01) * 100

    result = {
        "context_tokens": actual_tokens,
        "keep_ratio": keep_ratio,
        "keep_count": keep_count,
        "scoring": scoring_label,
        "kv_mode": kv_mode,
        "actual_kept": actual_kept,
        "metal_before_gb": round(metal_before_compact, 2),
        "metal_after_gb": round(metal_after_compact, 2),
        "memory_saved_pct": round(memory_saved_pct, 1),
        "kv_before": kv_offset_before,
        "kv_after": kv_offset_after,
        "needle_found": needle_found,
        "base_needle": base_needle,
        "agreement": round(agreement, 1),
        "base_text": base_text[:100],
        "skv_text": skv_text[:100],
    }

    status = "PASS" if needle_found else "FAIL"
    mem_status = "PASS" if memory_saved_pct > 5 else "FAIL"
    logger.info(f"  NIAH: {status}  |  Memory savings: {mem_status} ({memory_saved_pct:.0f}%)")

    del cache; gc.collect(); mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="SnapKV compaction benchmark")
    parser.add_argument("--context", type=int, default=4096,
                        help="Context length in tokens (default: 4096)")
    parser.add_argument("--keep-ratios", type=str, default="0.25,0.50,0.75",
                        help="Comma-separated keep ratios to test")
    parser.add_argument("--caote", action="store_true", default=False,
                        help="Use CAOTE scoring (attention × value distinctiveness)")
    parser.add_argument("--segment-size", type=int, default=0,
                        help="BUZZ segmented eviction: per-segment top-K (0=global)")
    parser.add_argument("--skip-baseline", action="store_true", default=False,
                        help="Skip baseline generation (for 64K+ where O(n²) is too slow)")
    parser.add_argument("--kv-mode", choices=["fp16", "native"], default="fp16",
                        help="KV cache mode: fp16 (default) or native (3-bit QuantizedKVCache)")
    args = parser.parse_args()

    keep_ratios = [float(r) for r in args.keep_ratios.split(",")]

    t0 = time.perf_counter()

    from mlx_lm import load
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch

    logger.info(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)
    apply_prefill_last_logit_patch(model)

    load_time = time.perf_counter() - t0
    logger.info(f"Model loaded in {load_time:.1f}s")

    # Warmup
    from mlx_lm.models.cache import KVCache
    n_layers = len(model.layers)
    warmup_cache = [KVCache() for _ in range(n_layers)]
    warmup_ids = tokenizer.encode("Hello")
    logits = model(mx.array([warmup_ids]), cache=warmup_cache)
    mx.eval(logits)
    for _ in range(4):
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        logits = model(tok.reshape(1, 1), cache=warmup_cache)
        mx.eval(logits)
    del warmup_cache, logits; gc.collect(); mx.clear_cache()
    logger.info("Metal warmup done")

    results = []
    for ratio in keep_ratios:
        result = run_test(model, tokenizer, args.context, ratio,
                          use_caote=args.caote,
                          segment_size=args.segment_size,
                          skip_baseline=args.skip_baseline,
                          kv_mode=args.kv_mode)
        results.append(result)

    elapsed = time.perf_counter() - t0

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SNAPKV COMPACTION BENCHMARK RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Context: {args.context} tokens")
    logger.info(f"Elapsed: {elapsed:.1f}s")
    logger.info("")

    all_niah_pass = True
    any_memory_saved = False
    for r in results:
        niah_s = "PASS" if r["needle_found"] else "FAIL"
        mem_s = f"{r['memory_saved_pct']:.0f}%"
        logger.info(f"  Keep {r['keep_ratio']:.0%}: "
                    f"NIAH={niah_s}  "
                    f"Agreement={r['agreement']:.0f}%  "
                    f"Memory saved={mem_s}  "
                    f"Metal {r['metal_before_gb']:.1f}->{r['metal_after_gb']:.1f} GB")
        if not r["needle_found"]:
            all_niah_pass = False
        if r["memory_saved_pct"] > 5:
            any_memory_saved = True

    overall = "PASS" if all_niah_pass and any_memory_saved else "FAIL"
    logger.info(f"\nOverall: {overall}")
    logger.info(f"  NIAH all pass: {all_niah_pass}")
    logger.info(f"  Memory savings: {any_memory_saved}")

    # Save results
    out_path = Path("research/snapkv_compaction_bench.json")
    out_path.write_text(json.dumps({
        "results": results,
        "model": MODEL_ID,
        "context": args.context,
        "elapsed_s": round(elapsed, 1),
        "all_niah_pass": all_niah_pass,
        "any_memory_saved": any_memory_saved,
        "overall": overall,
    }, indent=2))
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
