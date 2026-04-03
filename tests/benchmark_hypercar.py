#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark script for the Hypercar build.

Measures:
  - Prefill speed (tok/s) at various context lengths
  - Decode speed (tok/s) for generation
  - Peak memory usage
  - Context window stress test
"""

import argparse
import logging
import time
import sys

import mlx.core as mx

logger = logging.getLogger(__name__)


def sample_token(logits: mx.array, temperature: float = 0.7, top_p: float = 0.9) -> mx.array:
    """Sample a token with temperature and top-p (nucleus) sampling."""
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)

    logits = logits / temperature

    # Top-p filtering
    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = mx.cumsum(sorted_probs, axis=-1)

    # Zero out tokens beyond top_p
    cutoff = (cumulative - sorted_probs) >= top_p
    sorted_probs = mx.where(cutoff, 0.0, sorted_probs)

    # Renormalize and sample
    sorted_probs = sorted_probs / sorted_probs.sum(axis=-1, keepdims=True)
    token_idx = mx.random.categorical(mx.log(sorted_probs + 1e-10))
    return mx.take_along_axis(sorted_indices, token_idx[..., None], axis=-1).squeeze(-1)


def benchmark_prefill(model, tokenizer, context_lengths, warmup=True):
    """Measure prefill throughput at various context lengths."""
    print("\n--- Prefill Benchmark ---")
    print(f"{'Context':>10} {'Time (ms)':>12} {'Tok/s':>10} {'Memory (GB)':>12}")
    print("-" * 50)

    # Build a long prompt by repeating text
    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    base_tokens = tokenizer.encode(base_text)

    if warmup:
        # Warmup pass
        x = mx.array([base_tokens[:32]])
        model(x)
        mx.eval(model(x))

    for ctx_len in context_lengths:
        # Pad/repeat tokens to target length
        tokens = (base_tokens * ((ctx_len // len(base_tokens)) + 1))[:ctx_len]
        x = mx.array([tokens])

        # Clear cache before measurement
        mx.synchronize()
        mx.clear_cache()

        start = time.perf_counter()
        logits = model(x)
        mx.eval(logits)
        elapsed = time.perf_counter() - start

        tok_per_sec = ctx_len / elapsed
        mem_gb = mx.get_active_memory() / 1e9

        print(f"{ctx_len:>10,} {elapsed*1000:>12.1f} {tok_per_sec:>10,.0f} {mem_gb:>12.1f}")

        del logits, x
        mx.synchronize()
        mx.clear_cache()


def benchmark_decode(model, tokenizer, prompt, max_tokens=50):
    """Measure decode throughput (autoregressive generation)."""
    print("\n--- Decode Benchmark (Standard) ---")

    tokens = tokenizer.encode(prompt)
    cache = model.make_cache() if hasattr(model, 'make_cache') else None

    # Prefill
    x = mx.array([tokens])
    logits = model(x, cache=cache)
    mx.eval(logits)

    # Decode loop (temperature sampling to avoid repetition)
    generated = []
    start = time.perf_counter()

    for i in range(max_tokens):
        next_token = sample_token(logits[:, -1, :], temperature=0.7, top_p=0.9)
        mx.eval(next_token)
        generated.append(next_token.item())

        x = next_token.reshape(1, 1)
        logits = model(x, cache=cache)
        mx.eval(logits)

    elapsed = time.perf_counter() - start
    tok_per_sec = max_tokens / elapsed

    text = tokenizer.decode(generated)
    print(f"Prompt: {prompt!r}")
    print(f"Generated {max_tokens} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
    print(f"Output: {text!r}")
    print(f"Memory: {mx.get_active_memory()/1e9:.1f}GB")

    return tok_per_sec


def benchmark_medusa_decode(model, tokenizer, prompt, max_tokens=50, num_heads=3, draft_heads=None):
    """Measure decode throughput with Medusa speculative decoding."""
    label = "distilled" if draft_heads is not None else "random"
    print(f"\n--- Decode Benchmark (Medusa {num_heads} heads, {label}) ---")

    from omlx.medusa_decode import medusa_generate

    generated, stats = medusa_generate(
        model, tokenizer, prompt,
        max_tokens=max_tokens, num_heads=num_heads,
        draft_heads=draft_heads,
        temperature=0.7, top_p=0.9,
    )

    text = tokenizer.decode(generated)
    print(f"Prompt: {prompt!r}")
    print(f"Generated {stats.total_tokens} tokens in {stats.elapsed_seconds:.2f}s "
          f"({stats.tokens_per_second:.1f} tok/s)")
    print(f"Steps: {stats.total_steps} | Tok/step: {stats.tokens_per_step:.2f} | "
          f"Draft acceptance: {stats.acceptance_rate:.1%}")
    print(f"Output: {text!r}")
    print(f"Memory: {mx.get_active_memory()/1e9:.1f}GB")

    return stats


def benchmark_needle_haystack(model, tokenizer, context_sizes=[1024, 4096]):
    """Needle-in-a-haystack coherence test with strict exact-match scoring.

    The model must retrieve the exact needle phrase. Partial matches are
    scored but only EXACT counts as a pass.
    """
    print("\n--- Needle-in-Haystack Coherence Test ---")

    NEEDLE = "BLUE ELEPHANT 42"
    needle_sentence = f"The secret code is {NEEDLE}."
    hay_sentence = "This is filler text about various topics in science and technology. "
    # Use multiple different filler sentences to avoid pattern collapse
    filler_variants = [
        "The weather today is partly cloudy with a chance of rain in the afternoon. ",
        "Recent advances in renewable energy have made solar panels more efficient than ever. ",
        "The history of mathematics spans thousands of years across many cultures worldwide. ",
        "Modern architecture emphasizes sustainable design and integration with natural environments. ",
        "Ocean currents play a crucial role in regulating global climate and weather patterns. ",
    ]

    results = []
    print(f"  Needle: {NEEDLE!r}")
    print(f"  {'Context':>10} {'Score':>7} {'Status':>8} {'Answer'}")
    print(f"  {'-'*70}")

    for ctx_len in context_sizes:
        # Build diverse haystack
        hay_tokens = []
        for i in range(ctx_len * 2 // len(tokenizer.encode(hay_sentence))):
            variant = filler_variants[i % len(filler_variants)]
            hay_tokens.extend(tokenizer.encode(variant))
        hay_tokens = hay_tokens[:ctx_len]

        # Insert needle at ~40% position (not dead center — harder)
        needle_tokens = tokenizer.encode(needle_sentence)
        insert_pos = int(len(hay_tokens) * 0.4)
        full_tokens = hay_tokens[:insert_pos] + needle_tokens + hay_tokens[insert_pos:]
        full_tokens = full_tokens[:ctx_len]

        # Precise retrieval prompt
        question = (
            "\n\nBased on the text above, what is the exact secret code? "
            "Reply with ONLY the code, nothing else.\n\n"
            "The secret code is: "
        )
        q_tokens = tokenizer.encode(question)
        all_tokens = full_tokens + q_tokens

        x = mx.array([all_tokens])
        cache = model.make_cache() if hasattr(model, 'make_cache') else None
        if cache is not None and _tq_kv_bits > 0:
            cache = _apply_tq_to_cache(cache, _tq_kv_bits)

        # Prefill
        logits = model(x, cache=cache)
        mx.eval(logits)

        # Generate tokens (greedy for exact retrieval — no sampling)
        generated = []
        for _ in range(15):
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)
            t = tok.item()
            generated.append(t)
            # Stop on EOS or newline
            decoded = tokenizer.decode([t])
            if "<|end" in decoded or "\n" in decoded:
                break
            logits = model(tok.reshape(1, 1), cache=cache)
            mx.eval(logits)

        answer = tokenizer.decode(generated).strip().rstrip(".")

        # Strict scoring
        answer_upper = answer.upper()
        needle_upper = NEEDLE.upper()

        if needle_upper in answer_upper:
            score = "EXACT"
            status = "PASS"
        else:
            # Score individual components
            parts = NEEDLE.split()
            matched = sum(1 for p in parts if p.upper() in answer_upper)
            if matched == len(parts):
                score = "EXACT"
                status = "PASS"
            elif matched > 0:
                score = f"{matched}/{len(parts)}"
                status = "PARTIAL"
            else:
                score = "0/3"
                status = "FAIL"

        print(f"  {len(all_tokens):>10,} {score:>7} {status:>8} {answer!r}")
        results.append({"ctx": len(all_tokens), "score": score, "status": status, "answer": answer})

        del logits, x, cache
        mx.synchronize()
        mx.clear_cache()

    # Summary
    exact = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n  Result: {exact}/{total} EXACT matches")
    return results


def _apply_tq_to_cache(cache, tq_bits):
    """Convert KVCache entries to TurboQuantKVCache in-place."""
    if tq_bits <= 0:
        return cache
    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache
    converted = 0
    for i, c in enumerate(cache):
        if isinstance(c, KVCache):
            cache[i] = TurboQuantKVCache(bits=tq_bits)
            converted += 1
    if converted > 0:
        logger.info(f"TQ KV: converted {converted} cache layers to {tq_bits}-bit")
    return cache


# Module-level tq_kv_bits for cache creation (set by main)
_tq_kv_bits = 0


def benchmark_context_stress(model, tokenizer, max_ctx=None):
    """Find the maximum context length before OOM."""
    print("\n--- Context Window Stress Test ---")

    base_text = "The quick brown fox jumps over the lazy dog. " * 200
    base_tokens = tokenizer.encode(base_text)

    # Try increasing context sizes
    test_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 256000]
    if max_ctx:
        test_sizes = [s for s in test_sizes if s <= max_ctx]

    max_achieved = 0

    for ctx_len in test_sizes:
        tokens = (base_tokens * ((ctx_len // len(base_tokens)) + 1))[:ctx_len]
        x = mx.array([tokens])

        try:
            mx.synchronize()
            mx.clear_cache()

            # Create cache with optional TQ KV
            cache = model.make_cache() if hasattr(model, 'make_cache') else None
            if cache is not None and _tq_kv_bits > 0:
                cache = _apply_tq_to_cache(cache, _tq_kv_bits)

            start = time.perf_counter()
            logits = model(x, cache=cache)
            mx.eval(logits)
            elapsed = time.perf_counter() - start

            tok_per_sec = ctx_len / elapsed
            mem_gb = mx.get_active_memory() / 1e9
            max_achieved = ctx_len
            print(f"  {ctx_len:>10,} tokens: OK ({tok_per_sec:,.0f} tok/s, {mem_gb:.1f}GB)")

            del logits, x
            mx.synchronize()
            mx.clear_cache()

        except Exception as e:
            print(f"  {ctx_len:>10,} tokens: FAILED ({type(e).__name__}: {e})")
            del x
            mx.synchronize()
            mx.clear_cache()
            break

    print(f"\nMax context achieved: {max_achieved:,} tokens")
    return max_achieved


def main():
    parser = argparse.ArgumentParser(description="Hypercar Benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF ID")
    parser.add_argument("--max-tokens", type=int, default=50, help="Decode tokens to generate")
    parser.add_argument("--max-ctx", type=int, default=None, help="Max context to stress test")
    parser.add_argument("--skip-prefill", action="store_true", help="Skip prefill benchmark")
    parser.add_argument("--skip-decode", action="store_true", help="Skip decode benchmark")
    parser.add_argument("--skip-stress", action="store_true", help="Skip context stress test")
    parser.add_argument("--sanitize-patch", action="store_true", help="Apply TQ3.5 sanitize patch")
    parser.add_argument("--tq-runtime", action="store_true", help="Apply TQ3.5 runtime rotation")
    parser.add_argument("--fp16-layers", type=int, default=0, help="Layers to skip rotation")
    parser.add_argument("--expert-choice", action="store_true", help="Apply Expert-Choice MoE")
    parser.add_argument("--medusa", type=int, default=0, help="Medusa draft heads")
    parser.add_argument("--medusa-distill", type=int, default=0, help="Medusa distillation steps (0=random heads)")
    parser.add_argument("--starc", action="store_true", help="Apply STARC sparse attention")
    parser.add_argument("--tq-kv", type=int, default=0, help="TurboQuant KV cache bits (3 or 4, 0=disabled)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    if args.sanitize_patch:
        from omlx.patches.granitemoehybrid_sanitize import apply_sanitize_patch
        apply_sanitize_patch()

    from mlx_lm import load
    model, tokenizer = load(args.model)

    # Apply hypercar patches
    if args.tq_runtime:
        from omlx.patches.turboquant_runtime import apply_turboquant_runtime_patch
        n = apply_turboquant_runtime_patch(model, fp16_layers=args.fp16_layers)
        print(f"  TQ3.5 runtime rotation: {n} layers patched")

    if args.expert_choice:
        import omlx.patches.expert_choice_router as ecr
        ecr._patch_applied = False
        from omlx.patches.expert_choice_router import apply_expert_choice_patch
        n = apply_expert_choice_patch(model, capacity_factor=1.2)
        print(f"  Expert-Choice MoE: {n} routers replaced")

    if args.medusa > 0:
        import omlx.patches.medusa_patch as mp
        mp._patch_applied = False
        from omlx.patches.medusa_patch import apply_medusa_patch
        apply_medusa_patch(model, num_heads=args.medusa)
        print(f"  Medusa: {args.medusa} draft heads attached")

    if args.starc:
        import omlx.patches.starc_attention as sa
        sa._PATCHED = False
        from omlx.patches.starc_attention import apply_starc_attention_patch
        apply_starc_attention_patch(budget_pct=0.15, min_seq_len=512)
        print(f"  STARC: sparse attention enabled (15% budget)")

    if args.tq_kv > 0:
        from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
        apply_turboquant_attention_patch()
        global _tq_kv_bits
        _tq_kv_bits = args.tq_kv
        print(f"  TQ KV cache: {args.tq_kv}-bit (attention patches installed)")

    print(f"Model type: {model.model_type}")
    if hasattr(model, 'args'):
        a = model.args
        print(f"  hidden={getattr(a, 'hidden_size', '?')}, "
              f"layers={getattr(a, 'num_hidden_layers', '?')}, "
              f"experts={getattr(a, 'num_local_experts', '?')}")
    print(f"Memory after load: {mx.get_active_memory()/1e9:.1f}GB "
          f"(peak: {mx.get_peak_memory()/1e9:.1f}GB)")

    if not args.skip_prefill:
        benchmark_prefill(model, tokenizer, [128, 512, 1024, 2048, 4096])

    # Medusa distillation (if requested, run before benchmarks)
    distilled_heads = None
    if args.medusa > 0 and args.medusa_distill > 0:
        print(f"\n--- Medusa Distillation ({args.medusa_distill} steps) ---")
        from omlx.medusa_distill import distill_medusa_heads
        distilled_heads = distill_medusa_heads(
            model, tokenizer,
            num_heads=args.medusa,
            num_steps=args.medusa_distill,
        )

    if not args.skip_decode:
        benchmark_decode(model, tokenizer,
                        "The meaning of life is",
                        max_tokens=args.max_tokens)
        if args.medusa > 0:
            benchmark_medusa_decode(model, tokenizer,
                                   "The meaning of life is",
                                   max_tokens=args.max_tokens,
                                   num_heads=args.medusa,
                                   draft_heads=distilled_heads)

    if not args.skip_stress:
        benchmark_needle_haystack(model, tokenizer, context_sizes=[1024, 4096, 16384])
        benchmark_context_stress(model, tokenizer, max_ctx=args.max_ctx)

    print(f"\nFinal memory: {mx.get_active_memory()/1e9:.1f}GB "
          f"(peak: {mx.get_peak_memory()/1e9:.1f}GB)")

    # Explicit cleanup to prevent Metal memory leaks between runs
    del model
    import gc
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    print("Metal memory released.")


if __name__ == "__main__":
    main()
