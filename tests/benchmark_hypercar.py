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
import time
import sys

import mlx.core as mx


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
    print("\n--- Decode Benchmark ---")

    tokens = tokenizer.encode(prompt)
    cache = model.make_cache() if hasattr(model, 'make_cache') else None

    # Prefill
    x = mx.array([tokens])
    logits = model(x, cache=cache)
    mx.eval(logits)

    # Decode loop
    generated = []
    start = time.perf_counter()

    for i in range(max_tokens):
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
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

            start = time.perf_counter()
            logits = model(x)
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
    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    if args.sanitize_patch:
        from omlx.patches.granitemoehybrid_sanitize import apply_sanitize_patch
        apply_sanitize_patch()

    from mlx_lm import load
    model, tokenizer = load(args.model)

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

    if not args.skip_decode:
        benchmark_decode(model, tokenizer,
                        "The meaning of life is",
                        max_tokens=args.max_tokens)

    if not args.skip_stress:
        benchmark_context_stress(model, tokenizer, max_ctx=args.max_ctx)

    print(f"\nFinal memory: {mx.get_active_memory()/1e9:.1f}GB "
          f"(peak: {mx.get_peak_memory()/1e9:.1f}GB)")


if __name__ == "__main__":
    main()
