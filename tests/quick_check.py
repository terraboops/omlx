#!/usr/bin/env python3
"""Quick sanity check: load model, generate 10 tokens, report tok/s. ~10 seconds total."""

import sys, time, argparse
import mlx.core as mx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--sanitize-patch", action="store_true")
    p.add_argument("--tq-runtime", action="store_true")
    p.add_argument("--fp16-layers", type=int, default=0)
    args = p.parse_args()

    if args.sanitize_patch:
        from omlx.patches.granitemoehybrid_sanitize import apply_sanitize_patch
        apply_sanitize_patch()

    from mlx_lm import load
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"Load: {time.perf_counter()-t0:.1f}s | {mx.get_active_memory()/1e9:.1f}GB", flush=True)

    if args.tq_runtime:
        from omlx.patches.turboquant_runtime import apply_turboquant_runtime_patch
        n = apply_turboquant_runtime_patch(model, fp16_layers=args.fp16_layers)
        print(f"TQ rotation: {n} layers", flush=True)

    tokens = tokenizer.encode("The meaning of life is")
    cache = model.make_cache()

    # Prefill
    t0 = time.perf_counter()
    logits = model(mx.array([tokens]), cache=cache)
    mx.eval(logits)
    print(f"Prefill: {time.perf_counter()-t0:.2f}s ({len(tokens)} tokens)", flush=True)

    # Decode 10 tokens
    generated = []
    t0 = time.perf_counter()
    for _ in range(10):
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        generated.append(tok.item())
        logits = model(tok.reshape(1, 1), cache=cache)
        mx.eval(logits)
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(generated)
    print(f"Decode: {10/elapsed:.1f} tok/s | {text!r}", flush=True)
    print(f"Memory: {mx.get_active_memory()/1e9:.1f}GB", flush=True)

if __name__ == "__main__":
    main()
