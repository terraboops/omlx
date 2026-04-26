# SPDX-License-Identifier: Apache-2.0
"""Traced decode benchmark — explicit DuoKV cache + tracer instrumentation.

Mirrors the production cache wiring from `omlx/bench/hypercar_bench.py:313-356`
so DuoKVCache is actually used (instead of the default mlx_lm cache).
Installs `trace_class(DuoKVCache, mlx=True)` and similar before the first
forward pass, then runs prefill + decode at the requested context.

Usage:
    .venv/bin/python -m tools.analyst_kit.traced_decode \\
        --context-tokens 16384 \\
        --decode-steps 64 \\
        --output research/analyst_runs/2026-04-26/traced_decode_16K.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import mlx.core as mx

logger = logging.getLogger("analyst.traced_decode")

_materialize = getattr(mx, "eval")


def build_long_prompt(target_tokens: int, tokenizer) -> str:
    base = (
        "The high-performance inference stack uses compressed key-value caches "
        "to hold long context efficiently. DuoAttention partitions heads into "
        "retrieval (full fp16) and streaming (ring buffer) groups. SnapKV evicts "
        "tokens by attention score with re-RoPE position correction. "
    )
    text = base
    while len(tokenizer.encode(text)) < target_tokens:
        text += base
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


def make_duokv_cache(n_layers: int):
    from omlx.duo_kv_cache import DuoKVCache, load_duo_policy
    policy = load_duo_policy()
    return [DuoKVCache(policy, layer_idx=i, bits=3) for i in range(n_layers)]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from omlx.observability import registry, reset
    from omlx.observability.heap import snapshot, snapshot_diff
    from omlx.observability.tracer import trace_class

    reset()

    # Install tracers BEFORE model load so any imports during loading
    # see the wrapped methods.
    from omlx.duo_kv_cache import DuoKVCache, StreamingKVCache
    h_duo = trace_class(DuoKVCache, prefix="duokv", mlx=True)
    h_str = trace_class(StreamingKVCache, prefix="streamkv", mlx=True)

    logger.info("loading model: %s", args.model)
    snap0 = snapshot("pre-load")
    from mlx_lm import load
    model, tokenizer = load(args.model)
    snap1 = snapshot("post-load")
    logger.info(snapshot_diff(snap0, snap1).format())

    # Apply prefill_last_logit patch (mirrors bench)
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)

    # Build prompt
    logger.info("building %d-token prompt", args.context_tokens)
    prompt = build_long_prompt(args.context_tokens, tokenizer)
    tokens = tokenizer.encode(prompt)
    logger.info("encoded %d tokens", len(tokens))

    # Build DuoKV cache list (explicit)
    n_layers = len(model.layers)
    cache = make_duokv_cache(n_layers)
    logger.info("built %d DuoKVCache layers", len(cache))

    # Chunked prefill (mirrors bench)
    PREFILL_CHUNK = 4096
    snap2 = snapshot("pre-prefill")
    t_pf0 = time.perf_counter()
    for chunk_start in range(0, len(tokens), PREFILL_CHUNK):
        chunk_end = min(chunk_start + PREFILL_CHUNK, len(tokens))
        x = mx.array([tokens[chunk_start:chunk_end]])
        logits = model(x, cache=cache)
        _materialize(logits)
    pf_dt = time.perf_counter() - t_pf0
    snap3 = snapshot("post-prefill")
    logger.info("prefill: %.3fs for %d tokens (%.1f tok/s)",
                pf_dt, len(tokens), len(tokens) / pf_dt if pf_dt > 0 else 0)
    logger.info(snapshot_diff(snap2, snap3).format())

    # Decode loop
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    snap4 = snapshot("pre-decode")
    t_dec0 = time.perf_counter()
    n_decoded = 0
    for _ in range(args.decode_steps):
        x = next_token.reshape(1, 1)
        logits_d = model(x, cache=cache)
        next_token = mx.argmax(logits_d[:, -1, :], axis=-1)
        _materialize(next_token)
        n_decoded += 1
    dec_dt = time.perf_counter() - t_dec0
    snap5 = snapshot("post-decode")
    logger.info("decode: %d tokens in %.3fs (%.2f tok/s)",
                n_decoded, dec_dt, n_decoded / dec_dt if dec_dt > 0 else 0)
    logger.info(snapshot_diff(snap4, snap5).format())

    # Final report
    obs = registry.snapshot()
    obs["meta"] = {
        "model": args.model,
        "kv_mode": "duo",
        "context_tokens": args.context_tokens,
        "n_input_tokens": len(tokens),
        "decode_steps": n_decoded,
        "prefill_dt_s": round(pf_dt, 4),
        "decode_dt_s": round(dec_dt, 4),
        "prefill_tok_per_s": round(len(tokens) / pf_dt, 2) if pf_dt > 0 else 0.0,
        "decode_tok_per_s": round(n_decoded / dec_dt, 2) if dec_dt > 0 else 0.0,
        "load_diff": snapshot_diff(snap0, snap1).to_dict(),
        "prefill_diff": snapshot_diff(snap2, snap3).to_dict(),
        "decode_diff": snapshot_diff(snap4, snap5).to_dict(),
    }
    out_path.write_text(json.dumps(obs, indent=2))
    logger.info("wrote %s", out_path)
    registry.print_summary(top_n=40)
    return 0


if __name__ == "__main__":
    sys.exit(main())
