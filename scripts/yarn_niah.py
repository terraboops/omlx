# SPDX-License-Identifier: Apache-2.0
"""Long-context NIAH for Qwen3.6 with runtime YaRN rope_scaling override.

Qwen3.6 ships with `max_position_embeddings=262144` (256K) and no
`rope_scaling` config — the marketing "262K → 1M without fine-tuning" claim
relies on the user manually configuring YaRN at load time.

This script loads the model, replaces each attention module's RoPE with a
``YarnRoPE`` configured for the requested scaling factor, then runs an NIAH
test at the target context. It's intentionally bench-machinery-free.

Usage:
    python scripts/yarn_niah.py 512K
    python scripts/yarn_niah.py 1M --factor 4.0
"""

import argparse
import logging
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
from mlx_lm.models.rope_utils import YarnRoPE

sys.path.insert(0, ".")
import omlx.bench.hypercar_bench as hb
from omlx.bench.hypercar_bench import (
    NIAH_NEEDLE,
    NIAH_QUESTION,
    _build_code_haystack,
    _format_chat_prompt,
    _eos_token_ids,
    _make_cache,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def parse_context(s: str) -> int:
    s = s.upper().strip()
    if s.endswith("K"):
        return int(s[:-1]) * 1024
    if s.endswith("M"):
        return int(s[:-1]) * 1024 * 1024
    return int(s)


def patch_rope_with_yarn(model, scaling_factor: float,
                          original_max_position_embeddings: int):
    """Replace each attention module's RoPE with a YarnRoPE."""
    n_patched = 0
    for layer in model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        rope = getattr(attn, "rope", None)
        if rope is None:
            continue
        # Pull dims and base from the existing RoPE module so we don't have
        # to know the model's internal naming.
        dims = getattr(rope, "dims", None) or getattr(attn, "head_dim", None)
        base = getattr(rope, "base", None)
        if base is None:
            # Standard fallback for Qwen models
            base = 10_000_000.0
        traditional = getattr(rope, "traditional", False)
        new_max = int(scaling_factor * original_max_position_embeddings)
        attn.rope = YarnRoPE(
            dims=dims,
            max_position_embeddings=new_max,
            traditional=traditional,
            scaling_factor=scaling_factor,
            base=base,
            original_max_position_embeddings=original_max_position_embeddings,
        )
        n_patched += 1
    logger.info(f"Patched RoPE on {n_patched} attention layers "
                f"(factor={scaling_factor}, original_max={original_max_position_embeddings}, "
                f"new_max={new_max})")
    return n_patched


def run_niah(model, tokenizer, ctx_len: int, max_tokens: int = 32,
             prefill_chunk: int = 4096):
    haystack = _build_code_haystack(tokenizer, ctx_len - 200, NIAH_NEEDLE, 50.0)
    user_msg = (f"Here is some code:\n\n{haystack}\n\nBased on the code "
                f"above, answer this question: {NIAH_QUESTION}\n"
                "Respond with ONLY the answer, nothing else.")
    prompt = _format_chat_prompt(tokenizer, user_msg, enable_thinking=False)
    tokens = tokenizer.encode(prompt)[:ctx_len]
    logger.info(f"Prompt: {len(tokens)} tokens (target {ctx_len})")

    cache = _make_cache(len(model.layers), model)
    eos_ids = _eos_token_ids(tokenizer)

    PREFILL_CHUNK = prefill_chunk
    if prefill_chunk == 0:
        from omlx.patches.adaptive_prefill import AdaptivePrefillController
        controller = AdaptivePrefillController(
            target_metal_pct=0.65, min_chunk=512, max_chunk=4096,
            throughput_floor=80.0,
        )
        logger.info("Prefill chunk size: ADAPTIVE (controller-driven)")
    else:
        controller = None
        logger.info(f"Prefill chunk size: {PREFILL_CHUNK}")

    prefill_t0 = time.perf_counter()
    chunk_start = 0
    last_log_at = 0
    while chunk_start < len(tokens):
        if controller is not None:
            remaining = len(tokens) - chunk_start
            this_chunk = controller.next_chunk_size(remaining)
        else:
            this_chunk = PREFILL_CHUNK
        chunk_end = min(chunk_start + this_chunk, len(tokens))
        x = mx.array([tokens[chunk_start:chunk_end]])
        t0 = time.perf_counter()
        logits = model(x, cache=cache)
        mx.eval(logits)
        if controller is not None:
            elapsed_chunk = time.perf_counter() - t0
            tok_per_sec = (chunk_end - chunk_start) / max(elapsed_chunk, 1e-6)
            metal_gb = mx.get_active_memory() / 1e9
            controller.feedback(metal_gb, tok_per_sec)

        chunk_start = chunk_end
        # Log progress every ~64K tokens
        if chunk_start - last_log_at >= 65536 or chunk_start == len(tokens):
            elapsed = time.perf_counter() - prefill_t0
            logger.info(f"  prefill {chunk_start}/{len(tokens)} "
                        f"({elapsed:.0f}s, {chunk_start/elapsed:.0f} tok/s, "
                        f"this_chunk={this_chunk})")
            last_log_at = chunk_start
    prefill_time = time.perf_counter() - prefill_t0
    logger.info(f"Prefill: {prefill_time:.1f}s, "
                f"{len(tokens)/prefill_time:.1f} tok/s")
    logger.info(f"Metal peak: {mx.metal.get_peak_memory()/1e9:.1f} GB")

    # Decode the answer
    generated = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tok_id = token.item()
        if tok_id in eos_ids:
            break
        generated.append(tok_id)
        x = token.reshape(1, 1)
        logits = model(x, cache=cache)
        mx.eval(logits)

    response = tokenizer.decode(generated).strip()
    needle_value = NIAH_NEEDLE.split("is ")[-1]
    found = needle_value in response
    status = "✅ PASS" if found else "❌ FAIL"
    logger.info(f"\n{status} @ {ctx_len // 1024}K — {response[:80]!r}")
    return found, response, prefill_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ctx", help="Context size, e.g. '512K' or '1M'")
    parser.add_argument("--factor", type=float, default=None,
                        help="YaRN scaling factor (default: ceil(ctx / 256K))")
    parser.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    parser.add_argument("--max-answer-tokens", type=int, default=32)
    parser.add_argument("--kv-mode", default="native",
                        choices=["fp16", "native", "tq3"])
    parser.add_argument("--kv-bits", type=int, default=4, choices=[2, 4, 8])
    parser.add_argument("--prefill-chunk", type=int, default=4096,
                        help="Per-chunk prefill size. Smaller chunks reduce "
                             "the per-allocation Metal buffer demand at the "
                             "cost of more chunks. Pass 0 for adaptive — "
                             "uses AdaptivePrefillController (Metal-pressure "
                             "and throughput-feedback).")
    args = parser.parse_args()
    hb._KV_MODE = args.kv_mode
    hb.KV_BITS = args.kv_bits

    ctx_len = parse_context(args.ctx)
    original_max = 262144
    factor = args.factor or max(1.0, (ctx_len + original_max - 1) // original_max)

    logger.info(f"Loading {args.model}...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    logger.info(f"Load: {time.perf_counter() - t0:.1f}s")

    logger.info(f"Applying YaRN: factor={factor}, "
                f"original_max_position={original_max}, "
                f"target context {ctx_len // 1024}K")
    patch_rope_with_yarn(model, factor, original_max)

    found, response, prefill_time = run_niah(
        model, tokenizer, ctx_len, max_tokens=args.max_answer_tokens,
        prefill_chunk=args.prefill_chunk,
    )
    sys.exit(0 if found else 1)


if __name__ == "__main__":
    main()
