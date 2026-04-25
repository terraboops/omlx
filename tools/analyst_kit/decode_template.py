# SPDX-License-Identifier: Apache-2.0
"""Decode microbenchmark template — copy & adapt.

Loads the model once, instruments DuoKVCache + SnapKV via the tracer,
runs N decode steps at a target context length, dumps observability
JSON, and prints a heap diff.

This is the "first script the analyst writes" — pre-written so the
analyst can spend its budget hunting, not plumbing.

Usage
-----
    .venv/bin/python -m tools.analyst_kit.decode_template \\
        --context-tokens 16000 \\
        --decode-steps 200 \\
        --output research/analyst_runs/$(date +%F)/decode_16k.json

Adapt
-----
- For prefill-only profiling: set `--decode-steps 0`.
- To target a specific KV mode: pass `--kv-mode` (defaults to duo).
- To inspect a specific class: edit `INSTRUMENT_TARGETS` below.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("analyst.decode_template")


# Classes to wrap with the tracer. Edit to focus on suspected hot spots.
INSTRUMENT_TARGETS: List[Tuple[str, str | None]] = [
    ("omlx.duo_kv_cache", "DuoKVCache"),
    ("omlx.streaming_attention", None),
    ("omlx.patches.snapkv", None),
]


def build_long_prompt(target_tokens: int, tokenizer) -> str:
    """Synthesize a deterministic prompt of approximately N tokens."""
    # Reuse a paragraph that tokenizes ~consistently across English text.
    base = (
        "The high-performance inference stack uses compressed key-value caches "
        "to hold long context efficiently. DuoAttention partitions heads into "
        "retrieval (full fp16) and streaming (ring buffer) groups. SnapKV evicts "
        "tokens by attention score with re-RoPE position correction. "
    )
    text = base
    while len(tokenizer.encode(text)) < target_tokens:
        text += base
    # Trim to exactly target_tokens.
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--context-tokens", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--kv-mode", choices=["duo", "native", "tq3", "fp16"], default="duo")
    ap.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-trace", action="store_true",
                    help="Skip tracer monkey-patching (use pure observability calls only)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Imports are inside main so --help is fast.
    from omlx.observability import registry, reset
    from omlx.observability.heap import snapshot, snapshot_diff
    from omlx.observability.tracer import trace_class, trace_module, untrace

    reset()

    logger.info("loading model: %s (kv-mode=%s)", args.model, args.kv_mode)
    snap_load_before = snapshot("pre-load")
    from mlx_lm import load
    model, tokenizer = load(args.model)
    snap_load_after = snapshot("post-load")
    logger.info(snapshot_diff(snap_load_before, snap_load_after).format())

    # Install tracers.
    handles = []
    if not args.no_trace:
        for mod_name, cls_name in INSTRUMENT_TARGETS:
            try:
                mod = importlib.import_module(mod_name)
                if cls_name is None:
                    handles.append(trace_module(mod, mlx=True))
                else:
                    cls = getattr(mod, cls_name, None)
                    if cls is not None:
                        handles.append(trace_class(cls, mlx=True))
            except Exception as e:
                logger.warning("could not trace %s: %s", mod_name, e)
        logger.info("installed %d tracers", len(handles))

    # Build prompt.
    logger.info("building %d-token prompt", args.context_tokens)
    prompt = build_long_prompt(args.context_tokens, tokenizer)

    # Prefill.
    snap_pf_before = snapshot("pre-prefill")
    t_pf0 = time.perf_counter()
    from mlx_lm.generate import stream_generate
    gen = stream_generate(model, tokenizer, prompt=prompt, max_tokens=args.decode_steps)
    # Force prefill by pulling one token.
    iterator = iter(gen)
    first = next(iterator, None)
    pf_dt = time.perf_counter() - t_pf0
    snap_pf_after = snapshot("post-prefill")
    logger.info("prefill: %.3fs for ~%d ctx tokens", pf_dt, args.context_tokens)
    logger.info(snapshot_diff(snap_pf_before, snap_pf_after).format())

    # Decode loop.
    snap_dec_before = snapshot("pre-decode")
    t_dec0 = time.perf_counter()
    n_decoded = 1 if first is not None else 0
    for tok in iterator:
        n_decoded += 1
    dec_dt = time.perf_counter() - t_dec0
    snap_dec_after = snapshot("post-decode")
    logger.info(
        "decode: %d tokens in %.3fs → %.2f tok/s",
        n_decoded, dec_dt, n_decoded / dec_dt if dec_dt > 0 else 0.0,
    )
    logger.info(snapshot_diff(snap_dec_before, snap_dec_after).format())

    # Tear down tracers.
    for h in handles:
        untrace(h)

    # Final report.
    obs = registry.snapshot()
    obs["meta"] = {
        "model": args.model,
        "kv_mode": args.kv_mode,
        "context_tokens": args.context_tokens,
        "decode_steps_requested": args.decode_steps,
        "decode_steps_observed": n_decoded,
        "prefill_dt_s": round(pf_dt, 4),
        "decode_dt_s": round(dec_dt, 4),
        "decode_tok_per_s": round(n_decoded / dec_dt, 2) if dec_dt > 0 else 0.0,
        "load_diff": snapshot_diff(snap_load_before, snap_load_after).to_dict(),
        "prefill_diff": snapshot_diff(snap_pf_before, snap_pf_after).to_dict(),
        "decode_diff": snapshot_diff(snap_dec_before, snap_dec_after).to_dict(),
    }
    out_path.write_text(json.dumps(obs, indent=2))
    logger.info("wrote %s", out_path)
    registry.print_summary(top_n=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
