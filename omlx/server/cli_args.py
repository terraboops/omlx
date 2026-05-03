# SPDX-License-Identifier: Apache-2.0
"""Hypercar server CLI argparse declarations (Task 355 Cycle 3 extraction
from hypercar_server.py).

Single function `build_parser()` returns the fully-configured
`ArgumentParser` for the hypercar server. Extracted here so the flag
surface (~90 lines of `add_argument` calls) is in its own module
instead of buried inside `main()`.

Adding a new flag: edit `build_parser()` here, then update the
flag-conditional dispatch in `hypercar_server.main()`.

Design note
-----------
The parser includes flags for ALL hypercar features even when they
require the user to also pass another flag (e.g., `--caote` requires
`--snapkv-keep > 0`). Validation of cross-flag dependencies happens
in `main()` after `parse_args()` — the parser itself is permissive
so users get the full surface from `--help`.
"""

from __future__ import annotations

import argparse

from omlx.model_constants import SERVER_DEFAULT_MODEL_ID


def build_parser(epilog: str | None = None) -> argparse.ArgumentParser:
    """Return the hypercar-server argparse.ArgumentParser.

    Args:
        epilog: optional epilog text (typically the host module's
                __doc__). When None, no epilog is set.

    Returns:
        A configured ArgumentParser. Call `.parse_args()` on it.
    """
    parser = argparse.ArgumentParser(
        description="Hypercar MLX server (OpenAI-compatible, TQ3 KV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("--model",
                        default=SERVER_DEFAULT_MODEL_ID,
                        help="Model to serve (HF ID or local path)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fp16-layers", type=int, default=1,
                        help="Number of fp16 layers (rest use TQ3). Default 1.")
    parser.add_argument("--kv-mode",
                        choices=["native", "tq3", "fp16", "duo", "duo-int4-fused"],
                        default="duo",
                        help="KV cache: duo (DuoAttention, best quality+speed), "
                             "duo-int4-fused (Task 281 — int4 quantize at decode + "
                             "fused Metal kernel; opt-in for memory savings at long "
                             "context, slower decode), native (MLX affine), "
                             "tq3 (WHT codebook), fp16 (no quant)")
    parser.add_argument("--duo-quantize", action="store_true", default=False,
                        help="DuoKV: use QuantizedKVCache(3-bit) for retrieval heads (saves ~8x memory, enables 1M context)")
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
    parser.add_argument("--ttt-router-policy", type=str, default=None,
                        help="Path to a DuoAttention policy JSON. When set, "
                             "installs a TTTHeadRouter SDPA monkey-patch "
                             "(Task 388 Phase 2). Without --ttt-router-blocks-dir "
                             "the router runs in BIT-EQUIVALENCE mode (no TTT "
                             "blocks loaded → all heads → original SDPA). With "
                             "the blocks dir, streaming-tagged heads with a "
                             "loaded TTT block route through the recurrence.")
    parser.add_argument("--ttt-router-blocks-dir", type=str, default=None,
                        help="Path to a TTT-Linear blocks directory written by "
                             "scripts/ttt_distill_single_head.py --save-block-dir "
                             "or by Phase 1 distillation. Requires "
                             "--ttt-router-policy to also be set.")
    parser.add_argument("--prefill-step-size", type=int, default=8192,
                        help="Tokens per prefill chunk (default 8192, was 2048 for TQ3)")
    parser.add_argument("--adaptive-chunk", action="store_true", default=False,
                        help="Adaptive prefill chunking: auto-adjusts chunk size based on "
                             "Metal memory pressure and throughput. Overrides --prefill-step-size.")
    parser.add_argument("--draft-model", type=str, default=None,
                        help="Speculative decoding drafter model (Task 341). When set, "
                             "stream_generate uses the drafter to propose tokens "
                             "verified by the main model. Drafter MUST use the same "
                             "tokenizer family as --model (mlx_lm does not translate "
                             "token IDs). Recommended: "
                             "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit (per "
                             "UAG-MLX-LM finding). Default off.")
    parser.add_argument("--num-draft-tokens", type=int, default=4,
                        help="Tokens drafted per main-model verification when "
                             "--draft-model is set (default 4 per Task 341 probe). "
                             "Higher N: more potential speedup at high acceptance "
                             "rate, more wasted compute at low rate. See "
                             "research/speculative_decoding_lever.md.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt-cache-size", type=int, default=8,
                        help="LRU prompt cache entries")
    parser.add_argument("--prompt-cache-bytes", type=int, default=20_000_000_000,
                        help="Max bytes across all cached prompts")
    parser.add_argument("--allowed-origins", default="*")
    parser.add_argument("--log-level", default="INFO")
    return parser
