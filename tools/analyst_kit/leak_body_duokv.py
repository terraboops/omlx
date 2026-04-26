# SPDX-License-Identifier: Apache-2.0
"""Leak-loop body that exercises DuoKVCache without loading the model.

Synthesizes a fixed prefill, then runs decode-shaped appends in a loop.
Used by `python -m omlx.observability.leak_loop --target ...:decode_body`
to detect Metal peak climbing across iterations.

Usage:
    python -m omlx.observability.leak_loop \\
        --target tools.analyst_kit.leak_body_duokv:decode_body \\
        --iterations 600 \\
        --warmup 60 \\
        --csv research/analyst_runs/2026-04-26/leak.csv \\
        --gc-every 60 \\
        --clear-cache-every 60 \\
        --log-every 60

Slope analysis:
  - Slope flattens with --gc-every    → Python ref leak
  - Slope flattens with --clear-cache → Metal pool fragmentation
  - Slope persists with both          → true Metal leak
"""

from __future__ import annotations

import mlx.core as mx

from omlx.duo_kv_cache import DuoKVCache, load_duo_policy

# Module-level cache state — reused across leak-loop iterations.
_state = {"cache": None, "k_step": None, "v_step": None}

# Hypercar-realistic shapes (Qwen3-Coder-30B-A3B GQA=8)
B = 1
H_KV = 4
D = 128
DTYPE = mx.float16

# Initial context filled before the leak-loop steps. 8K is enough to
# trigger the trim path (capacity = sink + window = 260) without burning
# memory or time on prefill.
INITIAL_CONTEXT = 8192

_materialize = getattr(mx, "eval")


def _setup() -> None:
    """One-time setup: build cache, prime with initial context."""
    policy = load_duo_policy()
    cache = DuoKVCache(policy, layer_idx=0, n_kv_heads=H_KV)
    k_pre = mx.random.normal((B, H_KV, INITIAL_CONTEXT, D), dtype=DTYPE)
    v_pre = mx.random.normal((B, H_KV, INITIAL_CONTEXT, D), dtype=DTYPE)
    cache.update_and_fetch(k_pre, v_pre)
    mx.synchronize()
    # Single-token decode shapes, reused every iteration to avoid
    # measuring random-tensor allocation overhead.
    k_step = mx.random.normal((B, H_KV, 1, D), dtype=DTYPE)
    v_step = mx.random.normal((B, H_KV, 1, D), dtype=DTYPE)
    _materialize(k_step, v_step)
    _state["cache"] = cache
    _state["k_step"] = k_step
    _state["v_step"] = v_step


def decode_body(i: int) -> None:
    """One decode-step body: append one token + materialize the trim output."""
    if _state["cache"] is None:
        _setup()
    cache = _state["cache"]
    out_k, out_v = cache.update_and_fetch(_state["k_step"], _state["v_step"])
    _materialize(out_k, out_v)
