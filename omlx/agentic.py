# SPDX-License-Identifier: Apache-2.0
"""Agentic inference loop with TQ3 stateful KV cache.

Demonstrates the three superpowers of TQ3 over native quantized KV:
  1. Fork: deepcopy cache, explore two paths, pick the winner
  2. Rewind: O(1) context undo, no re-prefill
  3. Save/Load: persist session to disk, resume in 1.5s

Usage:
    python -m omlx.agentic --demo fork
    python -m omlx.agentic --demo rewind
    python -m omlx.agentic --demo save-load
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import time
from pathlib import Path
from typing import List

import mlx.core as mx

logger = logging.getLogger("omlx.agentic")

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"


def _load():
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    from omlx.patches.vertical_eval import apply_vertical_eval_patch

    apply_turboquant_attention_patch()
    model, tokenizer = load(MODEL_ID)
    apply_prefill_last_logit_patch(model)
    apply_vertical_eval_patch(model)
    n = model.args.num_hidden_layers
    return model, tokenizer, n


def _make_tq3_cache(n_layers):
    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache
    return [KVCache() if i == 0 else TurboQuantKVCache(bits=3) for i in range(n_layers)]


def _generate(model, tokenizer, cache, prompt_tokens=None, prompt_text=None,
              max_tokens=64, stop_on="\n\n"):
    """Generate text, return (text, cache, logits)."""
    if prompt_tokens is None:
        prompt_tokens = tokenizer.encode(prompt_text)

    # Prefill
    logits = model(mx.array([prompt_tokens]), cache=cache)
    mx.eval(logits)

    # Decode
    generated = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tid = token.item()
        generated.append(tid)
        text = tokenizer.decode(generated)
        if stop_on and stop_on in text:
            break
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)

    return tokenizer.decode(generated), cache, logits


# ---------------------------------------------------------------------------
# Demo 1: Fork — explore two approaches, pick the winner
# ---------------------------------------------------------------------------

def demo_fork(model, tokenizer, n_layers):
    """Fork the KV cache to explore two coding approaches in parallel."""
    logger.info("=== DEMO: Context Forking ===\n")

    # Set up initial context
    prompt = "Write a Python function to compute the nth Fibonacci number.\n\n"
    logger.info(f"Prompt: {prompt.strip()}")

    cache = _make_tq3_cache(n_layers)
    tokens = tokenizer.encode(prompt)
    logits = model(mx.array([tokens]), cache=cache)
    mx.eval(logits)

    # Get offset before forking
    offsets = [c.offset if hasattr(c, 'offset') else 0 for c in cache]
    logger.info(f"Cache offset before fork: {offsets[1]} tokens\n")

    # Fork: deepcopy the cache
    t0 = time.perf_counter()
    cache_a = copy.deepcopy(cache)
    cache_b = copy.deepcopy(cache)
    fork_time = time.perf_counter() - t0
    logger.info(f"Fork time: {fork_time*1000:.1f}ms (two independent branches)\n")

    # Branch A: generate with "iterative" hint
    hint_a = "# Approach: iterative (for loop)\ndef fibonacci(n):\n"
    text_a, cache_a, _ = _generate(
        model, tokenizer, cache_a,
        prompt_text=hint_a, max_tokens=80, stop_on="\n\n",
    )
    logger.info(f"Branch A (iterative):\n{hint_a}{text_a}\n")

    # Branch B: generate with "recursive" hint
    hint_b = "# Approach: recursive with memoization\ndef fibonacci(n):\n"
    text_b, cache_b, _ = _generate(
        model, tokenizer, cache_b,
        prompt_text=hint_b, max_tokens=80, stop_on="\n\n",
    )
    logger.info(f"Branch B (recursive):\n{hint_b}{text_b}\n")

    # Test both
    for label, hint, text in [("A", hint_a, text_a), ("B", hint_b, text_b)]:
        code = hint + text.split("\n\n")[0]
        try:
            glob = {}
            exec(code, glob)
            assert glob["fibonacci"](10) == 55
            logger.info(f"Branch {label}: PASS (fibonacci(10) == 55)")
        except Exception as e:
            logger.info(f"Branch {label}: FAIL ({e})")

    # Show memory savings: we explored two paths without re-prefilling the prompt
    logger.info(f"\nBoth branches shared the {len(tokens)}-token prompt prefix.")
    logger.info(f"No re-prefill needed — fork is O(cache_size) copy, not O(prefill).")

    del cache_a, cache_b
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Demo 2: Rewind — undo a bad generation, try again
# ---------------------------------------------------------------------------

def demo_rewind(model, tokenizer, n_layers):
    """Rewind the cache after a bad generation, retry without re-prefill."""
    logger.info("=== DEMO: Context Rewind ===\n")

    # Build a code context
    context = """class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(('add', a, b, result))
        return result

    def subtract(self, a, b):
        result = a - b
        self.history.append(('subtract', a, b, result))
        return result

"""
    prompt = context + "    # Add a multiply method:\n    def multiply(self, a, b):\n"
    logger.info(f"Context: {len(tokenizer.encode(context))} tokens of Calculator class")

    cache = _make_tq3_cache(n_layers)
    tokens = tokenizer.encode(prompt)
    logits = model(mx.array([tokens]), cache=cache)
    mx.eval(logits)

    checkpoint_offset = cache[1].offset if hasattr(cache[1], 'offset') else 0
    logger.info(f"Checkpoint saved at offset: {checkpoint_offset}\n")

    # Generate attempt 1
    text1, cache, logits = _generate(
        model, tokenizer, cache,
        prompt_tokens=[], max_tokens=60, stop_on="\n\n",
    )
    post_gen_offset = cache[1].offset if hasattr(cache[1], 'offset') else 0
    logger.info(f"Attempt 1 ({post_gen_offset - checkpoint_offset} new tokens):")
    logger.info(f"  {text1.strip()[:100]}\n")

    # Rewind to checkpoint
    t0 = time.perf_counter()
    for c in cache:
        if hasattr(c, 'rewind_to'):
            c.rewind_to(checkpoint_offset)
    rewind_time = time.perf_counter() - t0
    rewound_offset = cache[1].offset if hasattr(cache[1], 'offset') else 0
    logger.info(f"Rewound to offset {rewound_offset} in {rewind_time*1000:.2f}ms")
    logger.info(f"Dropped {post_gen_offset - rewound_offset} tokens. No re-prefill needed.\n")

    # Generate attempt 2 (different temperature or prompt tweak)
    retry_hint = "        # Use the same pattern as add/subtract\n        result = a * b\n"
    text2, cache, _ = _generate(
        model, tokenizer, cache,
        prompt_text=retry_hint, max_tokens=60, stop_on="\n\n",
    )
    logger.info(f"Attempt 2 (after rewind + hint):")
    logger.info(f"  {retry_hint}{text2.strip()[:100]}\n")

    logger.info(f"Rewind cost: {rewind_time*1000:.2f}ms (vs ~5-30s for re-prefill at long context)")

    del cache
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Demo 3: Save/Load — persist session, resume instantly
# ---------------------------------------------------------------------------

def demo_save_load(model, tokenizer, n_layers):
    """Save a session to disk, reload it without re-prefill."""
    logger.info("=== DEMO: Session Persistence ===\n")

    # Build a meaningful context
    context = """# Project: Task Management API
# This is a large codebase with many utility functions.

from typing import List, Optional, Dict
from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

class TaskManager:
    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self._next_id = 1

    def create(self, title: str, description: str = "", tags: List[str] = None) -> Task:
        task = Task(id=self._next_id, title=title, description=description, tags=tags or [])
        self.tasks[self._next_id] = task
        self._next_id += 1
        return task

    def complete(self, task_id: int) -> bool:
        if task_id in self.tasks:
            self.tasks[task_id].completed = True
            return True
        return False

"""
    prompt = context + "\n# Add a method to filter tasks by tag:\n"
    tokens = tokenizer.encode(prompt)
    logger.info(f"Prefilling {len(tokens)} tokens...")

    cache = _make_tq3_cache(n_layers)
    t0 = time.perf_counter()
    logits = model(mx.array([tokens]), cache=cache)
    mx.eval(logits)
    prefill_time = time.perf_counter() - t0
    logger.info(f"Prefill: {len(tokens)/prefill_time:.0f} tok/s ({prefill_time:.2f}s)\n")

    # Save to disk
    save_path = "/tmp/hypercar_session.npz"
    t0 = time.perf_counter()
    for i, c in enumerate(cache):
        if hasattr(c, 'save_to_disk'):
            c.save_to_disk(f"/tmp/hypercar_session_layer{i}")
    save_time = time.perf_counter() - t0

    total_size = sum(
        Path(f"/tmp/hypercar_session_layer{i}.npz").stat().st_size
        for i in range(n_layers)
        if Path(f"/tmp/hypercar_session_layer{i}.npz").exists()
    ) / 1e6
    logger.info(f"Saved {n_layers} layer caches in {save_time:.2f}s ({total_size:.1f}MB)")

    # Clear everything
    del cache, logits
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    logger.info("Cache cleared from memory.\n")

    # Load from disk (simulates "next morning" resume)
    logger.info("Loading session from disk...")
    cache2 = _make_tq3_cache(n_layers)
    t0 = time.perf_counter()
    for i, c in enumerate(cache2):
        path = f"/tmp/hypercar_session_layer{i}.npz"
        if hasattr(c, 'load_from_disk') and Path(path).exists():
            c.load_from_disk(path)
    load_time = time.perf_counter() - t0

    loaded_offset = cache2[1].offset if hasattr(cache2[1], 'offset') else 0
    logger.info(f"Loaded {loaded_offset} tokens in {load_time:.2f}s")
    logger.info(f"Compare: original prefill was {prefill_time:.2f}s\n")

    # Generate from the restored cache
    text, _, _ = _generate(
        model, tokenizer, cache2,
        prompt_tokens=[], max_tokens=80, stop_on="\n\n",
    )
    logger.info(f"Generation from restored cache:")
    logger.info(f"  {text.strip()[:200]}\n")

    speedup = prefill_time / max(load_time, 0.001)
    logger.info(f"Resume speedup: {speedup:.1f}x faster than re-prefill")
    logger.info(f"At 256K context, this would be ~45min prefill → ~1.5s load")

    # Cleanup
    for i in range(n_layers):
        path = Path(f"/tmp/hypercar_session_layer{i}.npz")
        if path.exists():
            path.unlink()

    del cache2
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agentic TQ3 demos")
    parser.add_argument("--demo", choices=["fork", "rewind", "save-load", "all"],
                        default="all", help="Which demo to run")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading model...")
    model, tokenizer, n_layers = _load()
    logger.info(f"Model loaded: {mx.get_active_memory()/1e9:.1f}GB\n")

    demos = {
        "fork": demo_fork,
        "rewind": demo_rewind,
        "save-load": demo_save_load,
    }

    if args.demo == "all":
        for name, fn in demos.items():
            fn(model, tokenizer, n_layers)
            print()
    else:
        demos[args.demo](model, tokenizer, n_layers)


if __name__ == "__main__":
    main()
