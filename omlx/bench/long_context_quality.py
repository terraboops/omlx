# SPDX-License-Identifier: Apache-2.0
"""Long-context quality validation for TQ3.

Injects code filler to reach target context length, appends a coding
problem, checks if the model still produces correct code.

Validates that TQ3's Givens rotation doesn't accumulate floating-point
drift over long distances.
"""

from __future__ import annotations

import gc
import logging
import sys
import time

import mlx.core as mx

logger = logging.getLogger(__name__)

# Realistic code filler — diverse enough to exercise the codebook
CODE_FILLER = '''"""Data processing utilities module."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

class DataProcessor:
    """Generic data processor with configurable pipeline stages."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stages: List[Callable] = []
        self.metrics: Dict[str, int] = {"processed": 0, "errors": 0}

    def add_stage(self, func: Callable) -> "DataProcessor":
        self.stages.append(func)
        return self

    def process(self, item: Any) -> Optional[Any]:
        for stage in self.stages:
            try:
                item = stage(item)
                if item is None:
                    break
            except Exception as e:
                self.metrics["errors"] += 1
                return None
        self.metrics["processed"] += 1
        return item


def batch_iterator(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """Yield batches of items."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_config(path: str) -> Dict[str, Any]:
    """Load JSON configuration from disk."""
    with open(path) as f:
        return json.load(f)


def merge_dicts(*dicts: Dict) -> Dict:
    """Shallow-merge multiple dicts, later keys override earlier."""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def compose(*functions: Callable) -> Callable:
    """Compose functions left-to-right."""
    def composed(x):
        for f in functions:
            x = f(x)
        return x
    return composed


def chunked(seq: List, n: int) -> List[List]:
    """Split sequence into chunks of size n."""
    return [seq[i:i+n] for i in range(0, len(seq), n)]

'''


def build_long_prompt(tokenizer, target_tokens: int, problem: str) -> str:
    """Build a prompt of target_tokens length with problem at the end."""
    # Calculate filler needed
    filler_tokens = len(tokenizer.encode(CODE_FILLER))
    problem_tokens = len(tokenizer.encode(problem))
    filler_needed = target_tokens - problem_tokens
    reps = max(1, filler_needed // filler_tokens)

    # Build the prompt: many copies of filler, then the problem
    prompt = (CODE_FILLER * reps) + "\n\n# ============\n# New task:\n# ============\n" + problem
    return prompt


TEST_PROBLEMS = [
    {
        "prompt": "def factorial(n: int) -> int:\n    \"\"\"Return n! (n factorial). Assume n >= 0.\"\"\"\n",
        "test": "assert factorial(0) == 1 and factorial(5) == 120 and factorial(1) == 1",
        "name": "factorial",
    },
    {
        "prompt": "def is_even(n: int) -> bool:\n    \"\"\"Return True if n is even.\"\"\"\n",
        "test": "assert is_even(4) == True and is_even(7) == False and is_even(0) == True",
        "name": "is_even",
    },
    {
        "prompt": "def reverse_list(lst: list) -> list:\n    \"\"\"Return the list reversed.\"\"\"\n",
        "test": "assert reverse_list([1,2,3]) == [3,2,1] and reverse_list([]) == []",
        "name": "reverse_list",
    },
]


def run_long_context_test(model, tokenizer, cache_factory, n_layers: int,
                          context_length: int, prefill_chunk: int = 2048) -> dict:
    """Run coding problems after a long context preamble.

    Returns dict with per-problem results and pass rate.
    """
    results = []

    for problem in TEST_PROBLEMS:
        prompt_text = build_long_prompt(tokenizer, context_length, problem["prompt"])
        tokens = tokenizer.encode(prompt_text)[:context_length + 200]  # Allow for problem

        logger.info(f"  {problem['name']} @ {len(tokens):,} tokens...")

        # Create fresh cache
        cache = cache_factory(n_layers)

        # Chunked prefill
        from omlx.memory_budget import compute_budget, compute_dequant_chunk_size
        from omlx.streaming_attention import set_chunk_hint

        for cs in range(0, len(tokens), prefill_chunk):
            ce = min(cs + prefill_chunk, len(tokens))
            budget = compute_budget(model_gb=17.2, active_kv_gb=0, decoding=False)
            adaptive = compute_dequant_chunk_size(
                query_len=ce - cs, num_query_heads=32, head_dim=128,
                num_layers=48, budget=budget,
            )
            set_chunk_hint(adaptive)
            logits = model(mx.array([tokens[cs:ce]]), cache=cache)
            mx.eval(logits)
            mx.synchronize()
            mx.clear_cache()

        # Generate completion (~80 tokens)
        generated = []
        for _ in range(80):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tid = token.item()
            generated.append(tid)
            text = tokenizer.decode(generated)
            # Stop on double newline or new def
            if "\n\n" in text or "\ndef " in text:
                break
            logits = model(token.reshape(1, 1), cache=cache)
            mx.eval(logits)

        completion = tokenizer.decode(generated).split("\n\n")[0].split("\ndef ")[0]
        full_code = problem["prompt"] + completion

        # Execute and test
        try:
            glob = {}
            exec(full_code, glob)
            exec(problem["test"], glob)
            ok = True
            err = ""
        except Exception as e:
            ok = False
            err = f"{type(e).__name__}: {str(e)[:100]}"

        results.append({
            "name": problem["name"],
            "passed": ok,
            "error": err,
            "completion": completion[:80],
        })
        status = "PASS" if ok else "FAIL"
        logger.info(f"    {status}: {completion[:60]}")

        del cache
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    logger.info(f"  Pass rate: {pass_rate*100:.0f}% ({sum(1 for r in results if r['passed'])}/{len(results)})")

    return {
        "context_length": context_length,
        "pass_rate": pass_rate,
        "results": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Long-context quality validation")
    parser.add_argument("--contexts", nargs="+", type=int, default=[16384, 32768])
    parser.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    from omlx.patches.vertical_eval import apply_vertical_eval_patch

    apply_turboquant_attention_patch()
    model, tokenizer = load(args.model)
    apply_prefill_last_logit_patch(model)
    apply_vertical_eval_patch(model)
    n_layers = model.args.num_hidden_layers

    def tq_factory(n):
        return [KVCache() if i == 0 else TurboQuantKVCache(bits=3) for i in range(n)]

    all_results = []
    for ctx in args.contexts:
        logger.info(f"\n=== Context: {ctx:,} tokens ===")
        result = run_long_context_test(model, tokenizer, tq_factory, n_layers, ctx)
        all_results.append(result)

    logger.info("\n=== SUMMARY ===")
    for r in all_results:
        logger.info(f"  {r['context_length']:>7,} tok: {r['pass_rate']*100:>3.0f}% pass")


if __name__ == "__main__":
    main()
