# SPDX-License-Identifier: Apache-2.0
"""Benchmark phases for oMLX hypercar validation.

Four phases, each building on the previous:
  Phase 1: Smoke — does it run? (memory, speed, coherence)
  Phase 2: Quality — does it think correctly? (NIAH, TQ3 fidelity)
  Phase 3: Stress — does it survive? (real code, sustained decode, leaks)
  Phase 4: Intelligence — is it smart? (SWE-bench, LiveCodeBench, etc.)

Target: >1M context, >GPT-4 intelligence, >25 tok/s decode.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

logger = logging.getLogger("omlx.bench")


# ---------------------------------------------------------------------------
# Phase 2: Quality — NIAH + TQ3 fidelity
# ---------------------------------------------------------------------------

NIAH_NEEDLE = "The secret passphrase for the vault is: CRYSTALLINE-AURORA-7742"
NIAH_QUESTION = "What is the secret passphrase for the vault?"
NIAH_ANSWER = "CRYSTALLINE-AURORA-7742"


def generate_haystack(tokenizer, target_tokens: int, needle: str, depth_pct: float = 50.0) -> str:
    """Generate a haystack of code with a needle buried at depth_pct%.

    Uses realistic Python code (not 'x=1' repeated) as the distractor content.
    """
    # Realistic code filler — diverse tokens, not repetitive
    code_blocks = [
        '''def fibonacci(n: int) -> list[int]:
    """Generate fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib[:n]
''',
        '''class DataProcessor:
    """Process and transform data records."""

    def __init__(self, config: dict):
        self.config = config
        self.records = []
        self._cache = {}

    def add_record(self, record: dict) -> None:
        self.records.append(record)
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._cache.clear()

    def get_summary(self) -> dict:
        if "summary" not in self._cache:
            self._cache["summary"] = {
                "count": len(self.records),
                "keys": list(set(k for r in self.records for k in r.keys())),
            }
        return self._cache["summary"]
''',
        '''import asyncio
from typing import AsyncIterator

async def stream_data(url: str, chunk_size: int = 1024) -> AsyncIterator[bytes]:
    """Stream data from a URL in chunks."""
    reader = await asyncio.open_connection(url, 443)
    try:
        while True:
            chunk = await reader.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        reader.close()
''',
        '''# Configuration constants
DATABASE_URL = "postgresql://localhost:5432/app_db"
REDIS_URL = "redis://localhost:6379/0"
MAX_CONNECTIONS = 100
TIMEOUT_SECONDS = 30
RETRY_COUNT = 3
LOG_LEVEL = "INFO"
BATCH_SIZE = 256
WORKER_THREADS = 8
CACHE_TTL = 3600
''',
        '''def merge_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left: list, right: list) -> list:
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
    ]

    # Build haystack by cycling through code blocks
    filler = "\n\n".join(code_blocks)
    filler_tokens = len(tokenizer.encode(filler))
    needle_tokens = len(tokenizer.encode(needle))

    # Calculate how many repetitions we need
    needed_filler = target_tokens - needle_tokens
    reps = (needed_filler // filler_tokens) + 1

    # Build the full haystack with needle at the right depth
    all_blocks = []
    for i in range(reps):
        all_blocks.append(filler)

    # Insert needle at depth_pct position
    needle_idx = max(1, int(len(all_blocks) * (depth_pct / 100.0)))
    all_blocks.insert(needle_idx, f"\n# IMPORTANT NOTE: {needle}\n")

    haystack = "\n\n".join(all_blocks)

    # Trim to target token count
    tokens = tokenizer.encode(haystack)[:target_tokens]
    return tokenizer.decode(tokens)


def phase2_niah(
    model,
    tokenizer,
    n_layers: int,
    context_length: int,
    cache_factory,
    prefill_chunk: int = 8192,
    depths: list[float] | None = None,
) -> dict:
    """Needle-in-a-Haystack test at various depths.

    Returns dict with results per depth.
    """
    if depths is None:
        depths = [10.0, 25.0, 50.0, 75.0, 90.0]

    results = {}
    for depth in depths:
        logger.info(f"  NIAH depth={depth}%...")

        # Build haystack with needle
        haystack = generate_haystack(tokenizer, context_length, NIAH_NEEDLE, depth)
        prompt = f"{haystack}\n\nQuestion: {NIAH_QUESTION}\nAnswer:"

        tokens = tokenizer.encode(prompt)[:context_length]

        # Chunked prefill
        cache = cache_factory(n_layers)
        for chunk_start in range(0, len(tokens), prefill_chunk):
            chunk_end = min(chunk_start + prefill_chunk, len(tokens))
            x = mx.array([tokens[chunk_start:chunk_end]])
            logits = model(x, cache=cache)
            mx.eval(logits)

        # Generate response (up to 32 tokens)
        generated = []
        for _ in range(32):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            generated.append(token.item())
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)

        response = tokenizer.decode(generated).strip()
        found = NIAH_ANSWER in response

        results[depth] = {
            "found": found,
            "response": response[:100],
            "context_tokens": len(tokens),
        }

        status = "PASS" if found else "FAIL"
        logger.info(f"    {status}: depth={depth}% response='{response[:60]}'")

        del cache
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    return results


def phase2_tq3_fidelity(
    model,
    tokenizer,
    n_layers: int,
    context_length: int = 4096,
    prefill_chunk: int = 8192,
) -> dict:
    """Compare TQ3 compressed output vs fp16 baseline.

    Runs the same prompt through both cache types and measures cosine similarity.
    """
    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache

    prompt = "Write a Python function that implements binary search in a sorted list:"
    tokens = tokenizer.encode(prompt)

    def get_logits(cache_list):
        x = mx.array([tokens])
        logits = model(x, cache=cache_list)
        mx.eval(logits)
        return logits[:, -1, :]  # Last token logits

    # fp16 baseline
    fp16_cache = [KVCache() for _ in range(n_layers)]
    fp16_logits = get_logits(fp16_cache)
    del fp16_cache

    # TQ3 compressed
    tq3_cache = [TurboQuantKVCache(bits=3, dequant_chunk_size=2048) for _ in range(n_layers)]
    tq3_logits = get_logits(tq3_cache)
    del tq3_cache

    # Compare
    cos_sim = (
        mx.sum(fp16_logits.reshape(-1) * tq3_logits.reshape(-1))
        / (mx.linalg.norm(fp16_logits.reshape(-1)) * mx.linalg.norm(tq3_logits.reshape(-1)) + 1e-8)
    ).item()

    # Top-1 agreement
    fp16_top1 = mx.argmax(fp16_logits, axis=-1).item()
    tq3_top1 = mx.argmax(tq3_logits, axis=-1).item()
    top1_match = fp16_top1 == tq3_top1

    logger.info(f"  TQ3 fidelity: cosine={cos_sim:.6f} top1_match={top1_match}")

    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    return {
        "cosine_similarity": cos_sim,
        "top1_match": top1_match,
        "fp16_top1_token": tokenizer.decode([fp16_top1]),
        "tq3_top1_token": tokenizer.decode([tq3_top1]),
    }


# ---------------------------------------------------------------------------
# Phase 3: Stress — sustained decode, memory stability, real content
# ---------------------------------------------------------------------------

def phase3_sustained_decode(
    model,
    tokenizer,
    n_layers: int,
    context_length: int,
    cache_factory,
    prefill_chunk: int = 8192,
    decode_tokens: int = 128,
    min_toks: float = 25.0,
) -> dict:
    """Sustained decode test after long-context prefill.

    Generates decode_tokens tokens and measures:
    - Average tok/s
    - Min tok/s (worst single token)
    - Memory stability (does active memory grow during decode?)
    """
    # Prefill with realistic code
    code = '''"""Module for processing and analyzing data."""
import json
from pathlib import Path
from typing import Any

def load_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())

'''
    code_tokens = tokenizer.encode(code)
    tokens = (code_tokens * ((context_length // len(code_tokens)) + 1))[:context_length]

    cache = cache_factory(n_layers)

    # Chunked prefill
    for chunk_start in range(0, len(tokens), prefill_chunk):
        chunk_end = min(chunk_start + prefill_chunk, len(tokens))
        x = mx.array([tokens[chunk_start:chunk_end]])
        logits = model(x, cache=cache)
        mx.eval(logits)

    mem_before = mx.get_active_memory() / 1e9

    # Sustained decode
    token_times = []
    generated_tokens = []
    for i in range(decode_tokens):
        t0 = time.perf_counter()
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        generated_tokens.append(token.item())
        x = token.reshape(1, 1)
        logits = model(x, cache=cache)
        mx.eval(logits)
        elapsed = time.perf_counter() - t0
        token_times.append(elapsed)

    mem_after = mx.get_active_memory() / 1e9

    toks_per_sec = [1.0 / t for t in token_times]
    avg_toks = sum(toks_per_sec) / len(toks_per_sec)
    min_toks_actual = min(toks_per_sec)
    max_toks_actual = max(toks_per_sec)
    mem_delta = mem_after - mem_before

    # Check for memory leak (>0.5GB growth during decode = leak)
    has_leak = mem_delta > 0.5

    generated_text = tokenizer.decode(generated_tokens)

    logger.info(
        f"  Sustained decode ({decode_tokens} tokens at {context_length:,} ctx): "
        f"avg={avg_toks:.1f} min={min_toks_actual:.1f} max={max_toks_actual:.1f} tok/s | "
        f"mem delta={mem_delta:+.2f}GB {'LEAK!' if has_leak else 'stable'}"
    )

    del cache
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    return {
        "context_length": context_length,
        "decode_tokens": decode_tokens,
        "avg_toks": round(avg_toks, 1),
        "min_toks": round(min_toks_actual, 1),
        "max_toks": round(max_toks_actual, 1),
        "mem_before_gb": round(mem_before, 2),
        "mem_after_gb": round(mem_after, 2),
        "mem_delta_gb": round(mem_delta, 3),
        "has_leak": has_leak,
        "speed_pass": min_toks_actual >= min_toks,
        "generated_preview": generated_text[:200],
    }


# ---------------------------------------------------------------------------
# Phase 4: Intelligence — external benchmark runners
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = {
    "evalplus": {
        "name": "EvalPlus (HumanEval+)",
        "install": "pip install evalplus",
        "run": (
            'evalplus.evaluate --model "{model}" '
            '--dataset humaneval --backend openai '
            '--base-url http://localhost:{port}/v1 --greedy'
        ),
        "what": "164 Python coding problems with 80x extra test cases",
        "gpt4_score": "90%+",
        "target": "90%+ (saturated benchmark)",
    },
    "bigcodebench": {
        "name": "BigCodeBench",
        "install": 'pip install "git+https://github.com/bigcode-project/bigcodebench.git"',
        "run": (
            'bigcodebench.generate --model {model} --split instruct --subset full '
            '--backend openai --base-url http://localhost:{port}/v1 && '
            'bigcodebench.evaluate --model {model} --split instruct --subset full'
        ),
        "what": "Real-world coding with library APIs (pandas, numpy, etc.)",
        "gpt4_score": "61.1% (Complete)",
        "target": ">50%",
    },
    "livecodebench": {
        "name": "LiveCodeBench",
        "install": "git clone https://github.com/LiveCodeBench/LiveCodeBench && cd LiveCodeBench && pip install -e .",
        "run": (
            'python -m lcb_runner.runner.main --model {model} '
            '--scenario codegeneration --evaluate'
        ),
        "what": "Fresh competitive programming (Codeforces/LeetCode, avoids data contamination)",
        "gpt4_score": "~35%",
        "target": ">40% (Qwen3-30B-A3B baseline: 40.3%)",
    },
    "swe_bench": {
        "name": "SWE-bench Verified",
        "install": "pip install swe-agent",
        "run": (
            'swe-agent run --model openai:{model} '
            '--base-url http://localhost:{port}/v1 '
            '--dataset princeton-nlp/SWE-bench_Verified --split test'
        ),
        "what": "Real GitHub issue resolution across 41 repos",
        "gpt4_score": "33.2%",
        "target": ">50% (Qwen3-30B-A3B baseline: ~51%)",
        "requires_docker": True,
    },
}


def phase4_print_setup_guide(port: int = 8000):
    """Print setup instructions for Phase 4 intelligence benchmarks.

    These benchmarks require external tools and a running oMLX server,
    so we print the commands rather than running them directly.
    """
    logger.info("\n=== PHASE 4: INTELLIGENCE BENCHMARKS ===")
    logger.info("These require a running oMLX server + external tools.\n")
    logger.info(f"Step 1: Start oMLX server:")
    logger.info(f"  omlx serve --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --port {port}\n")

    for key, cfg in BENCHMARK_CONFIGS.items():
        logger.info(f"--- {cfg['name']} ---")
        logger.info(f"  Tests: {cfg['what']}")
        logger.info(f"  GPT-4 score: {cfg['gpt4_score']}")
        logger.info(f"  Target: {cfg['target']}")
        if cfg.get("requires_docker"):
            logger.info(f"  Requires: Docker")
        logger.info(f"  Install: {cfg['install']}")
        logger.info(f"  Run: {cfg['run'].format(model='qwen3-coder-30b', port=port)}")
        logger.info("")


def phase4_evalplus_quick(
    model,
    tokenizer,
    n_layers: int,
    cache_factory,
    n_problems: int = 5,
) -> dict:
    """Quick inline EvalPlus-style test (no external deps).

    Runs a small set of HumanEval-like problems to sanity-check coding ability
    before deploying the full benchmark suite.
    """
    problems = [
        {
            "prompt": 'def add(a: int, b: int) -> int:\n    """Return the sum of a and b."""\n',
            "test": "assert add(2, 3) == 5 and add(-1, 1) == 0 and add(0, 0) == 0",
            "name": "add",
        },
        {
            "prompt": 'def reverse_string(s: str) -> str:\n    """Return the reversed string."""\n',
            "test": 'assert reverse_string("hello") == "olleh" and reverse_string("") == ""',
            "name": "reverse_string",
        },
        {
            "prompt": 'def is_palindrome(s: str) -> bool:\n    """Check if string is a palindrome (case insensitive)."""\n',
            "test": 'assert is_palindrome("racecar") == True and is_palindrome("hello") == False and is_palindrome("Aba") == True',
            "name": "is_palindrome",
        },
        {
            "prompt": 'def factorial(n: int) -> int:\n    """Return the factorial of n. n >= 0."""\n',
            "test": "assert factorial(0) == 1 and factorial(5) == 120 and factorial(1) == 1",
            "name": "factorial",
        },
        {
            "prompt": 'def flatten(lst: list) -> list:\n    """Flatten a nested list."""\n',
            "test": "assert flatten([[1,2],[3,[4,5]]]) == [1,2,3,4,5] and flatten([]) == []",
            "name": "flatten",
        },
    ]

    results = []
    for prob in problems[:n_problems]:
        # Generate completion
        tokens = tokenizer.encode(prob["prompt"])
        cache = cache_factory(n_layers)
        x = mx.array([tokens])
        logits = model(x, cache=cache)
        mx.eval(logits)

        generated = []
        for _ in range(128):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tok_id = token.item()
            generated.append(tok_id)
            text_so_far = tokenizer.decode(generated)
            # Stop on double newline or new function def
            if "\n\n" in text_so_far or "\ndef " in text_so_far:
                break
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)

        completion = tokenizer.decode(generated).split("\n\n")[0].split("\ndef ")[0]
        full_code = prob["prompt"] + completion

        # Test the generated code
        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            exec(prob["test"], exec_globals)
            passed = True
        except Exception as e:
            passed = False

        results.append({
            "name": prob["name"],
            "passed": passed,
            "completion": completion[:100],
        })

        status = "PASS" if passed else "FAIL"
        logger.info(f"  {status}: {prob['name']}")

        del cache
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    pass_rate = sum(1 for r in results if r["passed"]) / len(results)
    logger.info(f"  Quick EvalPlus: {pass_rate*100:.0f}% ({sum(1 for r in results if r['passed'])}/{len(results)})")

    return {
        "pass_rate": pass_rate,
        "results": results,
    }
