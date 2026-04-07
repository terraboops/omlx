# SPDX-License-Identifier: Apache-2.0
"""Gated benchmark for oMLX Hypercar — run before every commit.

Five phases, each gated: a failure stops the pipeline.

  Phase 0: Smoke       — model loads, 10 tokens, Metal < 20GB       (<30s)
  Phase 1: Coherence   — "2+2" => "4", "hello world" => "print"     (<60s)
  Phase 2: Code Intel  — 5 coding problems, exec + assert, >=60%    (<5min)
  Phase 3: NIAH        — needle retrieval at 4K and 8K context       (<5min)
  Phase 4: Memory      — background profiler, swap < 8GB, peak < 38GB
  Phase 5: Profiling   — write summary JSON

Usage:
    .venv/bin/python -m omlx.bench.hypercar_bench
    .venv/bin/python -m omlx.bench.hypercar_bench --quick   # Phase 0+1 only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

from omlx.bench.profiler import Profiler

logger = logging.getLogger("omlx.bench.hypercar")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
KV_BITS = 3
KV_GROUP_SIZE = 64
PREFILL_CHUNK = 4096

# Gate thresholds
MAX_METAL_LOAD_GB = 20.0
MAX_METAL_PEAK_GB = 38.0
MAX_SWAP_DELTA_GB = 8.0
MIN_CODE_PASS_RATE = 0.6

PROFILE_PATH = Path("/tmp/hypercar_profile.json")
RESULTS_PATH = Path("/tmp/hypercar_bench_results.json")

NIAH_NEEDLE = "The secret code is ALPHA-7749"
NIAH_QUESTION = "What is the secret code?"
NIAH_ANSWER = "ALPHA-7749"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class PhaseResult:
    name: str
    passed: bool
    elapsed_s: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    prefill_toks: float = 0.0
    decode_toks: float = 0.0


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _metal_gb() -> float:
    return mx.get_active_memory() / 1e9


def _peak_metal_gb() -> float:
    return mx.get_peak_memory() / 1e9


def _load_model():
    """Load model + tokenizer, apply prefill_last_logit patch."""
    from mlx_lm import load
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch

    model, tokenizer = load(MODEL_ID)
    apply_prefill_last_logit_patch(model)
    return model, tokenizer


def _make_cache(n_layers: int):
    """Create QuantizedKVCache(bits=3, group_size=64) for all layers."""
    from mlx_lm.models.cache import QuantizedKVCache
    return [QuantizedKVCache(group_size=KV_GROUP_SIZE, bits=KV_BITS)
            for _ in range(n_layers)]


def _generate(model, tokenizer, prompt: str, max_tokens: int = 64,
              cache=None) -> tuple[str, float, float]:
    """Generate tokens and return (text, prefill_tok/s, decode_tok/s)."""
    tokens = tokenizer.encode(prompt)
    n_layers = len(model.layers)

    if cache is None:
        cache = _make_cache(n_layers)

    # Chunked prefill
    t0 = time.perf_counter()
    for chunk_start in range(0, len(tokens), PREFILL_CHUNK):
        chunk_end = min(chunk_start + PREFILL_CHUNK, len(tokens))
        x = mx.array([tokens[chunk_start:chunk_end]])
        logits = model(x, cache=cache)
        mx.eval(logits)
    prefill_time = time.perf_counter() - t0
    prefill_toks = len(tokens) / prefill_time if prefill_time > 0 else 0.0

    # Decode
    generated = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tok_id = token.item()
        generated.append(tok_id)
        # Stop on EOS
        if hasattr(tokenizer, 'eos_token_id') and tok_id == tokenizer.eos_token_id:
            break
        x = token.reshape(1, 1)
        logits = model(x, cache=cache)
        mx.eval(logits)
    decode_time = time.perf_counter() - t0
    decode_toks = len(generated) / decode_time if decode_time > 0 else 0.0

    text = tokenizer.decode(generated)

    del cache
    gc.collect()
    mx.clear_cache()

    return text, prefill_toks, decode_toks


# ---------------------------------------------------------------------------
# Phase 0: Smoke
# ---------------------------------------------------------------------------

def phase0_smoke(model, tokenizer) -> PhaseResult:
    """Load model, generate 10 tokens, check Metal < 20GB."""
    t0 = time.perf_counter()

    metal_after_load = _metal_gb()
    logger.info(f"  Metal after load: {metal_after_load:.1f} GB")

    if metal_after_load > MAX_METAL_LOAD_GB:
        return PhaseResult(
            name="Phase 0: Smoke",
            passed=False,
            elapsed_s=time.perf_counter() - t0,
            details={"metal_after_load_gb": round(metal_after_load, 2),
                     "reason": f"Metal {metal_after_load:.1f}GB > {MAX_METAL_LOAD_GB}GB limit"},
        )

    # Generate 10 tokens
    text, prefill_toks, decode_toks = _generate(model, tokenizer,
                                                 "Hello, world!", max_tokens=10)
    logger.info(f"  Smoke output: {text[:80]!r}")
    logger.info(f"  Prefill: {prefill_toks:.0f} tok/s  Decode: {decode_toks:.1f} tok/s")

    return PhaseResult(
        name="Phase 0: Smoke",
        passed=True,
        elapsed_s=time.perf_counter() - t0,
        details={"metal_after_load_gb": round(metal_after_load, 2),
                 "output_preview": text[:80]},
        prefill_toks=prefill_toks,
        decode_toks=decode_toks,
    )


# ---------------------------------------------------------------------------
# Phase 1: Coherence
# ---------------------------------------------------------------------------

def phase1_coherence(model, tokenizer) -> PhaseResult:
    """Basic coherence: math + code generation."""
    t0 = time.perf_counter()
    checks = {}

    # Check 1: 2+2 must contain "4"
    text, p_toks, d_toks = _generate(model, tokenizer,
                                      "What is 2+2? Answer with just the number.",
                                      max_tokens=32)
    checks["math"] = {"output": text[:100], "passed": "4" in text}
    logger.info(f"  Math check: {'PASS' if checks['math']['passed'] else 'FAIL'} — {text[:60]!r}")

    # Check 2: hello world must contain "print"
    text2, p_toks2, d_toks2 = _generate(model, tokenizer,
                                          "Write hello world in Python. Just the code, nothing else.",
                                          max_tokens=64)
    checks["code"] = {"output": text2[:100], "passed": "print" in text2.lower()}
    logger.info(f"  Code check: {'PASS' if checks['code']['passed'] else 'FAIL'} — {text2[:60]!r}")

    all_passed = all(c["passed"] for c in checks.values())

    return PhaseResult(
        name="Phase 1: Coherence",
        passed=all_passed,
        elapsed_s=time.perf_counter() - t0,
        details=checks,
        prefill_toks=(p_toks + p_toks2) / 2,
        decode_toks=(d_toks + d_toks2) / 2,
    )


# ---------------------------------------------------------------------------
# Phase 2: Code Intelligence
# ---------------------------------------------------------------------------

CODING_PROBLEMS = [
    {
        "name": "factorial",
        "prompt": 'def factorial(n: int) -> int:\n    """Return the factorial of n. n >= 0."""\n',
        "test": "assert factorial(0) == 1 and factorial(5) == 120 and factorial(1) == 1",
    },
    {
        "name": "reverse_string",
        "prompt": 'def reverse_string(s: str) -> str:\n    """Return the reversed string."""\n',
        "test": 'assert reverse_string("hello") == "olleh" and reverse_string("") == ""',
    },
    {
        "name": "is_palindrome",
        "prompt": 'def is_palindrome(s: str) -> bool:\n    """Check if string is a palindrome (case insensitive)."""\n',
        "test": 'assert is_palindrome("racecar") == True and is_palindrome("hello") == False and is_palindrome("Aba") == True',
    },
    {
        "name": "fibonacci",
        "prompt": 'def fibonacci(n: int) -> list[int]:\n    """Return the first n Fibonacci numbers starting from 0."""\n',
        "test": "assert fibonacci(1) == [0] and fibonacci(5) == [0, 1, 1, 2, 3] and fibonacci(0) == []",
    },
    {
        "name": "flatten_list",
        "prompt": 'def flatten_list(lst: list) -> list:\n    """Flatten a nested list into a single list."""\n',
        "test": "assert flatten_list([[1,2],[3,[4,5]]]) == [1,2,3,4,5] and flatten_list([]) == []",
    },
]


def phase2_code_intelligence(model, tokenizer) -> PhaseResult:
    """Run 5 coding problems, exec + assert."""
    t0 = time.perf_counter()
    n_layers = len(model.layers)
    results = []

    for prob in CODING_PROBLEMS:
        tokens = tokenizer.encode(prob["prompt"])
        cache = _make_cache(n_layers)
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
            if "\n\n" in text_so_far or "\ndef " in text_so_far:
                break
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)

        completion = tokenizer.decode(generated).split("\n\n")[0].split("\ndef ")[0]
        full_code = prob["prompt"] + completion

        try:
            exec_globals = {}
            exec(full_code, exec_globals)
            exec(prob["test"], exec_globals)
            passed = True
        except Exception as e:
            passed = False
            logger.debug(f"    {prob['name']} error: {e}")

        results.append({"name": prob["name"], "passed": passed,
                        "completion": completion[:100]})
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {status}: {prob['name']}")

        del cache
        gc.collect()
        mx.clear_cache()

    pass_count = sum(1 for r in results if r["passed"])
    pass_rate = pass_count / len(results)
    gate_passed = pass_rate >= MIN_CODE_PASS_RATE

    logger.info(f"  Code intelligence: {pass_count}/{len(results)} "
                f"({pass_rate*100:.0f}%) — gate {'PASS' if gate_passed else 'FAIL'}")

    return PhaseResult(
        name="Phase 2: Code Intelligence",
        passed=gate_passed,
        elapsed_s=time.perf_counter() - t0,
        details={"pass_rate": pass_rate, "pass_count": pass_count,
                 "total": len(results), "results": results},
    )


# ---------------------------------------------------------------------------
# Phase 3: Needle in Haystack
# ---------------------------------------------------------------------------

def _build_code_haystack(tokenizer, target_tokens: int, needle: str,
                         depth_pct: float = 50.0) -> str:
    """Build a code haystack with a needle at depth_pct%."""
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

    def get_summary(self) -> dict:
        return {"count": len(self.records)}
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

    filler = "\n\n".join(code_blocks)
    filler_tokens = len(tokenizer.encode(filler))
    needle_tokens = len(tokenizer.encode(needle))

    needed_filler = target_tokens - needle_tokens
    reps = max(1, (needed_filler // filler_tokens) + 1)

    all_blocks = [filler for _ in range(reps)]
    needle_idx = max(1, int(len(all_blocks) * (depth_pct / 100.0)))
    all_blocks.insert(needle_idx, f"\n# IMPORTANT NOTE: {needle}\n")

    haystack = "\n\n".join(all_blocks)
    tokens = tokenizer.encode(haystack)[:target_tokens]
    return tokenizer.decode(tokens)


def phase3_niah(model, tokenizer) -> PhaseResult:
    """Needle-in-a-haystack at 4K and 8K context."""
    t0 = time.perf_counter()
    n_layers = len(model.layers)
    results = {}

    for ctx_len in [4096, 8192]:
        logger.info(f"  NIAH @ {ctx_len // 1024}K context...")

        haystack = _build_code_haystack(tokenizer, ctx_len, NIAH_NEEDLE, 50.0)
        prompt = f"{haystack}\n\nQuestion: {NIAH_QUESTION}\nAnswer:"
        tokens = tokenizer.encode(prompt)[:ctx_len]

        cache = _make_cache(n_layers)

        # Chunked prefill
        prefill_t0 = time.perf_counter()
        for chunk_start in range(0, len(tokens), PREFILL_CHUNK):
            chunk_end = min(chunk_start + PREFILL_CHUNK, len(tokens))
            x = mx.array([tokens[chunk_start:chunk_end]])
            logits = model(x, cache=cache)
            mx.eval(logits)
        prefill_time = time.perf_counter() - prefill_t0
        prefill_toks = len(tokens) / prefill_time if prefill_time > 0 else 0.0

        # Generate response
        generated = []
        decode_t0 = time.perf_counter()
        for _ in range(32):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            generated.append(token.item())
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)
        decode_time = time.perf_counter() - decode_t0
        decode_toks = len(generated) / decode_time if decode_time > 0 else 0.0

        response = tokenizer.decode(generated).strip()
        found = NIAH_ANSWER in response

        results[ctx_len] = {
            "found": found,
            "response": response[:100],
            "context_tokens": len(tokens),
            "prefill_toks": round(prefill_toks, 1),
            "decode_toks": round(decode_toks, 1),
        }

        status = "PASS" if found else "FAIL"
        logger.info(f"    {status}: {ctx_len // 1024}K — {response[:60]!r}")

        del cache
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    # Gate: must pass at 4K
    gate_passed = results[4096]["found"]

    return PhaseResult(
        name="Phase 3: Needle in Haystack",
        passed=gate_passed,
        elapsed_s=time.perf_counter() - t0,
        details=results,
        prefill_toks=results[4096].get("prefill_toks", 0.0),
        decode_toks=results[4096].get("decode_toks", 0.0),
    )


# ---------------------------------------------------------------------------
# Phase 4: Memory Profile (evaluated from profiler running across all phases)
# ---------------------------------------------------------------------------

def phase4_memory_check(profiler_result) -> PhaseResult:
    """Check memory gates from the profiler that ran across all phases."""
    t0 = time.perf_counter()

    metal_peak = profiler_result.metal_peak_gb
    swap_peak = profiler_result.swap_peak_gb

    gates = {
        "metal_peak_gb": round(metal_peak, 2),
        "swap_peak_gb": round(swap_peak, 2),
        "metal_limit_gb": MAX_METAL_PEAK_GB,
        "swap_limit_gb": MAX_SWAP_DELTA_GB,
    }

    metal_ok = metal_peak <= MAX_METAL_PEAK_GB
    swap_ok = swap_peak <= MAX_SWAP_DELTA_GB

    if not metal_ok:
        gates["reason"] = f"Metal peak {metal_peak:.1f}GB > {MAX_METAL_PEAK_GB}GB"
    if not swap_ok:
        gates["reason"] = f"Swap delta {swap_peak:.1f}GB > {MAX_SWAP_DELTA_GB}GB"

    logger.info(f"  Metal peak: {metal_peak:.1f} GB (limit {MAX_METAL_PEAK_GB}GB) "
                f"{'PASS' if metal_ok else 'FAIL'}")
    logger.info(f"  Swap peak:  {swap_peak:.1f} GB (limit {MAX_SWAP_DELTA_GB}GB) "
                f"{'PASS' if swap_ok else 'FAIL'}")

    return PhaseResult(
        name="Phase 4: Memory Profile",
        passed=metal_ok and swap_ok,
        elapsed_s=time.perf_counter() - t0,
        details=gates,
    )


# ---------------------------------------------------------------------------
# Phase 5: Profiling Snapshot (write results JSON)
# ---------------------------------------------------------------------------

def phase5_write_results(phases: List[PhaseResult], profiler_result,
                         total_elapsed: float) -> PhaseResult:
    """Write summary JSON to /tmp/hypercar_bench_results.json."""
    t0 = time.perf_counter()

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": _git_commit_hash(),
        "total_elapsed_s": round(total_elapsed, 2),
        "all_passed": all(p.passed for p in phases),
        "memory": {
            "metal_peak_gb": round(profiler_result.metal_peak_gb, 2),
            "metal_avg_gb": round(profiler_result.metal_avg_gb, 2),
            "swap_peak_gb": round(profiler_result.swap_peak_gb, 2),
            "rss_peak_gb": round(profiler_result.rss_peak_gb, 2),
        },
        "phases": {},
    }

    for p in phases:
        results["phases"][p.name] = {
            "passed": p.passed,
            "elapsed_s": round(p.elapsed_s, 2),
            "prefill_toks": round(p.prefill_toks, 1),
            "decode_toks": round(p.decode_toks, 1),
            "details": p.details,
        }

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  Results written to {RESULTS_PATH}")

    # Also write the raw profile
    profile_data = {
        "summary": profiler_result.summary(),
        "samples": [
            {
                "t": round(s.t, 2),
                "metal_active_gb": round(s.metal_active_gb, 2),
                "metal_peak_gb": round(s.metal_peak_gb, 2),
                "rss_gb": round(s.rss_gb, 2),
                "swap_gb": round(s.swap_gb, 2),
            }
            for s in profiler_result.samples
        ],
    }
    PROFILE_PATH.write_text(json.dumps(profile_data, indent=2))
    logger.info(f"  Profile written to {PROFILE_PATH}")

    return PhaseResult(
        name="Phase 5: Profiling Snapshot",
        passed=True,
        elapsed_s=time.perf_counter() - t0,
        details={"results_path": str(RESULTS_PATH),
                 "profile_path": str(PROFILE_PATH)},
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(phases: List[PhaseResult], total_elapsed: float):
    """Print a clear summary table."""
    print("\n" + "=" * 60)
    print("HYPERCAR BENCHMARK RESULTS")
    print("=" * 60)

    for p in phases:
        status = "PASS" if p.passed else "FAIL"
        marker = "  " if p.passed else ">>"
        print(f"  {marker} [{status}] {p.name:40s} ({p.elapsed_s:.1f}s)")

    print("-" * 60)
    all_passed = all(p.passed for p in phases)
    final = "ALL GATES PASSED" if all_passed else "GATE FAILURE"
    print(f"  {'  ' if all_passed else '>>'} [{final}]  Total: {total_elapsed:.1f}s")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hypercar gated benchmark — run before every commit",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Only run Phase 0 + 1 (smoke + coherence)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    total_t0 = time.perf_counter()
    phases: List[PhaseResult] = []

    # Start background profiler (Phase 4 — runs continuously)
    profiler = Profiler(sample_interval=1.0)
    profiler.start()

    try:
        # Load model once
        logger.info("Loading model: %s", MODEL_ID)
        load_t0 = time.perf_counter()
        model, tokenizer = _load_model()
        mx.eval(model.parameters())
        load_time = time.perf_counter() - load_t0
        logger.info(f"Model loaded in {load_time:.1f}s")

        # Phase 0: Smoke
        logger.info("\n=== Phase 0: Smoke ===")
        p0 = phase0_smoke(model, tokenizer)
        phases.append(p0)
        if not p0.passed:
            logger.error("Phase 0 FAILED — aborting")
            return _finish(phases, profiler, total_t0)

        # Phase 1: Coherence
        logger.info("\n=== Phase 1: Coherence ===")
        p1 = phase1_coherence(model, tokenizer)
        phases.append(p1)
        if not p1.passed:
            logger.error("Phase 1 FAILED — aborting")
            return _finish(phases, profiler, total_t0)

        if args.quick:
            logger.info("\n--quick mode: skipping Phase 2-3")
            return _finish(phases, profiler, total_t0)

        # Phase 2: Code Intelligence
        logger.info("\n=== Phase 2: Code Intelligence ===")
        p2 = phase2_code_intelligence(model, tokenizer)
        phases.append(p2)
        if not p2.passed:
            logger.error("Phase 2 FAILED — aborting")
            return _finish(phases, profiler, total_t0)

        # Phase 3: Needle in Haystack
        logger.info("\n=== Phase 3: Needle in Haystack ===")
        p3 = phase3_niah(model, tokenizer)
        phases.append(p3)
        if not p3.passed:
            logger.error("Phase 3 FAILED — aborting")
            return _finish(phases, profiler, total_t0)

    except Exception as e:
        logger.exception("Benchmark crashed: %s", e)
        phases.append(PhaseResult(name="CRASH", passed=False,
                                  details={"error": str(e)}))

    return _finish(phases, profiler, total_t0)


def _finish(phases: List[PhaseResult], profiler: Profiler,
            total_t0: float) -> int:
    """Stop profiler, evaluate memory gates, write results, print summary."""
    profiler_result = profiler.stop()
    total_elapsed = time.perf_counter() - total_t0

    # Phase 4: Memory check
    logger.info("\n=== Phase 4: Memory Profile ===")
    p4 = phase4_memory_check(profiler_result)
    phases.append(p4)

    # Phase 5: Write results
    logger.info("\n=== Phase 5: Profiling Snapshot ===")
    p5 = phase5_write_results(phases, profiler_result, total_elapsed)
    phases.append(p5)

    _print_summary(phases, total_elapsed)

    all_passed = all(p.passed for p in phases)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
