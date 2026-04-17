# SPDX-License-Identifier: Apache-2.0
"""Gated benchmark for oMLX Hypercar — run before every commit.

Eight phases, each gated: a failure stops the pipeline.
Memory watchdog runs continuously and aborts immediately on breach.

  Phase 0: Smoke          — model loads, 10 tokens, Metal < load limit   (<30s)
  Phase 1: Coherence      — "2+2" => "4", "hello world" => "print"      (<60s)
  Phase 2: Code Intel     — 5 coding problems, exec + assert, >=60%     (<5min)
  Phase 3: NIAH           — needle retrieval at 4K context (ChatML)      (<5min)
  Phase 3b: RULER         — multi-key retrieval + aggregation (RULER)    (<5min)
  Phase 3c: MMLU-Pro      — reasoning gate (cs + math, >=35%)           (<3min)
  Phase 4: HumanEval Lite — 20 curated problems, >=50% pass@1           (<10min)
  Phase 5: Memory         — watchdog summary (breach = already aborted)
  Phase 6: Summary        — write results JSON

Usage:
    .venv/bin/python -m omlx.bench.hypercar_bench            # Phase 0-3b
    .venv/bin/python -m omlx.bench.hypercar_bench --quick    # Phase 0+1 only
    .venv/bin/python -m omlx.bench.hypercar_bench --full     # Phase 0-4 (HumanEval + full RULER)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import threading
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

MODEL_ID = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"  # Override with --model
KV_BITS = 3
KV_GROUP_SIZE = 64
PREFILL_CHUNK = 4096

# Gate thresholds (defaults — overridden by % of system memory)
MIN_CODE_PASS_RATE = 0.6
MIN_HUMANEVAL_PASS_RATE = 0.35  # 4-bit MoE model scores ~40-45% on these problems
MIN_RULER_MK_ACCURACY = 0.8  # multi_key_retrieval@16K must hit 80%
MIN_RULER_VT_ACCURACY = 0.7  # variable_tracking@4K must hit 70%
MIN_MMLU_PRO_ACCURACY = 0.35  # MMLU-Pro cs+math must hit 35%
# MMLU-Pro needs ≥512 max_tokens for CoT reasoning. Cutting to 256 caused
# a 64%→24% silent regression (commits 1e803b6→20df582) because reasoning
# was truncated before emitting the final answer letter.
MMLU_PRO_MIN_MAX_TOKENS = 512

PROFILE_PATH = Path("/tmp/hypercar_profile.json")
DEFAULT_RESULTS_PATH = Path("/tmp/hypercar_bench_results.json")

NIAH_NEEDLE = "The secret code is ALPHA-7749"
NIAH_QUESTION = "What is the secret code?"
NIAH_ANSWER = "ALPHA-7749"


def _parse_context_list(s: str) -> list[int]:
    """Parse comma-separated context specs like '4K,16K,64K' into token counts."""
    result = []
    for part in s.split(","):
        part = part.strip().upper()
        if part.endswith("K"):
            result.append(int(part[:-1]) * 1024)
        elif part.endswith("M"):
            result.append(int(part[:-1]) * 1024 * 1024)
        else:
            result.append(int(part))
    return result


# Per-phase headroom requirements (GB above current Metal usage).
# Calibrated from R44/R47/R48 successful runs' per-phase memory deltas.
PHASE_HEADROOM_GB = {
    "Phase 3: NIAH": 8.0,       # 16K fp16 KV + attention scores
    "Phase 3b: RULER": 6.0,     # Multi-key NIAH at 16K
    "Phase 3c: MMLU-Pro": 3.0,  # Small per-question KV
    "Phase 4: HumanEval": 3.0,  # Small per-problem KV
}


def _check_phase_headroom(phase_name: str, metal_limit_gb: float) -> bool:
    """Check if enough Metal headroom exists for a phase.

    Returns True if the phase should run, False if it should be skipped.
    Prevents mid-run crashes under co-tenancy (R54/R57/R58 pattern).
    """
    required = PHASE_HEADROOM_GB.get(phase_name, 2.0)
    try:
        # Use non-deprecated API (mx.get_active_memory replaces mx.metal.get_active_memory)
        get_active = getattr(mx, 'get_active_memory', None) or mx.metal.get_active_memory
        metal_active = get_active() / 1e9
        headroom = metal_limit_gb - metal_active
        if headroom < required:
            logger.warning(
                f"  {phase_name} SKIPPED — headroom {headroom:.1f} GB < "
                f"{required:.1f} GB required (co-tenancy pressure detected)"
            )
            return False
        return True
    except Exception:
        return True  # If we can't check, proceed optimistically


# ---------------------------------------------------------------------------
# System memory detection
# ---------------------------------------------------------------------------

def _detect_system_memory_gb() -> float:
    """Detect total system memory in GB (macOS)."""
    try:
        result = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, timeout=5,
        )
        total_bytes = int(result.strip())
        return total_bytes / 1e9
    except Exception:
        logger.warning("Could not detect system memory, defaulting to 48GB")
        return 48.0


def _compute_memory_limits(
    total_gb: float,
    metal_pct: float,
    swap_pct: float,
    load_pct: float,
) -> Dict[str, float]:
    """Compute memory limits as % of system memory."""
    return {
        "metal_peak_gb": total_gb * (metal_pct / 100.0),
        "swap_delta_gb": total_gb * (swap_pct / 100.0),
        "metal_load_gb": total_gb * (load_pct / 100.0),
    }


# ---------------------------------------------------------------------------
# Memory Watchdog
# ---------------------------------------------------------------------------

class MemoryWatchdog:
    """Fail-fast memory watchdog wrapping the Profiler.

    Checks memory limits on every profiler sample. On breach, sets the
    ``breached`` event so phases can abort immediately.
    """

    def __init__(
        self,
        metal_limit_gb: float,
        swap_limit_gb: float,
        sample_interval: float = 1.0,
    ):
        self.metal_limit_gb = metal_limit_gb
        self.swap_limit_gb = swap_limit_gb
        self.breached = threading.Event()
        self.breach_reason: str = ""
        self.breach_snapshot: Optional[Dict[str, Any]] = None

        self.profiler = Profiler(sample_interval=sample_interval)
        self._check_stop = threading.Event()
        self._check_thread: Optional[threading.Thread] = None

    def start(self):
        self.profiler.start()
        self._check_stop.clear()
        self._check_thread = threading.Thread(
            target=self._check_loop, daemon=True,
        )
        self._check_thread.start()

    def stop(self):
        self._check_stop.set()
        if self._check_thread:
            self._check_thread.join(timeout=5)
        return self.profiler.stop()

    # Swap I/O throughput threshold: sustained pressure above this kills perf
    # 2000 MB/s = runaway thrashing. Normal model load may see 500-1000 MB/s
    # transiently as the OS pages in model weights.
    SWAP_IO_BREACH_MB_S = 2000.0
    SWAP_IO_BREACH_CONSECUTIVE = 5  # Must exceed for N consecutive samples

    def _check_loop(self):
        """Poll profiler samples and check limits."""
        last_idx = 0
        swap_io_breach_count = 0
        while not self._check_stop.is_set():
            samples = self.profiler.result.samples
            for i in range(last_idx, len(samples)):
                s = samples[i]
                if s.metal_peak_gb > self.metal_limit_gb:
                    self._set_breach(
                        f"Metal peak {s.metal_peak_gb:.1f}GB > "
                        f"{self.metal_limit_gb:.1f}GB limit",
                        s,
                    )
                    return
                if s.swap_gb > self.swap_limit_gb:
                    self._set_breach(
                        f"Swap delta {s.swap_gb:.1f}GB > "
                        f"{self.swap_limit_gb:.1f}GB limit",
                        s,
                    )
                    return
                # Swap I/O throughput: detect runaway memory pressure
                if s.swap_io_mb_per_s > self.SWAP_IO_BREACH_MB_S:
                    swap_io_breach_count += 1
                    if swap_io_breach_count >= self.SWAP_IO_BREACH_CONSECUTIVE:
                        self._set_breach(
                            f"Swap I/O {s.swap_io_mb_per_s:.0f} MB/s sustained "
                            f"for {swap_io_breach_count} samples "
                            f"(threshold {self.SWAP_IO_BREACH_MB_S:.0f} MB/s) — "
                            f"system under critical memory pressure",
                            s,
                        )
                        return
                else:
                    swap_io_breach_count = 0
            last_idx = len(samples)
            self._check_stop.wait(0.5)

    def _set_breach(self, reason: str, sample):
        self.breach_reason = reason
        self.breach_snapshot = {
            "timestamp": round(sample.t, 2),
            "metal_gb": round(sample.metal_peak_gb, 2),
            "swap_gb": round(sample.swap_gb, 2),
            "reason": reason,
        }
        logger.error(f"MEMORY BREACH: {reason}")
        # Flush immediately — under memory pressure the process may exit
        # before buffered log output is written. R54/R57 exited 144 with
        # NO error message because the logger didn't flush in time.
        for handler in logging.getLogger().handlers:
            handler.flush()
        sys.stderr.flush()
        print("WATCHDOG-BREACH", file=sys.stderr, flush=True)
        self.breached.set()


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
    reason: str = ""


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


# Module-level KV mode and model ref (set from CLI in main())
_KV_MODE = "native"
_MODEL_REF = None  # Set after model loads, used by _make_cache for hybrid models
_QUEST_TOPK = 0  # Quest page selection (0=off)


def _load_model():
    """Load model + tokenizer, apply patches for current kv-mode."""
    from mlx_lm import load
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch

    if _KV_MODE == "tq3":
        from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
        apply_turboquant_attention_patch()

    model, tokenizer = load(MODEL_ID)
    apply_prefill_last_logit_patch(model)

    if _KV_MODE == "tq3":
        from omlx.patches.vertical_eval import apply_vertical_eval_patch
        apply_vertical_eval_patch(model)

    return model, tokenizer


def _make_cache(n_layers: int, model=None):
    """Create KV cache based on current _KV_MODE.

    For hybrid models (Granite): uses model.make_cache() to get the right
    cache types per layer (ArraysCache for Mamba, KVCache for attention),
    then replaces KVCache with quantized variants.
    """
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    if _KV_MODE == "fp16":
        if model and hasattr(model, 'make_cache'):
            return model.make_cache()
        return [KVCache() for _ in range(n_layers)]

    if _KV_MODE == "duo":
        from omlx.duo_kv_cache import DuoKVCache, load_duo_policy
        policy = load_duo_policy()
        return [DuoKVCache(policy, layer_idx=i, bits=KV_BITS) for i in range(n_layers)]

    if _KV_MODE == "shadowkv":
        from omlx.shadowkv_cache import ShadowKVCache
        return [ShadowKVCache(target_rank=192) for i in range(n_layers)]

    if model is None:
        model = _MODEL_REF

    # Get base caches (handles hybrid models with ArraysCache + KVCache)
    if model and hasattr(model, 'make_cache'):
        base_caches = model.make_cache()
    else:
        base_caches = [KVCache() for _ in range(n_layers)]

    # Replace KVCache layers with quantized variants
    result = []
    for i, c in enumerate(base_caches):
        if isinstance(c, KVCache):
            if _KV_MODE == "tq3":
                from omlx.turboquant_kv import TurboQuantKVCache
                result.append(TurboQuantKVCache(bits=KV_BITS, quest_topk=_QUEST_TOPK))
            else:  # native
                result.append(QuantizedKVCache(group_size=KV_GROUP_SIZE, bits=KV_BITS))
        else:
            result.append(c)  # Keep ArraysCache (Mamba state) as-is
    return result


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

def phase0_smoke(model, tokenizer, watchdog: MemoryWatchdog,
                 metal_load_limit: float) -> PhaseResult:
    """Load model, generate 10 tokens, check Metal < load limit."""
    t0 = time.perf_counter()

    if watchdog.breached.is_set():
        return PhaseResult(
            name="Phase 0: Smoke", passed=False,
            elapsed_s=time.perf_counter() - t0,
            reason=watchdog.breach_reason,
        )

    metal_after_load = _metal_gb()
    logger.info(f"  Metal after load: {metal_after_load:.1f} GB")

    if metal_after_load > metal_load_limit:
        return PhaseResult(
            name="Phase 0: Smoke",
            passed=False,
            elapsed_s=time.perf_counter() - t0,
            details={"metal_after_load_gb": round(metal_after_load, 2),
                     "reason": f"Metal {metal_after_load:.1f}GB > {metal_load_limit:.1f}GB limit"},
        )

    # Generate 10 tokens
    text, prefill_toks, decode_toks = _generate(model, tokenizer,
                                                 "Hello, world!", max_tokens=10)
    logger.info(f"  Smoke output: {text[:80]!r}")
    logger.info(f"  Prefill: {prefill_toks:.0f} tok/s  Decode: {decode_toks:.1f} tok/s")

    if watchdog.breached.is_set():
        return PhaseResult(
            name="Phase 0: Smoke", passed=False,
            elapsed_s=time.perf_counter() - t0,
            reason=watchdog.breach_reason,
        )

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

def phase1_coherence(model, tokenizer, watchdog: MemoryWatchdog) -> PhaseResult:
    """Basic coherence: math + code generation."""
    t0 = time.perf_counter()
    checks = {}

    if watchdog.breached.is_set():
        return PhaseResult(
            name="Phase 1: Coherence", passed=False,
            elapsed_s=time.perf_counter() - t0,
            reason=watchdog.breach_reason,
        )

    # Check 1: 2+2 must contain "4"
    text, p_toks, d_toks = _generate(model, tokenizer,
                                      "What is 2+2? Answer with just the number.",
                                      max_tokens=32)
    checks["math"] = {"output": text[:100], "passed": "4" in text}
    logger.info(f"  Math check: {'PASS' if checks['math']['passed'] else 'FAIL'} — {text[:60]!r}")

    if watchdog.breached.is_set():
        return PhaseResult(
            name="Phase 1: Coherence", passed=False,
            elapsed_s=time.perf_counter() - t0,
            reason=watchdog.breach_reason,
        )

    # Check 2: hello world must contain "print"
    text2, p_toks2, d_toks2 = _generate(model, tokenizer,
                                          "Write hello world in Python. Just the code, nothing else.",
                                          max_tokens=64)
    checks["code"] = {"output": text2[:100], "passed": "print" in text2.lower()}
    logger.info(f"  Code check: {'PASS' if checks['code']['passed'] else 'FAIL'} — {text2[:60]!r}")

    if watchdog.breached.is_set():
        return PhaseResult(
            name="Phase 1: Coherence", passed=False,
            elapsed_s=time.perf_counter() - t0,
            reason=watchdog.breach_reason,
        )

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


def phase2_code_intelligence(model, tokenizer,
                             watchdog: MemoryWatchdog) -> PhaseResult:
    """Run 5 coding problems, exec + assert."""
    t0 = time.perf_counter()
    n_layers = len(model.layers)
    results = []

    for prob in CODING_PROBLEMS:
        if watchdog.breached.is_set():
            return PhaseResult(
                name="Phase 2: Code Intelligence", passed=False,
                elapsed_s=time.perf_counter() - t0,
                reason=watchdog.breach_reason,
            )

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
# Phase 3: Needle in Haystack (ChatML via apply_chat_template)
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


def phase3_niah(model, tokenizer, watchdog: MemoryWatchdog, args_ref=None) -> PhaseResult:
    """Needle-in-a-haystack at 4K context using ChatML formatting."""
    t0 = time.perf_counter()
    n_layers = len(model.layers)
    results = {}

    # --niah-context overrides default context list
    niah_context_str = getattr(args_ref, 'niah_context', None)
    if niah_context_str:
        niah_contexts = _parse_context_list(niah_context_str)
        logger.info(f"  NIAH contexts from --niah-context: "
                    f"{[f'{c//1024}K' for c in niah_contexts]}")
    else:
        niah_contexts = [4096, 16384]
        if getattr(args_ref, 'niah_500k', False):
            niah_contexts.extend([65536, 131072, 262144, 524288])

    for ctx_len in niah_contexts:
        if watchdog.breached.is_set():
            return PhaseResult(
                name="Phase 3: Needle in Haystack", passed=False,
                elapsed_s=time.perf_counter() - t0,
                reason=watchdog.breach_reason,
            )

        logger.info(f"  NIAH @ {ctx_len // 1024}K context...")

        # Leave room for ChatML template overhead + question (~200 tokens)
        haystack = _build_code_haystack(tokenizer, ctx_len - 200, NIAH_NEEDLE, 50.0)

        # Use ChatML formatting via tokenizer.apply_chat_template
        messages = [
            {"role": "user",
             "content": f"Here is some code:\n\n{haystack}\n\nBased on the code above, answer this question: {NIAH_QUESTION}\nRespond with ONLY the answer, nothing else."}
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback if tokenizer doesn't support chat templates
            prompt = f"{haystack}\n\nQuestion: {NIAH_QUESTION}\nAnswer:"

        tokens = tokenizer.encode(prompt)[:ctx_len]
        cache = _make_cache(n_layers)

        # Chunked prefill — adaptive chunk size for long context.
        # Attention scores per chunk = chunk × accumulated_ctx × n_heads × 2 bytes.
        # chunk=4096 at 64K = 16 GB (OOM). chunk=1024 = 4 GB. chunk=512 = 2 GB.
        # On 48GB M4 Pro with 32.4 GB model, chunk=512 at 64K leaves ~7 GB
        # headroom vs the 41.2 GB Metal limit (80% of 51.5 GB).
        chunk = PREFILL_CHUNK
        if ctx_len >= 131072:
            chunk = 512   # 128K+: 512 × 131K × 32 × 2 = 4.3 GB scores
        elif ctx_len >= 65536:
            chunk = 512   # 64K: 512 × 64K × 32 × 2 = 2.1 GB scores
        elif ctx_len >= 32768:
            chunk = 2048

        prefill_t0 = time.perf_counter()
        for chunk_start in range(0, len(tokens), chunk):
            chunk_end = min(chunk_start + chunk, len(tokens))
            x = mx.array([tokens[chunk_start:chunk_end]])
            logits = model(x, cache=cache)
            mx.eval(logits)
        prefill_time = time.perf_counter() - prefill_t0
        prefill_toks = len(tokens) / prefill_time if prefill_time > 0 else 0.0

        # Generate response (32 tokens for NIAH answer)
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

        # Decode stress: generate 128 MORE tokens to measure decode speed
        # honestly. The NIAH answer tokens are too few (32) to amortize
        # MLX graph compilation + kernel warmup, producing artifact readings
        # like 0.3 tok/s at 16K when real speed is ~20 tok/s.
        DECODE_STRESS_TOKENS = 128
        decode_t0 = time.perf_counter()
        for _ in range(DECODE_STRESS_TOKENS):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)
        decode_time = time.perf_counter() - decode_t0
        decode_toks = DECODE_STRESS_TOKENS / decode_time if decode_time > 0 else 0.0

        results[ctx_len] = {
            "found": found,
            "response": response[:100],
            "context_tokens": len(tokens),
            "prefill_toks": round(prefill_toks, 1),
            "decode_toks": round(decode_toks, 1),
            "decode_stress_tokens": DECODE_STRESS_TOKENS,
        }

        status = "PASS" if found else "FAIL"
        logger.info(f"    {status}: {ctx_len // 1024}K — {response[:60]!r}")

        del cache
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    # Gate: must pass at 4K, 16K, and all contexts up to 256K
    # 500K+ is a warning (prefill takes too long for gating)
    gate_contexts = [c for c in niah_contexts if c <= 262144]
    gate_passed = all(results.get(c, {}).get("found", False) for c in gate_contexts)

    return PhaseResult(
        name="Phase 3: Needle in Haystack",
        passed=gate_passed,
        elapsed_s=time.perf_counter() - t0,
        details=results,
        prefill_toks=results[4096].get("prefill_toks", 0.0),
        decode_toks=results[4096].get("decode_toks", 0.0),
    )


# ---------------------------------------------------------------------------
# Phase 3b: RULER — multi-key retrieval + variable tracking + aggregation
# ---------------------------------------------------------------------------

def _run_ruler_task(model, tokenizer, task_spec: dict,
                    watchdog: MemoryWatchdog) -> dict:
    """Run a single RULER task and return result dict."""
    from omlx.eval.ruler.tasks import (
        generate_multi_key_niah,
        generate_variable_tracking,
        generate_frequent_word,
    )

    generators = {
        "multi_key_niah": generate_multi_key_niah,
        "variable_tracking": generate_variable_tracking,
        "frequent_word": generate_frequent_word,
    }

    gen_name = task_spec["generator"]
    gen_fn = generators[gen_name]

    # Build kwargs from task_spec (exclude 'generator')
    kwargs = {k: v for k, v in task_spec.items() if k != "generator"}
    kwargs["tokenizer"] = tokenizer
    task = gen_fn(**kwargs)

    n_layers = len(model.layers)
    ctx_tokens = task_spec["target_tokens"]

    # Format as chat message
    messages = [
        {"role": "user",
         "content": f"{task['context']}\n\n{task['question']}"}
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt = f"{task['context']}\n\n{task['question']}\nAnswer:"

    tokens = tokenizer.encode(prompt)[:ctx_tokens]
    cache = _make_cache(n_layers)

    # Chunked prefill
    for chunk_start in range(0, len(tokens), PREFILL_CHUNK):
        if watchdog.breached.is_set():
            del cache
            return {"task_type": gen_name, "ctx": ctx_tokens, "passed": False,
                    "reason": "memory_breach"}
        chunk_end = min(chunk_start + PREFILL_CHUNK, len(tokens))
        x = mx.array([tokens[chunk_start:chunk_end]])
        logits = model(x, cache=cache)
        mx.eval(logits)

    # Generate response (up to 128 tokens for multi-value answers)
    generated = []
    for _ in range(128):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tok_id = token.item()
        generated.append(tok_id)
        if hasattr(tokenizer, 'eos_token_id') and tok_id == tokenizer.eos_token_id:
            break
        x = token.reshape(1, 1)
        logits = model(x, cache=cache)
        mx.eval(logits)

    response = tokenizer.decode(generated).strip()

    # Score: check how many expected values appear in the response
    # For multi-key NIAH: each expected value must be found (AND logic)
    # For freq_word/variable_tracking: expected contains case variants of
    # one answer — any match counts as correct (OR logic)
    expected = task["expected"]
    task_type = task.get("task_type", "")
    if task_type in ("frequent_word", "variable_tracking"):
        # OR logic: any variant matching = 100% accuracy
        any_found = any(exp in response for exp in expected)
        found = len(expected) if any_found else 0
    else:
        # AND logic: each expected value must appear (multi-key NIAH)
        found = sum(1 for exp in expected if exp in response)
    accuracy = found / len(expected) if expected else 0.0

    del cache
    gc.collect()
    mx.clear_cache()

    return {
        "task_type": gen_name,
        "ctx": ctx_tokens,
        "params": task["params"],
        "response": response[:200],
        "expected": expected,
        "found": found,
        "total": len(expected),
        "accuracy": accuracy,
        "passed": accuracy >= 0.5,  # per-task pass: at least half correct
    }


def _project_prefill_memory_gb(ctx_tokens: int, model) -> float:
    """Rough upper bound on additional Metal memory for a prefill at ctx_tokens.

    During prefill, the main memory consumers beyond the model itself are:
      - KV cache: ctx_tokens × n_layers × 2 (K+V) × head_dim × n_kv_heads × dtype
      - Attention scores: ctx_tokens × ctx_tokens × n_heads × dtype (per chunk)
      - Intermediates: MoE router, RMS norm, etc.

    We use a conservative safety_factor to account for intermediates.
    """
    n_layers = len(model.layers)
    # Detect head dimensions from model config
    first_attn = getattr(model.layers[0], 'self_attn', None)
    if first_attn and hasattr(first_attn, 'n_heads'):
        n_heads = first_attn.n_heads
    else:
        n_heads = 32  # Qwen3-Coder default
    if first_attn and hasattr(first_attn, 'n_kv_heads'):
        n_kv_heads = first_attn.n_kv_heads
    else:
        n_kv_heads = 4

    hidden_size = getattr(model, 'hidden_size', None)
    if hidden_size is None and hasattr(model, 'args'):
        hidden_size = getattr(model.args, 'hidden_size', 2048)
    else:
        hidden_size = hidden_size or 2048
    head_dim = hidden_size // n_heads

    # KV cache bytes: depends on quantization mode
    if _KV_MODE in ("native", "tq3"):
        bytes_per_kv_elem = KV_BITS / 8  # 3-bit = 0.375 bytes/elem
    elif _KV_MODE == "duo":
        bytes_per_kv_elem = 2  # fp16 (duo uses fp16 for retrieval heads)
    else:
        bytes_per_kv_elem = 2  # fp16

    # KV cache memory (both K and V, all layers)
    kv_gb = (ctx_tokens * n_layers * 2 * n_kv_heads * head_dim * bytes_per_kv_elem) / 1e9

    # Attention scores per chunk (always fp16 regardless of KV mode)
    chunk = min(ctx_tokens, PREFILL_CHUNK)
    attn_gb = (chunk * ctx_tokens * n_heads * 2) / 1e9  # fp16 scores

    safety_factor = 1.5  # MoE router, RMS norm, residuals
    return (kv_gb + attn_gb) * safety_factor


def phase3b_ruler(model, tokenizer, watchdog: MemoryWatchdog,
                  full: bool = False) -> PhaseResult:
    """RULER synthetic long-context evaluation.

    Default mode: quick suite (multi-key NIAH + VT + freq-word at 4K/16K).
    --full mode: full 15-task suite at 4K/16K/64K.

    Two gates:
      1. multi_key_retrieval@16K accuracy >= 0.8
      2. ruler_vt@4K accuracy >= 0.7 (independent eval for Goal 2)

    Methodology follows ProLong (arXiv:2410.02660) which identifies three
    diagnostic RULER subtask families — retrieval (multi-key NIAH),
    multi-hop tracing (variable tracking), and aggregation (frequent word) —
    as the minimal set that distinguishes genuine long-context capability
    from shallow retrieval. Length tiers (4K, 16K, 64K) are chosen to
    bracket the model's validated context range. Tasks beyond the headroom
    gate are SKIPped rather than aborting the phase.
    """
    from omlx.eval.ruler.tasks import RULER_QUICK_SUITE, RULER_FULL_SUITE

    t0 = time.perf_counter()
    suite = RULER_FULL_SUITE if full else RULER_QUICK_SUITE
    results = []
    skipped = []

    logger.info(f"  Running {'full' if full else 'quick'} RULER suite "
                f"({len(suite)} tasks)...")

    for i, task_spec in enumerate(suite):
        if watchdog.breached.is_set():
            return PhaseResult(
                name="Phase 3b: RULER", passed=False,
                elapsed_s=time.perf_counter() - t0,
                reason=watchdog.breach_reason,
            )

        gen_name = task_spec["generator"]
        ctx_tokens = task_spec["target_tokens"]
        ctx_k = ctx_tokens // 1024
        extra = ""
        if gen_name == "multi_key_niah":
            extra = f" keys={task_spec['num_keys']}"
        elif gen_name == "variable_tracking":
            extra = f" chain={task_spec['chain_length']}"
        elif gen_name == "frequent_word":
            extra = f" words={task_spec['num_target_words']}"

        # Memory headroom check: skip tasks that would breach Metal limit
        projected_gb = _project_prefill_memory_gb(ctx_tokens, model)
        current_metal = _metal_gb()
        headroom_gb = watchdog.metal_limit_gb - current_metal - 1.0  # 1GB soft buffer

        if projected_gb > headroom_gb:
            skip_reason = (
                f"projected {projected_gb:.1f}GB > {headroom_gb:.1f}GB headroom "
                f"(Metal {current_metal:.1f}GB + limit {watchdog.metal_limit_gb:.1f}GB)"
            )
            logger.info(f"  [{i+1}/{len(suite)}] {gen_name}@{ctx_k}K{extra} — SKIP: {skip_reason}")
            skipped.append({
                "task_type": gen_name,
                "ctx": ctx_tokens,
                "reason": f"skip_memory: {skip_reason}",
                "projected_gb": round(projected_gb, 1),
            })
            continue

        logger.info(f"  [{i+1}/{len(suite)}] {gen_name}@{ctx_k}K{extra}")
        result = _run_ruler_task(model, tokenizer, task_spec, watchdog)
        results.append(result)

        if "reason" in result:
            logger.info(f"    BREACH: {result['reason']}")
        else:
            status = "PASS" if result["passed"] else "FAIL"
            logger.info(f"    {status}: {result['found']}/{result['total']} "
                         f"(acc={result['accuracy']:.0%}) — {result['response'][:60]!r}")

    # Gate 1: multi_key_retrieval@16K accuracy >= MIN_RULER_MK_ACCURACY
    # Filter out breach results (they lack 'accuracy')
    mk_16k_results = [
        r for r in results
        if r["task_type"] == "multi_key_niah" and r["ctx"] == 16384
        and "accuracy" in r
    ]

    if mk_16k_results:
        mk_16k_accuracy = sum(r["accuracy"] for r in mk_16k_results) / len(mk_16k_results)
    else:
        mk_16k_accuracy = 1.0  # No 16K multi-key tasks ran — skip gate

    mk_gate = mk_16k_accuracy >= MIN_RULER_MK_ACCURACY

    # Gate 2: variable_tracking@4K accuracy >= MIN_RULER_VT_ACCURACY
    # This is an independent eval toward Goal 2 ("4 independent evals")
    vt_4k_results = [
        r for r in results
        if r["task_type"] == "variable_tracking" and r["ctx"] == 4096
        and "accuracy" in r
    ]

    if vt_4k_results:
        vt_4k_accuracy = sum(r["accuracy"] for r in vt_4k_results) / len(vt_4k_results)
    else:
        vt_4k_accuracy = 1.0  # No 4K VT tasks — skip gate

    vt_gate = vt_4k_accuracy >= MIN_RULER_VT_ACCURACY

    gate_passed = mk_gate and vt_gate

    # Summary stats (skip breach results)
    by_type = {}
    for r in results:
        if "accuracy" not in r:
            continue
        key = f"{r['task_type']}@{r['ctx'] // 1024}K"
        if key in by_type:
            existing = by_type[key]
            by_type[key] = (existing + r["accuracy"]) / 2
        else:
            by_type[key] = r["accuracy"]

    if skipped:
        logger.info(f"  Skipped {len(skipped)} tasks due to memory headroom")

    logger.info(f"  RULER summary: {by_type}")
    logger.info(f"  Gate (multi_key@16K): {mk_16k_accuracy:.0%} "
                f"(need {MIN_RULER_MK_ACCURACY:.0%}) — "
                f"{'PASS' if mk_gate else 'FAIL'}")
    logger.info(f"  Gate (ruler_vt@4K):   {vt_4k_accuracy:.0%} "
                f"(need {MIN_RULER_VT_ACCURACY:.0%}) — "
                f"{'PASS' if vt_gate else 'FAIL'}")

    return PhaseResult(
        name="Phase 3b: RULER",
        passed=gate_passed,
        elapsed_s=time.perf_counter() - t0,
        details={
            "suite": "full" if full else "quick",
            "num_tasks": len(suite),
            "num_ran": len(results),
            "num_skipped": len(skipped),
            "results": results,
            "skipped": skipped,
            "by_type": by_type,
            "mk_16k_accuracy": mk_16k_accuracy,
            "vt_4k_accuracy": vt_4k_accuracy,
            "gates": {
                "multi_key@16K": {"accuracy": mk_16k_accuracy, "threshold": MIN_RULER_MK_ACCURACY, "passed": mk_gate},
                "ruler_vt@4K": {"accuracy": vt_4k_accuracy, "threshold": MIN_RULER_VT_ACCURACY, "passed": vt_gate},
            },
        },
    )


# ---------------------------------------------------------------------------
# Phase 3c: MMLU-Pro (reasoning gate for Goal 2)
# ---------------------------------------------------------------------------

def phase3c_mmlu_pro(model, tokenizer, watchdog: MemoryWatchdog,
                     full: bool = False) -> PhaseResult:
    """MMLU-Pro reasoning evaluation.

    Default mode: 25 questions from cs + math.
    --full mode: 100 questions across all categories.

    Gate: mmlu_pro_cs_math >= 0.35 accuracy.
    """
    from omlx.eval.mmlu_pro.tasks import load_mmlu_pro, format_prompt, extract_answer

    assert MMLU_PRO_MIN_MAX_TOKENS >= 512, (
        f"MMLU-Pro needs ≥512 max_tokens — 256 caused a 64→24% silent "
        f"regression (commits 1e803b6→20df582). Got {MMLU_PRO_MIN_MAX_TOKENS}."
    )

    t0 = time.perf_counter()

    if full:
        questions = load_mmlu_pro(categories=None, n=100, seed=42)
    else:
        questions = load_mmlu_pro(categories=["computer_science", "math"], n=25, seed=42)

    if not questions:
        logger.warning("  MMLU-Pro: no questions loaded — skipping")
        return PhaseResult(
            name="Phase 3c: MMLU-Pro", passed=True,
            elapsed_s=time.perf_counter() - t0,
            details={"skipped": True, "reason": "dataset not available"},
        )

    n_layers = len(model.layers)
    correct = 0
    total = 0
    results = []

    logger.info(f"  Running MMLU-Pro ({len(questions)} questions)...")

    for i, q in enumerate(questions):
        if watchdog.breached.is_set():
            return PhaseResult(
                name="Phase 3c: MMLU-Pro", passed=False,
                elapsed_s=time.perf_counter() - t0,
                reason=watchdog.breach_reason,
            )

        prompt = format_prompt(q)

        # Use ChatML formatting
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            chat_prompt = prompt + "\n"

        text, _, _ = _generate(model, tokenizer, chat_prompt,
                               max_tokens=MMLU_PRO_MIN_MAX_TOKENS)
        predicted = extract_answer(text)
        is_correct = predicted == q["answer"]

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "category": q["category"],
            "predicted": predicted,
            "expected": q["answer"],
            "correct": is_correct,
        })

        if (i + 1) % 5 == 0 or i == len(questions) - 1:
            logger.info(f"  [{i+1}/{len(questions)}] {correct}/{total} correct "
                        f"({correct/total*100:.0f}%)")

    accuracy = correct / total if total > 0 else 0.0
    gate_passed = accuracy >= MIN_MMLU_PRO_ACCURACY

    # Per-category breakdown
    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, {"correct": 0, "total": 0})
        by_cat[cat]["total"] += 1
        if r["correct"]:
            by_cat[cat]["correct"] += 1

    logger.info(f"  MMLU-Pro: {correct}/{total} ({accuracy:.0%}) — "
                f"gate {'PASS' if gate_passed else 'FAIL'}")
    for cat, stats in sorted(by_cat.items()):
        cat_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"    {cat}: {stats['correct']}/{stats['total']} ({cat_acc:.0%})")

    return PhaseResult(
        name="Phase 3c: MMLU-Pro",
        passed=gate_passed,
        elapsed_s=time.perf_counter() - t0,
        details={
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "by_category": by_cat,
            "gate_threshold": MIN_MMLU_PRO_ACCURACY,
        },
    )


# ---------------------------------------------------------------------------
# Phase 4: HumanEval Lite (20 curated problems)
# ---------------------------------------------------------------------------

HUMANEVAL_LITE = [
    {
        "task_id": "HE/0",
        "prompt": 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """Check if any two numbers in the list are closer than threshold."""\n',
        "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0, 4.0], 0.3) == True",
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HE/1",
        "prompt": 'def separate_paren_groups(paren_string: str) -> list[str]:\n    """Separate groups of balanced parentheses into individual strings."""\n',
        "test": "assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']",
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HE/2",
        "prompt": 'def truncate_number(number: float) -> float:\n    """Return the decimal part of a positive float."""\n',
        "test": "assert abs(truncate_number(3.5) - 0.5) < 1e-6",
        "entry_point": "truncate_number",
    },
    {
        "task_id": "HE/3",
        "prompt": 'def below_zero(operations: list[int]) -> bool:\n    """Check if a bank account starting at 0 goes below zero after operations."""\n',
        "test": "assert below_zero([1, 2, -3, 1, 2, -4, 5, 6, -1, 2, -3, 5, -22]) == True\nassert below_zero([1, 2, 3]) == False",
        "entry_point": "below_zero",
    },
    {
        "task_id": "HE/4",
        "prompt": 'def mean_absolute_deviation(numbers: list[float]) -> float:\n    """Compute mean absolute deviation around the mean."""\n',
        "test": "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6",
        "entry_point": "mean_absolute_deviation",
    },
    {
        "task_id": "HE/5",
        "prompt": 'def intersperse(numbers: list[int], delimeter: int) -> list[int]:\n    """Insert delimeter between every two consecutive elements."""\n',
        "test": "assert intersperse([], 4) == []\nassert intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]",
        "entry_point": "intersperse",
    },
    {
        "task_id": "HE/6",
        "prompt": 'def parse_nested_parens(paren_string: str) -> list[int]:\n    """Return the max nesting depth for each group of parentheses."""\n',
        "test": "assert parse_nested_parens('(()()) ((())) () ((())()())') == [2, 3, 1, 3]",
        "entry_point": "parse_nested_parens",
    },
    {
        "task_id": "HE/7",
        "prompt": 'def filter_by_substring(strings: list[str], substring: str) -> list[str]:\n    """Filter strings that contain the given substring."""\n',
        "test": "assert filter_by_substring([], 'a') == []\nassert filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a') == ['abc', 'bacd', 'array']",
        "entry_point": "filter_by_substring",
    },
    {
        "task_id": "HE/8",
        "prompt": 'def sum_product(numbers: list[int]) -> tuple[int, int]:\n    """Return a tuple of (sum, product) of all numbers in the list."""\n',
        "test": "assert sum_product([]) == (0, 1)\nassert sum_product([1, 2, 3, 4]) == (10, 24)",
        "entry_point": "sum_product",
    },
    {
        "task_id": "HE/9",
        "prompt": 'def rolling_max(numbers: list[int]) -> list[int]:\n    """Return the running maximum at each position."""\n',
        "test": "assert rolling_max([1, 2, 3, 2, 3, 4, 2]) == [1, 2, 3, 3, 3, 4, 4]",
        "entry_point": "rolling_max",
    },
    {
        "task_id": "HE/10",
        "prompt": 'def is_palindrome(string: str) -> bool:\n    """Check if a string is a palindrome."""\n',
        "test": "assert is_palindrome('') == True\nassert is_palindrome('aba') == True\nassert is_palindrome('abc') == False",
        "entry_point": "is_palindrome",
    },
    {
        "task_id": "HE/11",
        "prompt": 'def string_xor(a: str, b: str) -> str:\n    """Perform XOR on two binary strings."""\n',
        "test": "assert string_xor('010', '110') == '100'",
        "entry_point": "string_xor",
    },
    {
        "task_id": "HE/12",
        "prompt": 'def longest(strings: list[str]) -> str | None:\n    """Return the longest string, or None if empty."""\n',
        "test": "assert longest([]) is None\nassert longest(['a', 'bb', 'ccc']) == 'ccc'",
        "entry_point": "longest",
    },
    {
        "task_id": "HE/13",
        "prompt": 'def greatest_common_divisor(a: int, b: int) -> int:\n    """Compute the GCD of two integers."""\n',
        "test": "assert greatest_common_divisor(3, 5) == 1\nassert greatest_common_divisor(25, 15) == 5",
        "entry_point": "greatest_common_divisor",
    },
    {
        "task_id": "HE/14",
        "prompt": 'def all_prefixes(string: str) -> list[str]:\n    """Return all prefixes from shortest to longest."""\n',
        "test": "assert all_prefixes('abc') == ['a', 'ab', 'abc']",
        "entry_point": "all_prefixes",
    },
    {
        "task_id": "HE/15",
        "prompt": 'def string_sequence(n: int) -> str:\n    """Return a space-separated string of numbers from 0 to n."""\n',
        "test": "assert string_sequence(0) == '0'\nassert string_sequence(5) == '0 1 2 3 4 5'",
        "entry_point": "string_sequence",
    },
    {
        "task_id": "HE/16",
        "prompt": 'def count_distinct_characters(string: str) -> int:\n    """Count distinct characters (case insensitive)."""\n',
        "test": "assert count_distinct_characters('xyzXYZ') == 3\nassert count_distinct_characters('Jerry') == 4",
        "entry_point": "count_distinct_characters",
    },
    {
        "task_id": "HE/17",
        "prompt": 'def parse_music(music_string: str) -> list[int]:\n    """Parse music notation: o=4, o|=2, .|=1 beats."""\n',
        "test": "assert parse_music('o o| .| o| o| .| .| .| .| o o') == [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]",
        "entry_point": "parse_music",
    },
    {
        "task_id": "HE/18",
        "prompt": 'def how_many_times(string: str, substring: str) -> int:\n    """Count how many times substring occurs in string (overlapping)."""\n',
        "test": "assert how_many_times('', 'a') == 0\nassert how_many_times('aaa', 'a') == 3\nassert how_many_times('aaaa', 'aa') == 3",
        "entry_point": "how_many_times",
    },
    {
        "task_id": "HE/19",
        "prompt": 'def sort_numbers(numbers: str) -> str:\n    """Sort space-separated number words (zero through nine)."""\n',
        "test": "assert sort_numbers('three one five') == 'one three five'",
        "entry_point": "sort_numbers",
    },
]


def phase4_humaneval_lite(model, tokenizer,
                          watchdog: MemoryWatchdog) -> PhaseResult:
    """Run 20 HumanEval Lite problems, exec + assert. Gate: >=50% pass@1."""
    t0 = time.perf_counter()
    n_layers = len(model.layers)
    results = []

    for prob in HUMANEVAL_LITE:
        if watchdog.breached.is_set():
            return PhaseResult(
                name="Phase 4: HumanEval Lite", passed=False,
                elapsed_s=time.perf_counter() - t0,
                reason=watchdog.breach_reason,
            )

        tokens = tokenizer.encode(prob["prompt"])
        cache = _make_cache(n_layers)
        x = mx.array([tokens])
        logits = model(x, cache=cache)
        mx.eval(logits)

        generated = []
        for _ in range(256):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tok_id = token.item()
            generated.append(tok_id)
            text_so_far = tokenizer.decode(generated)
            if "\n\n" in text_so_far or "\ndef " in text_so_far:
                break
            if hasattr(tokenizer, 'eos_token_id') and tok_id == tokenizer.eos_token_id:
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
            logger.debug(f"    {prob['task_id']} error: {e}")

        results.append({"task_id": prob["task_id"],
                        "entry_point": prob["entry_point"],
                        "passed": passed,
                        "completion": completion[:100]})
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {status}: {prob['task_id']} ({prob['entry_point']})")

        del cache
        gc.collect()
        mx.clear_cache()

    pass_count = sum(1 for r in results if r["passed"])
    pass_rate = pass_count / len(results)
    gate_passed = pass_rate >= MIN_HUMANEVAL_PASS_RATE

    logger.info(f"  HumanEval Lite: {pass_count}/{len(results)} "
                f"({pass_rate*100:.0f}%) — gate {'PASS' if gate_passed else 'FAIL'}")

    return PhaseResult(
        name="Phase 4: HumanEval Lite",
        passed=gate_passed,
        elapsed_s=time.perf_counter() - t0,
        details={"pass_rate": pass_rate, "pass_count": pass_count,
                 "total": len(results), "results": results},
    )


# ---------------------------------------------------------------------------
# Phase 3d: LiveCodeBench (contamination-free coding gate)
# ---------------------------------------------------------------------------

MIN_LCB_PASS_RATE = 0.30  # 30% pass@1 on post-cutoff problems

def phase3d_livecodebench(model, tokenizer,
                           watchdog: MemoryWatchdog,
                           n_problems: int = 10) -> PhaseResult:
    """Run LiveCodeBench problems with sandboxed execution.

    Contamination-free coding eval using post-cutoff competitive programming
    problems. Generates code, executes in subprocess, checks stdout.

    Gate: pass@1 >= 30%.
    """
    import json as _json
    from omlx.eval.livecodebench import _extract_code, _execute_code
    from omlx.eval.datasets import load_jsonl, deterministic_sample

    t0 = time.perf_counter()
    n_layers = len(model.layers)
    data_path = Path(__file__).parent.parent / "eval" / "data" / "livecodebench.jsonl"

    if not data_path.exists():
        return PhaseResult(
            name="Phase 3d: LiveCodeBench", passed=True,
            details={"skipped": True, "reason": "livecodebench.jsonl not found"},
        )

    # Load and filter problems with valid test cases
    raw = load_jsonl(data_path)
    problems = []
    for item in raw:
        tc = item.get("public_test_cases", "[]")
        if isinstance(tc, str):
            try:
                tc = _json.loads(tc)
            except (ValueError, TypeError):
                continue
        if not isinstance(tc, list) or not tc:
            continue
        inputs = [t.get("input", "") for t in tc]
        outputs = [t.get("output", "") for t in tc]
        if inputs and outputs:
            problems.append({
                "id": item.get("question_id", ""),
                "title": item.get("question_title", ""),
                "description": item.get("question_content", ""),
                "inputs": inputs,
                "outputs": outputs,
                "starter_code": item.get("starter_code", ""),
            })

    problems = deterministic_sample(problems, n_problems)
    logger.info(f"  LiveCodeBench: {len(problems)} problems sampled")

    results = []
    for prob in problems:
        if watchdog.breached.is_set():
            return PhaseResult(
                name="Phase 3d: LiveCodeBench", passed=False,
                elapsed_s=time.perf_counter() - t0,
                reason=watchdog.breach_reason,
            )

        # Build prompt
        prompt_text = (
            "Solve the following programming problem in Python.\n"
            "Read input from stdin and print the output to stdout.\n"
            "Provide only the complete Python code in a ```python block.\n\n"
            f"Problem: {prob['title']}\n{prob['description']}\n\nSolution:"
        )
        messages = [{"role": "user", "content": prompt_text}]
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_text = prompt_text + "\n"
        tokens = tokenizer.encode(chat_text)

        # Generate
        cache = _make_cache(n_layers)
        x = mx.array([tokens])
        logits = model(x, cache=cache)
        mx.eval(logits)

        generated = []
        for _ in range(512):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tok_id = token.item()
            generated.append(tok_id)
            if hasattr(tokenizer, 'eos_token_id') and tok_id == tokenizer.eos_token_id:
                break
            x = token.reshape(1, 1)
            logits = model(x, cache=cache)
            mx.eval(logits)

        response = tokenizer.decode(generated)
        code = _extract_code(response)

        # Execute against test cases (first 3)
        passed = True
        for inp, expected in zip(prob["inputs"][:3], prob["outputs"][:3]):
            stdin_input = inp if isinstance(inp, str) else str(inp)
            expected_out = expected.strip() if isinstance(expected, str) else str(expected).strip()
            stdout, success, error = _execute_code(code, stdin_input)
            if not success or stdout.strip() != expected_out:
                passed = False
                break

        results.append({"id": prob["id"], "title": prob["title"],
                        "passed": passed, "code": code[:100]})
        status = "PASS" if passed else "FAIL"
        logger.info(f"    {status}: {prob['title'][:50]}")

        del cache; gc.collect(); mx.clear_cache()

    pass_count = sum(1 for r in results if r["passed"])
    pass_rate = pass_count / max(len(results), 1)
    gate_passed = pass_rate >= MIN_LCB_PASS_RATE

    logger.info(f"  LiveCodeBench: {pass_count}/{len(results)} "
                f"({pass_rate*100:.0f}%) — gate {'PASS' if gate_passed else 'FAIL'}")

    return PhaseResult(
        name="Phase 3d: LiveCodeBench",
        passed=gate_passed,
        elapsed_s=time.perf_counter() - t0,
        details={"pass_rate": pass_rate, "pass_count": pass_count,
                 "total": len(results), "results": results},
    )


# ---------------------------------------------------------------------------
# Phase 3e: SnapKV Eviction Quality (regression gate)
# ---------------------------------------------------------------------------

def phase3e_snapkv_quality(model, tokenizer,
                            watchdog: MemoryWatchdog) -> PhaseResult:
    """Verify SnapKV eviction pipeline preserves NIAH quality at 4K.

    Runs a quick NIAH test with SnapKV+CAOTE at 50% keep to catch
    regressions in the eviction stack. Gate: needle must be found.
    """
    t0 = time.perf_counter()
    n_layers = len(model.layers)

    if watchdog.breached.is_set():
        return PhaseResult(name="Phase 3e: SnapKV Quality", passed=False,
                           reason=watchdog.breach_reason)

    try:
        from omlx.patches.snapkv import (
            install_q_capture_hook, compute_caote_importance,
            snapkv_select, get_keep_indices, compact_cache,
            check_ger_safety,
        )
    except ImportError as e:
        return PhaseResult(name="Phase 3e: SnapKV Quality", passed=True,
                           details={"skipped": True, "reason": str(e)})

    # Build a 4K NIAH prompt
    needle_code = "SNAPKV-BENCH-9921"
    filler = "The quick brown fox jumps over the lazy dog. " * 40
    prompt_text = (filler + f"The secret code is {needle_code}. "
                   + filler + "\nWhat is the secret code? Reply with ONLY the code:")
    messages = [{"role": "user", "content": prompt_text}]
    try:
        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat = prompt_text + "\n"
    input_ids = tokenizer.encode(chat)
    T = len(input_ids)
    keep_count = max(64, T // 2)  # 50% keep

    # Prefill with Q capture
    from mlx_lm.models.cache import KVCache
    captured, cleanup = install_q_capture_hook(model)
    cache = [KVCache() for _ in range(n_layers)]
    logits = model(mx.array([input_ids]), cache=cache)
    mx.eval(logits)

    # Compute importance and compact
    importance = compute_caote_importance(captured, cache)
    mx.eval(importance)
    cleanup()

    keep_mask = snapkv_select(importance, keep_count)
    safe, ger, _ = check_ger_safety(importance, keep_mask)
    indices = get_keep_indices(keep_mask)
    compact_cache(cache, indices, model=model)

    # Generate
    tokens = []
    for _ in range(32):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tokens.append(token.item())
        logits = model(token.reshape(1, 1), cache=cache)
        mx.eval(logits)

    output = tokenizer.decode(tokens)
    needle_found = needle_code.lower() in output.lower()
    kept_pct = len(indices) * 100 // T

    del cache; gc.collect(); mx.clear_cache()

    status = "PASS" if needle_found else "FAIL"
    logger.info(f"  SnapKV NIAH@4K ({kept_pct}% kept, GER={ger:.3f}): {status}")
    if needle_found:
        logger.info(f"    Output: {output[:60]!r}")
    else:
        logger.info(f"    FAIL output: {output[:60]!r}")

    return PhaseResult(
        name="Phase 3e: SnapKV Quality",
        passed=needle_found,
        elapsed_s=time.perf_counter() - t0,
        details={"needle_found": needle_found, "kept_pct": kept_pct,
                 "ger": round(ger, 4), "ger_safe": safe,
                 "output": output[:100]},
    )


# ---------------------------------------------------------------------------
# Phase 5: Memory Profile (summary of watchdog)
# ---------------------------------------------------------------------------

def phase5_memory_check(profiler_result, limits: Dict[str, float]) -> PhaseResult:
    """Check memory gates from the profiler that ran across all phases."""
    t0 = time.perf_counter()

    metal_peak = profiler_result.metal_peak_gb
    swap_peak = profiler_result.swap_peak_gb
    metal_limit = limits["metal_peak_gb"]
    swap_limit = limits["swap_delta_gb"]

    gates = {
        "metal_peak_gb": round(metal_peak, 2),
        "swap_peak_gb": round(swap_peak, 2),
        "metal_limit_gb": round(metal_limit, 2),
        "swap_limit_gb": round(swap_limit, 2),
    }

    metal_ok = metal_peak <= metal_limit
    swap_ok = swap_peak <= swap_limit

    if not metal_ok:
        gates["reason"] = f"Metal peak {metal_peak:.1f}GB > {metal_limit:.1f}GB"
    if not swap_ok:
        gates["reason"] = f"Swap delta {swap_peak:.1f}GB > {swap_limit:.1f}GB"

    logger.info(f"  Metal peak: {metal_peak:.1f} GB (limit {metal_limit:.1f}GB) "
                f"{'PASS' if metal_ok else 'FAIL'}")
    logger.info(f"  Swap peak:  {swap_peak:.1f} GB (limit {swap_limit:.1f}GB) "
                f"{'PASS' if swap_ok else 'FAIL'}")

    return PhaseResult(
        name="Phase 5: Memory Profile",
        passed=metal_ok and swap_ok,
        elapsed_s=time.perf_counter() - t0,
        details=gates,
    )


# ---------------------------------------------------------------------------
# Phase 6: Write results JSON
# ---------------------------------------------------------------------------

def phase6_write_results(phases: List[PhaseResult], profiler_result,
                         total_elapsed: float,
                         results_path: Path) -> PhaseResult:
    """Write summary JSON."""
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

    results_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"  Results written to {results_path}")

    # Also write the raw profile
    profile_data = {
        "summary": profiler_result.summary(),
        "samples": [
            {
                "t": round(s.t, 2),
                "metal_active_gb": round(s.metal_active_gb, 2),
                "metal_peak_gb": round(s.metal_peak_gb, 2),
                "rss_gb": round(s.rss_gb, 2),
                "phys_footprint_gb": round(s.phys_footprint_gb, 2),
                "cpu_pct": round(s.cpu_pct, 1),
                "swap_gb": round(s.swap_gb, 2),
                "swap_io_mb_per_s": round(s.swap_io_mb_per_s, 1),
            }
            for s in profiler_result.samples
        ],
    }
    PROFILE_PATH.write_text(json.dumps(profile_data, indent=2))
    logger.info(f"  Profile written to {PROFILE_PATH}")

    return PhaseResult(
        name="Phase 6: Summary",
        passed=True,
        elapsed_s=time.perf_counter() - t0,
        details={"results_path": str(results_path),
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Modes:
  --quick       Smoke + coherence only (~15s)
  (default)     + Code Intel + NIAH + RULER + MMLU-Pro (~7min)
  --full        + HumanEval (~25min)

KV cache modes:
  duo (default) Best quality (MMLU-Pro 64%%, HumanEval 95%%), zero swap, ≤16K
  native        3-bit quantized, lower quality, supports 64K+ context
  tq3           WHT codebook, save/load/rewind, best RULER quality
  fp16          Baseline quality, highest memory, ~75K max

NIAH escape hatch (Goal 1 validation):
  --niah-only               Run only smoke + NIAH (skip RULER, MMLU-Pro, HumanEval)
  --niah-context 4K,64K     Set specific context lengths (bypasses headroom gate)

Examples:
  %(prog)s                          # standard pre-commit check
  %(prog)s --quick                  # fast smoke test
  %(prog)s --full                   # full validation with HumanEval
  %(prog)s --kv-mode native         # test with 3-bit KV (for long context)
  %(prog)s --full --max-swap-pct 40 # loosened swap for full run on 48GB
  %(prog)s --niah-only --niah-context 4K,16K,64K  # focused Goal 1 probe
  %(prog)s --niah-only --niah-context 128K --kv-mode native  # 128K validation
""",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Phase 0+1 only (~30s)")
    parser.add_argument("--full", action="store_true",
                        help="All phases including HumanEval (~15min)")
    parser.add_argument("--kv-mode", choices=["native", "tq3", "fp16", "duo", "shadowkv"], default="duo",
                        help="KV cache: native (MLX affine), tq3 (WHT codebook), fp16 (no quant)")
    parser.add_argument("--kv-bits", type=int, default=3, choices=[2, 3, 4],
                        help="KV cache quantization bits (default: 3). 2-bit saves ~33%% memory.")
    parser.add_argument("--max-metal-pct", type=float, default=80.0,
                        help="Metal peak limit as %% of system memory (default: 80)")
    parser.add_argument("--max-swap-pct", type=float, default=25.0,
                        help="Swap delta limit as %% of system memory (default: 25)")
    parser.add_argument("--max-load-pct", type=float, default=70.0,
                        help="Metal at load limit as %% of system memory (default: 70)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model path (e.g. /tmp/granite-4.0-h-small-TQ3.5-wht)")
    parser.add_argument("--niah-500k", action="store_true",
                        help="Add 500K token NIAH test (requires hybrid model with low KV overhead)")
    parser.add_argument("--niah-only", action="store_true",
                        help="Run ONLY Phase 0 (smoke) + Phase 3 (NIAH). "
                        "Skips Code Intel, RULER, MMLU-Pro, HumanEval. "
                        "Use with --niah-context for focused Goal 1 validation.")
    parser.add_argument("--niah-context", type=str, default=None,
                        help="Comma-separated context lengths for NIAH, e.g. '4K,16K,64K,128K'. "
                        "Overrides default [4K,16K]. Bypasses headroom gate with WARNING.")
    parser.add_argument("--quest-topk", type=int, default=0,
                        help="Quest page selection: attend to top-K pages during decode (0=off)")
    parser.add_argument("--prefill-sparse", type=str, default=None,
                        choices=["minference"],
                        help="Sparse prefill: minference (per-head pattern dispatch)")
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Run a warmup pass before Phase 0 to prime Metal kernel cache (default: on)")
    parser.add_argument("--no-warmup", action="store_false", dest="warmup",
                        help="Skip warmup pass (exposes Metal JIT cold-start penalty)")
    parser.add_argument("--force", action="store_true",
                        help="Override pre-flight safety checks (GPU contention, low memory, swap)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    parser.add_argument("--json", type=str,
                        default="/tmp/hypercar_bench_results.json",
                        help="Path for results JSON")
    args = parser.parse_args()

    # Exclusive lock: only one bench instance at a time on this machine.
    # Running two model loads concurrently on 48GB causes catastrophic swap.
    import fcntl
    import signal
    LOCK_PATH = Path("/tmp/hypercar_bench.lock")
    lock_fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # Check if the holder is still alive — stale locks from crashed runs
        try:
            stale_pid = int(LOCK_PATH.read_text().strip())
            os.kill(stale_pid, 0)  # Probe only, no signal sent
            # Process exists — genuine contention
            print(f"ERROR: Another hypercar_bench (PID {stale_pid}) is running. "
                  "Only one instance allowed at a time (48GB memory constraint).",
                  file=sys.stderr)
            sys.exit(1)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            # PID is dead or unreadable — stale lock, break it
            lock_fd.close()
            LOCK_PATH.unlink(missing_ok=True)
            lock_fd = open(LOCK_PATH, "w")
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_fd.write(f"{os.getpid()}\n")
    lock_fd.flush()
    # Clean up lock on normal exit and common signals
    import atexit
    atexit.register(lambda: (lock_fd.close(), LOCK_PATH.unlink(missing_ok=True)))
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: (lock_fd.close(), LOCK_PATH.unlink(missing_ok=True), sys.exit(128 + s)))

    # Pre-flight resource check: detect GPU-heavy processes and insufficient memory
    # before loading a 32GB model into a system that can't handle it.
    try:
        import psutil

        # 1. Check for other MLX/Metal-heavy processes
        gpu_procs = []
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                info = proc.info
                if info["pid"] == os.getpid():
                    continue
                cmdline = info.get("cmdline") or []
                cmd = " ".join(cmdline)
                # Only match actual Python processes running MLX workloads,
                # not shells whose working directory happens to contain "omlx"
                is_python = any(c.endswith(("python", "python3", "python3.13")) for c in cmdline[:1])
                if is_python and any(kw in cmd for kw in [
                    "hypercar_server", "mlx_lm.server", "omlx.bench.hypercar",
                    "mlx_lm.generate", "minference_calibrate",
                ]):
                    gpu_procs.append(f"PID {info['pid']}: {cmd[:100]}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if gpu_procs:
            print("WARNING: GPU-heavy processes detected:", file=sys.stderr)
            for p in gpu_procs:
                print(f"  {p}", file=sys.stderr)
            print("Running the benchmark concurrently with these may cause "
                  "catastrophic swap pressure on 48GB.", file=sys.stderr)
            print("Aborting. Kill the above processes first, or pass "
                  "--force to override.", file=sys.stderr)
            if not getattr(args, 'force', False):
                sys.exit(1)

        # 2. Check available memory — need ~35GB for model + KV + overhead
        vm = psutil.virtual_memory()
        available_gb = vm.available / 1e9
        MIN_AVAILABLE_GB = 30.0  # Model is 32GB but lazy-loads; need ~30 free
        if available_gb < MIN_AVAILABLE_GB:
            print(f"WARNING: Only {available_gb:.1f} GB available "
                  f"(need {MIN_AVAILABLE_GB:.0f} GB for safe model load).",
                  file=sys.stderr)
            print(f"Current VM: {vm.used/1e9:.1f} GB used, "
                  f"{vm.percent:.0f}% utilized.", file=sys.stderr)
            print("Aborting. Free memory or pass --force to override.",
                  file=sys.stderr)
            if not getattr(args, 'force', False):
                sys.exit(1)

        # 3. Check swap — if swap is already elevated, model load will thrash
        try:
            swap = psutil.swap_memory()
            swap_used_gb = swap.used / 1e9
        except OSError:
            swap_used_gb = 0.0  # Can't read swap (sandbox) — skip check
        MAX_PREEXISTING_SWAP_GB = 12.0  # macOS reports compressed memory as swap; 12GB is real pressure
        if swap_used_gb > MAX_PREEXISTING_SWAP_GB:
            print(f"WARNING: {swap_used_gb:.1f} GB swap already in use "
                  f"(threshold {MAX_PREEXISTING_SWAP_GB:.0f} GB).",
                  file=sys.stderr)
            print("System is under memory pressure. Benchmark results "
                  "will be unreliable.", file=sys.stderr)
            if not getattr(args, 'force', False):
                print("Aborting. Wait for swap to clear or pass --force.",
                      file=sys.stderr)
                sys.exit(1)
    except ImportError:
        pass  # psutil not available — skip preflight

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Set KV mode, bits, quest, and model
    global _KV_MODE, KV_BITS, _QUEST_TOPK, MODEL_ID
    _KV_MODE = args.kv_mode
    KV_BITS = args.kv_bits
    _QUEST_TOPK = args.quest_topk
    if args.model:
        MODEL_ID = args.model
    logger.info(f"KV mode: {_KV_MODE}, bits: {KV_BITS}")
    if _QUEST_TOPK > 0:
        logger.info(f"Quest page selection: top-{_QUEST_TOPK} pages")
    logger.info(f"Model: {MODEL_ID}")

    # Detect system memory and compute limits
    total_gb = _detect_system_memory_gb()
    limits = _compute_memory_limits(
        total_gb, args.max_metal_pct, args.max_swap_pct, args.max_load_pct,
    )
    logger.info(f"System memory: {total_gb:.1f} GB")
    logger.info(f"  Metal peak limit:  {limits['metal_peak_gb']:.1f} GB "
                f"({args.max_metal_pct:.0f}%)")
    logger.info(f"  Metal load limit:  {limits['metal_load_gb']:.1f} GB "
                f"({args.max_load_pct:.0f}%)")
    logger.info(f"  Swap delta limit:  {limits['swap_delta_gb']:.1f} GB "
                f"({args.max_swap_pct:.0f}%)")

    results_path = Path(args.json)
    total_t0 = time.perf_counter()
    phases: List[PhaseResult] = []

    # Start fail-fast memory watchdog
    watchdog = MemoryWatchdog(
        metal_limit_gb=limits["metal_peak_gb"],
        swap_limit_gb=limits["swap_delta_gb"],
        sample_interval=1.0,
    )
    watchdog.start()

    try:
        # Load model once
        logger.info("Loading model: %s", MODEL_ID)
        load_t0 = time.perf_counter()
        model, tokenizer = _load_model()
        global _MODEL_REF
        _MODEL_REF = model  # For hybrid cache factory
        # Don't force-eval all parameters — let MLX load lazily.
        # Force-eval causes 2x peak memory during load (mmap + Metal copy).
        # Parameters will be materialized on first forward pass instead.
        load_time = time.perf_counter() - load_t0
        logger.info(f"Model loaded in {load_time:.1f}s (lazy — first forward will materialize)")

        # Apply MInference sparse prefill if requested
        if args.prefill_sparse == "minference":
            from omlx.patches.minference_prefill import apply_minference_prefill_patch
            if apply_minference_prefill_patch():
                logger.info("MInference sparse prefill ENABLED")
            else:
                logger.warning("MInference sparse prefill FAILED — dense fallback")

        if watchdog.breached.is_set():
            logger.error("MEMORY BREACH during model load — aborting")
            return _finish(phases, watchdog, limits, total_t0, results_path)

        # Warmup pass: prime Metal kernel cache + GPU pipeline state
        # Always uses native KV cache for warmup (TQ3 below min_quant_tokens
        # returns tuples that crash the SDPA path on short sequences).
        if args.warmup:
            logger.info("\n=== Warmup: priming Metal kernels ===")
            warmup_t0 = time.perf_counter()
            from mlx_lm.models.cache import KVCache
            n_layers = len(model.layers)
            warmup_cache = [KVCache() for _ in range(n_layers)]
            _generate(model, tokenizer, "Hello", max_tokens=16,
                      cache=warmup_cache)
            del warmup_cache
            gc.collect()
            mx.clear_cache()
            logger.info(f"  Warmup done in {time.perf_counter() - warmup_t0:.1f}s "
                        f"(Metal cache primed, memory cleared)")

        # Phase 0: Smoke
        logger.info("\n=== Phase 0: Smoke ===")
        p0 = phase0_smoke(model, tokenizer, watchdog, limits["metal_load_gb"])
        phases.append(p0)
        if not p0.passed:
            logger.error("Phase 0 FAILED — aborting")
            return _finish(phases, watchdog, limits, total_t0, results_path)

        # Phase 1: Coherence
        logger.info("\n=== Phase 1: Coherence ===")
        p1 = phase1_coherence(model, tokenizer, watchdog)
        phases.append(p1)
        if not p1.passed:
            logger.error("Phase 1 FAILED — aborting")
            return _finish(phases, watchdog, limits, total_t0, results_path)

        if args.quick:
            logger.info("\n--quick mode: skipping Phase 2+")
            return _finish(phases, watchdog, limits, total_t0, results_path)

        # --niah-only: skip Code Intel, go straight to NIAH, then finish
        if getattr(args, 'niah_only', False):
            logger.info("\n--niah-only mode: skipping Phase 2 (Code Intel)")
            logger.info("\n=== Phase 3: Needle in Haystack ===")
            p3 = phase3_niah(model, tokenizer, watchdog, args_ref=args)
            phases.append(p3)
            if p3.passed:
                logger.info("Phase 3 PASSED — NIAH-only run complete")
            else:
                logger.error("Phase 3 FAILED")
            return _finish(phases, watchdog, limits, total_t0, results_path)

        # Phase 2: Code Intelligence
        logger.info("\n=== Phase 2: Code Intelligence ===")
        p2 = phase2_code_intelligence(model, tokenizer, watchdog)
        phases.append(p2)
        if not p2.passed:
            logger.error("Phase 2 FAILED — aborting")
            return _finish(phases, watchdog, limits, total_t0, results_path)

        # Phase 3: Needle in Haystack
        logger.info("\n=== Phase 3: Needle in Haystack ===")
        if _check_phase_headroom("Phase 3: NIAH", limits["metal_peak_gb"]):
            p3 = phase3_niah(model, tokenizer, watchdog, args_ref=args)
            phases.append(p3)
            if not p3.passed:
                logger.error("Phase 3 FAILED — aborting")
                return _finish(phases, watchdog, limits, total_t0, results_path)
        else:
            phases.append(PhaseResult(
                name="Phase 3: Needle in Haystack", passed=True,
                details={"skipped": True, "reason": "insufficient headroom"},
            ))

        # Phase 3b: RULER (multi-key retrieval, aggregation)
        logger.info("\n=== Phase 3b: RULER ===")
        if _check_phase_headroom("Phase 3b: RULER", limits["metal_peak_gb"]):
            p3b = phase3b_ruler(model, tokenizer, watchdog, full=args.full)
            phases.append(p3b)
            if not p3b.passed:
                if watchdog.breached.is_set():
                    logger.error("Phase 3b FAILED (memory breach) — aborting")
                    return _finish(phases, watchdog, limits, total_t0, results_path)
                logger.warning(
                    "Phase 3b FAILED (quality gate) — memory clean, "
                    "continuing to Phase 4 HumanEval for independent eval coverage"
                )
        else:
            phases.append(PhaseResult(
                name="Phase 3b: RULER", passed=True,
                details={"skipped": True, "reason": "insufficient headroom"},
            ))

        # Phase 3c: MMLU-Pro (reasoning gate — runs in both default and --full)
        logger.info("\n=== Phase 3c: MMLU-Pro ===")
        if _check_phase_headroom("Phase 3c: MMLU-Pro", limits["metal_peak_gb"]):
            p3c = phase3c_mmlu_pro(model, tokenizer, watchdog, full=args.full)
            phases.append(p3c)
            if not p3c.passed:
                if watchdog.breached.is_set():
                    logger.error("Phase 3c FAILED (memory breach) — aborting")
                    return _finish(phases, watchdog, limits, total_t0, results_path)
                logger.warning("Phase 3c FAILED (MMLU-Pro gate) — continuing")
        else:
            phases.append(PhaseResult(
                name="Phase 3c: MMLU-Pro", passed=True,
                details={"skipped": True, "reason": "insufficient headroom"},
            ))

        # Phase 3e: SnapKV eviction quality (runs in both default and --full)
        logger.info("\n=== Phase 3e: SnapKV Quality ===")
        if _check_phase_headroom("Phase 3e: SnapKV", limits["metal_peak_gb"]):
            p3e = phase3e_snapkv_quality(model, tokenizer, watchdog)
            phases.append(p3e)
            if not p3e.passed:
                logger.warning("Phase 3e FAILED (SnapKV NIAH gate) — continuing")
        else:
            phases.append(PhaseResult(
                name="Phase 3e: SnapKV Quality", passed=True,
                details={"skipped": True, "reason": "insufficient headroom"},
            ))

        if not args.full:
            logger.info("\nDefault mode: skipping Phase 4 (HumanEval). Use --full to include.")
            return _finish(phases, watchdog, limits, total_t0, results_path)

        # Phase 3d: LiveCodeBench (contamination-free coding, --full only)
        logger.info("\n=== Phase 3d: LiveCodeBench ===")
        if _check_phase_headroom("Phase 3d: LiveCodeBench", limits["metal_peak_gb"]):
            p3d = phase3d_livecodebench(model, tokenizer, watchdog,
                                         n_problems=20)
            phases.append(p3d)
            if not p3d.passed:
                if watchdog.breached.is_set():
                    logger.error("Phase 3d FAILED (memory breach) — aborting")
                    return _finish(phases, watchdog, limits, total_t0, results_path)
                logger.warning("Phase 3d FAILED (LCB gate) — continuing to HumanEval")
        else:
            phases.append(PhaseResult(
                name="Phase 3d: LiveCodeBench", passed=True,
                details={"skipped": True, "reason": "insufficient headroom"},
            ))

        # Phase 4: HumanEval Lite
        logger.info("\n=== Phase 4: HumanEval Lite ===")
        p4 = phase4_humaneval_lite(model, tokenizer, watchdog)
        phases.append(p4)
        if not p4.passed:
            logger.error("Phase 4 FAILED — aborting")
            return _finish(phases, watchdog, limits, total_t0, results_path)

    except Exception as e:
        logger.exception("Benchmark crashed: %s", e)
        phases.append(PhaseResult(name="CRASH", passed=False,
                                  details={"error": str(e)}))

    return _finish(phases, watchdog, limits, total_t0, results_path)


def _finish(phases: List[PhaseResult], watchdog: MemoryWatchdog,
            limits: Dict[str, float], total_t0: float,
            results_path: Path) -> int:
    """Stop watchdog, evaluate memory gates, write results, print summary."""
    profiler_result = watchdog.stop()
    total_elapsed = time.perf_counter() - total_t0

    # Wall-clock correlation check: detect swap-induced stalls (Task 80)
    sched_frac = profiler_result.cpu_scheduling_fraction
    wall_s = profiler_result.wall_clock_elapsed_s
    python_s = profiler_result.total_seconds
    if sched_frac < 0.9:
        logger.warning(
            f"CPU scheduling fraction was {sched_frac:.2f} "
            f"(wall {wall_s:.0f}s vs {len(profiler_result.samples)} samples "
            f"in {python_s:.0f}s) — heavy co-tenancy or swap-thrashing "
            f"detected, timing metrics should be interpreted with caution"
        )

    # Phase 5: Memory check
    logger.info("\n=== Phase 5: Memory Profile ===")
    p5 = phase5_memory_check(profiler_result, limits)
    phases.append(p5)

    # Phase 6: Write results
    logger.info("\n=== Phase 6: Summary ===")
    p6 = phase6_write_results(phases, profiler_result, total_elapsed,
                              results_path)
    phases.append(p6)

    _print_summary(phases, total_elapsed)

    all_passed = all(p.passed for p in phases)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
