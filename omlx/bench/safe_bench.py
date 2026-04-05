# SPDX-License-Identifier: Apache-2.0
"""Safe benchmark harness for oMLX context window tests.

Prevents the 97GB OOM scenario by enforcing:
  1. Lock file — only one benchmark at a time
  2. Swap monitoring — fail if swap exceeds limit
  3. Metal memory monitoring — fail if active memory exceeds limit
  4. tok/s minimum — fail early if prefill is too slow
  5. Coherence test — fail if model output is garbage

Usage:
    python -m omlx.bench.safe_bench \
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit \
        --contexts 65536 131072 196608 262144 \
        --cache-mode turbo3 \
        --max-metal-gb 38 \
        --max-swap-gb 8

Memory budget (48GB M4 Pro):
    Model weights:   17.2 GB (fixed)
    OS + apps:        4.0 GB
    Safety margin:    6.0 GB
    Available:       ~21 GB for KV + working set
    Max Metal:        38 GB (soft limit — includes some swap headroom)
    Max Swap:          8 GB (hard limit — beyond = death spiral)
"""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

logger = logging.getLogger("omlx.bench")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCK_FILE = Path(tempfile.gettempdir()) / "omlx_bench.lock"
DEFAULT_MAX_METAL_GB = 38.0   # 38 GB Metal active memory
DEFAULT_MAX_SWAP_GB = 8.0     # 8 GB swap — beyond = SSD thrashing
DEFAULT_MIN_PREFILL_TOKS = 50.0   # Min prefill tok/s (one-time buffer cost)
DEFAULT_MIN_DECODE_TOKS = 25.0    # Min decode tok/s (the speed that matters)
PREFILL_CHUNK_SIZE = 2048     # Tokens per prefill chunk (matches dequant chunk)
DECODE_TEST_TOKENS = 16       # Tokens to generate for decode speed check
MONITOR_INTERVAL = 2.0        # Seconds between memory checks


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Result of a single context-length benchmark."""
    context_length: int
    status: str  # "pass", "fail_memory", "fail_prefill_speed", "fail_decode_speed", "fail_coherence", "error", "skipped"
    prefill_toks: float = 0.0
    decode_toks: float = 0.0
    active_gb: float = 0.0
    peak_gb: float = 0.0
    swap_gb: float = 0.0
    elapsed_s: float = 0.0
    error_msg: str = ""
    coherence_ok: bool = False
    # Profile aggregates
    cpu_avg_pct: float = 0.0
    cpu_peak_pct: float = 0.0
    rss_peak_gb: float = 0.0


@dataclass
class BenchConfig:
    """Benchmark configuration."""
    model_path: str = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
    contexts: list[int] = field(default_factory=lambda: [65536, 131072])
    cache_mode: str = "turbo3"
    bits: int = 3
    dequant_chunk_size: int = 2048
    max_metal_gb: float = DEFAULT_MAX_METAL_GB
    max_swap_gb: float = DEFAULT_MAX_SWAP_GB
    min_prefill_toks: float = DEFAULT_MIN_PREFILL_TOKS
    min_decode_toks: float = DEFAULT_MIN_DECODE_TOKS
    prefill_chunk: int = PREFILL_CHUNK_SIZE
    decode_test_tokens: int = DECODE_TEST_TOKENS
    coherence_test: bool = True
    run_phase2: bool = True   # Quality: NIAH + TQ3 fidelity
    run_phase3: bool = True   # Stress: sustained decode + memory leak
    run_phase4: bool = True   # Intelligence: quick EvalPlus + setup guide

    # Hypercar feature flags (all default ON)
    use_fp16_layer0: bool = True          # fp16 layer 0 anchor
    fp16_layers: int = 16                 # Number of fp16 layers (was 1, needs 16 for quality)
    use_vertical_eval: bool = True        # mx.eval every 8 layers
    use_streaming_dequant: bool = True    # Online softmax for long history
    use_adaptive_budget: bool = True      # Dynamic chunk sizing
    min_quant_tokens: int = 512           # Stay fp16 below this threshold
    # Decoder features (future)
    use_medusa: bool = False              # Requires distilled draft heads
    medusa_num_heads: int = 3
    medusa_distill_steps: int = 0         # 0 = use random (bad), 200+ = distill
    use_prompt_lookup: bool = False       # N-gram lookup decoding
    use_starc: bool = False               # Sparse attention (DROPPED — slower)


# ---------------------------------------------------------------------------
# Lock file — only one benchmark at a time
# ---------------------------------------------------------------------------

def acquire_lock() -> bool:
    """Acquire exclusive benchmark lock. Returns True if acquired."""
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        atexit.register(_release_lock)
        return True
    except FileExistsError:
        # Check if the holding process is still alive
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if alive
            return False  # Another benchmark is running
        except (ProcessLookupError, ValueError, OSError):
            # Stale lock — remove and retry
            LOCK_FILE.unlink(missing_ok=True)
            return acquire_lock()


def _release_lock():
    """Release the benchmark lock."""
    LOCK_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Swap monitoring
# ---------------------------------------------------------------------------

def get_swap_used_gb() -> float:
    """Get current swap usage in GB via sysctl."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"],
            timeout=5, text=True,
        )
        # Format: "total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)"
        for part in out.split():
            if part.endswith("M") and "used" in out[:out.index(part)]:
                # Find the "used = X.XXM" value
                pass
        # More robust parsing
        parts = out.split()
        for i, p in enumerate(parts):
            if p == "used" and i + 2 < len(parts):
                val = parts[i + 2].rstrip("M")
                return float(val) / 1024.0  # MB to GB
    except Exception:
        pass
    return 0.0


def get_memory_pressure() -> str:
    """Get macOS memory pressure level."""
    try:
        out = subprocess.check_output(
            ["memory_pressure", "-Q"],
            timeout=5, text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "System-wide memory free percentage" in line:
                pct = int(line.split(":")[-1].strip().rstrip("%"))
                if pct < 10:
                    return "critical"
                elif pct < 25:
                    return "warning"
                return "normal"
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Memory watchdog thread
# ---------------------------------------------------------------------------

class MemoryWatchdog:
    """Background thread that monitors Metal and swap memory.

    Raises a flag when limits are exceeded so the benchmark loop can abort.
    """

    def __init__(self, max_metal_gb: float, max_swap_gb: float):
        self.max_metal_bytes = int(max_metal_gb * 1e9)
        self.max_swap_gb = max_swap_gb
        self.violated = False
        self.violation_reason = ""
        self.peak_metal_gb = 0.0
        self.peak_swap_gb = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._swap_baseline_gb = 0.0

    def start(self):
        """Start the watchdog thread."""
        self._swap_baseline_gb = get_swap_used_gb()
        self.violated = False
        self.violation_reason = ""
        self.peak_metal_gb = 0.0
        self.peak_swap_gb = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the watchdog thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def check(self) -> bool:
        """Check if limits are violated. Returns True if OK."""
        return not self.violated

    def _monitor(self):
        """Background monitoring loop."""
        while not self._stop.is_set():
            try:
                # Check Metal active memory
                metal_bytes = mx.get_active_memory()
                metal_gb = metal_bytes / 1e9
                self.peak_metal_gb = max(self.peak_metal_gb, metal_gb)

                if metal_bytes > self.max_metal_bytes:
                    self.violated = True
                    self.violation_reason = (
                        f"Metal memory {metal_gb:.1f}GB exceeds "
                        f"limit {self.max_metal_bytes/1e9:.1f}GB"
                    )
                    logger.error(f"WATCHDOG: {self.violation_reason}")
                    return

                # Check swap
                swap_gb = get_swap_used_gb()
                swap_delta = swap_gb - self._swap_baseline_gb
                self.peak_swap_gb = max(self.peak_swap_gb, swap_delta)

                if swap_delta > self.max_swap_gb:
                    self.violated = True
                    self.violation_reason = (
                        f"Swap grew by {swap_delta:.1f}GB (baseline {self._swap_baseline_gb:.1f}GB), "
                        f"exceeds limit {self.max_swap_gb:.1f}GB"
                    )
                    logger.error(f"WATCHDOG: {self.violation_reason}")
                    return

            except Exception as e:
                logger.warning(f"Watchdog error: {e}")

            self._stop.wait(MONITOR_INTERVAL)


# ---------------------------------------------------------------------------
# Coherence test
# ---------------------------------------------------------------------------

def check_coherence(model, tokenizer, cache, n_layers: int) -> bool:
    """Quick coherence test: ask a trivial question and check the answer.

    Uses the already-loaded model and generates a few tokens to verify
    the model is producing sensible output after prefill.
    """
    # Pad with ~512 tokens of realistic code so TQ3 codebook has real signal.
    # 15 tokens is too few — quantization noise dominates at tiny context.
    preamble = '''"""Utility module for basic arithmetic operations."""

def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Return the difference of two integers."""
    return a - b

def multiply(a: int, b: int) -> int:
    """Return the product of two integers."""
    return a * b

def divide(a: float, b: float) -> float:
    """Return the quotient of two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Examples:
# add(2, 3) = 5
# subtract(10, 4) = 6
# multiply(3, 7) = 21
# divide(15, 3) = 5.0

'''
    prompt = preamble + "What is 2 + 2? Answer with just the number:"
    tokens = tokenizer.encode(prompt)
    x = mx.array([tokens])

    # Create fresh cache for coherence test
    from mlx_lm.models.cache import KVCache as _KVCache
    from omlx.turboquant_kv import TurboQuantKVCache as _TQCache
    test_cache = [_KVCache() if i == 0 else _TQCache(bits=3, dequant_chunk_size=2048)
                  for i in range(n_layers)]

    logits = model(x, cache=test_cache)
    mx.eval(logits)

    # Generate 5 tokens
    generated = []
    for _ in range(5):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        generated.append(token.item())
        x = token.reshape(1, 1)
        logits = model(x, cache=test_cache)
        mx.eval(logits)

    text = tokenizer.decode(generated).strip()
    del test_cache

    # Check if "4" appears in the output
    ok = "4" in text
    if ok:
        logger.info(f"Coherence PASS: '{text}'")
    else:
        logger.error(f"Coherence FAIL: got '{text}', expected '4'")

    return ok


# ---------------------------------------------------------------------------
# Single context benchmark
# ---------------------------------------------------------------------------

def bench_context(
    model,
    tokenizer,
    n_layers: int,
    ctx: int,
    config: BenchConfig,
    watchdog: MemoryWatchdog,
) -> BenchResult:
    """Benchmark a single context length with all safety checks."""

    result = BenchResult(context_length=ctx, status="error")
    profile = None  # Will be populated

    # Reset Metal state
    mx.synchronize()
    mx.clear_cache()

    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache
    cache = [
        KVCache() if i == 0 else TurboQuantKVCache(
            bits=config.bits,
            dequant_chunk_size=config.dequant_chunk_size,
        )
        for i in range(n_layers)
    ]

    # Generate tokens
    base = "x = 1\n"
    base_tokens = tokenizer.encode(base)
    tokens = (base_tokens * ((ctx // len(base_tokens)) + 1))[:ctx]

    logger.info(f"--- {ctx:,} tokens ---")

    # Start watchdog + profiler for this test
    watchdog.start()
    from .profiler import Profiler
    profiler = Profiler(sample_interval=0.5)
    profiler.start()
    t0 = time.perf_counter()

    try:
        # Chunked prefill with progress and safety checks
        processed = 0
        for chunk_start in range(0, len(tokens), config.prefill_chunk):
            chunk_end = min(chunk_start + config.prefill_chunk, len(tokens))
            chunk = tokens[chunk_start:chunk_end]
            x = mx.array([chunk])

            # Adaptive dequant chunk hint (if enabled) — set ONCE per step.
            if config.use_adaptive_budget:
                from omlx.memory_budget import compute_budget, compute_dequant_chunk_size
                from omlx.streaming_attention import set_chunk_hint
                budget = compute_budget(model_gb=17.2, active_kv_gb=0, decoding=False)
                adaptive = compute_dequant_chunk_size(
                    query_len=len(chunk), num_query_heads=32, head_dim=128,
                    num_layers=48, budget=budget,
                )
                set_chunk_hint(adaptive)

            chunk_t0 = time.perf_counter()
            logits = model(x, cache=cache)
            mx.eval(logits)
            mx.synchronize()
            mx.clear_cache()
            chunk_elapsed = time.perf_counter() - chunk_t0

            processed = chunk_end
            chunk_toks = len(chunk) / chunk_elapsed if chunk_elapsed > 0 else 0

            # Check watchdog
            if not watchdog.check():
                result.status = "fail_memory"
                result.error_msg = watchdog.violation_reason
                logger.error(f"  ABORT at {processed:,}/{ctx:,}: {watchdog.violation_reason}")
                break

            # Prefill is a ONE-TIME cost (user pastes code once, then iterates).
            # Decode speed is what matters. Scale prefill floor with context:
            # 50 @ 65K, 30 @ 128K, 15 @ 256K — prefill is buffer time we accept
            # to unlock longer context. Abort only if pathologically slow.
            scaled_min = config.min_prefill_toks * (65536 / max(ctx, 1))
            scaled_min = max(scaled_min, 10.0)  # Absolute floor: 10 tok/s
            if processed >= 32768 and chunk_toks < scaled_min:
                result.status = "fail_prefill_speed"
                result.error_msg = f"Prefill {chunk_toks:.0f} tok/s < {scaled_min:.0f} scaled minimum (base {config.min_prefill_toks:.0f})"
                logger.error(f"  ABORT at {processed:,}/{ctx:,}: {result.error_msg}")
                break

            # Progress every 32K
            if processed % 32768 == 0 or processed == len(tokens):
                metal_gb = mx.get_active_memory() / 1e9
                swap_gb = get_swap_used_gb()
                logger.info(
                    f"  {processed:>7,}/{ctx:,} | "
                    f"{chunk_toks:,.0f} tok/s | "
                    f"metal {metal_gb:.1f}GB | "
                    f"swap {swap_gb:.1f}GB"
                )

        elapsed = time.perf_counter() - t0
        watchdog.stop()
        profile = profiler.stop()
        logger.info(f"  Profile: {profile.summary()}")
        result.cpu_avg_pct = profile.cpu_avg_pct
        result.cpu_peak_pct = profile.cpu_peak_pct
        result.rss_peak_gb = profile.rss_peak_gb

        if result.status not in ("fail_memory", "fail_prefill_speed"):
            result.prefill_toks = processed / elapsed if elapsed > 0 else 0
            result.active_gb = mx.get_active_memory() / 1e9
            result.peak_gb = mx.get_peak_memory() / 1e9
            result.swap_gb = watchdog.peak_swap_gb
            result.elapsed_s = elapsed

            logger.info(
                f"  Prefill done: {ctx:,} tok | "
                f"{result.prefill_toks:,.0f} tok/s | "
                f"active {result.active_gb:.1f}GB | "
                f"peak {result.peak_gb:.1f}GB"
            )

            # --- Decode speed test ---
            decoder_name = "greedy"
            if config.use_medusa:
                decoder_name = f"medusa-{config.medusa_num_heads}h"
            elif config.use_prompt_lookup:
                decoder_name = "prompt-lookup"
            logger.info(f"  Decode test ({decoder_name}): {config.decode_test_tokens} tokens...")

            decode_t0 = time.perf_counter()
            tokens_generated = 0

            if config.use_medusa and config.medusa_distill_steps > 0:
                # TODO: Medusa distillation + TQ3 decode integration
                # For now, warn and fall back to greedy
                logger.warning("  Medusa not fully wired yet — using greedy decode")

            # Standard greedy decode (works with all cache types including TQ3)
            for _ in range(config.decode_test_tokens):
                token = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(token)
                x = token.reshape(1, 1)
                logits = model(x, cache=cache)
                mx.eval(logits)
                tokens_generated += 1

                if not watchdog.check():
                    result.status = "fail_memory"
                    result.error_msg = f"During decode: {watchdog.violation_reason}"
                    break

            decode_elapsed = time.perf_counter() - decode_t0

            if result.status != "fail_memory":
                result.decode_toks = tokens_generated / decode_elapsed if decode_elapsed > 0 else 0

                if result.decode_toks < config.min_decode_toks:
                    result.status = "fail_decode_speed"
                    result.error_msg = (
                        f"Decode {result.decode_toks:.1f} tok/s < "
                        f"{config.min_decode_toks:.0f} minimum at {ctx:,} context"
                    )
                    logger.error(f"  ABORT: {result.error_msg}")
                else:
                    result.status = "pass"
                    logger.info(
                        f"  PASS: {ctx:,} tok | "
                        f"prefill {result.prefill_toks:,.0f} tok/s | "
                        f"decode {result.decode_toks:.1f} tok/s | "
                        f"active {result.active_gb:.1f}GB | "
                        f"peak {result.peak_gb:.1f}GB | "
                        f"swap delta {result.swap_gb:.1f}GB"
                    )

    except Exception as e:
        watchdog.stop()
        profiler.stop()
        result.status = "error"
        result.error_msg = f"{type(e).__name__}: {str(e)[:200]}"
        logger.error(f"  ERROR at {ctx:,}: {result.error_msg}")

    # Cleanup
    del cache
    if "logits" in dir():
        del logits
    gc.collect()
    mx.synchronize()
    mx.clear_cache()

    return result


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(config: BenchConfig) -> list[BenchResult]:
    """Run the full benchmark suite with all safety checks."""

    # 1. Acquire lock
    if not acquire_lock():
        logger.error("Another benchmark is already running (lock file exists)")
        sys.exit(1)

    logger.info(f"Benchmark lock acquired (PID {os.getpid()})")
    logger.info(f"Config: {config}")

    # 2. Load model
    from mlx_lm import load
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch

    apply_turboquant_attention_patch()
    model, tokenizer = load(config.model_path)
    apply_prefill_last_logit_patch(model)

    if config.use_vertical_eval:
        from omlx.patches.vertical_eval import apply_vertical_eval_patch
        apply_vertical_eval_patch(model)
        logger.info("Feature: vertical_eval ENABLED (every 8 layers)")
    else:
        logger.warning("Feature: vertical_eval DISABLED (may cause 45GB peaks)")

    n_layers = model.args.num_hidden_layers

    model_gb = mx.get_active_memory() / 1e9
    logger.info(f"Model loaded: {model_gb:.1f}GB ({n_layers} layers)")

    # 3. Coherence check before running benchmarks
    if config.coherence_test:
        logger.info("Running coherence test...")
        if not check_coherence(model, tokenizer, None, n_layers):
            logger.error("COHERENCE TEST FAILED — model is producing garbage. Aborting.")
            _release_lock()
            sys.exit(1)

    # 4. Create watchdog
    watchdog = MemoryWatchdog(config.max_metal_gb, config.max_swap_gb)

    # 5. Run benchmarks (sorted by context length)
    results = []
    for ctx in sorted(config.contexts):
        result = bench_context(model, tokenizer, n_layers, ctx, config, watchdog)
        results.append(result)

        # If a context length fails, don't try larger ones
        if result.status != "pass":
            logger.warning(
                f"Context {ctx:,} failed ({result.status}). "
                f"Skipping larger contexts."
            )
            for remaining_ctx in sorted(config.contexts):
                if remaining_ctx > ctx and remaining_ctx not in [r.context_length for r in results]:
                    results.append(BenchResult(
                        context_length=remaining_ctx,
                        status="skipped",
                        error_msg=f"Skipped: {ctx:,} failed ({result.status})",
                    ))
            break

    # 6. Summary
    logger.info("\n=== BENCHMARK RESULTS ===")
    logger.info(f"{'Context':>10} | {'Status':>16} | {'Prefill':>9} | {'Decode':>9} | {'Active':>8} | {'Peak':>8} | {'Swap':>8}")
    logger.info("-" * 90)
    for r in results:
        logger.info(
            f"{r.context_length:>10,} | {r.status:>16} | "
            f"{r.prefill_toks:>7,.0f} t/s | "
            f"{r.decode_toks:>7.1f} t/s | "
            f"{r.active_gb:>7.1f}G | "
            f"{r.peak_gb:>7.1f}G | "
            f"{r.swap_gb:>7.1f}G"
        )
        if r.error_msg:
            logger.info(f"{'':>10}   {r.error_msg}")

    # --- Gate: Phase 1 must pass before continuing ---
    from omlx.turboquant_kv import TurboQuantKVCache

    from mlx_lm.models.cache import KVCache

    def _cache_factory(n):
        # First `fp16_layers` layers stay fp16 for quality.
        # At fp16_layers=1 (old default), TQ3 produces garbage at 2K+ context.
        # At fp16_layers=16, quality matches fp16 baseline at 2.5K context.
        # The fp16_layer0 flag is legacy (sets fp16_layers=1 when flag off via=0).
        fp16_count = config.fp16_layers if config.use_fp16_layer0 else 0
        caches = []
        for i in range(n):
            if i < fp16_count:
                caches.append(KVCache())
            else:
                caches.append(TurboQuantKVCache(
                    bits=config.bits,
                    dequant_chunk_size=config.dequant_chunk_size,
                    min_quant_tokens=config.min_quant_tokens,
                ))
        return caches

    logger.info("Features:")
    logger.info(f"  fp16_layers:       {config.fp16_layers if config.use_fp16_layer0 else 0}")
    logger.info(f"  vertical_eval:     {config.use_vertical_eval}")
    logger.info(f"  adaptive_budget:   {config.use_adaptive_budget}")
    logger.info(f"  streaming_dequant: {config.use_streaming_dequant}")
    logger.info(f"  min_quant_tokens:  {config.min_quant_tokens}")
    logger.info(f"  medusa:            {config.use_medusa} (heads={config.medusa_num_heads}, distill={config.medusa_distill_steps})")
    logger.info(f"  prompt_lookup:     {config.use_prompt_lookup}")
    logger.info(f"  dequant_chunk:     {config.dequant_chunk_size}")
    logger.info(f"  prefill_chunk:     {config.prefill_chunk}")

    all_passed = all(r.status == "pass" for r in results)
    any_passed = any(r.status == "pass" for r in results)
    max_passed_ctx = max((r.context_length for r in results if r.status == "pass"), default=0)

    if not any_passed:
        logger.error("\nPhase 1 FAILED — no context lengths passed. Skipping all later phases.")

    # 7. Phase 2: Quality (requires Phase 1 pass)
    phase2_passed = False
    if config.run_phase2 and any_passed:
        from .phases import phase2_niah, phase2_tq3_fidelity

        logger.info("\n=== PHASE 2: QUALITY ===")

        # TQ3 fidelity (quick, short context)
        logger.info("TQ3 vs fp16 fidelity test...")
        fidelity = phase2_tq3_fidelity(model, tokenizer, n_layers)

        if fidelity["cosine_similarity"] < 0.95:
            logger.error(
                f"  TQ3 fidelity FAILED: cosine={fidelity['cosine_similarity']:.6f} < 0.95. "
                f"Skipping later phases."
            )
        else:
            # NIAH at the largest passing context (uncapped — streaming dequant
            # guarantees flat memory, so NIAH at 256K is safe if Phase 1 passed it)
            niah_ctx = max_passed_ctx
            logger.info(f"Needle-in-a-Haystack at {niah_ctx:,} tokens...")
            niah = phase2_niah(model, tokenizer, n_layers, niah_ctx, _cache_factory,
                              config.prefill_chunk)

            niah_pass = sum(1 for v in niah.values() if v["found"])
            niah_total = len(niah)
            logger.info(f"  NIAH: {niah_pass}/{niah_total} depths found needle")
            logger.info(f"  TQ3 fidelity: cosine={fidelity['cosine_similarity']:.6f} "
                         f"top1={'match' if fidelity['top1_match'] else 'MISMATCH'}")

            # NIAH must find needle in at least 3/5 depths
            if niah_pass >= 3:
                phase2_passed = True
            else:
                logger.error(f"  NIAH FAILED: only {niah_pass}/{niah_total} depths. Skipping later phases.")

    # 8. Phase 3: Stress (requires Phase 2 pass)
    phase3_passed = False
    if config.run_phase3 and phase2_passed:
        from .phases import phase3_sustained_decode

        logger.info("\n=== PHASE 3: STRESS ===")

        stress_ctx = max_passed_ctx
        logger.info(f"Sustained decode (128 tokens) at {stress_ctx:,} context...")
        stress = phase3_sustained_decode(
            model, tokenizer, n_layers, stress_ctx, _cache_factory,
            config.prefill_chunk, decode_tokens=128,
            min_toks=config.min_decode_toks,
        )

        if stress["has_leak"]:
            logger.error(f"  Memory leak detected: {stress['mem_delta_gb']:+.3f}GB. Skipping Phase 4.")
        elif not stress["speed_pass"]:
            logger.error(
                f"  Sustained decode too slow: {stress['min_toks']} tok/s < "
                f"{config.min_decode_toks} minimum. Skipping Phase 4."
            )
        else:
            phase3_passed = True

    # 9. Phase 4: Intelligence (requires Phase 3 pass)
    if config.run_phase4 and phase3_passed:
        from .phases import phase4_print_setup_guide, phase4_evalplus_quick

        logger.info("\n=== PHASE 4: INTELLIGENCE ===")

        logger.info("Quick EvalPlus (5 problems)...")
        evalplus = phase4_evalplus_quick(model, tokenizer, n_layers, _cache_factory)

        if evalplus["pass_rate"] < 0.6:
            logger.error(
                f"  Quick EvalPlus FAILED: {evalplus['pass_rate']*100:.0f}% < 60%. "
                f"Model quality too low for full benchmarks."
            )
        else:
            # Only print the full benchmark guide if everything passed
            phase4_print_setup_guide()

    # 10. Final verdict
    phases_run = ["Phase 1"]
    if config.run_phase2 and any_passed:
        phases_run.append(f"Phase 2 {'PASS' if phase2_passed else 'FAIL'}")
    if config.run_phase3 and phase2_passed:
        phases_run.append(f"Phase 3 {'PASS' if phase3_passed else 'FAIL'}")
    if config.run_phase4 and phase3_passed:
        phases_run.append("Phase 4")

    logger.info(f"\nPhases: {' → '.join(phases_run)}")
    if all_passed and phase2_passed and phase3_passed:
        logger.info("ALL PHASES PASSED")
    else:
        logger.info("SOME PHASES FAILED — see above for details")

    # 11. Cleanup
    del model
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    _release_lock()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Safe oMLX benchmark with memory watchdog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--contexts", nargs="+", type=int,
        default=[65536, 131072],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--cache-mode", default="turbo3",
        help="Cache mode (turbo3, turbo4, fp16)",
    )
    parser.add_argument(
        "--max-metal-gb", type=float, default=DEFAULT_MAX_METAL_GB,
        help=f"Max Metal active memory in GB (default: {DEFAULT_MAX_METAL_GB})",
    )
    parser.add_argument(
        "--max-swap-gb", type=float, default=DEFAULT_MAX_SWAP_GB,
        help=f"Max swap growth in GB (default: {DEFAULT_MAX_SWAP_GB})",
    )
    parser.add_argument(
        "--min-prefill-toks", type=float, default=DEFAULT_MIN_PREFILL_TOKS,
        help=f"Min prefill tok/s (default: {DEFAULT_MIN_PREFILL_TOKS})",
    )
    parser.add_argument(
        "--min-decode-toks", type=float, default=DEFAULT_MIN_DECODE_TOKS,
        help=f"Min decode tok/s (default: {DEFAULT_MIN_DECODE_TOKS})",
    )
    parser.add_argument(
        "--prefill-chunk", type=int, default=PREFILL_CHUNK_SIZE,
        help=f"Prefill chunk size (default: {PREFILL_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--dequant-chunk", type=int, default=16384,
        help="Dequant chunk size for streaming attention (default: 16384)",
    )
    parser.add_argument(
        "--no-coherence", action="store_true",
        help="Skip coherence test",
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=[1, 2, 3, 4],
        help="Which phases to run (default: 1 2 3 4)",
    )
    # Hypercar feature flags
    parser.add_argument("--no-fp16-layer0", action="store_true",
                        help="Disable fp16 layer 0 anchor (all layers TQ3)")
    parser.add_argument("--fp16-layers", type=int, default=16,
                        help="Number of leading fp16 layers (default: 16, was 1)")
    parser.add_argument("--no-vertical-eval", action="store_true",
                        help="Disable mx.eval per 8 layers (causes 45GB peaks)")
    parser.add_argument("--no-adaptive-budget", action="store_true",
                        help="Use fixed dequant_chunk_size (no adaptation)")
    parser.add_argument("--min-quant-tokens", type=int, default=512,
                        help="Stay fp16 below this threshold (default: 512)")
    parser.add_argument("--use-medusa", action="store_true",
                        help="Enable Medusa draft heads (requires --medusa-distill)")
    parser.add_argument("--medusa-heads", type=int, default=3,
                        help="Number of Medusa draft heads (default: 3)")
    parser.add_argument("--medusa-distill", type=int, default=0,
                        help="Medusa distillation steps (0=random/slow, 200+=trained)")
    parser.add_argument("--use-prompt-lookup", action="store_true",
                        help="Enable n-gram prompt lookup decoding")
    parser.add_argument(
        "--json", type=str, default=None,
        help="Write results to JSON file",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    config = BenchConfig(
        model_path=args.model,
        contexts=args.contexts,
        cache_mode=args.cache_mode,
        max_metal_gb=args.max_metal_gb,
        max_swap_gb=args.max_swap_gb,
        min_prefill_toks=args.min_prefill_toks,
        min_decode_toks=args.min_decode_toks,
        prefill_chunk=args.prefill_chunk,
        dequant_chunk_size=args.dequant_chunk,
        coherence_test=not args.no_coherence,
        run_phase2=2 in args.phases,
        run_phase3=3 in args.phases,
        run_phase4=4 in args.phases,
        use_fp16_layer0=not args.no_fp16_layer0,
        fp16_layers=args.fp16_layers,
        use_vertical_eval=not args.no_vertical_eval,
        use_adaptive_budget=not args.no_adaptive_budget,
        min_quant_tokens=args.min_quant_tokens,
        use_medusa=args.use_medusa,
        medusa_num_heads=args.medusa_heads,
        medusa_distill_steps=args.medusa_distill,
        use_prompt_lookup=args.use_prompt_lookup,
    )

    results = run_benchmark(config)

    # Write JSON if requested
    if args.json:
        out = []
        for r in results:
            out.append({
                "context_length": r.context_length,
                "status": r.status,
                "prefill_toks": round(r.prefill_toks, 1),
                "decode_toks": round(r.decode_toks, 1),
                "active_gb": round(r.active_gb, 2),
                "peak_gb": round(r.peak_gb, 2),
                "swap_gb": round(r.swap_gb, 2),
                "elapsed_s": round(r.elapsed_s, 1),
                "cpu_avg_pct": round(r.cpu_avg_pct, 1),
                "cpu_peak_pct": round(r.cpu_peak_pct, 1),
                "rss_peak_gb": round(r.rss_peak_gb, 2),
                "error_msg": r.error_msg,
            })
        Path(args.json).write_text(json.dumps(out, indent=2))
        logger.info(f"Results written to {args.json}")

    # Exit with error if any test failed
    if any(r.status != "pass" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
