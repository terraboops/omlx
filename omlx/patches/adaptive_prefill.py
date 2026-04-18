# SPDX-License-Identifier: Apache-2.0
"""Adaptive prefill chunk-size controller with memory-aware feedback.

Replaces the fixed prefill_step_size with a proportional controller
that adjusts chunk size based on Metal memory pressure and throughput.

At the start of prefill (KV cache small), uses large chunks for maximum
throughput. As the KV cache grows and Metal pressure builds, shrinks
chunks to avoid swap thrashing. If tok/s drops below a floor (O(n^2)
attention cliff), shrinks aggressively.

Derived from Memory-aware Dynamic Batching (arXiv:2503.05248).

Usage:
    from omlx.patches.adaptive_prefill import apply_adaptive_prefill
    apply_adaptive_prefill()  # monkey-patches generate_step
"""

from __future__ import annotations

import logging
import time

import mlx.core as mx

logger = logging.getLogger("hypercar.adaptive_prefill")

# System memory detection (same approach as baseline.py)
try:
    import ctypes
    import ctypes.util
    _libc = ctypes.CDLL(ctypes.util.find_library("c"))
    _buf = ctypes.c_uint64(0)
    _size = ctypes.c_size_t(8)
    _libc.sysctlbyname(b"hw.memsize", ctypes.byref(_buf),
                        ctypes.byref(_size), None, 0)
    SYSTEM_MEMORY_GB = _buf.value / 1e9
except Exception:
    SYSTEM_MEMORY_GB = 48.0  # fallback for M4 Pro


class AdaptivePrefillController:
    """Proportional controller for prefill chunk sizing.

    Two signals, one output:
      - Metal memory residency → shrink when above target
      - Prefill throughput → shrink when below floor (O(n²) cliff)
      - Output: next chunk size (clamped to [min_chunk, max_chunk])

    The controller starts at max_chunk and adjusts per iteration.
    """

    def __init__(
        self,
        target_metal_pct: float = 0.65,
        min_chunk: int = 512,
        max_chunk: int = 16384,
        throughput_floor: float = 200.0,
        system_memory_gb: float = SYSTEM_MEMORY_GB,
    ):
        self.target_metal_gb = system_memory_gb * target_metal_pct
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.throughput_floor = throughput_floor
        self.chunk_size = max_chunk
        self.history: list[dict] = []

    def next_chunk_size(self, remaining: int) -> int:
        """Return the chunk size for the next prefill iteration."""
        return min(self.chunk_size, remaining)

    def feedback(self, metal_gb: float, tok_per_sec: float) -> None:
        """Update controller state after a chunk completes."""
        metal_ratio = metal_gb / max(self.target_metal_gb, 0.1)
        prev = self.chunk_size

        if tok_per_sec < self.throughput_floor and tok_per_sec > 0:
            # Throughput falling — shrink aggressively (O(n²) cliff)
            self.chunk_size = max(self.min_chunk, self.chunk_size // 2)
        elif metal_ratio > 1.0:
            # Over memory target — shrink proportionally
            factor = max(0.5, 1.0 / metal_ratio)
            self.chunk_size = max(self.min_chunk, int(self.chunk_size * factor))
        elif metal_ratio < 0.8:
            # Under target with headroom — grow gradually
            self.chunk_size = min(self.max_chunk, int(self.chunk_size * 1.25))
        # else: in the sweet spot [0.8, 1.0] of target — hold steady

        self.history.append({
            "chunk_size": self.chunk_size,
            "prev_chunk": prev,
            "metal_gb": round(metal_gb, 2),
            "tok_per_sec": round(tok_per_sec, 1),
            "metal_ratio": round(metal_ratio, 3),
        })

    def summary(self) -> dict:
        """Return controller summary for profiling."""
        if not self.history:
            return {"adaptive_chunks": 0}
        sizes = [h["chunk_size"] for h in self.history]
        return {
            "adaptive_chunks": len(self.history),
            "chunk_min": min(sizes),
            "chunk_max": max(sizes),
            "chunk_final": sizes[-1],
            "metal_peak_gb": max(h["metal_gb"] for h in self.history),
            "throughput_min": min(h["tok_per_sec"] for h in self.history),
        }


def apply_adaptive_prefill(
    target_metal_pct: float = 0.65,
    min_chunk: int = 512,
    max_chunk: int = 16384,
    throughput_floor: float = 200.0,
    min_prompt_tokens: int = 1024,
) -> None:
    """Monkey-patch generate_step to use adaptive prefill chunking.

    For prompts >= min_prompt_tokens, runs a custom prefill loop with
    memory-aware chunk sizing before handing off to the original
    generate_step for decode.

    Short prompts (< min_prompt_tokens) pass through unchanged.

    Args:
        target_metal_pct: Target Metal usage as fraction of system memory.
        min_chunk: Minimum chunk size (floor).
        max_chunk: Maximum chunk size (ceiling / starting point).
        throughput_floor: If tok/s drops below this, shrink chunks.
        min_prompt_tokens: Minimum prompt length to activate adaptive mode.
    """
    import importlib
    gen_mod = importlib.import_module("mlx_lm.generate")
    import mlx_lm.models.cache as cache_mod

    _original_generate_step = gen_mod.generate_step

    def adaptive_generate_step(prompt, model, **kwargs):
        prompt_len = prompt.shape[0] if hasattr(prompt, 'shape') else len(prompt)

        # Short prompts: pass through (no benefit from adaptive chunking)
        if prompt_len < min_prompt_tokens:
            yield from _original_generate_step(prompt, model, **kwargs)
            return

        # Create cache ourselves for adaptive prefill
        prompt_cache = kwargs.pop('prompt_cache', None)
        if prompt_cache is None:
            prompt_cache = cache_mod.make_prompt_cache(
                model, max_kv_size=kwargs.get('max_kv_size'))

        controller = AdaptivePrefillController(
            target_metal_pct=target_metal_pct,
            min_chunk=min_chunk,
            max_chunk=max_chunk,
            throughput_floor=throughput_floor,
        )

        # Adaptive prefill: process all tokens except the last one
        # (the last token goes to generate_step for sampling)
        if not isinstance(prompt, mx.array):
            prompt = mx.array(prompt)

        tokens_to_prefill = prompt_len - 1
        processed = 0

        with mx.stream(mx.cpu):
            pass  # ensure we're on default stream

        while processed < tokens_to_prefill:
            remaining = tokens_to_prefill - processed
            chunk_size = controller.next_chunk_size(remaining)

            chunk = prompt[processed:processed + chunk_size]
            t0 = time.perf_counter()

            logits = model(chunk[None], cache=prompt_cache)
            mx.eval([c.state for c in prompt_cache])

            elapsed = time.perf_counter() - t0
            tok_per_sec = chunk_size / max(elapsed, 1e-6)
            metal_gb = mx.get_active_memory() / 1e9

            controller.feedback(metal_gb, tok_per_sec)
            processed += chunk_size
            mx.clear_cache()

        # Log adaptive summary
        s = controller.summary()
        logger.info(
            f"Adaptive prefill: {prompt_len} tokens, "
            f"{s['adaptive_chunks']} chunks "
            f"({s.get('chunk_min', 0)}-{s.get('chunk_max', 0)} range), "
            f"Metal peak {s.get('metal_peak_gb', 0):.1f} GB"
        )

        # Hand off to original generate_step with pre-filled cache
        # Only the last token remains — generate_step skips its prefill loop
        kwargs['prompt_cache'] = prompt_cache
        yield from _original_generate_step(prompt[-1:], model, **kwargs)

    gen_mod.generate_step = adaptive_generate_step

    # Also patch server's reference if imported
    try:
        import mlx_lm.server as srv
        srv.generate_step = adaptive_generate_step
    except ImportError:
        pass

    logger.info(
        f"Adaptive prefill enabled: target={target_metal_pct:.0%} Metal, "
        f"chunks {min_chunk}-{max_chunk}, floor={throughput_floor} tok/s"
    )
