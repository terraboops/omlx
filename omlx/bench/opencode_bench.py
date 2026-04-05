# SPDX-License-Identifier: Apache-2.0
"""OpenCode benchmark — realistic coding workload, long context.

Uses the hypercar config that previously achieved 256K context:
  - Streaming TQ3 KV (minimal fp16 layers)
  - Vertical eval every 8 layers
  - Adaptive memory budget
  - Fused Givens kernel
  - last-logit patch

Tests the scenarios that matter for an OpenCode/Copilot use case:
  1. Paste realistic code file (~Python/TS from disk)
  2. Ask a coding question about it
  3. Measure: time-to-first-token, decode tok/s, answer quality

Usage:
    # Quick test at 8K-65K
    python -m omlx.bench.opencode_bench --contexts 8192 32768 65536

    # Full stress test to 256K
    python -m omlx.bench.opencode_bench --contexts 65536 131072 196608 262144
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx

logger = logging.getLogger("opencode")


# ---------------------------------------------------------------------------
# Realistic code corpus — used as haystack
# ---------------------------------------------------------------------------

# Real Python code samples that exercise diverse token distributions.
# These are designed to mimic what a user would paste: mixed libraries,
# comments, docstrings, data structures, algorithms.
CODE_SAMPLES = [
    # FastAPI-style API code
    '''from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="Task API", version="1.0.0")


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    priority: int = Field(default=0, ge=0, le=10)
    tags: List[str] = Field(default_factory=list)


class Task(TaskCreate):
    id: int
    created_at: datetime
    updated_at: datetime
    completed: bool = False


tasks_db: dict[int, Task] = {}
next_id = 1


@app.post("/tasks", response_model=Task, status_code=201)
async def create_task(task: TaskCreate):
    global next_id
    now = datetime.utcnow()
    new_task = Task(
        id=next_id,
        created_at=now,
        updated_at=now,
        **task.model_dump()
    )
    tasks_db[next_id] = new_task
    next_id += 1
    logger.info(f"Created task {new_task.id}: {new_task.title}")
    return new_task


@app.get("/tasks", response_model=List[Task])
async def list_tasks(
    completed: Optional[bool] = None,
    priority_min: int = 0,
    limit: int = 100
):
    results = list(tasks_db.values())
    if completed is not None:
        results = [t for t in results if t.completed == completed]
    if priority_min > 0:
        results = [t for t in results if t.priority >= priority_min]
    return results[:limit]
''',

    # Data processing pipeline
    '''import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Any, Iterator


class DataPipeline:
    """Composable data transformation pipeline with metrics."""

    def __init__(self, name: str):
        self.name = name
        self.stages: list[tuple[str, Callable]] = []
        self.metrics: Dict[str, Any] = {"rows_processed": 0, "errors": 0}

    def add(self, name: str, fn: Callable) -> "DataPipeline":
        self.stages.append((name, fn))
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for stage_name, fn in self.stages:
            try:
                before = len(result)
                result = fn(result)
                after = len(result)
                self.metrics[f"{stage_name}_dropped"] = before - after
            except Exception as e:
                self.metrics["errors"] += 1
                raise RuntimeError(f"Pipeline {self.name} failed at {stage_name}: {e}")
        self.metrics["rows_processed"] = len(result)
        return result


def deduplicate(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep="first")


def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def normalize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / (std + 1e-10)
    return df
''',

    # Algorithms
    '''from collections import deque, defaultdict
from heapq import heappush, heappop
from typing import TypeVar, Generic, Callable, Optional, Any


T = TypeVar("T")


class Graph(Generic[T]):
    """Directed weighted graph with common traversal algorithms."""

    def __init__(self):
        self.adjacency: dict[T, list[tuple[T, float]]] = defaultdict(list)
        self.nodes: set[T] = set()

    def add_edge(self, src: T, dst: T, weight: float = 1.0) -> None:
        self.adjacency[src].append((dst, weight))
        self.nodes.add(src)
        self.nodes.add(dst)

    def bfs(self, start: T) -> list[T]:
        visited = {start}
        order = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor, _ in self.adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return order

    def dfs(self, start: T) -> list[T]:
        visited = set()
        order = []
        def _dfs(node: T):
            if node in visited:
                return
            visited.add(node)
            order.append(node)
            for neighbor, _ in self.adjacency[node]:
                _dfs(neighbor)
        _dfs(start)
        return order

    def dijkstra(self, start: T, end: T) -> tuple[float, list[T]]:
        distances = {node: float("inf") for node in self.nodes}
        distances[start] = 0
        previous: dict[T, Optional[T]] = {node: None for node in self.nodes}
        pq = [(0.0, start)]

        while pq:
            dist, current = heappop(pq)
            if current == end:
                break
            if dist > distances[current]:
                continue
            for neighbor, weight in self.adjacency[current]:
                new_dist = dist + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heappush(pq, (new_dist, neighbor))

        if distances[end] == float("inf"):
            return float("inf"), []

        path = []
        node: Optional[T] = end
        while node is not None:
            path.append(node)
            node = previous[node]
        path.reverse()
        return distances[end], path
''',
]


@dataclass
class OpenCodeResult:
    context_length: int
    prefill_tokens: int
    prefill_seconds: float
    prefill_toks_per_sec: float
    ttft_seconds: float  # Time to first decoded token
    decode_tokens: int
    decode_seconds: float
    decode_toks_per_sec: float
    active_gb: float
    peak_gb: float
    response: str
    status: str = "ok"  # ok, oom, timeout


def build_code_context(tokenizer, target_tokens: int) -> str:
    """Build a realistic code context of approximately target_tokens size.

    Repeats CODE_SAMPLES until target reached. Adds a final question.
    """
    parts = []
    tokens_so_far = 0
    i = 0
    while tokens_so_far < target_tokens:
        sample = CODE_SAMPLES[i % len(CODE_SAMPLES)]
        header = f"\n# === File {i + 1}: sample_module_{i + 1}.py ===\n\n"
        parts.append(header + sample)
        tokens_so_far += len(tokenizer.encode(header + sample))
        i += 1
    return "".join(parts)


QUESTIONS = [
    "\n# Question: Looking at the code above, what endpoints does the Task API expose? List them briefly:",
    "\n# Question: Does the DataPipeline class support error recovery or does it re-raise on failure?",
    "\n# Question: In the Graph class, does dijkstra() return just the distance or also the path?",
]


def run_opencode_benchmark(
    model, tokenizer, cache_factory, n_layers: int,
    context_length: int, decode_tokens: int = 50,
    prefill_chunk: int = 2048,
) -> OpenCodeResult:
    """Run one OpenCode benchmark: load codebase, ask question, decode."""
    from omlx.memory_budget import compute_budget, compute_dequant_chunk_size
    from omlx.streaming_attention import set_chunk_hint

    # Build prompt: codebase + question
    target = context_length - 200  # Leave room for question
    codebase = build_code_context(tokenizer, target)
    question = QUESTIONS[0]  # Ask about the API endpoints
    prompt = codebase + question + "\n# Answer:\n"

    tokens = tokenizer.encode(prompt)
    actual_len = len(tokens)
    logger.info(f"  Prompt: {actual_len:,} tokens (target {context_length:,})")

    # Create cache
    mx.synchronize()
    mx.clear_cache()
    cache = cache_factory(n_layers)

    # PREFILL
    prefill_t0 = time.perf_counter()
    for chunk_start in range(0, actual_len, prefill_chunk):
        chunk_end = min(chunk_start + prefill_chunk, actual_len)
        budget = compute_budget(model_gb=17.2, active_kv_gb=0, decoding=False)
        chunk = compute_dequant_chunk_size(
            query_len=chunk_end - chunk_start,
            num_query_heads=32, head_dim=128, num_layers=48,
            budget=budget,
        )
        set_chunk_hint(chunk)
        logits = model(mx.array([tokens[chunk_start:chunk_end]]), cache=cache)
        mx.eval(logits)
        mx.synchronize()
        mx.clear_cache()
    prefill_sec = time.perf_counter() - prefill_t0

    active_gb = mx.get_active_memory() / 1e9
    peak_gb = mx.get_peak_memory() / 1e9

    # DECODE — measure time-to-first-token separately
    ttft_t0 = time.perf_counter()
    first_token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(first_token)
    ttft = time.perf_counter() - ttft_t0
    generated = [first_token.item()]

    decode_t0 = time.perf_counter()
    for _ in range(decode_tokens - 1):
        x = mx.array([[generated[-1]]])
        logits = model(x, cache=cache)
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        generated.append(tok.item())
    decode_sec = time.perf_counter() - decode_t0

    response = tokenizer.decode(generated).strip()

    result = OpenCodeResult(
        context_length=actual_len,
        prefill_tokens=actual_len,
        prefill_seconds=prefill_sec,
        prefill_toks_per_sec=actual_len / prefill_sec if prefill_sec > 0 else 0,
        ttft_seconds=ttft,
        decode_tokens=len(generated),
        decode_seconds=decode_sec,
        decode_toks_per_sec=(len(generated) - 1) / decode_sec if decode_sec > 0 else 0,
        active_gb=active_gb,
        peak_gb=peak_gb,
        response=response[:200],
    )

    del cache, logits
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="OpenCode realistic benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")
    parser.add_argument("--contexts", nargs="+", type=int,
                        default=[8192, 32768, 65536, 131072])
    parser.add_argument("--fp16-layers", type=int, default=1,
                        help="fp16 layers (default 1 = full hypercar)")
    parser.add_argument("--prefill-chunk", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=50)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model with hypercar patches
    from mlx_lm import load
    from mlx_lm.models.cache import KVCache
    from omlx.turboquant_kv import TurboQuantKVCache
    from omlx.patches.turboquant_attention import apply_turboquant_attention_patch
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    from omlx.patches.vertical_eval import apply_vertical_eval_patch

    logger.info(f"Loading {args.model}...")
    apply_turboquant_attention_patch()
    model, tokenizer = load(args.model)
    apply_prefill_last_logit_patch(model)
    apply_vertical_eval_patch(model)
    n_layers = model.args.num_hidden_layers
    logger.info(f"Model loaded: {mx.get_active_memory()/1e9:.1f}GB ({n_layers} layers)")
    logger.info(f"Hypercar: fp16_layers={args.fp16_layers}, prefill_chunk={args.prefill_chunk}")

    def cache_factory(n):
        return [
            KVCache() if i < args.fp16_layers
            else TurboQuantKVCache(bits=3, dequant_chunk_size=2048, min_quant_tokens=512)
            for i in range(n)
        ]

    # Run benchmarks
    results = []
    for ctx in args.contexts:
        logger.info(f"\n=== Context: {ctx:,} tokens ===")
        try:
            result = run_opencode_benchmark(
                model, tokenizer, cache_factory, n_layers,
                ctx, decode_tokens=args.decode_tokens,
                prefill_chunk=args.prefill_chunk,
            )
            results.append(result)
            logger.info(
                f"  prefill: {result.prefill_toks_per_sec:>5.0f} tok/s | "
                f"ttft: {result.ttft_seconds*1000:>4.0f}ms | "
                f"decode: {result.decode_toks_per_sec:>4.1f} tok/s"
            )
            logger.info(
                f"  memory: active {result.active_gb:>4.1f}GB | peak {result.peak_gb:>4.1f}GB"
            )
            logger.info(f"  response: {result.response[:120]!r}")
        except Exception as e:
            logger.error(f"FAILED at {ctx:,}: {type(e).__name__}: {str(e)[:120]}")
            results.append(OpenCodeResult(
                context_length=ctx, prefill_tokens=0, prefill_seconds=0,
                prefill_toks_per_sec=0, ttft_seconds=0, decode_tokens=0,
                decode_seconds=0, decode_toks_per_sec=0,
                active_gb=0, peak_gb=0, response="", status="error",
            ))
            break

    # Summary table
    logger.info("\n=== OpenCode Benchmark Results ===")
    logger.info(f"{'Context':>10} | {'Prefill':>9} | {'TTFT':>7} | {'Decode':>9} | {'Peak':>6}")
    logger.info("-" * 60)
    for r in results:
        if r.status == "ok":
            logger.info(
                f"{r.context_length:>10,} | "
                f"{r.prefill_toks_per_sec:>7.0f}t/s | "
                f"{r.ttft_seconds*1000:>5.0f}ms | "
                f"{r.decode_toks_per_sec:>7.1f}t/s | "
                f"{r.peak_gb:>5.1f}G"
            )

    if args.json:
        out = [{
            "context_length": r.context_length,
            "prefill_toks_per_sec": round(r.prefill_toks_per_sec, 1),
            "ttft_seconds": round(r.ttft_seconds, 3),
            "decode_toks_per_sec": round(r.decode_toks_per_sec, 2),
            "active_gb": round(r.active_gb, 2),
            "peak_gb": round(r.peak_gb, 2),
            "status": r.status,
            "response_preview": r.response,
        } for r in results]
        Path(args.json).write_text(json.dumps(out, indent=2))
        logger.info(f"Results → {args.json}")


if __name__ == "__main__":
    main()
