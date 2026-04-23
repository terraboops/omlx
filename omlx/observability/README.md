# `omlx.observability` — analyst instrumentation toolkit

Drop-in named timers, counters, and heap snapshots for forensic
performance work. Lives alongside the production code; stays silent
when disabled; costs microseconds when on.

This module is **analyst tooling** — not part of the hypercar serving
path. Import it from benchmarks, profiling scripts, or ad-hoc
investigation. Production code may also `from omlx.observability import
timer` without harm — unused timers are free, and the whole registry
can be disabled with `OMLX_OBSERVABILITY=0`.

## Quick start

### Timing a hot path

```python
from omlx.observability import timer, mlx_timer, registry

# CPU-only region — wall clock is honest
with timer("snapkv.freshness_score"):
    scores = compute_freshness_scores(cache)

# Region that dispatches MLX ops — force materialization so async
# GPU work is measured, not hidden
with mlx_timer("snapkv.rerope", new_k):
    new_k = rerope(k, positions)

# Flush at end of run
registry.print_summary()
registry.dump("/tmp/run.json")
```

### Counting events

```python
from omlx.observability import counter

counter("decode.tokens").add(1)
counter("snapkv.evicted").add(n_evicted)
counter("mx_eval.calls").add(1)
```

### Decorator form

```python
from omlx.observability.timers import timed

@timed()                         # name = "module.fn_qualname"
def score_tokens(cache): ...

@timed("snapkv.rerope", mlx=True)  # forces mx.synchronize on exit
def rerope(k, positions): ...
```

### Heap snapshot & diff (leak hunting)

```python
from omlx.observability.heap import snapshot, snapshot_diff

before = snapshot("before-prefill")
run_prefill(ctx_tokens=64_000)
after = snapshot("after-prefill")

print(snapshot_diff(before, after).format())
# heap diff [before-prefill → after-prefill] dt=23.45s
#   metal active: +38.404 GB (0.00 → 38.40)
#   metal peak  : +38.404 GB (0.00 → 38.40)
#   rss         : +1.020 GB
#   phys fp     : +39.512 GB
#   swap        : +4.900 GB
```

On Apple Silicon, **use `phys_footprint` as the truth** — RSS does not
include Metal-allocated unified memory.

### Leak detection loop

```bash
# 30-min fixed-prompt decode loop against your body
python -m omlx.observability.leak_loop \
    --target "my_bench:decode_body" \
    --iterations 1800 \
    --warmup 60 \
    --csv /tmp/leak.csv \
    --log-every 60
# → CLEAN: metal_peak=+0.042 MB/iter  phys_fp=+0.018 MB/iter
# or
# → LEAK:  metal_peak=+4.310 MB/iter  phys_fp=+4.305 MB/iter
```

The body callable receives the iteration index. Use `--gc-every N` or
`--clear-cache-every N` to attribute a climbing slope:

- Slope flattens under `--gc-every 10` → Python reference leak somewhere.
- Slope flattens under `--clear-cache-every 10` → MLX buffer-pool
  fragmentation, not a true leak.
- Slope survives both → genuine Metal leak.

## Design notes

- **Zero dependencies outside stdlib / mlx / psutil.**
- Timers retain up to `sample_cap=4096` samples via reservoir sampling
  (Algorithm R) so percentiles stay accurate without unbounded memory.
- The global `registry` is thread-safe (guarded by `threading.Lock`).
- `timer()` and `counter()` fast-path short-circuit when
  `OMLX_OBSERVABILITY=0` — safe to leave in hot code.
- `mlx_timer()` forces graph materialization on exit. Without this,
  async dispatch makes every MLX region look like microseconds.
- Heap snapshots capture Metal active/peak/cache, RSS, phys_footprint,
  swap used. Optional Python top-N via `tracemalloc` (caller must start
  tracemalloc — we don't, because it costs).

## Analyst playbook (60-second version)

1. **Baseline.** `mx.clear_cache(); snap0 = snapshot("t0")`.
2. **Sprinkle.** Wrap the top-20 functions from `cProfile -s cumulative`
   in `mlx_timer(...)`. Wrap per-iter loops with `counter(...).add(1)`.
3. **Run.** Execute the benchmark or reproduction.
4. **Dump.** `registry.dump("run.json")` and `print(snapshot_diff(snap0).format())`.
5. **Diff.** Compare `run.json` across commits. Any timer that grew
   >20% between commits is a regression or a feature — the analyst
   decides which.
6. **Leak-hunt.** `leak_loop --iterations 300 --gc-every 30` on
   anything that might retain state across requests.
