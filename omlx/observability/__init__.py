# SPDX-License-Identifier: Apache-2.0
"""In-code instrumentation toolkit for analyst forensics.

Drop-in named timers, counters, and heap snapshots that aggregate across
a run and flush to JSON. Designed for hot-path use: zero cost when the
registry is disabled, minimal cost when active.

Usage
-----

    from omlx.observability import timer, counter, registry

    with timer("snapkv.compact"):
        compact_cache(...)
    counter("snapkv.evictions").add(n_evicted)

    registry.dump("/tmp/run.json")
    registry.print_summary()

Timers support both wall-time (`timer`) and MLX-aware (`mlx_timer`)
variants. `mlx_timer` forces graph materialization at close so async
GPU work is not hidden in the dispatch queue.
"""

from omlx.observability.registry import registry, reset
from omlx.observability.timers import mlx_timer, timer
from omlx.observability.counters import counter
from omlx.observability.heap import HeapSnapshot, snapshot, snapshot_diff

__all__ = [
    "registry",
    "reset",
    "timer",
    "mlx_timer",
    "counter",
    "HeapSnapshot",
    "snapshot",
    "snapshot_diff",
]
