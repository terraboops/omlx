# SPDX-License-Identifier: Apache-2.0
"""Named counters / gauges.

Thin wrapper over `registry.counter()` for a fluent hot-path API:

    counter("decode.tokens").add(1)
    counter("snapkv.evictions").add(n)
    counter("kv.active_bytes").set(mx.get_active_memory())

Counters are monotonic-by-convention (add) with optional gauge semantics
(set). For histograms, use a named timer — buckets are pre-computed by
percentile at flush time.
"""

from __future__ import annotations

from omlx.observability.registry import CounterStats, registry


class _Noop:
    __slots__ = ()

    def add(self, delta: float = 1.0) -> None:
        pass

    def set(self, v: float) -> None:
        pass


_NOOP = _Noop()


def counter(name: str) -> CounterStats:
    """Get the named counter. Returns a no-op proxy if registry is disabled."""
    if not registry.enabled:
        return _NOOP  # type: ignore[return-value]
    return registry.counter(name)
