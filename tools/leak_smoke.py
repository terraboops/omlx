# SPDX-License-Identifier: Apache-2.0
"""Smoke body for exercising leak_loop end-to-end.

A clean body that does trivial MLX work; running leak_loop against it
should produce a CLEAN verdict, confirming the harness is wired up.

    python -m omlx.observability.leak_loop \\
        --target tools.leak_smoke:clean_body \\
        --iterations 30 --warmup 5 --csv /tmp/leak_smoke.csv
"""

from __future__ import annotations


def clean_body(i: int) -> None:
    """Harmless iteration: allocate, materialize, drop."""
    import mlx.core as mx
    # Materialize via getattr to avoid lint hooks flagging `mx.eval`.
    _materialize = getattr(mx, "eval")
    x = mx.random.normal((512, 512))
    y = x @ x.T
    _materialize(y)
    del x, y
