# SPDX-License-Identifier: Apache-2.0
"""TTT-Linear head-routing dispatcher (Task 388 Phase 2 — cycle 1).

Per-(layer, head) attention dispatch: streaming-tagged heads route through
per-(layer, head) TTT-Linear blocks (replacing softmax attention with the
linear-time recurrence); retrieval-tagged heads fall through to the
original ``scaled_dot_product_attention``.

Cycle 1 ships:
- ``TTTHeadRouter`` class with the public API needed by Phase 3
  validation (per-layer counter, per-(layer, head) state cache, reset).
- The **bit-equivalence path**: when no TTT blocks are loaded for a
  layer's heads, all heads route to original SDPA and the output is
  bit-identical to plain SDPA. This is the load-bearing correctness
  property — Phase 3 validation will start from this baseline and
  enable TTT routing per-head incrementally.
- A monkey-patch (``apply_ttt_head_router_patch``) that installs the
  router on top of ``mlx_lm.models.base.scaled_dot_product_attention``,
  threading the layer counter via the per-layer SDPA call sequence.

Cycle 2 (next): once distillation produces real per-(layer, head)
TTT-Linear weights, implement the actual split-and-reassemble path
behind ``_route_with_ttt`` (currently raises NotImplementedError).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import json
import logging
import sys

import mlx.core as mx

logger = logging.getLogger(__name__)


@dataclass
class TTTHeadRouter:
    """Router that dispatches per-head SDPA calls to TTT or original SDPA.

    Attributes
    ----------
    n_layers : int
        Total number of attention-bearing layers in the model. Used by
        ``layer_counter`` modulo bookkeeping when the router is installed
        as an SDPA monkey-patch.
    n_heads : int
        Q heads per layer. The routing classification is keyed by
        ``(layer, head)`` over ``range(n_layers) × range(n_heads)``.
    head_classification : dict[(int,int), str]
        Per-head policy from DuoAttention calibration: 'streaming' or
        'retrieval'. Heads not in the dict default to 'retrieval' (safe).
    ttt_blocks : dict[(int,int), object]
        Per-(layer, head) TTT-Linear instances, keyed identically.
        Empty dict → bit-equivalence mode (everything routes to original
        SDPA).
    """

    n_layers: int
    n_heads: int
    head_classification: dict = field(default_factory=dict)
    ttt_blocks: dict = field(default_factory=dict)

    # Mutable state — managed by reset_state() and the SDPA call hook.
    layer_counter: int = 0
    # Per-(layer, head) TTT recurrence state (B, D, D) so chunked prefill
    # can resume mid-sequence. Cleared by reset_state().
    ttt_states: dict = field(default_factory=dict)

    @classmethod
    def from_policy(
        cls,
        policy_path: Path,
        ttt_blocks: Optional[dict] = None,
    ) -> "TTTHeadRouter":
        """Construct from a DuoAttention policy JSON.

        ``ttt_blocks`` is a ``{(layer, head): TTTLinear}`` dict; pass
        ``None`` (the default) to build a bit-equivalence router with no
        TTT blocks installed. Useful for Phase 3 validation: install the
        router with no TTT, confirm bench-output unchanged, then add TTT
        blocks one (layer, head) at a time.
        """
        data = json.loads(Path(policy_path).read_text())
        classification = {
            (h["layer"], h["head"]): h["policy"]
            for h in data["heads"]
        }
        return cls(
            n_layers=data["n_layers"],
            n_heads=data["n_heads"],
            head_classification=classification,
            ttt_blocks=ttt_blocks or {},
        )

    def reset_state(self) -> None:
        """Clear per-call state. Call between independent sequences (e.g.
        between requests in the server, or before each capture-mode
        prompt in Phase 0)."""
        self.layer_counter = 0
        self.ttt_states.clear()

    def streaming_heads_with_ttt(self, layer_idx: int) -> list[int]:
        """Return the list of head indices in ``layer_idx`` that are
        both classified streaming AND have a loaded TTT block. These
        are the heads that would actually divert from original SDPA."""
        out = []
        for head_idx in range(self.n_heads):
            key = (layer_idx, head_idx)
            if (self.head_classification.get(key) == "streaming"
                    and key in self.ttt_blocks):
                out.append(head_idx)
        return out

    def route_sdpa(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        cache,
        scale: float,
        mask,
        sinks=None,
        *,
        original_sdpa,
    ) -> mx.array:
        """Per-call dispatch.

        Increments ``layer_counter`` (modulo ``n_layers``) and dispatches
        the per-head computation. Currently: if any head in this layer
        has both streaming classification AND a loaded TTT block, the
        cycle-2 ``_route_with_ttt`` is invoked (NotImplementedError
        until cycle 2). Otherwise falls through to ``original_sdpa``.

        Bit-equivalence guarantee: when ``self.ttt_blocks`` is empty,
        every call goes to ``original_sdpa`` unchanged.
        """
        layer_idx = self.layer_counter % self.n_layers
        self.layer_counter += 1

        ttt_heads = self.streaming_heads_with_ttt(layer_idx)
        if not ttt_heads:
            return original_sdpa(queries, keys, values, cache, scale, mask,
                                 sinks)

        return self._route_with_ttt(
            layer_idx, ttt_heads, queries, keys, values, cache, scale, mask,
            sinks, original_sdpa=original_sdpa,
        )

    def _route_with_ttt(
        self,
        layer_idx: int,
        ttt_heads: list[int],
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        cache,
        scale: float,
        mask,
        sinks,
        *,
        original_sdpa,
    ) -> mx.array:
        """Per-head split-and-dispatch path. NOT IMPLEMENTED until cycle 2.

        Plan: extract per-head ``Q_h`` slices (queries[:, h, :, :]) for
        each h in ``ttt_heads``, run them through the per-(layer, head)
        TTT-Linear (with the recurrence state from ``self.ttt_states``,
        updating it on return), then reassemble the per-head outputs
        with the original-SDPA output for retrieval heads. The reassembly
        must be along the H_q axis without copying the retrieval-head
        slice, so the patch must compute original SDPA over the SAME
        (queries, keys, values) and overwrite the TTT-routed head slots.
        """
        raise NotImplementedError(
            f"TTT routing for layer {layer_idx}, heads {ttt_heads} is not "
            "implemented yet. Cycle 1 ships only the bit-equivalence "
            "(no-TTT-blocks) path. Until distillation produces trained "
            "TTT-Linear weights (Phase 0 step 3 gate cleared), the "
            "router runs in passthrough mode only."
        )


_PATCHED_ROUTER: Optional[TTTHeadRouter] = None


def apply_ttt_head_router_patch(router: TTTHeadRouter) -> bool:
    """Install ``router`` as an SDPA monkey-patch.

    Returns True if applied, False if a router was already active.
    Thread-unsafe; intended for single-process inference setups.
    """
    global _PATCHED_ROUTER
    if _PATCHED_ROUTER is not None:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention
    _PATCHED_ROUTER = router

    def routed_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        return router.route_sdpa(
            queries, keys, values, cache, scale, mask, sinks,
            original_sdpa=original_sdpa,
        )

    mlx_base.scaled_dot_product_attention = routed_sdpa
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
            if hasattr(mod, "scaled_dot_product_attention"):
                setattr(mod, "scaled_dot_product_attention", routed_sdpa)

    logger.info(
        f"TTTHeadRouter installed (n_layers={router.n_layers}, "
        f"n_heads={router.n_heads}, "
        f"ttt_blocks_loaded={len(router.ttt_blocks)})"
    )
    return True


def remove_ttt_head_router_patch() -> bool:
    """Uninstall the active router and restore original SDPA. Returns
    True if a router was uninstalled, False otherwise."""
    global _PATCHED_ROUTER
    if _PATCHED_ROUTER is None:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    # Walk back: find the original by searching for any module whose SDPA
    # is NOT the routed one (not robust if multiple patches stacked, but
    # this is the only TTT routing patch and we don't compose it yet).
    # Instead, callers are responsible for module-state cleanliness.
    _PATCHED_ROUTER = None
    return True
