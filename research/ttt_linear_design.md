# TTT-Linear design note (Task 388 Phase 0)

_Drafted 2026-05-02 alongside the scaffolding in `omlx/state_space/ttt_linear.py`._

## Why this exists

Goal 4 (≥500 tok/s prefill, **constant across context window**) is the open frontier on Qwen3.6 — Goal 1 (1M context memory) is solved by the hybrid SSM:attn layout + int4 KV. The remaining Goal 4 obstacle is the O(N²) softmax-attention cost in the 10 full-attention layers; even the lightest sparse-attention pattern keeps that asymptote.

DuoAttention's existing per-head classification (`omlx/duo_kv_cache.py::load_duo_policy()`) tags ~59% of heads as **streaming** — they attend mostly to a small local window plus sink tokens. Streaming heads are precisely the heads that don't *need* full softmax; their behavior is well-approximated by a recurrence with bounded state. TTT-Linear is one such recurrence whose hidden state is itself a learnable linear map, updated per token via an inner gradient descent.

**The structural claim**: replacing the 59% streaming heads with TTT-Linear blocks makes 59% of the per-token attention cost O(D²) (constant) instead of O(N×D) (growing). That's the move that *rotates* the prefill curve — every other Goal-4 lever (Quest, MInference, MC#) just lowers the constant in front of the N².

## Math (paper Section 3.1)

Per token at position `t`, with input `x_t ∈ R^D`:

    Q_t = W_Q x_t
    K_t = W_K x_t
    V_t = W_V x_t

The hidden state is a linear layer `W_t ∈ R^{D×D}`. The per-token update is one step of SGD on a reconstruction loss:

    l(W; x_t) = ‖ W K_t  −  V_t ‖²
    ∇_W l    = 2 (W K_t − V_t) K_t^T               # closed form
    W_t      = W_{t-1}  −  η · ∇_W l(W_{t-1}; x_t)
    o_t      = LayerNorm(W_t Q_t)                  # paper uses LN here

The recurrence is causal (W_t depends only on tokens ≤ t), and per-token compute is O(D²) — independent of `t`. That's the property we're after.

## Mini-batch TTT (paper Section 3.3)

The naive online update is one SGD step per token, requiring a Python loop. For prefill efficiency, the paper introduces *mini-batch TTT*: process `b` tokens against the same hidden state `W_{t-b}`, then take one gradient step using the batch-averaged loss:

    ∇_W L_b  =  (2 / b) · Σ_{i=t-b+1..t} (W_{t-b} K_i − V_i) K_i^T
    W_t      =  W_{t-b}  −  η · ∇_W L_b

This is two matmuls per mini-batch (one for `W K_i` accumulation, one for the gradient outer product) — comparable in cost to softmax attention's QK^T at chunk granularity. For Hypercar's 4096-token chunked prefill, `b = 4096` is the natural choice and lines up exactly with `PREFILL_CHUNK`.

The cost: outputs `o_i = W_t Q_i` for tokens within a mini-batch all use the *same* `W_t`, so they don't see the in-batch updates. Tradeoff: lose some sequential precision, gain GPU-friendly batched compute. Paper validates that quality holds at b=64 and beyond.

## API contract (matches the stub)

A TTT-Linear block is one (layer, head) instance. It exposes:

```python
class TTTLinear(nn.Module):
    def init_state(self, batch_size, dtype=mx.float32) -> mx.array:
        """Return W_0 of shape (B, D, D)."""

    def __call__(self, x: mx.array, state: mx.array | None = None
                 ) -> tuple[mx.array, mx.array]:
        """Process L tokens, return (outputs (B,L,D), new_state (B,D,D))."""
```

The state-threading API mirrors how MLX's `KVCache` is used in `mlx_lm` model forwards — a list of states is passed in, one per layer/head, and updated in place across chunks. The integration patch will create one TTTLinear instance per (layer, streaming-head-index) and route those heads' compute through it instead of `mx.fast.scaled_dot_product_attention`.

## Implementation strategy

**Phase 0 step 1 (this filing — done)**: scaffolding module + design note.

**Phase 0 step 2 (next cycle)**: implement the core forward pass.
- Online (mini_batch_size=1) version first — easy to verify correctness.
- Vectorized mini-batch (size=B) version next — the performance path.
- Numerical-correctness test: a hand-computed 3-token reference (D=4) where I work out W_1, W_2, W_3 by hand and compare against the implementation.

**Phase 0 step 3 (subsequent cycle)**: single-head distillation.
- Pick one (layer, head) tagged as streaming in the DuoAttention policy.
- Capture (input x_t, output o_t) pairs from the original attention head on a 1K-token calibration prompt.
- Initialize a TTTLinear with random `W_Q`/`W_K`/`W_V`, train via MSE/cos-sim loss for ~10K steps.
- Evaluate cos sim against held-out 1K tokens.

**Phase 0 gate**: cos sim ≥ 0.95 → proceed to Phase 1 (scale to all streaming heads). Cos sim < 0.95 → fall back to TTT-MLP (richer hidden state) or pivot to Mamba-2 SSM.

## Risks the scaffolding doesn't yet address

1. **Mini-batch update vectorization in MLX**: the inner gradient computation involves an outer product summed over the mini-batch, which doesn't have a one-line `mx.matmul` form. Likely needs `mx.einsum` or hand-rolled. Could be a Phase 0 step 2 stumbling block.
2. **Per-(layer, head) parameter explosion**: 10 attn layers × ~16 streaming heads × 3 projections × D² = 31 MB of new weights at D=256. Trivial. But if we go to TTT-MLP, the hidden state grows and so does the parameter footprint — flag for monitoring.
3. **LN placement**: paper has multiple variants (LN on Q, LN on output, both). Distillation may want all three searched.
4. **Distillation data**: 10K tokens of generic code+chat may not capture the streaming heads' specialized behavior. May need to source from RULER multi-key prompts (where streaming heads are most active) for better distillation signal.

## What's next-cycle-ready

Once `TTTLinear.__call__` is implemented (~2-3 days of focused work), the next dependency is the head-routing patch — replacing `self_attn`'s SDPA call with a per-head dispatcher that routes streaming heads to TTTLinear. That's a separate task (could be Task 389 when filed).

## References

- Sun, Y. et al. *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* arXiv:2407.04620 (2024).
- DuoAttention head classification: `omlx/duo_kv_cache.py::load_duo_policy()` and `scripts/duoattention_calibrate.py`.
- Existing Hypercar streaming-vs-retrieval split policy file: `omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json`.
