#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TTT-Linear single-head distillation prototype (Task 388 Phase 0 step 3).

Picks one streaming-tagged head from the DuoAttention calibration policy,
captures per-head input ``Q_h`` and per-head SDPA output ``o_h`` on a
calibration prompt, and trains a TTT-Linear block to regress ``o_h`` from
``Q_h``.

Phase 0 spike gate: cosine similarity ≥ 0.95 on held-out tokens. Pass →
proceed to Phase 1 (scale to all streaming heads). Fail → falsification
note pointing at TTT-MLP / Mamba / RWKV alternatives.

Usage:
    # Synthetic-mode smoke test (no model load, ~10s):
    .venv/bin/python scripts/ttt_distill_single_head.py --synthetic

    # Real distillation (~5 min: model load + capture + train):
    .venv/bin/python scripts/ttt_distill_single_head.py --capture \
        --policy omlx/patches/duoattention_policies/qwen3_coder_30b_a3b_instruct_8bit.json \
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit \
        --context-len 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as mxopt
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, ".")
from omlx.state_space import TTTLinear, TTTLinearConfig, attention_layer_indices

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ttt_distill")


# ---------------------------------------------------------------------------
# Policy + head selection
# ---------------------------------------------------------------------------


@dataclass
class HeadSelection:
    layer: int
    head: int
    local_fraction: float
    n_layers: int
    n_heads: int


def pick_streaming_head(policy_path: Path) -> HeadSelection:
    """Choose the streaming head with the highest local_fraction.

    Picking the highest local_fraction maximizes the chance of distillation
    success on the prototype: that head is the most cleanly local in its
    attention pattern, so a recurrence-based replacement should fit best.
    """
    data = json.loads(Path(policy_path).read_text())
    streaming = [
        h for h in data["heads"] if h.get("policy") == "streaming"
    ]
    if not streaming:
        raise ValueError(
            f"No streaming heads in {policy_path} — calibration "
            "produced no candidates for distillation."
        )
    best = max(streaming, key=lambda h: h["local_fraction"])
    return HeadSelection(
        layer=best["layer"],
        head=best["head"],
        local_fraction=best["local_fraction"],
        n_layers=data["n_layers"],
        n_heads=data["n_heads"],
    )


# ---------------------------------------------------------------------------
# Loss + metric functions
# ---------------------------------------------------------------------------


def mse_loss(pred: mx.array, target: mx.array) -> mx.array:
    return mx.mean((pred - target) ** 2)


def cos_sim_per_token(pred: mx.array, target: mx.array) -> mx.array:
    """Cosine similarity along the last dim, returns per-(B,L) sims."""
    eps = 1e-8
    p_norm = mx.sqrt(mx.sum(pred * pred, axis=-1) + eps)
    t_norm = mx.sqrt(mx.sum(target * target, axis=-1) + eps)
    dot = mx.sum(pred * target, axis=-1)
    return dot / (p_norm * t_norm)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_ttt(
    ttt: TTTLinear,
    x_train: mx.array,
    o_train: mx.array,
    x_val: mx.array,
    o_val: mx.array,
    *,
    n_steps: int = 2000,
    lr: float = 1e-3,
    log_every: int = 200,
    early_stop: bool = True,
) -> dict:
    """Adam-train a TTT-Linear block to regress ``o`` from ``x``.

    Inputs ``x_train`` / ``o_train`` / ``x_val`` / ``o_val`` may be
    multi-sequence batches: shape ``(N, L, D)`` where N=1 collapses to
    the single-sequence case. The TTT forward already handles per-batch
    state.

    With ``early_stop=True`` (the default), tracks the best val cos-sim
    across eval points and restores those parameters at the end of
    training. Mitigates the single-sequence overfitting found in the
    Phase 0 step 3 prototype (Phase 0 follow-up note).

    Returns a dict with train/val loss curves, the final cos-sim summary
    (computed from restored-best parameters), and the best-step pointer.
    """
    optimizer = mxopt.Adam(learning_rate=lr)

    def loss_fn(model, x, o):
        pred, _ = model(x)
        return mse_loss(pred, o)

    loss_and_grad = nn.value_and_grad(ttt, loss_fn)

    train_losses, val_losses, val_cos_curve = [], [], []
    best_cos = -2.0
    best_step = 0
    best_params = None

    for step in range(n_steps):
        loss, grads = loss_and_grad(ttt, x_train, o_train)
        optimizer.update(ttt, grads)
        mx.eval(ttt.parameters(), optimizer.state)

        if step % log_every == 0 or step == n_steps - 1:
            pred_v, _ = ttt(x_val)
            v_loss = mse_loss(pred_v, o_val).item()
            t_loss = loss.item()
            sims = cos_sim_per_token(pred_v, o_val)
            mean_cos = mx.mean(sims).item()
            train_losses.append((step, t_loss))
            val_losses.append((step, v_loss))
            val_cos_curve.append((step, mean_cos))
            marker = ""
            if early_stop and mean_cos > best_cos:
                best_cos = mean_cos
                best_step = step
                best_params = tree_flatten(ttt.parameters())
                marker = "  ★ best"
            logger.info(
                f"  step {step:>5d}  train_mse={t_loss:.4e}  "
                f"val_mse={v_loss:.4e}  val_cos_mean={mean_cos:.4f}{marker}"
            )

    # Restore best params (if early_stop tracked any)
    if early_stop and best_params is not None:
        ttt.update(tree_unflatten(best_params))
        logger.info(f"  Restored best params from step {best_step} "
                    f"(val_cos_mean={best_cos:.4f})")

    # Final eval (against restored-best params)
    pred_v, _ = ttt(x_val)
    sims = cos_sim_per_token(pred_v, o_val)
    sims_flat = sims.reshape(-1)
    sims_np = sorted(sims_flat.tolist())
    n = len(sims_np)
    summary = {
        "mean_cos": sum(sims_np) / n,
        "p10_cos": sims_np[int(0.10 * n)],
        "p50_cos": sims_np[int(0.50 * n)],
        "p90_cos": sims_np[int(0.90 * n)],
        "min_cos": sims_np[0],
        "max_cos": sims_np[-1],
        "final_train_mse": train_losses[-1][1],
        "final_val_mse": val_losses[-1][1],
        "best_step": best_step,
        "best_val_cos_mean": best_cos,
    }
    return {
        "summary": summary,
        "train_curve": train_losses,
        "val_curve": val_losses,
        "val_cos_curve": val_cos_curve,
    }


# ---------------------------------------------------------------------------
# Synthetic data generator (validates training loop without a model load)
# ---------------------------------------------------------------------------


def make_synthetic_pairs(
    L: int = 1024, D: int = 128, eta_truth: float = 0.05, seed: int = 0,
    mini_batch_size: int = 64, use_layer_norm: bool = True,
) -> tuple[mx.array, mx.array]:
    """Generate ground-truth (x, o) by forward-running a TRUE TTT-Linear
    with known random projections. Training another TTT-Linear from random
    init should drive cos-sim → 1 on held-out tokens — that confirms the
    optimizer + loss landscape work in the regime where the target is
    *exactly* expressible as a TTT-Linear.

    Defaults mirror the deployment regime: small η + mini-batch + LN. The
    online-η=0.5 unbounded recurrence overflows fp32 well before L=1024.
    """
    mx.random.seed(seed)
    cfg = TTTLinearConfig(head_dim=D, eta=eta_truth,
                          mini_batch_size=mini_batch_size,
                          use_layer_norm=use_layer_norm)
    truth = TTTLinear(cfg)
    x = mx.random.normal((1, L, D))
    o, _ = truth(x)
    return x, o


def make_synthetic_pairs_multi(
    n_seqs: int, L: int = 1024, D: int = 128, eta_truth: float = 0.05,
    truth_seed: int = 0, data_seed: int = 0,
    mini_batch_size: int = 64, use_layer_norm: bool = True,
) -> tuple[mx.array, mx.array]:
    """Generate ``n_seqs`` independent sequences sharing one ground-truth
    TTT-Linear. Returns ``(x, o)`` each shape ``(n_seqs, L, D)``.

    Two seeds: ``truth_seed`` controls the ground-truth projections (held
    fixed across train/val), ``data_seed`` controls the input distribution
    (varied across train/val). This is the multi-sequence analog of
    `make_synthetic_pairs`; methodology fix from the Phase 0 step 3
    follow-up note.
    """
    # Ground truth fixed across all sequences
    mx.random.seed(truth_seed)
    cfg = TTTLinearConfig(head_dim=D, eta=eta_truth,
                          mini_batch_size=mini_batch_size,
                          use_layer_norm=use_layer_norm)
    truth = TTTLinear(cfg)
    # Then sample input sequences with the data seed
    mx.random.seed(data_seed)
    x = mx.random.normal((n_seqs, L, D))
    o, _ = truth(x)
    return x, o


# ---------------------------------------------------------------------------
# Real-model capture (heavyweight; user-invoked)
# ---------------------------------------------------------------------------


# Default calibration prompts — varied content so the captured per-head
# (Q_h, o_h) sequences sample TTT state space from multiple starting
# distributions. Streaming heads are local-attention dominant, but the
# *content* of those local windows differs sharply across these snippets;
# train/val split across prompts (rather than within one) prevents the
# overfitting documented in the Phase 0 step 3 follow-up note.
DEFAULT_CALIBRATION_PROMPTS: list[str] = [
    # Code: pure-Python data transform
    ("import json\nfrom pathlib import Path\n\n"
     "def process(items):\n    return [str(x).upper() for x in items]\n\n"
     "def main():\n    data = json.loads(Path('input.json').read_text())\n"
     "    print(process(data))\n\n"),
    # Code: numerical / scientific
    ("import numpy as np\n\n"
     "def gaussian(x, mu=0.0, sigma=1.0):\n"
     "    z = (x - mu) / sigma\n"
     "    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))\n\n"
     "def kl_divergence(p, q, eps=1e-12):\n"
     "    return float(np.sum(p * np.log((p + eps) / (q + eps))))\n\n"),
    # Prose: documentation / chat-like
    ("The hypercar inference stack runs on Apple M4 Pro. The KV cache is\n"
     "compressed via Walsh-Hadamard rotation followed by 3-bit affine\n"
     "quantization. This composes with DuoAttention's per-head streaming\n"
     "and retrieval split, where streaming heads attend mainly to a\n"
     "local 256-token window plus 4 sink tokens, and retrieval heads\n"
     "attend across the full context.\n\n"),
    # Code: SQL-flavored / structured
    ("CREATE TABLE events (id BIGINT PRIMARY KEY, ts TIMESTAMP, kind TEXT);\n"
     "CREATE INDEX events_ts_idx ON events(ts);\n\n"
     "SELECT kind, COUNT(*) AS n FROM events\n"
     "WHERE ts >= NOW() - INTERVAL '7 days'\n"
     "GROUP BY kind ORDER BY n DESC LIMIT 20;\n\n"),
    # Code: error-handling pattern
    ("def safe_divide(a: float, b: float) -> float:\n"
     "    try:\n        return a / b\n    except ZeroDivisionError:\n"
     "        return float('inf') if a > 0 else float('-inf')\n\n"
     "assert safe_divide(1, 0) == float('inf')\n"
     "assert safe_divide(-1, 0) == float('-inf')\n\n"),
    # Prose: long-form prose
    ("The proof proceeds by induction. Base case n=1: the recurrence\n"
     "reduces to the identity map, which trivially satisfies the bound.\n"
     "Inductive step: assume the claim for n=k; we show n=k+1. Apply\n"
     "the update rule once more and observe that each component is\n"
     "bounded above by the inductive hypothesis times the contraction\n"
     "factor.\n\n"),
    # Code: regex-heavy
    ("import re\n\n"
     "EMAIL = re.compile(r'^[\\w.+-]+@[\\w-]+\\.[\\w.-]+$')\n"
     "URL   = re.compile(r'^https?://[^\\s/$.?#].[^\\s]*$')\n\n"
     "def classify(token: str) -> str:\n"
     "    if EMAIL.match(token): return 'email'\n"
     "    if URL.match(token):   return 'url'\n"
     "    return 'plain'\n\n"),
    # Prose: chat-like Q&A
    ("Q: How does TTT-Linear differ from softmax attention?\n"
     "A: TTT-Linear uses a learnable linear map W as its hidden state,\n"
     "updated per token via online SGD on a reconstruction loss. Per-\n"
     "token compute is O(D^2) regardless of context length, vs softmax\n"
     "attention's O(N*D) for the same step.\n\n"),
]


def tile_prompt_to_length(tokenizer, prompt: str, target_len: int) -> list[int]:
    """Encode + tile a prompt up to (or above) ``target_len``, then trim."""
    tokens = tokenizer.encode(prompt)
    if not tokens:
        raise ValueError(f"Empty token list for prompt: {prompt[:80]!r}")
    reps = (target_len // len(tokens)) + 1
    return (tokens * reps)[:target_len]


def _patch_sdpa_for_capture(
    target_call_idx: int, head_idx: int, n_attn_calls_per_forward: int,
) -> tuple[dict, list[int], object]:
    """Install a capturing SDPA that records (Q_h, o_h) for one chosen
    attention call. Returns (captured_dict, layer_counter,
    original_sdpa); call ``_unpatch_sdpa(original)`` when done.

    ``target_call_idx`` is the position (0-indexed) within the per-
    forward attention-call sequence to capture. For Qwen3-Coder this
    equals ``layer_idx`` since every layer is attention. For Qwen3.6
    it's ``attention_layer_indices(model).index(layer_idx)``.

    ``n_attn_calls_per_forward`` is the modulo for wraparound across
    multi-prompt loops — equals ``len(attention_layer_indices(model))``.

    The capturing SDPA mirrors ``duoattention_calibrate.py`` — handles
    GQA expansion, ``mask='causal'`` string sentinel, and array masks.
    Captured sub-tensors stay attached to MLX's compute graph; caller
    is responsible for ``mx.eval`` after the prefill.
    """
    import mlx_lm.models.base as mlx_base

    captured: dict = {}
    layer_counter = [0]
    original_sdpa = mlx_base.scaled_dot_product_attention

    def capturing_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        B, H_q, L_q, D = queries.shape
        H_kv = keys.shape[1]
        gqa = H_q // H_kv
        if H_kv < H_q:
            keys_exp = mx.repeat(keys, gqa, axis=1)
            values_exp = mx.repeat(values, gqa, axis=1)
        else:
            keys_exp = keys
            values_exp = values
        scores = (queries @ keys_exp.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                L_kv = keys_exp.shape[2]
                causal = mx.triu(mx.full((L_q, L_kv), -3.4e4), k=1)
                scores = scores + causal
            elif isinstance(mask, mx.array):
                scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        out = weights @ values_exp                 # (B, H_q, L, D)

        cur_call = layer_counter[0] % n_attn_calls_per_forward
        layer_counter[0] += 1
        if cur_call == target_call_idx:
            captured["Q_h"] = queries[:, head_idx, :, :]   # (B, L, D)
            captured["o_h"] = out[:, head_idx, :, :]       # (B, L, D)
            captured["head_dim"] = D
        return out

    mlx_base.scaled_dot_product_attention = capturing_sdpa
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
            if hasattr(mod, "scaled_dot_product_attention"):
                setattr(mod, "scaled_dot_product_attention", capturing_sdpa)

    return captured, layer_counter, original_sdpa


def _unpatch_sdpa(original_sdpa) -> None:
    import mlx_lm.models.base as mlx_base
    mlx_base.scaled_dot_product_attention = original_sdpa
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
            if hasattr(mod, "scaled_dot_product_attention"):
                setattr(mod, "scaled_dot_product_attention", original_sdpa)


def capture_attention_pairs_multi(
    model_id: str,
    layer_idx: int,
    head_idx: int,
    context_len: int,
    prompts: Optional[list[str]] = None,
) -> tuple[mx.array, mx.array, int]:
    """Load model, run prefill on N calibration prompts, return per-head
    queries and per-head SDPA output stacked along the batch dim.

    Returns ``(Q_h, o_h, head_dim)`` where each of ``Q_h``/``o_h`` has
    shape ``(N, context_len, head_dim)``. N defaults to
    ``len(DEFAULT_CALIBRATION_PROMPTS)``.

    Multi-prompt (vs the single-prompt earlier API) is the third
    methodology fix from the Phase 0 step 3 follow-up note: train/val
    split across prompts gives the optimizer signal to learn the
    streaming-head dynamics rather than memorize one prompt's specifics.
    """
    from mlx_lm import load

    if prompts is None:
        prompts = DEFAULT_CALIBRATION_PROMPTS

    logger.info(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)

    # Hybrid-safe: SDPA only fires on attention layers. Convert the
    # policy-file `layer_idx` (model.layers index) to the position
    # within the per-forward attention-call sequence.
    attn_indices = attention_layer_indices(model)
    if layer_idx not in attn_indices:
        raise ValueError(
            f"layer_idx={layer_idx} is not an attention layer in this "
            f"model (attention layers are at {attn_indices}). The policy "
            f"file may have been calibrated for a different architecture."
        )
    target_call_idx = attn_indices.index(layer_idx)
    logger.info(
        f"Hybrid-aware capture: model has {len(model.layers)} layers, "
        f"{len(attn_indices)} attention. Target layer {layer_idx} = "
        f"attention call {target_call_idx} of {len(attn_indices)}."
    )

    captured, layer_counter, original_sdpa = _patch_sdpa_for_capture(
        target_call_idx, head_idx, len(attn_indices))

    Q_h_list: list[mx.array] = []
    o_h_list: list[mx.array] = []
    head_dim: Optional[int] = None
    try:
        for i, prompt in enumerate(prompts):
            tokens = tile_prompt_to_length(tokenizer, prompt, context_len)
            x = mx.array([tokens])
            captured.clear()
            layer_counter[0] = 0  # reset so layer_idx hits on this prompt
            t0 = time.perf_counter()
            logits = model(x)
            mx.eval(logits)
            logger.info(
                f"  prompt {i+1}/{len(prompts)}: {context_len} tok in "
                f"{time.perf_counter() - t0:.1f}s"
            )
            if "Q_h" not in captured:
                raise RuntimeError(
                    f"Capture missed layer {layer_idx} on prompt {i} — "
                    "layer_counter wraparound or SDPA dispatch path "
                    "skipped that layer."
                )
            Q_h_list.append(captured["Q_h"])    # (1, L, D)
            o_h_list.append(captured["o_h"])    # (1, L, D)
            if head_dim is None:
                head_dim = captured["head_dim"]
    finally:
        _unpatch_sdpa(original_sdpa)

    Q_h = mx.concatenate(Q_h_list, axis=0)       # (N, L, D)
    o_h = mx.concatenate(o_h_list, axis=0)       # (N, L, D)
    return Q_h, o_h, head_dim


def capture_attention_pairs(
    model_id: str,
    layer_idx: int,
    head_idx: int,
    context_len: int,
) -> tuple[mx.array, mx.array, int]:
    """Single-prompt capture (legacy API kept for back-compat).

    Wraps ``capture_attention_pairs_multi`` with a one-element prompt
    list — the first DEFAULT_CALIBRATION_PROMPTS entry. Returns shape
    ``(1, L, D)`` to match the original contract.
    """
    return capture_attention_pairs_multi(
        model_id, layer_idx, head_idx, context_len,
        prompts=[DEFAULT_CALIBRATION_PROMPTS[0]],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_synthetic(D: int, n_steps: int, lr: float, eta: float,
                  mini_batch_size: int, n_train_seqs: int = 8,
                  n_val_seqs: int = 4, L: int = 1024) -> dict:
    """Synthetic recovery via multi-sequence training.

    Generates ``n_train_seqs`` + ``n_val_seqs`` independent sequences
    sharing one ground-truth TTT-Linear's projections. Trains the student
    on the train sequences and evals on the val sequences (which share
    the dynamics but are drawn from different inputs). This is the
    Phase 0 step 3 follow-up methodology — single-sequence positional
    split overfits at step ~200 (peak val cos-sim 0.69 → degrades).
    """
    logger.info(f"Synthetic mode (multi-sequence): D={D} L={L} "
                f"n_train={n_train_seqs} n_val={n_val_seqs} n_steps={n_steps} "
                f"lr={lr} eta={eta} mini_batch_size={mini_batch_size}")
    x_train, o_train = make_synthetic_pairs_multi(
        n_seqs=n_train_seqs, L=L, D=D, eta_truth=eta,
        truth_seed=0, data_seed=42, mini_batch_size=mini_batch_size)
    x_val, o_val = make_synthetic_pairs_multi(
        n_seqs=n_val_seqs, L=L, D=D, eta_truth=eta,
        truth_seed=0, data_seed=99, mini_batch_size=mini_batch_size)

    cfg = TTTLinearConfig(head_dim=D, eta=eta,
                          mini_batch_size=mini_batch_size,
                          use_layer_norm=True)
    ttt = TTTLinear(cfg)
    return train_ttt(ttt, x_train, o_train, x_val, o_val,
                     n_steps=n_steps, lr=lr)


def run_capture(model_id: str, policy_path: Path, context_len: int,
                n_steps: int, lr: float, eta: float,
                mini_batch_size: int,
                prompts: Optional[list[str]] = None,
                n_val_prompts: int = 2,
                save_block_dir: Optional[Path] = None) -> dict:
    """Capture per-head pairs across N prompts, train/val-split across
    prompts (not within), train via the multi-sequence path that cleared
    the synthetic gate at 0.9997.

    If ``save_block_dir`` is given, the trained TTT-Linear (with the
    early-stop-restored best-val params) is written via
    ``omlx.patches.ttt_head_router.save_blocks`` keyed by the picked
    ``(layer, head)``. A ``TTTHeadRouter.from_policy(ttt_dir=...)`` can
    then load it directly.
    """
    sel = pick_streaming_head(policy_path)
    logger.info(f"Selected streaming head: layer={sel.layer} head={sel.head} "
                f"local_fraction={sel.local_fraction:.4f}")
    if prompts is None:
        prompts = DEFAULT_CALIBRATION_PROMPTS
    if n_val_prompts >= len(prompts):
        raise ValueError(
            f"n_val_prompts ({n_val_prompts}) must be < len(prompts) "
            f"({len(prompts)})."
        )
    Q_h, o_h, D = capture_attention_pairs_multi(
        model_id, sel.layer, sel.head, context_len, prompts=prompts)
    logger.info(f"Captured {Q_h.shape} per-head queries + outputs "
                f"(head_dim={D})")

    # Train on the first N - n_val prompts; val on the last n_val.
    n_total = Q_h.shape[0]
    split = n_total - n_val_prompts
    x_train, o_train = Q_h[:split, :, :], o_h[:split, :, :]
    x_val, o_val = Q_h[split:, :, :], o_h[split:, :, :]
    logger.info(f"Train/val split: {x_train.shape[0]} prompts / "
                f"{x_val.shape[0]} prompts")

    cfg = TTTLinearConfig(head_dim=D, eta=eta,
                          mini_batch_size=mini_batch_size,
                          use_layer_norm=True)
    ttt = TTTLinear(cfg)
    out = train_ttt(ttt, x_train, o_train, x_val, o_val,
                    n_steps=n_steps, lr=lr)
    out["selection"] = {
        "layer": sel.layer, "head": sel.head,
        "local_fraction": sel.local_fraction,
    }
    out["n_train_prompts"] = int(x_train.shape[0])
    out["n_val_prompts"] = int(x_val.shape[0])

    if save_block_dir is not None:
        from omlx.patches.ttt_head_router import save_blocks
        save_blocks({(sel.layer, sel.head): ttt}, save_block_dir)
        logger.info(
            f"  trained block written to {save_block_dir}/L{sel.layer}_H{sel.head}.* "
            "(load via TTTHeadRouter.from_policy(ttt_dir=...))"
        )
        out["save_block_dir"] = str(save_block_dir)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip model load; train against a known "
                             "TTT-Linear ground truth (validates math).")
    parser.add_argument("--capture", action="store_true",
                        help="Load model + capture per-head pairs.")
    parser.add_argument("--policy", type=Path,
                        default=Path("omlx/patches/duoattention_policies/"
                                     "qwen3_coder_30b_a3b_instruct_8bit.json"))
    parser.add_argument("--model",
                        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=128,
                        help="Synthetic-mode only — match Qwen3-Coder.")
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=0.05,
                        help="Inner-loop SGD rate. Online η=0.5 overflows "
                             "fp32 at long L; deployment uses small η + "
                             "mini-batch + LN for stability.")
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--n-val-prompts", type=int, default=2,
                        help="Capture mode: how many prompts to hold out as "
                             "val. Train uses the rest.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results to this path.")
    parser.add_argument("--save-block-dir", type=Path, default=None,
                        help="Capture mode: write trained TTT-Linear "
                             "weights to this directory using "
                             "save_blocks() so a TTTHeadRouter can load "
                             "them later via from_policy(ttt_dir=...).")
    args = parser.parse_args()

    if not (args.synthetic or args.capture):
        parser.error("Pass --synthetic or --capture.")

    if args.synthetic:
        result = run_synthetic(
            D=args.head_dim, n_steps=args.n_steps, lr=args.lr,
            eta=args.eta, mini_batch_size=args.mini_batch_size,
        )
    else:
        result = run_capture(
            model_id=args.model, policy_path=args.policy,
            context_len=args.context_len,
            n_steps=args.n_steps, lr=args.lr,
            eta=args.eta, mini_batch_size=args.mini_batch_size,
            n_val_prompts=args.n_val_prompts,
            save_block_dir=args.save_block_dir,
        )

    s = result["summary"]
    gate_pass = s["mean_cos"] >= 0.95
    logger.info("")
    logger.info(f"=== Phase 0 spike gate: cos-sim ≥ 0.95 → "
                f"{'PASS' if gate_pass else 'FAIL'} ===")
    logger.info(f"  mean   = {s['mean_cos']:.4f}")
    logger.info(f"  p10    = {s['p10_cos']:.4f}")
    logger.info(f"  p50    = {s['p50_cos']:.4f}")
    logger.info(f"  p90    = {s['p90_cos']:.4f}")
    logger.info(f"  min    = {s['min_cos']:.4f}")
    logger.info(f"  final train MSE = {s['final_train_mse']:.4e}")
    logger.info(f"  final val   MSE = {s['final_val_mse']:.4e}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        logger.info(f"  results written to {args.output}")

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
