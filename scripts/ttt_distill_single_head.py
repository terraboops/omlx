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

sys.path.insert(0, ".")
from omlx.state_space import TTTLinear, TTTLinearConfig

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
) -> dict:
    """Adam-train a TTT-Linear block to regress ``o`` from ``x``.

    Returns a dict with the train/val loss curves and the final cos-sim
    summary (mean, p10, p50, p90 per-token).
    """
    optimizer = mxopt.Adam(learning_rate=lr)

    def loss_fn(model, x, o):
        pred, _ = model(x)
        return mse_loss(pred, o)

    loss_and_grad = nn.value_and_grad(ttt, loss_fn)

    train_losses, val_losses = [], []
    for step in range(n_steps):
        loss, grads = loss_and_grad(ttt, x_train, o_train)
        optimizer.update(ttt, grads)
        mx.eval(ttt.parameters(), optimizer.state)

        if step % log_every == 0 or step == n_steps - 1:
            pred_v, _ = ttt(x_val)
            v_loss = mse_loss(pred_v, o_val).item()
            t_loss = loss.item()
            train_losses.append((step, t_loss))
            val_losses.append((step, v_loss))
            sims = cos_sim_per_token(pred_v, o_val)
            mean_cos = mx.mean(sims).item()
            logger.info(
                f"  step {step:>5d}  train_mse={t_loss:.4e}  "
                f"val_mse={v_loss:.4e}  val_cos_mean={mean_cos:.4f}"
            )

    # Final eval
    pred_v, _ = ttt(x_val)
    sims = cos_sim_per_token(pred_v, o_val)        # (B, L)
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
    }
    return {
        "summary": summary,
        "train_curve": train_losses,
        "val_curve": val_losses,
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


# ---------------------------------------------------------------------------
# Real-model capture (heavyweight; user-invoked)
# ---------------------------------------------------------------------------


def capture_attention_pairs(
    model_id: str,
    layer_idx: int,
    head_idx: int,
    context_len: int,
) -> tuple[mx.array, mx.array, int]:
    """Load model, run prefill on a calibration prompt, capture per-head
    queries and per-head SDPA output for ``(layer_idx, head_idx)``.

    Returns ``(Q_h, o_h, head_dim)`` where each of ``Q_h``/``o_h`` has
    shape ``(1, L, head_dim)``.
    """
    from mlx_lm import load
    import mlx_lm.models.base as mlx_base

    logger.info(f"Loading model: {model_id}")
    model, tokenizer = load(model_id)

    n_layers = len(model.layers)
    captured: dict = {}      # filled by capturing_sdpa
    layer_counter = [0]

    original_sdpa = mlx_base.scaled_dot_product_attention

    def capturing_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
        """Drop-in SDPA that records (Q_h, o_h) for the chosen layer."""
        # mask handling — mirror duoattention_calibrate.py
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

        cur_layer = layer_counter[0] % n_layers
        layer_counter[0] += 1
        if cur_layer == layer_idx:
            captured["Q_h"] = queries[:, head_idx, :, :]   # (B, L, D)
            captured["o_h"] = out[:, head_idx, :, :]       # (B, L, D)
            captured["head_dim"] = D
        return out

    # Patch SDPA + walk imported model modules (D13 lesson)
    mlx_base.scaled_dot_product_attention = capturing_sdpa
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if mod_name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
            if hasattr(mod, "scaled_dot_product_attention"):
                setattr(mod, "scaled_dot_product_attention", capturing_sdpa)

    # Prefill on a code-heavy prompt
    code = (
        "import json\nfrom pathlib import Path\n\n"
        "def process(items):\n    return [str(x).upper() for x in items]\n\n"
    )
    tokens = tokenizer.encode(code)
    reps = (context_len // len(tokens)) + 1
    full_tokens = (tokens * reps)[:context_len]
    logger.info(f"Prefilling {len(full_tokens)} tokens to capture (l={layer_idx}, h={head_idx})...")
    try:
        x = mx.array([full_tokens])
        t0 = time.perf_counter()
        logits = model(x)
        mx.eval(logits)
        logger.info(f"Prefill {time.perf_counter() - t0:.1f}s")
    finally:
        mlx_base.scaled_dot_product_attention = original_sdpa
        for mod_name, mod in list(sys.modules.items()):
            if mod is None:
                continue
            if mod_name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
                if hasattr(mod, "scaled_dot_product_attention"):
                    setattr(mod, "scaled_dot_product_attention", original_sdpa)

    if "Q_h" not in captured:
        raise RuntimeError(
            f"Capture missed layer {layer_idx} — layer_counter wraparound or "
            "SDPA dispatch path skipped that layer."
        )
    return captured["Q_h"], captured["o_h"], captured["head_dim"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_synthetic(D: int, n_steps: int, lr: float, eta: float,
                  mini_batch_size: int) -> dict:
    """Synthetic recovery: positional split on one sequence (mirrors
    capture mode) so train and val share the recurrence dynamics."""
    logger.info(f"Synthetic mode: D={D} n_steps={n_steps} lr={lr} "
                f"eta={eta} mini_batch_size={mini_batch_size}")
    L_total = 1280
    x_full, o_full = make_synthetic_pairs(
        L=L_total, D=D, eta_truth=eta, mini_batch_size=mini_batch_size)
    split = int(0.8 * L_total)
    x_train, o_train = x_full[:, :split, :], o_full[:, :split, :]
    x_val, o_val = x_full[:, split:, :], o_full[:, split:, :]

    cfg = TTTLinearConfig(head_dim=D, eta=eta,
                          mini_batch_size=mini_batch_size,
                          use_layer_norm=True)
    ttt = TTTLinear(cfg)
    return train_ttt(ttt, x_train, o_train, x_val, o_val,
                     n_steps=n_steps, lr=lr)


def run_capture(model_id: str, policy_path: Path, context_len: int,
                n_steps: int, lr: float, eta: float,
                mini_batch_size: int) -> dict:
    sel = pick_streaming_head(policy_path)
    logger.info(f"Selected streaming head: layer={sel.layer} head={sel.head} "
                f"local_fraction={sel.local_fraction:.4f}")
    Q_h, o_h, D = capture_attention_pairs(
        model_id, sel.layer, sel.head, context_len)
    logger.info(f"Captured {Q_h.shape} per-head queries + outputs "
                f"(head_dim={D})")
    # Train/val split (90/10)
    L = Q_h.shape[1]
    split = int(0.9 * L)
    x_train, o_train = Q_h[:, :split, :], o_h[:, :split, :]
    x_val, o_val = Q_h[:, split:, :], o_h[:, split:, :]

    cfg = TTTLinearConfig(head_dim=D, eta=eta,
                          mini_batch_size=mini_batch_size,
                          use_layer_norm=False)
    ttt = TTTLinear(cfg)
    out = train_ttt(ttt, x_train, o_train, x_val, o_val,
                    n_steps=n_steps, lr=lr)
    out["selection"] = {
        "layer": sel.layer, "head": sel.head,
        "local_fraction": sel.local_fraction,
    }
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
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results to this path.")
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
