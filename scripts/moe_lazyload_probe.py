#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ProMoE lazy-load probe for Qwen3-Coder expert weights.

Tests whether partially masking MoE experts reduces Metal memory
while maintaining model quality (smoke + coherence).

Qwen3-Coder-30B-A3B uses QuantizedSwitchLinear — all 128 experts
are fused into a single (128, H, W) tensor per projection. To
simulate partial loading, we zero-mask the router logits for
non-resident experts so they are never dispatched to.

Usage:
    .venv/bin/python scripts/moe_lazyload_probe.py --resident-fraction 0.5
"""

from __future__ import annotations

import argparse
import gc
import logging
import time

import mlx.core as mx

logger = logging.getLogger("moe_probe")


def probe_partial_experts(
    model_id: str,
    resident_fraction: float,
    max_tokens: int = 32,
):
    """Load model and mask non-resident experts at the router level.

    The QuantizedSwitchLinear stores all experts in a fused tensor.
    We can't avoid loading them, but we CAN prevent the router from
    dispatching to the cold experts by biasing their gate logits to -inf.
    This simulates the memory access pattern of a partial-load system.
    """
    from mlx_lm import load

    print(f"Loading model: {model_id}")
    t0 = time.perf_counter()
    model, tokenizer = load(model_id)
    load_time = time.perf_counter() - t0

    metal_at_load = mx.get_active_memory() / 1e9
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Metal at load: {metal_at_load:.1f} GB")

    n_layers = len(model.layers)
    num_experts = model.layers[0].mlp.num_experts
    top_k = model.layers[0].mlp.top_k
    n_resident = max(top_k, int(num_experts * resident_fraction))

    print(f"\nArchitecture: {n_layers} layers, {num_experts} experts, top-{top_k}")
    print(f"Resident fraction: {resident_fraction} → {n_resident}/{num_experts} experts")
    print(f"Cold experts: {num_experts - n_resident}")

    # Measure expert weight sizes
    sw = model.layers[0].mlp.switch_mlp
    expert_mb_per_layer = 0
    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
        proj = sw[proj_name]
        expert_mb_per_layer += proj.weight.nbytes / 1e6
    total_expert_gb = expert_mb_per_layer * n_layers / 1e3
    cold_expert_gb = total_expert_gb * (1 - resident_fraction)

    print(f"\nExpert memory per layer: {expert_mb_per_layer:.0f} MB")
    print(f"Total expert memory: {total_expert_gb:.1f} GB")
    print(f"Cold expert memory (could be saved): {cold_expert_gb:.1f} GB")

    # --- Patch gate modules to mask cold experts ---
    # The original forward (qwen3_moe.py line 123-139):
    #   gates = self.gate(x)          # (B, T, 128)
    #   gates = softmax(gates)        # probabilities
    #   inds = argpartition(gates, kth=-k)[..., -k:]  # top-k indices
    #   scores = take_along_axis(gates, inds)
    #   y = switch_mlp(x, inds)
    #   y = (y * scores[..., None]).sum(axis=-2)
    #
    # We patch gate.__call__ to add -1e9 bias to cold expert logits
    # BEFORE softmax, so they get ~0 probability and are never selected.

    # Patch the MoE block's __call__ to mask gate logits before softmax.
    # We patch the CLASS method, not instance, so it affects all layers.
    import mlx_lm.models.qwen3_moe as qwen3_mod
    _original_moe_call = qwen3_mod.Qwen3MoeSparseMoeBlock.__call__

    def _patched_moe_call(self, x):
        gates = self.gate(x)
        # Apply cold-expert mask before softmax
        if n_resident < self.num_experts:
            cold_mask = mx.arange(self.num_experts) >= n_resident
            gates = mx.where(cold_mask, gates - 1e9, gates)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y

    qwen3_mod.Qwen3MoeSparseMoeBlock.__call__ = _patched_moe_call

    print(f"\nRouter masking applied: experts >= {n_resident} get -inf gate logits")

    # --- Test quality ---
    from omlx.patches.prefill_last_logit import apply_prefill_last_logit_patch
    apply_prefill_last_logit_patch(model)

    print("\n--- Quality Tests ---")

    def _chat_encode(text):
        messages = [{"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return tokenizer.encode(prompt)

    # Test 1: Coherence (2+2)
    tokens = _chat_encode("What is 2+2? Answer with just the number.")
    x = mx.array([tokens])
    logits = model(x)
    mx.eval(logits)

    generated = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        generated.append(token.item())
        x = token.reshape(1, 1)
        logits = model(x)
        mx.eval(logits)

    text = tokenizer.decode(generated)
    math_pass = "4" in text
    print(f"Math (2+2): {'PASS' if math_pass else 'FAIL'} — {text[:60]!r}")

    # Test 2: Code
    gc.collect()
    mx.clear_cache()

    tokens = _chat_encode("Write hello world in Python. Just the code, nothing else.")
    x = mx.array([tokens])
    logits = model(x)
    mx.eval(logits)

    generated = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        generated.append(token.item())
        x = token.reshape(1, 1)
        logits = model(x)
        mx.eval(logits)

    text2 = tokenizer.decode(generated)
    code_pass = "print" in text2.lower()
    print(f"Code:       {'PASS' if code_pass else 'FAIL'} — {text2[:60]!r}")

    metal_at_inference = mx.get_active_memory() / 1e9
    metal_peak = mx.get_peak_memory() / 1e9

    print(f"\n--- Memory ---")
    print(f"Metal at load:      {metal_at_load:.1f} GB")
    print(f"Metal at inference: {metal_at_inference:.1f} GB")
    print(f"Metal peak:         {metal_peak:.1f} GB")
    print(f"Could save with lazy-load: {cold_expert_gb:.1f} GB")

    print(f"\n--- Verdict ---")
    quality_ok = math_pass and code_pass
    if quality_ok:
        print(f"VIABLE: {resident_fraction*100:.0f}% expert residency maintains quality.")
        print(f"  Lazy-load could save ~{cold_expert_gb:.1f} GB Metal memory.")
        print(f"  Requires per-expert weight files (current: fused QuantizedSwitchLinear).")
    else:
        print(f"FAIL: {resident_fraction*100:.0f}% expert residency degrades quality.")
        print(f"  Need higher resident fraction or 4-bit re-quant on cold experts.")

    return {
        "resident_fraction": resident_fraction,
        "n_resident": n_resident,
        "n_total": num_experts,
        "metal_at_load_gb": round(metal_at_load, 1),
        "metal_peak_gb": round(metal_peak, 1),
        "cold_expert_gb": round(cold_expert_gb, 1),
        "math_pass": math_pass,
        "code_pass": code_pass,
        "quality_ok": quality_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="ProMoE lazy-load probe")
    parser.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    parser.add_argument("--resident-fraction", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    probe_partial_experts(args.model, args.resident_fraction)


if __name__ == "__main__":
    main()
