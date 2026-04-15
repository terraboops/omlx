#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""EAGLE-2 tree-attention feasibility probe for MLX (Task 28).

Tests whether MLX's attention primitives accept tree-shaped (non-causal)
masks, which EAGLE-2 speculative decoding requires for verifying multiple
draft-token candidates laid out as a tree.

EAGLE-2 tree structure example (4 branches, depth 3):
    root -> [c1, c2, c3, c4]        # 4 candidates at depth 1
    c1   -> [c1a, c1b]              # 2 sub-candidates at depth 2
    c2   -> [c2a]                   # 1 sub-candidate
    ...
    Total: 12 candidate tokens verified in one forward pass

The tree mask is a boolean matrix where position (i,j) is True if token j
is an ancestor-or-self of token i in the tree. This is NOT a simple causal
(lower-triangular) mask — siblings cannot attend to each other.

Usage:
    python scripts/probe_eagle_tree_attn.py

Reports:
    - Whether mx.fast.scaled_dot_product_attention accepts tree masks
    - Numerical accuracy vs NumPy reference
    - Performance comparison: fused SDPA vs vanilla attention with tree mask
"""

from __future__ import annotations

import time
import numpy as np

import mlx.core as mx

# Qwen3-Coder attention shapes
B = 1
H = 32
D = 64
SCALE = D ** -0.5


def build_tree_mask(tree_structure: list[list[int]]) -> np.ndarray:
    """Build a tree attention mask from parent indices.

    Args:
        tree_structure: list of parent indices for each node.
            tree_structure[i] = list of ancestor indices (including self)
            for node i. Root node (index 0) has ancestors = [0].

    Returns:
        Boolean mask of shape (n_nodes, n_nodes) where mask[i][j] = True
        means node i can attend to node j.
    """
    n = len(tree_structure)
    mask = np.zeros((n, n), dtype=bool)
    for i, ancestors in enumerate(tree_structure):
        for j in ancestors:
            mask[i][j] = True
    return mask


def build_eagle_tree(n_branches: int = 4, depth: int = 3) -> tuple[list[list[int]], int]:
    """Build a realistic EAGLE-2 tree structure.

    Returns:
        (tree_structure, n_nodes) where tree_structure[i] = ancestor list
    """
    # Node 0 = root (the last accepted token)
    tree = [[0]]  # root attends to itself

    node_id = 1
    # Build tree breadth-first
    queue = [(0, 0)]  # (parent_id, current_depth)
    while queue:
        parent, d = queue.pop(0)
        if d >= depth:
            continue
        # Number of children decreases with depth (realistic tree shape)
        n_children = max(1, n_branches // (d + 1))
        for _ in range(n_children):
            # This node's ancestors = parent's ancestors + self
            ancestors = tree[parent] + [node_id]
            tree.append(ancestors)
            queue.append((node_id, d + 1))
            node_id += 1

    return tree, len(tree)


def numpy_reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                                mask: np.ndarray) -> np.ndarray:
    """Reference attention implementation in NumPy."""
    # q, k, v: (B, H, S, D)
    scores = np.einsum('bhid,bhjd->bhij', q, k) * SCALE

    # Apply mask: set masked positions to -inf
    # mask shape: (S, S), broadcast to (B, H, S, S)
    mask_4d = mask[np.newaxis, np.newaxis, :, :]
    scores = np.where(mask_4d, scores, -1e9)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    exp_scores = np.where(mask_4d, exp_scores, 0.0)
    attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-10)

    # Weighted sum
    return np.einsum('bhij,bhjd->bhid', attn_weights, v)


def test_sdpa_tree_mask(tree: list[list[int]], n_nodes: int,
                        prefix_len: int = 128) -> dict:
    """Test mx.fast.scaled_dot_product_attention with a tree mask.

    In EAGLE-2 verification, the KV cache already contains `prefix_len`
    accepted tokens. The tree mask applies to the `n_nodes` draft tokens
    being verified, but they also attend to all prefix tokens.

    Full mask shape: (n_nodes, prefix_len + n_nodes)
    - Left block (n_nodes × prefix_len): all True (attend to full prefix)
    - Right block (n_nodes × n_nodes): tree mask (ancestors only)
    """
    total_kv = prefix_len + n_nodes

    # Build full mask
    bool_mask = np.ones((n_nodes, total_kv), dtype=bool)
    tree_mask = build_tree_mask(tree)
    bool_mask[:, prefix_len:] = tree_mask

    # Create additive mask for MLX SDPA (-inf for masked positions)
    additive_mask_np = np.where(bool_mask, 0.0, -1e9).astype(np.float32)

    # Generate random Q, K, V
    np.random.seed(42)
    q_np = np.random.randn(B, H, n_nodes, D).astype(np.float32)
    k_np = np.random.randn(B, H, total_kv, D).astype(np.float32)
    v_np = np.random.randn(B, H, total_kv, D).astype(np.float32)

    # NumPy reference
    ref_output = numpy_reference_attention(q_np, k_np, v_np, bool_mask)

    # MLX tensors
    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)
    mask_mx = mx.array(additive_mask_np)
    mx.eval(q_mx, k_mx, v_mx, mask_mx)

    result = {
        "n_nodes": n_nodes,
        "prefix_len": prefix_len,
        "total_kv": total_kv,
        "mask_shape": f"({n_nodes}, {total_kv})",
    }

    # Test 1: mx.fast.scaled_dot_product_attention
    try:
        sdpa_out = mx.fast.scaled_dot_product_attention(
            q_mx, k_mx, v_mx, scale=SCALE, mask=mask_mx)
        mx.eval(sdpa_out)
        sdpa_np = np.array(sdpa_out, dtype=np.float32)

        # Check accuracy vs reference
        max_err = np.max(np.abs(sdpa_np - ref_output))
        mean_err = np.mean(np.abs(sdpa_np - ref_output))
        result["sdpa_supported"] = True
        result["sdpa_max_error"] = float(max_err)
        result["sdpa_mean_error"] = float(mean_err)
        result["sdpa_verdict"] = "PASS" if max_err < 0.01 else "NUMERICAL_ISSUE"
    except Exception as e:
        result["sdpa_supported"] = False
        result["sdpa_error"] = str(e)
        result["sdpa_verdict"] = "FAIL"

    # Test 2: Vanilla MLX attention (QK^T + mask + softmax + @V)
    try:
        scores = (q_mx @ k_mx.swapaxes(-1, -2)) * SCALE
        scores = scores + mask_mx
        weights = mx.softmax(scores, axis=-1)
        vanilla_out = weights @ v_mx
        mx.eval(vanilla_out)
        vanilla_np = np.array(vanilla_out, dtype=np.float32)

        max_err = np.max(np.abs(vanilla_np - ref_output))
        mean_err = np.mean(np.abs(vanilla_np - ref_output))
        result["vanilla_supported"] = True
        result["vanilla_max_error"] = float(max_err)
        result["vanilla_mean_error"] = float(mean_err)
        result["vanilla_verdict"] = "PASS" if max_err < 0.01 else "NUMERICAL_ISSUE"
    except Exception as e:
        result["vanilla_supported"] = False
        result["vanilla_error"] = str(e)
        result["vanilla_verdict"] = "FAIL"

    return result


def bench_tree_attention(tree: list[list[int]], n_nodes: int,
                          prefix_len: int = 2048,
                          iterations: int = 50, warmup: int = 10) -> dict:
    """Benchmark tree-masked attention performance."""
    total_kv = prefix_len + n_nodes

    # Build mask
    bool_mask = np.ones((n_nodes, total_kv), dtype=bool)
    tree_mask = build_tree_mask(tree)
    bool_mask[:, prefix_len:] = tree_mask
    additive_mask_np = np.where(bool_mask, 0.0, -1e9).astype(np.float32)

    q_mx = mx.random.normal((B, H, n_nodes, D))
    k_mx = mx.random.normal((B, H, total_kv, D))
    v_mx = mx.random.normal((B, H, total_kv, D))
    mask_mx = mx.array(additive_mask_np)
    mx.eval(q_mx, k_mx, v_mx, mask_mx)

    result = {"prefix_len": prefix_len, "n_nodes": n_nodes}

    # Bench SDPA with tree mask
    for _ in range(warmup):
        out = mx.fast.scaled_dot_product_attention(
            q_mx, k_mx, v_mx, scale=SCALE, mask=mask_mx)
        mx.eval(out)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(
            q_mx, k_mx, v_mx, scale=SCALE, mask=mask_mx)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    result["sdpa_tree_median_ms"] = round(sorted(times)[len(times) // 2] * 1000, 3)

    # Bench SDPA with causal mask (baseline)
    causal_mask = mx.triu(mx.full((n_nodes, total_kv), -1e9), k=1)
    # Actually for decode verification the causal mask isn't right.
    # Use no mask (pure causal via SDPA internal) as baseline:
    for _ in range(warmup):
        out = mx.fast.scaled_dot_product_attention(
            q_mx, k_mx, v_mx, scale=SCALE)
        mx.eval(out)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(
            q_mx, k_mx, v_mx, scale=SCALE)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    result["sdpa_nomask_median_ms"] = round(sorted(times)[len(times) // 2] * 1000, 3)

    # Bench vanilla attention with tree mask (fallback cost)
    for _ in range(warmup):
        scores = (q_mx @ k_mx.swapaxes(-1, -2)) * SCALE + mask_mx
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v_mx
        mx.eval(out)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        scores = (q_mx @ k_mx.swapaxes(-1, -2)) * SCALE + mask_mx
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v_mx
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    result["vanilla_tree_median_ms"] = round(sorted(times)[len(times) // 2] * 1000, 3)

    # Overhead ratios
    if result["sdpa_nomask_median_ms"] > 0:
        result["tree_mask_overhead"] = round(
            result["sdpa_tree_median_ms"] / result["sdpa_nomask_median_ms"], 2)
    if result["sdpa_tree_median_ms"] > 0:
        result["vanilla_fallback_ratio"] = round(
            result["vanilla_tree_median_ms"] / result["sdpa_tree_median_ms"], 2)

    return result


def main():
    print("EAGLE-2 Tree-Attention Feasibility Probe")
    print(f"MLX version: {mx.__version__}")
    print(f"Metal device: {mx.default_device()}")
    print(f"Shapes: B={B}, H={H}, D={D}")
    print()

    # Build a realistic EAGLE-2 tree
    tree, n_nodes = build_eagle_tree(n_branches=4, depth=3)
    print(f"Tree structure: {n_branches_label(tree)} = {n_nodes} nodes")
    print(f"Tree depth: {max(len(a) for a in tree) - 1}")
    print()

    # === Correctness Tests ===
    print("=" * 60)
    print("CORRECTNESS: Tree mask vs NumPy reference")
    print("=" * 60)

    for prefix in [0, 128, 2048]:
        r = test_sdpa_tree_mask(tree, n_nodes, prefix_len=prefix)
        print(f"\n  Prefix={prefix}, Tree={n_nodes} nodes, "
              f"Mask={r['mask_shape']}")

        print(f"    SDPA:    {r['sdpa_verdict']}", end="")
        if r["sdpa_supported"]:
            print(f"  max_err={r['sdpa_max_error']:.6f}  "
                  f"mean_err={r['sdpa_mean_error']:.6f}")
        else:
            print(f"  error: {r.get('sdpa_error', 'unknown')}")

        print(f"    Vanilla: {r['vanilla_verdict']}", end="")
        if r["vanilla_supported"]:
            print(f"  max_err={r['vanilla_max_error']:.6f}  "
                  f"mean_err={r['vanilla_mean_error']:.6f}")
        else:
            print(f"  error: {r.get('vanilla_error', 'unknown')}")

    # === Performance Tests ===
    print()
    print("=" * 60)
    print("PERFORMANCE: Tree mask overhead at decode-time shapes")
    print("=" * 60)

    for prefix in [128, 2048, 8192]:
        r = bench_tree_attention(tree, n_nodes, prefix_len=prefix)
        print(f"\n  Prefix={prefix}, Tree={n_nodes} nodes:")
        print(f"    SDPA (tree mask):   {r['sdpa_tree_median_ms']:>8.3f} ms")
        print(f"    SDPA (no mask):     {r['sdpa_nomask_median_ms']:>8.3f} ms  "
              f"(tree overhead: {r.get('tree_mask_overhead', '?')}x)")
        print(f"    Vanilla (tree mask):{r['vanilla_tree_median_ms']:>8.3f} ms  "
              f"(fallback cost: {r.get('vanilla_fallback_ratio', '?')}x vs fused)")

    # === Verdict ===
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    # Run final correctness check
    r = test_sdpa_tree_mask(tree, n_nodes, prefix_len=128)
    if r["sdpa_supported"] and r["sdpa_verdict"] == "PASS":
        print("  mx.fast.scaled_dot_product_attention: SUPPORTS tree masks ✓")
        print("  EAGLE-2 can use the fused SDPA path for verification.")
        overhead = bench_tree_attention(tree, n_nodes, prefix_len=2048)
        oh = overhead.get("tree_mask_overhead", 0)
        if oh < 2.0:
            print(f"  Tree mask overhead: {oh}x — acceptable for spec decoding.")
            print("  RECOMMENDATION: EAGLE-2 is FEASIBLE on MLX.")
        else:
            print(f"  Tree mask overhead: {oh}x — may limit spec decoding gains.")
            print("  RECOMMENDATION: EAGLE-2 MARGINAL — tree overhead erodes "
                  "the draft acceptance gains.")
    else:
        print("  mx.fast.scaled_dot_product_attention: REJECTS tree masks ✗")
        if r["vanilla_supported"]:
            fb = bench_tree_attention(tree, n_nodes, prefix_len=2048)
            ratio = fb.get("vanilla_fallback_ratio", 0)
            print(f"  Vanilla fallback cost: {ratio}x slower than fused SDPA")
            print("  RECOMMENDATION: EAGLE-2 feasible only with vanilla attention")
            print("  (expect ~1.5-3x verification overhead vs causal)")
        else:
            print("  Both paths FAILED — tree attention not feasible on MLX")


def n_branches_label(tree):
    """Human-readable tree shape."""
    depths = [len(a) - 1 for a in tree]
    by_depth = {}
    for d in depths:
        by_depth[d] = by_depth.get(d, 0) + 1
    parts = [f"d{d}={n}" for d, n in sorted(by_depth.items())]
    return ", ".join(parts)


if __name__ == "__main__":
    main()
