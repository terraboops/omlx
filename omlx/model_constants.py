# SPDX-License-Identifier: Apache-2.0
"""Central model identifier registry.

Before this module, the target model was hard-coded in 10+ places
(hypercar_server, agentic, tq_calibrate, ttt, every bench script).
Migrating to a new target meant grep-and-replace across the repo.

Import from here so a future migration (Task 253, Qwen3-Coder → Qwen3.6)
is a one-line edit.

Convention: call sites should default to `DEFAULT_MODEL_ID`.
CLI flags named `--model` still allow per-invocation overrides
(the flag layer is untouched by this centralization).
"""

# Previous target (Qwen3-Coder-30B family). Kept for backwards-compat:
# the bench helpers handle this architecture cleanly via the same paths
# used for Qwen3.6, and CLI overrides via `--model` work as before.
MODEL_QWEN3_CODER_30B_8BIT = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
MODEL_QWEN3_CODER_30B_4BIT = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"

# Current target (Task 253 — bench-validated 2026-05-02; ALL GATES PASSED).
# Beats Qwen3-Coder-8bit on LCB (+25 pp), MMLU-Pro (+9 pp), matches on
# HumanEval/NIAH/RULER, runs 8 GB lighter at peak. Native 262K, validated
# to 512K with YaRN factor=2 + int4 KV + adaptive prefill controller.
# Architecture class: Qwen3_5MoeForConditionalGeneration — hybrid 30 SSM
# + 10 full-attention layers (every 4th is attention, head_dim=256).
MODEL_QWEN36_35B_4BIT = "mlx-community/Qwen3.6-35B-A3B-4bit"
MODEL_QWEN36_35B_8BIT_UNSLOTH = "unsloth/Qwen3.6-35B-A3B-MLX-8bit"

# The single source of truth. Update this (and only this) when flipping
# the target. Call sites should read it rather than hard-coding a string.
DEFAULT_MODEL_ID = MODEL_QWEN36_35B_4BIT

# For the `--model` default on hypercar_server (historically the 8-bit,
# per Task 128 fix). Matches DEFAULT_MODEL_ID today — exposed separately
# so a future split (server runs fast 4-bit; bench holds 8-bit) doesn't
# require re-plumbing.
SERVER_DEFAULT_MODEL_ID = DEFAULT_MODEL_ID
BENCH_DEFAULT_MODEL_ID = DEFAULT_MODEL_ID

# Agentic flows (`omlx/agentic.py`, `bench/opencode_bench.py`) historically
# default to the 4-bit variant for faster turn-around when running
# multi-step traces. Exposed separately so flipping to Qwen3.6 doesn't
# entangle 4-bit/8-bit choices with the main bench target.
AGENTIC_DEFAULT_MODEL_ID = MODEL_QWEN3_CODER_30B_4BIT

# Calibration scripts (`omlx/tq_calibrate.py`, `omlx/ttt.py`,
# `bench/profile_prefill.py`) — 4-bit by default; the codec/calibration
# math is bit-width-agnostic, but smaller weights make iteration faster.
CALIBRATION_DEFAULT_MODEL_ID = MODEL_QWEN3_CODER_30B_4BIT
