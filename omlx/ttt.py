# SPDX-License-Identifier: Apache-2.0
"""Test-Time Training (TTT) for hypercar agentic loop.

Implements online fine-tuning during inference using the "fast weights"
approach from In-Place TTT (arXiv:2604.06169). Key design:

  1. Freeze all base model weights (TQ3.5 or standard quantized)
  2. Create small "fast weight" adapters on MLP output projections
  3. Train ONLY these adapters using execution feedback
  4. Fork/rewind via the agentic session API

Memory budget:
  - Fast weights per layer: rank × hidden_dim × 2 (up + down) × 4 bytes
  - At rank=16, hidden_dim=4096: 16 × 4096 × 2 × 4 = 512KB per layer
  - 48 layers: ~24MB total (negligible vs 17GB model)
  - Gradients: same size = ~24MB
  - Total TTT overhead: ~50MB

The approach:
  1. Generate N candidate completions (temperature > 0)
  2. Execute each against compiler/test suite
  3. Use passing candidates as pseudo-labels
  4. Compute cross-entropy loss on the passing trajectory
  5. Update fast weights via micro-batch gradient descent
  6. mx.eval() after each update to prevent graph hoarding

Usage:
    from omlx.ttt import TTTAdapter, ttt_step

    adapter = TTTAdapter(model, rank=16)
    # Generate candidates
    candidates = [generate(model, prompt, temp=0.8) for _ in range(4)]
    # Test them
    results = [execute_code(c) for c in candidates]
    # Train on passing ones
    for c, r in zip(candidates, results):
        if r.passed:
            ttt_step(model, adapter, tokenizer, prompt + c, lr=1e-4)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger("omlx.ttt")


class FastWeightAdapter(nn.Module):
    """Low-rank adapter for MLP output projections (fast weights).

    Adds a trainable LoRA-style bypass: y = base_output + x @ A @ B
    where A is (in_dim, rank) and B is (rank, out_dim).

    Only A and B are trainable. The base model is frozen.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        # Initialize A with small random values, B with zeros
        # This means the adapter starts as identity (no effect)
        scale = 1.0 / (in_dim ** 0.5)
        self.A = mx.random.normal((in_dim, rank)) * scale
        self.B = mx.zeros((rank, out_dim))

    def __call__(self, x: mx.array) -> mx.array:
        """Compute adapter output: x @ A @ B."""
        return x @ self.A @ self.B


class TTTAdapter:
    """Test-Time Training adapter manager.

    Creates and manages fast weight adapters for each MLP layer.
    Provides methods for training and applying the adapters.
    """

    def __init__(self, model: nn.Module, rank: int = 16,
                 target_modules: Optional[List[str]] = None):
        """Create adapters for target MLP modules.

        Args:
            model: The frozen base model.
            rank: LoRA rank for fast weights.
            target_modules: Module name patterns to target.
                Default: ["down_proj"] (MLP output projection).
        """
        self.rank = rank
        self.adapters: Dict[str, FastWeightAdapter] = {}
        self.target_modules = target_modules or ["down_proj"]

        # Walk model tree and create adapters
        for name, module in model.named_modules():
            leaf = name.split(".")[-1] if name else ""
            if leaf in self.target_modules:
                if hasattr(module, 'weight'):
                    # Get dimensions from the module
                    if hasattr(module, 'input_dims'):
                        in_dim = module.input_dims
                        out_dim = module.output_dims
                    elif hasattr(module, 'scales'):
                        # QuantizedLinear: infer from scales
                        in_dim = module.scales.shape[-1] * module.group_size
                        out_dim = module.scales.shape[-2]
                    else:
                        w = module.weight
                        out_dim, in_dim = w.shape[0], w.shape[-1]

                    self.adapters[name] = FastWeightAdapter(in_dim, out_dim, rank)

        # Compute memory usage
        total_params = sum(
            a.A.size + a.B.size for a in self.adapters.values()
        )
        total_bytes = total_params * 4  # float32
        logger.info(
            f"TTT: created {len(self.adapters)} adapters "
            f"(rank={rank}, {total_bytes/1e6:.1f}MB)"
        )

    def parameters(self) -> List[mx.array]:
        """Return all trainable adapter parameters."""
        params = []
        for adapter in self.adapters.values():
            params.extend([adapter.A, adapter.B])
        return params

    def named_parameters(self) -> List[Tuple[str, mx.array]]:
        """Return named trainable parameters."""
        result = []
        for name, adapter in self.adapters.items():
            result.append((f"{name}.A", adapter.A))
            result.append((f"{name}.B", adapter.B))
        return result

    def apply(self, model: nn.Module):
        """Hook adapters into model forward pass.

        Monkey-patches each target module's __call__ to add the adapter
        output on top of the base computation.
        """
        for name, adapter in self.adapters.items():
            # Navigate to the module
            parts = name.split(".")
            module = model
            for p in parts:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)

            # Store adapter reference and patch __call__
            module._ttt_adapter = adapter
            if not hasattr(module, '_ttt_orig_call'):
                module._ttt_orig_call = module.__call__

                def make_patched(mod):
                    def patched_call(x):
                        base_out = mod._ttt_orig_call(x)
                        if hasattr(mod, '_ttt_adapter'):
                            return base_out + mod._ttt_adapter(x)
                        return base_out
                    return patched_call

                module.__call__ = make_patched(module)

        logger.info(f"TTT: applied {len(self.adapters)} adapters to model")

    def remove(self, model: nn.Module):
        """Remove adapters from model (restore original forward pass)."""
        for name in self.adapters:
            parts = name.split(".")
            module = model
            for p in parts:
                if p.isdigit():
                    module = module[int(p)]
                else:
                    module = getattr(module, p)

            if hasattr(module, '_ttt_orig_call'):
                module.__call__ = module._ttt_orig_call
                del module._ttt_orig_call
            if hasattr(module, '_ttt_adapter'):
                del module._ttt_adapter

    def reset(self):
        """Reset all adapters to zero (fresh start)."""
        for adapter in self.adapters.values():
            adapter.B = mx.zeros_like(adapter.B)

    def fork(self) -> "TTTAdapter":
        """Create an independent copy of all adapters."""
        import copy
        new = copy.copy(self)
        new.adapters = {}
        for name, adapter in self.adapters.items():
            new_adapter = FastWeightAdapter.__new__(FastWeightAdapter)
            new_adapter.rank = adapter.rank
            new_adapter.A = mx.array(adapter.A)
            new_adapter.B = mx.array(adapter.B)
            new.adapters[name] = new_adapter
        return new


def ttt_loss(model, tokenizer, text: str, cache=None) -> mx.array:
    """Compute cross-entropy loss on a text sequence.

    Uses teacher forcing: predict each token given all previous tokens.
    The loss is computed on the COMPLETION tokens only (not the prompt).

    Args:
        model: The model (with adapters applied).
        tokenizer: The tokenizer.
        text: The full text (prompt + completion).
        cache: Optional pre-filled KV cache.

    Returns:
        Scalar loss value.
    """
    tokens = mx.array([tokenizer.encode(text)])
    # Forward pass
    logits = model(tokens, cache=cache)
    # Shift for next-token prediction
    # logits[:, :-1] predicts tokens[:, 1:]
    shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = tokens[:, 1:].reshape(-1)
    # Cross-entropy loss
    loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
    return loss


def ttt_step(
    model: nn.Module,
    adapter: TTTAdapter,
    optimizer: optim.Optimizer,
    tokenizer: Any,
    text: str,
    cache=None,
) -> float:
    """One TTT gradient step on a text sequence.

    Computes loss, backpropagates through adapter parameters only,
    updates via optimizer, and evaluates immediately to prevent
    graph hoarding.

    Args:
        model: Model with adapters applied.
        adapter: The TTTAdapter managing fast weights.
        optimizer: MLX optimizer (e.g., AdamW).
        tokenizer: Tokenizer.
        text: Training text (passing code solution).
        cache: Optional KV cache.

    Returns:
        Loss value (float).
    """
    # Compute loss and gradients w.r.t. adapter parameters
    loss_fn = lambda params: ttt_loss(model, tokenizer, text, cache)

    # Use value_and_grad on the adapter parameters
    loss, grads = nn.value_and_grad(model, loss_fn)(model.parameters())

    # Update only adapter parameters
    optimizer.update(model, grads)

    # CRITICAL: evaluate immediately to prevent graph hoarding
    mx.eval(loss)
    mx.eval(adapter.parameters())

    return float(loss.item())
