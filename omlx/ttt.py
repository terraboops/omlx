# SPDX-License-Identifier: Apache-2.0
"""Test-Time Training (TTT) for hypercar agentic loop.

Online fine-tuning during inference using fast weight adapters.
Three feedback signals:
  1. Tool call success/fail — deterministic, train on successful trajectories
  2. Harness rating — human or CI feedback, DPO-style preference learning
  3. Solution discovery — verifiable outcomes (tests pass, code compiles)

Memory budget: ~50MB total (adapters + gradients + optimizer state).
The 17GB base model stays frozen.

Usage:
    from omlx.ttt import TTTEngine

    engine = TTTEngine(model, tokenizer, rank=16)

    # Generate candidates
    candidates = engine.generate_candidates(prompt, n=4, temp=0.8)

    # Provide feedback
    for c in candidates:
        result = execute_code(c.code)
        engine.feedback(c.id, reward=1.0 if result.passed else -1.0,
                       signal="tool_call")

    # Train on positive feedback
    stats = engine.train_step()

    # If quality degrades, rewind
    engine.rewind()
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

logger = logging.getLogger("omlx.ttt")


# ---------------------------------------------------------------------------
# Fast Weight Adapter
# ---------------------------------------------------------------------------

class FastWeightAdapter(nn.Module):
    """Low-rank adapter: y = base_output + x @ A @ B.

    A: (in_dim, rank) — initialized with small random values
    B: (rank, out_dim) — initialized to zeros (starts as no-op)
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int = 16):
        super().__init__()
        self.rank = rank
        scale = 0.01 / (in_dim ** 0.5)
        self.A = scale * mx.random.normal((in_dim, rank))
        self.B = mx.zeros((rank, out_dim))

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.A @ self.B


# ---------------------------------------------------------------------------
# Feedback Signals
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A generated candidate with feedback."""
    id: str
    prompt: str
    completion: str
    tokens: List[int] = field(default_factory=list)
    reward: float = 0.0
    signal: str = ""  # "tool_call", "harness", "solution"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainStats:
    """Statistics from a training step."""
    loss: float = 0.0
    grad_norm: float = 0.0
    num_positive: int = 0
    num_negative: int = 0
    elapsed_s: float = 0.0
    adapter_norm: float = 0.0


# ---------------------------------------------------------------------------
# Code Execution Verifier
# ---------------------------------------------------------------------------

def verify_code(code: str, test: str = "", timeout: int = 10) -> Tuple[bool, str]:
    """Execute Python code and optionally run a test assertion.

    Returns (passed, error_message).
    Only trains on VERIFIABLY correct code — never partial success.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        if test:
            f.write(f"\n\n# Test\n{test}\nprint('PASS')\n")
        f.flush()
        try:
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            passed = result.returncode == 0
            if test:
                passed = passed and "PASS" in result.stdout
            error = result.stderr[:200] if not passed else ""
            return passed, error
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, str(e)[:200]
        finally:
            Path(f.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TTT Engine
# ---------------------------------------------------------------------------

class TTTEngine:
    """Test-Time Training engine with three feedback signals.

    Manages the full cycle: generate → execute → feedback → train → verify.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        rank: int = 16,
        lr: float = 1e-4,
        target_modules: Optional[List[str]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.rank = rank

        # Create adapters
        self.target_modules = target_modules or ["down_proj"]
        self.adapters: Dict[str, FastWeightAdapter] = {}
        self._create_adapters()

        # Optimizer for adapter parameters only
        self.optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)

        # Feedback buffer
        self.candidates: Dict[str, Candidate] = {}
        self.history: List[TrainStats] = []

        # Checkpoint for rewind
        self._checkpoint: Optional[Dict[str, Tuple[mx.array, mx.array]]] = None
        self.save_checkpoint()

    def _create_adapters(self):
        """Walk model tree and create adapters for target modules."""
        for name, module in self.model.named_modules():
            leaf = name.split(".")[-1] if name else ""
            if leaf not in self.target_modules:
                continue

            # Infer dimensions
            if hasattr(module, 'scales'):
                in_dim = module.scales.shape[-1] * module.group_size
                out_dim = module.scales.shape[-2]
            elif hasattr(module, 'weight'):
                w = module.weight
                out_dim, in_dim = w.shape[0], w.shape[-1]
            else:
                continue

            self.adapters[name] = FastWeightAdapter(in_dim, out_dim, self.rank)

        # Apply adapters to model
        self._patch_model()

        total_params = sum(a.A.size + a.B.size for a in self.adapters.values())
        logger.info(f"TTT: {len(self.adapters)} adapters, {total_params * 4 / 1e6:.1f}MB")

    def _patch_model(self):
        """Monkey-patch target modules to add adapter bypass."""
        for name, adapter in self.adapters.items():
            parts = name.split(".")
            module = self.model
            for p in parts:
                module = module[int(p)] if p.isdigit() else getattr(module, p)

            module._ttt_adapter = adapter
            if not hasattr(module, '_ttt_orig_call'):
                orig = module.__call__

                def make_patched(mod, orig_fn):
                    def patched(x):
                        out = orig_fn(x)
                        if hasattr(mod, '_ttt_adapter'):
                            out = out + mod._ttt_adapter(x)
                        return out
                    return patched

                module.__call__ = make_patched(module, orig)
                module._ttt_orig_call = orig

    # ----- Generation -----

    def generate_candidates(
        self, prompt: str, n: int = 4, max_tokens: int = 256,
        temperature: float = 0.8,
    ) -> List[Candidate]:
        """Generate N diverse candidate completions."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=temperature)
        candidates = []
        for _ in range(n):
            text = generate(
                self.model, self.tokenizer,
                prompt=prompt, max_tokens=max_tokens,
                sampler=sampler, verbose=False,
            )
            cid = str(uuid.uuid4())[:8]
            c = Candidate(
                id=cid, prompt=prompt, completion=text,
                tokens=self.tokenizer.encode(text),
            )
            candidates.append(c)
            self.candidates[cid] = c

        return candidates

    # ----- Feedback -----

    def feedback(self, candidate_id: str, reward: float, signal: str = "tool_call",
                 metadata: Optional[Dict] = None):
        """Provide feedback on a candidate.

        Args:
            candidate_id: ID from generate_candidates.
            reward: +1.0 for success, -1.0 for failure, 0-5 for rating.
            signal: "tool_call", "harness", or "solution".
            metadata: Optional extra info (error message, test output, etc).
        """
        if candidate_id not in self.candidates:
            logger.warning(f"Unknown candidate: {candidate_id}")
            return

        c = self.candidates[candidate_id]
        c.reward = reward
        c.signal = signal
        if metadata:
            c.metadata.update(metadata)

        logger.info(f"TTT feedback: {candidate_id} reward={reward} signal={signal}")

    def feedback_from_execution(self, candidate_id: str, test: str = ""):
        """Auto-feedback by executing the candidate's code.

        Only rewards VERIFIABLY correct code (compiles + tests pass).
        """
        c = self.candidates.get(candidate_id)
        if not c:
            return

        passed, error = verify_code(c.completion, test=test)
        c.reward = 1.0 if passed else -1.0
        c.signal = "solution" if test else "tool_call"
        c.metadata["passed"] = passed
        c.metadata["error"] = error

        logger.info(f"TTT exec: {candidate_id} {'PASS' if passed else 'FAIL'} {error[:50]}")

    # ----- Training -----

    def train_step(self) -> TrainStats:
        """Train on positively-rewarded candidates.

        Only uses candidates with reward > 0. Computes cross-entropy loss
        on each positive trajectory and updates adapter parameters.

        Returns training statistics.
        """
        t0 = time.perf_counter()
        positive = [c for c in self.candidates.values() if c.reward > 0]
        negative = [c for c in self.candidates.values() if c.reward <= 0]

        if not positive:
            logger.info("TTT: no positive candidates, skipping training")
            return TrainStats(num_negative=len(negative))

        total_loss = 0.0

        for c in positive:
            full_text = c.prompt + c.completion.split("\n\n")[0]
            tokens = mx.array([self.tokenizer.encode(full_text)])

            prompt_len = len(self.tokenizer.encode(c.prompt))
            if prompt_len >= tokens.shape[1] - 1:
                continue

            # Build loss function that closes over tokens
            def loss_fn(adapter_params):
                # Set adapter parameters from the flat list
                idx = 0
                for adapter in self.adapters.values():
                    adapter.A = adapter_params[idx]
                    adapter.B = adapter_params[idx + 1]
                    idx += 2

                logits = self.model(tokens)
                shift_logits = logits[:, prompt_len:-1, :].reshape(-1, logits.shape[-1])
                shift_labels = tokens[:, prompt_len+1:].reshape(-1)
                return nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")

            # Collect adapter parameters
            params = []
            for adapter in self.adapters.values():
                params.extend([adapter.A, adapter.B])

            # Compute loss and gradients w.r.t adapter params only
            loss_and_grad = mx.value_and_grad(loss_fn)
            loss_val, grads = loss_and_grad(params)

            # SGD update on adapter params
            lr = self.optimizer.learning_rate
            idx = 0
            for adapter in self.adapters.values():
                adapter.A = adapter.A - lr * grads[idx]
                adapter.B = adapter.B - lr * grads[idx + 1]
                idx += 2

            total_loss += float(loss_val.item())

            # CRITICAL: eval immediately to prevent graph hoarding
            mx.eval(loss_val)
            for adapter in self.adapters.values():
                mx.eval(adapter.A, adapter.B)

        # Compute adapter norm for monitoring
        adapter_norm = sum(
            float(mx.sum(a.A ** 2).item()) + float(mx.sum(a.B ** 2).item())
            for a in self.adapters.values()
        ) ** 0.5

        elapsed = time.perf_counter() - t0
        stats = TrainStats(
            loss=total_loss / max(len(positive), 1),
            num_positive=len(positive),
            num_negative=len(negative),
            elapsed_s=elapsed,
            adapter_norm=adapter_norm,
        )
        self.history.append(stats)

        logger.info(
            f"TTT step: loss={stats.loss:.4f} "
            f"+{stats.num_positive}/-{stats.num_negative} "
            f"norm={stats.adapter_norm:.4f} ({stats.elapsed_s:.2f}s)"
        )

        # Clear processed candidates
        self.candidates.clear()

        return stats

    # ----- Checkpoint / Rewind -----

    def save_checkpoint(self):
        """Save current adapter state for rewind."""
        self._checkpoint = {
            name: (mx.array(a.A), mx.array(a.B))
            for name, a in self.adapters.items()
        }
        logger.info("TTT: checkpoint saved")

    def rewind(self):
        """Rewind adapters to last checkpoint."""
        if self._checkpoint is None:
            logger.warning("TTT: no checkpoint to rewind to")
            return
        for name, (a_saved, b_saved) in self._checkpoint.items():
            if name in self.adapters:
                self.adapters[name].A = mx.array(a_saved)
                self.adapters[name].B = mx.array(b_saved)
        self.candidates.clear()
        logger.info("TTT: rewound to checkpoint")

    def reset(self):
        """Reset all adapters to zero (fresh start, no learning)."""
        for adapter in self.adapters.values():
            adapter.B = mx.zeros_like(adapter.B)
        self.candidates.clear()
        self.history.clear()
        self._checkpoint = None
        logger.info("TTT: reset to zero")

    # ----- Diagnostics -----

    def stats(self) -> Dict[str, Any]:
        """Return current TTT state."""
        return {
            "num_adapters": len(self.adapters),
            "adapter_memory_mb": sum(
                (a.A.size + a.B.size) * 4 for a in self.adapters.values()
            ) / 1e6,
            "total_train_steps": len(self.history),
            "pending_candidates": len(self.candidates),
            "has_checkpoint": self._checkpoint is not None,
            "last_loss": self.history[-1].loss if self.history else None,
        }


# ---------------------------------------------------------------------------
# Demo: self-improving code generation
# ---------------------------------------------------------------------------

def demo_ttt(model=None, tokenizer=None):
    """Demo: generate factorial, test it, train on passing solutions."""
    if model is None:
        from mlx_lm import load
        model, tokenizer = load("mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit")

    engine = TTTEngine(model, tokenizer, rank=16, lr=1e-4)
    print(f"TTT Engine: {engine.stats()}")

    prompt = "def factorial(n):\n    if n <= 1: return 1\n    return n *"
    test = "assert factorial(5) == 120 and factorial(0) == 1"

    for round_num in range(3):
        print(f"\n=== Round {round_num + 1} ===")

        # Generate candidates — enough tokens to complete the function
        candidates = engine.generate_candidates(prompt, n=4, max_tokens=30, temperature=0.8)

        # Execute and provide feedback
        for c in candidates:
            # Build executable code: prompt + completion (stop at double newline)
            code = c.prompt + c.completion.split("\n\n")[0].split("\ndef ")[0]
            passed, error = verify_code(code, test=test)
            c.reward = 1.0 if passed else -1.0
            c.signal = "solution"
            c.metadata["passed"] = passed
            c.metadata["error"] = error
            status = "PASS" if passed else "FAIL"
            print(f"  {c.id}: {status} — {code.strip()[:80]!r}")

        # Train on successful ones
        stats = engine.train_step()
        print(f"  Train: loss={stats.loss:.4f} +{stats.num_positive}/-{stats.num_negative}")

        # Checkpoint after each round
        engine.save_checkpoint()

    print(f"\nFinal stats: {engine.stats()}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    demo_ttt()
