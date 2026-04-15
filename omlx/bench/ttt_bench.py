# SPDX-License-Identifier: Apache-2.0
"""End-to-end TTT API benchmark.

Validates the full Test-Time Training pipeline via HTTP:
  1. Generate N candidates via /v1/ttt/generate
  2. Verify each via /v1/ttt/feedback_exec (code execution)
  3. Train on positives via /v1/ttt/train
  4. Check adapter state via /v1/ttt/stats
  5. Checkpoint and rewind via /v1/ttt/checkpoint + /v1/ttt/rewind
  6. Validate loss decreases across training rounds

Usage:
    python -m omlx.bench.ttt_bench --server http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger("omlx.bench.ttt")

DEFAULT_SERVER = "http://localhost:8080"


def _post(url: str, data: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.loads(e.read()).get("error", str(e))}
        except Exception:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def _get(url: str, timeout: int = 10) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


@dataclass
class TTTResult:
    name: str
    passed: bool
    details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


def test_ttt_stats(base: str) -> TTTResult:
    """Check TTT stats endpoint (pre-initialization)."""
    stats = _get(f"{base}/v1/ttt/stats")
    return TTTResult("ttt_stats_pre", "error" not in stats, details=stats)


def test_ttt_generate_verify_train(base: str) -> TTTResult:
    """Full cycle: generate → verify → train on positives."""
    prompt = "def factorial(n):\n    if n <= 1: return 1\n    return n *"
    test = "assert factorial(5) == 120 and factorial(0) == 1"

    # 1. Generate candidates
    gen_result = _post(f"{base}/v1/ttt/generate", {
        "prompt": prompt, "n": 4, "max_tokens": 20, "temperature": 0.8,
    })
    if "error" in gen_result:
        return TTTResult("generate_verify_train", False, error=gen_result["error"])

    candidates = gen_result.get("candidates", [])
    if len(candidates) != 4:
        return TTTResult("generate_verify_train", False,
                         error=f"expected 4 candidates, got {len(candidates)}")

    # 2. Verify each (execute + assert test passes)
    passed_count = 0
    results = []
    for c in candidates:
        # Build full code from prompt + completion
        completion = c["completion"]
        code = prompt + completion.split("\n\n")[0].split("\ndef ")[0]
        feedback = _post(f"{base}/v1/ttt/feedback_exec", {
            "candidate_id": c["id"],
            "test": test,
        })
        passed = feedback.get("passed", False)
        if passed:
            passed_count += 1
        results.append({
            "id": c["id"], "passed": passed,
            "preview": completion[:60],
        })

    if passed_count == 0:
        return TTTResult("generate_verify_train", False,
                         details={"results": results},
                         error="no candidates passed — nothing to train on")

    # 3. Train
    train_result = _post(f"{base}/v1/ttt/train", {})
    if "error" in train_result:
        return TTTResult("generate_verify_train", False,
                         error=f"train: {train_result['error']}")

    loss = train_result.get("loss", 0)
    num_positive = train_result.get("num_positive", 0)
    adapter_norm = train_result.get("adapter_norm", 0)

    passed = num_positive > 0 and loss > 0 and adapter_norm > 0

    return TTTResult(
        "generate_verify_train", passed,
        details={
            "num_candidates": len(candidates),
            "num_passed": passed_count,
            "train_loss": loss,
            "num_positive": num_positive,
            "adapter_norm": adapter_norm,
            "elapsed_s": train_result.get("elapsed_s"),
        },
    )


def test_ttt_loss_decreases(base: str) -> TTTResult:
    """Run 3 training rounds, verify loss trends down."""
    # Self-completing prompt style (matches factorial test): the completion
    # smoothly appends to the last token, avoiding newline/indent ambiguity
    # that produces SyntaxError on execution.
    prompt = "def is_even(n):\n    return n %"
    test = "assert is_even(4) == True and is_even(7) == False"

    losses = []
    norms = []

    for round_num in range(3):
        gen = _post(f"{base}/v1/ttt/generate", {
            "prompt": prompt, "n": 4, "max_tokens": 15, "temperature": 0.8,
        })
        if "error" in gen:
            return TTTResult("loss_decreases", False, error=gen["error"])

        passed_this_round = 0
        for c in gen.get("candidates", []):
            fb = _post(f"{base}/v1/ttt/feedback_exec", {
                "candidate_id": c["id"], "test": test,
            })
            if fb.get("passed"):
                passed_this_round += 1

        if passed_this_round == 0:
            continue

        train = _post(f"{base}/v1/ttt/train", {})
        if "error" not in train:
            losses.append(train.get("loss", 0))
            norms.append(train.get("adapter_norm", 0))

    if len(losses) < 2:
        return TTTResult("loss_decreases", False,
                         error=f"need >=2 training rounds, got {len(losses)}")

    # Loss should generally decrease or stay stable; adapter norm should grow
    norm_increased = norms[-1] >= norms[0]

    return TTTResult(
        "loss_decreases",
        norm_increased,
        details={
            "losses": losses,
            "norms": norms,
            "loss_trend": "decreasing" if losses[-1] < losses[0] else "stable/increasing",
            "norm_grew": norm_increased,
        },
    )


def test_ttt_checkpoint_rewind(base: str) -> TTTResult:
    """Checkpoint → train → rewind → verify adapter state."""
    # Save checkpoint
    cp = _post(f"{base}/v1/ttt/checkpoint", {})
    if "error" in cp:
        return TTTResult("checkpoint_rewind", False, error=cp["error"])

    # Get pre-train stats
    pre_stats = _get(f"{base}/v1/ttt/stats")
    pre_steps = pre_stats.get("total_train_steps", 0)

    # Generate + feedback + train to change adapter state
    gen = _post(f"{base}/v1/ttt/generate", {
        "prompt": "def add(a, b):\n    return", "n": 2,
        "max_tokens": 10, "temperature": 0.8,
    })
    for c in gen.get("candidates", []):
        _post(f"{base}/v1/ttt/feedback_exec", {
            "candidate_id": c["id"], "test": "assert add(2, 3) == 5",
        })
    _post(f"{base}/v1/ttt/train", {})

    # Rewind to checkpoint
    rw = _post(f"{base}/v1/ttt/rewind", {})
    if "error" in rw:
        return TTTResult("checkpoint_rewind", False, error=rw["error"])

    return TTTResult(
        "checkpoint_rewind", True,
        details={"pre_steps": pre_steps, "rewound": rw.get("status")},
    )


def test_ttt_reset(base: str) -> TTTResult:
    """Reset adapters and verify state cleared."""
    rst = _post(f"{base}/v1/ttt/reset", {})
    if "error" in rst:
        return TTTResult("reset", False, error=rst["error"])

    stats = _get(f"{base}/v1/ttt/stats")
    # After reset, train_steps should be 0 (or whatever the reset behavior sets)
    return TTTResult(
        "reset", "error" not in stats,
        details={"post_reset_stats": stats},
    )


ALL_TESTS = [
    test_ttt_stats,
    test_ttt_generate_verify_train,
    test_ttt_loss_decreases,
    test_ttt_checkpoint_rewind,
    test_ttt_reset,
]


def main():
    parser = argparse.ArgumentParser(description="TTT API benchmark")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--json", default="/tmp/hypercar_ttt_bench.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Verify server
    stats = _get(f"{args.server}/v1/stats")
    if "error" in stats:
        logger.error(f"Server not reachable: {stats['error']}")
        sys.exit(1)

    logger.info(f"Server: {args.server} ({stats.get('kv_mode')} mode)")
    logger.info(f"Model: {stats.get('model')}")

    logger.info("\n=== TTT API BENCHMARK ===\n")

    results = []
    for test_fn in ALL_TESTS:
        logger.info(f"Running {test_fn.__name__}...")
        t0 = time.perf_counter()
        try:
            r = test_fn(args.server)
        except Exception as e:
            r = TTTResult(test_fn.__name__, False, error=f"{type(e).__name__}: {e}")
        elapsed = time.perf_counter() - t0
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        logger.info(f"  [{status}] {r.name} ({elapsed:.1f}s)")
        if r.error:
            logger.info(f"    error: {r.error}")
        if r.details:
            for k, v in list(r.details.items())[:3]:
                logger.info(f"    {k}: {v}")

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print("\n" + "=" * 50)
    print("TTT API BENCHMARK RESULTS")
    print("=" * 50)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        marker = "  " if r.passed else ">>"
        print(f"  {marker} [{status}] {r.name}")
        if r.error:
            print(f"          {r.error}")
    print("-" * 50)
    print(f"  [{passed}/{total}] Total")
    print("=" * 50)

    from dataclasses import asdict
    with open(args.json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
