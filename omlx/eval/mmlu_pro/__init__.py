# SPDX-License-Identifier: Apache-2.0
"""MMLU-Pro reasoning evaluation (arXiv:2406.01574).

Loads TIGER-Lab/MMLU-Pro from HuggingFace, formats chain-of-thought
prompts, and extracts answer letters for scoring.
"""

from omlx.eval.mmlu_pro.tasks import load_mmlu_pro, format_prompt, extract_answer

__all__ = ["load_mmlu_pro", "format_prompt", "extract_answer"]
