# SPDX-License-Identifier: Apache-2.0
"""RULER synthetic long-context evaluation tasks.

Implements task generators from RULER (arXiv:2404.06654):
  - Multi-key NIAH: hide N key-value pairs, retrieve all values
  - Variable tracking: follow chains of variable reassignment
  - Frequent-word aggregation: identify most frequent word

All generators produce (context, question, expected_answer) triples
at arbitrary token lengths, making them composable with any bench harness.
"""

from omlx.eval.ruler.tasks import (
    generate_multi_key_niah,
    generate_variable_tracking,
    generate_frequent_word,
    RULER_QUICK_SUITE,
    RULER_FULL_SUITE,
)

__all__ = [
    "generate_multi_key_niah",
    "generate_variable_tracking",
    "generate_frequent_word",
    "RULER_QUICK_SUITE",
    "RULER_FULL_SUITE",
]
