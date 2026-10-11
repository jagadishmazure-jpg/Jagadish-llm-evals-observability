"""Quality metrics over scored results."""

from __future__ import annotations

import math
from typing import Sequence


def accuracy(scores: Sequence[float], threshold: float = 0.5) -> float:
    """Fraction of ``scores`` at or above ``threshold``.

    Returns 0.0 for an empty sequence.
    """
    if not scores:
        return 0.0
    hits = sum(1 for s in scores if s >= threshold)
    return hits / len(scores)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021).

    Given ``n`` samples with ``c`` correct, estimates the probability that at
    least one of ``k`` draws is correct: ``1 - C(n-c, k) / C(n, k)``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0 <= c <= n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
