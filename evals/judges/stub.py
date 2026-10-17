"""Offline stub judge for tests.

Deterministic, keyword-based scoring with zero network access and no API
keys. Never use in production; it exists so the entire test suite runs
offline and stays fast.
"""

from __future__ import annotations

import re
from typing import Optional

from evals.judges.base import Judge, Judgment
from evals.judges.rubric import RubricCriterion

_WORD_RE = re.compile(r"[a-z0-9]+")


def _keywords(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _coverage(response: str, reference: str) -> float:
    """Fraction of reference keywords present in the response (0..1)."""
    ref_words = _keywords(reference)
    if not ref_words:
        return 1.0 if not _keywords(response) else 0.0
    resp_words = _keywords(response)
    return len(ref_words & resp_words) / len(ref_words)


class OfflineStubJudge:
    """Deterministic stand-in for an LLM judge.

    Scoring rules (all deterministic for identical inputs):
    - ``judge``: keyword coverage of the reference, scaled to 0..1. With no
      reference, scores 1.0 for a non-empty response and 0.0 for an empty one.
    - ``rubric_scorer``: maps coverage to the 1..5 rubric scale.
    - ``pairwise_prefer``: the response with higher reference coverage wins;
      ties when coverage is equal.
    """

    def judge(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        **kwargs: object,
    ) -> Judgment:
        if reference is None:
            score = 1.0 if response.strip() else 0.0
            label = "pass" if score == 1.0 else "fail"
        else:
            score = _coverage(response, reference)
            label = "pass" if score >= 0.5 else "fail"
        return Judgment(
            score=score,
            label=label,
            rationale=f"offline stub: keyword coverage {score:.2f}",
            metadata={"judge": "offline-stub"},
        )

    def rubric_scorer(
        self,
        criterion: RubricCriterion,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
    ) -> int:
        if reference is None:
            return 5 if response.strip() else 1
        coverage = _coverage(response, reference)
        # Map [0,1] coverage onto the 1..5 rubric scale.
        return min(5, max(1, int(round(1 + coverage * 4))))

    def pairwise_prefer(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        reference: Optional[str] = None,
    ) -> str:
        if reference is None:
            # Deterministic fallback: longer non-empty response wins.
            a, b = bool(response_a.strip()), bool(response_b.strip())
            if a and not b:
                return "a"
            if b and not a:
                return "b"
            return "tie" if len(response_a) == len(response_b) else (
                "a" if len(response_a) > len(response_b) else "b"
            )
        cov_a = _coverage(response_a, reference)
        cov_b = _coverage(response_b, reference)
        if cov_a == cov_b:
            return "tie"
        return "a" if cov_a > cov_b else "b"


# Type-check: the stub satisfies the Judge protocol.
assert isinstance(OfflineStubJudge(), Judge)
