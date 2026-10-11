"""Rubric judge: 1-5 scores per criterion, combined into a weighted score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence


@dataclass(frozen=True)
class RubricCriterion:
    """One scored dimension of a rubric."""

    name: str
    description: str = ""
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("criterion name must be non-empty")
        if self.weight <= 0:
            raise ValueError("criterion weight must be positive")


@dataclass(frozen=True)
class CriterionScore:
    """Score for a single criterion, on the 1-5 rubric scale."""

    criterion: str
    score: int
    rationale: str = ""


@dataclass(frozen=True)
class RubricResult:
    """Per-criterion scores plus the aggregate."""

    scores: tuple[CriterionScore, ...]
    weighted_score: float  # 1.0 - 5.0
    normalized_score: float  # 0.0 - 1.0

    def score_for(self, criterion: str) -> int:
        """Return the 1-5 score recorded for ``criterion``."""
        for item in self.scores:
            if item.criterion == criterion:
                return item.score
        raise KeyError(f"no score recorded for criterion {criterion!r}")


# Callable invoked per criterion; must return an int in 1..5.
Scorer = Callable[[RubricCriterion, str, str, Optional[str]], int]


class RubricJudge:
    """Scores responses against a weighted rubric of criteria.

    The ``scorer`` callable does the actual per-criterion scoring (for
    example an LLM call in production, or
    :meth:`OfflineStubJudge.rubric_scorer` in tests). This class validates
    the 1-5 scale and computes the weighted aggregate.
    """

    def __init__(self, criteria: Sequence[RubricCriterion], scorer: Scorer) -> None:
        if not criteria:
            raise ValueError("rubric requires at least one criterion")
        names = [c.name for c in criteria]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate criterion names: {names}")
        self._criteria = tuple(criteria)
        self._scorer = scorer

    @property
    def criteria(self) -> tuple[RubricCriterion, ...]:
        return self._criteria

    def judge(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
    ) -> RubricResult:
        scored: list[CriterionScore] = []
        weighted_total = 0.0
        weight_total = 0.0
        for criterion in self._criteria:
            score = self._scorer(criterion, prompt, response, reference)
            if not isinstance(score, int) or isinstance(score, bool):
                raise TypeError(
                    f"scorer must return an int in 1..5, got {score!r}"
                )
            if not 1 <= score <= 5:
                raise ValueError(
                    f"rubric scores must be in 1..5, got {score} "
                    f"for criterion {criterion.name!r}"
                )
            scored.append(CriterionScore(criterion.name, score))
            weighted_total += criterion.weight * score
            weight_total += criterion.weight
        weighted = weighted_total / weight_total
        normalized = (weighted - 1.0) / 4.0
        return RubricResult(
            scores=tuple(scored),
            weighted_score=weighted,
            normalized_score=normalized,
        )
