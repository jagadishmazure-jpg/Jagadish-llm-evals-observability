"""Pairwise judge: pick a winner between two responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

Winner = Literal["a", "b", "tie"]

# Callable returning which response is preferred.
PreferenceFn = Callable[[str, str, str, Optional[str]], Winner]


@dataclass(frozen=True)
class PairwiseResult:
    winner: Winner
    rationale: str = ""
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if self.winner not in ("a", "b", "tie"):
            raise ValueError(f"winner must be 'a', 'b' or 'tie', got {self.winner!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in 0..1")


class PairwiseJudge:
    """Compares two responses and declares a winner.

    The ``prefer`` callable does the actual comparison (an LLM call in
    production, :meth:`OfflineStubJudge.pairwise_prefer` in tests).
    """

    def __init__(self, prefer: PreferenceFn) -> None:
        self._prefer = prefer

    def judge(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        reference: Optional[str] = None,
    ) -> PairwiseResult:
        winner = self._prefer(prompt, response_a, response_b, reference)
        if winner not in ("a", "b", "tie"):
            raise ValueError(
                f"preference function must return 'a', 'b' or 'tie', got {winner!r}"
            )
        return PairwiseResult(winner=winner)
