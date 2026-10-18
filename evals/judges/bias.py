"""Bias probes for the pairwise judge.

Two classic LLM-judge failure modes, made executable:

- **position bias**: preferring whichever answer is presented first,
  regardless of content;
- **verbosity bias**: preferring the longer answer regardless of quality.

``position_swap_probe`` runs every pair twice with the order flipped and
counts pairs where the verdict *follows the position* instead of the
content. ``verbosity_probe`` pits a longer answer against a shorter one
of equal quality and counts pairs where length wins.

A bias rate above :data:`MAX_ACCEPTABLE_BIAS_RATE` means the judge cannot
be trusted on that dimension. The probes are deterministic for identical
inputs, like every other judge in this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from evals.judges.pairwise import PairwiseJudge

#: Maximum bias rate a judge may exhibit before it is considered unfit
#: for the probed dimension. Documented here so tests and CI share one
#: definition of "systematically biased".
MAX_ACCEPTABLE_BIAS_RATE = 0.10


@dataclass(frozen=True)
class BiasPair:
    """One probe item: two responses to compare for a prompt."""

    prompt: str
    response_a: str
    response_b: str
    reference: Optional[str] = None


@dataclass(frozen=True)
class PairDetail:
    """Per-pair outcome of a probe run."""

    index: int
    first_winner: str
    second_winner: str
    biased: bool
    note: str = ""


@dataclass(frozen=True)
class BiasReport:
    """Outcome of a bias probe over a set of pairs."""

    probe: str
    pairs: int
    biased_pairs: int
    details: List[PairDetail] = field(default_factory=list)

    @property
    def bias_rate(self) -> float:
        if self.pairs == 0:
            return 0.0
        return self.biased_pairs / self.pairs

    @property
    def passes(self) -> bool:
        """True when the judge is not systematically biased."""
        return self.bias_rate <= MAX_ACCEPTABLE_BIAS_RATE

    def summary(self) -> str:
        return (
            f"{self.probe}: {self.biased_pairs}/{self.pairs} biased "
            f"(rate {self.bias_rate:.2f}, threshold "
            f"{MAX_ACCEPTABLE_BIAS_RATE:.2f}) -> "
            f"{'PASS' if self.passes else 'FAIL'}"
        )


def _content_winner(first: str, second: str) -> Optional[str]:
    """Map the two ordered verdicts back to content identity.

    Returns "a" if both runs picked response_a's content, "b" if both
    picked response_b's content, "tie" if both tied, else None when the
    verdicts are inconsistent (neither content- nor position-stable).
    """
    if first == "tie" and second == "tie":
        return "tie"
    if first == "a" and second == "b":
        return "a"
    if first == "b" and second == "a":
        return "b"
    return None


def position_swap_probe(
    judge: PairwiseJudge, pairs: List[BiasPair]
) -> BiasReport:
    """Detect position bias: verdicts that follow the slot, not the content.

    Each pair is judged twice, order flipped. A pair counts as biased when
    the judge picks the first-presented answer in *both* runs (or the
    second in both) — i.e. the winner tracks position rather than content.
    """
    details: List[PairDetail] = []
    biased = 0
    for i, pair in enumerate(pairs):
        first = judge.judge(
            pair.prompt, pair.response_a, pair.response_b, pair.reference
        ).winner
        second = judge.judge(
            pair.prompt, pair.response_b, pair.response_a, pair.reference
        ).winner
        follows_position = (first == "a" and second == "a") or (
            first == "b" and second == "b"
        )
        content = _content_winner(first, second)
        if follows_position:
            biased += 1
            note = "verdict followed position, not content"
        elif content is None:
            note = "inconsistent verdicts across orders"
        else:
            note = f"content-stable: {content}"
        details.append(
            PairDetail(index=i, first_winner=first,
                       second_winner=second, biased=follows_position,
                       note=note)
        )
    return BiasReport(probe="position-swap", pairs=len(pairs),
                      biased_pairs=biased, details=details)


def verbosity_probe(judge: PairwiseJudge, pairs: List[BiasPair]) -> BiasReport:
    """Detect verbosity bias: the longer answer winning on length alone.

    Each pair should hold quality constant (equal reference coverage)
    while varying length. A pair counts as biased when the longer
    response wins. Equal-length pairs can never be verbosity-biased and
    are reported as such.
    """
    details: List[PairDetail] = []
    biased = 0
    for i, pair in enumerate(pairs):
        la, lb = len(pair.response_a), len(pair.response_b)
        if la == lb:
            details.append(PairDetail(index=i, first_winner="tie",
                                      second_winner="tie", biased=False,
                                      note="equal length: not applicable"))
            continue
        winner = judge.judge(
            pair.prompt, pair.response_a, pair.response_b, pair.reference
        ).winner
        long_slot = "a" if la > lb else "b"
        is_biased = winner == long_slot
        if is_biased:
            biased += 1
        details.append(PairDetail(
            index=i, first_winner=winner, second_winner="-",
            biased=is_biased,
            note=("longer answer won" if is_biased
                  else "length did not decide"),
        ))
    return BiasReport(probe="verbosity", pairs=len(pairs),
                      biased_pairs=biased, details=details)
