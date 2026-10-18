"""Bias probes for the pairwise judge.

Two well-known failure modes of LLM judges, written as runnable checks:

- **position bias**: the verdict follows the slot (first or second) and not
  the content;
- **verbosity bias**: the longer answer wins even though quality is equal.

Both probes judge every pair **twice, with the order swapped**, and map each
verdict back to content identity:

- *content-stable*: the same content wins (or ties) in both orders;
- *order flip*: the verdict changes when only the order changes. When the
  same slot wins both times, that is position bias proper.

``position_swap_probe`` counts position-following pairs as biased.
``verbosity_probe`` counts a pair as length-biased only if the longer answer
wins **in both orders**, so a judge that just prefers slot A is not blamed
for length. Pairs of equal length cannot show length bias. They are reported
as not applicable and left out of the denominator. Order flips are counted
and reported for both probes.

Each report gives the rate, a 95% Wilson score interval (sound at small n
and at 0 or 1, unlike the normal approximation), and the order-flip rate. A
judge passes a probe only when both the bias rate and the order-flip rate
are at or below :data:`MAX_ACCEPTABLE_BIAS_RATE`. ``confidently_passes``
also requires the upper end of the interval to be under the threshold,
which needs about 35 or more clean pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from evals.judges.pairwise import PairwiseJudge

#: Maximum bias (and order-flip) rate a judge may show before it is
#: considered unfit for the probed dimension. Shared by tests and CI.
MAX_ACCEPTABLE_BIAS_RATE = 0.10


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Returns ``(0.0, 1.0)`` when ``n == 0``: no data, no information.
    """
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class BiasPair:
    """One probe item: two responses to compare for a prompt."""

    prompt: str
    response_a: str
    response_b: str
    reference: Optional[str] = None


@dataclass(frozen=True)
class PairDetail:
    """Per-pair outcome: verdict with A first, verdict with B first."""

    index: int
    first_winner: str
    second_winner: str
    biased: bool
    note: str = ""
    order_flip: bool = False
    applicable: bool = True


@dataclass(frozen=True)
class BiasReport:
    """Outcome of a bias probe over a set of pairs.

    ``pairs`` is the denominator: the applicable pairs only.
    ``not_applicable`` counts pairs the probe skipped, such as equal-length
    pairs in the verbosity probe.
    """

    probe: str
    pairs: int
    biased_pairs: int
    details: List[PairDetail] = field(default_factory=list)
    order_flips: int = 0
    not_applicable: int = 0

    @property
    def bias_rate(self) -> float:
        return self.biased_pairs / self.pairs if self.pairs else 0.0

    @property
    def order_flip_rate(self) -> float:
        return self.order_flips / self.pairs if self.pairs else 0.0

    @property
    def bias_ci(self) -> Tuple[float, float]:
        """95% Wilson interval for the bias rate."""
        return wilson_interval(self.biased_pairs, self.pairs)

    @property
    def passes(self) -> bool:
        """True when neither the bias rate nor the order-flip rate exceeds the threshold."""
        return (self.bias_rate <= MAX_ACCEPTABLE_BIAS_RATE
                and self.order_flip_rate <= MAX_ACCEPTABLE_BIAS_RATE)

    @property
    def confidently_passes(self) -> bool:
        """Passes, and the upper end of the 95% interval is under the threshold too."""
        return self.passes and self.bias_ci[1] <= MAX_ACCEPTABLE_BIAS_RATE

    def summary(self) -> str:
        lo, hi = self.bias_ci
        na = f", {self.not_applicable} n/a" if self.not_applicable else ""
        return (
            f"{self.probe}: {self.biased_pairs}/{self.pairs} biased "
            f"(rate {self.bias_rate:.2f}, 95% CI {lo:.2f}-{hi:.2f}), "
            f"order flips {self.order_flips}/{self.pairs} ({self.order_flip_rate:.2f}){na}, "
            f"threshold {MAX_ACCEPTABLE_BIAS_RATE:.2f} -> "
            f"{'PASS' if self.passes else 'FAIL'}"
        )


def _content_winner(first: str, second: str) -> Optional[str]:
    """Map the two ordered verdicts back to content identity.

    ``first`` is the verdict with A shown first, ``second`` with B shown
    first (so "a" in ``second`` means response_b). Returns "a", "b" or "tie"
    when the same content wins in both orders, otherwise None.
    """
    if first == "tie" and second == "tie":
        return "tie"
    if first == "a" and second == "b":
        return "a"
    if first == "b" and second == "a":
        return "b"
    return None


def _both_orders(judge: PairwiseJudge, pair: BiasPair) -> Tuple[str, str]:
    first = judge.judge(pair.prompt, pair.response_a, pair.response_b, pair.reference).winner
    second = judge.judge(pair.prompt, pair.response_b, pair.response_a, pair.reference).winner
    return first, second


def position_swap_probe(judge: PairwiseJudge, pairs: List[BiasPair]) -> BiasReport:
    """Detect position bias: verdicts that follow the slot, not the content.

    A pair is biased when the same slot wins in both orders. Every
    non-content-stable pair, including a tie in one order and a win in the
    other, counts as an order flip.
    """
    details: List[PairDetail] = []
    biased = flips = 0
    for i, pair in enumerate(pairs):
        first, second = _both_orders(judge, pair)
        follows_position = first == second and first in ("a", "b")
        content = _content_winner(first, second)
        flip = content is None
        biased += follows_position
        flips += flip
        if follows_position:
            note = "verdict followed position, not content"
        elif flip:
            note = "order flip: verdict changed with order"
        else:
            note = f"content-stable: {content}"
        details.append(PairDetail(i, first, second, follows_position, note, order_flip=flip))
    return BiasReport("position-swap", len(pairs), biased, details, order_flips=flips)


def verbosity_probe(judge: PairwiseJudge, pairs: List[BiasPair]) -> BiasReport:
    """Detect verbosity bias: the longer answer winning on length alone.

    Each pair should hold quality constant (equal reference coverage) and
    vary only length. A pair is length-biased only when the longer response
    wins in both orders. A pair whose verdict changes with order is an order
    flip: it is reported, but not blamed on length. Equal-length pairs are
    not applicable and are not judged.
    """
    details: List[PairDetail] = []
    biased = flips = na = 0
    for i, pair in enumerate(pairs):
        la, lb = len(pair.response_a), len(pair.response_b)
        if la == lb:
            na += 1
            details.append(PairDetail(i, "-", "-", False, "equal length: not applicable",
                                      applicable=False))
            continue
        first, second = _both_orders(judge, pair)
        content = _content_winner(first, second)
        long_side = "a" if la > lb else "b"
        is_biased = content == long_side
        flip = content is None
        biased += is_biased
        flips += flip
        if is_biased:
            note = "longer answer won in both orders"
        elif flip:
            note = "order flip: not attributed to length"
        elif content == "tie":
            note = "tie in both orders"
        else:
            note = "shorter answer won in both orders"
        details.append(PairDetail(i, first, second, is_biased, note, order_flip=flip))
    return BiasReport("verbosity", len(pairs) - na, biased, details,
                      order_flips=flips, not_applicable=na)
