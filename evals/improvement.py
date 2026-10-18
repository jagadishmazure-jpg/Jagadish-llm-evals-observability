"""Close the loop: aggregate failures into patterns, compare before/after.

Evaluation only pays for itself when the numbers drive a fix. This module
provides the two halves of that loop:

- :class:`FailureAnalysis`: groups failing (case, metric) records by reason
  signature so the top failure patterns surface ("step_efficiency fails
  whenever get_today is called unnecessarily").
- :class:`ComparisonReport`: compares metric scores before and after a
  change, with a Markdown delta table for the PR or review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from evals.trajectory import MetricResult

__all__ = [
    "ScoredCase",
    "FailurePattern",
    "FailureAnalysis",
    "ComparisonReport",
]


@dataclass(frozen=True)
class ScoredCase:
    """One metric's verdict on one evaluation case."""

    case_id: str
    metric_name: str
    result: MetricResult


@dataclass(frozen=True)
class FailurePattern:
    """A recurring failure: one metric, one reason shape, N cases."""

    metric_name: str
    occurrences: int
    signature: str
    example_reason: str
    case_ids: tuple[str, ...] = ()

    def describe(self) -> str:
        cases = ", ".join(self.case_ids) if self.case_ids else "no cases"
        return (
            f"{self.metric_name} failed {self.occurrences}x "
            f"({cases}): {self.signature}"
        )


_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def _signature(reason: str) -> str:
    """Reason shape with concrete values abstracted away.

    Quoted spans become {} and numbers become #, so "unnecessary:
    'get_today'" and "unnecessary: 'check_date'" cluster into one pattern
    while genuinely different reasons stay apart.
    """
    s = _QUOTED.sub("{}", reason)
    return _NUMBER.sub("#", s)


class FailureAnalysis:
    """Aggregates scored cases into ranked failure patterns."""

    def __init__(self, records: Sequence[ScoredCase]) -> None:
        self._records = list(records)

    @property
    def records(self) -> list[ScoredCase]:
        return list(self._records)

    def failures(self) -> list[ScoredCase]:
        """Only the records that did not pass."""
        return [r for r in self._records if not r.result.passed]

    def metrics(self) -> list[str]:
        return sorted({r.metric_name for r in self._records})

    def failure_rate(self, metric_name: str) -> float:
        """Fraction of records for ``metric_name`` that failed."""
        recs = [r for r in self._records if r.metric_name == metric_name]
        if not recs:
            raise ValueError(f"no records for metric {metric_name!r}")
        return sum(1 for r in recs if not r.result.passed) / len(recs)

    def top_patterns(self, n: int = 5) -> list[FailurePattern]:
        """Most frequent failure patterns, most frequent first.

        Ties break deterministically by metric name then signature.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        groups: dict[tuple[str, str], list[ScoredCase]] = {}
        for record in self.failures():
            key = (record.metric_name, _signature(record.result.reason))
            groups.setdefault(key, []).append(record)
        patterns = [
            FailurePattern(
                metric_name=metric,
                occurrences=len(rs),
                signature=sig,
                example_reason=rs[0].result.reason,
                case_ids=tuple(sorted({r.case_id for r in rs})),
            )
            for (metric, sig), rs in groups.items()
        ]
        patterns.sort(key=lambda p: (-p.occurrences, p.metric_name, p.signature))
        return patterns[:n]


@dataclass(frozen=True)
class ComparisonReport:
    """Before/after metric scores for one improvement.

    ``before``/``after`` map metric name -> scores across cases. A metric
    present on only one side shows as n/a in the table; :meth:`delta`
    requires both sides.
    """

    before: Mapping[str, Sequence[float]] = field(default_factory=dict)
    after: Mapping[str, Sequence[float]] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for side in (self.before, self.after):
            for metric, scores in side.items():
                for s in scores:
                    if not 0.0 <= s <= 1.0:
                        raise ValueError(f"score {s} for {metric!r} not in 0..1")
        for metric, t in self.thresholds.items():
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"threshold {t} for {metric!r} not in 0..1")

    def metrics(self) -> list[str]:
        return sorted(set(self.before) | set(self.after))

    def _mean(self, side: Mapping[str, Sequence[float]], metric: str) -> float:
        scores = side.get(metric)
        if not scores:
            raise ValueError(f"no scores for metric {metric!r}")
        return sum(scores) / len(scores)

    def mean_before(self, metric: str) -> float:
        return self._mean(self.before, metric)

    def mean_after(self, metric: str) -> float:
        return self._mean(self.after, metric)

    def threshold_for(self, metric: str) -> float:
        return self.thresholds.get(metric, 0.8)

    def _pass_rate(self, side: Mapping[str, Sequence[float]], metric: str) -> float:
        scores = side.get(metric, ())
        if not scores:
            raise ValueError(f"no scores for metric {metric!r}")
        threshold = self.threshold_for(metric)
        return sum(1 for s in scores if s >= threshold) / len(scores)

    def delta(self, metric: str) -> float:
        """Mean after minus mean before. Positive means the fix helped."""
        return self.mean_after(metric) - self.mean_before(metric)

    def to_markdown(self) -> str:
        """Per-metric delta table plus an improvement summary."""
        header = (
            "| metric | before (mean) | after (mean) | delta | "
            "before pass rate | after pass rate |"
        )
        sep = "|---|---|---|---|---|---|"
        lines = [header, sep]
        improved = regressed = 0
        for metric in self.metrics():
            try:
                b, a = self.mean_before(metric), self.mean_after(metric)
                d = a - b
                improved += d > 0
                regressed += d < 0
                bp = self._pass_rate(self.before, metric)
                ap = self._pass_rate(self.after, metric)
                lines.append(
                    f"| {metric} | {b:.2f} | {a:.2f} | {d:+.2f} | "
                    f"{bp:.0%} | {ap:.0%} |"
                )
            except ValueError:
                lines.append(f"| {metric} | n/a | n/a | n/a | n/a | n/a |")
        lines += [
            "",
            f"Summary: {improved} metric(s) improved, {regressed} regressed, "
            f"{len(self.metrics()) - improved - regressed} unchanged or n/a.",
        ]
        return "\n".join(lines)
