"""Deterministic behavior metrics: tools, arguments, and final answers.

Mirrors DeepEval's ``ToolCorrectnessMetric``, ``ArgumentCorrectnessMetric``
and ``GEval``-style answer correctness, but every score is computed from
recorded values instead of an LLM judge. For *how* the agent worked (the
path it took), see :mod:`evals.trajectory`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from evals.trajectory import (
    MetricResult,
    _claim_coverage,
    _freeze,
    _verdict,
)

__all__ = [
    "BehaviorCase",
    "tool_correctness_score",
    "argument_correctness_score",
    "answer_correctness_score",
]

_MISSING = object()  # sentinel for "argument not supplied at all"


@dataclass(frozen=True)
class BehaviorCase:
    """One evaluation case with behavior-level expectations.

    Beyond the expected answer, the case records which tools the agent
    should call, with which arguments, and optionally the plan it should
    follow. This is the "expected behavior" contract: stricter than an
    answer-only gold set, and what makes tool/argument metrics possible.
    """

    input: str
    expected_answer: str
    expected_tools: Sequence[str] = ()
    expected_arguments: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    expected_plan: Sequence[str] = ()


def tool_correctness_score(
    actual_tools: Sequence[str],
    expected_tools: Sequence[str],
    threshold: float = 0.8,
) -> MetricResult:
    """Were the right tools called? (DeepEval: ToolCorrectnessMetric)

    Set-overlap F1 between called and expected tools: missing tools hurt
    recall, extra tools hurt precision. Order is not checked here; that is
    :func:`evals.trajectory.plan_quality_score`'s job.
    """
    actual, expected = set(actual_tools), set(expected_tools)
    if not expected and not actual:
        return _verdict(1.0, threshold, "no tools expected and none called")
    if not expected:
        return _verdict(
            0.0, threshold, f"no tools expected but called: {', '.join(sorted(actual))}"
        )
    if not actual:
        return _verdict(
            0.0, threshold, f"expected {', '.join(sorted(expected))} but no tools called"
        )
    hits = len(actual & expected)
    precision = hits / len(actual)
    recall = hits / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    reason = f"F1 {f1:.2f} (precision {precision:.2f}, recall {recall:.2f})"
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        reason += f"; missing: {', '.join(repr(m) for m in missing)}"
    if extra:
        reason += f"; extra: {', '.join(repr(e) for e in extra)}"
    return _verdict(f1, threshold, reason)


def argument_correctness_score(
    actual_args: Mapping[str, Mapping[str, object]],
    expected_args: Mapping[str, Mapping[str, object]],
    threshold: float = 0.8,
) -> MetricResult:
    """Were tools called with the right arguments? (DeepEval: ArgumentCorrectnessMetric)

    Per-tool match rate, averaged over expected tools: a tool that was
    never called scores 0 for that tool. Values compare canonically, so
    ``{"a": 1}`` never accidentally equals ``{"a": "1"}``. Extra tools the
    agent called are not penalized here (see
    :func:`tool_correctness_score`).
    """
    if not expected_args:
        return _verdict(1.0, threshold, "no expected arguments to check")
    per_tool: list[float] = []
    problems: list[str] = []
    for tool in sorted(expected_args):
        expected = expected_args[tool]
        actual = actual_args.get(tool)
        if actual is None:
            per_tool.append(0.0)
            problems.append(f"{tool!r} was not called")
            continue
        if not expected:
            per_tool.append(1.0)
            continue
        matched = sum(
            1 for k, v in expected.items() if _freeze(actual.get(k, _MISSING)) == _freeze(v)
        )
        per_tool.append(matched / len(expected))
        if matched < len(expected):
            bad = sorted(
                k
                for k, v in expected.items()
                if _freeze(actual.get(k, _MISSING)) != _freeze(v)
            )
            problems.append(f"{tool!r}: wrong/missing arguments: {', '.join(bad)}")
    score = sum(per_tool) / len(per_tool)
    if problems:
        reason = f"argument match {score:.2f} across {len(per_tool)} tools; " + "; ".join(problems)
    else:
        reason = f"all arguments matched across {len(per_tool)} tools"
    return _verdict(score, threshold, reason)


def answer_correctness_score(
    actual: str,
    expected: str,
    threshold: float = 0.8,
) -> MetricResult:
    """Does the final answer reach the expected conclusion? (DeepEval: GEval)

    Logical-conclusion match, NOT exact string match: the text is
    normalized (case, punctuation) and the expected answer's key claims
    must appear. Paraphrases pass; a wrong or missing conclusion fails even
    when most words overlap.
    """
    score, covered, missed = _claim_coverage(actual, expected)
    total = len(covered) + len(missed)
    if missed:
        reason = (
            f"{len(covered)}/{total} expected claims present; "
            f"missing or contradicted: {'; '.join(missed)}"
        )
    else:
        reason = f"conclusion matches: all {total} expected claims present"
    return _verdict(score, threshold, reason)
