"""Deterministic trajectory metrics for agent evaluation.

Mirrors DeepEval's agentic metrics (``TaskCompletionMetric``,
``StepEfficiencyMetric``, ``PlanQualityMetric``, ``PlanAdherenceMetric``)
without an LLM judge: every score is a deterministic function of the
recorded trajectory, so identical runs always produce identical scores.

Use these when you want to score *how* the agent worked (the path), not
just what it concluded. For *what it did* (tools, arguments, final
answer), see :mod:`evals.behavior`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "MetricResult",
    "TrajectoryStep",
    "Trajectory",
    "task_completion_score",
    "step_efficiency_score",
    "plan_quality_score",
    "plan_adherence_score",
]


@dataclass(frozen=True)
class MetricResult:
    """Outcome of one deterministic metric evaluation."""

    score: float
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in 0..1")
        if not self.reason:
            raise ValueError("reason must explain the score")


@dataclass(frozen=True)
class TrajectoryStep:
    """One recorded agent action."""

    tool_name: str
    arguments: Mapping[str, object] = field(default_factory=dict)
    reasoning_summary: str = ""


@dataclass(frozen=True)
class Trajectory:
    """The full recorded run of an agent: what it did and what it said."""

    steps: Sequence[TrajectoryStep] = ()
    final_answer: str = ""
    declared_plan: Sequence[str] = ()


# ---------------------------------------------------------------------------
# Shared text helpers (also used by evals.behavior).
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but if then else when while with without for of to in on at by "
    "from as is are was were be been being it its this that these those i you he she "
    "we they them his her our your their my me him us do does did done have has had "
    "having will would can could shall should may might must not no yes so such than "
    "too very just only also however therefore because what which who whom whose how "
    "where why there here all any each every both few more most other some nor own "
    "same over under again once up out into onto off".split()
)

_CLAIM_SPLIT = re.compile(r"[.!?;\n]+")
_WORD = re.compile(r"[a-z0-9]+")


def _normalize_text(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; splits on punctuation and underscores."""
    return _WORD.findall(text.lower())


def _content_tokens(text: str) -> set[str]:
    """Normalized tokens minus stopwords: the words that carry meaning."""
    return {t for t in _normalize_text(text) if t not in _STOPWORDS}


def _split_claims(text: str) -> list[str]:
    return [c.strip() for c in _CLAIM_SPLIT.split(text) if c.strip()]


def _claim_coverage(actual: str, expected: str) -> tuple[float, list[str], list[str]]:
    """Fraction of expected claims whose content appears in ``actual``.

    A claim counts as covered when at least 60% of its content tokens occur
    in the actual text. Matching is case- and punctuation-insensitive and
    ignores stopwords, so paraphrases pass while missing or wrong
    conclusions fail.
    """
    claims = [c for c in _split_claims(expected) if _content_tokens(c)]
    if not claims:
        raise ValueError("expected text contains no claims to check")
    actual_tokens = _content_tokens(actual)
    covered: list[str] = []
    missed: list[str] = []
    for claim in claims:
        tokens = _content_tokens(claim)
        hit = len(tokens & actual_tokens) / len(tokens)
        (covered if hit >= 0.6 else missed).append(claim)
    return len(covered) / len(claims), covered, missed


def _freeze(value: object) -> str:
    """Canonical string form of (possibly nested) argument values."""
    if isinstance(value, Mapping):
        inner = ",".join(
            f"{_freeze(k)}:{_freeze(v)}"
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + inner + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_freeze(v) for v in value) + "]"
    return repr(value)


def _check_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in 0..1")


def _verdict(score: float, threshold: float, reason: str) -> MetricResult:
    _check_threshold(threshold)
    return MetricResult(score=score, passed=score >= threshold, reason=reason)


# ---------------------------------------------------------------------------
# The four trajectory metrics.
# ---------------------------------------------------------------------------


def task_completion_score(
    trajectory: Trajectory,
    expected_behavior: str,
    threshold: float = 0.8,
) -> MetricResult:
    """Did the agent accomplish what was asked? (DeepEval: TaskCompletionMetric)

    Deterministic: extracts claims from ``expected_behavior`` and scores the
    fraction covered by the final answer. Catches wrong conclusions and
    missing outcomes. It does not judge *how* the agent got there; the
    trajectory metrics below cover the path.
    """
    score, covered, missed = _claim_coverage(trajectory.final_answer, expected_behavior)
    total = len(covered) + len(missed)
    if missed:
        reason = (
            f"{len(covered)}/{total} expected claims covered; "
            f"missing: {'; '.join(missed)}"
        )
    else:
        reason = f"all {total} expected claims covered in the final answer"
    return _verdict(score, threshold, reason)


def step_efficiency_score(
    trajectory: Trajectory,
    minimal_tools: set[str] | frozenset[str],
    threshold: float = 0.8,
) -> MetricResult:
    """Did the agent avoid unnecessary tool calls? (DeepEval: StepEfficiencyMetric)

    score = (calls to tools in ``minimal_tools``) / (total calls), so every
    extra call lowers the score. Calling nothing scores 0 unless the task
    genuinely needs no tools.
    """
    actual = [s.tool_name for s in trajectory.steps]
    if not actual:
        if not minimal_tools:
            return _verdict(1.0, threshold, "no tools needed and none called")
        return _verdict(
            0.0,
            threshold,
            f"no tools called; expected one of: {', '.join(sorted(minimal_tools))}",
        )
    needed = sum(1 for t in actual if t in minimal_tools)
    extra = sorted({t for t in actual if t not in minimal_tools})
    score = needed / len(actual)
    if extra:
        quoted = ", ".join(repr(t) for t in extra)
        reason = f"{needed}/{len(actual)} calls necessary; unnecessary: {quoted}"
    else:
        reason = f"all {len(actual)} tool calls were necessary"
    return _verdict(score, threshold, reason)


_INFO_PREFIXES = ("get", "check", "fetch", "lookup", "calculate", "search", "retrieve", "list")
_ACTION_PREFIXES = ("issue", "process", "send", "create", "update", "delete", "submit", "approve")


def _tool_kind(tool_name: str) -> str:
    name = tool_name.lower()
    if name.startswith(_ACTION_PREFIXES):
        return "action"
    if name.startswith(_INFO_PREFIXES):
        return "info"
    return "other"


def plan_quality_score(
    trajectory: Trajectory,
    task_description: str,
    threshold: float = 0.8,
) -> MetricResult:
    """Was the tool sequence an effective plan? (DeepEval: PlanQualityMetric)

    Deterministic structural heuristics, each costing 0.25 (floor 0.0):

    - redundant re-fetch: the same tool called twice with identical arguments
    - action before information: an action tool runs with no prior
      info-gathering call (e.g. issuing a refund without looking anything up)
    - calls after a terminal action: tools called after the last action tool,
      when the outcome was already determined
    """
    steps = list(trajectory.steps)
    if not steps:
        if task_description.strip():
            return _verdict(0.0, threshold, "no tool calls executed for the task")
        return _verdict(1.0, threshold, "no task and no tool calls")
    penalties: list[str] = []
    seen: set[tuple[str, str]] = set()
    for s in steps:
        key = (s.tool_name, _freeze(s.arguments))
        if key in seen:
            penalties.append(
                f"redundant re-fetch: {s.tool_name!r} called again with identical arguments"
            )
        seen.add(key)
    kinds = [_tool_kind(s.tool_name) for s in steps]
    if "action" in kinds:
        first_action = kinds.index("action")
        if "info" not in kinds[:first_action]:
            penalties.append(
                f"action before information: {steps[first_action].tool_name!r} "
                "ran with no prior info-gathering call"
            )
        terminal_at = max(i for i, k in enumerate(kinds) if k == "action")
        trailing = sorted({s.tool_name for s in steps[terminal_at + 1 :]})
        if trailing:
            quoted = ", ".join(repr(t) for t in trailing)
            penalties.append(
                f"calls after terminal action: {quoted} ran after "
                f"{steps[terminal_at].tool_name!r}"
            )
    score = max(0.0, 1.0 - 0.25 * len(penalties))
    if penalties:
        reason = f"score {score:.2f}; " + "; ".join(penalties)
    else:
        reason = (
            f"effective sequence: {len(steps)} calls, info before action, "
            "no re-fetches, nothing after the terminal action"
        )
    return _verdict(score, threshold, reason)


def plan_adherence_score(
    trajectory: Trajectory,
    threshold: float = 0.8,
) -> MetricResult:
    """Did execution follow the declared plan? (DeepEval: PlanAdherenceMetric)

    Each planned step counts as followed when at least half of its content
    tokens appear across the executed steps (tool names plus reasoning
    summaries). A good plan that was ignored scores low here even if the
    outcome was right.
    """
    plan = [p for p in trajectory.declared_plan if p.strip()]
    if not plan:
        return _verdict(0.0, threshold, "no declared plan to adhere to")
    executed = " ".join(
        [s.tool_name for s in trajectory.steps]
        + [s.reasoning_summary for s in trajectory.steps]
    )
    executed_tokens = _content_tokens(executed)
    followed = 0
    for step in plan:
        tokens = _content_tokens(step)
        if not tokens:
            continue
        if len(tokens & executed_tokens) / len(tokens) >= 0.5:
            followed += 1
    score = followed / len(plan)
    reason = f"followed {followed}/{len(plan)} planned steps"
    return _verdict(score, threshold, reason)
