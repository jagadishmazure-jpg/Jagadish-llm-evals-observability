"""Regression suite runner: dataset in, model function, aggregated results out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from evals.judges.base import Judge, Judgment
from evals.suites.datasets import EvalCase


@dataclass(frozen=True)
class ModelOutput:
    """What a model-under-test returns for one case."""

    response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    judgment: Judgment | None
    output: ModelOutput | None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True)
class SuiteResult:
    """Aggregate outcome of a full regression run."""

    results: tuple[CaseResult, ...]
    total_latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.output.prompt_tokens for r in self.results if r.output)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.output.completion_tokens for r in self.results if r.output)

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]


# Model under test: case in, ModelOutput out. May raise; the runner catches it.
ModelFn = Callable[[EvalCase], ModelOutput]


class RegressionSuite:
    """Runs every case through ``model_fn`` and scores outputs with judges.

    A case passes when it produces output without error and every judge
    scores it at or above ``pass_threshold`` (judge scores are expected on a
    0..1 scale; use :class:`RubricJudge` via an adapter or normalize first).
    With no judges, a case passes when the model returns without error.
    Model exceptions are caught and recorded as failures, never raised.
    """

    def __init__(
        self,
        cases: Sequence[EvalCase],
        model_fn: ModelFn,
        judges: Sequence[Judge] = (),
        pass_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("pass_threshold must be in 0..1")
        self._cases = list(cases)
        self._model_fn = model_fn
        self._judges = list(judges)
        self._threshold = pass_threshold

    @property
    def cases(self) -> list[EvalCase]:
        return list(self._cases)

    def run(self) -> SuiteResult:
        results: list[CaseResult] = []
        total_latency = 0.0
        for case in self._cases:
            started = time.perf_counter()
            try:
                output = self._model_fn(case)
                error: str | None = output.error
            except Exception as exc:  # noqa: BLE001 - runner must not crash
                output = None
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000.0
            total_latency += latency_ms

            judgment: Judgment | None = None
            passed = error is None
            if passed and output is not None:
                for judge in self._judges:
                    judgment = judge.judge(
                        case.input, output.response, reference=case.expected
                    )
                    if judgment.score < self._threshold:
                        passed = False
                        break
            results.append(
                CaseResult(
                    case_id=case.id,
                    passed=passed,
                    judgment=judgment,
                    output=output,
                    error=error,
                    latency_ms=latency_ms,
                )
            )
        return SuiteResult(results=tuple(results), total_latency_ms=total_latency)
