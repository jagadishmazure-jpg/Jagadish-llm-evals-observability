"""Turn finished spans into per-model/per-agent cost and latency reports.

Conventions: spans record ``model`` and ``agent`` in their attributes, plus
``prompt_tokens`` / ``completion_tokens`` for LLM calls. Spans missing an
attribute are grouped under ``"unknown"``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from evals.metrics.cost import (
    DEFAULT_PRICING,
    LatencyStats,
    TokenUsage,
    estimate_cost_usd,
    latency_stats,
)
from observability.tracing.spans import Span, SpanStatus

UNKNOWN = "unknown"


@dataclass(frozen=True)
class CostLatencyAggregate:
    key: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency: LatencyStats | None


def _aggregate(
    spans: Sequence[Span],
    key_fn,
    pricing: Mapping[str, tuple[float, float]] | None,
) -> dict[str, CostLatencyAggregate]:
    groups: dict[str, list[Span]] = defaultdict(list)
    for span in spans:
        groups[key_fn(span)].append(span)

    report: dict[str, CostLatencyAggregate] = {}
    for key, group in groups.items():
        prompt = sum(int(span.attributes.get("prompt_tokens", 0)) for span in group)
        completion = sum(
            int(span.attributes.get("completion_tokens", 0)) for span in group
        )
        usage = TokenUsage(prompt, completion)
        # Cost is attributed per model. Models missing from the pricing table
        # fall back to the default rates so analytics never crash on a new
        # model name; pass an explicit table for accurate billing.
        table = dict(DEFAULT_PRICING)
        if pricing:
            table.update(pricing)
        model = group[0].attributes.get("model", "default")
        priced_as = model if model in table else "default"
        cost = estimate_cost_usd(usage, model=priced_as, pricing=table).cost_usd
        latencies = [
            span.duration_ms for span in group if span.duration_ms is not None
        ]
        report[key] = CostLatencyAggregate(
            key=key,
            calls=len(group),
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=cost,
            latency=latency_stats(latencies) if latencies else None,
        )
    return report


def aggregate_by_model(
    spans: Sequence[Span],
    pricing: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, CostLatencyAggregate]:
    """Cost/latency per ``model`` attribute (``"unknown"`` when missing)."""
    return _aggregate(
        spans, lambda s: str(s.attributes.get("model", UNKNOWN)), pricing
    )


def aggregate_by_agent(
    spans: Sequence[Span],
    pricing: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, CostLatencyAggregate]:
    """Cost/latency per ``agent`` attribute (``"unknown"`` when missing)."""
    return _aggregate(
        spans, lambda s: str(s.attributes.get("agent", UNKNOWN)), pricing
    )


@dataclass(frozen=True)
class FailureBreakdown:
    total_spans: int
    failed_spans: int
    by_error_type: dict[str, int]
    by_span_name: dict[str, int]

    @property
    def failure_rate(self) -> float:
        return self.failed_spans / self.total_spans if self.total_spans else 0.0


def failure_breakdown(spans: Iterable[Span]) -> FailureBreakdown:
    """Summarize failures: counts by error type and by span name."""
    items = list(spans)
    by_error: Counter[str] = Counter()
    by_name: Counter[str] = Counter()
    failed = 0
    for span in items:
        if span.status is SpanStatus.ERROR:
            failed += 1
            by_error[span.error_type or UNKNOWN] += 1
            by_name[span.name] += 1
    return FailureBreakdown(
        total_spans=len(items),
        failed_spans=failed,
        by_error_type=dict(by_error),
        by_span_name=dict(by_name),
    )
