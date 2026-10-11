"""Token accounting, cost estimation, and latency statistics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

# USD per 1M tokens: (input, output). Illustrative defaults; override per run.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "default": (1.00, 3.00),
}


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("token counts must be non-negative")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


def total_usage(usages: Sequence[TokenUsage]) -> TokenUsage:
    """Sum a sequence of token usages."""
    total = TokenUsage()
    for usage in usages:
        total = total + usage
    return total


@dataclass(frozen=True)
class CostReport:
    usage: TokenUsage
    model: str
    cost_usd: float


def estimate_cost_usd(
    usage: TokenUsage,
    model: str = "default",
    pricing: Mapping[str, tuple[float, float]] | None = None,
) -> CostReport:
    """Estimate USD cost from token usage and a per-1M-token pricing table."""
    table = dict(DEFAULT_PRICING)
    if pricing:
        table.update(pricing)
    if model not in table:
        raise KeyError(f"no pricing for model {model!r}")
    in_rate, out_rate = table[model]
    cost = (usage.prompt_tokens / 1_000_000) * in_rate + (
        usage.completion_tokens / 1_000_000
    ) * out_rate
    return CostReport(usage=usage, model=model, cost_usd=cost)


@dataclass(frozen=True)
class LatencyStats:
    count: int
    mean_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    frac = rank - low
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + frac * (sorted_values[high] - sorted_values[low])


def latency_stats(latencies_ms: Sequence[float]) -> LatencyStats:
    """Mean/min/max/p50/p95 over latency samples in milliseconds."""
    values = sorted(latencies_ms)
    if not values:
        raise ValueError("latency_stats requires at least one sample")
    if any(v < 0 for v in values):
        raise ValueError("latencies must be non-negative")
    return LatencyStats(
        count=len(values),
        mean_ms=statistics.fmean(values),
        min_ms=values[0],
        max_ms=values[-1],
        p50_ms=_percentile(values, 50),
        p95_ms=_percentile(values, 95),
    )
