"""Tests for token accounting, cost estimation, and latency stats."""

import pytest

from evals.metrics import (
    TokenUsage,
    estimate_cost_usd,
    latency_stats,
    total_usage,
)


def test_token_usage_total():
    assert TokenUsage(100, 50).total_tokens == 150


def test_token_usage_rejects_negative():
    with pytest.raises(ValueError):
        TokenUsage(-1, 0)
    with pytest.raises(ValueError):
        TokenUsage(0, -5)


def test_token_usage_add():
    total = TokenUsage(100, 50) + TokenUsage(25, 25)
    assert total == TokenUsage(125, 75)


def test_token_usage_add_wrong_type():
    with pytest.raises(TypeError):
        TokenUsage(1, 1) + 5  # type: ignore[operator]


def test_total_usage_sums():
    total = total_usage([TokenUsage(10, 5), TokenUsage(20, 15), TokenUsage()])
    assert total == TokenUsage(30, 20)


def test_total_usage_empty():
    assert total_usage([]) == TokenUsage(0, 0)


def test_estimate_cost_default_pricing():
    report = estimate_cost_usd(TokenUsage(1_000_000, 1_000_000))
    assert report.cost_usd == pytest.approx(4.0)
    assert report.model == "default"


def test_estimate_cost_custom_pricing():
    report = estimate_cost_usd(
        TokenUsage(500_000, 500_000),
        model="demo",
        pricing={"demo": (2.0, 4.0)},
    )
    assert report.cost_usd == pytest.approx(3.0)


def test_estimate_cost_unknown_model():
    with pytest.raises(KeyError):
        estimate_cost_usd(TokenUsage(10, 10), model="nope")


def test_estimate_cost_zero_usage():
    report = estimate_cost_usd(TokenUsage(0, 0))
    assert report.cost_usd == 0.0


def test_latency_stats_basic():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0])
    assert stats.count == 4
    assert stats.mean_ms == pytest.approx(25.0)
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(40.0)
    assert stats.p50_ms == pytest.approx(25.0)
    assert stats.p95_ms == pytest.approx(38.5)


def test_latency_stats_single_value():
    stats = latency_stats([7.5])
    assert stats.mean_ms == pytest.approx(7.5)
    assert stats.p50_ms == pytest.approx(7.5)
    assert stats.p95_ms == pytest.approx(7.5)


def test_latency_stats_empty_raises():
    with pytest.raises(ValueError):
        latency_stats([])


def test_latency_stats_negative_raises():
    with pytest.raises(ValueError):
        latency_stats([10.0, -1.0])


def test_latency_stats_unsorted_input():
    stats = latency_stats([40.0, 10.0, 30.0, 20.0])
    assert stats.min_ms == pytest.approx(10.0)
    assert stats.max_ms == pytest.approx(40.0)
