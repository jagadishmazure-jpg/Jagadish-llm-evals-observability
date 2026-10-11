"""Tests for trace analytics: cost/latency aggregates and failures."""

import pytest

from observability.analytics import (
    aggregate_by_agent,
    aggregate_by_model,
    failure_breakdown,
)
from observability.tracing import Span, SpanStatus

PRICING = {"m1": (2.0, 6.0), "m2": (1.0, 1.0)}


def finished_span(name="llm.call", **attrs):
    span = Span(name=name, attributes=dict(attrs))
    span.finish()
    return span


def test_aggregate_by_model_groups_and_sums():
    spans = [
        finished_span(model="m1", prompt_tokens=100, completion_tokens=50),
        finished_span(model="m1", prompt_tokens=100, completion_tokens=50),
        finished_span(model="m2", prompt_tokens=10, completion_tokens=10),
    ]
    report = aggregate_by_model(spans, pricing=PRICING)
    assert set(report) == {"m1", "m2"}
    m1 = report["m1"]
    assert m1.calls == 2
    assert m1.prompt_tokens == 200
    assert m1.completion_tokens == 100
    # (200/1e6)*2 + (100/1e6)*6 = 0.001
    assert m1.cost_usd == pytest.approx(0.001)


def test_aggregate_by_agent():
    spans = [
        finished_span(agent="researcher", model="m1", prompt_tokens=10),
        finished_span(agent="coder", model="m1", prompt_tokens=20),
        finished_span(agent="researcher", model="m1", prompt_tokens=30),
    ]
    report = aggregate_by_agent(spans, pricing=PRICING)
    assert report["researcher"].calls == 2
    assert report["researcher"].prompt_tokens == 40
    assert report["coder"].calls == 1


def test_aggregate_missing_attributes_grouped_unknown():
    spans = [finished_span(prompt_tokens=5)]
    report = aggregate_by_model(spans, pricing=PRICING)
    assert set(report) == {"unknown"}
    assert report["unknown"].prompt_tokens == 5


def test_aggregate_latency_stats_present():
    spans = [finished_span(model="m1"), finished_span(model="m1")]
    report = aggregate_by_model(spans, pricing=PRICING)
    assert report["m1"].latency is not None
    assert report["m1"].latency.count == 2


def test_aggregate_latency_none_when_unfinished():
    spans = [Span(name="open", attributes={"model": "m1"})]
    report = aggregate_by_model(spans, pricing=PRICING)
    assert report["m1"].latency is None


def test_aggregate_empty():
    assert aggregate_by_model([]) == {}
    assert aggregate_by_agent([]) == {}


def test_failure_breakdown_counts():
    ok = finished_span("ok.span")
    bad = finished_span("bad.span")
    bad.finish(SpanStatus.ERROR, error=TimeoutError("slow"))
    worse = finished_span("bad.span")
    worse.finish(SpanStatus.ERROR, error=ValueError("bad"))
    report = failure_breakdown([ok, bad, worse])
    assert report.total_spans == 3
    assert report.failed_spans == 2
    assert report.failure_rate == pytest.approx(2 / 3)
    assert report.by_error_type == {"TimeoutError": 1, "ValueError": 1}
    assert report.by_span_name == {"bad.span": 2}


def test_failure_breakdown_all_ok():
    report = failure_breakdown([finished_span("a"), finished_span("b")])
    assert report.failed_spans == 0
    assert report.failure_rate == 0.0
    assert report.by_error_type == {}


def test_failure_breakdown_empty():
    report = failure_breakdown([])
    assert report.total_spans == 0
    assert report.failure_rate == 0.0


def test_failure_breakdown_error_without_exception():
    span = finished_span("x")
    span.finish(SpanStatus.ERROR)  # no error object
    report = failure_breakdown([span])
    assert report.by_error_type == {"unknown": 1}
