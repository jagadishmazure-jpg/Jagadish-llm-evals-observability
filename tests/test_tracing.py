"""Tests for spans, the tracer context manager, and exporters."""

import pytest

from observability.tracing import (
    InMemoryExporter,
    Span,
    SpanStatus,
    Trace,
    Tracer,
)


def test_span_defaults():
    span = Span(name="op")
    assert span.status is SpanStatus.OK
    assert span.is_root
    assert span.duration_ms is None  # unfinished
    assert span.attributes == {}
    assert span.events == []


def test_span_finish_records_duration():
    span = Span(name="op")
    span.finish()
    assert span.duration_ms is not None
    assert span.duration_ms >= 0.0


def test_span_finish_with_error():
    span = Span(name="op")
    span.finish(SpanStatus.ERROR, error=RuntimeError("boom"))
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "RuntimeError"
    assert span.error_message == "boom"


def test_span_attributes_and_events():
    span = Span(name="op")
    span.set_attribute("model", "demo")
    span.add_event("retry", attempt=2)
    assert span.attributes["model"] == "demo"
    assert span.events[0].name == "retry"
    assert span.events[0].attributes["attempt"] == 2


def test_tracer_nests_spans():
    exporter = InMemoryExporter()
    tracer = Tracer(exporter)
    with tracer.start_span("parent") as parent:
        with tracer.start_span("child") as child:
            assert child.parent_id == parent.span_id
            assert child.trace_id == parent.trace_id
            assert not child.is_root
    assert parent.is_root
    assert len(exporter.spans) == 2


def test_tracer_exports_finished_spans():
    exporter = InMemoryExporter()
    tracer = Tracer(exporter)
    with tracer.start_span("work", model="m1"):
        pass
    assert len(exporter.spans) == 1
    assert exporter.spans[0].name == "work"
    assert exporter.spans[0].attributes["model"] == "m1"
    assert exporter.spans[0].status is SpanStatus.OK


def test_tracer_captures_exceptions():
    exporter = InMemoryExporter()
    tracer = Tracer(exporter)
    with pytest.raises(ValueError, match="bad"):
        with tracer.start_span("risky"):
            raise ValueError("bad")
    span = exporter.spans[0]
    assert span.status is SpanStatus.ERROR
    assert span.error_type == "ValueError"


def test_tracer_restores_parent_after_child():
    exporter = InMemoryExporter()
    tracer = Tracer(exporter)
    with tracer.start_span("parent") as parent:
        with tracer.start_span("child"):
            pass
        assert tracer.current_span is parent
    assert tracer.current_span is None


def test_tracer_without_exporter():
    tracer = Tracer()
    with tracer.start_span("solo") as span:
        assert span.name == "solo"


def test_exporter_clear():
    exporter = InMemoryExporter()
    tracer = Tracer(exporter)
    with tracer.start_span("a"):
        pass
    exporter.clear()
    assert exporter.spans == []


def test_exporter_spans_for_trace():
    exporter = InMemoryExporter()
    t1, t2 = Tracer(exporter), Tracer(exporter)
    with t1.start_span("a"):
        pass
    with t2.start_span("b"):
        pass
    assert len(exporter.spans_for_trace(t1.trace_id)) == 1
    assert exporter.spans_for_trace(t1.trace_id)[0].name == "a"


def test_exporter_export_batch():
    exporter = InMemoryExporter()
    exporter.export_batch([Span(name="a"), Span(name="b")])
    assert len(exporter.spans) == 2


def test_trace_root_and_named():
    root = Span(name="root", trace_id="t1")
    child = Span(name="child", trace_id="t1", parent_id=root.span_id)
    trace = Trace(trace_id="t1", spans=(root, child))
    assert trace.root is root
    assert trace.spans_named("child") == [child]
    assert trace.spans_named("missing") == []


def test_trace_root_missing():
    orphan = Span(name="x", trace_id="t", parent_id="nope")
    assert Trace(trace_id="t", spans=(orphan,)).root is None
