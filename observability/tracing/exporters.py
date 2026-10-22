"""Span exporters."""

from __future__ import annotations

from typing import Protocol, Sequence

from observability.tracing.spans import Span


class Exporter(Protocol):
    """Receives finished spans."""

    def export(self, span: Span) -> None:
        """Handle one finished span."""


class InMemoryExporter:
    """Collects finished spans in memory; ideal for tests and notebooks."""

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def export(self, span: Span) -> None:
        self._spans.append(span)

    @property
    def spans(self) -> list[Span]:
        return list(self._spans)

    def clear(self) -> None:
        self._spans.clear()

    def spans_for_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]

    def export_batch(self, spans: Sequence[Span]) -> None:
        self._spans.extend(spans)
