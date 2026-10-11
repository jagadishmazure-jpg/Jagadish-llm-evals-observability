"""Context-manager API for creating spans inside a trace."""

from __future__ import annotations

import contextlib
import contextvars
import uuid
from typing import Any, Iterator, Optional

from observability.tracing.exporters import Exporter
from observability.tracing.spans import Span, SpanStatus

_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    "current_span", default=None
)


class Tracer:
    """Creates nested spans and hands finished spans to an exporter.

    Usage::

        tracer = Tracer(exporter)
        with tracer.start_span("agent.run", model="gpt-4o") as span:
            span.set_attribute("prompt_tokens", 120)
            with tracer.start_span("llm.call") as child:
                ...
    """

    def __init__(self, exporter: Optional[Exporter] = None) -> None:
        self._exporter = exporter
        self._trace_id = uuid.uuid4().hex

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def current_span(self) -> Optional[Span]:
        return _current_span.get()

    @contextlib.contextmanager
    def start_span(self, name: str, **attributes: Any) -> Iterator[Span]:
        parent = _current_span.get()
        span = Span(
            name=name,
            trace_id=self._trace_id,
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes),
        )
        token = _current_span.set(span)
        try:
            yield span
            span.finish(SpanStatus.OK)
        except BaseException as exc:
            span.finish(SpanStatus.ERROR, error=exc)
            raise
        finally:
            _current_span.reset(token)
            if self._exporter is not None:
                self._exporter.export(span)
