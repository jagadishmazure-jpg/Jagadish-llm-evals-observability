"""Span and Trace dataclasses."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class SpanStatus(enum.Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class SpanEvent:
    """Timestamped annotation attached to a span."""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """One unit of work inside a trace."""

    name: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.OK
    error_type: str | None = None
    error_message: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    def finish(
        self,
        status: SpanStatus = SpanStatus.OK,
        error: BaseException | None = None,
    ) -> None:
        """Close the span, recording status and optional error details."""
        self.end_time = time.time()
        self.status = status
        if error is not None:
            self.error_type = type(error).__name__
            self.error_message = str(error)

    def add_event(self, name: str, **attributes: Any) -> None:
        self.events.append(SpanEvent(name=name, attributes=dict(attributes)))

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    @property
    def duration_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


@dataclass(frozen=True)
class Trace:
    """A completed trace: its id plus every finished span."""

    trace_id: str
    spans: tuple[Span, ...]

    @property
    def root(self) -> Span | None:
        for span in self.spans:
            if span.is_root:
                return span
        return None

    def spans_named(self, name: str) -> list[Span]:
        return [s for s in self.spans if s.name == name]
