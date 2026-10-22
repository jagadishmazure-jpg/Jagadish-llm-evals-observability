"""Judge interface shared by every scoring implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Judgment:
    """The outcome of scoring a single model response."""

    score: float
    label: str = ""
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Judge(Protocol):
    """Anything that can score a model response against a prompt."""

    def judge(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> Judgment:
        """Score ``response`` for ``prompt``.

        Returns a :class:`Judgment`. Implementations must be deterministic
        for identical inputs unless documented otherwise.
        """
