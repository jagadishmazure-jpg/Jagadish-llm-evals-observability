"""Load evaluation datasets from JSONL or JSON files.

Expected row format::

    {"id": "case-1", "input": "What is 2+2?", "expected": "4"}

``expected`` is optional (judges can score without a reference).
``metadata`` is an optional free-form dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union


class DatasetError(ValueError):
    """Raised when a dataset file cannot be loaded or validated."""


@dataclass(frozen=True)
class EvalCase:
    id: str
    input: str
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _row_to_case(row: Any, lineno: int | None) -> EvalCase:
    where = f"line {lineno}" if lineno is not None else "entry"
    if not isinstance(row, dict):
        raise DatasetError(f"{where}: expected an object, got {type(row).__name__}")
    case_id = row.get("id")
    text = row.get("input")
    if not isinstance(case_id, str) or not case_id:
        raise DatasetError(f"{where}: 'id' must be a non-empty string")
    if not isinstance(text, str):
        raise DatasetError(f"{where}: 'input' must be a string")
    expected = row.get("expected")
    if expected is not None and not isinstance(expected, str):
        raise DatasetError(f"{where}: 'expected' must be a string when present")
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"{where}: 'metadata' must be an object when present")
    return EvalCase(id=case_id, input=text, expected=expected, metadata=metadata)


def load_dataset(path: Union[str, Path]) -> list[EvalCase]:
    """Load cases from a ``.jsonl`` (one JSON object per line) or ``.json`` file.

    Blank lines in JSONL are skipped. Raises :class:`DatasetError` for
    missing files, invalid JSON, or invalid rows. Duplicate ids raise
    :class:`DatasetError`.
    """
    file = Path(path)
    if not file.is_file():
        raise DatasetError(f"dataset not found: {file}")
    suffix = file.suffix.lower()
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"cannot read dataset {file}: {exc}") from exc

    rows: list[tuple[Any, int | None]] = []
    if suffix == ".jsonl":
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    rows.append((json.loads(line), lineno))
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"line {lineno}: invalid JSON: {exc}"
                    ) from exc
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON in {file}: {exc}") from exc
        if not isinstance(data, list):
            raise DatasetError(f"{file}: top-level JSON must be a list of cases")
        rows = [(row, None) for row in data]
    else:
        raise DatasetError(
            f"unsupported dataset extension {suffix!r}; use .jsonl or .json"
        )

    cases = [_row_to_case(row, lineno) for row, lineno in rows]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise DatasetError(f"duplicate case ids: {dupes}")
    return cases
