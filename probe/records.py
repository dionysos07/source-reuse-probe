"""Reading and writing the JSON files the stages hand to each other."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RecordError(Exception):
    """A results file is missing or unreadable, reported as a message."""


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def read_records(path: Path, what: str) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecordError(f"no {what} found at {path}; run that stage first") from exc
    except json.JSONDecodeError as exc:
        raise RecordError(f"{path} is not valid JSON: {exc}") from exc
