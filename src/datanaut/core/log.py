"""Pipeline log: an ordered, serializable record of every action taken on the data."""

from __future__ import annotations

import platform
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

import datanaut


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class StepLog(BaseModel):
    """One logged step: what was done, with which parameters, and its effect on row counts."""

    model_config = ConfigDict(frozen=True)

    step_id: int
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    summary: str
    counts: dict[str, float] = Field(default_factory=dict)
    rows_before: int | None = None
    rows_after: int | None = None
    read_only: bool = False
    created_at: str = Field(default_factory=_utc_now_iso)


class _LogDocument(BaseModel):
    metadata: dict[str, str]
    entries: list[StepLog]


def library_versions() -> dict[str, str]:
    """Return the versions of datanaut, Python, pandas and numpy."""
    return {
        "datanaut": datanaut.__version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


class PipelineLog:
    """Append-only list of StepLog entries with JSON round trip and version metadata."""

    def __init__(
        self,
        entries: list[StepLog] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._entries: list[StepLog] = list(entries or [])
        self.metadata: dict[str, str] = dict(metadata) if metadata else library_versions()

    def append(
        self,
        operation: str,
        summary: str,
        *,
        params: dict[str, Any] | None = None,
        counts: dict[str, float] | None = None,
        rows_before: int | None = None,
        rows_after: int | None = None,
        read_only: bool = False,
    ) -> StepLog:
        """Add an entry with the next step_id and return it."""
        entry = StepLog(
            step_id=len(self._entries) + 1,
            operation=operation,
            params=params or {},
            summary=summary,
            counts=counts or {},
            rows_before=rows_before,
            rows_after=rows_after,
            read_only=read_only,
        )
        self._entries.append(entry)
        return entry

    def extend(self, entries: list[StepLog]) -> list[StepLog]:
        """Append existing entries (for example from a loader result), renumbering them.

        Each entry keeps its original `created_at`, so the log records when work really happened.
        """
        added = []
        for entry in entries:
            renumbered = entry.model_copy(update={"step_id": len(self._entries) + 1})
            self._entries.append(renumbered)
            added.append(renumbered)
        return added

    @property
    def entries(self) -> list[StepLog]:
        """A copy of the entries in order."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[StepLog]:
        return iter(self._entries)

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize metadata and entries to JSON."""
        doc = _LogDocument(metadata=self.metadata, entries=self._entries)
        return doc.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> PipelineLog:
        """Rebuild a log from the output of to_json."""
        doc = _LogDocument.model_validate_json(text)
        return cls(entries=doc.entries, metadata=doc.metadata)
