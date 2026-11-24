"""Data shapes for the AI analyst: the structured query, its result and the answer.

The query is the only thing the model can ask for. Everything in it is checked against an
allowlist (`executor.validate_query`) before any pandas operation is built.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from datanaut.ai import config

Operator = Literal["==", "!=", "<", "<=", ">", ">=", "between", "in"]
Aggregation = Literal["mean", "sum", "min", "max", "count", "median", "std", "quantile"]
Sort = Literal["none", "asc", "desc"]
FIXED_GROUPS = ("none", "hour", "weekday", "date", "week", "month")
VIRTUAL_COLUMNS = ("hour", "weekday", "month")
"""Calendar fields that can be filtered on (weekday: Monday is 0, Sunday is 6)."""


def _as_text(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return repr(value)
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Filter(_Strict):
    """Keep rows where `column <operator> value`. `between` and `in` use `values`."""

    column: str
    operator: Operator
    value: str | None = None
    values: list[str] | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _value_as_text(cls, v: Any) -> Any:
        return _as_text(v)

    @field_validator("values", mode="before")
    @classmethod
    def _values_as_text(cls, v: Any) -> Any:
        return [_as_text(x) for x in v] if isinstance(v, list | tuple) else v


class TimeScope(_Strict):
    """Keep rows from `start` to `end` (dates, both inclusive)."""

    start: str | None = None
    end: str | None = None


class Compare(_Strict):
    """Two filtered groups, aggregated separately (for example weekday against weekend)."""

    label_a: str = "A"
    filters_a: list[Filter] = Field(default_factory=list)
    label_b: str = "B"
    filters_b: list[Filter] = Field(default_factory=list)


class QuerySpec(_Strict):
    """A structured question about the data."""

    value_columns: list[str] = Field(min_length=1)
    filters: list[Filter] = Field(default_factory=list)
    time_scope: TimeScope | None = None
    group_by: str = "none"
    aggregation: Aggregation = "mean"
    q: float | None = None
    sort: Sort = "none"
    limit: int = config.DEFAULT_LIMIT
    compare: Compare | None = None


class ResultTable(BaseModel):
    """What the query computed. `counts` holds the rows that went into each result row."""

    columns: list[str]
    rows: list[list[Any]]
    counts: list[int] = Field(default_factory=list)
    truncated: bool = False
    n_observations: int = 0


class ChartHint(BaseModel):
    """Which chart suits the result (drawn by the UI from the table)."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvidenceCard(BaseModel):
    """Everything behind an answer: what was asked, what was computed and from how much data."""

    columns: list[str]
    start: str | None = None
    end: str | None = None
    n_observations: int
    method: str
    table: ResultTable
    query: dict[str, Any]
    chart: ChartHint


class AnalystAnswer(BaseModel):
    """The reply to one question."""

    explanation: str
    evidence: EvidenceCard | None = None
    provider: str | None = None
    fell_back: bool = False
    rewritten: bool = False
    """The number check found a figure that is not in the result, so the text was rewritten."""
    templated: bool = False
    """The model's text failed the number check twice; Python wrote the summary instead."""
    refused: bool = False
    cached: bool = False
    error: str | None = None


class Suggestion(BaseModel):
    """A suggested next step that points at something real (a finding or a log entry)."""

    title: str
    rationale: str
    kind: Literal["cleaning", "anomaly_hypothesis", "peak_reduction"]
    finding_id: str | None = None
    log_entry_id: int | None = None
    action: str = ""
