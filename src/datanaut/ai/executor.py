"""Validate a structured query and run it with pandas.

The pandas operation is built only from validated, allowlisted fields. There is no path from a
string to code: no eval, exec, compile or string-based query expression anywhere in this package.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from datanaut.ai import config
from datanaut.ai.schemas import (
    FIXED_GROUPS,
    VIRTUAL_COLUMNS,
    Filter,
    QuerySpec,
    ResultTable,
)
from datanaut.ai.text import sanitize_text
from datanaut.errors import QueryError

OPERATORS = ("==", "!=", "<", "<=", ">", ">=", "between", "in")
AGGREGATIONS = ("mean", "sum", "min", "max", "count", "median", "std", "quantile")
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
MAX_VALUE_COLUMNS = 8
_VIRTUAL_RANGE = {"hour": (0, 23), "weekday": (0, 6), "month": (1, 12)}

_COMPARE: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
_AGGREGATE: dict[str, Callable[[Any, float | None], Any]] = {
    "mean": lambda g, q: g.mean(),
    "sum": lambda g, q: g.sum(),
    "min": lambda g, q: g.min(),
    "max": lambda g, q: g.max(),
    "count": lambda g, q: g.count(),
    "median": lambda g, q: g.median(),
    "std": lambda g, q: g.std(),
    "quantile": lambda g, q: g.quantile(q),
}


# ------------------------------------------------------------------------------------ schema


@dataclass(frozen=True)
class ColumnInfo:
    """A column as the model sees it (`name`, cleaned) and as it is in the data (`real`)."""

    name: str
    real: str
    kind: str  # "numeric" or "categorical"


@dataclass(frozen=True)
class DataSchema:
    """The columns a query may use, and the time range of the data."""

    columns: dict[str, ColumnInfo]
    start: pd.Timestamp
    end: pd.Timestamp
    n_rows: int

    def numeric(self) -> list[str]:
        return [c.name for c in self.columns.values() if c.kind == "numeric"]

    def categorical(self) -> list[str]:
        return [c.name for c in self.columns.values() if c.kind == "categorical"]


def schema_from_frame(df: pd.DataFrame) -> DataSchema:
    """Describe a time-indexed table. Column names are cleaned and made unique."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        raise QueryError("The data needs a time index and at least one row.")
    columns: dict[str, ColumnInfo] = {}
    for real in df.columns:
        series = df[real]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        name = base = sanitize_text(real, config.MAX_COLUMN_NAME) or "column"
        n = 2
        while name in columns:
            name, n = f"{base}_{n}", n + 1
        columns[name] = ColumnInfo(name, str(real), "numeric" if numeric else "categorical")
    return DataSchema(columns, df.index.min(), df.index.max(), len(df))


# --------------------------------------------------------------------------------- validation


def parse_query(raw: dict[str, Any]) -> QuerySpec:
    """Build a QuerySpec from a model's arguments, with errors the model can read."""
    try:
        return QuerySpec.model_validate(raw)
    except ValidationError as exc:
        problems = []
        for err in exc.errors()[:8]:
            where = ".".join(str(p) for p in err["loc"]) or "query"
            if err["type"] == "literal_error":
                problems.append(f"{where}: not allowed; expected {err['ctx']['expected']}")
            elif err["type"] == "extra_forbidden":
                problems.append(f"{where}: unknown field")
            else:
                problems.append(f"{where}: {err['msg']}")
        raise QueryError("Invalid query: " + "; ".join(problems) + ".") from exc


def _number(text: str | None) -> float | None:
    try:
        value = float(str(text))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _check_filter(f: Filter, schema: DataSchema, where: str, problems: list[str]) -> None:
    info = schema.columns.get(f.column)
    virtual = info is None and f.column in VIRTUAL_COLUMNS
    if info is None and not virtual:
        known = ", ".join(list(schema.columns) + list(VIRTUAL_COLUMNS))
        problems.append(f"{where}: unknown column '{f.column}'. Known columns: {known}")
        return
    numeric = virtual or (info is not None and info.kind == "numeric")
    if f.operator in ("between", "in"):
        if f.values is None or f.value is not None:
            problems.append(f"{where}: '{f.operator}' needs 'values' (a list) and no 'value'")
            return
        if f.operator == "between" and len(f.values) != 2:
            problems.append(f"{where}: 'between' needs exactly two values")
            return
        if f.operator == "in" and not 1 <= len(f.values) <= config.MAX_IN_VALUES:
            problems.append(f"{where}: 'in' needs 1 to {config.MAX_IN_VALUES} values")
            return
        items = f.values
    else:
        if f.value is None or f.values is not None:
            problems.append(f"{where}: '{f.operator}' needs a single 'value' and no 'values'")
            return
        items = [f.value]
    if not numeric and f.operator not in ("==", "!=", "in"):
        problems.append(f"{where}: '{f.operator}' only works on numeric columns")
        return
    if numeric:
        numbers = [_number(v) for v in items]
        if any(n is None for n in numbers):
            problems.append(f"{where}: '{f.column}' is numeric, so values must be numbers")
            return
        if f.operator == "between" and numbers[0] > numbers[1]:  # type: ignore[operator]
            problems.append(f"{where}: 'between' needs the lower value first")
        if virtual:
            lo, hi = _VIRTUAL_RANGE[f.column]
            if any(not float(n).is_integer() or not lo <= n <= hi for n in numbers):  # type: ignore[arg-type]
                problems.append(f"{where}: '{f.column}' must be a whole number from {lo} to {hi}")


def _check_scope(spec: QuerySpec, schema: DataSchema, problems: list[str]) -> None:
    scope = spec.time_scope
    if scope is None:
        return
    lo, hi = schema.start.normalize(), schema.end.normalize()
    stamps: dict[str, pd.Timestamp] = {}
    for label, text in (("start", scope.start), ("end", scope.end)):
        if text is None:
            continue
        try:
            stamp = pd.Timestamp(text)
        except (ValueError, TypeError):
            problems.append(f"time_scope.{label}: '{sanitize_text(text, 40)}' is not a date")
            continue
        if stamp.tzinfo is not None:
            stamp = stamp.tz_localize(None)
        if not lo <= stamp.normalize() <= hi:
            problems.append(
                f"time_scope.{label}: {stamp.date()} is outside the data, which runs from "
                f"{lo.date()} to {hi.date()}"
            )
        stamps[label] = stamp
    if len(stamps) == 2 and stamps["start"] > stamps["end"]:
        problems.append("time_scope: start is after end")


def validate_query(spec: QuerySpec, schema: DataSchema) -> None:
    """Reject anything outside the allowlist. Raises QueryError listing every problem found."""
    problems: list[str] = []
    cols = spec.value_columns
    if len(cols) > MAX_VALUE_COLUMNS:
        problems.append(f"value_columns: at most {MAX_VALUE_COLUMNS} columns")
    for c in cols:
        info = schema.columns.get(c)
        if info is None:
            problems.append(
                f"value_columns: unknown column '{sanitize_text(c, 64)}'. "
                f"Known columns: {', '.join(schema.columns)}"
            )
        elif info.kind != "numeric" and spec.aggregation != "count":
            problems.append(f"value_columns: '{c}' is not numeric; only 'count' works on it")
    if spec.aggregation not in AGGREGATIONS:
        problems.append(f"aggregation: not allowed; use one of {', '.join(AGGREGATIONS)}")
    if spec.aggregation == "quantile":
        if spec.q is None or not 0 < spec.q < 1:
            problems.append("q: 'quantile' needs q strictly between 0 and 1")
    elif spec.q is not None:
        problems.append("q: only allowed with the 'quantile' aggregation")
    if not 1 <= spec.limit <= config.MAX_LIMIT:
        problems.append(f"limit: must be from 1 to {config.MAX_LIMIT}")
    if len(spec.filters) > config.MAX_FILTERS:
        problems.append(f"filters: at most {config.MAX_FILTERS}")
    for i, f in enumerate(spec.filters):
        _check_filter(f, schema, f"filters[{i}]", problems)
    _check_scope(spec, schema, problems)
    group = spec.group_by
    if group not in FIXED_GROUPS and group not in schema.categorical():
        options = ", ".join(FIXED_GROUPS + tuple(schema.categorical()))
        problems.append(
            f"group_by: '{sanitize_text(group, 64)}' is not allowed; use one of {options}"
        )
    if spec.compare is not None:
        if group != "none":
            problems.append("compare: group_by must be 'none' when comparing two groups")
        for side, filters in (("a", spec.compare.filters_a), ("b", spec.compare.filters_b)):
            if not filters:
                problems.append(f"compare.filters_{side}: needs at least one filter")
            for i, f in enumerate(filters):
                _check_filter(f, schema, f"compare.filters_{side}[{i}]", problems)
        for label in (spec.compare.label_a, spec.compare.label_b):
            if not label.strip() or len(label) > 40:
                problems.append("compare: labels must be 1 to 40 characters")
    if problems:
        raise QueryError("Invalid query: " + "; ".join(problems) + ".")


# ---------------------------------------------------------------------------------- execution


def _filter_values(df: pd.DataFrame, f: Filter, schema: DataSchema) -> pd.Series:
    info = schema.columns.get(f.column)
    if info is not None:
        return df[info.real]
    index = df.index
    if f.column == "hour":
        return pd.Series(index.hour, index=index)
    if f.column == "weekday":
        return pd.Series(index.day_of_week, index=index)
    return pd.Series(index.month, index=index)


def _filter_mask(df: pd.DataFrame, f: Filter, schema: DataSchema) -> pd.Series:
    series = _filter_values(df, f, schema)
    info = schema.columns.get(f.column)
    numeric = info is None or info.kind == "numeric"
    if numeric:
        convert: Callable[[str], Any] = float
    else:
        convert = str
        series = (
            series.astype(object)
            .where(series.notna(), None)
            .map(lambda v: None if v is None else str(v))
        )
    if f.operator == "between":
        lo, hi = (convert(v) for v in f.values or [])
        return series.between(lo, hi)
    if f.operator == "in":
        return series.isin({convert(v) for v in f.values or []})
    result = _COMPARE[f.operator](series, convert(str(f.value)))
    return result.fillna(False).astype(bool)


def _apply_filters(df: pd.DataFrame, filters: list[Filter], schema: DataSchema) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for f in filters:
        mask &= _filter_mask(df, f, schema)
    return df[mask]


def _scope(df: pd.DataFrame, spec: QuerySpec) -> pd.DataFrame:
    scope = spec.time_scope
    if scope is None:
        return df
    mask = pd.Series(True, index=df.index)
    if scope.start:
        mask &= df.index >= pd.Timestamp(scope.start).normalize()
    if scope.end:
        mask &= df.index < pd.Timestamp(scope.end).normalize() + pd.Timedelta(days=1)
    return df[mask]


def _native(value: Any) -> Any:
    if value is None or (isinstance(value, float | np.floating) and not math.isfinite(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _group_key(sub: pd.DataFrame, group: str, schema: DataSchema) -> pd.Series | np.ndarray:
    index = sub.index
    if group == "hour":
        return index.hour.to_numpy()
    if group == "weekday":
        return index.day_of_week.to_numpy()
    if group == "date":
        return index.normalize().to_numpy()
    if group == "week":
        return (index.normalize() - pd.to_timedelta(index.day_of_week, unit="D")).to_numpy()
    if group == "month":
        return index.to_period("M").astype(str).to_numpy()
    return sub[schema.columns[group].real].astype(str).to_numpy()


def _label(group: str, key: Any) -> str:
    if group == "hour":
        return f"{int(key):02d}:00"
    if group == "weekday":
        return WEEKDAYS[int(key)]
    if group in ("date", "week"):
        return str(pd.Timestamp(key).date())
    return str(key)


def execute_query(
    df: pd.DataFrame, spec: QuerySpec, schema: DataSchema | None = None
) -> ResultTable:
    """Run a validated query. Raises QueryError if it is not allowed or no rows match."""
    schema = schema or schema_from_frame(df)
    validate_query(spec, schema)
    real = [schema.columns[c].real for c in spec.value_columns]
    base = _apply_filters(_scope(df, spec), spec.filters, schema)
    if base.empty:
        raise QueryError("No rows match these filters and dates. Relax them and try again.")
    q = spec.q if spec.aggregation == "quantile" else None
    agg = _AGGREGATE[spec.aggregation]

    rows: list[list[Any]] = []
    counts: list[int] = []
    if spec.compare is not None:
        label_col = "group"
        for label, filters in (
            (spec.compare.label_a, spec.compare.filters_a),
            (spec.compare.label_b, spec.compare.filters_b),
        ):
            part = _apply_filters(base, filters, schema)
            values = agg(part[real], q) if len(part) else pd.Series(np.nan, index=real)
            rows.append([sanitize_text(label, 40), *(_native(values[c]) for c in real)])
            counts.append(len(part))
    elif spec.group_by == "none":
        label_col = "all"
        values = agg(base[real], q)
        rows.append(["all rows", *(_native(values[c]) for c in real)])
        counts.append(len(base))
    else:
        label_col = spec.group_by
        key = _group_key(base, spec.group_by, schema)
        grouped = base[real].groupby(key, sort=True)
        values = agg(grouped, q)
        sizes = base.groupby(key, sort=True).size()
        for k in values.index:
            rows.append([_label(spec.group_by, k), *(_native(values.loc[k, c]) for c in real)])
            counts.append(int(sizes.loc[k]))

    if spec.sort != "none" and len(rows) > 1:
        order = sorted(
            range(len(rows)),
            key=lambda i: (rows[i][1] is None, rows[i][1] if rows[i][1] is not None else 0.0),
            reverse=False,
        )
        if spec.sort == "desc":
            present = [i for i in order if rows[i][1] is not None][::-1]
            order = present + [i for i in order if rows[i][1] is None]
        rows, counts = [rows[i] for i in order], [counts[i] for i in order]
    truncated = len(rows) > spec.limit
    return ResultTable(
        columns=[label_col, *spec.value_columns],
        rows=rows[: spec.limit],
        counts=counts[: spec.limit],
        truncated=truncated,
        n_observations=len(base),
    )


def observed_range(df: pd.DataFrame, spec: QuerySpec, schema: DataSchema | None = None):
    """The first and last timestamp of the rows a query used (for the evidence card)."""
    schema = schema or schema_from_frame(df)
    base = _apply_filters(_scope(df, spec), spec.filters, schema)
    if base.empty:
        return None, None
    return base.index.min(), base.index.max()
