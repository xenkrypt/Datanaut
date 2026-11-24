"""The evidence block every analysis result carries, and the log entry that goes with it.

Evidence says exactly what a number is based on: the columns used, the time range analyzed,
how many observations went in, the method and its parameters. Analyses only describe
association in the data; they never state cause.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from datanaut.core.log import PipelineLog, StepLog
from datanaut.errors import AnalysisError


class Evidence(BaseModel):
    """What an analysis result is based on."""

    columns: list[str]
    start: str | None = None
    end: str | None = None
    n_observations: int
    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)


def jsonable(value: Any) -> Any:
    """Convert numpy, pandas and date values (also inside containers) to plain JSON types."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp | datetime | date):
        return str(value)
    if value is None or isinstance(value, bool | int | str):
        return value
    return str(value)


def make_evidence(
    index: pd.Index | Iterable[Any] | None,
    columns: list[str],
    n_observations: int,
    method: str,
    parameters: Mapping[str, Any] | None = None,
) -> Evidence:
    """Build an evidence block. `index` holds the timestamps of the observations used."""
    start = end = None
    if index is not None and len(index):
        idx = pd.Index(index)
        start, end = str(idx.min()), str(idx.max())
    return Evidence(
        columns=[str(c) for c in columns],
        start=start,
        end=end,
        n_observations=int(n_observations),
        method=method,
        parameters=jsonable(dict(parameters or {})),
    )


def analysis_log(
    operation: str,
    summary: str,
    evidence: Evidence,
    *,
    counts: Mapping[str, float] | None = None,
    rows: int | None = None,
) -> StepLog:
    """A read-only log entry for an analysis. Analyses never change the data."""
    return PipelineLog().append(
        operation,
        summary,
        params={"columns": evidence.columns, "method": evidence.method, **evidence.parameters},
        counts={
            k: float(v) for k, v in (counts or {"observations": evidence.n_observations}).items()
        },
        rows_before=rows,
        rows_after=rows,
        read_only=True,
    )


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """The column as floats, after checking that the frame has a time index and the column fits."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AnalysisError("This analysis needs a time-indexed table. Build the time index first.")
    if column not in df.columns:
        raise AnalysisError(f"Column '{column}' was not found in the data.")
    s = df[column]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        raise AnalysisError(f"Column '{column}' is not numeric.")
    return s.astype(float)


def finite_part(s: pd.Series) -> tuple[pd.Series, int]:
    """The finite values of a series, and how many values were left out (missing or infinite)."""
    keep = np.isfinite(s.to_numpy())
    return s[keep], int((~keep).sum())


_MAX_GRID_FACTOR = 50


def median_step(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    """The typical gap between distinct timestamps, or None if there are fewer than two."""
    unique = pd.DatetimeIndex(index.dropna().unique()).sort_values()
    if len(unique) < 2:
        return None
    return pd.Series(unique).diff().dropna().median()


def to_regular_grid(s: pd.Series) -> tuple[pd.Series, pd.Timedelta]:
    """Put a series on an even time grid at its typical step.

    Duplicate timestamps are averaged; each reading goes to the nearest grid time within half a
    step, and grid times with no reading are NaN. Raises AnalysisError if the series is too
    short or so sparse in time that the grid would be huge.
    """
    step = median_step(s.index)
    if step is None or step <= pd.Timedelta(0):
        raise AnalysisError("The series needs at least two distinct timestamps.")
    unique = s.groupby(level=0).mean().sort_index()
    grid = pd.date_range(unique.index[0], unique.index[-1], freq=step)
    if len(grid) > _MAX_GRID_FACTOR * len(unique) + 1000:
        raise AnalysisError("The series is too sparse in time for this analysis.")
    return unique.reindex(grid, method="nearest", tolerance=step / 2), step
