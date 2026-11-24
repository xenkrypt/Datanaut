"""Data profiler: a read-only diagnosis of a (time-indexed) DataFrame.

The profiler never changes the data. It returns a `DataProfile` that includes one read-only
log entry and a binned missingness matrix (a DataFrame, ready for any chart).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from datanaut.core.log import PipelineLog, StepLog

REGULAR_TOLERANCE = 0.1
"""An interval counts as regular if it is within this share of the median interval."""

GAP_FACTOR = 1.5
"""An interval longer than this many median intervals is a gap (missing expected timestamps)."""

HIGH_MISSING_PCT = 50.0
"""Columns missing more than this percentage get a warning."""

_CATEGORICAL_MAX_UNIQUE = 50
_CATEGORICAL_MAX_SHARE = 0.5


class ColumnProfile(BaseModel):
    """Facts about one column."""

    name: str
    dtype: str
    inferred_type: Literal["numeric", "boolean", "datetime", "categorical", "text", "empty"]
    missing_count: int
    missing_pct: float
    unique_count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    skew: float | None = None
    missing_runs: int = 0
    longest_missing_run: int = 0


class SamplingProfile(BaseModel):
    """How regularly the index is spaced (based on distinct timestamps)."""

    median_interval_seconds: float
    median_interval_label: str
    regular_share: float
    irregular_count: int
    off_grid_count: int = 0  # irregular intervals that are not gaps (jitter, extra readings)


class Gap(BaseModel):
    """A stretch of missing expected timestamps: first and last missing timestamp."""

    start: str
    end: str
    missing_timestamps: int
    duration_seconds: float


class GapProfile(BaseModel):
    """Missing expected timestamps relative to the median interval."""

    gap_count: int
    missing_timestamps: int
    longest_gap_timestamps: int
    longest_gap_seconds: float
    largest: list[Gap] = Field(default_factory=list)


class DuplicateProfile(BaseModel):
    """Duplicate rows, and duplicate timestamps split by whether the values agree."""

    full_row_duplicates: int
    duplicate_timestamps: int | None = None
    duplicate_timestamps_identical: int | None = None
    duplicate_timestamps_conflicting: int | None = None


class DataProfile(BaseModel):
    """The complete profile of a dataset."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rows: int
    n_columns: int
    memory_bytes: int
    columns: list[ColumnProfile]
    sampling: SamplingProfile | None = None
    gaps: GapProfile | None = None
    duplicates: DuplicateProfile
    constant_columns: list[str] = Field(default_factory=list)
    empty_columns: list[str] = Field(default_factory=list)
    index_monotonic: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    missingness_matrix: pd.DataFrame = Field(exclude=True)
    log_entry: StepLog

    def column(self, name: str) -> ColumnProfile:
        """Look up one column's profile by name."""
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


def profile(df: pd.DataFrame, *, max_bins: int = 50) -> DataProfile:
    """Profile `df` without changing it. `max_bins` caps the missingness matrix rows."""
    n_rows = len(df)
    columns = [_profile_column(str(name), df[name]) for name in df.columns]
    is_time = isinstance(df.index, pd.DatetimeIndex)

    sampling, gaps = _sampling_and_gaps(df.index) if is_time else (None, None)
    duplicates = _duplicates(df, is_time)
    constant = [c.name for c in columns if c.unique_count == 1]
    empty = [c.name for c in columns if c.missing_count == n_rows and n_rows > 0]
    monotonic = bool(df.index.is_monotonic_increasing) if is_time else None
    memory = int(df.memory_usage(deep=True, index=True).sum())

    warnings = _warnings(columns, sampling, gaps, duplicates, constant, empty, monotonic)
    log = PipelineLog()
    missing_cells = sum(c.missing_count for c in columns)
    entry = log.append(
        "profile",
        f"Profiled {n_rows} rows x {len(columns)} columns: {missing_cells} missing cell(s), "
        f"{duplicates.full_row_duplicates} duplicate row(s), {len(warnings)} warning(s). "
        "No data was changed.",
        params={"max_bins": max_bins},
        counts={
            "rows": n_rows,
            "columns": len(columns),
            "missing_cells": missing_cells,
            "full_row_duplicates": duplicates.full_row_duplicates,
            "duplicate_timestamps": duplicates.duplicate_timestamps or 0,
            "missing_expected_timestamps": gaps.missing_timestamps if gaps else 0,
            "warnings": len(warnings),
        },
        rows_before=n_rows,
        rows_after=n_rows,
        read_only=True,
    )
    return DataProfile(
        rows=n_rows,
        n_columns=len(columns),
        memory_bytes=memory,
        columns=columns,
        sampling=sampling,
        gaps=gaps,
        duplicates=duplicates,
        constant_columns=constant,
        empty_columns=empty,
        index_monotonic=monotonic,
        warnings=warnings,
        missingness_matrix=_missingness_matrix(df, max_bins),
        log_entry=entry,
    )


# --------------------------------------------------------------------------------- columns


def _num(value: Any) -> float | None:
    """A finite float, or None."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _infer_type(s: pd.Series, unique: int, present: int) -> str:
    if present == 0:
        return "empty"
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if unique <= _CATEGORICAL_MAX_UNIQUE and unique / present <= _CATEGORICAL_MAX_SHARE:
        return "categorical"
    return "text"


def _missing_runs(mask: np.ndarray) -> tuple[int, int]:
    """Number of runs of True and the length of the longest."""
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return 0, 0
    return len(starts), int((ends - starts).max())


def _profile_column(name: str, s: pd.Series) -> ColumnProfile:
    n = len(s)
    mask = s.isna().to_numpy()
    missing = int(mask.sum())
    present = n - missing
    try:
        unique = int(s.nunique(dropna=True))
    except TypeError:  # unhashable cells
        unique = int(s.astype(str).nunique())
    kind = _infer_type(s, unique, present)
    stats: dict[str, float | None] = {}
    if kind == "numeric":
        values = s.dropna().astype(float)
        stats = {
            "min": _num(values.min()),
            "max": _num(values.max()),
            "mean": _num(values.mean()),
            "std": _num(values.std()),
            "skew": _num(values.skew()) if present >= 3 else None,
        }
    runs, longest = _missing_runs(mask)
    return ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        inferred_type=kind,
        missing_count=missing,
        missing_pct=round(100.0 * missing / n, 4) if n else 0.0,
        unique_count=unique,
        missing_runs=runs,
        longest_missing_run=longest,
        **stats,
    )


# ------------------------------------------------------------------- sampling, gaps, dupes


def interval_label(seconds: float) -> str:
    for size, unit in ((86400, "d"), (3600, "h"), (60, "min")):
        if seconds >= size and seconds % size == 0:
            return f"{int(seconds // size)}{unit}"
    return f"{seconds:g}s"


def _sampling_and_gaps(index: pd.DatetimeIndex) -> tuple[SamplingProfile | None, GapProfile | None]:
    stamps = pd.DatetimeIndex(index.dropna().unique()).sort_values()
    if len(stamps) < 2:
        return None, None
    diffs = stamps.to_series().diff().dt.total_seconds().dropna().to_numpy()
    median = float(np.median(diffs))
    if median <= 0:
        return None, None
    regular = np.abs(diffs - median) <= REGULAR_TOLERANCE * median
    sampling = SamplingProfile(
        median_interval_seconds=median,
        median_interval_label=interval_label(median),
        regular_share=round(float(regular.mean()), 6),
        irregular_count=int((~regular).sum()),
        off_grid_count=int((~regular & ~(diffs > GAP_FACTOR * median)).sum()),
    )
    positions = np.flatnonzero(diffs > GAP_FACTOR * median)
    steps = np.maximum(np.rint(diffs[positions] / median).astype(int) - 1, 1)
    order = np.lexsort((positions, -steps))[:5]
    step = pd.Timedelta(seconds=median)
    largest = [
        Gap(
            start=str(stamps[positions[i]] + step),
            end=str(stamps[positions[i] + 1] - step),
            missing_timestamps=int(steps[i]),
            duration_seconds=float(steps[i] * median),
        )
        for i in order
    ]
    gaps = GapProfile(
        gap_count=len(positions),
        missing_timestamps=int(steps.sum()),
        longest_gap_timestamps=int(steps.max()) if len(steps) else 0,
        longest_gap_seconds=float(steps.max() * median) if len(steps) else 0.0,
        largest=largest,
    )
    return sampling, gaps


def _duplicates(df: pd.DataFrame, is_time: bool) -> DuplicateProfile:
    if not is_time:
        return DuplicateProfile(full_row_duplicates=int(df.duplicated().sum()))
    with_index = df.reset_index(drop=True)
    with_index.insert(0, "__datanaut_index__", df.index.array)
    full = with_index.duplicated(keep="first").to_numpy()
    same_ts = df.index.duplicated(keep="first")
    identical = int((same_ts & full).sum())
    return DuplicateProfile(
        full_row_duplicates=int(full.sum()),
        duplicate_timestamps=int(same_ts.sum()),
        duplicate_timestamps_identical=identical,
        duplicate_timestamps_conflicting=int(same_ts.sum()) - identical,
    )


# -------------------------------------------------------------------------- missingness map


def _missingness_matrix(df: pd.DataFrame, max_bins: int) -> pd.DataFrame:
    """Fraction of missing cells per column in each bin (NaN where a bin has no rows)."""
    n = len(df)
    if n == 0 or df.shape[1] == 0 or max_bins < 1:
        return pd.DataFrame(columns=[str(c) for c in df.columns], dtype=float)
    bins = min(max_bins, n)
    labels: pd.Index
    ids: np.ndarray
    if isinstance(df.index, pd.DatetimeIndex) and df.index.notna().all() and n > 1:
        t = df.index.asi8.astype(float)
        span = t.max() - t.min()
    else:
        span = 0.0
    if span > 0:
        ids = np.minimum(((t - t.min()) / span * bins).astype(int), bins - 1)
        width = (df.index.max() - df.index.min()).total_seconds() / bins
        labels = pd.DatetimeIndex(
            df.index.min() + pd.to_timedelta(np.arange(bins) * width, unit="s"),
            name=df.index.name,
        )
    else:
        ids = np.arange(n) * bins // n
        labels = pd.Index(np.arange(bins) * n // bins, name="row")
    missing = df.isna().astype(float)
    matrix = missing.groupby(ids).mean().reindex(range(bins))
    matrix.index = labels
    matrix.columns = [str(c) for c in df.columns]
    return matrix


# --------------------------------------------------------------------------------- warnings


def _warnings(
    columns: list[ColumnProfile],
    sampling: SamplingProfile | None,
    gaps: GapProfile | None,
    duplicates: DuplicateProfile,
    constant: list[str],
    empty: list[str],
    monotonic: bool | None,
) -> list[str]:
    out: list[str] = []
    for c in columns:
        if c.name in empty:
            out.append(f"Column '{c.name}' is entirely missing.")
        elif c.missing_pct > HIGH_MISSING_PCT:
            out.append(f"Column '{c.name}' is {c.missing_pct:.1f}% missing.")
    for name in constant:
        out.append(f"Column '{name}' is constant (a single value).")
    if monotonic is False:
        out.append("The time index is not in chronological order.")
    if sampling is not None and sampling.off_grid_count:
        out.append(
            f"Irregular sampling: {sampling.off_grid_count} interval(s) are neither the typical "
            f"{sampling.median_interval_label} nor a gap ({sampling.regular_share:.1%} regular)."
        )
    if gaps is not None and gaps.gap_count:
        out.append(
            f"{gaps.missing_timestamps} expected timestamp(s) are missing in {gaps.gap_count} "
            f"gap(s); the longest is {gaps.longest_gap_timestamps} timestamp(s)."
        )
    if duplicates.duplicate_timestamps_conflicting:
        out.append(
            f"{duplicates.duplicate_timestamps_conflicting} duplicate timestamp(s) have "
            "conflicting values."
        )
    if duplicates.full_row_duplicates:
        out.append(f"{duplicates.full_row_duplicates} fully duplicated row(s).")
    long_runs = [c for c in columns if c.longest_missing_run >= 24 and c.name not in empty]
    for c in long_runs:
        out.append(
            f"Column '{c.name}' has a run of {c.longest_missing_run} consecutive missing values."
        )
    return out
