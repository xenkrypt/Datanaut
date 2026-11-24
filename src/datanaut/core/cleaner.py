"""Cleaner: parameterized, previewable, replayable fixes for time-series data.

Every fix is an `OpSpec` (operation name plus a JSON-serializable params dict) applied with
`apply_op(df, spec)`, which returns a new DataFrame and one log entry. The input is never
modified, and anything skipped, clipped or left unfilled is counted in the log entry.
Because specs are plain data, a chain of them can be replayed later (see versioning).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from statsmodels.tsa.seasonal import STL

from datanaut.core.characterize import SeriesCharacterization
from datanaut.core.log import PipelineLog, StepLog
from datanaut.core.profiler import DataProfile
from datanaut.errors import CleaningError

MAD_TO_SIGMA = 1.4826
"""Scales a median absolute deviation to a standard deviation for normal data."""

FLAG_SUFFIX = "__outlier"
"""Suffix of the boolean column added by outliers(action='flag')."""

DEFAULT_MAX_GAP = 6
CONSUMPTION_HINTS = ("consum", "energy", "kwh", "power", "demand", "usage")
_UNIT_SCALE_RATIO = 1000.0
_STAT_COLUMN_CAP = 12
_CHUNK_CELLS = 2_000_000  # window cells held at once by the Hampel filter
MAX_WINDOW = 501


# ------------------------------------------------------------------------------- spec and result


class OpSpec(BaseModel):
    """One cleaning operation: its name and parameters. Plain data, safe to store as JSON."""

    operation: str
    params: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class CleanResult:
    """A new DataFrame and the log entry describing exactly what happened."""

    df: pd.DataFrame
    log_entry: StepLog


@dataclass
class Outcome:
    df: pd.DataFrame
    summary: str
    counts: dict[str, float] = field(default_factory=dict)
    read_only: bool = False
    params_extra: dict[str, Any] = field(default_factory=dict)  # logged next to the params


_Handler = Callable[[pd.DataFrame, Any], Outcome]


def _operations() -> dict[str, tuple[type[BaseModel], _Handler]]:
    """Every operation apply_op accepts, built on each call so no module state is mutated."""
    from datanaut.core.validation import RULE_ACTION  # lazy: validation imports this module

    operations: dict[str, tuple[type[BaseModel], _Handler]] = {
        "drop_duplicates": (DropDuplicatesParams, _drop_duplicates),
        "handle_missing": (HandleMissingParams, _handle_missing),
        "outliers": (OutliersParams, _outliers),
        "resample": (ResampleParams, _resample),
        "check_warnings": (CheckWarningsParams, _check_warnings),
    }
    name, params_model, handler = RULE_ACTION
    operations[name] = (params_model, handler)
    return operations


def available_operations() -> list[str]:
    """Names of all operations apply_op accepts."""
    return sorted(_operations())


def apply_op(df: pd.DataFrame, spec: OpSpec) -> CleanResult:
    """Apply one operation to a copy of `df`. Raises CleaningError for bad params or data."""
    operations = _operations()
    if spec.operation not in operations:
        raise CleaningError(
            f"Unknown operation '{spec.operation}'. Available: {', '.join(sorted(operations))}."
        )
    model, handler = operations[spec.operation]
    try:
        params = model.model_validate(spec.params)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "params"
        raise CleaningError(
            f"Invalid {spec.operation} parameter '{where}': {first['msg']}"
        ) from exc
    outcome = handler(df, params)
    entry = PipelineLog().append(
        spec.operation,
        outcome.summary,
        params={**params.model_dump(), **outcome.params_extra},
        counts=outcome.counts,
        rows_before=len(df),
        rows_after=len(outcome.df),
        read_only=outcome.read_only,
    )
    return CleanResult(df=outcome.df, log_entry=entry)


def _require_column(df: pd.DataFrame, name: str, *, numeric: bool = False) -> None:
    if name not in df.columns:
        raise CleaningError(f"Column '{name}' was not found in the data.")
    if numeric and (
        not pd.api.types.is_numeric_dtype(df[name]) or pd.api.types.is_bool_dtype(df[name])
    ):
        raise CleaningError(f"Column '{name}' is not numeric.")


def _require_time_index(df: pd.DataFrame, *, monotonic: bool = True) -> pd.DatetimeIndex:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise CleaningError("This step needs a time index. Build the time index first.")
    if monotonic and not df.index.is_monotonic_increasing:
        raise CleaningError("The time index is not in chronological order. Sort it first.")
    return df.index


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) positions of each run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# ----------------------------------------------------------------------------- drop_duplicates


class DropDuplicatesParams(BaseModel):
    """keep: which exact duplicate row survives. conflicts: how same-timestamp rows with
    different values are resolved ('none' leaves them and reports them)."""

    model_config = ConfigDict(extra="forbid")

    keep: Literal["first", "last"] = "first"
    conflicts: Literal["none", "first", "last", "mean", "median", "sum"] = "none"


def _drop_duplicates(df: pd.DataFrame, p: DropDuplicatesParams) -> Outcome:
    n_before = len(df)
    frame = df.reset_index(drop=True)
    frame.insert(0, "__datanaut_index__", df.index.array)
    exact = frame.duplicated(keep=p.keep).to_numpy()
    out = df.loc[~exact].copy()
    n_exact = int(exact.sum())

    dup_rows = out.index.duplicated(keep=False)
    conflict_rows = int(out.index.duplicated(keep="first").sum())
    conflict_groups = int(out.index[dup_rows].nunique()) if dup_rows.any() else 0
    resolved = 0
    widened: list[str] = []
    if p.conflicts in ("first", "last") and conflict_rows:
        out = out.loc[~out.index.duplicated(keep=p.conflicts)]
        resolved = conflict_rows
    elif p.conflicts in ("mean", "median", "sum") and conflict_rows:
        out, widened = _aggregate_conflicts(out, p.conflicts)
        resolved = conflict_rows

    left = conflict_rows - resolved
    parts = [f"removed {n_exact} exact duplicate row(s) (kept the {p.keep})"]
    if conflict_rows:
        parts.append(
            f"{conflict_rows} extra row(s) in {conflict_groups} timestamp(s) with conflicting "
            + (
                f"values resolved with '{p.conflicts}'"
                if resolved
                else "values were left as they are"
            )
        )
    if widened:
        parts.append(
            f"{len(widened)} integer column(s) became decimal because the result is "
            f"fractional: {', '.join(widened)}"
        )
    return Outcome(
        df=out,
        summary="Duplicates: " + "; ".join(parts) + ".",
        counts={
            "exact_duplicates_removed": n_exact,
            "conflicting_timestamp_groups": conflict_groups,
            "conflicting_rows_resolved": resolved,
            "conflicting_rows_left": left,
            "integer_columns_widened": len(widened),
            "rows_removed": n_before - len(out),
        },
    )


def _aggregate_conflicts(df: pd.DataFrame, how: str) -> tuple[pd.DataFrame, list[str]]:
    """Collapse rows sharing a timestamp.

    Numeric columns are aggregated; every other column keeps the value from the first row.
    Integer columns stay integer when the result is whole; the ones that had to become decimal
    are returned by name so the log can say so.
    """
    groups = df.groupby(level=0, sort=False, dropna=False)
    numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    others = [c for c in df.columns if c not in numeric]
    pieces: list[pd.DataFrame] = []
    if numeric:
        pieces.append(
            groups[numeric].sum(min_count=1) if how == "sum" else getattr(groups[numeric], how)()
        )
    if others:
        first_rows = df.loc[~df.index.duplicated(keep="first"), others]
        if pieces:
            first_rows = first_rows.set_axis(pieces[0].index)  # same labels, same order
        pieces.append(first_rows)
    out = pd.concat(pieces, axis=1)[list(df.columns)]
    out.index.name = df.index.name
    widened: list[str] = []
    for c in numeric:
        if pd.api.types.is_integer_dtype(df[c]) and not pd.api.types.is_integer_dtype(out[c]):
            col = out[c]
            if col.notna().all() and (col == np.floor(col)).all():
                out[c] = col.astype(df[c].dtype)
            else:
                widened.append(str(c))
    return out, widened


# ----------------------------------------------------------------------------- handle_missing


class HandleMissingParams(BaseModel):
    """Fill or drop missing values in one column."""

    model_config = ConfigDict(extra="forbid")

    column: str
    method: Literal["interpolate_time", "ffill", "seasonal_fill", "drop"]
    max_gap: int | None = Field(default=DEFAULT_MAX_GAP, ge=1)  # interpolate_time; None = any
    limit: int | None = Field(default=None, ge=1)  # ffill
    period: int | None = Field(default=None, ge=2)  # seasonal_fill, in rows
    max_seasons: int = Field(
        default=2, ge=1
    )  # seasonal_fill: how many seasons to look back/forward

    @model_validator(mode="after")
    def _needs_period(self) -> HandleMissingParams:
        if self.method == "seasonal_fill" and self.period is None:
            raise ValueError("seasonal_fill needs a period (number of rows per season)")
        return self


def _handle_missing(df: pd.DataFrame, p: HandleMissingParams) -> Outcome:
    _require_column(df, p.column, numeric=p.method in ("interpolate_time", "seasonal_fill"))
    s = df[p.column]
    before = int(s.isna().sum())
    out = df.copy()
    counts: dict[str, float] = {"missing_before": before}

    if p.method == "drop":
        out = out.loc[s.notna().to_numpy()]
        counts["rows_dropped"] = before
        summary = f"'{p.column}': dropped {before} row(s) where it was missing."
    elif p.method == "ffill":
        out[p.column] = s.ffill(limit=p.limit)
        filled = before - int(out[p.column].isna().sum())
        counts["filled"] = filled
        counts["left_missing"] = before - filled
        summary = (
            f"'{p.column}': forward-filled {filled} of {before} missing value(s)"
            + (f" (limit {p.limit})" if p.limit else "")
            + (f"; {before - filled} left missing." if before - filled else ".")
        )
    elif p.method == "interpolate_time":
        _require_time_index(df)
        filled_series, detail = _interpolate_gaps(s, p.max_gap)
        out[p.column] = filled_series
        counts.update(detail)
        summary = (
            f"'{p.column}': interpolated {detail['filled']:g} of {before} missing value(s) in "
            f"{detail['gaps_filled']:g} gap(s) of at most {p.max_gap or 'any'} point(s); "
            f"{detail['left_too_long']:g} left in {detail['gaps_too_long']:g} longer gap(s) and "
            f"{detail['left_at_edges']:g} at the series edges."
        )
    else:  # seasonal_fill
        filled_series = _seasonal_fill(s, int(p.period or 0), p.max_seasons)
        out[p.column] = filled_series
        filled = before - int(filled_series.isna().sum())
        counts["filled"] = filled
        counts["left_missing"] = before - filled
        summary = (
            f"'{p.column}': filled {filled} of {before} missing value(s) from the same position "
            f"in neighboring seasons (period {p.period}); {before - filled} could not be filled."
        )
    counts["missing_after"] = int(out[p.column].isna().sum()) if p.method != "drop" else 0
    return Outcome(df=out, summary=summary, counts=counts)


def _interpolate_gaps(s: pd.Series, max_gap: int | None) -> tuple[pd.Series, dict[str, float]]:
    """Time-linear interpolation of gaps no longer than max_gap; longer gaps are left whole."""
    mask = s.isna().to_numpy()
    starts, ends = _runs(mask)
    lengths = ends - starts
    n = len(s)
    inside = (starts > 0) & (ends < n)
    short = np.ones(len(lengths), bool) if max_gap is None else lengths <= max_gap
    fill = inside & short
    fill_mask = np.zeros(n, bool)
    fill_mask[np.flatnonzero(mask)] = np.repeat(fill, lengths)
    interpolated = s.interpolate(method="time", limit_area="inside")
    out = s.where(~fill_mask, interpolated)
    too_long = inside & ~short
    detail = {
        "filled": float(lengths[fill].sum()),
        "gaps_filled": float(fill.sum()),
        "left_too_long": float(lengths[too_long].sum()),
        "gaps_too_long": float(too_long.sum()),
        "left_at_edges": float(lengths[~inside].sum()),
    }
    return out, detail


def _seasonal_fill(s: pd.Series, period: int, max_seasons: int) -> pd.Series:
    """Fill each gap from the mean of the same position one (then two, ...) seasons away."""
    values = s.to_numpy(dtype=float)
    missing = np.isnan(values)
    filled = values.copy()
    todo = missing.copy()
    if isinstance(s.index, pd.DatetimeIndex) and len(s) > 1:
        unique = s[~s.index.duplicated()]
        gaps = unique.index.to_series().diff().dropna()
        step = gaps.median() if len(gaps) else pd.Timedelta(0)
        if step <= pd.Timedelta(0):
            return s.copy()

        def at(shift: int) -> np.ndarray:
            return unique.reindex(s.index + shift * period * step).to_numpy(dtype=float)

    else:

        def at(shift: int) -> np.ndarray:
            out = np.full(len(values), np.nan)
            k = shift * period
            if k > 0:
                out[k:] = values[:-k]
            elif k < 0:
                out[:k] = values[-k:]
            return out

    for season in range(1, max_seasons + 1):
        before, after = at(-season), at(season)
        total = np.nan_to_num(before) + np.nan_to_num(after)
        count = (~np.isnan(before)).astype(int) + (~np.isnan(after)).astype(int)
        usable = todo & (count > 0)
        filled[usable] = total[usable] / count[usable]
        todo &= ~usable
    return pd.Series(filled, index=s.index, name=s.name)


# --------------------------------------------------------------------------------- outliers


class OutliersParams(BaseModel):
    """Detect outliers in one column and flag, blank or clip them."""

    model_config = ConfigDict(extra="forbid")

    column: str
    method: Literal["hampel", "seasonal_residual"] = "hampel"
    action: Literal["flag", "set_missing", "clip"] = "flag"
    window: int = Field(default=11, ge=3, le=MAX_WINDOW)  # hampel: rolling window in rows (odd)
    n_sigmas: float = Field(default=3.0, gt=0)
    period: int | None = Field(default=None, ge=2)  # seasonal_residual: rows per season

    @model_validator(mode="after")
    def _check(self) -> OutliersParams:
        if self.window % 2 == 0:
            raise ValueError("window must be odd")
        if self.method == "seasonal_residual" and self.period is None:
            raise ValueError("seasonal_residual needs a period (number of rows per season)")
        return self


@dataclass
class _Detection:
    mask: np.ndarray  # True where an outlier
    lower: np.ndarray  # NaN where not evaluated
    upper: np.ndarray
    unevaluated: int


def _rolling_median_mad(
    values: np.ndarray, window: int, min_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centered rolling median and scaled MAD (spread of the window around its own median)."""
    n, half = len(values), window // 2
    padded = np.concatenate([np.full(half, np.nan), values, np.full(half, np.nan)])
    view = np.lib.stride_tricks.sliding_window_view(padded, window)
    median = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    rows = max(1, _CHUNK_CELLS // window)
    for start in range(0, n, rows):
        block = view[start : start + rows]
        enough = (~np.isnan(block)).sum(axis=1) >= min_periods
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows
            m = np.nanmedian(block, axis=1)
            d = np.nanmedian(np.abs(block - m[:, None]), axis=1)
        median[start : start + len(block)] = np.where(enough, m, np.nan)
        mad[start : start + len(block)] = np.where(enough, d * MAD_TO_SIGMA, np.nan)
    return median, mad


def _hampel(x: pd.Series, window: int, n_sigmas: float) -> _Detection:
    """Hampel filter: a value is an outlier if it is more than n_sigmas scaled MADs from the
    median of its centered window. Windows with no spread use a small floor scale."""
    values = x.to_numpy(dtype=float)
    median, scale = _rolling_median_mad(values, window, window // 2 + 1)
    typical = float(np.nanmedian(scale)) if np.isfinite(scale).any() else 0.0
    scale = np.maximum(scale, max(0.1 * typical, 1e-12))
    lower, upper = median - n_sigmas * scale, median + n_sigmas * scale
    with np.errstate(invalid="ignore"):
        mask = (values < lower) | (values > upper)
    unevaluated = int((~np.isnan(values) & np.isnan(median)).sum())
    return _Detection(mask, lower, upper, unevaluated)


def _seasonal_smoother(n: int, period: int) -> int:
    """A smooth seasonal window when there are many cycles, so the fit cannot follow noise."""
    return 13 if n // period >= 10 else 7


def _stl_fit(values: np.ndarray, period: int) -> np.ndarray:
    try:
        result = STL(
            values, period=period, seasonal=_seasonal_smoother(len(values), period), robust=True
        ).fit()
    except ValueError as exc:
        raise CleaningError(f"Seasonal fit failed: {exc}") from exc
    return result.trend + result.seasonal


def _stl_residual(x: pd.Series, period: int, n_sigmas: float) -> _Detection:
    """Seasonal-trend fit; outliers are residuals beyond n_sigmas scaled MADs.

    The fit runs twice: suspect points are replaced by the first fit's values and the fit is
    repeated, so a large spike cannot pull the fit toward itself.
    """
    values = x.to_numpy(dtype=float)
    if len(values) < 2 * period + 2:
        raise CleaningError(
            f"'{x.name}' has {len(values)} rows, too few for a seasonal fit with period {period}: "
            f"at least {2 * period + 2} are needed."
        )
    if np.isnan(values).mean() > 0.5:
        raise CleaningError(f"'{x.name}' is more than half missing, too sparse for a seasonal fit.")
    positions = np.arange(len(values))
    present = ~np.isnan(values)
    filled = np.interp(positions, positions[present], values[present])  # analysis copy only

    def spread(residual: np.ndarray) -> tuple[float, float]:
        center = float(np.median(residual))
        return center, max(MAD_TO_SIGMA * float(np.median(np.abs(residual - center))), 1e-12)

    fitted = _stl_fit(filled, period)
    center, scale = spread(filled - fitted)
    suspect = np.abs(filled - fitted - center) > n_sigmas * scale
    if suspect.any():
        fitted = _stl_fit(np.where(suspect, fitted, filled), period)
        center, scale = spread(filled - fitted)
    lower, upper = fitted + center - n_sigmas * scale, fitted + center + n_sigmas * scale
    with np.errstate(invalid="ignore"):
        mask = (values < lower) | (values > upper)
    return _Detection(mask, lower, upper, 0)


def detect_outliers(df: pd.DataFrame, p: OutliersParams) -> _Detection:
    """Run the chosen detector on a column (used by the operation and by proposals)."""
    _require_column(df, p.column, numeric=True)
    x = df[p.column].astype(float)
    if p.method == "hampel":
        return _hampel(x, p.window, p.n_sigmas)
    return _stl_residual(x, int(p.period or 0), p.n_sigmas)


def _outliers(df: pd.DataFrame, p: OutliersParams) -> Outcome:
    det = detect_outliers(df, p)
    out = df.copy()
    n_out = int(det.mask.sum())
    n_eval = int(df[p.column].notna().sum()) - det.unevaluated
    counts: dict[str, float] = {
        "outliers": n_out,
        "evaluated": n_eval,
        "not_evaluated": det.unevaluated,
        "missing_skipped": int(df[p.column].isna().sum()),
    }
    where = f"'{p.column}' ({p.method}, {p.n_sigmas:g} sigma)"
    if p.action == "flag":
        flag = p.column + FLAG_SUFFIX
        counts["replaced_existing_flag"] = float(flag in out.columns)
        out[flag] = pd.Series(det.mask, index=out.index, dtype=bool)
        summary = f"{where}: flagged {n_out} outlier(s) in '{flag}'; original values kept."
    elif p.action == "set_missing":
        out.loc[det.mask, p.column] = np.nan
        counts["set_missing"] = n_out
        summary = f"{where}: set {n_out} outlier(s) to missing."
    else:
        values = out[p.column].to_numpy(dtype=float)
        clipped = np.where(det.mask, np.clip(values, det.lower, det.upper), values)
        out[p.column] = clipped
        counts["clipped"] = n_out
        summary = f"{where}: clipped {n_out} outlier(s) to the detection bounds."
    if det.unevaluated:
        summary += f" {det.unevaluated} value(s) near the edges could not be evaluated."
    return Outcome(df=out, summary=summary, counts=counts)


# ---------------------------------------------------------------------------------- resample


class ResampleParams(BaseModel):
    """Resample to a regular frequency (a pandas rule such as '1h', '15min' or '1D')."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    agg: Literal["mean", "sum", "median", "min", "max", "last"] = "mean"

    @model_validator(mode="after")
    def _valid_rule(self) -> ResampleParams:
        try:
            offset = pd.tseries.frequencies.to_offset(self.rule)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"'{self.rule}' is not a valid frequency") from exc
        if offset is None:
            raise ValueError(f"'{self.rule}' is not a valid frequency")
        return self


def _resample(df: pd.DataFrame, p: ResampleParams) -> Outcome:
    _require_time_index(df)
    resampler = df.resample(p.rule)
    sizes = resampler.size()
    numeric = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ]
    booleans = [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]
    others = [c for c in df.columns if c not in numeric and c not in booleans]
    pieces: list[pd.DataFrame] = []
    if numeric:
        part = resampler[numeric]
        pieces.append(part.sum(min_count=1) if p.agg == "sum" else getattr(part, p.agg)())
    if booleans:
        pieces.append(resampler[booleans].max().fillna(False).astype(bool))
    if others:
        pieces.append(resampler[others].first())
    out = pd.concat(pieces, axis=1)[list(df.columns)] if pieces else df.iloc[0:0].copy()
    out.index.name = df.index.name
    empty = int((sizes == 0).sum())
    return Outcome(
        df=out,
        summary=(
            f"Resampled to '{p.rule}' ({p.agg}): {len(df)} -> {len(out)} row(s); "
            f"{empty} empty bin(s) left missing."
        ),
        counts={"rows_before": len(df), "rows_after": len(out), "empty_bins": empty},
    )


# ------------------------------------------------------------------------------------- warnings


class QualityWarning(BaseModel):
    """A data-quality concern. Warnings never change data."""

    kind: Literal["negative_values", "constant_column", "suspect_unit_scale", "implausible_jumps"]
    column: str
    count: int
    detail: str
    examples: list[str] = Field(default_factory=list)


class CheckWarningsParams(BaseModel):
    """consumption_columns: columns that must not be negative (names with hints such as
    'energy' or 'power' are included automatically). jump_factor: jump size in robust sigmas."""

    model_config = ConfigDict(extra="forbid")

    consumption_columns: list[str] = Field(default_factory=list)
    jump_factor: float = Field(default=10.0, gt=0)


def _plain_numeric(df: pd.DataFrame) -> list[str]:
    return [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and not str(c).endswith(FLAG_SUFFIX)
    ]


def quality_warnings(
    df: pd.DataFrame,
    consumption_columns: list[str] | None = None,
    jump_factor: float = 10.0,
) -> list[QualityWarning]:
    """Find suspicious values without changing anything."""
    out: list[QualityWarning] = []
    flagged = set(consumption_columns or [])
    numeric = _plain_numeric(df)
    for col in numeric:
        if col in flagged or any(h in col.lower() for h in CONSUMPTION_HINTS):
            negative = df[col] < 0
            if negative.any():
                out.append(
                    QualityWarning(
                        kind="negative_values",
                        column=col,
                        count=int(negative.sum()),
                        detail=f"'{col}' looks like a consumption column but has negative values.",
                        examples=[str(v) for v in df.index[negative.to_numpy()][:3]],
                    )
                )
    for col in df.columns:
        if str(col).endswith(FLAG_SUFFIX):
            continue
        if df[col].nunique(dropna=True) == 1 and len(df) > 1:
            out.append(
                QualityWarning(
                    kind="constant_column",
                    column=str(col),
                    count=int(df[col].notna().sum()),
                    detail=f"'{col}' has a single value.",
                )
            )
    medians = {
        c: float(np.nanmedian(np.abs(df[c].to_numpy(float))))
        for c in numeric
        if df[c].notna().any()
    }
    positive = {c: m for c, m in medians.items() if m > 0}
    if len(positive) > 1:
        small = min(positive, key=lambda c: positive[c])
        for col, med in positive.items():
            if col != small and med / positive[small] > _UNIT_SCALE_RATIO:
                out.append(
                    QualityWarning(
                        kind="suspect_unit_scale",
                        column=col,
                        count=int(df[col].notna().sum()),
                        detail=(
                            f"Median of '{col}' ({med:.4g}) is over {_UNIT_SCALE_RATIO:g}x the "
                            f"median of '{small}' ({positive[small]:.4g}); the units may differ."
                        ),
                    )
                )
    for col in numeric:
        diffs = df[col].astype(float).diff().to_numpy()
        valid = diffs[~np.isnan(diffs)]
        if len(valid) < 10:
            continue
        scale = MAD_TO_SIGMA * float(np.median(np.abs(valid - np.median(valid))))
        if scale <= 0:
            continue
        with np.errstate(invalid="ignore"):
            jumps = np.abs(diffs - np.median(valid)) > jump_factor * scale
        if jumps.any():
            out.append(
                QualityWarning(
                    kind="implausible_jumps",
                    column=col,
                    count=int(jumps.sum()),
                    detail=(
                        f"'{col}' has {int(jumps.sum())} step(s) larger than {jump_factor:g} "
                        "robust standard deviations of its usual change."
                    ),
                    examples=[str(v) for v in df.index[jumps][:3]],
                )
            )
    return out


def _check_warnings(df: pd.DataFrame, p: CheckWarningsParams) -> Outcome:
    found = quality_warnings(df, p.consumption_columns, p.jump_factor)
    by_kind: dict[str, float] = {}
    for w in found:
        by_kind[w.kind] = by_kind.get(w.kind, 0) + 1
    summary = (
        f"Quality check: {len(found)} warning(s). No data was changed."
        if found
        else "Quality check: no warnings. No data was changed."
    )
    return Outcome(
        df=df.copy(),
        summary=summary,
        counts={"warnings": len(found), **{f"warnings_{k}": v for k, v in by_kind.items()}},
        read_only=True,
    )


# -------------------------------------------------------------------------------------- preview


class ColumnStats(BaseModel):
    """Summary statistics of one column."""

    count: int
    missing: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class PreviewResult:
    """What an operation would do: stats before and after, the log entry, and the new frame."""

    before: dict[str, ColumnStats]
    after: dict[str, ColumnStats]
    rows_before: int
    rows_after: int
    log_entry: StepLog
    df: pd.DataFrame


def _stats(s: pd.Series) -> ColumnStats:
    present = s.dropna()
    base = {"count": int(len(present)), "missing": int(s.isna().sum())}
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or present.empty:
        return ColumnStats(**base)

    def finite(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    return ColumnStats(
        **base,
        mean=finite(present.mean()),
        std=finite(present.std()) if len(present) > 1 else None,
        min=finite(present.min()),
        max=finite(present.max()),
    )


def preview(df: pd.DataFrame, spec: OpSpec) -> PreviewResult:
    """Apply `spec` to a copy and summarize the effect. `df` is not modified."""
    result = apply_op(df, spec)
    column = spec.params.get("column")
    if isinstance(column, str) and column in df.columns:
        names = [column]
    else:
        names = [c for c in _plain_numeric(df)][:_STAT_COLUMN_CAP]
    before = {c: _stats(df[c]) for c in names if c in df.columns}
    after = {c: _stats(result.df[c]) for c in names if c in result.df.columns}
    return PreviewResult(
        before=before,
        after=after,
        rows_before=len(df),
        rows_after=len(result.df),
        log_entry=result.log_entry,
        df=result.df,
    )


# ------------------------------------------------------------------------------------ proposals


class Alternative(BaseModel):
    """Another way the user could handle the same issue."""

    label: str
    spec: OpSpec


class Proposal(BaseModel):
    """A suggested fix with plain-language counts. Nothing is applied until the user agrees."""

    id: str
    category: Literal["duplicates", "missing", "outliers", "resample", "warning"]
    spec: OpSpec
    description: str
    default_selected: bool
    counts: dict[str, float] = Field(default_factory=dict)
    alternatives: list[Alternative] = Field(default_factory=list)


def _gap_summary(s: pd.Series, max_gap: int) -> tuple[int, int, int]:
    """(missing total, fillable by interpolation with max_gap, longest run)."""
    mask = s.isna().to_numpy()
    starts, ends = _runs(mask)
    lengths = ends - starts
    inside = (starts > 0) & (ends < len(s))
    fillable = int(lengths[inside & (lengths <= max_gap)].sum())
    return int(mask.sum()), fillable, int(lengths.max()) if len(lengths) else 0


def propose_fixes(
    df: pd.DataFrame,
    prof: DataProfile,
    characterization: SeriesCharacterization | None = None,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
) -> list[Proposal]:
    """Suggest fixes, listed in the order they should be applied.

    Only exact-duplicate removal and short-gap interpolation are selected by default.
    """
    proposals: list[Proposal] = []
    period = characterization.seasonal_period if characterization else None
    time_ok = isinstance(df.index, pd.DatetimeIndex) and df.index.is_monotonic_increasing

    d = prof.duplicates
    if d.full_row_duplicates:
        proposals.append(
            Proposal(
                id="drop_duplicates:exact",
                category="duplicates",
                spec=OpSpec(operation="drop_duplicates", params={"keep": "first"}),
                description=(
                    f"{d.full_row_duplicates:,} exact duplicate row(s): keep the first of each."
                ),
                default_selected=True,
                counts={"exact_duplicates": d.full_row_duplicates},
                alternatives=[
                    Alternative(
                        label="keep the last of each",
                        spec=OpSpec(operation="drop_duplicates", params={"keep": "last"}),
                    )
                ],
            )
        )
    if d.duplicate_timestamps_conflicting:
        n = d.duplicate_timestamps_conflicting
        proposals.append(
            Proposal(
                id="drop_duplicates:conflicts",
                category="duplicates",
                spec=OpSpec(operation="drop_duplicates", params={"conflicts": "first"}),
                description=(
                    f"{n:,} row(s) share a timestamp with different values: keep the first."
                ),
                default_selected=False,
                counts={"conflicting_rows": n},
                alternatives=[
                    Alternative(
                        label=f"{how}",
                        spec=OpSpec(operation="drop_duplicates", params={"conflicts": how}),
                    )
                    for how in ("last", "mean", "median", "sum")
                ],
            )
        )

    for col in _plain_numeric(df):
        total, fillable, longest = _gap_summary(df[col], max_gap)
        if total == 0 or total == len(df):
            continue
        base = {"column": col}
        interp = OpSpec(
            operation="handle_missing",
            params={**base, "method": "interpolate_time", "max_gap": max_gap},
        )
        forward_fill = Alternative(
            label="forward fill",
            spec=OpSpec(
                operation="handle_missing", params={**base, "method": "ffill", "limit": max_gap}
            ),
        )
        unlimited = Alternative(
            label="interpolate every gap, however long",
            spec=OpSpec(
                operation="handle_missing",
                params={**base, "method": "interpolate_time", "max_gap": None},
            ),
        )
        alts = [
            forward_fill,
            unlimited,
            Alternative(
                label="drop rows with a missing value",
                spec=OpSpec(operation="handle_missing", params={**base, "method": "drop"}),
            ),
        ]
        seasonal = (
            OpSpec(
                operation="handle_missing",
                params={**base, "method": "seasonal_fill", "period": period},
            )
            if period
            else None
        )
        if seasonal is not None:
            alts.insert(
                0,
                Alternative(
                    label=f"fill from the same time in nearby seasons (period {period})",
                    spec=seasonal,
                ),
            )
        if time_ok and fillable:
            proposals.append(
                Proposal(
                    id=f"handle_missing:{col}",
                    category="missing",
                    spec=interp,
                    description=(
                        f"{total:,} missing value(s) in {col}: interpolate gaps up to {max_gap} "
                        f"point(s) ({fillable:,} fillable, {total - fillable:,} left; "
                        f"longest gap {longest})."
                    ),
                    default_selected=True,
                    counts={"missing": total, "fillable": fillable, "longest_gap": longest},
                    alternatives=alts,
                )
            )
        else:
            # Every gap is longer than max_gap (or the index cannot support interpolation):
            # suggest the least invented option, but never select it by default.
            lead = seasonal or (unlimited if time_ok else forward_fill)
            how = (
                "fill from nearby seasons"
                if seasonal
                else ("interpolate across the long gaps" if time_ok else "forward fill")
            )
            proposals.append(
                Proposal(
                    id=f"handle_missing:{col}",
                    category="missing",
                    spec=lead.spec if isinstance(lead, Alternative) else lead,
                    description=(
                        f"{total:,} missing value(s) in {col}, none in gaps of {max_gap} point(s) "
                        f"or fewer (longest gap {longest}): {how}."
                    ),
                    default_selected=False,
                    counts={"missing": total, "fillable": fillable, "longest_gap": longest},
                    alternatives=[
                        a
                        for a in alts
                        if a.spec != (lead.spec if isinstance(lead, Alternative) else lead)
                    ],
                )
            )

    for col in _plain_numeric(df):
        if df[col].nunique(dropna=True) < 3:
            continue
        flag = OutliersParams(column=col)
        try:
            found = int(detect_outliers(df, flag).mask.sum())
        except CleaningError:
            continue
        if not found:
            continue
        alts = [
            Alternative(
                label="set them to missing",
                spec=OpSpec(operation="outliers", params={"column": col, "action": "set_missing"}),
            ),
            Alternative(
                label="clip them to the detection bounds",
                spec=OpSpec(operation="outliers", params={"column": col, "action": "clip"}),
            ),
        ]
        if period and len(df) >= 2 * period + 2:
            alts.append(
                Alternative(
                    label=f"flag using a seasonal fit (period {period})",
                    spec=OpSpec(
                        operation="outliers",
                        params={"column": col, "method": "seasonal_residual", "period": period},
                    ),
                )
            )
        proposals.append(
            Proposal(
                id=f"outliers:{col}",
                category="outliers",
                spec=OpSpec(operation="outliers", params={"column": col}),
                description=(
                    f"{found:,} suspected outlier(s) in {col} (rolling median, window "
                    f"{flag.window}, {flag.n_sigmas:g} sigma): flag them and keep the values."
                ),
                default_selected=False,
                counts={"outliers": found},
                alternatives=alts,
            )
        )

    sampling = prof.sampling
    if time_ok and sampling is not None and sampling.off_grid_count:
        rule = sampling.median_interval_label
        proposals.append(
            Proposal(
                id="resample:regular",
                category="resample",
                spec=OpSpec(operation="resample", params={"rule": rule, "agg": "mean"}),
                description=(
                    f"{sampling.off_grid_count:,} interval(s) are not {rule}: resample to a "
                    f"regular {rule} grid (mean)."
                ),
                default_selected=False,
                counts={"off_grid_intervals": sampling.off_grid_count},
                alternatives=[
                    Alternative(
                        label=f"{agg} instead of mean",
                        spec=OpSpec(operation="resample", params={"rule": rule, "agg": agg}),
                    )
                    for agg in ("sum", "median", "min", "max", "last")
                ],
            )
        )

    found_warnings = quality_warnings(df)
    if found_warnings:
        proposals.append(
            Proposal(
                id="check_warnings:all",
                category="warning",
                spec=OpSpec(operation="check_warnings", params={}),
                description=(
                    f"{len(found_warnings)} data-quality warning(s): "
                    + "; ".join(
                        sorted(
                            {f"{w.kind.replace('_', ' ')} in {w.column}" for w in found_warnings}
                        )
                    )
                    + ". Review only, nothing is changed."
                ),
                default_selected=False,
                counts={"warnings": len(found_warnings)},
            )
        )
    return proposals
