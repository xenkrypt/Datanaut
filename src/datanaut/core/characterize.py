"""Series characterization: frequency, seasonal period, trend and stationarity.

All analysis runs on a regularized, interpolated COPY of the series; the caller's DataFrame is
never changed. Long series are summarized in blocks (trend) or by their most recent points
(stationarity test) for speed, and the result says so.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from statsmodels.tsa.stattools import acf, adfuller

from datanaut.core.log import PipelineLog, StepLog
from datanaut.core.profiler import interval_label
from datanaut.errors import ParseError

DAY, WEEK, YEAR = 86_400.0, 604_800.0, 365 * 86_400.0
MIN_PEAK_STRENGTH = 0.2
"""Autocorrelation a candidate period needs to count as seasonality."""

_STRONG, _MODERATE = 0.6, 0.35
_MIN_ADF_POINTS = 20
_MIN_TREND_BLOCKS = 8
_MAX_GRID_FACTOR = 50

_FREQUENCIES: tuple[tuple[float, str], ...] = (
    (1, "secondly"),
    (60, "minutely"),
    (300, "5-minutely"),
    (600, "10-minutely"),
    (900, "15-minutely"),
    (1800, "30-minutely"),
    (3600, "hourly"),
    (10_800, "3-hourly"),
    (21_600, "6-hourly"),
    (43_200, "12-hourly"),
    (DAY, "daily"),
    (WEEK, "weekly"),
)
_CALENDAR: tuple[tuple[float, float, str], ...] = (
    (28 * DAY, 31 * DAY, "monthly"),
    (89 * DAY, 92 * DAY, "quarterly"),
    (365 * DAY, 366 * DAY, "yearly"),
)


class SeasonalCandidate(BaseModel):
    """One sensible period for the frequency, with its autocorrelation strength."""

    period: int
    label: str
    strength: float
    is_peak: bool
    significant: bool


class TrendResult(BaseModel):
    """Theil-Sen trend with a Kendall tau check (p-values are indicative: data is correlated)."""

    direction: Literal["increasing", "decreasing", "none", "insufficient data"]
    slope_per_day: float | None = None
    total_change: float | None = None
    relative_change_pct: float | None = None
    effect_size: float | None = None
    effect_label: Literal["small", "moderate", "large"] | None = None
    kendall_tau: float | None = None
    p_value: float | None = None
    points_used: int = 0
    block_size: int = 1


class StationarityResult(BaseModel):
    """Augmented Dickey-Fuller result."""

    label: Literal["stationary", "non-stationary", "insufficient data"]
    statistic: float | None = None
    p_value: float | None = None
    points_used: int = 0


class SeriesCharacterization(BaseModel):
    """How a series behaves, used to choose defaults for later steps."""

    column: str
    n_rows: int
    n_observations: int
    missing_share: float
    frequency_label: str
    median_interval_seconds: float | None = None
    analysis_points: int = 0
    downsampled: bool = False
    seasonal_candidates: list[SeasonalCandidate] = Field(default_factory=list)
    seasonal_period: int | None = None
    seasonal_label: str | None = None
    seasonal_strength: float | None = None
    trend: TrendResult
    stationarity: StationarityResult
    recommendation: str
    notes: list[str] = Field(default_factory=list)
    log_entry: StepLog


def frequency_label(seconds: float | None) -> str:
    """Name a sampling interval, for example 3600 -> 'hourly'."""
    if seconds is None or seconds <= 0:
        return "unknown"
    for target, name in _FREQUENCIES:
        if abs(seconds - target) <= 0.05 * target:
            return name
    for low, high, name in _CALENDAR:
        if low <= seconds <= high:
            return name
    return f"every {interval_label(seconds)}"


def _period_label(period: int, step: float) -> str:
    span = period * step
    for target, name in ((DAY, "daily"), (WEEK, "weekly"), (YEAR, "yearly")):
        if abs(span - target) <= 0.05 * target:
            return name
    return f"{period}-step"


def candidate_periods(step_seconds: float, n_points: int) -> list[int]:
    """Sensible seasonal periods (in steps): day, week and year, if the data covers 2 cycles."""
    out: list[int] = []
    for seconds in (DAY, WEEK, YEAR):
        exact = seconds / step_seconds
        period = round(exact)
        if period >= 2 and abs(exact - period) / period <= 0.03 and n_points >= 2 * period:
            if period not in out:
                out.append(period)
    return out


def _regularize(s: pd.Series, step: float) -> pd.Series | None:
    """Put the series on a regular grid and interpolate gaps. Works on a copy."""
    grid = pd.date_range(s.index[0], s.index[-1], freq=pd.Timedelta(seconds=step))
    if len(grid) > _MAX_GRID_FACTOR * len(s) + 1000:
        return None
    tolerance = pd.Timedelta(seconds=step / 2)
    on_grid = s.reindex(grid, method="nearest", tolerance=tolerance)
    if on_grid.isna().mean() > 0.5:
        return None
    return on_grid.interpolate(method="time", limit_direction="both")


def _seasonal(x: np.ndarray, step: float) -> list[SeasonalCandidate]:
    periods = candidate_periods(step, len(x))
    if not periods:
        return []
    t = np.arange(len(x), dtype=float)
    detrended = x - np.polyval(np.polyfit(t, x, 1), t)
    correlations = acf(detrended, nlags=max(periods) + 1, fft=True)
    out: list[SeasonalCandidate] = []
    for p in periods:
        radius = max(1, round(0.04 * p))
        lo, hi = max(1, p - radius), min(len(correlations) - 1, p + radius)
        strength = float(correlations[p])
        is_peak = bool(strength >= correlations[lo : hi + 1].max() - 1e-12)
        out.append(
            SeasonalCandidate(
                period=p,
                label=_period_label(p, step),
                strength=round(strength, 4),
                is_peak=is_peak,
                significant=bool(is_peak and strength >= MIN_PEAK_STRENGTH),
            )
        )
    return sorted(out, key=lambda c: -c.strength)


def _primary(candidates: list[SeasonalCandidate]) -> SeasonalCandidate | None:
    """Strongest significant candidate, preferring a shorter period that is nearly as strong."""
    significant = [c for c in candidates if c.significant]
    if not significant:
        return None
    best = max(c.strength for c in significant)
    return min((c for c in significant if c.strength >= 0.9 * best), key=lambda c: c.period)


def _trend(x: np.ndarray, step: float, period: int | None, max_points: int) -> TrendResult:
    block = 1
    if period and len(x) // period >= _MIN_TREND_BLOCKS:
        block = period
    if len(x) // block > max_points:
        factor = int(np.ceil(len(x) / block / max_points))
        block *= factor
    n_blocks = len(x) // block
    if n_blocks < _MIN_TREND_BLOCKS or np.ptp(x) == 0:
        return TrendResult(direction="insufficient data", points_used=n_blocks, block_size=block)
    y = x[: n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    days = (np.arange(n_blocks) * block + (block - 1) / 2) * step / DAY
    slope, intercept, _, _ = stats.theilslopes(y, days)
    tau, p_value = stats.kendalltau(days, y)
    residual = y - (slope * days + intercept)
    scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    total = float(slope * len(x) * step / DAY)
    mean = float(np.mean(x))
    effect = abs(total) / scale if scale > 0 else None
    significant = bool(p_value < 0.05)
    direction: Literal["increasing", "decreasing", "none"] = (
        "none" if not significant else ("increasing" if slope > 0 else "decreasing")
    )
    label = None
    if effect is not None:
        label = "small" if effect < 1 else ("moderate" if effect < 3 else "large")
    return TrendResult(
        direction=direction,
        slope_per_day=float(slope),
        total_change=total,
        relative_change_pct=(100 * total / abs(mean)) if abs(mean) > 1e-12 else None,
        effect_size=effect,
        effect_label=label,
        kendall_tau=float(tau),
        p_value=float(p_value),
        points_used=n_blocks,
        block_size=block,
    )


def _remove_seasonal_means(x: np.ndarray, periods: list[int]) -> np.ndarray:
    """Subtract the average pattern of each period (shortest first), keeping the level."""
    out = x.copy()
    for period in sorted(periods):
        phase = np.arange(len(out)) % period
        sums = np.bincount(phase, weights=out, minlength=period)
        counts = np.bincount(phase, minlength=period)
        pattern = sums / counts
        out = out - pattern[phase] + pattern.mean()
    return out


def _stationarity(x: np.ndarray, max_points: int, periods: list[int]) -> StationarityResult:
    """ADF test on the series with its seasonal pattern removed (the test is weak otherwise)."""
    window = _remove_seasonal_means(x, periods)[-max_points:]
    if len(window) < _MIN_ADF_POINTS or np.ptp(window) == 0:
        return StationarityResult(label="insufficient data", points_used=len(window))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            statistic, p_value, *_ = adfuller(window, autolag="AIC")
        except (ValueError, np.linalg.LinAlgError):
            return StationarityResult(label="insufficient data", points_used=len(window))
    return StationarityResult(
        label="stationary" if p_value < 0.05 else "non-stationary",
        statistic=float(statistic),
        p_value=float(p_value),
        points_used=len(window),
    )


def _recommend(
    primary: SeasonalCandidate | None,
    trend: TrendResult,
    stationarity: StationarityResult,
    usable: bool,
    notes: list[str],
) -> str:
    if not usable:
        return (
            "not enough usable data to characterize this series: check missing values and sampling"
        )
    parts: list[str] = []
    if primary is not None:
        kind = (
            "strong"
            if primary.strength >= _STRONG
            else ("moderate" if primary.strength >= _MODERATE else "weak")
        )
        advice = "use seasonal methods" if kind != "weak" else "seasonal methods may help"
        parts.append(f"{kind} {primary.label} seasonality: {advice}")
    else:
        parts.append("no clear seasonality: non-seasonal methods are fine")
    if trend.direction in ("increasing", "decreasing"):
        parts.append(f"{trend.direction} trend ({trend.effect_label} effect): include a trend term")
    if stationarity.label == "non-stationary":
        parts.append("non-stationary: consider differencing or detrending before modeling")
    if any("irregular" in n or "gap" in n for n in notes):
        parts.append("sampling has gaps: they were interpolated for this analysis only")
    return "; ".join(parts)


def characterize(
    df: pd.DataFrame,
    column: str,
    *,
    max_trend_points: int = 2000,
    max_adf_points: int = 5000,
) -> SeriesCharacterization:
    """Describe how `column` behaves over its DatetimeIndex (read-only; the input is unchanged)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ParseError("Characterization needs a time-indexed table. Build the time index first.")
    if column not in df.columns:
        raise ParseError(f"Column '{column}' was not found in the data.")
    if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
        raise ParseError(f"Column '{column}' is not numeric.")

    raw = df[column].astype(float)
    n_inf = int(np.isinf(raw.to_numpy()).sum())
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw[raw.index.notna()]
    n_rows, n_obs = len(raw), int(raw.notna().sum())
    missing_share = float(1 - n_obs / n_rows) if n_rows else 1.0
    series = raw.groupby(level=0).mean().sort_index()  # a copy; duplicate stamps averaged
    notes: list[str] = []
    if n_inf:
        notes.append(f"{n_inf} infinite value(s) were treated as missing.")
    if len(series) < len(raw):
        notes.append(f"{len(raw) - len(series)} duplicate timestamp(s) were averaged for analysis.")

    step: float | None = None
    if len(series) >= 2:
        diffs = series.index.to_series().diff().dt.total_seconds().dropna()
        step = float(diffs.median()) or None
    label = frequency_label(step)

    x: np.ndarray | None = None
    if step is not None and n_obs >= 2:
        regular = _regularize(series, step)
        if regular is None:
            notes.append("Too many missing or sparse timestamps to analyze reliably.")
        else:
            x = regular.to_numpy()
            if len(regular) > len(series):
                notes.append(
                    f"{len(regular) - len(series)} gap timestamp(s) were interpolated "
                    "(on a copy used only for this analysis)."
                )
    elif n_obs < 2:
        notes.append("Fewer than two observations.")

    candidates: list[SeasonalCandidate] = []
    primary: SeasonalCandidate | None = None
    if x is not None and step is not None and np.ptp(x) > 0:
        candidates = _seasonal(x, step)
        primary = _primary(candidates)
        trend = _trend(x, step, primary.period if primary else None, max_trend_points)
        seasonal_periods = [c.period for c in candidates if c.significant]
        stationarity = _stationarity(x, max_adf_points, seasonal_periods)
        if seasonal_periods:
            notes.append(
                "Stationarity was tested after removing the "
                + " and ".join(
                    c.label for c in sorted(candidates, key=lambda c: c.period) if c.significant
                )
                + " pattern."
            )
    else:
        if x is not None:
            notes.append("The series is constant.")
        trend = TrendResult(direction="insufficient data")
        stationarity = StationarityResult(label="insufficient data")
    analysis_points = 0 if x is None else len(x)
    natural_block = 1
    if primary is not None and analysis_points // primary.period >= _MIN_TREND_BLOCKS:
        natural_block = primary.period
    downsampled = bool(
        analysis_points
        and (trend.block_size > natural_block or stationarity.points_used < analysis_points)
    )
    if trend.block_size > 1:
        notes.append(
            f"Trend was estimated on {trend.points_used} block mean(s) of {trend.block_size} "
            "readings each."
        )
    if x is not None and stationarity.points_used < analysis_points and stationarity.points_used:
        notes.append(
            f"Stationarity test used the most recent {stationarity.points_used} of "
            f"{analysis_points} points."
        )
    usable = primary is not None or trend.direction != "insufficient data"
    recommendation = _recommend(primary, trend, stationarity, usable, notes)

    log = PipelineLog()
    counts: dict[str, float] = {
        "n_observations": n_obs,
        "analysis_points": analysis_points,
        "seasonal_period": primary.period if primary else 0,
    }
    if stationarity.p_value is not None:
        counts["adf_p_value"] = stationarity.p_value
    if trend.p_value is not None:
        counts["trend_p_value"] = trend.p_value
    entry = log.append(
        "characterize",
        f"Characterized '{column}' ({label}): "
        + (
            f"{primary.label} seasonality (period {primary.period}), "
            if primary
            else "no clear seasonality, "
        )
        + f"trend {trend.direction}, {stationarity.label}. No data was changed.",
        params={
            "column": column,
            "max_trend_points": max_trend_points,
            "max_adf_points": max_adf_points,
        },
        counts=counts,
        rows_before=n_rows,
        rows_after=n_rows,
        read_only=True,
    )
    return SeriesCharacterization(
        column=column,
        n_rows=n_rows,
        n_observations=n_obs,
        missing_share=round(missing_share, 6),
        frequency_label=label,
        median_interval_seconds=step,
        analysis_points=analysis_points,
        downsampled=downsampled,
        seasonal_candidates=candidates,
        seasonal_period=primary.period if primary else None,
        seasonal_label=primary.label if primary else None,
        seasonal_strength=primary.strength if primary else None,
        trend=trend,
        stationarity=stationarity,
        recommendation=recommendation,
        notes=notes,
        log_entry=entry,
    )
