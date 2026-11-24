"""Weather link: Open-Meteo data, lined up with a series, and how the two move together.

There is no network code here. `fetch_open_meteo` builds the request URL and checks the reply,
but the caller supplies the `fetch` function (url -> parsed JSON), so tests use a fake and the
application decides how to reach the internet. Correlations describe association only.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from datanaut.core.evidence import (
    Evidence,
    analysis_log,
    make_evidence,
    median_step,
    numeric_series,
    to_regular_grid,
)
from datanaut.core.log import StepLog
from datanaut.errors import AnalysisError, WeatherError

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MIN_PAIRS = 10
_VARIABLE = re.compile(r"^[a-z0-9_]{1,64}$")

Fetch = Callable[[str], Any]


# ------------------------------------------------------------------------- Open-Meteo


def _as_date(value: date | str, label: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise WeatherError(f"The {label} date '{value}' is not a valid YYYY-MM-DD date.") from exc


def open_meteo_url(
    lat: float,
    lon: float,
    start: date | str,
    end: date | str,
    variables: list[str],
    *,
    source: Literal["archive", "forecast"] = "archive",
    timezone: str = "auto",
) -> str:
    """The request URL for hourly data. Inputs are validated; nothing is sent."""
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise WeatherError(
            "Latitude must be between -90 and 90 and longitude between -180 and 180."
        )
    first, last = _as_date(start, "start"), _as_date(end, "end")
    if first > last:
        raise WeatherError("The start date is after the end date.")
    if not variables:
        raise WeatherError("Name at least one weather variable, for example temperature_2m.")
    bad = [v for v in variables if not _VARIABLE.match(v)]
    if bad:
        raise WeatherError(f"'{bad[0]}' is not a valid weather variable name.")
    if not re.match(r"^[A-Za-z0-9_/+\-]{1,64}$", timezone):
        raise WeatherError(f"'{timezone}' is not a valid timezone name (use 'auto' or e.g. 'UTC').")
    query = urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": first.isoformat(),
            "end_date": last.isoformat(),
            "hourly": ",".join(variables),
            "timezone": timezone,
        },
        safe=",",
    )
    return f"{ARCHIVE_URL if source == 'archive' else FORECAST_URL}?{query}"


class _HourlyBlock(BaseModel):
    time: list[str]
    values: dict[str, list[float | None]] = Field(default_factory=dict)


def _parse_response(payload: Any, variables: list[str]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise WeatherError("The weather service sent a reply that is not a JSON object.")
    if payload.get("error"):
        raise WeatherError(
            f"The weather service refused the request: {payload.get('reason', 'no reason given')}."
        )
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise WeatherError(
            "The weather reply has no hourly data (expected an 'hourly' block with 'time')."
        )
    missing = [v for v in variables if v not in hourly]
    if missing:
        raise WeatherError(f"The weather reply is missing the variable '{missing[0]}'.")
    times = hourly["time"]
    if not isinstance(times, list) or not times:
        raise WeatherError("The weather reply has no timestamps.")
    columns: dict[str, list[float | None]] = {}
    for name in variables:
        series = hourly[name]
        if not isinstance(series, list) or len(series) != len(times):
            raise WeatherError(
                f"The weather variable '{name}' has "
                f"{len(series) if isinstance(series, list) else 'no'} "
                f"values for {len(times)} timestamps."
            )
        if any(
            v is not None and (isinstance(v, bool) or not isinstance(v, int | float))
            for v in series
        ):
            raise WeatherError(
                f"The weather variable '{name}' contains values that are not numbers."
            )
        columns[name] = series
    try:
        index = pd.DatetimeIndex(pd.to_datetime(times, format="ISO8601"), name="time")
    except (ValueError, TypeError) as exc:
        raise WeatherError("The weather reply has timestamps that could not be read.") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    frame = pd.DataFrame(
        {
            k: pd.array([np.nan if v is None else v for v in vals], dtype="float64")
            for k, vals in columns.items()
        },
        index=index,
    )
    return frame[~frame.index.duplicated(keep="first")].sort_index()


def fetch_open_meteo(
    lat: float,
    lon: float,
    start: date | str,
    end: date | str,
    variables: list[str],
    fetch: Fetch,
    *,
    source: Literal["archive", "forecast"] = "archive",
    timezone: str = "auto",
) -> pd.DataFrame:
    """Hourly weather as a time-indexed DataFrame, using the caller's `fetch(url) -> JSON`.

    Times are in the location's timezone when `timezone='auto'`, which matches sensor data
    logged in local time. A malformed or error reply raises WeatherError.
    """
    url = open_meteo_url(lat, lon, start, end, list(variables), source=source, timezone=timezone)
    try:
        payload = fetch(url)
    except WeatherError:
        raise
    except Exception as exc:  # the caller's fetch can fail in any way (network, HTTP, JSON)
        raise WeatherError(f"The weather request failed: {exc}") from exc
    return _parse_response(payload, list(variables))


def read_weather_csv(path: str | Path) -> pd.DataFrame:
    """Read weather from a local CSV: the first column is the time, the others are numeric.

    Use this when weather is supplied as a file instead of being fetched. Rows whose time cannot
    be read are an error, not silently dropped.
    """
    file = Path(path)
    try:
        raw = pd.read_csv(file, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise WeatherError(f"The weather file '{file.name}' could not be read.") from exc
    if raw.shape[1] < 2:
        raise WeatherError("The weather file needs a time column and at least one value column.")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(raw.iloc[:, 0], format="ISO8601"), name="time")
    except (ValueError, TypeError) as exc:
        raise WeatherError(
            "The first column of the weather file must hold ISO dates or times."
        ) from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").astype(float)
    values.index = index
    if values.dropna(how="all").empty:
        raise WeatherError("The weather file has no numeric values.")
    values = values[~values.index.duplicated(keep="first")].sort_index()
    return values.dropna(axis=1, how="all")


# ----------------------------------------------------------------------------- alignment


class AlignedWeather(BaseModel):
    """Weather lined up with a series' timestamps, and how much of the series it covers."""

    model_config = {"arbitrary_types_allowed": True}

    df: pd.DataFrame = Field(exclude=True)
    method: Literal["nearest", "resample_mean", "interpolate"]
    series_step_seconds: float | None
    weather_step_seconds: float | None
    coverage: dict[str, float]
    rows: int
    rows_with_weather: int
    evidence: Evidence
    log_entry: StepLog


def align_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> AlignedWeather:
    """Bring weather onto the timestamps of `df`.

    Coarser series (for example daily) get the mean of the weather inside each interval;
    finer series get time-linear interpolation between weather readings (never extrapolated);
    otherwise the nearest reading within half a step is used. Only the index of `df` is used.
    """
    for name, frame in (("series", df), ("weather", weather_df)):
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise WeatherError(f"The {name} needs a time index.")
    if (df.index.tz is None) != (weather_df.index.tz is None):
        raise WeatherError(
            "The series and the weather disagree about timezones: one is naive, one is not."
        )
    numeric = [c for c in weather_df.columns if pd.api.types.is_numeric_dtype(weather_df[c])]
    if not numeric:
        raise WeatherError("The weather table has no numeric columns.")
    weather = weather_df[numeric].astype(float)
    weather = weather[~weather.index.duplicated(keep="first")].sort_index()
    step_s, step_w = median_step(df.index), median_step(weather.index)
    if step_w is None:
        raise WeatherError("The weather table needs at least two timestamps.")
    tolerance = (step_s or step_w) / 2

    if step_s is not None and step_s >= step_w * 1.5:
        method = "resample_mean"
        binned = weather.resample(step_s).mean()
        aligned = binned.reindex(df.index, method="nearest", tolerance=tolerance)
    elif step_s is not None and step_s * 1.5 <= step_w:
        method = "interpolate"
        union = weather.index.union(df.index)
        filled = weather.reindex(union).interpolate(method="time", limit_area="inside")
        aligned = filled.reindex(df.index)
    else:
        method = "nearest"
        aligned = weather.reindex(df.index, method="nearest", tolerance=tolerance)

    coverage = {
        str(c): float(aligned[c].notna().mean()) if len(aligned) else 0.0 for c in aligned.columns
    }
    both = int(aligned.notna().all(axis=1).sum())
    overlap = aligned.dropna(how="all").index
    evidence = make_evidence(
        overlap,
        [str(c) for c in aligned.columns],
        both,
        "align_weather",
        {"method": method},
    )
    summary = (
        f"Aligned {len(numeric)} weather variable(s) to {len(df)} timestamp(s) ({method}); "
        f"{both} row(s) have weather for every variable."
    )
    return AlignedWeather(
        df=aligned,
        method=method,
        series_step_seconds=step_s.total_seconds() if step_s is not None else None,
        weather_step_seconds=step_w.total_seconds(),
        coverage=coverage,
        rows=len(df),
        rows_with_weather=both,
        evidence=evidence,
        log_entry=analysis_log(
            "align_weather",
            summary,
            evidence,
            counts={"rows": len(df), "rows_with_weather": both},
            rows=len(df),
        ),
    )


# ---------------------------------------------------------------------------- weather link


class LagCorrelation(BaseModel):
    """Correlation when the weather leads the series by `lag_steps` readings."""

    lag_steps: int
    lag_hours: float
    pearson: float | None = None
    n: int


class DegreeDayLink(BaseModel):
    """Association with heating or cooling degree values (degrees beyond the base)."""

    kind: Literal["heating", "cooling"]
    base_temperature: float
    pearson: float | None = None
    spearman: float | None = None
    n: int
    note: str | None = None


class WeatherLink(BaseModel):
    """How a series is associated with a weather variable. Association is not causation."""

    column: str
    variable: str
    n_pairs: int
    pearson: float | None = None
    spearman: float | None = None
    best_lag_steps: int | None = None
    best_lag_hours: float | None = None
    best_lag_pearson: float | None = None
    max_lag_steps: int
    lag_correlations: list[LagCorrelation] = Field(default_factory=list)
    heating: DegreeDayLink
    cooling: DegreeDayLink
    coverage: float
    step_seconds: float
    caveat: str = (
        "These correlations describe how the series and the weather move together; they do not "
        "show that one causes the other. Readings are correlated in time, so no p-values are given."
    )
    evidence: Evidence
    log_entry: StepLog


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < MIN_PAIRS or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return None, None, n
    return float(stats.pearsonr(a[ok], b[ok])[0]), float(stats.spearmanr(a[ok], b[ok])[0]), n


def weather_link(
    df: pd.DataFrame,
    column: str,
    weather_df: pd.DataFrame,
    max_lag: int = 48,
    *,
    variable: str | None = None,
    base_temperature: float = 18.0,
) -> WeatherLink:
    """Pearson and Spearman association, best lag, and degree-day association with a variable.

    `variable` defaults to temperature_2m if present, else the first weather column. Lags are
    counted in readings of the series (weather leading the series), from 0 to `max_lag`.
    Heating and cooling values are degrees below or above `base_temperature` (default 18 C).
    """
    if max_lag < 0:
        raise AnalysisError("max_lag cannot be negative.")
    series = numeric_series(df, column)
    series = series[np.isfinite(series.to_numpy())]
    if variable is None:
        variable = (
            "temperature_2m"
            if "temperature_2m" in weather_df.columns
            else str(weather_df.columns[0])
        )
    if variable not in weather_df.columns:
        raise WeatherError(f"The weather table has no column '{variable}'.")
    grid_series, step = to_regular_grid(series)
    aligned = align_weather(pd.DataFrame(index=grid_series.index), weather_df[[variable]])
    temp = aligned.df[variable]
    x, w = grid_series.to_numpy(), temp.to_numpy()
    pearson, spearman, n_pairs = _corr(x, w)
    if n_pairs < MIN_PAIRS:
        raise AnalysisError(
            f"Only {n_pairs} time(s) have both '{column}' and weather '{variable}'; "
            f"at least {MIN_PAIRS} are needed."
        )
    limit = min(max_lag, max(0, len(x) // 4))
    lags: list[LagCorrelation] = []
    for k in range(limit + 1):
        shifted = temp.shift(k).to_numpy()
        r, _, n = _corr(x, shifted)
        lags.append(
            LagCorrelation(lag_steps=k, lag_hours=k * step.total_seconds() / 3600, pearson=r, n=n)
        )
    scored = [lc for lc in lags if lc.pearson is not None]
    best = max(scored, key=lambda lc: abs(lc.pearson)) if scored else None  # type: ignore[arg-type]

    def degree_link(kind: Literal["heating", "cooling"]) -> DegreeDayLink:
        values = (base_temperature - temp) if kind == "heating" else (temp - base_temperature)
        values = values.clip(lower=0).to_numpy()
        r, rho, n = _corr(x, values)
        note = None
        if r is None:
            note = f"No {kind} degrees in this period, or too few overlapping readings."
        return DegreeDayLink(
            kind=kind, base_temperature=base_temperature, pearson=r, spearman=rho, n=n, note=note
        )

    heating, cooling = degree_link("heating"), degree_link("cooling")
    used_index = grid_series.index[np.isfinite(x) & np.isfinite(w)]
    evidence = make_evidence(
        used_index,
        [column, variable],
        n_pairs,
        "weather_link",
        {"max_lag": max_lag, "base_temperature": base_temperature},
    )
    summary = (
        f"'{column}' is associated with weather '{variable}' (Pearson {pearson:.2f}, Spearman "
        f"{spearman:.2f}, n={n_pairs})"
        + (
            f"; strongest at a lag of {best.lag_hours:g} h (r={best.pearson:.2f})."
            if best is not None
            else "."
        )
        if pearson is not None and spearman is not None
        else f"'{column}' and weather '{variable}': too little variation to correlate "
        f"(n={n_pairs})."
    )
    return WeatherLink(
        column=column,
        variable=variable,
        n_pairs=n_pairs,
        pearson=pearson,
        spearman=spearman,
        best_lag_steps=best.lag_steps if best else None,
        best_lag_hours=best.lag_hours if best else None,
        best_lag_pearson=best.pearson if best else None,
        max_lag_steps=limit,
        lag_correlations=lags,
        heating=heating,
        cooling=cooling,
        coverage=float(np.isfinite(w).mean()),
        step_seconds=step.total_seconds(),
        evidence=evidence,
        log_entry=analysis_log(
            "weather_link",
            summary,
            evidence,
            counts={"pairs": n_pairs, "lags_tested": len(lags)},
            rows=len(df),
        ),
    )
