"""Analysis logic for the Analyze and Insights screens: cached, timed calls into the core.

No streamlit import. Every result is cached by (data fingerprint, name, parameters), so a rerun
never recomputes it. A freshly computed result adds its read-only log entry to the pipeline log.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
import pandas as pd

from datanaut.core.anomalies import (
    compare_detectors,
    multivariate_anomalies,
    stl_anomalies,
)
from datanaut.core.changepoints import detect_change_points
from datanaut.core.characterize import characterize
from datanaut.core.cleaner import seasonal_fit
from datanaut.core.evidence import to_regular_grid
from datanaut.core.findings import FindingsReport, generate_findings
from datanaut.core.forecast import backtest, forecast, recommend_model
from datanaut.core.investigate import investigate
from datanaut.core.patterns import energy_metrics, load_profile, peak_periods
from datanaut.core.weather import align_weather, fetch_open_meteo, weather_link
from datanaut.core.whatif import Tariff, peak_shift
from datanaut.errors import AnalysisError, WeatherError
from datanaut.ui import net
from datanaut.ui.state import WizardState, cache_key

Fetch = Callable[[str], Any]


@dataclass(frozen=True)
class Run:
    """A result, how long it took to compute, and whether this call reused an earlier one."""

    result: Any
    seconds: float
    cached: bool

    def caption(self) -> str:
        if self.cached:
            return f"Reused an earlier result (first computed in {self.seconds:.1f} s)."
        return f"Computed in {self.seconds:.1f} s."


def _key(state: WizardState, name: str, params: dict[str, Any]) -> tuple:
    return cache_key(state.fingerprint, name, column=column(state), **params)


def run(state: WizardState, name: str, fn: Callable[[], Any], **params: Any) -> Run:
    """Compute (or reuse) one analysis."""
    key = _key(state, name, params)
    cached = key in state.cache

    def timed() -> tuple[Any, float]:
        start = perf_counter()
        value = fn()
        return value, perf_counter() - start

    started = perf_counter()
    result, seconds = state.get_or_compute(key, timed)
    if not cached and hasattr(result, "log_entry"):
        state.log.extend([result.log_entry])
    return Run(result, seconds if cached else perf_counter() - started, cached)


def peek(state: WizardState, name: str, **params: Any) -> Run | None:
    """An earlier result for these parameters, or None (never computes anything)."""
    key = _key(state, name, params)
    if key not in state.cache:
        return None
    result, seconds = state.cache.get_or_compute(key, lambda: None)
    return Run(result, seconds, True)


def column(state: WizardState) -> str:
    """The column being analyzed (the first value column unless the user chose another)."""
    if state.analysis_column in state.value_columns:
        return str(state.analysis_column)
    return state.value_columns[0]


def frame(state: WizardState) -> pd.DataFrame:
    assert state.store is not None
    return state.store.current()


def series(state: WizardState) -> pd.Series:
    return frame(state)[column(state)]


# --- characterization and patterns --------------------------------------------------------


def characterization(state: WizardState) -> Run:
    col = column(state)
    return run(state, "characterize", lambda: characterize(frame(state), col))


def period(state: WizardState) -> int | None:
    try:
        return characterization(state).result.seasonal_period
    except AnalysisError:
        return None


def consumption(state: WizardState) -> bool:
    return bool(state.consumption.get(column(state), False))


def hour_profile(state: WizardState) -> Run:
    return run(state, "profile_hour", lambda: load_profile(frame(state), column(state), "hour"))


def weekday_profile(state: WizardState) -> Run:
    return run(
        state, "profile_weekday", lambda: load_profile(frame(state), column(state), "weekday")
    )


def heat_profile(state: WizardState) -> Run:
    return run(
        state,
        "profile_heat",
        lambda: load_profile(frame(state), column(state), "hour_x_weekday"),
    )


def metrics(state: WizardState) -> Run:
    flag = consumption(state)
    return run(
        state,
        "metrics",
        lambda: energy_metrics(frame(state), column(state), consumption=flag),
        consumption=flag,
    )


def peaks(state: WizardState, percentile: float = 95.0) -> Run:
    return run(
        state,
        "peaks",
        lambda: peak_periods(frame(state), column(state), percentile),
        percentile=percentile,
    )


# --- anomalies and change points ----------------------------------------------------------


def stl(state: WizardState, threshold: float = 3.5) -> Run:
    p = period(state)
    if not p:
        raise AnalysisError(
            "No repeating pattern was detected in this column, so seasonal anomaly detection "
            "cannot run."
        )
    return run(
        state,
        "stl",
        lambda: stl_anomalies(frame(state), column(state), p, threshold),
        threshold=threshold,
        period=p,
    )


def multivariate(state: WizardState, contamination: float = 0.01) -> Run:
    cols = list(state.value_columns)
    if len(cols) < 2:
        raise AnalysisError("The multivariate check needs at least two value columns.")
    return run(
        state,
        "multivariate",
        lambda: multivariate_anomalies(frame(state), cols, contamination=contamination),
        contamination=contamination,
        columns=cols,
    )


def detector_comparison(state: WizardState, threshold: float, contamination: float) -> Run:
    uni = stl(state, threshold).result
    multi = multivariate(state, contamination).result
    return run(
        state,
        "compare_detectors",
        lambda: compare_detectors(uni, multi),
        threshold=threshold,
        contamination=contamination,
    )


def change_points(
    state: WizardState,
    sensitivity: Literal["low", "medium", "high"] = "medium",
    cost: Literal["rbf", "l2"] = "rbf",
) -> Run:
    p = period(state)
    return run(
        state,
        "change_points",
        lambda: detect_change_points(
            frame(state), column(state), sensitivity=sensitivity, cost=cost, period=p
        ),
        sensitivity=sensitivity,
        cost=cost,
        period=p,
    )


def stl_components(state: WizardState) -> tuple[pd.Series, pd.Series] | None:
    """The series and its robust trend-plus-seasonal fit on an even grid, for plotting."""
    p = period(state)
    if not p:
        return None
    grid, _ = to_regular_grid(series(state).replace([np.inf, -np.inf], np.nan))
    if len(grid) < 2 * p + 2 or grid.isna().mean() > 0.5:
        return None
    filled = grid.interpolate(limit_direction="both")
    fitted, _, _ = seasonal_fit(filled.to_numpy(dtype=float), p, 3.5)
    return grid, pd.Series(fitted, index=grid.index, name="trend + seasonal")


# --- forecasting --------------------------------------------------------------------------

BACKTEST_MODELS = ("seasonal_naive", "holt_winters")
"""SARIMA is left out of the interactive screen because it is slow on long series."""


def run_backtest(state: WizardState, horizon: int) -> Run:
    p = period(state)
    return run(
        state,
        "backtest",
        lambda: backtest(frame(state), column(state), list(BACKTEST_MODELS), horizon, period=p),
        horizon=horizon,
        period=p,
    )


def recommend(bt):
    """The recommended model for a backtest (the simplest one within the tolerance of the best)."""
    return recommend_model(bt)


def run_forecast(state: WizardState, horizon: int) -> Run:
    bt = run_backtest(state, horizon).result
    rec = recommend_model(bt)
    if rec.model is None:
        raise AnalysisError("No model could be scored, so there is nothing to forecast with.")
    p = period(state)
    return run(
        state,
        "forecast",
        lambda: forecast(frame(state), column(state), rec.model, horizon, p, backtest_result=bt),
        horizon=horizon,
        model=rec.model,
        period=p,
    )


# --- weather ------------------------------------------------------------------------------


def parse_weather_csv(data: bytes) -> pd.DataFrame:
    """Read an uploaded weather CSV from memory: first column time, the others numeric."""
    try:
        raw = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise WeatherError("The weather file could not be read as a CSV table.") from exc
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


def set_weather_file(state: WizardState, data: bytes) -> None:
    state.weather = parse_weather_csv(data)
    state.weather_id = "file:" + hashlib.sha256(data).hexdigest()[:16]


def date_range(state: WizardState) -> tuple[str, str]:
    index = frame(state).index
    return str(index.min().date()), str(index.max().date())


def fetch_weather(
    state: WizardState,
    lat: float,
    lon: float,
    fetch: Fetch = net.fetch_json,
    variables: tuple[str, ...] = ("temperature_2m",),
) -> None:
    """Download hourly weather for the data's date range through the injected `fetch`."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise WeatherError("Latitude must be between -90 and 90, longitude between -180 and 180.")
    start, end = date_range(state)
    state.weather = fetch_open_meteo(lat, lon, start, end, list(variables), fetch)
    state.weather_id = f"open-meteo:{lat:.3f},{lon:.3f},{start},{end}"


def weather_result(state: WizardState, variable: str | None = None) -> Run:
    if state.weather is None:
        raise WeatherError("Add weather data first: fetch it or upload a CSV file.")
    weather = state.weather
    return run(
        state,
        "weather_link",
        lambda: weather_link(frame(state), column(state), weather, variable=variable),
        weather=state.weather_id,
        variable=variable,
    )


def aligned_weather(state: WizardState, variable: str) -> pd.Series | None:
    """The chosen weather variable on the series' timestamps (for the scatter chart)."""
    if state.weather is None:
        return None
    return align_weather(frame(state), state.weather).df.get(variable)


# --- insights -----------------------------------------------------------------------------


def findings(state: WizardState) -> tuple[FindingsReport, list[str]]:
    """Run the analyses the findings need (reusing cached ones) and rank the findings.

    Returns the report and a list of analyses that were skipped, with the reason.
    """
    skipped: list[str] = []

    def attempt(label: str, fn: Callable[[], Run]) -> Any:
        try:
            return fn().result
        except (AnalysisError, WeatherError) as exc:
            skipped.append(f"{label}: {exc.message}")
            return None

    peaks_r = attempt("Peak periods", lambda: peaks(state))
    metrics_r = attempt("Energy metrics", lambda: metrics(state))
    weekday_r = attempt("Weekday profile", lambda: weekday_profile(state))
    anomalies_r = attempt("Anomalies", lambda: stl(state))
    changes_r = attempt("Change points", lambda: change_points(state))
    backtest_r = attempt("Forecast backtest", lambda: run_backtest(state, 24))
    weather_r = None
    if state.weather is not None:
        weather_r = attempt("Weather", lambda: weather_result(state))
    report = run(
        state,
        "findings",
        lambda: generate_findings(
            peaks=peaks_r,
            metrics=metrics_r,
            weekday_profile=weekday_r,
            weather=weather_r,
            anomalies=anomalies_r,
            change_points=changes_r,
            backtest=backtest_r,
        ),
        weather=state.weather_id,
    ).result
    return report, skipped


def what_if(
    state: WizardState,
    shift_fraction: float,
    percentile: float,
    peak_rate: float | None = None,
    offpeak_rate: float | None = None,
    currency: str = "",
) -> Run:
    tariff = None
    if peak_rate is not None and offpeak_rate is not None:
        tariff = Tariff(peak_rate=peak_rate, offpeak_rate=offpeak_rate, currency=currency)
    return run(
        state,
        "peak_shift",
        lambda: peak_shift(frame(state), column(state), shift_fraction, percentile, tariff=tariff),
        shift_fraction=shift_fraction,
        percentile=percentile,
        tariff=None if tariff is None else tariff.model_dump(),
    )


def event_options(state: WizardState) -> list[tuple[str, str, str]]:
    """Listed anomalies and change points as (label, start, end) periods to investigate."""
    options: list[tuple[str, str, str]] = []
    r = peek(state, "stl", threshold=3.5, period=period(state))
    if r is not None:
        for day in r.result.days[:10]:
            options.append((f"Anomalies on {day.date}", day.date, f"{day.date} 23:59:59"))
    for r in _cached_change_points(state):
        for b in r.breaks[:10]:
            ts = pd.Timestamp(b.timestamp)
            options.append(
                (
                    f"After the {b.direction} at {b.timestamp}",
                    str(ts),
                    str(ts + 7 * pd.Timedelta(days=1)),
                )
            )
    return options


def _cached_change_points(state: WizardState) -> list[Any]:
    p = period(state)
    return [
        r.result
        for sens in ("medium", "low", "high")
        for cost in ("rbf", "l2")
        if (r := peek(state, "change_points", sensitivity=sens, cost=cost, period=p)) is not None
    ][:1]


def run_investigation(
    state: WizardState,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    baseline: Literal["previous", "same_weekdays"],
) -> Run:
    if frame(state).index.has_duplicates:
        raise AnalysisError(
            "The data has repeated timestamps, which the investigation cannot compare. "
            "Apply the duplicate-removal fix on the Clean screen first."
        )
    others = [c for c in state.value_columns if c != column(state)] or None
    cps = _cached_change_points(state)
    cp = cps[0] if cps else None
    return run(
        state,
        "investigate",
        lambda: investigate(
            frame(state),
            column(state),
            start,
            end,
            baseline,
            other_columns=others,
            weather_df=state.weather,
            change_points=cp,
        ),
        start=str(start),
        end=str(end),
        baseline=baseline,
        weather=state.weather_id,
        with_change_points=cp is not None,
    )
