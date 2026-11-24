import copy
import json
from datetime import date
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from datanaut.core.fingerprint import fingerprint
from datanaut.core.weather import (
    ARCHIVE_URL,
    FORECAST_URL,
    align_weather,
    fetch_open_meteo,
    open_meteo_url,
    weather_link,
)
from datanaut.errors import AnalysisError, WeatherError


def temperature(index):
    """A believable temperature: a daily cycle plus a slow wander, so lags are distinguishable."""
    t = np.arange(len(index))
    rng = np.random.default_rng(3)
    wander = np.cumsum(rng.normal(0, 0.25, len(index)))
    wander -= np.linspace(wander[0], wander[-1], len(index))  # keep it bounded
    return 10 + 8 * np.sin(2 * np.pi * (t - 9) / 24) + wander


def fake_fetch_factory(value_fn=lambda t: 10.0 + t % 24):
    calls: list[str] = []

    def fetch(url: str):
        calls.append(url)
        q = parse_qs(urlparse(url).query)
        start, end = date.fromisoformat(q["start_date"][0]), date.fromisoformat(q["end_date"][0])
        times = pd.date_range(
            start, pd.Timestamp(end) + pd.Timedelta(days=1), freq="h", inclusive="left"
        )
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
        for name in q["hourly"][0].split(","):
            hourly[name] = [value_fn(i) for i in range(len(times))]
        return {"hourly": hourly, "latitude": 52.5}

    return fetch, calls


def weather_frame(days=40, start="2024-01-01", freq="h"):
    idx = pd.date_range(start, periods=days * 24, freq=freq)
    return pd.DataFrame({"temperature_2m": temperature(idx)}, index=idx)


# ------------------------------------------------------------------ request and response


def test_url_for_archive_and_forecast_has_validated_parameters():
    url = open_meteo_url(
        52.52, 13.405, "2024-01-01", date(2024, 1, 31), ["temperature_2m", "windspeed_10m"]
    )
    parts = urlparse(url)
    assert url.startswith(ARCHIVE_URL) and parts.scheme == "https"
    q = parse_qs(parts.query)
    assert q["latitude"] == ["52.5200"] and q["longitude"] == ["13.4050"]
    assert q["start_date"] == ["2024-01-01"] and q["end_date"] == ["2024-01-31"]
    assert q["hourly"] == ["temperature_2m,windspeed_10m"] and q["timezone"] == ["auto"]
    assert open_meteo_url(
        0, 0, "2024-01-01", "2024-01-02", ["temperature_2m"], source="forecast"
    ).startswith(FORECAST_URL)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"lat": 91}, "Latitude"),
        ({"lon": -181}, "Latitude"),
        ({"start": "2024-02-01", "end": "2024-01-01"}, "after the end"),
        ({"start": "not-a-date"}, "not a valid YYYY-MM-DD"),
        ({"variables": []}, "at least one weather variable"),
        ({"variables": ["temperature_2m&evil=1"]}, "not a valid weather variable"),
        ({"timezone": "UTC; drop"}, "not a valid timezone"),
    ],
)
def test_bad_requests_are_rejected_before_anything_is_sent(kwargs, message):
    fetch, calls = fake_fetch_factory()
    args = {
        "lat": 52.5,
        "lon": 13.4,
        "start": "2024-01-01",
        "end": "2024-01-02",
        "variables": ["temperature_2m"],
    }
    args.update(kwargs)
    with pytest.raises(WeatherError, match=message):
        fetch_open_meteo(
            args["lat"],
            args["lon"],
            args["start"],
            args["end"],
            args["variables"],
            fetch,
            timezone=args.get("timezone", "auto"),
        )
    assert calls == []


def test_fetch_returns_a_time_indexed_frame_using_only_the_injected_function():
    fetch, calls = fake_fetch_factory(lambda i: 5.0 + i)
    df = fetch_open_meteo(52.5, 13.4, "2024-01-01", "2024-01-02", ["temperature_2m"], fetch)
    assert len(calls) == 1 and "latitude=52.5000" in calls[0]
    assert (
        isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None and df.index.name == "time"
    )
    assert (
        len(df) == 48
        and df.index[0] == pd.Timestamp("2024-01-01 00:00")
        and df.index[-1] == pd.Timestamp("2024-01-02 23:00")
    )
    assert (
        df["temperature_2m"].iloc[:3].tolist() == [5.0, 6.0, 7.0]
        and df["temperature_2m"].dtype == float
    )


def test_null_readings_become_nan_and_timezone_suffixes_are_dropped():
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00Z", "2024-01-01T01:00Z"],
            "temperature_2m": [None, 3.5],
        }
    }
    df = fetch_open_meteo(0, 0, "2024-01-01", "2024-01-01", ["temperature_2m"], lambda url: payload)
    assert (
        np.isnan(df["temperature_2m"].iloc[0])
        and df["temperature_2m"].iloc[1] == 3.5
        and df.index.tz is None
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not json", "not a JSON object"),
        (
            {"error": True, "reason": "Latitude must be in range"},
            "refused the request: Latitude must be in range",
        ),
        ({}, "no hourly data"),
        ({"hourly": {"temperature_2m": [1.0]}}, "no hourly data"),
        ({"hourly": {"time": ["2024-01-01T00:00"]}}, "missing the variable 'temperature_2m'"),
        ({"hourly": {"time": [], "temperature_2m": []}}, "no timestamps"),
        (
            {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"], "temperature_2m": [1.0]}},
            "1 values for 2 timestamps",
        ),
        ({"hourly": {"time": ["2024-01-01T00:00"], "temperature_2m": ["warm"]}}, "not numbers"),
        ({"hourly": {"time": ["2024-01-01T00:00"], "temperature_2m": [True]}}, "not numbers"),
        ({"hourly": {"time": ["yesterday"], "temperature_2m": [1.0]}}, "could not be read"),
        ({"hourly": {"time": "2024-01-01", "temperature_2m": 1.0}}, "no timestamps"),
    ],
)
def test_malformed_weather_responses_are_rejected_with_a_friendly_error(payload, message):
    with pytest.raises(WeatherError, match=message):
        fetch_open_meteo(0, 0, "2024-01-01", "2024-01-01", ["temperature_2m"], lambda url: payload)


def test_a_failing_fetch_function_becomes_a_weather_error():
    def broken(url):
        raise ConnectionError("no route to host")

    with pytest.raises(WeatherError, match=r"request failed \(ConnectionError\)") as err:
        fetch_open_meteo(0, 0, "2024-01-01", "2024-01-01", ["temperature_2m"], broken)
    assert "no route to host" not in err.value.message  # the failure's own text is never echoed


def test_the_module_contains_no_network_code():
    source = (
        Path(__file__).resolve().parent.parent / "src" / "datanaut" / "core" / "weather.py"
    ).read_text(encoding="utf-8")
    for forbidden in ("import requests", "urlopen", "import httpx", "import socket", "http.client"):
        assert forbidden not in source


# ------------------------------------------------------------------ alignment


def test_same_frequency_aligns_exactly_with_full_coverage():
    w = weather_frame(days=3)
    series = pd.DataFrame({"v": range(48)}, index=w.index[10:58])
    a = align_weather(series, w)
    assert a.method == "nearest" and a.coverage == {"temperature_2m": 1.0}
    assert a.df["temperature_2m"].tolist() == w["temperature_2m"].iloc[10:58].tolist()
    assert (a.rows, a.rows_with_weather) == (48, 48)


def test_hourly_weather_onto_a_daily_series_uses_the_daily_mean():
    w = weather_frame(days=5)
    daily = pd.DataFrame({"v": range(5)}, index=pd.date_range("2024-01-01", periods=5, freq="D"))
    a = align_weather(daily, w)
    assert a.method == "resample_mean"
    expected = w["temperature_2m"].resample("D").mean()
    np.testing.assert_allclose(a.df["temperature_2m"].to_numpy(), expected.to_numpy())


def test_coarser_weather_is_interpolated_in_time_but_never_extrapolated():
    idx3 = pd.date_range("2024-01-01", periods=5, freq="3h")
    w = pd.DataFrame({"temperature_2m": [0.0, 3.0, 6.0, 9.0, 12.0]}, index=idx3)
    hourly = pd.DataFrame({"v": 0.0}, index=pd.date_range("2024-01-01", periods=20, freq="h"))
    a = align_weather(hourly, w)
    assert a.method == "interpolate"
    assert a.df["temperature_2m"].iloc[:13].tolist() == pytest.approx(list(np.arange(13.0)))
    assert a.df["temperature_2m"].iloc[13:].isna().all()  # beyond the last reading
    assert a.coverage["temperature_2m"] == pytest.approx(13 / 20) and a.rows_with_weather == 13


def test_alignment_reports_partial_coverage_and_rejects_bad_inputs():
    w = weather_frame(days=2)
    series = pd.DataFrame({"v": 0.0}, index=pd.date_range("2024-01-01", periods=96, freq="h"))
    assert align_weather(series, w).coverage["temperature_2m"] == pytest.approx(0.5)
    with pytest.raises(WeatherError, match="timezones"):
        align_weather(series.tz_localize("UTC"), w)
    with pytest.raises(WeatherError, match="needs a time index"):
        align_weather(pd.DataFrame({"v": [1.0]}), w)
    with pytest.raises(WeatherError, match="no numeric columns"):
        align_weather(series, w.assign(temperature_2m="x"))
    with pytest.raises(WeatherError, match="at least two timestamps"):
        align_weather(series, w.iloc[:1])


# ------------------------------------------------------------------ the link


def heating_series(w, lag=3, slope=-1.5, noise=1.0, seed=1):
    """Load that rises when it is cold, reacting to the temperature `lag` hours earlier."""
    rng = np.random.default_rng(seed)
    temp = w["temperature_2m"].shift(lag).bfill()
    return pd.DataFrame({"load": 50 + slope * temp + rng.normal(0, noise, len(w))}, index=w.index)


def test_temperature_driven_series_shows_negative_association_and_the_true_lag():
    w = weather_frame(days=40)
    df = heating_series(w, lag=3)
    link = weather_link(df, "load", w, max_lag=24)
    assert link.pearson < -0.4 and link.spearman < -0.4  # associated: colder, higher load
    assert link.best_lag_steps == 3 and link.best_lag_hours == 3.0
    assert link.best_lag_pearson < -0.95
    assert link.heating.pearson > 0.7 and abs(link.cooling.pearson or 0) < link.heating.pearson
    assert link.heating.kind == "heating" and link.heating.base_temperature == 18.0
    assert link.max_lag_steps == 24 and len(link.lag_correlations) == 25
    assert link.n_pairs == len(df) and link.coverage == 1.0 and link.step_seconds == 3600


def test_positive_driver_gives_positive_sign_and_cooling_degrees_associate():
    w = weather_frame(days=40)
    w["temperature_2m"] += 12  # a hot place: often above 18
    df = heating_series(w, lag=0, slope=+2.0)
    link = weather_link(df, "load", w, max_lag=6)
    assert link.pearson > 0.8 and link.best_lag_steps == 0
    assert link.cooling.pearson > 0.8 and link.cooling.kind == "cooling"


def test_the_degree_day_base_is_a_parameter():
    w = weather_frame(days=30)
    df = heating_series(w)
    cold_base = weather_link(df, "load", w, max_lag=3, base_temperature=0.0)
    assert cold_base.heating.base_temperature == 0.0
    assert (
        cold_base.heating.pearson is None
        or cold_base.heating.pearson != weather_link(df, "load", w, max_lag=3).heating.pearson
    )


def test_unrelated_weather_shows_no_strong_association():
    w = weather_frame(days=40)
    rng = np.random.default_rng(99)
    df = pd.DataFrame({"load": rng.normal(50, 5, len(w))}, index=w.index)
    link = weather_link(df, "load", w, max_lag=12)
    assert abs(link.pearson) < 0.15 and abs(link.spearman) < 0.15
    assert abs(link.best_lag_pearson) < 0.2


def test_daily_series_against_hourly_weather():
    w = weather_frame(days=120)
    daily_temp = w["temperature_2m"].resample("D").mean()
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {"load": 100 - 2.0 * daily_temp.to_numpy() + rng.normal(0, 0.5, len(daily_temp))},
        index=daily_temp.index,
    )
    link = weather_link(df, "load", w, max_lag=7)
    assert link.step_seconds == 86400 and link.pearson < -0.95 and link.best_lag_steps == 0


def test_not_enough_overlap_or_variation_is_reported_clearly():
    w = weather_frame(days=5)
    later = pd.DataFrame(
        {"load": np.arange(48.0)}, index=pd.date_range("2025-01-01", periods=48, freq="h")
    )
    with pytest.raises(AnalysisError, match="Only 0 time"):
        weather_link(later, "load", w)
    flat = w.assign(temperature_2m=15.0)
    df = heating_series(weather_frame(days=5))
    link = weather_link(df, "load", flat, max_lag=4)
    assert link.pearson is None and link.best_lag_steps is None
    assert "too little variation" in link.log_entry.summary
    with pytest.raises(WeatherError, match="no column 'nope'"):
        weather_link(df, "load", w, variable="nope")
    with pytest.raises(AnalysisError, match="cannot be negative"):
        weather_link(df, "load", w, max_lag=-1)


def test_link_has_evidence_log_json_and_leaves_inputs_alone():
    w = weather_frame(days=20)
    df = heating_series(w)
    snapshots = (copy.deepcopy(df), copy.deepcopy(w))
    before = (fingerprint(df), fingerprint(w))
    link = weather_link(df, "load", w, max_lag=6)
    pd.testing.assert_frame_equal(df, snapshots[0])
    pd.testing.assert_frame_equal(w, snapshots[1])
    assert (fingerprint(df), fingerprint(w)) == before
    ev = link.evidence
    assert ev.columns == ["load", "temperature_2m"] and ev.method == "weather_link"
    assert ev.n_observations == link.n_pairs and ev.parameters == {
        "max_lag": 6,
        "base_temperature": 18.0,
    }
    assert ev.start == str(df.index.min()) and ev.end == str(df.index.max())
    assert link.log_entry.read_only and "associated with" in link.log_entry.summary
    assert "causes" not in link.caveat.lower() or "do not show that one causes" in link.caveat
    assert json.loads(link.model_dump_json())["best_lag_steps"] == 3
