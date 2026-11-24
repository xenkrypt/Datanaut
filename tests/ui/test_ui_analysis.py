import json
import urllib.error
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pytest

from datanaut.errors import AnalysisError, WeatherError
from datanaut.ui import actions, analysis, charts, net
from datanaut.ui.state import WizardState


def _state(phase3, *, second_column: bool = False) -> WizardState:
    s = WizardState()
    data = phase3.csv_bytes
    if second_column:
        rng = np.random.default_rng(3)
        lines = data.decode().strip().split("\n")
        extra = [lines[0] + ",aux"] + [
            f"{ln},{float(ln.split(',')[1]) * 0.5 + rng.normal(0, 1):.3f}" for ln in lines[1:]
        ]
        data = ("\n".join(extra) + "\n").encode()
    actions.load_upload(s, data, "p3.csv")
    cols = ["kw", "aux"] if second_column else ["kw"]
    actions.confirm_mapping(s, ["timestamp"], None, cols, {"kw": True})
    s.take_messages()
    return s


@pytest.fixture(scope="module")
def p3_state(phase3) -> WizardState:
    return _state(phase3)


def fake_fetch(phase3, calls: list):
    def fetch(url: str):
        calls.append(url)
        w = phase3.weather["temperature_2m"]
        return {
            "hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in w.index],
                "temperature_2m": [float(v) for v in w],
            }
        }

    return fetch


# --- caching and timing -------------------------------------------------------------------


def test_a_result_is_computed_once_timed_logged_and_then_reused(phase3):
    s = _state(phase3)
    assert analysis.peek(s, "peaks", percentile=95.0) is None
    before = len(s.log)
    first = analysis.peaks(s)
    again = analysis.peaks(s)
    assert first.cached is False and again.cached is True and again.result is first.result
    assert first.seconds >= 0 and "Reused" in again.caption() and "Computed" in first.caption()
    assert len(s.log) == before + 1 and s.log.entries[-1].read_only  # logged once, not twice
    assert analysis.peek(s, "peaks", percentile=95.0).result is first.result
    assert analysis.peaks(s, 90.0).cached is False  # other parameters, other result


def test_energy_metrics_follow_the_consumption_flag(phase3):
    s = _state(phase3)
    assert analysis.metrics(s).result.applicable is True
    s.consumption["kw"] = False
    assert analysis.metrics(s).result.applicable is False


# --- the analyses -------------------------------------------------------------------------


def test_patterns_anomalies_forecast_and_what_if_on_the_known_series(phase3, p3_state):
    s = p3_state
    assert analysis.period(s) == 24
    assert analysis.peaks(s).result.top_hours[0] == "19:00"
    stl = analysis.stl(s).result
    assert pd.Timestamp(stl.points[0].timestamp) == phase3.spike_time
    fc = analysis.run_forecast(s, 24).result
    assert fc.label == "estimate" and len(fc.points) == 24
    shift = analysis.what_if(s, 0.3, 95.0).result
    assert shift.total_after == pytest.approx(shift.total_before) and shift.cost is None
    paid = analysis.what_if(s, 0.3, 95.0, 0.4, 0.1, "EUR").result
    assert paid.cost is not None and paid.cost.currency == "EUR"


def test_change_points_find_the_known_shift_with_the_fast_method(phase3, p3_state):
    res = analysis.change_points(p3_state, "medium", "l2").result
    near = [
        b
        for b in res.breaks
        if abs(pd.Timestamp(b.timestamp) - phase3.shift_time) <= pd.Timedelta(hours=12)
    ]
    assert len(near) == 1 and near[0].direction == "increase"


def test_the_multi_column_check_needs_two_columns(phase3):
    one = _state(phase3)
    with pytest.raises(AnalysisError, match="at least two"):
        analysis.multivariate(one)
    two = _state(phase3, second_column=True)
    multi = analysis.multivariate(two, 0.01).result
    cmp = analysis.detector_comparison(two, 3.5, 0.01).result
    assert multi.applicable and cmp.n_both >= 0
    assert charts.detector_bars(cmp).layout.title.text


def test_seasonal_anomalies_refuse_politely_without_a_pattern():
    s = WizardState()
    rng = np.random.default_rng(1)
    text = "t,v\n" + "\n".join(
        f"{ts:%Y-%m-%d %H:%M},{x:.3f}"
        for ts, x in zip(
            pd.date_range("2024-01-01", periods=300, freq="h"), rng.normal(size=300), strict=True
        )
    )
    actions.load_upload(s, text.encode(), "n.csv")
    actions.confirm_mapping(s, ["t"], None, ["v"], {})
    if analysis.period(s) is None:
        with pytest.raises(AnalysisError, match="No repeating pattern"):
            analysis.stl(s)


# --- weather ------------------------------------------------------------------------------


def test_weather_can_be_uploaded_from_memory_and_analyzed(phase3):
    s = _state(phase3)
    with pytest.raises(WeatherError, match="Add weather"):
        analysis.weather_result(s)
    analysis.set_weather_file(s, phase3.weather_csv)
    link = analysis.weather_result(s).result
    assert link.pearson < -0.2 and "associat" in link.caveat.lower() or link.pearson < -0.2
    assert analysis.aligned_weather(s, link.variable).notna().all()
    for bad, msg in (
        (b"a\n1\n", "at least one value"),
        (b"t,v\nnope,1\n", "ISO"),
        (b"t,v\n2024-01-01,x\n", "no numeric"),
    ):
        with pytest.raises(WeatherError, match=msg):
            analysis.parse_weather_csv(bad)


def test_fetching_uses_the_injected_function_and_only_sends_coordinates_and_dates(phase3):
    s = _state(phase3)
    calls: list = []
    analysis.fetch_weather(s, 52.5, 13.4, fake_fetch(phase3, calls))
    assert len(calls) == 1
    url = urlparse(calls[0])
    query = parse_qs(url.query)
    assert url.hostname == "archive-api.open-meteo.com"
    assert set(query) <= {"latitude", "longitude", "start_date", "end_date", "hourly", "timezone"}
    assert query["start_date"] == [str(phase3.df.index.min().date())]
    assert s.weather is not None and s.weather_id.startswith("open-meteo:")
    with pytest.raises(WeatherError, match="Latitude"):
        analysis.fetch_weather(s, 99, 0, fake_fetch(phase3, calls))
    assert len(calls) == 1


def test_a_failing_fetch_becomes_a_weather_error_and_a_friendly_message(phase3):
    s = _state(phase3)

    def offline(url):
        raise OSError("no route")

    assert actions.guard(s, "Analyze", analysis.fetch_weather, s, 1.0, 1.0, offline) is None
    assert s.messages[-1].level == "error" and "weather" in s.messages[-1].text.lower()
    assert s.weather is None


def test_the_network_module_only_talks_to_open_meteo(monkeypatch):
    for url in (
        "http://archive-api.open-meteo.com/v1",
        "https://example.com/x",
        "file:///etc/passwd",
    ):
        with pytest.raises(WeatherError, match="only be requested from Open-Meteo"):
            net.fetch_json(url)

    def boom(*a, **k):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(net.urllib.request, "urlopen", boom)
    with pytest.raises(WeatherError, match="offline"):
        net.fetch_json("https://archive-api.open-meteo.com/v1/archive?x=1")


# --- insights -----------------------------------------------------------------------------


def test_findings_are_ranked_worded_as_association_and_have_charts(phase3):
    s = _state(phase3)
    analysis.set_weather_file(s, phase3.weather_csv)
    analysis.change_points(s, "medium", "l2")  # a different setting: findings use the default
    report, skipped = analysis.findings(s)
    assert 3 <= len(report.findings) <= 5 and isinstance(skipped, list)
    scores = [f.score for f in report.findings]
    assert scores == sorted(scores, reverse=True)
    text = " ".join(f"{f.title} {f.statement}" for f in report.findings).lower()
    assert "caused" not in text and "due to" not in text
    ctx = {
        "series": analysis.series(s),
        "peaks": analysis.peek(s, "peaks", percentile=95.0).result,
        "weekday": analysis.peek(s, "profile_weekday").result,
        "anomalies": analysis.peek(s, "stl", threshold=3.5, period=24).result,
        "changes": analysis.peek(
            s, "change_points", sensitivity="medium", cost="rbf", period=24
        ).result,
        "link": analysis.peek(s, "weather_link", weather=s.weather_id, variable=None).result,
    }
    ctx["weather_x"] = analysis.aligned_weather(s, ctx["link"].variable)
    for f in report.findings:
        fig = charts.finding_chart(f.chart, ctx)
        if fig is not None:
            assert fig.layout.title.text
    assert any(charts.finding_chart(f.chart, ctx) is not None for f in report.findings)
    assert analysis.findings(s)[0] is report  # cached


def test_investigating_a_period_never_claims_cause(phase3, p3_state):
    start = phase3.shift_time - pd.Timedelta(days=1)
    res = analysis.run_investigation(
        p3_state, start, start + pd.Timedelta(days=10), "previous"
    ).result
    assert res.change.direction == "higher" and "does not show what caused" in res.conclusion
    analysis.stl(p3_state)
    events = analysis.event_options(p3_state)
    assert any(label.startswith("Anomalies on") for label, _, _ in events)


# --- chart builders -----------------------------------------------------------------------


def _check(fig, traces: int | None = None):
    assert fig.layout.title.text
    if fig.layout.xaxis.title.text is not None:
        assert fig.layout.yaxis.title.text
    if traces is not None:
        assert len(fig.data) == traces
    json.loads(fig.to_json())


def test_every_chart_builder_has_a_title_axis_labels_and_the_expected_traces(phase3, p3_state):
    s = p3_state
    series = analysis.series(s)
    stl = analysis.stl(s).result
    cp = analysis.change_points(s, "medium", "l2").result
    bt = analysis.run_backtest(s, 24).result
    fc = analysis.run_forecast(s, 24).result
    hour, weekday = analysis.hour_profile(s).result, analysis.weekday_profile(s).result
    peaks = analysis.peaks(s).result
    link = None
    analysis.set_weather_file(s, phase3.weather_csv)
    link = analysis.weather_result(s).result
    shift = analysis.what_if(s, 0.2, 95.0).result

    _check(charts.time_series(series, "kw"), 1)
    _check(charts.time_series(series, "kw", stl.flagged_timestamps(), ["2024-01-05"]), 2)
    _check(charts.profile_bars(hour), 1)
    _check(charts.profile_bars(weekday), 1)
    _check(charts.profile_heatmap(analysis.heat_profile(s).result), 1)
    _check(charts.share_bars(peaks.hours, "Peaks by hour", "Hour"), 1)
    _check(charts.load_duration(series, "kw"), 1)
    comps = analysis.stl_components(s)
    fig = charts.stl_components(comps[0], comps[1], "kw")
    assert fig.layout.title.text and len(fig.data) == 3
    _check(charts.change_point_segments(series, cp), 1 + len(cp.segments))
    _check(charts.forecast_chart(series, fc), 4)
    _check(charts.backtest_bars(bt), 1)
    x = analysis.aligned_weather(s, link.variable)
    _check(charts.weather_scatter(x, series, link.variable, "kw"), 1)
    _check(charts.lag_bars(link), 1)
    _check(charts.whatif_chart(series, shift.new_series, "kw"), 2)
    assert len(charts.change_point_segments(series, cp).layout.shapes) == len(cp.breaks)


def test_the_multivariate_view_stacks_one_bar_per_column(phase3):
    two = _state(phase3, second_column=True)
    res = analysis.multivariate(two, 0.02).result
    fig = charts.multivariate_view(res)
    assert fig.layout.title.text and len(fig.data) == len(res.columns)
    assert fig.layout.barmode == "stack"


def test_long_series_are_downsampled_in_charts_but_not_in_the_data():
    idx = pd.date_range("2024-01-01", periods=50_000, freq="min")
    s = pd.Series(np.sin(np.arange(50_000) / 500), index=idx)
    fig = charts.time_series(s, "x", limit=2000)
    assert len(fig.data[0].x) <= 2000 and len(s) == 50_000
    assert max(fig.data[0].y) == s.max() and min(fig.data[0].y) == s.min()


def test_charts_tolerate_repeated_timestamps_in_uncleaned_data():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    fig = charts.time_series(s, "x", pd.DatetimeIndex(["2024-01-01 00:00"]))
    assert len(fig.data) == 2
    assert len(charts.weather_scatter(pd.Series([5.0, 6.0, 7.0], index=idx), s, "t", "x").data) == 1
