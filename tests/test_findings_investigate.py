import json

import numpy as np
import pandas as pd
import pytest

from datanaut.core.anomalies import stl_anomalies
from datanaut.core.changepoints import detect_change_points
from datanaut.core.findings import FORBIDDEN_WORDS, generate_findings
from datanaut.core.investigate import (
    check_level_change,
    check_peak_hour_shift,
    check_weekday_mix,
    investigate,
)
from datanaut.core.patterns import energy_metrics, load_profile, peak_periods
from datanaut.errors import AnalysisError


def two_regimes(
    days=20, shift_day=10, bump=10.0, peak_before=18, peak_after=18, seed=0, extra=None
):
    """Hourly load with an evening peak; from `shift_day` the level rises (peak may move)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")  # a Monday
    hour = idx.hour.to_numpy()
    after = np.arange(len(idx)) >= shift_day * 24
    peak_hour = np.where(after, peak_after, peak_before)
    values = (
        20
        + 15 * np.exp(-(((hour - peak_hour) / 1.5) ** 2))
        + rng.normal(0, 0.8, len(idx))
        + bump * after
    )
    df = pd.DataFrame({"kw": values}, index=idx)
    for name, make in (extra or {}).items():
        df[name] = make(df["kw"].to_numpy(), rng)
    return df


# ------------------------------------------------------------------ the individual checks


def test_level_change_reports_difference_percent_and_effect_size():
    a, b = pd.Series([20.0, 22.0, 18.0, 20.0] * 5), pd.Series([10.0, 12.0, 8.0, 10.0] * 5)
    c = check_level_change(a, b)
    assert c.direction == "higher" and c.difference == 10 and c.percent == pytest.approx(100.0)
    assert c.effect_size_d == pytest.approx(
        10 / np.sqrt(np.var([20, 22, 18, 20], ddof=1) * 19 / 19 * 1), rel=0.2
    )
    assert check_level_change(b, b).direction == "unchanged"
    assert check_level_change(b, a).direction == "lower"


def test_weekday_mix_and_peak_hour_shift_checks():
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="h")
    weekdays_only = pd.Series(1.0, index=idx[idx.day_of_week < 5])
    weekends_only = pd.Series(1.0, index=idx[idx.day_of_week >= 5])
    mix = check_weekday_mix(weekends_only, weekdays_only)
    assert mix.verdict == "supported" and mix.effect_size == 1.0
    same = check_weekday_mix(weekdays_only, weekdays_only)
    assert same.verdict == "no evidence" and same.effect_size == 0

    def daily_peak(hour):
        v = pd.Series(0.0, index=idx)
        v[idx.hour == hour] = 5.0
        return v

    shifted = check_peak_hour_shift(daily_peak(21), daily_peak(18))
    assert shifted.verdict == "supported" and shifted.effect_size == pytest.approx(3.0)
    assert check_peak_hour_shift(daily_peak(19), daily_peak(18)).verdict == "weak"
    assert check_peak_hour_shift(daily_peak(18), daily_peak(18)).verdict == "no evidence"
    tiny = pd.Series([1.0, 2.0], index=idx[:2])
    assert check_peak_hour_shift(tiny, daily_peak(18)).verdict == "no evidence"


def test_peak_hour_shift_works_across_midnight():
    idx = pd.date_range("2024-01-01", periods=24 * 10, freq="h")

    def daily_peak(hour):
        v = pd.Series(0.0, index=idx)
        v[idx.hour == hour] = 5.0
        return v

    f = check_peak_hour_shift(daily_peak(1), daily_peak(23))  # 23:00 -> 01:00 is two hours
    assert f.effect_size == pytest.approx(2.0, abs=0.01) and f.verdict == "supported"


# ------------------------------------------------------------------ the investigation


def test_investigate_reports_the_change_and_ranks_supported_factors():
    df = two_regimes(peak_before=18, peak_after=21)
    cps = detect_change_points(df, "kw", cost="l2", period=24)
    res = investigate(df, "kw", "2024-01-11", "2024-01-20 23:00", change_points=cps)
    assert res.change.direction == "higher" and res.change.percent > 30
    assert res.change.effect_size_d > 1 and res.change.n_period == 240 and res.baseline.n == 240
    by = {f.name: f for f in res.factors}
    assert by["change_point"].verdict == "supported" and by["change_point"].numbers["location"] in (
        "inside",
        "near",
    )
    assert by["peak_hour_shift"].verdict == "supported" and by[
        "peak_hour_shift"
    ].effect_size == pytest.approx(3, abs=0.6)
    assert {"change_point", "peak_hour_shift"} <= set(res.supported)
    assert "does not show what caused the change" in res.conclusion
    assert "associat" in res.conclusion and "caused by" not in res.conclusion.lower()


def test_factors_without_support_are_left_out_of_the_conclusion_but_kept_in_the_audit_list():
    df = two_regimes(peak_before=18, peak_after=18)
    res = investigate(df, "kw", "2024-01-11", "2024-01-20 23:00")
    by = {f.name: f for f in res.factors}
    assert by["peak_hour_shift"].verdict == "no evidence"
    assert by["peak_hour_shift"].statement not in res.conclusion
    assert "peak_hour_shift" not in res.supported + res.weak
    assert {f.verdict for f in res.factors} <= {"supported", "weak", "no evidence"}


def test_no_factor_gives_an_honest_none_found_conclusion():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=24 * 42, freq="h")
    df = pd.DataFrame({"kw": 10 + rng.normal(0, 1, len(idx))}, index=idx)
    # two whole fortnights, so both windows hold the same weekdays
    res = investigate(df, "kw", "2024-01-22", "2024-02-04 23:00")
    assert not res.supported and "None of the checked factors shows evidence" in res.conclusion
    assert abs(res.change.effect_size_d) < 0.5


def test_other_columns_are_checked_for_association_in_the_period_and_the_baseline():
    df = two_regimes(
        extra={
            "related": lambda kw, rng: kw / 2 + rng.normal(0, 0.5, len(kw)),
            "unrelated": lambda kw, rng: rng.normal(0, 1, len(kw)),
        }
    )
    res = investigate(
        df, "kw", "2024-01-11", "2024-01-20 23:00", other_columns=["related", "unrelated"]
    )
    by = {f.name: f for f in res.factors}
    assert by["column:related"].verdict == "supported" and by["column:related"].effect_size > 0.9
    nums = by["column:related"].numbers
    assert nums["correlation_baseline"] is not None and nums["correlation_change"] is not None
    assert by["column:unrelated"].verdict == "no evidence"
    assert "column:unrelated" not in res.supported + res.weak
    assert "associated with column 'related'" in res.conclusion


def test_weather_association_uses_the_period_and_the_baseline():
    df = two_regimes(days=20, bump=0.0)
    t = np.arange(len(df))
    temp = 10 + 8 * np.sin(2 * np.pi * t / 24)
    df["kw"] = 50 - 1.5 * temp + np.random.default_rng(2).normal(0, 0.5, len(df))
    weather = pd.DataFrame({"temperature_2m": temp}, index=df.index)
    res = investigate(df, "kw", "2024-01-11", "2024-01-20 23:00", weather_df=weather)
    w = next(f for f in res.factors if f.name == "weather:temperature_2m")
    assert w.verdict == "supported" and w.effect_size > 0.9 and w.numbers["correlation_period"] < 0
    assert "temperature_2m" in res.evidence.columns


def test_same_weekdays_baseline_only_uses_matching_weekdays():
    df = two_regimes(days=40, shift_day=35)
    res = investigate(
        df, "kw", "2024-02-05", "2024-02-06 23:00", baseline="same_weekdays"
    )  # Mon-Tue
    assert res.baseline.kind == "same_weekdays" and res.baseline.n > 0
    start = pd.Timestamp(res.baseline.start)
    assert start >= pd.Timestamp("2024-02-05") - pd.Timedelta(days=28)
    # only Mondays and Tuesdays: 4 weeks x 2 days x 24 hours
    assert res.baseline.n == 4 * 2 * 24


def test_investigation_validation_and_evidence():
    df = two_regimes()
    with pytest.raises(AnalysisError, match="start before it ends"):
        investigate(df, "kw", "2024-01-11", "2024-01-11")
    with pytest.raises(AnalysisError, match="Not enough data"):
        investigate(df, "kw", "2024-01-01", "2024-01-02")  # nothing before it
    with pytest.raises(AnalysisError, match="baseline must be"):
        investigate(df, "kw", "2024-01-11", "2024-01-20", baseline="last_year")  # type: ignore[arg-type]
    res = investigate(df, "kw", "2024-01-11", "2024-01-20 23:00")
    ev = res.evidence
    assert ev.columns == ["kw"] and ev.method == "investigation_fixed_checks"
    assert ev.n_observations == 240 and ev.start == "2024-01-11 00:00:00"
    assert ev.parameters["baseline"] == "previous" and res.log_entry.read_only
    assert json.loads(res.model_dump_json())["change"]["direction"] == "higher"


# ------------------------------------------------------------------ findings


def test_findings_are_ranked_capped_deduplicated_and_never_claim_cause():
    df = two_regimes(days=40, shift_day=25, peak_before=19, peak_after=19, bump=12.0)
    peaks = peak_periods(df, "kw")
    metrics = energy_metrics(df, "kw", consumption=True)
    weekday = load_profile(df, "kw", "weekday")
    anomalies = stl_anomalies(
        df.assign(kw=df["kw"].where(df.index != df.index[300], df["kw"] + 60)), "kw", 24
    )
    changes = detect_change_points(df, "kw", cost="l2", period=24)
    report = generate_findings(
        peaks=peaks,
        metrics=metrics,
        weekday_profile=weekday,
        anomalies=anomalies,
        change_points=changes,
    )
    scores = [f.score for f in report.findings]
    assert scores == sorted(scores, reverse=True) and len(report.findings) <= 5
    ids = [f.id for f in report.findings]
    assert len(ids) == len(set(ids)) and "peak_hours" in ids
    assert any(i.startswith("change_point:") for i in ids) and any(
        i.startswith("anomalies:") for i in ids
    )
    for f in report.findings:
        assert 0 <= f.score <= 1 and f.chart.name and f.evidence.method
        text = f"{f.title} {f.statement}".lower()
        assert not any(w in text for w in FORBIDDEN_WORDS)
    assert report.n_candidates >= len(report.findings) and report.log_entry.read_only
    assert report.sources == [
        "peak_periods",
        "energy_metrics",
        "load_profile",
        "stl_anomalies",
        "change_points",
    ]
    capped = generate_findings(
        peaks=peaks, metrics=metrics, anomalies=anomalies, change_points=changes, max_findings=2
    )
    assert [f.id for f in capped.findings] == ids[:2]
    json.loads(report.model_dump_json())


def test_no_results_means_no_findings_and_weak_patterns_are_not_reported():
    assert generate_findings().findings == []
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    noise = pd.DataFrame({"kw": rng.normal(10, 1, len(idx))}, index=idx)
    quiet = generate_findings(
        peaks=peak_periods(noise, "kw"), weekday_profile=load_profile(noise, "kw", "weekday")
    )
    assert quiet.findings == []  # peaks spread evenly and no weekend effect


def test_weekend_and_load_factor_findings_use_effect_based_scores():
    idx = pd.date_range("2024-01-01", periods=24 * 28, freq="h")
    df = pd.DataFrame({"kw": np.where(idx.day_of_week >= 5, 30.0, 20.0)}, index=idx)
    report = generate_findings(
        weekday_profile=load_profile(df, "kw", "weekday"),
        metrics=energy_metrics(df, "kw", consumption=True),
    )
    by = {f.id: f for f in report.findings}
    assert by["weekday_vs_weekend"].numbers["difference_pct"] == pytest.approx(50.0)
    assert by["weekday_vs_weekend"].score == pytest.approx(1.0)
    assert "50% higher" in by["weekday_vs_weekend"].title
    assert by["load_factor"].score == pytest.approx(1 - by["load_factor"].numbers["load_factor"])
