import copy
import json

import numpy as np
import pandas as pd
import pytest

from datanaut.core.fingerprint import fingerprint
from datanaut.core.patterns import energy_metrics, load_profile, peak_periods
from datanaut.errors import AnalysisError

NaN = np.nan


def week(values_by_hour=None, days=14, name="v"):
    """Hourly frame; `values_by_hour(hour, weekday)` gives each reading."""
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h")  # a Monday
    f = values_by_hour or (lambda h, d: 10.0 + h)
    return pd.DataFrame({name: [f(t.hour, t.day_of_week) for t in idx]}, index=idx)


# ------------------------------------------------------------------ load_profile


def test_hour_profile_has_known_means_medians_and_spread():
    df = week(lambda h, d: 100.0 if h == 19 else 10.0)
    p = load_profile(df, "v", "hour")
    assert len(p.buckets) == 24 and [b.hour for b in p.buckets] == list(range(24))
    evening = p.buckets[19]
    assert (evening.n, evening.mean, evening.median, evening.p25, evening.p75) == (
        14,
        100,
        100,
        100,
        100,
    )
    assert p.buckets[3].mean == 10 and p.highest_bucket == "19:00" and p.lowest_bucket != "19:00"
    assert p.used == 24 * 14 and p.excluded == 0 and p.rows_total == 24 * 14


def test_spread_reflects_the_distribution_of_each_bucket():
    # at 08:00 the readings over 14 days are 0..13 -> quartiles of 0..13
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="h")
    values = [float(t.day_of_year - 1) if t.hour == 8 else 0.0 for t in idx]
    b = load_profile(pd.DataFrame({"v": values}, index=idx), "v", "hour").buckets[8]
    assert (b.mean, b.median) == (6.5, 6.5)
    assert (b.p25, b.p75) == (np.percentile(range(14), 25), np.percentile(range(14), 75))


def test_weekday_profile_and_names():
    df = week(lambda h, d: 5.0 if d >= 5 else 20.0)
    p = load_profile(df, "v", "weekday")
    assert [b.label for b in p.buckets] == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    assert [b.mean for b in p.buckets] == [20, 20, 20, 20, 20, 5, 5]
    assert all(b.n == 24 * 2 for b in p.buckets)


def test_hour_by_weekday_has_168_buckets_and_counts():
    df = week(lambda h, d: float(d * 100 + h))
    p = load_profile(df, "v", "hour_x_weekday")
    assert len(p.buckets) == 168
    tuesday_nine = next(b for b in p.buckets if b.label == "Tuesday 09:00")
    assert (tuesday_nine.n, tuesday_nine.mean) == (2, 109.0)
    assert p.highest_bucket == "Sunday 23:00"


def test_missing_and_infinite_values_are_excluded_and_counted_and_empty_buckets_are_listed():
    df = week(days=3)
    df.iloc[[0, 1, 2], 0] = [NaN, np.inf, -np.inf]
    df = df[df.index.hour != 5]  # no readings at all at 05:00
    p = load_profile(df, "v", "hour")
    assert p.excluded == 3 and p.used == len(df) - 3
    assert p.buckets[5].n == 0 and p.buckets[5].mean is None
    assert p.buckets[0].n == 2  # three days, one of them excluded
    assert p.evidence.n_observations == p.used


def test_profile_errors_and_unknown_grouping():
    with pytest.raises(AnalysisError, match="time-indexed"):
        load_profile(pd.DataFrame({"v": [1.0]}), "v")
    with pytest.raises(AnalysisError, match="not found"):
        load_profile(week(), "zz")
    with pytest.raises(AnalysisError, match="not numeric"):
        load_profile(week().assign(t="a"), "t")
    with pytest.raises(AnalysisError, match="Unknown profile grouping"):
        load_profile(week(), "v", "month")  # type: ignore[arg-type]


# ------------------------------------------------------------------ energy_metrics


def test_energy_metrics_known_values():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    df = pd.DataFrame({"kw": [10.0, 20.0, 40.0, 20.0, 10.0]}, index=idx)
    m = energy_metrics(df, "kw", consumption=True)
    assert m.applicable and m.peak == 40 and m.peak_time == "2024-01-01 02:00:00"
    assert m.mean == 20 and m.load_factor == 0.5
    assert m.baseload == pytest.approx(np.percentile([10, 20, 40, 20, 10], 5))
    assert m.baseload_percentile == 5.0 and m.used == 5 and m.excluded == 0


def test_energy_metrics_percentile_parameter_and_excluded_values():
    df = pd.DataFrame(
        {"kw": [NaN, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=pd.date_range("2024", periods=6, freq="h")
    )
    m = energy_metrics(df, "kw", consumption=True, baseload_percentile=20)
    assert m.baseload == pytest.approx(np.percentile([1, 2, 3, 4, 5], 20))
    assert (m.used, m.excluded) == (5, 1) and m.load_factor == pytest.approx(3 / 5)


def test_energy_metrics_only_runs_for_consumption_columns():
    m = energy_metrics(week(), "v")
    assert not m.applicable and "not marked as a consumption column" in m.reason
    assert m.peak is None and m.load_factor is None and m.log_entry.read_only
    assert m.log_entry.counts == {"applicable": 0.0}


def test_energy_metrics_flags_negative_readings_and_zero_peak():
    idx = pd.date_range("2024", periods=4, freq="h")
    neg = energy_metrics(
        pd.DataFrame({"kw": [1.0, -2.0, 3.0, 2.0]}, index=idx), "kw", consumption=True
    )
    assert neg.negative_values == 1 and "negative" in neg.notes[0]
    zero = energy_metrics(pd.DataFrame({"kw": [0.0] * 4}, index=idx), "kw", consumption=True)
    assert zero.load_factor is None and "undefined" in zero.notes[-1]
    with pytest.raises(AnalysisError, match="no usable values"):
        energy_metrics(pd.DataFrame({"kw": [NaN] * 3}, index=idx[:3]), "kw", consumption=True)
    with pytest.raises(AnalysisError, match="between 0 and 50"):
        energy_metrics(week(), "v", consumption=True, baseload_percentile=60)


# ------------------------------------------------------------------ peak_periods


def test_peak_periods_find_the_hours_and_weekdays_where_peaks_concentrate():
    # evening peaks 18:00-20:59 on Tuesday, Thursday and Friday only (5.4% of all readings)
    df = week(lambda h, d: 100.0 if (h in (18, 19, 20) and d in (1, 3, 4)) else 10.0, days=28)
    p = peak_periods(df, "v", 95)
    assert p.threshold == 100 and p.n_peak_observations == 3 * 3 * 4  # 3 hours x 3 days x 4 weeks
    assert p.top_hours == ["18:00", "19:00", "20:00"] and p.top_hours_share == 1.0
    assert p.top_weekdays == ["Tuesday", "Thursday"] and p.top_weekdays_share == pytest.approx(
        2 / 3
    )
    assert sum(e.share for e in p.hours) == pytest.approx(1.0) == sum(e.share for e in p.weekdays)
    assert next(e for e in p.hours if e.label == "03:00").count == 0


def test_peak_periods_spread_out_peaks_have_low_concentration():
    rng = np.random.default_rng(0)
    df = week(lambda h, d: float(rng.normal()), days=120)
    p = peak_periods(df, "v", 95)
    assert p.top_hours_share < 0.35  # chance level for three hours is 3/24 = 12.5%
    assert p.n_peak_observations >= int(0.05 * len(df))


def test_peak_periods_exclude_missing_and_validate():
    df = week(lambda h, d: float(h), days=7)
    df.iloc[:10, 0] = NaN
    p = peak_periods(df, "v", 90)
    assert p.excluded == 10 and p.evidence.n_observations == len(df) - 10
    with pytest.raises(AnalysisError, match="between 0 and 100"):
        peak_periods(df, "v", 100)
    with pytest.raises(AnalysisError, match="no usable values"):
        peak_periods(df.assign(v=NaN), "v")


# ------------------------------------------------------------------ evidence, logs, purity


@pytest.mark.parametrize(
    "call",
    [
        lambda df: load_profile(df, "v", "hour"),
        lambda df: load_profile(df, "v", "weekday"),
        lambda df: energy_metrics(df, "v", consumption=True),
        lambda df: peak_periods(df, "v"),
    ],
)
def test_every_result_has_evidence_a_readonly_log_entry_json_and_leaves_the_input_alone(call):
    df = week()
    snapshot, before = copy.deepcopy(df), fingerprint(df)
    result = call(df)
    pd.testing.assert_frame_equal(df, snapshot)
    assert fingerprint(df) == before
    ev = result.evidence
    assert ev.columns == ["v"] and ev.n_observations == len(df) and ev.method
    assert ev.start == str(df.index.min()) and ev.end == str(df.index.max())
    assert result.log_entry.read_only and result.log_entry.operation
    data = json.loads(result.model_dump_json())
    assert data["evidence"]["method"] == ev.method
    assert "caused" not in result.log_entry.summary.lower()
