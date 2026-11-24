import json
import time

import numpy as np
import pandas as pd
import pytest

from datanaut.core.characterize import candidate_periods, characterize, frequency_label
from datanaut.core.fingerprint import fingerprint
from datanaut.core.loader import load_file
from datanaut.core.log import PipelineLog
from datanaut.core.timeindex import build_time_index, detect_time_columns
from datanaut.errors import ParseError


def _series(values, freq="h", start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq=freq)
    return pd.DataFrame({"v": np.asarray(values, dtype=float)}, index=idx)


def _seasonal(n=24 * 56, seed=0, trend=0.0, noise=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (
        100
        + trend * t
        + 10 * np.sin(2 * np.pi * t / 24)
        + 3 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, noise, n)
    )


# ------------------------------------------------------------------ the required behaviours


def test_known_period_24_is_reported():
    c = characterize(_series(_seasonal()), "v")
    assert c.frequency_label == "hourly"
    assert c.seasonal_period == 24 and c.seasonal_label == "daily"
    assert c.seasonal_strength > 0.6
    periods = {s.period: s for s in c.seasonal_candidates}
    assert set(periods) == {24, 168}  # only sensible periods for hourly data
    assert periods[168].significant
    assert "strong daily seasonality: use seasonal methods" in c.recommendation


def test_random_walk_is_non_stationary():
    rng = np.random.default_rng(1)
    c = characterize(_series(np.cumsum(rng.normal(size=1500))), "v")
    assert c.stationarity.label == "non-stationary"
    assert c.stationarity.p_value > 0.05
    assert c.seasonal_period is None
    assert "non-stationary" in c.recommendation


def test_white_noise_is_stationary_without_trend_or_season():
    rng = np.random.default_rng(2)
    c = characterize(_series(rng.normal(size=1500)), "v")
    assert c.stationarity.label == "stationary"
    assert c.trend.direction == "none"
    assert c.seasonal_period is None
    assert "no clear seasonality" in c.recommendation


@pytest.mark.parametrize("n", [0, 1, 2, 5, 7, 12, 19])
def test_short_series_do_not_crash(n):
    c = characterize(_series(np.arange(n) % 3), "v")
    assert c.stationarity.label == "insufficient data"
    assert c.seasonal_period is None
    assert c.log_entry.operation == "characterize"
    if n < 8:
        assert c.trend.direction == "insufficient data"
        assert c.recommendation.startswith("not enough usable data")


# ------------------------------------------------------------------ trend and frequency


def test_trend_direction_and_size():
    up = characterize(_series(_seasonal(trend=0.05)), "v")
    down = characterize(_series(_seasonal(trend=-0.05)), "v")
    flat = characterize(_series(_seasonal(trend=0.0, seed=3)), "v")
    assert up.trend.direction == "increasing" and down.trend.direction == "decreasing"
    assert up.trend.slope_per_day == pytest.approx(0.05 * 24, rel=0.1)
    assert up.trend.total_change == pytest.approx(0.05 * 24 * 56, rel=0.1)
    assert up.trend.relative_change_pct > 0 > down.trend.relative_change_pct
    assert up.trend.effect_label == "large" and up.trend.p_value < 0.05
    assert flat.trend.direction == "none"
    assert up.trend.block_size == 24  # whole daily cycles, so the season does not leak in
    assert "increasing trend" in up.recommendation


def test_daily_data_weekly_season():
    rng = np.random.default_rng(4)
    t = np.arange(7 * 30)
    c = characterize(
        _series(50 + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, len(t)), "D"), "v"
    )
    assert c.frequency_label == "daily"
    assert c.seasonal_period == 7 and c.seasonal_label == "weekly"


def test_monthly_data_yearly_season():
    t = np.arange(12 * 8)
    rng = np.random.default_rng(5)
    c = characterize(
        _series(20 + 6 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, len(t)), "MS"), "v"
    )
    assert c.frequency_label == "monthly" and c.seasonal_period == 12
    assert c.seasonal_label == "yearly"


def test_frequency_labels_and_candidate_periods():
    assert [frequency_label(s) for s in (1, 60, 900, 3600, 86400, 604800)] == [
        "secondly",
        "minutely",
        "15-minutely",
        "hourly",
        "daily",
        "weekly",
    ]
    assert frequency_label(30 * 86400) == "monthly" and frequency_label(None) == "unknown"
    assert frequency_label(7 * 3600) == "every 7h"
    assert candidate_periods(3600, 24 * 20) == [24, 168]
    assert candidate_periods(3600, 24 * 800) == [24, 168, 8760]
    assert candidate_periods(900, 96 * 20) == [96, 672]
    assert candidate_periods(86400, 100) == [7]
    assert candidate_periods(86400 * 7, 200) == [52]
    assert candidate_periods(86400 * 2 * 365, 100) == []


# ------------------------------------------------------------------ messy inputs


def test_input_is_not_modified_and_gaps_are_interpolated_on_a_copy():
    df = _series(_seasonal())
    df = df.drop(df.index[200:206])
    df.loc[df.index[10], "v"] = np.nan
    before = fingerprint(df)
    c = characterize(df, "v")
    assert fingerprint(df) == before
    assert c.seasonal_period == 24
    assert c.analysis_points == 24 * 56
    assert any("6 gap timestamp" in n for n in c.notes)
    assert c.missing_share == pytest.approx(1 / len(df), abs=1e-6)
    assert "gaps" in c.recommendation


def test_duplicate_timestamps_and_jitter_are_handled():
    df = _series(_seasonal(n=24 * 30))
    rng = np.random.default_rng(6)
    jittered = df.copy()
    jittered.index = jittered.index + pd.to_timedelta(rng.integers(-60, 60, len(df)), unit="s")
    dup = pd.concat([jittered, jittered.iloc[:5]]).sort_index()
    c = characterize(dup, "v")
    assert c.seasonal_period == 24
    assert any("duplicate timestamp" in n for n in c.notes)


def test_mostly_missing_series_is_not_analyzed():
    values = _seasonal(n=24 * 30)
    values[: int(len(values) * 0.8)] = np.nan
    values[0], values[-1] = 1.0, 2.0
    c = characterize(_series(values), "v")
    assert c.seasonal_period is None and c.analysis_points > 0 or c.analysis_points == 0
    assert c.missing_share > 0.7


def test_constant_series():
    c = characterize(_series([5.0] * 100), "v")
    assert c.stationarity.label == "insufficient data" and c.seasonal_period is None
    assert any("constant" in n for n in c.notes)


def test_long_series_is_downsampled_and_says_so():
    n = 24 * 365 * 3
    start = time.perf_counter()
    c = characterize(_series(_seasonal(n=n, trend=0.001), start="2020-01-01"), "v")
    assert time.perf_counter() - start < 30
    assert c.downsampled
    assert c.seasonal_period == 24
    assert c.trend.points_used <= 2000 and c.trend.block_size % 24 == 0
    assert c.stationarity.points_used == 5000
    assert any("block mean" in n for n in c.notes)
    assert any("most recent 5000" in n for n in c.notes)
    assert {s.period for s in c.seasonal_candidates} == {24, 168, 8760}


def test_short_hourly_series_only_offers_periods_it_can_support():
    c = characterize(_series(_seasonal(n=24 * 10)), "v")
    assert [s.period for s in c.seasonal_candidates] == [24]


# ------------------------------------------------------------------ errors, log, variants


def test_bad_inputs():
    with pytest.raises(ParseError, match="time index"):
        characterize(pd.DataFrame({"v": [1.0, 2.0]}), "v")
    with pytest.raises(ParseError, match="not found"):
        characterize(_series([1, 2, 3]), "nope")
    df = _series([1, 2, 3])
    df["t"] = ["a", "b", "c"]
    df["b"] = [True, False, True]
    for col in ("t", "b"):
        with pytest.raises(ParseError, match="not numeric"):
            characterize(df, col)


def test_log_entry_and_json():
    c = characterize(_series(_seasonal()), "v")
    e = c.log_entry
    assert e.read_only and e.operation == "characterize"
    assert e.counts["seasonal_period"] == 24 and "adf_p_value" in e.counts
    assert json.loads(c.model_dump_json())["seasonal_period"] == 24
    log = PipelineLog()
    log.extend([e])
    assert PipelineLog.from_json(log.to_json()).entries == log.entries


def test_seasonal_series_without_trend_is_stationary_after_adjustment():
    c = characterize(_series(_seasonal(seed=11)), "v")
    assert c.stationarity.label == "stationary"
    assert any("Stationarity was tested after removing" in n for n in c.notes)


def test_every_variant_characterizes_to_daily_season(variant):
    df = load_file(variant.data, filename=variant.filename).df
    indexed = build_time_index(df, detect_time_columns(df)[0].columns).df
    c = characterize(indexed, "power_kw")
    assert c.frequency_label == "hourly"
    assert c.seasonal_period == 24
    assert c.log_entry.read_only


def test_infinite_values_are_reported_and_treated_as_missing():
    values = _seasonal(n=24 * 20)
    values[5] = np.inf
    values[9] = -np.inf
    c = characterize(_series(values), "v")
    assert any("2 infinite value" in n for n in c.notes)
    assert c.seasonal_period == 24
