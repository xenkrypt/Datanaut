import copy
import json

import numpy as np
import pandas as pd
import pytest

from datanaut.core.changepoints import default_penalty, detect_change_points
from datanaut.core.fingerprint import fingerprint
from datanaut.errors import AnalysisError


def frame(values, name="v"):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.DataFrame({name: np.asarray(values, dtype=float)}, index=idx)


def near(result, position, tolerance, df):
    target = df.index[position]
    return [
        b
        for b in result.breaks
        if abs(pd.Timestamp(b.timestamp) - target) <= pd.Timedelta(hours=tolerance)
    ]


def test_a_mean_shift_is_found_near_its_true_place_with_the_right_direction_and_size():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    x[200:] += 2.0
    df = frame(x)
    res = detect_change_points(df, "v")
    hits = near(res, 200, 12, df)  # tolerance: 12 observations either side of the true break
    assert len(hits) == 1 and res.n_breaks == 1
    b = hits[0]
    assert b.direction == "increase" and b.kind in ("mean", "both")
    assert b.mean_change == pytest.approx(2.0, abs=0.4) and b.mean_change_sd == pytest.approx(
        2.0, abs=0.5
    )
    assert b.mean_after > b.mean_before and b.p_welch < 0.001 and b.p_values_indicative
    assert [s.n for s in res.segments] == pytest.approx([200, 200], abs=14)
    assert res.segments[0].mean == pytest.approx(0, abs=0.3) and res.segments[
        1
    ].mean == pytest.approx(2, abs=0.3)
    assert "too optimistic" in res.p_value_note


def test_a_drop_in_level_is_reported_as_a_decrease():
    rng = np.random.default_rng(2)
    x = rng.normal(size=400)
    x[250:] -= 2.5
    df = frame(x)
    b = near(detect_change_points(df, "v", cost="l2"), 250, 12, df)[0]
    assert b.direction == "decrease" and b.mean_change < -2


def test_a_variance_change_is_detected():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    x[200:] *= 3.0
    df = frame(x)
    res = detect_change_points(df, "v")
    hits = near(res, 200, 20, df)
    assert hits and hits[0].kind in ("variance", "both")
    assert 4 < hits[0].variance_ratio < 20 and hits[0].p_levene < 0.001
    assert hits[0].direction == "none" or hits[0].kind == "both"


def test_white_noise_produces_no_change_points():
    # threshold: none in each of three seeds at the default (medium) sensitivity
    for seed in range(3):
        res = detect_change_points(frame(np.random.default_rng(10 + seed).normal(size=400)), "v")
        assert res.n_breaks == 0 and len(res.segments) == 1 and res.segments[0].n == 400


def test_higher_sensitivity_finds_at_least_as_many_breaks_and_a_given_penalty_wins():
    rng = np.random.default_rng(4)
    x = rng.normal(size=1500)
    x[500:] += 1.0
    x[1000:] -= 0.8
    df = frame(x)
    low = detect_change_points(df, "v", cost="l2", sensitivity="low")
    high = detect_change_points(df, "v", cost="l2", sensitivity="high")
    assert high.n_breaks >= low.n_breaks and high.penalty < low.penalty
    given = detect_change_points(df, "v", cost="l2", penalty=1e9)
    assert (
        given.n_breaks == 0 and given.penalty == 1e9 and given.evidence.parameters["penalty_given"]
    )
    assert default_penalty(1000, "l2", "high") < default_penalty(1000, "l2", "low")
    assert default_penalty(1000, "l2", "medium") == pytest.approx(3 * np.log(1000))


def test_a_daily_cycle_is_not_mistaken_for_change_but_a_real_shift_still_is():
    t = np.arange(24 * 30)
    rng = np.random.default_rng(5)
    x = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, len(t))
    x[24 * 15 :] += 5
    df = frame(x)
    plain = detect_change_points(df, "v", cost="l2", period=24)
    assert plain.seasonally_adjusted and "repeating pattern" in plain.notes[0]
    assert plain.n_breaks == 1 and near(plain, 24 * 15, 24, df)
    assert plain.breaks[0].mean_change == pytest.approx(5, abs=1.0)


def test_long_series_are_averaged_in_blocks_and_the_result_says_so():
    rng = np.random.default_rng(6)
    x = rng.normal(size=3000)
    x[1500:] += 2
    df = frame(x)
    res = detect_change_points(df, "v", max_points=500)
    assert res.downsampled and res.downsample_factor == 6
    assert any("averaged in blocks of 6" in n for n in res.notes)
    assert near(res, 1500, 24, df) and sum(s.n for s in res.segments) == 3000


def test_short_constant_and_gappy_series_are_handled_not_errors():
    short = detect_change_points(frame(np.arange(15.0)), "v")
    assert short.n_breaks == 0 and "at least" in short.notes[0] and short.segments[0].n == 15
    constant = detect_change_points(frame([5.0] * 100), "v")
    assert constant.n_breaks == 0 and "constant" in constant.notes[0]
    empty = detect_change_points(frame([np.nan] * 50), "v")
    assert empty.n_breaks == 0 and empty.segments == [] and empty.excluded_missing == 50
    x = np.random.default_rng(7).normal(size=300)
    x[100:110] = np.nan
    res = detect_change_points(frame(x), "v", cost="l2")
    assert res.excluded_missing == 10 and res.evidence.n_observations == 290


def test_bad_parameters_are_rejected():
    df = frame(np.random.default_rng(8).normal(size=100))
    for kwargs, message in (
        ({"sensitivity": "extreme"}, "sensitivity"),
        ({"cost": "l1"}, "cost must be"),
        ({"penalty": 0}, "penalty must be positive"),
        ({"min_size": 1}, "min_size"),
        ({"period": 1}, "period"),
    ):
        with pytest.raises(AnalysisError, match=message):
            detect_change_points(df, "v", **kwargs)
    with pytest.raises(AnalysisError, match="time-indexed"):
        detect_change_points(pd.DataFrame({"v": range(50)}), "v")


def test_evidence_log_json_and_the_input_is_untouched():
    rng = np.random.default_rng(9)
    x = rng.normal(size=300)
    x[150:] += 3
    df = frame(x)
    snapshot, before = copy.deepcopy(df), fingerprint(df)
    res = detect_change_points(df, "v", cost="l2", min_size=20)
    pd.testing.assert_frame_equal(df, snapshot)
    assert fingerprint(df) == before
    ev = res.evidence
    assert ev.columns == ["v"] and ev.method == "pelt_l2" and ev.n_observations == 300
    assert ev.start == str(df.index[0]) and ev.end == str(df.index[-1])
    assert ev.parameters["min_size"] == 20 and ev.parameters["cost"] == "l2"
    assert res.log_entry.read_only and res.log_entry.operation == "change_points"
    assert json.loads(res.model_dump_json())["n_breaks"] == res.n_breaks
