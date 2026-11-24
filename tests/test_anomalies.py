import copy
import json

import numpy as np
import pandas as pd
import pytest

from datanaut.core.anomalies import (
    MAX_LISTED,
    compare_detectors,
    multivariate_anomalies,
    stl_anomalies,
)
from datanaut.core.fingerprint import fingerprint
from datanaut.errors import AnalysisError

NaN = np.nan


def seasonal_frame(n_days=28, seed=0, noise=0.5, name="v"):
    rng = np.random.default_rng(seed)
    t = np.arange(24 * n_days)
    idx = pd.date_range("2024-01-01", periods=len(t), freq="h")
    values = 50 + 10 * np.sin(2 * np.pi * t / 24) + 0.01 * t + rng.normal(0, noise, len(t))
    return pd.DataFrame({name: values}, index=idx)


# ------------------------------------------------------------------ STL anomalies


def test_an_injected_spike_is_flagged_with_the_right_direction_and_score():
    df = seasonal_frame()
    df.iloc[300, 0] += 30
    df.iloc[500, 0] -= 30
    res = stl_anomalies(df, "v", period=24)
    flagged = {p.timestamp: p for p in res.points}
    high = flagged[str(df.index[300])]
    low = flagged[str(df.index[500])]
    assert high.direction == "high" and low.direction == "low"
    assert high.residual == pytest.approx(30, abs=3) and low.residual == pytest.approx(-30, abs=3)
    assert high.score > 10 and low.score > 10
    assert res.points[0].score >= res.points[-1].score  # strongest first
    assert 2 <= res.n_anomalies <= 8  # a few ordinary noise points may exceed 3.5 robust sigma
    assert {str(df.index[300].date()), str(df.index[500].date())} <= {d.date for d in res.days}


def test_clean_seasonal_data_has_few_flags():
    res = stl_anomalies(seasonal_frame(n_days=60, seed=2), "v", period=24)
    assert res.n_anomalies <= 0.01 * 24 * 60  # under 1% at the default 3.5 threshold


def test_threshold_controls_sensitivity():
    df = seasonal_frame()
    df.iloc[300, 0] += 4  # a modest bump
    assert (
        stl_anomalies(df, "v", 24, threshold=2.0).n_anomalies
        > stl_anomalies(df, "v", 24, threshold=6.0).n_anomalies
    )


def test_missing_values_are_not_flagged_and_are_counted():
    df = seasonal_frame()
    df.iloc[100:104, 0] = NaN
    df.iloc[300, 0] += 30
    res = stl_anomalies(df, "v", 24)
    assert res.excluded_missing == 4 and str(df.index[300]) in {p.timestamp for p in res.points}
    assert not ({str(t) for t in df.index[100:104]} & {p.timestamp for p in res.points})
    assert res.evidence.n_observations == len(df) - 4


def test_a_gap_in_the_timestamps_keeps_the_seasonal_phase_correct():
    df = seasonal_frame()
    df.iloc[400, 0] += 30
    df = df.drop(df.index[200:205])  # five hours of rows are simply absent
    res = stl_anomalies(df, "v", 24)
    assert str(seasonal_frame().index[400]) in {p.timestamp for p in res.points}
    assert res.n_anomalies <= 8


def test_too_short_or_too_sparse_series_give_clear_errors():
    short = seasonal_frame(n_days=2)
    with pytest.raises(AnalysisError, match="too few for a seasonal fit"):
        stl_anomalies(short, "v", 24)
    sparse = seasonal_frame()
    sparse.iloc[: int(len(sparse) * 0.6), 0] = NaN
    with pytest.raises(AnalysisError, match="more than half missing"):
        stl_anomalies(sparse, "v", 24)
    with pytest.raises(AnalysisError, match="at least 2"):
        stl_anomalies(seasonal_frame(), "v", 1)
    with pytest.raises(AnalysisError, match="positive"):
        stl_anomalies(seasonal_frame(), "v", 24, threshold=0)
    with pytest.raises(AnalysisError, match="time-indexed"):
        stl_anomalies(pd.DataFrame({"v": range(100)}), "v", 24)


def test_listing_is_capped_but_counts_are_complete():
    df = seasonal_frame(n_days=200)
    res = stl_anomalies(df, "v", 24, threshold=0.2)  # absurdly sensitive: thousands flagged
    assert res.n_anomalies > MAX_LISTED and len(res.points) == MAX_LISTED and res.truncated
    assert len(res.flagged_timestamps()) == res.n_anomalies


# ------------------------------------------------------------------ multivariate


def correlated_frame(n=1500, seed=0, rho=0.95):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"x": x, "y": y}, index=idx)


def test_a_jointly_unusual_point_is_flagged_only_by_the_multivariate_model():
    df = correlated_frame()
    spot = 700
    df.iloc[spot, 0] = 0.8  # each within 2 sd on its own, but far off the x ~ y ridge together
    df.iloc[spot, 1] = -2.0
    uni = {c: stl_anomalies(df, c, period=24) for c in ("x", "y")}
    multi = multivariate_anomalies(df, ["x", "y"], contamination=0.01, random_state=0)
    stamp = str(df.index[spot])
    assert stamp not in {p.timestamp for r in uni.values() for p in r.points}
    flagged = {p.timestamp: p for p in multi.points}
    assert stamp in flagged
    top = flagged[stamp].contributions
    assert [c.column for c in top] == ["y", "x"]  # y stands out most
    assert abs(top[0].robust_z) > abs(top[1].robust_z) and top[0].share > 0.5
    assert top[0].robust_z < 0  # below usual
    cmp = compare_detectors(uni, multi)
    only = {u.timestamp: u for u in cmp.only_multivariate}
    assert stamp in only and only[stamp].top_columns[0] == "y"
    assert cmp.n_only_multivariate == len(cmp.only_multivariate)


def test_a_big_spike_in_both_columns_is_flagged_by_both_detectors():
    df = correlated_frame()
    df.iloc[900, :] = 9.0
    uni = {c: stl_anomalies(df, c, period=24) for c in ("x", "y")}
    multi = multivariate_anomalies(df, ["x", "y"])
    cmp = compare_detectors(uni, multi)
    assert str(df.index[900]) in cmp.both and cmp.n_both >= 1
    assert cmp.n_both + cmp.n_only_multivariate == multi.n_flagged


def test_contamination_controls_how_many_rows_are_flagged_and_results_are_reproducible():
    df = correlated_frame()
    few = multivariate_anomalies(df, ["x", "y"], contamination=0.01, random_state=7)
    many = multivariate_anomalies(df, ["x", "y"], contamination=0.05, random_state=7)
    assert few.n_flagged == pytest.approx(0.01 * len(df), abs=3)
    assert many.n_flagged == pytest.approx(0.05 * len(df), abs=5) and many.n_flagged > few.n_flagged
    again = multivariate_anomalies(df, ["x", "y"], contamination=0.01, random_state=7)
    assert [p.timestamp for p in again.points] == [p.timestamp for p in few.points]
    assert [p.score for p in again.points] == [p.score for p in few.points]


def test_single_column_and_degenerate_inputs_are_not_applicable_not_errors():
    df = correlated_frame()
    one = multivariate_anomalies(df, ["x"])
    assert not one.applicable and "at least two" in one.reason and one.n_flagged == 0
    twice = multivariate_anomalies(df, ["x", "x"])
    assert not twice.applicable
    tiny = multivariate_anomalies(df.iloc[:10], ["x", "y"])
    assert not tiny.applicable and "at least 30" in tiny.reason
    flat = multivariate_anomalies(df.assign(z=1.0, w=2.0), ["z", "w"])
    assert not flat.applicable and "Fewer than two columns vary" in flat.reason
    assert one.log_entry.read_only and one.evidence.method == "isolation_forest"


def test_rows_with_a_missing_column_are_left_out_and_counted():
    df = correlated_frame(n=400)
    df.iloc[10:20, 0] = NaN
    res = multivariate_anomalies(df, ["x", "y"])
    assert res.rows_excluded == 10 and res.rows_used == 390 and res.evidence.n_observations == 390
    with pytest.raises(AnalysisError, match="greater than 0"):
        multivariate_anomalies(df, ["x", "y"], contamination=0)
    with pytest.raises(AnalysisError, match="not found"):
        multivariate_anomalies(df, ["x", "nope"])


def test_compare_detectors_accepts_plain_timestamps():
    a = pd.to_datetime(["2024-01-01 01:00", "2024-01-01 02:00"])
    b = ["2024-01-01 02:00", "2024-01-01 03:00"]
    cmp = compare_detectors(a, b)
    assert cmp.both == ["2024-01-01 02:00:00"]
    assert [u.timestamp for u in cmp.only_multivariate] == ["2024-01-01 03:00:00"]
    assert cmp.only_univariate == ["2024-01-01 01:00:00"]
    empty = compare_detectors([], [])
    assert empty.n_both == empty.n_only_multivariate == empty.n_only_univariate == 0


# ------------------------------------------------------------------ evidence and purity


def test_results_have_evidence_logs_json_and_never_touch_the_input():
    df = correlated_frame(n=600)
    df["z"] = seasonal_frame(n_days=25).iloc[:600, 0].to_numpy()
    snapshot, before = copy.deepcopy(df), fingerprint(df)
    stl = stl_anomalies(df, "z", 24)
    multi = multivariate_anomalies(df, ["x", "y", "z"])
    cmp = compare_detectors(stl, multi)
    pd.testing.assert_frame_equal(df, snapshot)
    assert fingerprint(df) == before
    for result, columns, method in (
        (stl, ["z"], "stl_residual_mad"),
        (multi, ["x", "y", "z"], "isolation_forest"),
        (cmp, ["z", "x", "y"], "compare_detectors"),
    ):
        assert result.evidence.columns == columns and result.evidence.method == method
        assert result.evidence.n_observations > 0 and result.evidence.start <= result.evidence.end
        assert result.log_entry.read_only
        json.loads(result.model_dump_json())
    assert stl.evidence.parameters == {"period": 24, "threshold": 3.5, "robust": True}
    assert (
        multi.evidence.parameters["contamination"] == 0.01
        and multi.evidence.parameters["random_state"] == 0
    )


def test_without_the_rotation_a_forest_misses_a_pure_combination_outlier():
    df = correlated_frame()
    df.iloc[700, 0] = 0.8
    df.iloc[700, 1] = -2.0
    stamp = str(df.index[700])
    plain = multivariate_anomalies(df, ["x", "y"], decorrelate=False)
    rotated = multivariate_anomalies(df, ["x", "y"], decorrelate=True)
    assert stamp not in {p.timestamp for p in plain.points}  # one axis at a time cannot see it
    assert stamp in {p.timestamp for p in rotated.points}
    assert plain.evidence.parameters["decorrelate"] is False
