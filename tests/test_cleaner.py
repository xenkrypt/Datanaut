import copy

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from datanaut.core.characterize import characterize
from datanaut.core.cleaner import (
    FLAG_SUFFIX,
    OpSpec,
    OutliersParams,
    apply_op,
    available_operations,
    detect_outliers,
    preview,
    propose_fixes,
    quality_warnings,
)
from datanaut.core.fingerprint import fingerprint
from datanaut.core.profiler import profile
from datanaut.errors import CleaningError

NaN = np.nan


def hourly(values, name="v", **cols):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.DataFrame({name: np.asarray(values, dtype=float), **cols}, index=idx)


def run(df, operation, **params):
    return apply_op(df, OpSpec(operation=operation, params=params))


def filled_values(res, col="v"):
    return res.df[col].tolist()


# ------------------------------------------------------------------ drop_duplicates


def _dup_frame():
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00"] * 3 + ["2024-01-01 01:00"] * 2 + ["2024-01-01 02:00"]
    )
    return pd.DataFrame({"v": [1.0, 2.0, 3.0, 5.0, 5.0, 9.0], "t": list("abcdde")}, index=idx)


def test_exact_duplicates_removed_and_counted():
    res = run(_dup_frame(), "drop_duplicates")
    assert len(res.df) == 5  # the (01:00, 5.0, 'e') pair collapses; 00:00 rows differ
    c = res.log_entry.counts
    assert c["exact_duplicates_removed"] == 1
    assert c["conflicting_timestamp_groups"] == 1 and c["conflicting_rows_left"] == 2
    assert c["conflicting_rows_resolved"] == 0  # reported, not silently changed
    assert "left as they are" in res.log_entry.summary


@pytest.mark.parametrize(
    "how, expected",
    [("first", 1.0), ("last", 3.0), ("mean", 2.0), ("median", 2.0), ("sum", 6.0)],
)
def test_conflicting_timestamps_resolution(how, expected):
    res = run(_dup_frame(), "drop_duplicates", conflicts=how)
    assert res.df.index.is_unique and len(res.df) == 3
    assert res.df["v"].iloc[0] == expected
    assert res.df["t"].iloc[0] == ("c" if how == "last" else "a")  # text follows first/last
    assert res.log_entry.counts["conflicting_rows_resolved"] == 2
    assert res.log_entry.counts["conflicting_rows_left"] == 0
    assert res.log_entry.counts["rows_removed"] == 3


def test_keep_last_for_exact_duplicates_and_sum_of_all_nan():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"v": [NaN, NaN, 1.0]}, index=idx)
    res = run(df, "drop_duplicates", keep="last", conflicts="sum")
    assert res.log_entry.counts["exact_duplicates_removed"] == 1
    assert np.isnan(res.df["v"].iloc[0])  # a sum of nothing is missing, not zero


def test_duplicates_on_a_plain_index():
    df = pd.DataFrame({"v": [1.0, 1.0, 2.0]}, index=["a", "a", "b"])
    assert len(run(df, "drop_duplicates").df) == 2


# ------------------------------------------------------------------ handle_missing


def test_interpolate_respects_max_gap_and_counts_everything():
    df = hourly([1, NaN, 3, NaN, NaN, NaN, 7, NaN])
    res = run(df, "handle_missing", column="v", method="interpolate_time", max_gap=2)
    out = res.df["v"].tolist()
    assert out[:3] == [1.0, 2.0, 3.0] and all(np.isnan(out[3:6]))  # long gap left whole
    assert out[6] == 7.0 and np.isnan(out[7])
    c = res.log_entry.counts
    assert (c["missing_before"], c["filled"], c["gaps_filled"]) == (5, 1, 1)
    assert (c["left_too_long"], c["gaps_too_long"], c["left_at_edges"]) == (3, 1, 1)
    assert c["missing_after"] == 4


def test_interpolation_is_time_aware_not_positional():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 10:00"])
    df = pd.DataFrame({"v": [0.0, NaN, 10.0]}, index=idx)
    res = run(df, "handle_missing", column="v", method="interpolate_time", max_gap=None)
    assert res.df["v"].iloc[1] == pytest.approx(1.0)  # 1h of 10h, not the midpoint 5.0


def test_interpolate_unlimited_gap_but_never_edges():
    df = hourly([NaN, 1, NaN, NaN, NaN, NaN, 5, NaN])
    res = run(df, "handle_missing", column="v", method="interpolate_time", max_gap=None)
    assert res.df["v"].iloc[2:6].tolist() == pytest.approx([1.8, 2.6, 3.4, 4.2])
    assert np.isnan(res.df["v"].iloc[0]) and np.isnan(res.df["v"].iloc[7])
    assert res.log_entry.counts["left_at_edges"] == 2


def test_ffill_with_limit():
    res = run(hourly([1, NaN, NaN, NaN]), "handle_missing", column="v", method="ffill", limit=2)
    assert res.df["v"].iloc[:3].tolist() == [1, 1, 1] and np.isnan(res.df["v"].iloc[3])
    assert res.log_entry.counts["filled"] == 2 and res.log_entry.counts["left_missing"] == 1


def test_seasonal_fill_uses_neighbouring_seasons_and_counts_the_rest():
    values = [1, 2, 3, 4] * 3
    values[1] = NaN
    values[5] = NaN
    df = hourly(values)
    one = run(df, "handle_missing", column="v", method="seasonal_fill", period=4, max_seasons=1)
    out = one.df["v"].tolist()
    assert np.isnan(out[1]) and out[5] == 2  # index 1 has no usable neighbour within 1 season
    assert one.log_entry.counts["left_missing"] == 1
    two = run(df, "handle_missing", column="v", method="seasonal_fill", period=4, max_seasons=2)
    assert two.df["v"].tolist() == [1, 2, 3, 4] * 3 and two.log_entry.counts["left_missing"] == 0


def test_seasonal_fill_averages_both_sides_and_works_across_missing_rows():
    df = hourly([10, 0, 20, 5] * 3)  # period 4: positions 2 -> 20 always
    df.iloc[6, 0] = NaN
    df.iloc[2, 0] = 30.0  # previous season value 30, next season 20 -> mean 25
    res = run(df, "handle_missing", column="v", method="seasonal_fill", period=4)
    assert res.df["v"].iloc[6] == 25.0
    holey = df.drop(df.index[3])  # a missing row shifts positions, timestamps still line up
    res = run(holey, "handle_missing", column="v", method="seasonal_fill", period=4)
    assert res.df["v"].loc[df.index[6]] == 25.0


def test_drop_missing_rows():
    res = run(hourly([1, NaN, 3, NaN]), "handle_missing", column="v", method="drop")
    assert res.df["v"].tolist() == [1, 3] and res.log_entry.counts["rows_dropped"] == 2
    assert res.log_entry.rows_before == 4 and res.log_entry.rows_after == 2


def test_handle_missing_errors():
    df = hourly([1, NaN, 3], t=["a", "b", "c"])
    with pytest.raises(CleaningError, match="not found"):
        run(df, "handle_missing", column="zzz", method="drop")
    with pytest.raises(CleaningError, match="not numeric"):
        run(df, "handle_missing", column="t", method="interpolate_time")
    with pytest.raises(CleaningError, match="period"):
        run(df, "handle_missing", column="v", method="seasonal_fill")
    with pytest.raises(CleaningError, match="time index"):
        run(df.reset_index(drop=True), "handle_missing", column="v", method="interpolate_time")
    with pytest.raises(CleaningError, match="chronological"):
        run(df.iloc[::-1], "handle_missing", column="v", method="interpolate_time")


# ------------------------------------------------------------------ outliers


def _spiky():
    i = np.arange(60)
    values = i + 0.1 * np.sin(i * 7.0)
    values[20] += 100
    return hourly(values)


def test_hampel_flags_exactly_the_spike_and_keeps_values():
    df = _spiky()
    res = run(df, "outliers", column="v")
    flag = res.df["v" + FLAG_SUFFIX]
    assert flag.dtype == bool and flag.to_numpy().nonzero()[0].tolist() == [20]
    pd.testing.assert_series_equal(res.df["v"], df["v"])  # flag only: values untouched
    assert res.log_entry.counts["outliers"] == 1 and res.log_entry.counts["evaluated"] == 60


def test_outlier_set_missing_and_clip():
    df = _spiky()
    gone = run(df, "outliers", column="v", action="set_missing")
    assert np.isnan(gone.df["v"].iloc[20]) and gone.df["v"].isna().sum() == 1
    assert gone.log_entry.counts["set_missing"] == 1 and "v__outlier" not in gone.df.columns
    det = detect_outliers(df, OutliersParams(column="v"))
    clipped = run(df, "outliers", column="v", action="clip")
    assert clipped.df["v"].iloc[20] == pytest.approx(det.upper[20])
    assert df["v"].iloc[20] > 100 > clipped.df["v"].iloc[20]
    assert clipped.log_entry.counts["clipped"] == 1


def test_outliers_skip_missing_values_and_say_so():
    df = _spiky()
    df.iloc[5, 0] = NaN
    res = run(df, "outliers", column="v")
    assert not res.df["v__outlier"].iloc[5]
    assert res.log_entry.counts["missing_skipped"] == 1 and res.log_entry.counts["outliers"] == 1


def test_hampel_on_series_shorter_than_window_reports_not_evaluated():
    res = run(hourly([1, 2, 3, 4]), "outliers", column="v", window=11)
    assert res.log_entry.counts["not_evaluated"] == 4 and res.log_entry.counts["outliers"] == 0
    assert "could not be evaluated" in res.log_entry.summary


def test_seasonal_residual_finds_spike_in_cyclic_data():
    rng = np.random.default_rng(0)
    t = np.arange(24 * 20)
    values = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, len(t))
    values[200] += 40
    res = run(hourly(values), "outliers", column="v", method="seasonal_residual", period=24)
    flagged = res.df["v__outlier"].to_numpy().nonzero()[0]
    assert 200 in flagged and len(flagged) <= 8


def test_seasonal_residual_too_short_is_a_clear_error():
    with pytest.raises(CleaningError, match="too few"):
        run(hourly(np.arange(30)), "outliers", column="v", method="seasonal_residual", period=24)
    with pytest.raises(CleaningError, match="period"):
        run(hourly(np.arange(30)), "outliers", column="v", method="seasonal_residual")


def test_outlier_param_validation():
    with pytest.raises(CleaningError, match="odd"):
        run(_spiky(), "outliers", column="v", window=10)
    with pytest.raises(CleaningError, match="not numeric"):
        run(hourly([1, 2, 3], t=["a", "b", "c"]), "outliers", column="t")


def test_flag_column_is_replaced_on_rerun_and_reported():
    first = run(_spiky(), "outliers", column="v")
    again = run(first.df, "outliers", column="v")
    assert again.log_entry.counts["replaced_existing_flag"] == 1
    assert list(again.df.columns).count("v__outlier") == 1


# ------------------------------------------------------------------ resample


def test_resample_sum_mean_last_and_counts():
    df = hourly([1, 2, 3, 4, 5, 6])
    total = run(df, "resample", rule="3h", agg="sum")
    assert total.df["v"].tolist() == [6, 15]
    assert total.log_entry.counts == {"rows_before": 6, "rows_after": 2, "empty_bins": 0}
    assert run(df, "resample", rule="3h", agg="mean").df["v"].tolist() == [2, 5]
    assert run(df, "resample", rule="3h", agg="last").df["v"].tolist() == [3, 6]
    assert run(df, "resample", rule="3h", agg="max").df["v"].tolist() == [3, 6]


def test_resample_empty_bins_stay_missing_not_zero():
    df = hourly([1, 2, 3, 0, 0, 0, 4, 5]).drop(hourly([0] * 8).index[3:6])
    res = run(df, "resample", rule="3h", agg="sum")
    assert res.df["v"].iloc[0] == 6 and res.df["v"].iloc[2] == 9
    assert np.isnan(res.df["v"].iloc[1])  # the empty middle bin is missing, not 0
    assert res.log_entry.counts["empty_bins"] == 1
    wide = run(hourly([1, 2, 3, 4]).iloc[[0, 3]], "resample", rule="1h", agg="sum")
    assert len(wide.df) == 4 and wide.log_entry.counts["empty_bins"] == 2
    assert wide.df["v"].isna().sum() == 2
    assert "2 empty bin" in wide.log_entry.summary


def test_resample_keeps_flags_and_text_columns():
    df = hourly([1, 2, 3, 4], flag=[False, True, False, False], t=list("abcd"))
    res = run(df, "resample", rule="2h", agg="mean")
    assert res.df["flag"].tolist() == [True, False] and res.df["flag"].dtype == bool
    assert res.df["t"].tolist() == ["a", "c"]
    assert list(res.df.columns) == ["v", "flag", "t"]


def test_resample_errors():
    with pytest.raises(CleaningError, match="valid frequency"):
        run(hourly([1, 2]), "resample", rule="banana")
    with pytest.raises(CleaningError, match="time index"):
        run(hourly([1, 2]).reset_index(drop=True), "resample", rule="1h")


# ------------------------------------------------------------------ warnings


def test_quality_warnings_cover_every_kind_and_change_nothing():
    n = 40
    rng = np.random.default_rng(1)
    df = hourly(
        rng.normal(5, 1, n),
        name="power_kw",
        flat=[3.0] * n,
        huge=rng.normal(5e6, 1e5, n),
        steppy=np.where(np.arange(n) == 20, 500.0, rng.normal(0, 1, n)),
    )
    df.iloc[3, 0] = -2.0
    found = {(w.kind, w.column): w for w in quality_warnings(df)}
    assert found[("negative_values", "power_kw")].count == 1
    assert ("constant_column", "flat") in found
    assert ("suspect_unit_scale", "huge") in found
    assert found[("implausible_jumps", "steppy")].count >= 1
    snapshot = df.copy()
    res = run(df, "check_warnings")
    assert res.log_entry.read_only and res.log_entry.counts["warnings"] >= 4
    pd.testing.assert_frame_equal(res.df, snapshot)


def test_negative_check_uses_user_flag_for_unhinted_names():
    df = hourly([1, -1, 2], name="widgets")
    assert quality_warnings(df) == []
    assert quality_warnings(df, consumption_columns=["widgets"])[0].kind == "negative_values"


# ------------------------------------------------------------------ the apply_op contract

_SPECS = [
    OpSpec(operation="drop_duplicates", params={"conflicts": "mean"}),
    OpSpec(operation="handle_missing", params={"column": "v", "method": "interpolate_time"}),
    OpSpec(operation="handle_missing", params={"column": "v", "method": "ffill"}),
    OpSpec(
        operation="handle_missing", params={"column": "v", "method": "seasonal_fill", "period": 4}
    ),
    OpSpec(operation="handle_missing", params={"column": "v", "method": "drop"}),
    OpSpec(operation="outliers", params={"column": "v", "action": "flag"}),
    OpSpec(operation="outliers", params={"column": "v", "action": "set_missing"}),
    OpSpec(operation="outliers", params={"column": "v", "action": "clip"}),
    OpSpec(operation="resample", params={"rule": "2h", "agg": "sum"}),
    OpSpec(operation="check_warnings", params={}),
]


@pytest.mark.parametrize(
    "spec",
    _SPECS,
    ids=lambda s: f"{s.operation}-{s.params.get('method', s.params.get('action', ''))}",
)
def test_input_is_never_modified(spec):
    df = _spiky()
    df.iloc[[4, 5, 30], 0] = NaN
    df = pd.concat([df, df.iloc[[10]]]).sort_index()
    snapshot = copy.deepcopy(df)
    before = fingerprint(df)
    result = apply_op(df, spec)
    assert result.df is not df
    pd.testing.assert_frame_equal(df, snapshot)
    assert fingerprint(df) == before
    result.df.iloc[0, 0] = -12345.0  # mutating the result must not touch the input either
    pd.testing.assert_frame_equal(df, snapshot)
    assert result.log_entry.operation == spec.operation
    assert result.log_entry.rows_before == len(df)


def test_unknown_operation_and_bad_params_are_friendly_errors():
    with pytest.raises(CleaningError, match="Unknown operation 'explode'"):
        apply_op(hourly([1]), OpSpec(operation="explode"))
    with pytest.raises(CleaningError, match="method"):
        run(hourly([1]), "handle_missing", column="v", method="teleport")
    with pytest.raises(CleaningError, match="Extra inputs"):
        run(hourly([1]), "drop_duplicates", frobnicate=True)
    assert set(available_operations()) >= {
        "drop_duplicates",
        "handle_missing",
        "outliers",
        "resample",
        "check_warnings",
    }


def test_specs_round_trip_as_json_and_log_has_resolved_params():
    spec = OpSpec(operation="handle_missing", params={"column": "v", "method": "ffill"})
    assert OpSpec.model_validate_json(spec.model_dump_json()) == spec
    entry = apply_op(hourly([1, NaN]), spec).log_entry
    assert entry.params["limit"] is None and entry.params["max_gap"] == 6  # defaults made explicit


# ------------------------------------------------------------------ preview


def test_preview_reports_before_after_without_modifying():
    df = hourly([1, NaN, 3, 5])
    snapshot = df.copy()
    pv = preview(
        df, OpSpec(operation="handle_missing", params={"column": "v", "method": "interpolate_time"})
    )
    pd.testing.assert_frame_equal(df, snapshot)
    assert pv.before["v"].missing == 1 and pv.after["v"].missing == 0
    assert pv.before["v"].mean == pytest.approx(3.0) and pv.after["v"].mean == pytest.approx(11 / 4)
    assert (pv.after["v"].min, pv.after["v"].max, pv.after["v"].count) == (1.0, 5.0, 4)
    assert pv.rows_before == pv.rows_after == 4
    assert pv.log_entry.operation == "handle_missing" and pv.df["v"].isna().sum() == 0


def test_preview_of_row_changing_ops_summarizes_numeric_columns():
    pv = preview(_dup_frame(), OpSpec(operation="drop_duplicates"))
    assert pv.rows_before == 6 and pv.rows_after == 5 and "v" in pv.before and "t" not in pv.before


# ------------------------------------------------------------------ proposals


def _messy():
    rng = np.random.default_rng(3)
    t = np.arange(24 * 14)
    values = 20 + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, len(t))
    values[50:52] = NaN  # short gap
    values[100:103] = NaN  # short gap
    values[200:230] = NaN  # long outage
    values[150] += 80  # spike
    df = hourly(values, name="power_kw", other=rng.normal(0, 1, len(t)))
    df = pd.concat([df, df.iloc[[10, 11]]]).sort_index(kind="stable")  # 2 exact duplicates
    return df.drop(df.index[300:303])  # a gap in the timestamps


def test_propose_fixes_content_flags_and_alternatives():
    df = _messy()
    props = {p.id: p for p in propose_fixes(df, profile(df))}
    dup = props["drop_duplicates:exact"]
    assert "2 exact duplicate" in dup.description and dup.default_selected
    miss = props["handle_missing:power_kw"]
    assert miss.default_selected and miss.spec.params["method"] == "interpolate_time"
    assert miss.counts == {"missing": 35, "fillable": 5, "longest_gap": 30}
    assert (
        "35 missing value(s) in power_kw" in miss.description and "5 fillable" in miss.description
    )
    assert {a.spec.params["method"] for a in miss.alternatives} >= {
        "ffill",
        "drop",
        "interpolate_time",
    }
    out = props["outliers:power_kw"]
    assert not out.default_selected and out.counts["outliers"] >= 1
    assert {a.spec.params["action"] for a in out.alternatives} == {"set_missing", "clip"}
    assert "resample:regular" not in props  # a timestamp gap is a gap, not jitter
    for p in props.values():
        if p.id not in ("drop_duplicates:exact", "handle_missing:power_kw"):
            assert not p.default_selected
        apply_op(df, p.spec)  # every proposal must be applicable as it stands
        for alt in p.alternatives:
            apply_op(df, alt.spec)


def test_proposals_use_the_series_period_when_characterized():
    df = _messy()
    c = characterize(df, "power_kw")
    props = {p.id: p for p in propose_fixes(df, profile(df), c)}
    methods = [a.spec.params["method"] for a in props["handle_missing:power_kw"].alternatives]
    assert "seasonal_fill" in methods
    assert any(
        a.spec.params.get("method") == "seasonal_residual"
        for a in props["outliers:power_kw"].alternatives
    )


def test_default_selected_proposals_fix_duplicates_and_short_gaps_only():
    df = _messy()
    out = df
    for p in propose_fixes(df, profile(df)):
        if p.default_selected:
            out = apply_op(out, p.spec).df
    assert not out.index.duplicated().any()
    gaps = out["power_kw"].isna().to_numpy()
    assert gaps.sum() == 30 and gaps[200:230].all()  # only the 30-hour outage is left alone
    assert (
        not out["power_kw"].iloc[48:54].isna().any()
        and not out["power_kw"].iloc[98:106].isna().any()
    )
    assert out["power_kw"].max() > 90  # the spike was only suggested, never applied


def test_irregular_timestamps_get_an_unselected_resample_proposal():
    rng = np.random.default_rng(5)
    jitter = pd.to_timedelta(rng.integers(-25, 25, 200), unit="min")
    idx = pd.date_range("2024-01-01", periods=200, freq="h") + jitter
    df = pd.DataFrame({"v": rng.normal(0, 1, 200)}, index=idx.sort_values())
    props = {p.id: p for p in propose_fixes(df, profile(df))}
    p = props["resample:regular"]
    assert not p.default_selected and p.spec.params == {"rule": "1h", "agg": "mean"}
    assert "not 1h" in p.description and {a.spec.params["agg"] for a in p.alternatives} == {
        "sum",
        "median",
        "min",
        "max",
        "last",
    }
    assert len(apply_op(df, p.spec).df) in range(195, 205)


def test_long_gaps_only_gives_unselected_proposal():
    df = hourly([1, 2, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 3, 4] * 3)
    (p,) = [x for x in propose_fixes(df, profile(df)) if x.category == "missing"]
    assert not p.default_selected and "none in gaps of 6" in p.description


def test_clean_data_gets_no_fixes_except_possible_noise_outliers():
    rng = np.random.default_rng(9)
    df = hourly(rng.normal(0, 1, 100), other=rng.normal(5, 1, 100))
    found = propose_fixes(df, profile(df))
    assert {p.category for p in found} <= {"outliers"}  # noise can exceed 3 sigma by chance
    assert not any(p.default_selected for p in found)


# ------------------------------------------------------------------ properties


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=4, max_size=40),
    st.lists(st.booleans(), min_size=4, max_size=40),
    st.lists(st.integers(1, 5), min_size=4, max_size=40),
)
def test_property_interpolation_stays_between_its_neighbours(values, holes, steps):
    n = min(len(values), len(holes), len(steps))
    x = np.array(values[:n], dtype=float)
    mask = np.array(holes[:n])
    mask[0] = mask[-1] = False
    x[mask] = NaN
    idx = pd.DatetimeIndex(
        pd.Timestamp("2024-01-01") + pd.to_timedelta(np.cumsum(steps[:n]), unit="h")
    )
    df = pd.DataFrame({"v": x}, index=idx)
    res = run(df, "handle_missing", column="v", method="interpolate_time", max_gap=None)
    out = res.df["v"].to_numpy()
    assert not np.isnan(out).any()
    observed = np.flatnonzero(~mask)
    for i in np.flatnonzero(mask):
        lo = observed[observed < i].max()
        hi = observed[observed > i].min()
        low, high = sorted((x[lo], x[hi]))
        assert low - 1e-9 <= out[i] <= high + 1e-9
    assert np.array_equal(out[~mask], x[~mask])


@settings(max_examples=50, deadline=None)
@given(
    st.lists(
        st.one_of(st.floats(-1e4, 1e4, allow_nan=False), st.just(NaN)), min_size=2, max_size=60
    ),
    st.sampled_from(["2h", "3h", "1D", "7h"]),
)
def test_property_resample_sum_preserves_the_total(values, rule):
    df = hourly(values)
    res = run(df, "resample", rule=rule, agg="sum")
    assert res.df["v"].sum() == pytest.approx(df["v"].sum(), abs=1e-6)
    assert res.log_entry.counts["rows_before"] == len(df)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.floats(-100, 100, allow_nan=False), min_size=15, max_size=60),
    st.floats(0.5, 4.0),
    st.integers(2, 6).map(lambda k: 2 * k + 1),
)
def test_property_clip_never_leaves_a_value_outside_the_bounds(values, sigmas, window):
    df = hourly(values)
    params = OutliersParams(column="v", action="clip", n_sigmas=sigmas, window=window)
    det = detect_outliers(df, params)
    out = (
        run(df, "outliers", column="v", action="clip", n_sigmas=sigmas, window=window)
        .df["v"]
        .to_numpy()
    )
    changed = det.mask
    assert (out[changed] <= det.upper[changed] + 1e-9).all()
    assert (out[changed] >= det.lower[changed] - 1e-9).all()
    assert np.array_equal(out[~changed], np.array(values)[~changed])


# ------------------------------------------------------------------ recovery of known corruption

RECOVERY_TOLERANCE = 6.0
RECOVERY_RMSE = 3.0
"""Tolerances: the series has noise sd 1.5 around a 10-unit daily cycle. A repaired point is
linearly interpolated from neighbours that each carry that noise plus the cycle's curvature
(about 2 units across a 4-hour gap), so we require every repaired point within 6 (4 sigma;
observed max 4.8) and the RMSE of repaired points within 3 (observed 2.6)."""


def test_cleaning_recovers_a_known_clean_series():
    rng = np.random.default_rng(42)
    t = np.arange(24 * 28)
    original = hourly(
        100
        + 10 * np.sin(2 * np.pi * t / 24)
        + 3 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 1.5, len(t)),
        name="power",
    )
    messy = original.copy()
    blocks = [(100, 103), (200, 204), (300, 301), (450, 456)]  # all <= 6 points
    long_block = (500, 520)  # an outage the cleaner must not invent
    spikes = [150, 250, 350, 400, 600]
    for a, b in blocks + [long_block]:
        messy.iloc[a:b, 0] = NaN
    messy.iloc[spikes, 0] += np.array([60, -60, 60, -60, 60])
    dup_positions = [50, 60, 70]
    messy = pd.concat([messy, messy.iloc[dup_positions]]).sort_index(kind="stable")
    assert len(messy) == len(original) + 3

    step1 = run(messy, "drop_duplicates")
    step2 = run(step1.df, "outliers", column="power", action="set_missing", n_sigmas=4)
    step3 = run(step2.df, "handle_missing", column="power", method="interpolate_time", max_gap=6)
    cleaned = step3.df

    assert step1.log_entry.counts["exact_duplicates_removed"] == 3
    assert step2.log_entry.counts["outliers"] >= len(spikes)
    pd.testing.assert_index_equal(cleaned.index, original.index, check_freq=False)
    assert cleaned["power"].isna().sum() == long_block[1] - long_block[0]  # outage left alone
    assert cleaned["power"].iloc[long_block[0] : long_block[1]].isna().all()

    ok = cleaned["power"].notna().to_numpy()
    error = (cleaned["power"] - original["power"]).abs().to_numpy()
    touched = ok & (error > 1e-12)
    corrupted = np.zeros(len(t), bool)
    for a, b in blocks:
        corrupted[a:b] = True
    corrupted[spikes] = True
    false_positives = touched & ~corrupted
    assert false_positives.sum() <= 4  # a 4-sigma rule may catch a few ordinary noisy points
    assert touched[corrupted & ok].all()  # every corrupted point that was repaired changed
    assert error[touched].max() <= RECOVERY_TOLERANCE
    assert np.sqrt(np.mean(error[touched] ** 2)) <= RECOVERY_RMSE
    untouched = ok & ~touched
    assert np.array_equal(
        cleaned["power"].to_numpy()[untouched], original["power"].to_numpy()[untouched]
    )


# ------------------------------------------------------------------ review fixes


def test_hampel_memory_is_bounded_by_cells_not_rows_and_the_window_is_capped():
    import tracemalloc

    rng = np.random.default_rng(0)
    big = pd.DataFrame(
        {"v": rng.normal(size=20_000)}, index=pd.date_range("2024", periods=20_000, freq="min")
    )
    tracemalloc.start()
    run(big, "outliers", column="v", window=501)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 250e6  # before the fix a 1001 window needed ~800 MB for 50k rows
    with pytest.raises(CleaningError, match="less than or equal to 501"):
        run(big, "outliers", column="v", window=503)


def test_hampel_results_do_not_depend_on_chunk_size(monkeypatch):
    import datanaut.core.cleaner as cleaner

    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    x[[40, 300]] += 25
    df = hourly(x)
    whole = detect_outliers(df, OutliersParams(column="v", window=7))
    monkeypatch.setattr(cleaner, "_CHUNK_CELLS", 21)  # three rows at a time
    pieces = detect_outliers(df, OutliersParams(column="v", window=7))
    assert np.array_equal(whole.mask, pieces.mask)
    assert np.allclose(whole.lower, pieces.lower, equal_nan=True)
    assert whole.mask[[40, 300]].all()


def _conflict_frame(**cols):
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-01"])
    return pd.DataFrame(cols, index=idx)


def test_conflicts_keep_the_first_rows_text_even_when_it_is_missing():
    df = _conflict_frame(v=[1.0, 2.0, 5.0], note=[None, "later", "x"])
    res = run(df, "drop_duplicates", conflicts="mean")
    assert res.df.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")]
    assert pd.isna(res.df["note"].iloc[0]) and res.df["note"].iloc[1] == "x"
    assert res.df["v"].tolist() == [1.5, 5.0]  # first-appearance order is kept


def test_conflicts_keep_integer_columns_integer_when_the_result_is_whole():
    whole = run(_conflict_frame(n=[1, 3, 7]), "drop_duplicates", conflicts="mean")
    assert whole.df["n"].dtype == np.int64 and whole.df["n"].tolist() == [2, 7]
    assert whole.log_entry.counts["integer_columns_widened"] == 0
    total = run(_conflict_frame(n=[1, 3, 7]), "drop_duplicates", conflicts="sum")
    assert total.df["n"].dtype == np.int64 and total.df["n"].tolist() == [4, 7]


def test_conflicts_report_integer_columns_that_had_to_become_decimal():
    res = run(_conflict_frame(n=[1, 2, 7], m=[1, 3, 3]), "drop_duplicates", conflicts="mean")
    assert res.df["n"].tolist() == [1.5, 7.0] and res.df["m"].dtype == np.int64
    assert res.log_entry.counts["integer_columns_widened"] == 1
    assert (
        "1 integer column(s) became decimal" in res.log_entry.summary
        and "n" in res.log_entry.summary
    )


def test_there_is_no_mutable_operation_registry_and_import_order_does_not_matter():
    import subprocess
    import sys

    import datanaut.core.cleaner as cleaner

    assert not hasattr(cleaner, "_REGISTRY") and not hasattr(cleaner, "register_operation")
    code = "import datanaut.core.cleaner as c;print(' '.join(c.available_operations()))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "apply_rule_action" in out.stdout.split() and "outliers" in out.stdout.split()
