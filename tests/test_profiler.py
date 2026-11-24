import json

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from datanaut.core.fingerprint import fingerprint
from datanaut.core.loader import load_file
from datanaut.core.log import PipelineLog
from datanaut.core.profiler import profile
from datanaut.core.timeindex import build_time_index, detect_time_columns

NUMERIC = ["power_kw", "temperature", "energy_kwh"]


def _indexed(variants, name):
    df = load_file(variants[name].data).df
    return build_time_index(df, detect_time_columns(df)[0].columns).df


def _hourly(n=48, **cols):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({k: v for k, v in cols.items()}, index=idx)


# ------------------------------------------------------------------ every messy variant


def test_variant_profile_matches_pandas_and_expectations(variant):
    df = load_file(variant.data, filename=variant.filename).df
    cands = detect_time_columns(df)[0].columns
    indexed = build_time_index(df, cands).df
    before = fingerprint(indexed)
    p = profile(indexed)
    assert fingerprint(indexed) == before  # read-only

    assert p.rows == variant.expect["rows"]
    assert p.n_columns == indexed.shape[1]
    assert p.memory_bytes > 0
    assert p.index_monotonic is True
    assert p.sampling is not None and p.sampling.median_interval_seconds == 3600.0
    assert p.sampling.median_interval_label == "1h"

    temp = variant.expect.get("temp_name", "temperature")
    for col in ("power_kw", temp, "energy_kwh"):
        c = p.column(col)
        s = indexed[col]
        assert c.inferred_type == "numeric"
        assert c.missing_count == int(s.isna().sum())
        assert np.isclose(c.mean, s.mean()) and np.isclose(c.std, s.std())
        assert np.isclose(c.skew, s.skew()) and c.min == s.min() and c.max == s.max()
    for col, n in variant.expect.get("missing", {}).items():
        assert p.column(col).missing_count == n

    assert p.log_entry.operation == "profile" and p.log_entry.read_only
    assert p.missingness_matrix.shape[1] == indexed.shape[1]
    assert json.loads(p.model_dump_json())["rows"] == p.rows


def test_gap_variant_reports_six_missing_hours(variants):
    p = profile(_indexed(variants, "gap_6h"))
    assert p.gaps.gap_count == 1
    assert p.gaps.missing_timestamps == 6 and p.gaps.longest_gap_timestamps == 6
    assert p.gaps.longest_gap_seconds == 6 * 3600
    g = p.gaps.largest[0]
    assert g.start == "2024-03-05 04:00:00" and g.end == "2024-03-05 09:00:00"
    assert any("6 expected timestamp" in w for w in p.warnings)
    assert p.sampling.irregular_count == 1 and p.sampling.off_grid_count == 0
    assert not any("Irregular sampling" in w for w in p.warnings)  # a gap is not jitter


def test_clean_variants_have_no_gaps_or_warnings(variants):
    p = profile(_indexed(variants, "baseline"))
    assert p.gaps.gap_count == 0 and p.gaps.largest == []
    assert p.sampling.regular_share == 1.0 and p.sampling.irregular_count == 0
    assert p.warnings == []
    assert p.duplicates.full_row_duplicates == 0


def test_duplicate_variant_is_split_into_identical(variants):
    d = profile(_indexed(variants, "duplicates")).duplicates
    assert d.full_row_duplicates == 5
    assert (d.duplicate_timestamps, d.duplicate_timestamps_identical) == (5, 5)
    assert d.duplicate_timestamps_conflicting == 0


def test_question_marks_variant_runs_and_matrix(variants):
    p = profile(_indexed(variants, "question_marks"), max_bins=10)
    power = p.column("power_kw")
    assert power.missing_count == 30 and 1 <= power.longest_missing_run <= power.missing_runs + 30
    assert p.missingness_matrix.shape == (10, 3)
    overall = float(p.missingness_matrix["power_kw"].mean())
    assert 0.03 < overall < 0.15


def test_text_columns_are_typed(variants):
    p = profile(load_file(variants["latin1"].data).df)
    assert p.column("site").inferred_type == "categorical" and p.column("site").unique_count == 1
    assert p.constant_columns == ["site"]
    assert any("'site' is constant" in w for w in p.warnings)
    assert p.column("timestamp").inferred_type == "text"
    assert p.sampling is None and p.gaps is None and p.index_monotonic is None


# ------------------------------------------------------------------ targeted behaviours


def test_conflicting_vs_identical_duplicate_timestamps():
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 01:00"]
    )
    df = pd.DataFrame({"v": [1.0, 1.0, 2.0, 5.0]}, index=idx)
    d = profile(df).duplicates
    assert d.duplicate_timestamps == 2
    assert d.duplicate_timestamps_identical == 1 and d.duplicate_timestamps_conflicting == 1
    assert d.full_row_duplicates == 1
    assert any("conflicting" in w for w in profile(df).warnings)


def test_duplicates_treat_nan_as_equal():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    d = profile(pd.DataFrame({"v": [np.nan, np.nan]}, index=idx)).duplicates
    assert d.full_row_duplicates == 1


def test_missing_runs_and_longest():
    s = [1, np.nan, np.nan, 3, np.nan, 5, np.nan, np.nan, np.nan]
    c = profile(_hourly(len(s), v=s)).column("v")
    assert (c.missing_count, c.missing_runs, c.longest_missing_run) == (6, 3, 3)
    assert profile(_hourly(3, v=[1, 2, 3])).column("v").missing_runs == 0


def test_high_missing_empty_and_constant_warnings():
    n = 40
    df = _hourly(
        n,
        mostly=[1.0, 2.0] * 5 + [np.nan] * 30,
        dead=[np.nan] * n,
        flat=[7.0] * n,
        fine=np.arange(n, dtype=float),
    )
    p = profile(df)
    text = " ".join(p.warnings)
    assert "'mostly' is 75.0% missing" in text
    assert "'dead' is entirely missing" in text and p.empty_columns == ["dead"]
    assert p.constant_columns == ["flat"]
    assert "'fine'" not in text
    assert p.column("dead").inferred_type == "empty"
    assert "consecutive missing" in text  # the 30-long run


def test_irregular_sampling_warning():
    rng = np.random.default_rng(0)
    offsets = np.cumsum(rng.integers(30, 90, size=100))
    idx = pd.DatetimeIndex(pd.Timestamp("2024-01-01") + pd.to_timedelta(offsets, unit="s"))
    p = profile(pd.DataFrame({"v": range(100)}, index=idx))
    assert 50 <= p.sampling.median_interval_seconds <= 70
    assert p.sampling.irregular_count > 10 and p.sampling.off_grid_count > 10
    assert any("Irregular sampling" in w for w in p.warnings)


def test_unsorted_index_flagged():
    df = _hourly(4, v=[1, 2, 3, 4]).iloc[[2, 0, 1, 3]]
    p = profile(df)
    assert p.index_monotonic is False
    assert any("chronological" in w for w in p.warnings)
    assert p.sampling is not None  # computed on sorted distinct timestamps


def test_interval_labels():
    for freq, label in (("15min", "15min"), ("D", "1d"), ("2h", "2h"), ("30s", "30s")):
        idx = pd.date_range("2024-01-01", periods=20, freq=freq)
        assert (
            profile(pd.DataFrame({"v": range(20)}, index=idx)).sampling.median_interval_label
            == label
        )


def test_monthly_data_is_regular_within_tolerance():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    p = profile(pd.DataFrame({"v": range(24)}, index=idx))
    assert p.sampling.irregular_count == 0 and p.gaps.gap_count == 0


def test_numeric_stats_none_when_too_short_for_skew():
    c = profile(_hourly(2, v=[1.0, 2.0])).column("v")
    assert c.skew is None and c.mean == 1.5


def test_boolean_and_datetime_columns():
    df = _hourly(5, flag=[True, False, True, True, False])
    df["when"] = pd.date_range("2025-01-01", periods=5)
    p = profile(df)
    assert p.column("flag").inferred_type == "boolean"
    assert p.column("when").inferred_type == "datetime"
    assert p.column("flag").mean is None


def test_empty_and_single_row_frames_do_not_crash():
    empty = profile(pd.DataFrame({"v": []}, index=pd.DatetimeIndex([])))
    assert empty.rows == 0 and empty.sampling is None and empty.missingness_matrix.empty
    one = profile(_hourly(1, v=[1.0]))
    assert one.sampling is None and one.gaps is None


def test_range_index_frame_uses_row_bins():
    df = pd.DataFrame({"a": [1.0] * 10 + [np.nan] * 10})
    p = profile(df, max_bins=2)
    assert list(p.missingness_matrix["a"]) == [0.0, 1.0]
    assert list(p.missingness_matrix.index) == [0, 10]
    assert p.duplicates.duplicate_timestamps is None


def test_missingness_matrix_time_bins_and_empty_bin_during_gap():
    idx = pd.date_range("2024-01-01", periods=100, freq="h").delete(slice(40, 60))
    a = np.where(np.arange(len(idx)) < 40, 0.0, np.nan)
    p = profile(pd.DataFrame({"a": a}, index=idx), max_bins=10)
    m = p.missingness_matrix["a"]
    assert len(m) == 10 and isinstance(m.index, pd.DatetimeIndex)
    assert m.iloc[0] == 0.0 and m.iloc[-1] == 1.0
    assert m.isna().sum() >= 1  # bins that fall entirely inside the outage have no rows


def test_matrix_bin_cap_and_large_frames():
    n = 100_000
    df = pd.DataFrame(
        {"v": np.arange(n, dtype=float)}, index=pd.date_range("2020", periods=n, freq="min")
    )
    assert profile(df, max_bins=50).missingness_matrix.shape == (50, 1)
    assert profile(df.iloc[:7], max_bins=50).missingness_matrix.shape == (7, 1)


def test_index_named_like_a_column_is_fine():
    df = _hourly(3, ts=[1.0, 2.0, 3.0])
    df.index.name = "ts"
    assert profile(df).duplicates.full_row_duplicates == 0


def test_log_entry_serializes():
    p = profile(_hourly(5, v=[1, 2, 3, 4, 5]))
    log = PipelineLog()
    log.extend([p.log_entry])
    assert PipelineLog.from_json(log.to_json()).entries == log.entries
    assert p.log_entry.counts["rows"] == 5


@settings(max_examples=40, deadline=None)
@given(
    st.integers(40, 120),
    st.lists(st.integers(1, 118), max_size=6, unique=True),
)
def test_property_missing_timestamps_match_removed_rows(n, removed):
    removed = sorted(r for r in removed if r < n - 1)
    # keep removals non-adjacent so the median interval stays one step
    kept_removed = [r for i, r in enumerate(removed) if i == 0 or r - removed[i - 1] > 1]
    idx = pd.date_range("2024-01-01", periods=n, freq="h").delete(kept_removed)
    p = profile(pd.DataFrame({"v": np.arange(len(idx), dtype=float)}, index=idx))
    assert p.gaps.missing_timestamps == len(kept_removed)
