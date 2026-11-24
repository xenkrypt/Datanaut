import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from datanaut.core.fingerprint import fingerprint
from datanaut.core.loader import load_file
from datanaut.core.timeindex import (
    build_time_index,
    detect_time_columns,
    detect_value_columns,
)
from datanaut.errors import ParseError

NUMERIC = ["power_kw", "temperature", "energy_kwh"]


def _frame(stamps, **cols):
    return pd.DataFrame({"when": stamps, **cols})


# ------------------------------------------------------------------ every messy variant


def test_variant_detect_and_build(variant):
    df = load_file(variant.data, filename=variant.filename).df
    cands = detect_time_columns(df)
    expected = ["Date", "Time"] if variant.expect.get("split") else ["timestamp"]
    assert cands[0].columns == expected
    assert cands[0].parse_share == 1.0

    before = fingerprint(df)
    res = build_time_index(df, cands[0].columns)
    assert fingerprint(df) == before  # input untouched

    out = res.df
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing
    assert len(out) == variant.expect["rows"] and len(res.rejected) == 0
    assert not any(c in out.columns for c in expected)
    assert res.flags.ambiguous is False and res.flags.unparseable_rows == 0

    # values stay attached to their timestamps after sorting
    truth = variant.truth.set_axis(pd.DatetimeIndex(variant.timestamps)).sort_index(kind="stable")
    got = out[["power_kw", variant.expect.get("temp_name", "temperature"), "energy_kwh"]].set_axis(
        NUMERIC, axis=1
    )
    assert list(out.index) == list(truth.index)
    np.testing.assert_allclose(
        got.to_numpy(float), truth.to_numpy(float), rtol=1e-9, equal_nan=True
    )
    assert [e.operation for e in res.log][0] == "build_time_index"


def test_variant_specific_flags(variants):
    def build(name):
        df = load_file(variants[name].data).df
        return build_time_index(df, detect_time_columns(df)[0].columns).flags

    assert build("shuffled").rows_moved_by_sort > 0
    assert build("baseline").rows_moved_by_sort == 0
    assert build("duplicates").duplicate_timestamps == 5
    assert build("baseline").duplicate_timestamps == 0
    gap = load_file(variants["gap_6h"].data).df
    idx = build_time_index(gap, "timestamp").df.index
    assert (idx.to_series().diff().max()) == pd.Timedelta(hours=7)


def test_value_columns_for_variants(variants):
    df = load_file(variants["latin1"].data).df
    assert detect_value_columns(df, ["timestamp"]) == ["power_kw", "température", "energy_kwh"]
    df = load_file(variants["date_time_columns"].data).df
    assert detect_value_columns(df, ["Date", "Time"]) == NUMERIC


# ------------------------------------------------------------------ detection


def test_ranking_prefers_parse_share_then_name_hint():
    df = pd.DataFrame(
        {
            "b_col": ["2024-01-01", "2024-01-02", "nope", "2024-01-04"],
            "created_at": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "other": ["2024-02-01", "2024-02-02", "2024-02-03", "2024-02-04"],
            "words": ["a", "b", "c", "d"],
        }
    )
    cands = detect_time_columns(df)
    assert [c.columns[0] for c in cands] == ["created_at", "other", "b_col"]
    assert cands[0].name_hint and not cands[1].name_hint  # hint wins the tie at share 1.0
    assert cands[2].parse_share == 0.75


def test_numeric_and_text_columns_are_not_timestamps():
    df = pd.DataFrame(
        {"n": [1, 2, 3], "numtext": ["10", "20", "30"], "t": ["12:30", "13:30", "14:30"]}
    )
    assert detect_time_columns(df) == []


def test_date_and_time_pair_replaces_date_alone():
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Time": ["00:00", "12:00", "00:00"],
            "v": [1.0, 2.0, 3.0],
        }
    )
    cands = detect_time_columns(df)
    assert [c.columns for c in cands] == [["Date", "Time"]]
    assert cands[0].kind == "date_time"


def test_daily_date_only_column_is_a_single_candidate():
    df = pd.DataFrame({"day": ["2024-01-01", "2024-01-02"], "v": [1, 2]})
    assert detect_time_columns(df)[0].columns == ["day"]


def test_excel_datetime_columns_detected():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5, freq="h"), "v": range(5)})
    cand = detect_time_columns(df)[0]
    assert cand.columns == ["ts"] and cand.parse_share == 1.0


def test_value_columns_exclusions():
    n = 20
    df = pd.DataFrame(
        {
            "ts": range(n),  # counter, unique and increasing
            "user_id": np.arange(n) * 7 % 13,  # name hint
            "const": [5.0] * n,
            "empty": [np.nan] * n,
            "flag": [True, False] * (n // 2),
            "label": ["a", "b"] * (n // 2),
            "temp": np.linspace(0, 1, n),
            "count": np.arange(n) % 4,
            "time": range(n),
        }
    )
    assert detect_value_columns(df, ["time"]) == ["temp", "count"]


# ------------------------------------------------------------------ day-first handling


def test_dayfirst_detected_when_only_one_reading_parses_all():
    df = _frame(["13/04/2024", "14/04/2024", "01/05/2024"], v=[1, 2, 3])
    res = build_time_index(df, "when")
    assert res.choices.dayfirst is True or res.choices.dayfirst_source == "not_needed"
    assert list(res.df.index) == list(pd.to_datetime(["2024-04-13", "2024-04-14", "2024-05-01"]))
    res = build_time_index(_frame(["04/13/2024", "04/14/2024", "05/01/2024"], v=[1, 2, 3]), "when")
    assert list(res.df.index) == list(pd.to_datetime(["2024-04-13", "2024-04-14", "2024-05-01"]))


def test_ambiguous_dates_are_not_guessed():
    df = _frame(["01/02/2024", "03/04/2024", "05/06/2024"], v=[1, 2, 3])
    res = build_time_index(df, "when")
    assert res.flags.ambiguous and not res.flags.index_set
    assert res.flags.ambiguous_values == 3
    first = res.flags.ambiguous_preview[0]
    assert (first.raw, first.dayfirst, first.monthfirst) == (
        "01/02/2024",
        "2024-02-01 00:00:00",
        "2024-01-02 00:00:00",
    )
    assert not isinstance(res.df.index, pd.DatetimeIndex)
    pd.testing.assert_frame_equal(res.df, df)
    assert res.log[0].read_only and "dayfirst" in res.log[0].summary


def test_user_override_resolves_ambiguity():
    df = _frame(["01/02/2024", "03/04/2024"], v=[1, 2])
    d = build_time_index(df, "when", dayfirst=True)
    m = build_time_index(df, "when", dayfirst=False)
    assert d.choices.dayfirst_source == "user"
    assert list(d.df.index) == list(pd.to_datetime(["2024-02-01", "2024-04-03"]))
    assert list(m.df.index) == list(pd.to_datetime(["2024-01-02", "2024-03-04"]))


def test_iso_dates_are_never_swapped_by_dayfirst():
    df = _frame(["2024-03-01 00:00:00", "2024-03-02 00:00:00"], v=[1, 2])
    for flag in (None, True, False):
        res = build_time_index(df, "when", dayfirst=flag)
        assert res.df.index[0] == pd.Timestamp("2024-03-01")
    assert build_time_index(df, "when").flags.ambiguous is False


# ------------------------------------------------------------------ bad rows, sorting, tz


def test_unparseable_and_missing_timestamps_are_set_aside_not_lost():
    df = _frame(["2024-01-03", "garbage", None, "2024-01-01", "2024-01-02", "??"], v=range(6))
    res = build_time_index(df, "when")
    assert res.flags.unparseable_rows == 2 and res.flags.missing_timestamp_rows == 1
    assert res.flags.unparseable_examples == ["garbage", "??"]
    assert list(res.df["v"]) == [3, 4, 0]
    assert list(res.rejected["v"]) == [1, 2, 5]
    assert len(res.df) + len(res.rejected) == len(df)
    ops = [e.operation for e in res.log]
    assert ops == ["build_time_index", "set_aside_untimed_rows"]
    assert res.log[0].rows_before == 6 and res.log[0].rows_after == 3
    assert res.log[1].counts == {"missing_timestamp_rows": 1, "unparseable_rows": 2}


def test_rows_moved_by_sort_counts_changed_positions():
    df = _frame(["2024-01-02", "2024-01-01", "2024-01-03"], v=[1, 2, 3])
    res = build_time_index(df, "when")
    assert res.flags.rows_moved_by_sort == 2
    assert list(res.df["v"]) == [2, 1, 3]


def test_stable_sort_keeps_duplicate_timestamps_in_file_order():
    df = _frame(["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-01"], v=[0, 1, 2, 3])
    res = build_time_index(df, "when")
    assert list(res.df["v"]) == [1, 2, 3, 0]
    assert res.flags.duplicate_timestamps == 2


def test_timezone_naive_stays_naive_and_offsets_are_kept():
    naive = build_time_index(_frame(["2024-03-01 00:00", "2024-03-01 01:00"], v=[1, 2]), "when")
    assert naive.df.index.tz is None and naive.choices.timezone is None
    aware = build_time_index(
        _frame(["2024-03-01T00:00:00+02:00", "2024-03-01T01:00:00+02:00"], v=[1, 2]), "when"
    )
    assert aware.df.index.tz is not None and aware.choices.timezone is not None
    assert not aware.flags.mixed_timezone_offsets


def test_mixed_offsets_are_converted_to_utc_and_flagged():
    df = _frame(["2024-03-01T00:00:00+02:00", "2024-03-01T00:30:00+01:00"], v=[1, 2])
    res = build_time_index(df, "when")
    assert res.flags.mixed_timezone_offsets
    assert str(res.df.index.tz) == "UTC"
    assert list(res.df.index.hour) == [22, 23]
    assert list(res.df["v"]) == [1, 2]


def test_date_plus_time_columns_with_ampm_and_missing():
    df = pd.DataFrame(
        {
            "Date": ["2024-03-01", "2024-03-01", None, "2024-03-02"],
            "Time": ["01:30 PM", "02:30 AM", "05:00 AM", "12:00 AM"],
            "v": [1, 2, 3, 4],
        }
    )
    res = build_time_index(df, ["Date", "Time"])
    assert res.choices.index_name == "timestamp"
    assert list(res.df.index.hour) == [2, 13, 0]
    assert res.flags.missing_timestamp_rows == 1
    assert list(res.rejected["v"]) == [3]


def test_excel_style_datetime_date_column_plus_text_time():
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-03-01", "2024-03-01"]),
            "t": ["00:00:00", "01:00:00"],
            "v": [1, 2],
        }
    )
    res = build_time_index(df, ["d", "t"])
    assert list(res.df.index) == list(pd.to_datetime(["2024-03-01 00:00", "2024-03-01 01:00"]))


def test_datetime64_column_used_directly():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-02", "2024-01-01"]), "v": [1, 2]})
    res = build_time_index(df, "ts")
    assert res.choices.dayfirst_source == "not_needed"
    assert list(res.df["v"]) == [2, 1] and res.flags.rows_moved_by_sort == 2


def test_errors():
    df = _frame(["a", "b"], v=[1, 2])
    with pytest.raises(ParseError, match="not found"):
        build_time_index(df, "missing")
    with pytest.raises(ParseError, match="could be read"):
        build_time_index(df, "when")
    with pytest.raises(ParseError):
        build_time_index(df, ["when", "v", "v"])


def test_index_name_does_not_collide():
    df = pd.DataFrame({"d": ["2024-01-01"], "t": ["00:00"], "timestamp": [5.0], "v": [1.0]})
    assert build_time_index(df, ["d", "t"]).choices.index_name == "timestamp_index"


def test_log_is_json_serializable():
    from datanaut.core.log import PipelineLog

    res = build_time_index(_frame(["2024-01-02", "bad", "2024-01-01"], v=[1, 2, 3]), "when")
    log = PipelineLog()
    log.extend(res.log)
    assert PipelineLog.from_json(log.to_json()).entries == log.entries


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(0, 10_000), min_size=1, max_size=40, unique=True), st.randoms())
def test_property_sorting_keeps_rows_aligned(offsets, rnd):
    base = pd.Timestamp("2024-01-01")
    stamps = [(base + pd.Timedelta(minutes=o)) for o in offsets]
    rnd.shuffle(stamps)
    df = pd.DataFrame(
        {"when": [s.strftime("%Y-%m-%d %H:%M:%S") for s in stamps], "v": [s.value for s in stamps]}
    )
    res = build_time_index(df, "when")
    assert res.df.index.is_monotonic_increasing
    assert list(res.df["v"]) == [t.value for t in res.df.index]
    assert sorted(res.df.index) == sorted(stamps)


def test_non_string_column_labels_work():
    df = pd.DataFrame({0: ["2024-01-02", "2024-01-01"], 1: [1.0, 2.0]}, index=["a", "b"])
    res = build_time_index(df, 0)
    assert list(res.df[1]) == [2.0, 1.0] and res.choices.time_columns == ["0"]
