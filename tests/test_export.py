import copy

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from datanaut.core.export import (
    DANGEROUS_PREFIXES,
    is_formula_like,
    neutralize_formulas,
    neutralize_text,
    write_safe_csv,
)


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@", "\t", "\r"])
def test_each_dangerous_prefix_is_quoted(prefix):
    assert neutralize_text(prefix + "1+1") == "'" + prefix + "1+1"


@pytest.mark.parametrize(
    "text", ["", "plain", "a=b", "x-y", "1+1", "a =1", "'=already", "é=", "名前"]
)
def test_safe_text_is_left_alone(text):
    assert neutralize_text(text) == text


@pytest.mark.parametrize("text", [" =1", "  +cmd", "\u00a0@x", "\n-1", " \t =x"])
def test_blanks_in_front_do_not_hide_a_formula(text):
    assert neutralize_text(text) == "'" + text


def _frame():
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    return pd.DataFrame(
        {
            "note": ["=cmd|' /C calc'!A0", "ok", "+1", None],
            "neg": [-1.5, -2.0, 3.0, np.nan],  # negative NUMBERS are not formulas
            "flag": [True, False, True, False],
            "mixed": ["@SUM(A1)", -5, "fine", 7],
            "-evil": [1, 2, 3, 4],
        },
        index=idx,
    )


def test_text_cells_are_quoted_and_numbers_never_touched():
    df = _frame()
    safe, count = neutralize_formulas(df)
    assert safe["note"].tolist()[:3] == ["'=cmd|' /C calc'!A0", "ok", "'+1"] and pd.isna(
        safe["note"].iloc[3]
    )
    assert safe["neg"].tolist()[:3] == [-1.5, -2.0, 3.0]
    assert safe["flag"].tolist() == df["flag"].tolist()
    assert safe["mixed"].tolist() == ["'@SUM(A1)", -5, "fine", 7]  # only the strings change
    assert isinstance(safe["mixed"].iloc[1], int | np.integer)
    assert list(safe.columns) == ["note", "neg", "flag", "mixed", "'-evil"]
    assert count == 2 + 1 + 1  # two in 'note', one in 'mixed', one column name
    assert safe.index.equals(df.index)


def test_input_is_not_modified_and_a_copy_is_always_returned():
    df = _frame()
    snapshot = copy.deepcopy(df)
    safe, _ = neutralize_formulas(df)
    pd.testing.assert_frame_equal(df, snapshot)
    clean = pd.DataFrame({"a": ["x", "y"]})
    same, count = neutralize_formulas(clean)
    assert count == 0 and same is not clean
    pd.testing.assert_frame_equal(same, clean)
    assert safe is not df


def test_text_index_labels_are_neutralized_but_datetime_index_is_not():
    df = pd.DataFrame({"v": [1, 2]}, index=["=A1", "ok"])
    safe, count = neutralize_formulas(df)
    assert list(safe.index) == ["'=A1", "ok"] and count == 1
    dated = pd.DataFrame({"v": [1]}, index=pd.date_range("2024-01-01", periods=1))
    assert neutralize_formulas(dated)[1] == 0


def test_all_missing_and_empty_frames_are_fine():
    assert neutralize_formulas(pd.DataFrame({"a": [None, None]}))[1] == 0
    assert neutralize_formulas(pd.DataFrame())[1] == 0


def test_write_safe_csv_output_and_report(tmp_path):
    df = _frame()
    path = tmp_path / "out.csv"
    result = write_safe_csv(df, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[-1] == "'-evil"
    assert "'=cmd|' /C calc'!A0" in lines[1] or "\"'=cmd|' /C calc'!A0\"" in lines[1]
    assert ",-1.5," in lines[1]  # a negative number stays a number
    assert result.cells_neutralized == 4 and (result.rows, result.columns) == (4, 5)
    entry = result.log_entry
    assert entry.operation == "export_csv" and entry.read_only
    assert entry.counts["cells_neutralized"] == 4 and "prefixed with a quote" in entry.summary
    back = pd.read_csv(path, index_col=0)
    assert back["note"].dropna().str.lstrip("'").tolist()[0] == "=cmd|' /C calc'!A0"


def test_written_csv_has_no_cell_that_starts_with_a_formula_character(tmp_path):
    df = pd.DataFrame({"t": ["=1", "+2", "-3", "@4", "ok"], "n": [-1, -2, -3, -4, -5]})
    path = tmp_path / "x.csv"
    write_safe_csv(df, path)
    back = pd.read_csv(path, index_col=0, dtype=str)
    assert not back["t"].str.startswith(DANGEROUS_PREFIXES).any()
    assert back["n"].tolist() == ["-1", "-2", "-3", "-4", "-5"]


@settings(max_examples=80, deadline=None)
@given(
    st.lists(
        st.text(alphabet=st.sampled_from(list("=+-@\t\rab 1'é")), max_size=6),
        min_size=1,
        max_size=30,
    )
)
def test_property_no_text_cell_starts_dangerously_afterwards(values):
    df = pd.DataFrame({"t": values})
    safe, count = neutralize_formulas(df)
    assert not any(is_formula_like(v) for v in safe["t"])
    assert count == sum(is_formula_like(v) for v in values)
    for before, after in zip(values, safe["t"], strict=True):
        assert after == before or after == "'" + before
