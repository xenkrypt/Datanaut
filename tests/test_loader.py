import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from datanaut.core.loader import load_file
from datanaut.core.log import PipelineLog
from datanaut.errors import (
    DatanautError,
    FileTooLargeError,
    ParseError,
    UnsupportedFormatError,
)

NUMERIC = ["power_kw", "temperature", "energy_kwh"]


def _csv(text: str, **kwargs):
    return load_file(text.encode("utf-8"), filename="t.csv", **kwargs)


def _ts_col(df: pd.DataFrame) -> str:
    return "timestamp" if "timestamp" in df.columns else "Date"


# ---------------------------------------------------------------- every messy variant


def test_variant_values_shape_and_log(variant):
    res = load_file(variant.data, filename=variant.filename)
    df = res.df
    assert len(df) == variant.expect["rows"]
    assert res.source_info.format == variant.expect["format"]
    assert res.source_info.size_bytes == len(variant.data)
    temp_col = variant.expect.get("temp_name", "temperature")
    for col in ("power_kw", temp_col, "energy_kwh"):
        assert pd.api.types.is_numeric_dtype(df[col]), col
    got = df[["power_kw", temp_col, "energy_kwh"]].set_axis(NUMERIC, axis=1)
    np.testing.assert_allclose(
        got.to_numpy(float), variant.truth.to_numpy(float), rtol=1e-9, equal_nan=True
    )
    assert res.log[0].operation == "load"
    assert [e.step_id for e in res.log] == list(range(1, len(res.log) + 1))
    assert {"load", "normalize_missing", "convert_numeric"} <= {e.operation for e in res.log}
    assert PipelineLog(entries=res.log).to_json()


def test_baseline_timestamps_kept_as_text(variants):
    v = variants["baseline"]
    res = load_file(v.data, filename=v.filename)
    assert res.source_info.encoding == "utf-8"
    assert res.source_info.delimiter == ","
    assert list(res.df["timestamp"]) == [t.strftime("%Y-%m-%d %H:%M:%S") for t in v.timestamps]


def test_semicolon_delimiter(variants):
    res = load_file(variants["semicolon"].data)
    assert res.source_info.delimiter == ";"
    assert list(res.df.columns) == ["timestamp", *NUMERIC]


def test_decimal_comma_and_eu_thousands(variants):
    res = load_file(variants["decimal_comma"].data)
    assert res.source_info.delimiter == ";"
    assert res.df["energy_kwh"].min() > 1000
    fmt = res.log[2].params["per_column"]["energy_kwh"]["format"]
    assert fmt == "decimal comma"


def test_us_thousands_separators_quoted(variants):
    res = load_file(variants["thousands_us"].data)
    assert res.source_info.delimiter == ","
    assert res.df["energy_kwh"].min() > 1000
    assert res.log[2].params["per_column"]["energy_kwh"]["format"] == "decimal point"


def test_latin1_encoding_decoded_correctly(variants):
    res = load_file(variants["latin1"].data)
    assert res.source_info.encoding not in ("utf-8", "utf-8-sig")
    assert "température" in res.df.columns
    assert set(res.df["site"]) == {"Zürich"}


def test_question_marks_and_other_markers_become_missing(variants):
    v = variants["question_marks"]
    res = load_file(v.data)
    assert int(res.df["power_kw"].isna().sum()) == 30
    assert int(res.df["temperature"].isna().sum()) == 16
    norm = next(e for e in res.log if e.operation == "normalize_missing")
    # blank cells are not "markers"; the other 14 temperature markers and 30 "?" are
    assert norm.params["per_column"]["power_kw"] == 30
    assert norm.params["per_column"]["temperature"] == 14
    assert norm.counts["cells_set_missing"] == 44


def test_separate_date_and_time_columns_kept(variants):
    res = load_file(variants["date_time_columns"].data)
    assert list(res.df.columns[:2]) == ["Date", "Time"]
    assert res.df["Date"].iloc[0] == "2024-03-01"
    assert res.df["Time"].iloc[1] == "01:00:00"


def test_duplicate_rows_are_kept_not_dropped(variants):
    res = load_file(variants["duplicates"].data)
    assert len(res.df) == len(variants["baseline"].truth) + 5
    assert int(res.df.duplicated().sum()) == 5


def test_gap_rows_not_filled(variants):
    res = load_file(variants["gap_6h"].data)
    assert len(res.df) == len(variants["baseline"].truth) - 6


def test_shuffled_order_is_preserved(variants):
    res = load_file(variants["shuffled"].data)
    ts = pd.to_datetime(res.df["timestamp"])
    assert not ts.is_monotonic_increasing
    assert list(res.df["timestamp"]) == [
        t.strftime("%Y-%m-%d %H:%M:%S") for t in variants["shuffled"].timestamps
    ]


def test_xlsx_types_sheets_and_text_numbers(variants):
    v = variants["xlsx"]
    res = load_file(v.data, filename=v.filename)
    assert res.source_info.format == "xlsx"
    assert res.source_info.sheet_names == ["data", "notes"]
    assert res.source_info.sheet == "data"
    assert res.source_info.encoding is None and res.source_info.delimiter is None
    assert pd.api.types.is_datetime64_any_dtype(res.df["timestamp"])
    # temperature was stored as text; power had "?" cells inside an otherwise numeric column
    assert list(res.log[2].params["per_column"]) == ["temperature"]
    assert pd.api.types.is_numeric_dtype(res.df["power_kw"])
    assert int(res.df["power_kw"].isna().sum()) == 10
    assert res.log[1].params["per_column"] == {"power_kw": 10}


def test_xlsx_sheet_selection(variants):
    v = variants["xlsx"]
    by_name = load_file(v.data, sheet="notes")
    by_index = load_file(v.data, sheet=1)
    assert list(by_name.df.columns) == ["note"]
    assert by_name.df.equals(by_index.df)
    with pytest.raises(ParseError, match="not found"):
        load_file(v.data, sheet="missing")
    with pytest.raises(ParseError):
        load_file(v.data, sheet=5)


# ---------------------------------------------------------------- sources and format checks


def test_path_bytes_and_filelike_agree(tmp_path, variants):
    v = variants["semicolon"]
    p = tmp_path / "data.csv"
    p.write_bytes(v.data)
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    a = load_file(p)
    b = load_file(v.data)
    with open(p, "rb") as fh:
        c = load_file(fh)
    d = load_file(io.BytesIO(v.data), filename="x.csv")
    assert a.df.equals(b.df) and b.df.equals(c.df) and c.df.equals(d.df)
    assert a.source_info.filename == "data.csv"
    assert c.source_info.filename == "data.csv"
    assert d.source_info.filename == "x.csv"
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before


def test_extension_is_not_trusted(variants):
    as_csv_name = load_file(variants["xlsx"].data, filename="really.csv")
    assert as_csv_name.source_info.format == "xlsx"
    as_xlsx_name = load_file(variants["baseline"].data, filename="really.xlsx")
    assert as_xlsx_name.source_info.format == "csv"


def test_size_cap_enforced_for_all_source_kinds(tmp_path, variants):
    data = variants["baseline"].data
    p = tmp_path / "big.csv"
    p.write_bytes(data)
    for src in (data, p, io.BytesIO(data)):
        with pytest.raises(FileTooLargeError):
            load_file(src, max_bytes=1000)


@pytest.mark.parametrize(
    "payload",
    [
        b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        bytes.fromhex("D0CF11E0A1B11AE1") + b"\x00" * 100,
        b"a,b\n1,2\x00\x00\x00\x00\n" + b"\x00" * 200,
        bytes(range(256)) * 4,
    ],
)
def test_unsupported_formats_rejected(payload):
    with pytest.raises(UnsupportedFormatError):
        load_file(payload)


def test_zip_that_is_not_a_workbook_is_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<x/>")
    with pytest.raises(UnsupportedFormatError):
        load_file(buf.getvalue())


def test_corrupt_zip_is_parse_error():
    with pytest.raises(ParseError):
        load_file(b"PK\x03\x04" + b"garbage" * 10)


def test_xlsx_unpacked_size_guard():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/workbook.xml", "0" * 5_000_000)
    assert len(buf.getvalue()) < 100_000
    with pytest.raises(FileTooLargeError):
        load_file(buf.getvalue(), max_unpacked_bytes=1_000_000)


@pytest.mark.parametrize("payload", [b"", b"   \n\n  ", b"a,b\n"])
def test_empty_inputs_are_parse_errors(payload):
    with pytest.raises(ParseError):
        load_file(payload)


def test_missing_file_is_friendly_error(tmp_path):
    with pytest.raises(DatanautError):
        load_file(tmp_path / "nope.csv")


def test_text_mode_filelike_rejected():
    with pytest.raises(ParseError):
        load_file(io.StringIO("a,b\n1,2\n"))


# ---------------------------------------------------------------- messy value handling


def test_numeric_threshold_is_95_percent():
    ok = "v\n" + "\n".join(["1.5"] * 96 + ["oops"] * 4) + "\n"  # 96%
    res = _csv(ok)
    assert pd.api.types.is_numeric_dtype(res.df["v"])
    assert int(res.df["v"].isna().sum()) == 4
    info = res.log[2].params["per_column"]["v"]
    assert info["converted"] == 96 and info["unparseable_set_missing"] == 4
    assert info["unparseable_examples"] == ["oops"] * 4

    bad = "v\n" + "\n".join(["1.5"] * 90 + ["oops"] * 10) + "\n"  # 90%
    res = _csv(bad)
    assert not pd.api.types.is_numeric_dtype(res.df["v"])
    assert (res.df["v"] == "oops").sum() == 10
    assert res.log[2].params["per_column"] == {}


def test_leading_zero_ids_stay_text():
    res = _csv("zip\n" + "\n".join(["01234", "00501", "02134"] * 5) + "\n")
    assert not pd.api.types.is_numeric_dtype(res.df["zip"])
    assert res.df["zip"].iloc[0] == "01234"


def test_ambiguous_separator_values_are_reported():
    res = _csv('v\n"1,234"\n"2,345"\n"3,456"\n')
    info = res.log[2].params["per_column"]["v"]
    assert info["ambiguous_separator_values"] == 3
    assert list(res.df["v"]) == [1234, 2345, 3456]


def test_marker_matching_is_trimmed_and_case_insensitive():
    res = _csv("a,b\n 1 , N/A \n2,Null\n3,--\n4,?\n")
    assert list(res.df["a"]) == [1, 2, 3, 4]
    assert res.df["b"].isna().all()


def test_negative_exponent_and_plus_signs():
    res = _csv("v\n+1.5\n-2\n1e3\n.5\n")
    np.testing.assert_allclose(res.df["v"], [1.5, -2, 1000, 0.5])


def test_ragged_and_empty_rows_reported_not_dropped_silently():
    text = "a,b,c\n1,2,p\n\n4,5\n6,7,q,r\n , , \n10,11,s\n"
    res = _csv(text)
    assert list(res.df["a"]) == [1, 4, 6, 10]
    assert res.df["c"].isna().sum() == 1  # padded short row
    assert res.df["c"].iloc[2] == "q,r"  # extra fields merged, not lost
    load = res.log[0]
    assert load.rows_before == 6 and load.rows_after == 4
    warn = next(e for e in res.log if e.operation == "parse_warnings")
    assert warn.counts == {"empty_rows_removed": 2, "short_rows": 1, "long_rows": 1}
    assert warn.read_only


def test_header_cleanup_is_reported():
    res = _csv("a,a, ,b,\n1,2,3,4,\n5,6,7,8,\n")
    assert list(res.df.columns) == ["a", "a.1", "unnamed_2", "b"]
    warn = next(e for e in res.log if e.operation == "parse_warnings")
    assert warn.params["renamed_columns"] == {"a.1": "a", "unnamed_2": ""}
    assert len(warn.params["dropped_empty_columns"]) == 1


def test_single_column_file_and_pipe_and_tab():
    assert list(_csv("v\n1\n2\n3\n").df.columns) == ["v"]
    assert _csv("a|b\n1|2\n3|4\n").source_info.delimiter == "|"
    assert _csv("a\tb\n1\t2\n3\t4\n").source_info.delimiter == "\t"


def test_bom_and_crlf():
    res = load_file(b"\xef\xbb\xbfa,b\r\n1,2\r\n3,4\r\n")
    assert res.source_info.encoding == "utf-8-sig"
    assert list(res.df.columns) == ["a", "b"]


def test_quoted_field_with_embedded_delimiter_and_newline():
    res = _csv('a,b\n"x,y",1\n"line1\nline2",2\n')
    assert list(res.df["a"]) == ["x,y", "line1\nline2"]


def test_log_round_trips_through_json(variants):
    res = load_file(variants["question_marks"].data)
    log = PipelineLog()
    log.extend(res.log)
    assert PipelineLog.from_json(log.to_json()).entries == log.entries


def test_marker_spellings_in_any_case():
    res = _csv("a,b\n1,NuLl\n2,n/A\n3,Na\n4,nA\n5,x\n")
    assert res.df["b"].isna().sum() == 4 and res.df["b"].iloc[4] == "x"
    assert res.log[1].params["per_column"] == {"b": 4}


def test_mixed_text_and_number_cells_in_excel_still_convert(variants):
    # the xlsx variant stores temperature as text and power as numbers with "?" cells
    res = load_file(variants["xlsx"].data)
    assert pd.api.types.is_numeric_dtype(res.df["temperature"])
    assert pd.api.types.is_numeric_dtype(res.df["power_kw"])
