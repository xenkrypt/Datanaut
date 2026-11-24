"""File loader: reads CSV and Excel (xlsx) data into a DataFrame and logs what it did.

Format is detected from the file content, never from the extension alone. Nothing is dropped
silently: empty rows, ragged rows, renamed columns and values that could not be parsed are all
reported in the returned log entries.
"""

from __future__ import annotations

import codecs
import csv
import io
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time
from itertools import islice, product
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import openpyxl
import pandas as pd
from charset_normalizer import from_bytes
from pandas.api.types import infer_dtype
from pydantic import BaseModel, Field

from datanaut.core.log import PipelineLog, StepLog
from datanaut.errors import FileTooLargeError, ParseError, UnsupportedFormatError

MISSING_MARKERS: frozenset[str] = frozenset({"", "?", "na", "n/a", "null", "-", "--"})
"""Lower-cased cell texts treated as missing (compared after stripping whitespace)."""

_MARKER_SPELLINGS: frozenset[str] = frozenset(
    "".join(chars)
    for marker in MISSING_MARKERS
    for chars in product(*[(c.lower(), c.upper()) for c in marker])
)
"""Every upper/lower-case spelling of the markers, so matching needs no per-cell lowercasing."""

NUMERIC_THRESHOLD = 0.95
"""Share of non-missing values that must convert for a text column to become numeric."""

_DELIMITERS = (",", ";", "\t", "|")
# Encodings of real-world CSV exports. Old DOS and Mac code pages are left out on purpose:
# on short Western text charset-normalizer otherwise tends to guess them wrongly.
_LIKELY_ENCODINGS = (
    "cp1252",
    "latin_1",
    "iso8859_15",
    "cp1250",
    "cp1251",
    "cp1253",
    "cp1254",
    "cp1256",
    "koi8_r",
    "gb18030",
    "big5",
    "shift_jis",
    "euc_kr",
)
_SAMPLE_CHARS = 100_000
_SAMPLE_ROWS = 500
_ZIP_MAGIC = b"PK\x03\x04"
_OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
_BINARY_MAGICS = (b"%PDF", b"\x89PNG", b"GIF8", b"\xff\xd8\xff", b"\x1f\x8b", b"MZ", b"\x7fELF")

_INT_US = r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)"
_INT_EU = r"(?:0|[1-9]\d*|[1-9]\d{0,2}(?:\.\d{3})+)"
_EXP = r"(?:[eE][+-]?\d+)?"
_NUM_US = re.compile(rf"[+-]?(?:{_INT_US}(?:\.\d+)?|\.\d+){_EXP}")
_NUM_EU = re.compile(rf"[+-]?(?:{_INT_EU}(?:,\d+)?|,\d+){_EXP}")


class SourceInfo(BaseModel):
    """Facts about the loaded file."""

    filename: str | None = None
    format: Literal["csv", "xlsx"]
    encoding: str | None = None
    delimiter: str | None = None
    sheet_names: list[str] = Field(default_factory=list)
    sheet: str | None = None
    size_bytes: int


@dataclass(frozen=True)
class LoadResult:
    """Result of load_file: the data, facts about the source and the log entries."""

    df: pd.DataFrame
    source_info: SourceInfo
    log: list[StepLog]


@dataclass
class _Parsed:
    frame: pd.DataFrame  # object cells, RangeIndex columns
    header: list[str]
    rows_in_file: int
    empty_rows: int = 0
    short_rows: int = 0
    long_rows: int = 0
    bad_row_examples: list[int] = field(default_factory=list)
    encoding: str | None = None
    delimiter: str | None = None
    sheet_names: list[str] = field(default_factory=list)
    sheet: str | None = None


@dataclass
class _Numeric:
    values: pd.Series
    fmt: Literal["us", "eu"]
    matched: int
    unparseable: int
    ambiguous: int
    examples: list[str]


def load_file(
    source: str | os.PathLike[str] | bytes | bytearray | BinaryIO,
    filename: str | None = None,
    max_bytes: int = 50_000_000,
    *,
    sheet: str | int | None = None,
    max_unpacked_bytes: int | None = None,
) -> LoadResult:
    """Load a CSV or xlsx file from a path, bytes or binary file-like object.

    `sheet` selects an Excel sheet by name or position (default: the first).
    `max_unpacked_bytes` caps the unpacked size of an xlsx archive (default: 10 x max_bytes).
    """
    raw, name = _read_source(source, filename, max_bytes)
    fmt = _detect_format(raw)
    if fmt == "xlsx":
        parsed = _parse_xlsx(raw, sheet, max_unpacked_bytes or 10 * max_bytes)
    else:
        parsed = _parse_csv(raw)
    df, missing_info, numeric_info, renamed, dropped_cols = _clean_frame(parsed, fmt == "xlsx")
    info = SourceInfo(
        filename=name,
        format=fmt,
        encoding=parsed.encoding,
        delimiter=parsed.delimiter,
        sheet_names=parsed.sheet_names,
        sheet=parsed.sheet,
        size_bytes=len(raw),
    )
    log = _build_log(info, parsed, df, missing_info, numeric_info, renamed, dropped_cols)
    return LoadResult(df=df, source_info=info, log=log)


# --------------------------------------------------------------------------- reading the source


def _read_source(
    source: str | os.PathLike[str] | bytes | bytearray | BinaryIO,
    filename: str | None,
    max_bytes: int,
) -> tuple[bytes, str | None]:
    limit_msg = f"This file is larger than the {max_bytes / 1_000_000:g} MB limit."
    if isinstance(source, bytes | bytearray | memoryview):
        raw = bytes(source)
        name = filename
    elif isinstance(source, str | os.PathLike):
        path = Path(source)
        try:
            if path.stat().st_size > max_bytes:
                raise FileTooLargeError(limit_msg)
            raw = path.read_bytes()
        except OSError as exc:
            raise ParseError(f"The file '{path.name}' could not be opened.") from exc
        name = filename or path.name
    elif hasattr(source, "read"):
        raw = source.read(max_bytes + 1)
        if not isinstance(raw, bytes | bytearray):
            raise ParseError("The file must be opened in binary mode.")
        raw = bytes(raw)
        name = filename or _basename(getattr(source, "name", None))
    else:
        raise ParseError("Unsupported source: pass a path, bytes or a binary file object.")
    if len(raw) > max_bytes:
        raise FileTooLargeError(limit_msg)
    return raw, name


def _basename(value: object) -> str | None:
    return Path(value).name if isinstance(value, str | os.PathLike) else None


def _detect_format(raw: bytes) -> Literal["csv", "xlsx"]:
    if not raw.strip():
        raise ParseError("The file is empty.")
    if raw.startswith(_ZIP_MAGIC):
        return "xlsx"
    if raw.startswith(_OLE_MAGIC):
        raise UnsupportedFormatError(
            "Legacy .xls files are not supported. Save the file as .xlsx or CSV and try again."
        )
    if raw.startswith(_BINARY_MAGICS):
        raise UnsupportedFormatError()
    utf16 = raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE))
    head = raw[:8192]
    if not utf16:
        control = sum(1 for b in head if b < 32 and b not in (9, 10, 13, 12))
        if b"\x00" in head or control / len(head) > 0.05:
            raise UnsupportedFormatError()
    return "csv"


# ------------------------------------------------------------------------------------------ CSV


def _decode(raw: bytes) -> tuple[str, str]:
    if raw.startswith(codecs.BOM_UTF8):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16"), "utf-16"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    match = from_bytes(raw, cp_isolation=list(_LIKELY_ENCODINGS)).best()
    if match is not None and match.encoding:
        try:
            return raw.decode(match.encoding), match.encoding
        except (UnicodeDecodeError, LookupError):
            pass
    return raw.decode("latin-1"), "latin-1"


def _score_delimiter(sample: str, delimiter: str) -> tuple[int, float, bool] | None:
    """Return (modal field count, share of rows with it, header matches) or None."""
    try:
        reader = csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter)
        rows = [r for r in islice(reader, _SAMPLE_ROWS) if r]
    except csv.Error:
        return None
    if not rows:
        return None
    modal, n = Counter(len(r) for r in rows).most_common(1)[0]
    return modal, n / len(rows), len(rows[0]) == modal


def _is_valid(score: tuple[int, float, bool] | None) -> bool:
    return score is not None and score[0] >= 2 and score[1] >= 0.95 and score[2]


def _detect_delimiter(text: str) -> str:
    """Sniff the delimiter, verify it, and otherwise try , ; tab | in that order."""
    sample = text[:_SAMPLE_CHARS]
    if len(text) > _SAMPLE_CHARS and "\n" in sample:
        sample = sample[: sample.rindex("\n")]
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS)).delimiter
    except csv.Error:
        sniffed = None
    if sniffed and _is_valid(_score_delimiter(sample, sniffed)):
        return sniffed
    scores = {d: _score_delimiter(sample, d) for d in _DELIMITERS}
    for d in _DELIMITERS:
        if _is_valid(scores[d]):
            return d
    usable = {d: s for d, s in scores.items() if s is not None and s[0] >= 2}
    if usable:
        return max(usable, key=lambda d: (usable[d][1], usable[d][0]))
    return ","


def _parse_csv(raw: bytes) -> _Parsed:
    text, encoding = _decode(raw)
    delimiter = _detect_delimiter(text)
    header: list[str] | None = None
    rows: list[list[str | None]] = []
    empty = short = long = 0
    examples: list[int] = []
    try:
        for fields in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter):
            if not any(f.strip() for f in fields):
                if header is not None:
                    empty += 1
                continue
            if header is None:
                header = [f.strip() for f in fields]
                continue
            width = len(header)
            if len(fields) < width:
                fields = [*fields, *([None] * (width - len(fields)))]
                short += 1
                examples.append(len(rows) + 1)
            elif len(fields) > width:
                fields = [*fields[: width - 1], delimiter.join(fields[width - 1 :])]
                long += 1
                examples.append(len(rows) + 1)
            rows.append(fields)
    except csv.Error as exc:
        raise ParseError(f"The CSV file could not be parsed: {exc}") from exc
    if header is None:
        raise ParseError("The file is empty.")
    if not rows:
        raise ParseError("The file has a header row but no data rows.")
    return _Parsed(
        frame=pd.DataFrame(rows, dtype=object),
        header=header,
        rows_in_file=len(rows) + empty,
        empty_rows=empty,
        short_rows=short,
        long_rows=long,
        bad_row_examples=examples[:5],
        encoding=encoding,
        delimiter=delimiter,
    )


# ----------------------------------------------------------------------------------------- xlsx


def _parse_xlsx(raw: bytes, sheet: str | int | None, max_unpacked: int) -> _Parsed:
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ParseError("The Excel file is corrupted and could not be opened.") from exc
    with archive:
        if "xl/workbook.xml" not in archive.namelist():
            raise UnsupportedFormatError(
                "This archive is not an Excel workbook. Please upload a CSV or Excel (.xlsx) file."
            )
        if sum(i.file_size for i in archive.infolist()) > max_unpacked:
            raise FileTooLargeError(
                "This Excel file expands to more than the allowed unpacked size."
            )
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        names = list(workbook.sheetnames)
        workbook.close()
        chosen = _choose_sheet(names, sheet)
        frame = pd.read_excel(
            io.BytesIO(raw),
            sheet_name=chosen,
            header=None,
            dtype=object,
            engine="openpyxl",
            keep_default_na=False,
        )
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError("The Excel file could not be read.") from exc
    frame = frame.map(_excel_cell)
    blank = frame.map(_is_blank).all(axis=1).to_numpy()
    if blank.all():
        raise ParseError("The selected sheet is empty.")
    first = int(np.argmax(~blank))
    header = ["" if _is_blank(v) else str(v).strip() for v in frame.iloc[first]]
    body = frame.iloc[first + 1 :]
    body_blank = blank[first + 1 :]
    data = body.loc[~body_blank].reset_index(drop=True)
    data.columns = range(data.shape[1])
    if data.empty:
        raise ParseError("The selected sheet has a header row but no data rows.")
    return _Parsed(
        frame=data,
        header=header,
        rows_in_file=len(body),
        empty_rows=int(body_blank.sum()),
        sheet_names=names,
        sheet=chosen,
    )


def _choose_sheet(names: list[str], sheet: str | int | None) -> str:
    if sheet is None:
        return names[0]
    if isinstance(sheet, int):
        if not -len(names) <= sheet < len(names):
            raise ParseError(f"Sheet number {sheet} does not exist. Sheets: {', '.join(names)}.")
        return names[sheet]
    if sheet not in names:
        raise ParseError(f"Sheet '{sheet}' was not found. Sheets: {', '.join(names)}.")
    return sheet


def _excel_cell(value: Any) -> Any:
    if isinstance(value, time):
        return value.isoformat()
    return value


# ------------------------------------------------------------------------------ cleaning columns


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float):
        return bool(np.isnan(value))
    return bool(value is pd.NaT)


def _unique_names(header: list[str]) -> tuple[list[str], dict[str, str]]:
    names: list[str] = []
    seen: set[str] = set()
    renamed: dict[str, str] = {}
    for i, original in enumerate(header):
        base = original or f"unnamed_{i}"
        candidate, k = base, 0
        while candidate in seen:
            k += 1
            candidate = f"{base}.{k}"
        seen.add(candidate)
        names.append(candidate)
        if candidate != original:
            renamed[candidate] = original
    return names, renamed


def _normalize_missing(s: pd.Series) -> tuple[pd.Series, int]:
    """Strip text, turn missing markers into NaN, and return how many markers were found."""
    present = s.dropna()
    if present.empty or infer_dtype(present, skipna=True) != "string":  # mixed cells: slow path
        stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
        lowered = stripped.map(lambda v: v.lower() if isinstance(v, str) else v)
        is_marker = lowered.isin(MISSING_MARKERS)
        blank = lowered.map(lambda v: isinstance(v, str) and v == "")
    else:
        stripped = s.str.strip()
        is_marker = stripped.isin(_MARKER_SPELLINGS)
        blank = stripped == ""
    cleaned = stripped.mask(is_marker | stripped.isna(), np.nan)
    return cleaned, int((is_marker & ~blank).sum())


def _numeric_masks(s: pd.Series) -> tuple[pd.Series, pd.Series] | None:
    """Which values fit decimal-point and decimal-comma numbers; None if clearly not numeric."""
    present = s.dropna()
    if present.empty:
        return None
    sample = present.iloc[np.linspace(0, len(present) - 1, min(len(present), 500)).astype(int)]
    fits = sample.str.fullmatch(_NUM_US, na=False) | sample.str.fullmatch(_NUM_EU, na=False)
    if fits.mean() < 0.5:  # a quick look is enough to rule out text columns
        return None
    return s.str.fullmatch(_NUM_US, na=False), s.str.fullmatch(_NUM_EU, na=False)


def _has_separator(s: pd.Series, mask: pd.Series) -> int:
    return int(s[mask].str.contains(r"[.,]", regex=True, na=False).sum())


def _separator_votes(s: pd.Series, masks: tuple[pd.Series, pd.Series]) -> tuple[int, int]:
    """Count values with a separator that only fit decimal-point (us) or decimal-comma (eu)."""
    us, eu = masks
    return _has_separator(s, us & ~eu), _has_separator(s, eu & ~us)


def _convert_numeric(
    s: pd.Series,
    masks: tuple[pd.Series, pd.Series],
    prefer: Literal["us", "eu"] = "us",
) -> _Numeric | None:
    """Convert a text column to numbers if at least NUMERIC_THRESHOLD of its values fit.

    When a column fits both conventions equally (for example "1,234"), `prefer` decides.
    """
    present = s.notna()
    n_present = int(present.sum())
    if n_present == 0:
        return None
    us, eu = masks
    n_us, n_eu = int(us.sum()), int(eu.sum())
    if n_us == n_eu:
        mask, fmt = (us, "us") if prefer == "us" else (eu, "eu")
    else:
        mask, fmt = (us, "us") if n_us > n_eu else (eu, "eu")
    matched = max(n_us, n_eu)
    if matched / n_present < NUMERIC_THRESHOLD:
        return None
    ambiguous = _has_separator(s, us & eu)
    kept = s.where(mask)
    if fmt == "us":
        normalized = kept.str.replace(",", "", regex=False)
    else:
        normalized = kept.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return _Numeric(
        values=pd.to_numeric(normalized, errors="coerce"),
        fmt=fmt,
        matched=matched,
        unparseable=n_present - matched,
        ambiguous=ambiguous,
        examples=[str(v) for v in s[present & ~mask].head(5)],
    )


def _num_to_text(value: Any) -> Any:
    if isinstance(value, bool) or pd.isna(value) or isinstance(value, str):
        return value
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return str(value)


def _kind(values: pd.Series) -> str:
    """Classify non-missing cells: 'text' (any str, no dates), 'number', 'datetime', 'other'."""
    present = values.dropna()
    if present.empty:
        return "empty"
    inferred = infer_dtype(present, skipna=True)
    if inferred == "string":
        return "text"
    if inferred in ("integer", "floating", "mixed-integer-float"):
        return "number"
    if inferred in ("datetime", "datetime64", "date"):
        return "datetime"
    if inferred in ("mixed", "mixed-integer"):
        return _kind_mixed(present)
    return "other"


def _kind_mixed(present: pd.Series) -> str:
    """Slow path for columns mixing strings, numbers and dates (typically Excel)."""
    is_str = present.map(lambda v: isinstance(v, str))
    is_dt = present.map(lambda v: isinstance(v, datetime | date))
    is_num = present.map(
        lambda v: isinstance(v, int | float | np.number) and not isinstance(v, bool)
    )
    if is_dt.all():
        return "datetime"
    if is_num.all():
        return "number"
    if (is_str | is_num).all() and is_str.any():
        return "text"
    return "other"


def _clean_frame(
    parsed: _Parsed, is_excel: bool
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[str, dict[str, Any]],
    dict[str, str],
    list[str],
]:
    frame = parsed.frame
    header = list(parsed.header)
    width = frame.shape[1]
    header = (header + [""] * width)[:width]
    drop = [
        j for j in range(width) if not header[j] and bool(frame.iloc[:, j].map(_is_blank).all())
    ]
    keep = [j for j in range(width) if j not in drop]
    dropped_cols = [f"column {j + 1}" for j in drop]
    names, renamed = _unique_names([header[j] for j in keep])

    columns: dict[str, pd.Series] = {}
    missing_info: dict[str, int] = {}
    numeric_info: dict[str, dict[str, Any]] = {}
    prepared: dict[str, tuple[str, pd.Series, tuple[pd.Series, pd.Series] | None]] = {}
    for name, j in zip(names, keep, strict=True):
        cleaned, markers = _normalize_missing(frame.iloc[:, j].reset_index(drop=True))
        if markers:
            missing_info[name] = markers
        kind = _kind(cleaned)
        masks = None
        if kind == "text":
            if infer_dtype(cleaned.dropna(), skipna=True) != "string":
                cleaned = cleaned.map(_num_to_text).astype(object)  # numbers among the text
            masks = _numeric_masks(cleaned)
        prepared[name] = (kind, cleaned, masks)
    # Columns like "110,457" are ambiguous alone; the rest of the file usually settles them.
    votes = [_separator_votes(s, m) for _, s, m in prepared.values() if m is not None]
    prefer: Literal["us", "eu"] = (
        "eu" if sum(v[1] for v in votes) > sum(v[0] for v in votes) else "us"
    )

    for name, (kind, cleaned, masks) in prepared.items():
        if kind == "datetime" and is_excel:
            try:
                columns[name] = pd.to_datetime(cleaned)
                continue
            except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime):
                pass
        elif kind == "number":
            columns[name] = pd.to_numeric(cleaned)
            continue
        elif kind == "text":
            outcome = _convert_numeric(cleaned, masks, prefer) if masks is not None else None
            if outcome is not None:
                columns[name] = outcome.values
                numeric_info[name] = {
                    "format": "decimal comma" if outcome.fmt == "eu" else "decimal point",
                    "converted": outcome.matched,
                    "unparseable_set_missing": outcome.unparseable,
                    "ambiguous_separator_values": outcome.ambiguous,
                    "unparseable_examples": outcome.examples,
                }
                continue
        columns[name] = cleaned
    df = pd.DataFrame(columns, index=pd.RangeIndex(len(frame)))
    df.columns = names
    return df, missing_info, numeric_info, renamed, dropped_cols


# ------------------------------------------------------------------------------------- the log


def _build_log(
    info: SourceInfo,
    parsed: _Parsed,
    df: pd.DataFrame,
    missing_info: dict[str, int],
    numeric_info: dict[str, dict[str, Any]],
    renamed: dict[str, str],
    dropped_cols: list[str],
) -> list[StepLog]:
    log = PipelineLog()
    where = f"'{info.filename}'" if info.filename else "the upload"
    if info.format == "csv":
        detail = f"encoding {info.encoding}, delimiter {info.delimiter!r}"
    else:
        detail = f"sheet '{info.sheet}'"
    log.append(
        "load",
        f"Loaded {len(df)} rows x {df.shape[1]} columns from {info.format.upper()} file "
        f"{where} ({detail}).",
        params=info.model_dump(),
        counts={"rows": len(df), "columns": df.shape[1], "size_bytes": info.size_bytes},
        rows_before=parsed.rows_in_file,
        rows_after=len(df),
    )

    total_markers = sum(missing_info.values())
    log.append(
        "normalize_missing",
        f"Treated {total_markers} cell(s) with missing-value markers as missing.",
        params={"markers": sorted(MISSING_MARKERS - {""}), "per_column": missing_info},
        counts={"cells_set_missing": total_markers},
        rows_before=len(df),
        rows_after=len(df),
    )

    converted = sum(v["converted"] for v in numeric_info.values())
    unparseable = sum(v["unparseable_set_missing"] for v in numeric_info.values())
    log.append(
        "convert_numeric",
        f"Converted {converted} value(s) in {len(numeric_info)} text column(s) to numbers"
        + (
            f"; {unparseable} value(s) could not be parsed and were set to missing."
            if unparseable
            else "."
        ),
        params={"threshold": NUMERIC_THRESHOLD, "per_column": numeric_info},
        counts={
            "columns_converted": len(numeric_info),
            "values_converted": converted,
            "values_unparseable": unparseable,
        },
        rows_before=len(df),
        rows_after=len(df),
    )

    issues: list[str] = []
    if parsed.empty_rows:
        issues.append(f"{parsed.empty_rows} empty row(s) removed")
    if parsed.short_rows:
        issues.append(f"{parsed.short_rows} row(s) with too few fields padded with missing values")
    if parsed.long_rows:
        issues.append(f"{parsed.long_rows} row(s) with extra fields merged into the last column")
    if renamed:
        issues.append(f"{len(renamed)} column name(s) cleaned or made unique")
    if dropped_cols:
        issues.append(f"{len(dropped_cols)} empty unnamed column(s) removed")
    if issues:
        log.append(
            "parse_warnings",
            "Reading the file needed adjustments: " + "; ".join(issues) + ".",
            params={
                "renamed_columns": renamed,
                "dropped_empty_columns": dropped_cols,
                "example_data_rows": parsed.bad_row_examples,
            },
            counts={
                "empty_rows_removed": parsed.empty_rows,
                "short_rows": parsed.short_rows,
                "long_rows": parsed.long_rows,
            },
            read_only=True,
        )
    return log.entries
