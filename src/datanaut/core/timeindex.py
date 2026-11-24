"""Time index handling: find timestamp and value columns, parse, sort and index.

Nothing is dropped silently. Rows whose timestamp is missing or cannot be parsed are set aside
in `TimeIndexResult.rejected` (with counts and examples in the log), and rows moved by sorting
are counted. If day-first and month-first readings both parse and disagree, nothing is guessed:
the result is flagged ambiguous with a preview of both readings.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from datanaut.core.log import PipelineLog, StepLog
from datanaut.errors import ParseError

MIN_PARSE_SHARE = 0.5
"""A column must parse as timestamps at least this often to be listed as a candidate."""

_SAMPLE_SIZE = 2000
_NAME_HINT = re.compile(
    r"date|time|stamp|(^|[_\W])ts($|[_\W])|period|day|_at$|created|recorded|logged|measured",
    re.IGNORECASE,
)
_ID_NAME = re.compile(
    r"(^|[_\s-])(id|uuid|guid|index|idx)$|^(id|uuid|guid|index|idx|row|rownum|row_number|record)"
    r"([_\s-]|$)",
    re.IGNORECASE,
)
_NUMERIC_TEXT = re.compile(r"[+-]?\d+(?:\.\d+)?")
_TIME_ONLY = re.compile(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?\s*(?:[AaPp]\.?[Mm]\.?)?")
_ISO_FULL = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:Z|[+-]\d{2}(?::?\d{2})?)?)?"
)
_YEAR_FIRST = re.compile(r"\d{4}[-/.]")


class TimeCandidate(BaseModel):
    """A column (or date + time column pair) that looks like timestamps."""

    columns: list[str]
    kind: Literal["single", "date_time"]
    parse_share: float
    name_hint: bool


class TimeChoices(BaseModel):
    """What build_time_index decided or was told."""

    time_columns: list[str]
    dayfirst: bool | None = None  # None: not needed (ISO text, real datetimes or no difference)
    dayfirst_source: Literal["user", "auto", "not_needed"] = "not_needed"
    timezone: str | None = None  # None means timezone-naive
    index_name: str = "timestamp"


class AmbiguousSample(BaseModel):
    """One value whose day-first and month-first readings differ."""

    raw: str
    dayfirst: str
    monthfirst: str


class TimeFlags(BaseModel):
    """Things the caller should know about."""

    index_set: bool = True
    ambiguous: bool = False
    ambiguous_preview: list[AmbiguousSample] = Field(default_factory=list)
    ambiguous_values: int = 0
    missing_timestamp_rows: int = 0
    unparseable_rows: int = 0
    unparseable_examples: list[str] = Field(default_factory=list)
    rows_moved_by_sort: int = 0
    duplicate_timestamps: int = 0
    mixed_timezone_offsets: bool = False


@dataclass(frozen=True)
class TimeIndexResult:
    """Result of build_time_index.

    `df` has a DatetimeIndex unless `flags.ambiguous` is set, in which case it is an unchanged
    copy of the input. `rejected` holds the original rows that had no usable timestamp.
    """

    df: pd.DataFrame
    choices: TimeChoices
    flags: TimeFlags
    rejected: pd.DataFrame
    log: list[StepLog]


# ------------------------------------------------------------------------------- detection


def _is_text_like(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def _sample_positions(n: int) -> np.ndarray:
    if n <= _SAMPLE_SIZE:
        return np.arange(n)
    return np.linspace(0, n - 1, _SAMPLE_SIZE).astype(int)


def _text(s: pd.Series, part: Literal["date", "time", "full"]) -> pd.Series:
    """Return a column as stripped text (datetime columns are formatted)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        fmt = {"date": "%Y-%m-%d", "time": "%H:%M:%S", "full": "%Y-%m-%d %H:%M:%S"}[part]
        return s.dt.strftime(fmt).astype(object)
    return s.astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)


def _parse_share(values: pd.Series) -> float:
    """Share of non-missing values that parse, under the better of the two readings."""
    present = values.dropna()
    if present.empty:
        return 0.0
    return max(float(_parse(values, d)[0].notna().sum()) for d in (False, True)) / len(present)


def _column_role(s: pd.Series, positions: np.ndarray) -> tuple[str | None, float]:
    """Classify a column as 'datetime', 'date', 'time' or None, with its parse share."""
    sample = s.iloc[positions]
    if pd.api.types.is_datetime64_any_dtype(s):
        present = sample.dropna()
        if present.empty:
            return None, 0.0
        has_time = bool((present != present.dt.normalize()).any())
        return ("datetime" if has_time else "date"), 1.0
    if not _is_text_like(s):
        return None, 0.0
    text = _text(sample, "full").dropna()
    text = text[text.map(lambda v: isinstance(v, str) and v != "")]
    if text.empty:
        return None, 0.0
    if text.str.fullmatch(_NUMERIC_TEXT).mean() >= 0.9:
        return None, 0.0
    if text.str.fullmatch(_TIME_ONLY).mean() >= 0.9:
        return "time", 1.0
    share = _parse_share(text)
    if share < MIN_PARSE_SHARE:
        return None, share
    has_clock = text.str.contains(":", regex=False).mean() > 0.5
    return ("datetime" if has_clock else "date"), share


def detect_time_columns(df: pd.DataFrame) -> list[TimeCandidate]:
    """Rank columns (or date + time pairs) by how many values parse as timestamps.

    Column-name hints break ties. A date-only column that has a separate time column is
    listed as a (date, time) pair rather than on its own.
    """
    positions = _sample_positions(len(df))
    roles: dict[str, tuple[str | None, float]] = {
        str(c): _column_role(df[c], positions) for c in df.columns
    }
    order = {str(c): i for i, c in enumerate(df.columns)}
    singles: list[TimeCandidate] = []
    pairs: list[TimeCandidate] = []
    paired_dates: set[str] = set()

    date_cols = [c for c, (r, _) in roles.items() if r == "date"]
    time_cols = [c for c, (r, _) in roles.items() if r == "time"]
    for d in date_cols:
        for t in time_cols:
            combined = _text(df[d].iloc[positions], "date").str.cat(
                _text(df[t].iloc[positions], "time"), sep=" "
            )
            share = _parse_share(combined)
            if share >= MIN_PARSE_SHARE:
                paired_dates.add(d)
                pairs.append(
                    TimeCandidate(
                        columns=[d, t],
                        kind="date_time",
                        parse_share=share,
                        name_hint=bool(_NAME_HINT.search(d) or _NAME_HINT.search(t)),
                    )
                )
    for c, (role, share) in roles.items():
        if role in ("date", "datetime") and c not in paired_dates:
            singles.append(
                TimeCandidate(
                    columns=[c],
                    kind="single",
                    parse_share=share,
                    name_hint=bool(_NAME_HINT.search(c)),
                )
            )
    return sorted(
        [*pairs, *singles],
        key=lambda c: (-round(c.parse_share, 6), not c.name_hint, order[c.columns[0]]),
    )


def detect_value_columns(df: pd.DataFrame, time_cols: Sequence[str] = ()) -> list[str]:
    """Numeric columns that could hold readings.

    Excludes the time columns, booleans, constant or empty columns, and obvious identifiers
    (names such as id or index, or an integer counter that is unique in every row).
    """
    skip = set(time_cols)
    out: list[str] = []
    for col in df.columns:
        s = df[col]
        if col in skip or not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        present = s.dropna()
        if present.nunique() <= 1 or _ID_NAME.search(str(col)):
            continue
        if pd.api.types.is_integer_dtype(s) and present.is_unique and len(present) > 2:
            if present.is_monotonic_increasing and bool((present.diff().dropna() == 1).all()):
                continue
        out.append(str(col))
    return out


# ------------------------------------------------------------------------------- parsing


def _parse(values: pd.Series, dayfirst: bool) -> tuple[pd.Series, bool]:
    """Parse text to datetimes; unparseable values become NaT. Returns (parsed, mixed_offsets)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values, False
    present = values.dropna().astype(str)
    # A majority decides the column's format; the odd value out is reported as unparseable.
    is_iso = bool(len(present)) and present.str.fullmatch(_ISO_FULL).mean() >= 0.5
    year_first = bool(len(present)) and present.str.match(_YEAR_FIRST).mean() >= 0.5
    # dayfirst would swap month and day in ISO and other year-first dates, so it is not applied
    kwargs: dict[str, Any] = (
        {"format": "ISO8601"} if is_iso else {"dayfirst": dayfirst and not year_first}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(values, errors="coerce", **kwargs), False
        except ValueError:  # mixed UTC offsets cannot share one dtype
            try:
                return pd.to_datetime(values, errors="coerce", utc=True, **kwargs), True
            except (ValueError, OverflowError):
                pass
        except OverflowError:
            pass
    return pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]"), False


def _combine(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if len(cols) == 1:
        s = df[cols[0]]
        return s if pd.api.types.is_datetime64_any_dtype(s) else _text(s, "full")
    return _text(df[cols[0]], "date").str.cat(_text(df[cols[1]], "time"), sep=" ")


def _fmt(ts: Any) -> str:
    return "NaT" if pd.isna(ts) else str(ts)


def build_time_index(
    df: pd.DataFrame,
    time_col: Any,
    dayfirst: bool | None = None,
) -> TimeIndexResult:
    """Parse the time column(s), sort chronologically and set a DatetimeIndex.

    `time_col` is one column, or (date_column, time_column) to combine. `dayfirst` overrides
    the day-first / month-first detection. The input DataFrame is never modified.
    """
    cols = list(time_col) if isinstance(time_col, list | tuple) else [time_col]
    if not 1 <= len(cols) <= 2:
        raise ParseError("Give one timestamp column, or a date column and a time column.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ParseError(f"Column '{missing[0]}' was not found in the data.")

    values = _combine(df, cols)
    n_rows = len(df)
    flags = TimeFlags()
    choices = TimeChoices(time_columns=[str(c) for c in cols], index_name=_index_name(df, cols))

    if pd.api.types.is_datetime64_any_dtype(values):
        parsed, mixed = values, False
    elif dayfirst is not None:
        parsed, mixed = _parse(values, dayfirst)
        choices = choices.model_copy(update={"dayfirst": dayfirst, "dayfirst_source": "user"})
    else:
        parsed_df, mixed_df = _parse(values, True)
        parsed_mf, mixed_mf = _parse(values, False)
        n_df, n_mf = int(parsed_df.notna().sum()), int(parsed_mf.notna().sum())
        if n_df > n_mf:
            parsed, mixed, pick = parsed_df, mixed_df, True
        elif n_mf > n_df:
            parsed, mixed, pick = parsed_mf, mixed_mf, False
        else:
            differ = parsed_df.notna() & parsed_mf.notna() & _differs(parsed_df, parsed_mf)
            if differ.any():
                return _ambiguous_result(df, values, parsed_df, parsed_mf, differ, choices, cols)
            parsed, mixed, pick = parsed_mf, mixed_mf, None
        if pick is None:
            choices = choices.model_copy(update={"dayfirst": None, "dayfirst_source": "not_needed"})
        else:
            choices = choices.model_copy(update={"dayfirst": pick, "dayfirst_source": "auto"})

    if int(parsed.notna().sum()) == 0:
        raise ParseError(
            "None of the values in "
            + " + ".join(f"'{c}'" for c in cols)
            + " could be read as dates or times."
        )

    tz = getattr(parsed.dt, "tz", None)
    choices = choices.model_copy(update={"timezone": None if tz is None else str(tz)})

    source_missing = values.isna().to_numpy()
    bad = parsed.isna().to_numpy()
    unparseable = bad & ~source_missing
    flags = flags.model_copy(
        update={
            "mixed_timezone_offsets": mixed,
            "missing_timestamp_rows": int(source_missing.sum()),
            "unparseable_rows": int(unparseable.sum()),
            "unparseable_examples": [str(v) for v in values[unparseable].head(5)],
        }
    )
    rejected = df.loc[bad].copy()
    kept = df.loc[~bad].drop(columns=cols)
    stamps = parsed[~bad]

    order = np.argsort(stamps.array.asi8, kind="stable")
    moved = int((order != np.arange(len(order))).sum())
    result = kept.iloc[order]
    result.index = pd.DatetimeIndex(stamps.iloc[order].array, name=choices.index_name)
    flags = flags.model_copy(
        update={
            "rows_moved_by_sort": moved,
            "duplicate_timestamps": int(result.index.duplicated().sum()),
        }
    )
    return TimeIndexResult(
        df=result,
        choices=choices,
        flags=flags,
        rejected=rejected,
        log=_build_log(choices, flags, n_rows, len(result)),
    )


def _differs(a: pd.Series, b: pd.Series) -> pd.Series:
    if a.dt.tz is not None or b.dt.tz is not None:
        return pd.Series(a.array.asi8 != b.array.asi8, index=a.index)
    return a != b


def _index_name(df: pd.DataFrame, cols: list[str]) -> str:
    if len(cols) == 1:
        return str(cols[0])
    remaining = set(df.columns) - set(cols)
    name = "timestamp"
    while name in remaining:
        name += "_index"
    return name


def _ambiguous_result(
    df: pd.DataFrame,
    values: pd.Series,
    parsed_df: pd.Series,
    parsed_mf: pd.Series,
    differ: pd.Series,
    choices: TimeChoices,
    cols: list[str],
) -> TimeIndexResult:
    first = np.flatnonzero(differ.to_numpy())[:3]
    preview = [
        AmbiguousSample(
            raw=str(values.iloc[i]),
            dayfirst=_fmt(parsed_df.iloc[i]),
            monthfirst=_fmt(parsed_mf.iloc[i]),
        )
        for i in first
    ]
    flags = TimeFlags(
        index_set=False,
        ambiguous=True,
        ambiguous_preview=preview,
        ambiguous_values=int(differ.sum()),
    )
    log = PipelineLog()
    log.append(
        "build_time_index",
        f"Could not choose between day-first and month-first for {' + '.join(cols)}: "
        f"{flags.ambiguous_values} value(s) read differently, so no time index was set. "
        "Pass dayfirst=True or False to decide.",
        params={"time_columns": cols, "ambiguous_preview": [p.model_dump() for p in preview]},
        counts={"ambiguous_values": flags.ambiguous_values},
        rows_before=len(df),
        rows_after=len(df),
        read_only=True,
    )
    return TimeIndexResult(
        df=df.copy(),
        choices=choices,
        flags=flags,
        rejected=df.iloc[0:0].copy(),
        log=log.entries,
    )


def _build_log(
    choices: TimeChoices, flags: TimeFlags, rows_before: int, rows_after: int
) -> list[StepLog]:
    log = PipelineLog()
    cols = " + ".join(f"'{c}'" for c in choices.time_columns)
    reading = {
        "user": f"{'day-first' if choices.dayfirst else 'month-first'} (chosen by user)",
        "auto": f"{'day-first' if choices.dayfirst else 'month-first'} (detected)",
        "not_needed": "no day/month ambiguity",
    }[choices.dayfirst_source]
    tz = f"timezone {choices.timezone}" if choices.timezone else "timezone-naive"
    log.append(
        "build_time_index",
        f"Parsed {cols} as the time index ({reading}; {tz}); sorted by time, "
        f"{flags.rows_moved_by_sort} row(s) changed position.",
        params=choices.model_dump(),
        counts={
            "rows_moved_by_sort": flags.rows_moved_by_sort,
            "duplicate_timestamps": flags.duplicate_timestamps,
            "mixed_timezone_offsets": float(flags.mixed_timezone_offsets),
        },
        rows_before=rows_before,
        rows_after=rows_after,
    )
    set_aside = flags.missing_timestamp_rows + flags.unparseable_rows
    if set_aside:
        log.append(
            "set_aside_untimed_rows",
            f"Set aside {set_aside} row(s) without a usable timestamp "
            f"({flags.missing_timestamp_rows} missing, {flags.unparseable_rows} unparseable). "
            "They are kept in the result's `rejected` table, not deleted.",
            params={"unparseable_examples": flags.unparseable_examples},
            counts={
                "missing_timestamp_rows": flags.missing_timestamp_rows,
                "unparseable_rows": flags.unparseable_rows,
            },
            rows_before=rows_before,
            rows_after=rows_after,
            read_only=True,
        )
    return log.entries
