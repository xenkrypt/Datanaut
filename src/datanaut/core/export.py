"""CSV export that is safe to open in a spreadsheet.

Spreadsheet programs treat a cell that starts with =, +, - or @ as a formula, so a malicious
or careless text value (or column name) could run when the file is opened. Such text is
prefixed with a single quote, which makes the spreadsheet show it as plain text. Numbers,
booleans and dates are never touched. The number of changed cells is always reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.api.types import infer_dtype

from datanaut.core.log import PipelineLog, StepLog

DANGEROUS_PREFIXES: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")
"""Leading characters that make a spreadsheet read text as a formula (tab and CR included)."""


@dataclass(frozen=True)
class ExportResult:
    """What was written and how many cells had to be neutralized."""

    path: Path
    rows: int
    columns: int
    cells_neutralized: int
    log_entry: StepLog


def neutralize_text(value: str) -> str:
    """Prefix text that a spreadsheet would read as a formula with a single quote."""
    return "'" + value if value.startswith(DANGEROUS_PREFIXES) else value


def _neutralize_series(s: pd.Series) -> tuple[pd.Series, int]:
    present = s.dropna()
    if present.empty:
        return s, 0
    if infer_dtype(present, skipna=True) == "string":
        flagged = s.str.startswith(DANGEROUS_PREFIXES, na=False)
    else:  # mixed cells: only the strings are text
        flagged = s.map(lambda v: isinstance(v, str) and v.startswith(DANGEROUS_PREFIXES))
    flagged = flagged.astype(bool)
    if not flagged.any():
        return s, 0
    fixed = s.astype(object).copy()
    fixed[flagged] = fixed[flagged].map(neutralize_text)
    return fixed, int(flagged.sum())


def _is_text_like(s: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def neutralize_formulas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return a copy with formula-like text neutralized, and how many cells were changed.

    Text columns, text column names and text index labels are checked; other dtypes are not.
    """
    out = df.copy()
    total = 0
    for position in range(out.shape[1]):
        column = out.iloc[:, position]
        if _is_text_like(column):
            fixed, n = _neutralize_series(column)
            if n:
                out.isetitem(position, fixed)
                total += n
    names = [neutralize_text(c) if isinstance(c, str) else c for c in out.columns]
    total += sum(1 for old, new in zip(out.columns, names, strict=True) if old != new)
    out.columns = names
    if _is_text_like(out.index.to_series()) and not isinstance(out.index, pd.DatetimeIndex):
        fixed, n = _neutralize_series(out.index.to_series())
        if n:
            out.index = pd.Index(fixed.to_numpy(), name=out.index.name)
            total += n
    return out, total


def write_safe_csv(df: pd.DataFrame, path: str | Path) -> ExportResult:
    """Write `df` (index included) to a UTF-8 CSV after neutralizing formula-like text."""
    safe, count = neutralize_formulas(df)
    target = Path(path)
    safe.to_csv(target, encoding="utf-8", lineterminator="\n")
    entry = PipelineLog().append(
        "export_csv",
        f"Wrote {len(df)} row(s) x {df.shape[1]} column(s) to CSV; {count} cell(s) starting with "
        "=, +, - or @ were prefixed with a quote so spreadsheets read them as text.",
        params={"file": target.name},
        counts={"rows": len(df), "columns": df.shape[1], "cells_neutralized": count},
        rows_before=len(df),
        rows_after=len(df),
        read_only=True,
    )
    return ExportResult(
        path=target,
        rows=len(df),
        columns=df.shape[1],
        cells_neutralized=count,
        log_entry=entry,
    )
