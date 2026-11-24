"""Formatting helpers: turn core results into plain rows and tables for display.

No streamlit import and no data logic: everything here only reshapes values the core computed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from datanaut.core.log import PipelineLog
from datanaut.core.profiler import DataProfile
from datanaut.core.versioning import VersionStore


def format_bytes(n: int) -> str:
    """12345 -> '12.1 KB'."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


def log_rows(log: PipelineLog) -> list[dict[str, Any]]:
    """One row per pipeline log entry, for the sidebar."""
    return [
        {
            "step": e.step_id,
            "operation": e.operation,
            "summary": e.summary,
            "counts": dict(e.counts),
            "rows_before": e.rows_before,
            "rows_after": e.rows_after,
            "read_only": e.read_only,
        }
        for e in log
    ]


def version_rows(store: VersionStore | None) -> list[dict[str, Any]]:
    """One row per version (V1..Vn), with the working version marked."""
    if store is None:
        return []
    rows = []
    for node in store.history():
        v = node.version
        rows.append(
            {
                "id": v.version_id,
                "label": v.op.operation if v.op else "raw data",
                "rows": v.rows,
                "columns": v.columns,
                "fingerprint": v.fingerprint[:12],
                "is_head": node.is_head,
                "depth": node.depth,
            }
        )
    return rows


def profile_cards(p: DataProfile) -> list[tuple[str, str]]:
    """The summary numbers shown at the top of the profile screen."""
    cells = p.rows * p.n_columns
    missing = sum(c.missing_count for c in p.columns)
    pct = 100 * missing / cells if cells else 0.0
    irregular = "n/a" if p.sampling is None else f"{p.sampling.irregular_count:,}"
    return [
        ("Rows", f"{p.rows:,}"),
        ("Columns", f"{p.n_columns:,}"),
        ("Missing values", f"{pct:.2f}%"),
        ("Duplicate rows", f"{p.duplicates.full_row_duplicates:,}"),
        ("Irregular intervals", irregular),
    ]


def column_table(p: DataProfile) -> pd.DataFrame:
    """The per-column table."""
    return pd.DataFrame(
        [
            {
                "Column": c.name,
                "Type": c.inferred_type,
                "Missing %": round(c.missing_pct, 2),
                "Unique": c.unique_count,
                "Min": c.min,
                "Mean": c.mean,
                "Max": c.max,
            }
            for c in p.columns
        ]
    )


def gap_table(p: DataProfile) -> pd.DataFrame:
    """The largest gaps in the time index (empty if there are none)."""
    gaps = p.gaps.largest if p.gaps else []
    return pd.DataFrame(
        [{"From": g.start, "To": g.end, "Missing timestamps": g.missing_timestamps} for g in gaps],
        columns=["From", "To", "Missing timestamps"],
    )


def sampling_line(p: DataProfile) -> str:
    if p.sampling is None:
        return "No time index yet, so the sampling interval is not known."
    s = p.sampling
    return (
        f"One reading every {s.median_interval_label}; "
        f"{s.regular_share:.1%} of intervals are regular."
    )


def downsample_minmax(series: pd.Series, limit: int = 5000) -> pd.Series:
    """Shrink a series for plotting by keeping the lowest and highest point of every bucket.

    For display only: the full data is still used for every calculation. The result has at
    most `limit` points, in time order, and always includes the overall minimum and maximum.
    """
    n = len(series)
    if limit < 2 or n <= limit:
        return series
    values = series.to_numpy(dtype=float)
    buckets = max(1, limit // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    keep: set[int] = set()
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        chunk = values[lo:hi]
        if np.isnan(chunk).all():
            keep.add(int(lo))
            continue
        keep.add(int(lo + np.nanargmin(chunk)))
        keep.add(int(lo + np.nanargmax(chunk)))
    return series.iloc[sorted(keep)]
