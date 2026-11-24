"""Every size limit in one place. They are enforced in code, not only in the server config.

The numbers suit free hosting, where memory is small and shared. Raise them only if the app runs
somewhere with more memory (`scripts/stress_test.py` measures what a given size costs).
"""

from __future__ import annotations

import pandas as pd

from datanaut.errors import AnalysisError, FileTooLargeError

MAX_UPLOAD_BYTES = 35_000_000
"""Largest file accepted. `.streamlit/config.toml` must say the same number of megabytes."""
MAX_ROWS = 750_000
MAX_COLUMNS = 200
DISPLAY_POINT_LIMIT = 5000
"""Charts plot at most this many points; the full data is still used for every calculation."""
AI_QUESTION_CAP = 20
"""Questions to the AI analyst per session."""
MAX_SEASONAL_WORK = 5_000_000
"""Most observations x readings-per-cycle the seasonal anomaly check accepts. It costs about
15 microseconds per unit, so this is roughly 75 seconds on a typical shared host (measured:
0.4 ms per hourly reading, 4.2 ms per 5-minute reading)."""

WHY = (
    "The free hosting this app runs on has limited memory, so larger inputs could slow it "
    "down or stop it for everyone."
)


def check_upload_size(n_bytes: int) -> None:
    """Refuse a file over the size limit, saying what the limit is and why."""
    if n_bytes > MAX_UPLOAD_BYTES:
        raise FileTooLargeError(
            f"This file is {n_bytes / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:g} MB. "
            f"{WHY} Try a shorter time range or fewer columns."
        )


def check_seasonal_work(n_observations: int, period: int) -> None:
    """Refuse the seasonal anomaly check when it would take too long, and say how to avoid that."""
    work = n_observations * period
    if work > MAX_SEASONAL_WORK:
        seconds = work * 15e-6
        raise AnalysisError(
            f"This series has {n_observations:,} readings with {period} per cycle, so the seasonal "
            f"anomaly check would take about {seconds / 60:.0f} minute(s). {WHY} Use a shorter "
            "time range, or readings at a coarser interval such as hourly."
        )


def check_dimensions(df: pd.DataFrame) -> None:
    """Refuse tables with too many rows or columns, saying what the limit is and why."""
    rows, cols = df.shape
    if rows > MAX_ROWS:
        raise FileTooLargeError(
            f"This file has {rows:,} rows; the limit is {MAX_ROWS:,}. {WHY} "
            "Try a shorter time range or fewer readings."
        )
    if cols > MAX_COLUMNS:
        raise FileTooLargeError(
            f"This file has {cols:,} columns; the limit is {MAX_COLUMNS:,}. {WHY} "
            "Keep only the columns you need."
        )
