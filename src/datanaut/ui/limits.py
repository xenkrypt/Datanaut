"""Size limits for free hosting, in one place. Enforced in code, not only in the server config."""

from __future__ import annotations

import pandas as pd

from datanaut.errors import FileTooLargeError

MAX_UPLOAD_BYTES = 50_000_000
MAX_ROWS = 1_000_000
MAX_COLUMNS = 200
DISPLAY_POINT_LIMIT = 5000
"""Charts plot at most this many points; the full data is still used for computation."""


def check_dimensions(df: pd.DataFrame) -> None:
    """Refuse tables that are too big for free hosting, with a message that says what to do."""
    rows, cols = df.shape
    if rows > MAX_ROWS:
        raise FileTooLargeError(
            f"This file has {rows:,} rows; the limit is {MAX_ROWS:,}. "
            "Try a shorter time range or fewer readings."
        )
    if cols > MAX_COLUMNS:
        raise FileTooLargeError(
            f"This file has {cols:,} columns; the limit is {MAX_COLUMNS:,}. "
            "Keep only the columns you need."
        )
