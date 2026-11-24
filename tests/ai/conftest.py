"""Fixtures for the AI tests: a seeded frame, a scripted model and a fake network transport."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """28 days of hourly readings: two numeric columns and one categorical column."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=28 * 24, freq="h", name="timestamp")
    hour = idx.hour.to_numpy()
    kw = (
        50
        + 20 * np.exp(-(((hour - 19) / 2.5) ** 2))
        + 5 * (idx.day_of_week >= 5)
        + rng.normal(0, 2, len(idx))
    )
    temp = 8 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1, len(idx))
    site = np.where(hour % 2 == 0, "north", "south")
    return pd.DataFrame({"kw": kw.round(3), "temp": temp.round(2), "site": site}, index=idx)
