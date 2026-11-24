"""The animated overview strip: which numbers it shows. No streamlit import."""

from __future__ import annotations

import numpy as np

from datanaut.ui import analysis, theme
from datanaut.ui.state import WizardState, cache_key

SPARK_POINTS = 90


def _spark_values(state: WizardState) -> list[float]:
    """The latest readings of the analyzed column, thinned to a handful of points."""

    def compute() -> list[float]:
        s = analysis.series(state).dropna()
        if len(s) > SPARK_POINTS:
            s = s.iloc[np.linspace(0, len(s) - 1, SPARK_POINTS).astype(int)]
        return [float(v) for v in s.to_numpy()]

    return state.get_or_compute(
        cache_key(state.fingerprint, "spark", column=analysis.column(state)), compute
    )


def overview_html(state: WizardState) -> str | None:
    """The dashboard strip for the working version, or None before the columns are mapped."""
    if state.store is None or state.profile is None or not state.value_columns:
        return None
    prof = state.profile
    cells = prof.rows * prof.n_columns
    missing = sum(c.missing_count for c in prof.columns)
    pct = 100 * missing / cells if cells else 0.0
    nodes = state.store.history()
    cards = [
        theme.count_card("Readings", prof.rows, f"{prof.n_columns} columns", 0),
        theme.ring_card("Missing values", pct, "of all cells", 1),
        theme.count_card("Versions", len(nodes), f"working on {state.store.head}", 2),
        theme.count_card("Pipeline steps", len(state.log), "all recorded", 3),
        theme.spark_card(analysis.column(state), _spark_values(state), "recent readings", 4),
    ]
    return theme.board(cards)
