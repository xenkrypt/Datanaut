"""Draw report charts with matplotlib from the chart payloads the report model prepared.

Figures are built with the object API (`Figure` and the Agg canvas), not pyplot, so no figure is
ever registered globally and nothing needs a display. Each figure is cleared after rendering.
Sizes, colours and fonts are fixed so charts look the same in every report. Plotly figures are
not exported; these are separate renderers for the same chart specs.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from datanaut.report.model import DISPLAY_POINT_LIMIT, ChartPayload, thin_positions
from datanaut.report.text import printable

FIG_WIDTH, FIG_HEIGHT, DPI = 7.0, 3.2, 150
BLUE, GREY, RED, ORANGE, GREEN = "#2457c5", "#8a93a3", "#c7372f", "#d9822b", "#2f9e5b"
_COLOURS = [BLUE, ORANGE, GREEN, GREY]
_BAND = "#9db8f5"


def _safe(text: str) -> str:
    """Matplotlib reads $...$ as math, and the font lacks some characters: show text literally."""
    return printable(text).replace("$", r"\$")


def _style(ax, payload: ChartPayload) -> None:
    ax.set_title(_safe(payload.title), fontsize=10, loc="left")
    ax.set_xlabel(_safe(payload.x_label), fontsize=8)
    ax.set_ylabel(_safe(payload.y_label), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _y(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def render_chart(payload: ChartPayload) -> bytes:
    """A PNG of one chart."""
    fig = Figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    FigureCanvasAgg(fig)
    try:
        ax = fig.add_subplot(1, 1, 1)
        _draw(ax, fig, payload)
        _style(ax, payload)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        return buffer.getvalue()
    finally:
        fig.clear()


def _draw(ax, fig: Figure, p: ChartPayload) -> None:
    if p.kind == "bars":
        n = max(1, len(p.series))
        width = 0.8 / n
        base = np.arange(len(p.x))
        for i, s in enumerate(p.series):
            ax.bar(
                base + (i - (n - 1) / 2) * width,
                _y(s.y),
                width,
                color=_COLOURS[i % 4],
                label=_safe(s.name),
            )
        ax.set_xticks(base)
        step = max(1, len(p.x) // 12)
        ax.set_xticklabels(
            [_safe(str(x)) if i % step == 0 else "" for i, x in enumerate(p.x)],
            rotation=0 if len(p.x) <= 8 else 45,
            ha="right" if len(p.x) > 8 else "center",
        )
        return
    if p.kind == "scatter":
        for i, s in enumerate(p.series):
            ax.scatter(np.asarray(p.x, dtype=float), _y(s.y), s=6, alpha=0.5, color=_COLOURS[i % 4])
        return
    if p.kind == "curve":
        for i, s in enumerate(p.series):
            ax.plot(np.asarray(p.x, dtype=float), _y(s.y), color=_COLOURS[i % 4], linewidth=1.2)
        return
    x = pd.to_datetime(pd.Index(p.x), format="ISO8601")  # time series ("line")
    if len(x) > DISPLAY_POINT_LIMIT:  # a second guard: thin again for display only
        keep = (
            thin_positions(_y(p.series[0].y), DISPLAY_POINT_LIMIT)
            if p.series
            else np.arange(len(x))
        )
    else:
        keep = np.arange(len(x))
    xs = x[keep]
    for i, s in enumerate(p.series):
        ax.plot(xs, _y(s.y)[keep], color=_COLOURS[i % 4], linewidth=1.2, label=_safe(s.name))
    if p.band_lower is not None and p.band_upper is not None:
        ax.fill_between(
            xs,
            _y(p.band_lower)[keep],
            _y(p.band_upper)[keep],
            color=_BAND,
            alpha=0.5,
            label="Prediction interval (estimate)",
        )
    if p.markers and p.series:
        marks = pd.to_datetime(pd.Index(p.markers), format="ISO8601")
        y = pd.Series(_y(p.series[0].y), index=x)
        y = y[~y.index.duplicated()]
        hit = y.reindex(marks)
        ax.scatter(marks, hit.to_numpy(), s=18, color=RED, zorder=3, label="Flagged")
    for stamp in p.vlines:
        ax.axvline(pd.Timestamp(stamp), color=ORANGE, linestyle="--", linewidth=1.0)
    fig.autofmt_xdate(rotation=20, ha="right")
    if len(p.series) > 1 or p.markers or p.band_lower is not None:
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4)
