"""Plotly figure builders: pure functions from core results to figures. No streamlit import.

Every figure has a title and labelled axes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datanaut.limits import DISPLAY_POINT_LIMIT
from datanaut.ui.display import downsample_minmax


def missingness_heatmap(matrix: pd.DataFrame, title: str = "Where values are missing") -> go.Figure:
    """Share of missing values per column (x) over time bins (y), from the profiler's matrix."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(dtype=float),
            x=[str(c) for c in matrix.columns],
            y=[str(i) for i in matrix.index],
            zmin=0,
            zmax=1,
            colorscale=_SCALE,
            colorbar={"title": "Share missing"},
            hovertemplate="%{x}<br>%{y}<br>missing: %{z:.0%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_BASE,
        title=title,
        xaxis_title="Column",
        yaxis_title="Time (binned)",
        yaxis={"autorange": "reversed"},
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
    )
    return fig


def before_after(
    before: pd.Series, after: pd.Series, column: str, limit: int = DISPLAY_POINT_LIMIT
) -> go.Figure:
    """The affected column before and after an operation (downsampled for display only)."""
    fig = go.Figure()
    for label, series, colour in (("Before", before, "#9aa5b1"), ("After", after, "#1f6feb")):
        shown = downsample_minmax(series, limit)
        fig.add_trace(
            go.Scatter(
                x=shown.index,
                y=shown.to_numpy(),
                mode="lines",
                name=label,
                line={"color": colour, "width": 1.5},
            )
        )
    fig.update_layout(
        **_BASE,
        title=f"{column}: before and after",
        xaxis_title="Time",
        yaxis_title=f"{column} (reading)",
        legend={"orientation": "h"},
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
    )
    return fig


# --------------------------------------------------------------------------- analysis charts

BLUE, GREY, RED, ORANGE, GREEN = "#7c9cff", "#6b7385", "#ff6b6b", "#ffb454", "#4ade80"
_MARGIN = {"l": 60, "r": 20, "t": 50, "b": 40}
_BASE = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "color": "#c9d1e0"},
    "transition": {"duration": 450, "easing": "cubic-in-out"},
    "hoverlabel": {"bgcolor": "#161b26", "bordercolor": "#2b3347"},
}
_SCALE = [[0.0, "#141824"], [1.0, "#7c9cff"]]


def _layout(fig: go.Figure, title: str, x: str, y: str) -> go.Figure:
    fig.update_layout(
        **_BASE,
        title=title,
        xaxis_title=x,
        yaxis_title=y,
        legend={"orientation": "h"},
        margin=_MARGIN,
    )
    return fig


def _line(fig: go.Figure, s: pd.Series, name: str, colour: str, limit: int) -> None:
    shown = downsample_minmax(s, limit)
    fig.add_trace(
        go.Scatter(
            x=shown.index,
            y=shown.to_numpy(),
            mode="lines",
            name=name,
            line={"color": colour, "width": 1.4},
        )
    )


def time_series(
    series: pd.Series,
    column: str,
    flagged: pd.DatetimeIndex | None = None,
    breaks: list[str] | None = None,
    title: str | None = None,
    limit: int = DISPLAY_POINT_LIMIT,
) -> go.Figure:
    """The series, with optional anomaly markers and change-point lines."""
    fig = go.Figure()
    _line(fig, series, column, BLUE, limit)
    if flagged is not None and len(flagged):
        hit = series[series.index.isin(flagged)].dropna()  # tolerates repeated timestamps
        fig.add_trace(
            go.Scatter(
                x=hit.index,
                y=hit.to_numpy(),
                mode="markers",
                name="Flagged",
                marker={"color": RED, "size": 8},
            )
        )
    for stamp in breaks or []:
        fig.add_vline(x=pd.Timestamp(stamp), line={"color": ORANGE, "dash": "dash", "width": 1.5})
    return _layout(fig, title or f"{column} over time", "Time", f"{column} (reading)")


def profile_bars(profile, title: str | None = None) -> go.Figure:
    """Average by hour of day or by weekday (a core LoadProfile)."""
    labels = [b.label for b in profile.buckets]
    means = [b.mean for b in profile.buckets]
    fig = go.Figure(go.Bar(x=labels, y=means, marker={"color": BLUE}, name="Average"))
    by = "hour of day" if profile.by == "hour" else "weekday"
    return _layout(
        fig,
        title or f"{profile.column}: average by {by}",
        by.capitalize(),
        f"Average {profile.column}",
    )


def profile_heatmap(profile) -> go.Figure:
    """Average by hour (columns) and weekday (rows) from an hour_x_weekday LoadProfile."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    z = [[None] * 24 for _ in range(7)]
    for b in profile.buckets:
        if b.hour is not None and b.weekday is not None:
            z[b.weekday][b.hour] = b.mean
    fig = go.Figure(
        go.Heatmap(z=z, x=list(range(24)), y=days, colorscale=_SCALE, colorbar={"title": "Average"})
    )
    return _layout(fig, f"{profile.column}: hour and weekday", "Hour of day", "Weekday")


def share_bars(entries: list, title: str, xlabel: str) -> go.Figure:
    """Share of peak readings per hour or weekday (core ShareEntry list)."""
    fig = go.Figure(
        go.Bar(
            x=[e.label for e in entries],
            y=[e.share for e in entries],
            marker={"color": ORANGE},
            name="Share of peaks",
        )
    )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, title, xlabel, "Share of peak readings")


def load_duration(series: pd.Series, column: str, limit: int = DISPLAY_POINT_LIMIT) -> go.Figure:
    """Readings sorted from highest to lowest."""
    ordered = pd.Series(np.sort(series.dropna().to_numpy())[::-1])
    shown = downsample_minmax(ordered, limit)
    fig = go.Figure(
        go.Scatter(
            x=shown.index, y=shown.to_numpy(), mode="lines", line={"color": BLUE}, name=column
        )
    )
    return _layout(
        fig, f"{column}: load duration curve", "Readings, highest first", f"{column} (reading)"
    )


def stl_components(
    observed: pd.Series, fitted: pd.Series, column: str, limit: int = DISPLAY_POINT_LIMIT
) -> go.Figure:
    """Observed, trend-plus-seasonal fit and what is left over, in three stacked panels."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Observed", "Trend and seasonal fit", "Residual"),
    )
    panels = (
        (observed, BLUE, "Observed"),
        (fitted, GREEN, "Fit"),
        (observed - fitted, GREY, "Residual"),
    )
    for row, (s, colour, name) in enumerate(panels, start=1):
        shown = downsample_minmax(s, limit)
        fig.add_trace(
            go.Scatter(
                x=shown.index,
                y=shown.to_numpy(),
                mode="lines",
                name=name,
                line={"color": colour, "width": 1.2},
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=column if row < 3 else "Difference", row=row, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(
        **_BASE,
        title=f"{column}: trend and seasonal components",
        margin=_MARGIN,
        height=620,
        showlegend=False,
    )
    return fig


def multivariate_view(result, limit: int = 15) -> go.Figure:
    """Each flagged point, split into the share each column contributed to its unusualness."""
    fig = go.Figure()
    points = result.points[:limit]
    for col in result.columns:
        shares = [next((c.share for c in p.contributions if c.column == col), 0.0) for p in points]
        fig.add_trace(go.Bar(name=col, x=[p.timestamp for p in points], y=shares))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%")
    return _layout(
        fig,
        "Flagged points: which columns contributed",
        "Flagged time",
        "Share of the contribution",
    )


def detector_bars(comparison) -> go.Figure:
    """How many points each detector flagged alone, and how many both did."""
    labels = ["Only the single-column check", "Both checks", "Only the multi-column check"]
    values = [comparison.n_only_univariate, comparison.n_both, comparison.n_only_multivariate]
    fig = go.Figure(go.Bar(x=labels, y=values, marker={"color": [GREY, GREEN, ORANGE]}))
    return _layout(fig, "Anomaly detectors compared", "Flagged by", "Number of points")


def change_point_segments(series: pd.Series, cp, limit: int = DISPLAY_POINT_LIMIT) -> go.Figure:
    """The series with each segment's mean and a line at every detected change."""
    fig = go.Figure()
    _line(fig, series, cp.column, GREY, limit)
    for i, seg in enumerate(cp.segments):
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(seg.start), pd.Timestamp(seg.end)],
                y=[seg.mean, seg.mean],
                mode="lines",
                name="Segment mean",
                showlegend=i == 0,
                line={"color": BLUE, "width": 3},
            )
        )
    for b in cp.breaks:
        fig.add_vline(x=pd.Timestamp(b.timestamp), line={"color": ORANGE, "dash": "dash"})
    return _layout(
        fig, f"{cp.column}: change points and segment means", "Time", f"{cp.column} (reading)"
    )


def forecast_chart(history: pd.Series, fc, tail: int = 240) -> go.Figure:
    """Recent history, then the forecast with its interval. Labelled as an estimate."""
    stamps = pd.to_datetime([p.timestamp for p in fc.points])
    fig = go.Figure()
    _line(fig, history.dropna().iloc[-tail:], "Actual (recent)", GREY, DISPLAY_POINT_LIMIT)
    fig.add_trace(
        go.Scatter(
            x=stamps,
            y=[p.upper for p in fc.points],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stamps,
            y=[p.lower for p in fc.points],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31,111,235,0.2)",
            name=f"{fc.interval_level:.0%} interval (estimate)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stamps,
            y=[p.forecast for p in fc.points],
            mode="lines",
            name=f"Forecast ({fc.model})",
            line={"color": BLUE, "width": 2},
        )
    )
    return _layout(
        fig,
        f"{fc.column}: forecast (an estimate, not a guarantee)",
        "Time",
        f"{fc.column} (reading)",
    )


def backtest_bars(bt) -> go.Figure:
    """Average error of each model across the backtest folds (lower is better)."""
    fig = go.Figure(
        go.Bar(x=[s.model for s in bt.scores], y=[s.mae for s in bt.scores], marker={"color": BLUE})
    )
    return _layout(
        fig,
        f"{bt.column}: backtest error by model",
        "Model",
        "Mean absolute error (lower is better)",
    )


def weather_scatter(
    weather: pd.Series,
    series: pd.Series,
    variable: str,
    column: str,
    limit: int = DISPLAY_POINT_LIMIT,
) -> go.Figure:
    """Each reading against the weather at the same time."""
    if len(weather) == len(
        series
    ):  # aligned to the series: pair by position (timestamps may repeat)
        both = pd.DataFrame({"w": weather.to_numpy(), "y": series.to_numpy()}).dropna()
    else:
        both = pd.concat([weather.rename("w"), series.rename("y")], axis=1).dropna()
    if len(both) > limit:
        both = both.iloc[np.linspace(0, len(both) - 1, limit).astype(int)]
    fig = go.Figure(
        go.Scattergl(
            x=both["w"],
            y=both["y"],
            mode="markers",
            marker={"color": BLUE, "size": 5, "opacity": 0.5},
            name="Readings",
        )
    )
    return _layout(fig, f"{column} and {variable}", variable, f"{column} (reading)")


def lag_bars(link) -> go.Figure:
    """Association at each lag, with the weather leading the series."""
    lags = [x for x in link.lag_correlations if x.pearson is not None]
    fig = go.Figure(
        go.Bar(
            x=[x.lag_hours for x in lags],
            y=[x.pearson for x in lags],
            marker={"color": BLUE},
            name="Pearson r",
        )
    )
    return _layout(
        fig,
        f"{link.column} and {link.variable}: association by lag",
        "Lag in hours (weather leads)",
        "Pearson correlation",
    )


def whatif_chart(
    before: pd.Series, after: pd.Series, column: str, limit: int = DISPLAY_POINT_LIMIT
) -> go.Figure:
    """The scenario series against the original."""
    fig = go.Figure()
    _line(fig, before, "Before", GREY, limit)
    _line(fig, after, "After shifting (scenario estimate)", BLUE, limit)
    return _layout(fig, f"{column}: what-if peak shifting", "Time", f"{column} (reading)")


def finding_chart(spec, ctx: dict) -> go.Figure | None:
    """The chart a finding asks for (by its ChartSpec name), or None if its inputs are missing."""
    name, series = spec.name, ctx.get("series")
    col = spec.params.get("column", "")
    if name == "hour_of_day_peak_shares" and ctx.get("peaks") is not None:
        return share_bars(ctx["peaks"].hours, f"{col}: peaks by hour of day", "Hour of day")
    if name == "weekday_peak_shares" and ctx.get("peaks") is not None:
        return share_bars(ctx["peaks"].weekdays, f"{col}: peaks by weekday", "Weekday")
    if name == "load_duration_curve" and series is not None:
        return load_duration(series, col)
    if name == "weekday_profile" and ctx.get("weekday") is not None:
        return profile_bars(ctx["weekday"])
    if name == "anomaly_timeline" and series is not None and ctx.get("anomalies") is not None:
        flagged = ctx["anomalies"].flagged_timestamps()
        return time_series(series, col, flagged, title=f"{col}: anomalies")
    if name == "change_point_timeline" and series is not None and ctx.get("changes") is not None:
        return change_point_segments(series, ctx["changes"])
    if name == "scatter_with_weather" and ctx.get("weather_x") is not None and series is not None:
        return weather_scatter(ctx["weather_x"], series, ctx["link"].variable, col)
    if name == "forecast_with_interval" and ctx.get("forecast") is not None and series is not None:
        return forecast_chart(series, ctx["forecast"])
    return None


def ask_chart(card) -> go.Figure | None:
    """The chart for an AI answer's result table (a core EvidenceCard), or None for one number."""
    table = card.table
    if card.chart.name == "single_value" or len(table.rows) < 2 or len(table.columns) < 2:
        return None
    labels = [r[0] for r in table.rows]
    fig = go.Figure()
    for i, name in enumerate(table.columns[1:], start=1):
        values = [r[i] for r in table.rows]
        if card.chart.name == "aggregate_over_time":
            fig.add_trace(go.Scatter(x=labels, y=values, mode="lines+markers", name=name))
        else:
            fig.add_trace(go.Bar(x=labels, y=values, name=name))
    return _layout(
        fig,
        f"{', '.join(card.columns)}: {card.method}",
        str(table.columns[0]).capitalize(),
        "Computed value",
    )
