# ruff: noqa: E501  (report text and chart definitions contain long lines)
"""What goes in the report: a model assembled from results that already exist. No rendering.

`build_report_data` reads the results it is given (findings, patterns, anomalies, change points,
forecast, what-if, investigation, AI suggestions, validation) and turns them into plain sections
of paragraphs, tables and charts. Sections with nothing to show are omitted, never left empty.
Text that comes from the user's file (column names, file name) is cleaned and shortened.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from datanaut.core.characterize import SeriesCharacterization, characterize
from datanaut.core.forecast import ESTIMATE_NOTE, recommend_model
from datanaut.core.log import PipelineLog, library_versions
from datanaut.core.profiler import DataProfile, profile
from datanaut.core.validation import ValidationReport
from datanaut.core.versioning import VersionStore
from datanaut.errors import DatanautError
from datanaut.limits import DISPLAY_POINT_LIMIT

PRODUCT = "Datanaut"
FORMAL_TITLE = "An AI-Powered Intelligent Platform for Time-Series Data Analysis, Analytics and Insight Generation"  # noqa: E501
SECTION_TITLES: dict[str, str] = {
    "summary": "Dataset summary",
    "quality": "Data quality",
    "pipeline": "Cleaning and pipeline log",
    "validation": "Validation results",
    "findings": "Key findings",
    "patterns": "Patterns and energy metrics",
    "anomalies": "Anomalies and change points",
    "forecast": "Forecast benchmark",
    "whatif": "What-if scenario",
    "investigation": "Investigation",
    "suggestions": "AI-generated suggestions",
    "limitations": "Limitations and assumptions",
    "reproducibility": "Reproducibility appendix",
}
MAX_NAME = 80
MAX_TEXT = 400
MAX_CELL = 120
AI_NOTE = (
    "These suggestions were produced by a language model from the computed findings and the "
    "pipeline log. They are ideas to check, not conclusions."
)
CHART_FOR_SPEC = {
    "hour_of_day_peak_shares": "peak_hours",
    "weekday_peak_shares": "peak_weekdays",
    "load_duration_curve": "load_duration",
    "weekday_profile": "weekday_profile",
    "anomaly_timeline": "anomalies",
    "change_point_timeline": "change_points",
    "scatter_with_weather": "weather_scatter",
    "forecast_with_interval": "forecast",
}


# --------------------------------------------------------------------------- small helpers


def clean(value: object, max_len: int = MAX_TEXT) -> str:
    """Text safe for the page: control characters removed, spaces collapsed, cut with an ellipsis."""
    text = "".join(ch for ch in str(value) if ch in "\n\t " or unicodedata.category(ch)[0] != "C")
    # a long unbroken word is almost always a column name or a value: shorten it wherever it appears
    words = [w if len(w) <= MAX_NAME else w[: MAX_NAME - 1] + "…" for w in text.split()]
    text = " ".join(words)
    return text if len(text) <= max_len else text[: max_len - 1].rstrip() + "…"


def num(value: Any, digits: int = 4) -> str:
    """A number for a table cell ('-' when missing)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int | np.integer):
        return f"{int(value):,}"
    if isinstance(value, float | np.floating):
        return f"{float(value):.{digits}g}"
    return clean(value, MAX_CELL)


def thin_positions(values: np.ndarray, limit: int = DISPLAY_POINT_LIMIT) -> np.ndarray:
    """Positions to plot: the lowest and highest point of every bucket (display only)."""
    n = len(values)
    if n <= limit or limit < 2:
        return np.arange(n)
    edges = np.linspace(0, n, max(1, limit // 2) + 1).astype(int)
    keep: set[int] = set()
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi <= lo:
            continue
        chunk = values[lo:hi]
        if np.isnan(chunk).all():
            keep.add(int(lo))
        else:
            keep.add(int(lo + np.nanargmin(chunk)))
            keep.add(int(lo + np.nanargmax(chunk)))
    return np.array(sorted(keep))


# --------------------------------------------------------------------------------- blocks


class Heading(BaseModel):
    type: Literal["heading"] = "heading"
    text: str


class Paragraph(BaseModel):
    type: Literal["paragraph"] = "paragraph"
    text: str
    style: Literal["normal", "note", "estimate"] = "normal"


class Bullets(BaseModel):
    type: Literal["bullets"] = "bullets"
    items: list[str]


class Table(BaseModel):
    type: Literal["table"] = "table"
    title: str = ""
    columns: list[str]
    rows: list[list[str]]


class KeyValues(BaseModel):
    type: Literal["kv"] = "kv"
    items: list[tuple[str, str]]


class ChartBlock(BaseModel):
    type: Literal["chart"] = "chart"
    chart_id: str


Block = Annotated[
    Heading | Paragraph | Bullets | Table | KeyValues | ChartBlock, Field(discriminator="type")
]


class Section(BaseModel):
    id: str
    title: str
    blocks: list[Block]


class SeriesPayload(BaseModel):
    name: str
    y: list[float | None]


class ChartPayload(BaseModel):
    """Everything a renderer needs to draw one chart; already thinned for display."""

    id: str
    kind: Literal["line", "bars", "scatter", "curve"]
    title: str
    x_label: str
    y_label: str
    x: list[str | float] = Field(default_factory=list)
    series: list[SeriesPayload] = Field(default_factory=list)
    markers: list[str] = Field(default_factory=list)
    vlines: list[str] = Field(default_factory=list)
    band_lower: list[float | None] | None = None
    band_upper: list[float | None] | None = None


class Cover(BaseModel):
    product: str = PRODUCT
    formal_title: str = FORMAL_TITLE
    dataset_name: str
    generated: str
    version_id: str
    fingerprint: str


class ReportData(BaseModel):
    cover: Cover
    sections: list[Section]
    charts: dict[str, ChartPayload] = Field(default_factory=dict)

    @property
    def section_ids(self) -> list[str]:
        return [s.id for s in self.sections]


# ---------------------------------------------------------------------------------- inputs


@dataclass
class ReportInputs:
    """Existing results and the data they came from. Nothing here is computed for the report."""

    store: VersionStore
    log: PipelineLog
    column: str
    dataset_name: str = "dataset"
    value_columns: list[str] = field(default_factory=list)
    consumption: dict[str, bool] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    profile: DataProfile | None = None
    characterization: SeriesCharacterization | None = None
    validation: ValidationReport | None = None
    suggestions: Any = None  # a SuggestionOutcome, if the user generated suggestions
    weather_x: pd.Series | None = None  # the weather variable on this version's timestamps
    results: Mapping[str, Any] = field(default_factory=dict)
    """Results by kind: findings, peaks, hour, weekday, metrics, stl, multivariate, changes,
    backtest, forecast, whatif, investigation, weather_link."""


def now_utc_text(moment: datetime | None = None) -> str:
    return (moment or datetime.now(UTC)).astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


# ------------------------------------------------------------------------------- charts


def _line_xy(series: pd.Series, limit: int) -> tuple[np.ndarray, list[str], list[float | None]]:
    values = series.to_numpy(dtype=float)
    pos = thin_positions(values, limit)
    x = [str(t) for t in series.index[pos]]
    y = [None if not np.isfinite(v) else float(v) for v in values[pos]]
    return pos, x, y


def _bars(chart_id, title, x_label, y_label, labels, values) -> ChartPayload:
    return ChartPayload(
        id=chart_id,
        kind="bars",
        title=title,
        x_label=x_label,
        y_label=y_label,
        x=[clean(s, 40) for s in labels],
        series=[SeriesPayload(name=y_label, y=[None if v is None else float(v) for v in values])],
    )


def _charts(inputs: ReportInputs, df: pd.DataFrame) -> dict[str, ChartPayload]:
    r = inputs.results
    col = inputs.column
    out: dict[str, ChartPayload] = {}
    series = df[col] if col in df.columns else None
    name = clean(col, MAX_NAME)

    if series is not None and r.get("stl") is not None:
        _, x, y = _line_xy(series, DISPLAY_POINT_LIMIT)
        shown = set(x)
        flagged = [str(t) for t in r["stl"].flagged_timestamps() if str(t) in shown][:300]
        out["anomalies"] = ChartPayload(
            id="anomalies",
            kind="line",
            title=f"{name}: unusual readings",
            x_label="Time",
            y_label=f"{name} (reading)",
            x=x,
            series=[SeriesPayload(name=name, y=y)],
            markers=flagged,
        )
    if series is not None and r.get("changes") is not None:
        cp = r["changes"]
        pos, x, y = _line_xy(series, DISPLAY_POINT_LIMIT)
        stamps = pd.to_datetime(pd.Index(x), format="ISO8601")
        seg_mean: list[float | None] = [None] * len(x)
        for seg in cp.segments:
            inside = (stamps >= pd.Timestamp(seg.start)) & (stamps <= pd.Timestamp(seg.end))
            for i in np.flatnonzero(inside):
                seg_mean[i] = float(seg.mean)
        out["change_points"] = ChartPayload(
            id="change_points",
            kind="line",
            title=f"{name}: change points and segment means",
            x_label="Time",
            y_label=f"{name} (reading)",
            x=x,
            series=[SeriesPayload(name=name, y=y), SeriesPayload(name="Segment mean", y=seg_mean)],
            vlines=[b.timestamp for b in cp.breaks][:30],
        )
    for key, label in (("hour", "hour of day"), ("weekday", "weekday")):
        prof = r.get(key)
        if prof is not None:
            out[f"{key}_profile"] = _bars(
                f"{key}_profile",
                f"{name}: average by {label}",
                label.capitalize(),
                f"Average {name}",
                [b.label for b in prof.buckets],
                [b.mean for b in prof.buckets],
            )
    peaks = r.get("peaks")
    if peaks is not None:
        out["peak_hours"] = _bars(
            "peak_hours",
            f"{name}: peaks by hour of day",
            "Hour of day",
            "Share of peak readings",
            [e.label for e in peaks.hours],
            [e.share for e in peaks.hours],
        )
        out["peak_weekdays"] = _bars(
            "peak_weekdays",
            f"{name}: peaks by weekday",
            "Weekday",
            "Share of peak readings",
            [e.label for e in peaks.weekdays],
            [e.share for e in peaks.weekdays],
        )
    if series is not None and series.notna().any():
        ordered = pd.Series(np.sort(series.dropna().to_numpy(dtype=float))[::-1])
        pos = thin_positions(ordered.to_numpy(), DISPLAY_POINT_LIMIT)
        out["load_duration"] = ChartPayload(
            id="load_duration",
            kind="curve",
            title=f"{name}: load duration curve",
            x_label="Readings, highest first",
            y_label=f"{name} (reading)",
            x=[float(i) for i in pos],
            series=[SeriesPayload(name=name, y=[float(ordered.iloc[i]) for i in pos])],
        )
    fc = r.get("forecast")
    if series is not None and fc is not None:
        hist = series.dropna().iloc[-240:]
        hx = [str(t) for t in hist.index]
        fx = [p.timestamp for p in fc.points]
        out["forecast"] = ChartPayload(
            id="forecast",
            kind="line",
            title=f"{name}: forecast (an estimate, not a guarantee)",
            x_label="Time",
            y_label=f"{name} (reading)",
            x=hx + fx,
            series=[
                SeriesPayload(
                    name="Actual (recent)", y=[float(v) for v in hist] + [None] * len(fx)
                ),
                SeriesPayload(
                    name=f"Forecast ({fc.model})",
                    y=[None] * len(hx) + [p.forecast for p in fc.points],
                ),
            ],
            band_lower=[None] * len(hx) + [p.lower for p in fc.points],
            band_upper=[None] * len(hx) + [p.upper for p in fc.points],
        )
    bt = r.get("backtest")
    if bt is not None:
        out["backtest"] = _bars(
            "backtest",
            f"{name}: backtest error by model",
            "Model",
            "Mean absolute error (lower is better)",
            [s.model for s in bt.scores],
            [s.mae for s in bt.scores],
        )
    shift = r.get("whatif")
    if series is not None and shift is not None:
        pos, x, before = _line_xy(series, DISPLAY_POINT_LIMIT)
        after_all = shift.new_series.to_numpy(dtype=float)
        after = [
            None if not np.isfinite(after_all[i]) else float(after_all[i])
            for i in pos
            if i < len(after_all)
        ]
        out["whatif"] = ChartPayload(
            id="whatif",
            kind="line",
            title=f"{name}: what-if peak shifting (scenario estimate)",
            x_label="Time",
            y_label=f"{name} (reading)",
            x=x,
            series=[
                SeriesPayload(name="Before", y=before),
                SeriesPayload(name="After shifting (scenario estimate)", y=after),
            ],
        )
    link = r.get("weather_link")
    if series is not None and link is not None and inputs.weather_x is not None:
        wx = inputs.weather_x.to_numpy(dtype=float)
        sy = series.to_numpy(dtype=float)
        if len(wx) == len(sy):
            ok = np.isfinite(wx) & np.isfinite(sy)
            idx = np.flatnonzero(ok)
            if len(idx) > DISPLAY_POINT_LIMIT:
                idx = idx[np.linspace(0, len(idx) - 1, DISPLAY_POINT_LIMIT).astype(int)]
            variable = clean(link.variable, MAX_NAME)
            out["weather_scatter"] = ChartPayload(
                id="weather_scatter",
                kind="scatter",
                title=f"{name} and {variable}",
                x_label=variable,
                y_label=f"{name} (reading)",
                x=[float(wx[i]) for i in idx],
                series=[SeriesPayload(name=name, y=[float(sy[i]) for i in idx])],
            )
    return out


# ------------------------------------------------------------------------------- sections


def _kv(*pairs: tuple[str, Any]) -> KeyValues:
    return KeyValues(items=[(k, clean(v, MAX_TEXT)) for k, v in pairs])


def _table(title: str, columns: list[str], rows: list[list[Any]]) -> Table:
    return Table(
        title=title,
        columns=columns,
        rows=[[clean(c, MAX_CELL) if isinstance(c, str) else num(c) for c in row] for row in rows],
    )


def _summary(inputs, version, df, prof, char) -> Section:
    start = df.index.min() if len(df) else None
    end = df.index.max() if len(df) else None
    pairs: list[tuple[str, Any]] = [
        ("Dataset", clean(inputs.dataset_name, MAX_NAME)),
        ("Dataset version", f"{version.version_id} (fingerprint {version.fingerprint[:12]})"),
        ("Rows", f"{version.rows:,}"),
        ("Columns", f"{version.columns:,}"),
        ("Time range", f"{start} to {end}"),
    ]
    if prof is not None and prof.sampling is not None:
        s = prof.sampling
        pairs.append(
            ("Sampling", f"every {s.median_interval_label}; {s.regular_share:.1%} regular")
        )
    pairs.append(("Analyzed column", clean(inputs.column, MAX_NAME)))
    if inputs.column in inputs.consumption:
        pairs.append(
            ("Marked as consumption", "yes" if inputs.consumption[inputs.column] else "no")
        )
    blocks: list[Any] = [_kv(*pairs)]
    if char is not None:
        t, st = char.trend, char.stationarity
        blocks += [
            Heading(text="How the series behaves"),
            _kv(
                ("Frequency", char.frequency_label),
                ("Observations", f"{char.n_observations:,}"),
                (
                    "Repeating pattern",
                    f"{char.seasonal_period} readings per cycle"
                    if char.seasonal_period
                    else "none detected",
                ),
                ("Trend", t.direction),
                ("Stationarity", st.label),
            ),
            Paragraph(text=clean(char.recommendation)),
        ]
    return Section(id="summary", title=SECTION_TITLES["summary"], blocks=blocks)


def _quality(prof: DataProfile) -> Section:
    cells = prof.rows * prof.n_columns
    missing = sum(c.missing_count for c in prof.columns)
    d = prof.duplicates
    blocks: list[Any] = [
        _kv(
            (
                "Missing values",
                f"{missing:,} of {cells:,} cells ({100 * missing / cells:.2f}%)" if cells else "0",
            ),
            ("Exact duplicate rows", f"{d.full_row_duplicates:,}"),
            (
                "Duplicate timestamps",
                "n/a"
                if d.duplicate_timestamps is None
                else f"{d.duplicate_timestamps:,} ({d.duplicate_timestamps_identical or 0} identical, "
                f"{d.duplicate_timestamps_conflicting or 0} conflicting)",
            ),
        )
    ]
    if prof.warnings:
        blocks += [Heading(text="Warnings"), Bullets(items=[clean(w) for w in prof.warnings])]
    if prof.gaps and prof.gaps.largest:
        blocks.append(
            _table(
                "Largest gaps in time",
                ["From", "To", "Missing timestamps"],
                [[g.start, g.end, g.missing_timestamps] for g in prof.gaps.largest[:10]],
            )
        )
    blocks.append(
        _table(
            "Columns",
            ["Column", "Type", "Missing %", "Min", "Mean", "Max"],
            [
                [
                    clean(c.name, MAX_NAME),
                    c.inferred_type,
                    round(c.missing_pct, 2),
                    c.min,
                    c.mean,
                    c.max,
                ]
                for c in prof.columns[:30]
            ],
        )
    )
    return Section(id="quality", title=SECTION_TITLES["quality"], blocks=blocks)


def _log_rows(inputs: ReportInputs, version_id: str) -> list[list[Any]]:
    chain_ids = {v.version_id for v in inputs.store.chain(version_id)}
    rows = []
    for e in inputs.log:
        made = e.params.get("dataset_version") if isinstance(e.params, dict) else None
        if isinstance(made, dict) and made.get("version_id") not in chain_ids:
            continue  # a change that belongs to another branch of the history
        counts = ", ".join(f"{clean(k, 30)}: {num(v)}" for k, v in e.counts.items())
        rows.append(
            [
                e.step_id,
                clean(e.operation, 40),
                clean(e.summary, MAX_TEXT),
                counts,
                "" if e.rows_before is None else e.rows_before,
                "" if e.rows_after is None else e.rows_after,
            ]
        )
    return rows


def _pipeline(inputs: ReportInputs, version_id: str) -> Section | None:
    rows = _log_rows(inputs, version_id)
    if not rows:
        return None
    return Section(
        id="pipeline",
        title=SECTION_TITLES["pipeline"],
        blocks=[
            Paragraph(
                text="Every step applied to the data, in order. Steps that belong to other "
                "branches of the version history are left out.",
                style="note",
            ),
            _table(
                "Pipeline log",
                ["Step", "Operation", "What happened", "Counts", "Rows before", "Rows after"],
                rows,
            ),
        ],
    )


def _validation(report: ValidationReport | None) -> Section | None:
    if report is None or not report.results:
        return None
    rows = [
        [
            clean(r.name, MAX_CELL),
            r.severity,
            r.status.replace("_", " "),
            r.violations,
            clean(r.message, MAX_CELL),
        ]
        for r in report.results
    ]
    verdict = (
        "All error-level rules are satisfied."
        if report.ok
        else "Some error-level rules are broken."
    )
    return Section(
        id="validation",
        title=SECTION_TITLES["validation"],
        blocks=[
            Paragraph(text=f"Rule set '{clean(report.ruleset_name, MAX_NAME)}'. {verdict}"),
            _table("", ["Rule", "Severity", "Status", "Violations", "Note"], rows),
        ],
    )


def _findings(r, charts) -> Section | None:
    report = r.get("findings")
    if report is None or not report.findings:
        return None
    blocks: list[Any] = [
        Paragraph(
            text="Ranked by strength. Each statement describes an association in the data, not a cause.",
            style="note",
        )
    ]
    for i, f in enumerate(report.findings, start=1):
        ev = f.evidence
        blocks += [
            Heading(text=f"{i}. {clean(f.title, 120)}"),
            Paragraph(text=clean(f.statement)),
            Paragraph(
                text=clean(
                    f"Evidence: {', '.join(ev.columns)} | {ev.start} to {ev.end} | "
                    f"{ev.n_observations:,} observations | {ev.method}"
                ),
                style="note",
            ),
        ]
        cid = CHART_FOR_SPEC.get(f.chart.name)
        if cid in charts:
            blocks.append(ChartBlock(chart_id=cid))
    return Section(id="findings", title=SECTION_TITLES["findings"], blocks=blocks)


def _patterns(inputs: ReportInputs, charts) -> Section | None:
    r = inputs.results
    metrics, peaks = r.get("metrics"), r.get("peaks")
    blocks: list[Any] = []
    if metrics is not None and metrics.applicable:
        blocks += [
            Heading(text="Energy metrics"),
            _kv(
                ("Peak", num(metrics.peak)),
                ("Peak time", metrics.peak_time or "-"),
                ("Baseload", num(metrics.baseload)),
                ("Mean", num(metrics.mean)),
                ("Load factor", num(metrics.load_factor)),
            ),
        ] + [Paragraph(text=clean(n), style="note") for n in metrics.notes]
    if peaks is not None:
        blocks += [
            Heading(text="Peak periods"),
            Paragraph(
                text=f"Peaks are most common around {', '.join(peaks.top_hours) or 'n/a'} "
                f"({peaks.top_hours_share:.0%} of them) and on "
                f"{', '.join(peaks.top_weekdays) or 'n/a'}."
            ),
        ]
    blocks += [
        ChartBlock(chart_id=c)
        for c in ("hour_profile", "weekday_profile", "peak_hours")
        if c in charts
    ]
    return (
        Section(id="patterns", title=SECTION_TITLES["patterns"], blocks=blocks) if blocks else None
    )


def _anomalies(inputs: ReportInputs, charts) -> Section | None:
    r = inputs.results
    stl, cp, multi = r.get("stl"), r.get("changes"), r.get("multivariate")
    blocks: list[Any] = []
    if stl is not None:
        blocks += [
            Heading(text="Unusual readings"),
            Paragraph(
                text=f"{stl.n_anomalies:,} reading(s) differ strongly from the column's usual pattern "
                f"(threshold {stl.threshold:g})."
            ),
        ]
        if stl.points:
            blocks.append(
                _table(
                    "Most unusual readings",
                    ["Time", "Value", "Direction", "Score"],
                    [[p.timestamp, p.value, p.direction, p.score] for p in stl.points[:10]],
                )
            )
        if "anomalies" in charts:
            blocks.append(ChartBlock(chart_id="anomalies"))
    if multi is not None and multi.applicable and multi.points:
        blocks += [
            Heading(text="Unusual combinations across columns"),
            _table(
                "",
                ["Time", "Score", "Main contributors"],
                [
                    [
                        p.timestamp,
                        p.score,
                        ", ".join(clean(c.column, 40) for c in p.contributions[:2]),
                    ]
                    for p in multi.points[:10]
                ],
            ),
        ]
    if cp is not None:
        blocks += [
            Heading(text="Change points"),
            Paragraph(text=f"{cp.n_breaks} change(s) found. {clean(cp.p_value_note)}"),
        ]
        if cp.breaks:
            blocks.append(
                _table(
                    "Detected changes",
                    ["When", "Mean before", "Mean after", "Change", "Kind"],
                    [
                        [b.timestamp, b.mean_before, b.mean_after, b.mean_change, b.kind]
                        for b in cp.breaks[:15]
                    ],
                )
            )
        if "change_points" in charts:
            blocks.append(ChartBlock(chart_id="change_points"))
    return (
        Section(id="anomalies", title=SECTION_TITLES["anomalies"], blocks=blocks)
        if blocks
        else None
    )


def _forecast(inputs: ReportInputs, charts) -> Section | None:
    bt, fc = inputs.results.get("backtest"), inputs.results.get("forecast")
    if bt is None and fc is None:
        return None
    blocks: list[Any] = [Paragraph(text=f"Estimate. {ESTIMATE_NOTE}", style="estimate")]
    if bt is not None:
        blocks.append(
            _table(
                "Backtest: models scored on the end of the data",
                ["Model", "MAE", "RMSE", "sMAPE %"],
                [[s.model, s.mae, s.rmse, s.smape] for s in bt.scores],
            )
        )
        rec = recommend_model(bt)
        if rec.model:
            blocks.append(Paragraph(text=f"Recommended model: {rec.model}. {clean(rec.reason)}"))
        if "backtest" in charts:
            blocks.append(ChartBlock(chart_id="backtest"))
    if fc is not None:
        blocks += [
            Paragraph(
                text=f"Forecast of {fc.horizon} reading(s) with {fc.model}, shown with a "
                f"{fc.interval_level:.0%} prediction interval ({clean(fc.interval_method)}). "
                "This is an estimate.",
                style="estimate",
            )
        ]
        if "forecast" in charts:
            blocks.append(ChartBlock(chart_id="forecast"))
    return Section(id="forecast", title=SECTION_TITLES["forecast"], blocks=blocks)


def _whatif(inputs: ReportInputs, charts) -> Section | None:
    s = inputs.results.get("whatif")
    if s is None:
        return None
    pairs = [
        ("Share of the peak shifted", f"{s.shift_fraction:.0%}"),
        ("Peak threshold", f"percentile {s.percentile:g}"),
        ("Peak before / after", f"{num(s.peak_before)} / {num(s.peak_after)}"),
        (
            "Load factor before / after",
            f"{num(s.load_factor_before, 3)} / {num(s.load_factor_after, 3)}",
        ),
        ("Energy moved", num(s.energy_moved)),
        ("Energy not placed", num(s.energy_not_placed)),
    ]
    if s.cost is not None:
        pairs.append(
            ("Estimated saving", f"{s.cost.estimated_saving:,.2f} {clean(s.cost.currency, 10)}")
        )
    blocks: list[Any] = [
        Paragraph(
            text=f"Scenario estimate. {s.label.capitalize()}: not a measurement.", style="estimate"
        ),
        Heading(text="Assumptions"),
        Bullets(items=[clean(a) for a in s.assumptions]),
        _kv(*pairs),
    ]
    if "whatif" in charts:
        blocks.append(ChartBlock(chart_id="whatif"))
    return Section(id="whatif", title=SECTION_TITLES["whatif"], blocks=blocks)


def _investigation(inputs: ReportInputs) -> Section | None:
    res = inputs.results.get("investigation")
    if res is None:
        return None
    c = res.change
    pct = f" ({c.percent:+.1f}%)" if c.percent is not None else ""
    blocks: list[Any] = [
        Paragraph(
            text=f"Period {res.period_start} to {res.period_end}, compared with "
            f"{clean(res.baseline.description)}."
        ),
        Paragraph(
            text=f"Average {clean(res.column, MAX_NAME)} was {num(c.mean_period)} in the period "
            f"against {num(c.mean_baseline)} in the baseline: {c.direction}{pct}."
        ),
        _table(
            "Checks",
            ["Check", "Verdict", "Effect", "What it shows"],
            [
                [f.name.replace("_", " "), f.verdict, f.effect_size, clean(f.statement, MAX_CELL)]
                for f in res.factors
            ],
        ),
        Paragraph(text=clean(res.conclusion)),
    ]
    return Section(id="investigation", title=SECTION_TITLES["investigation"], blocks=blocks)


def _suggestions(inputs: ReportInputs) -> Section | None:
    out = inputs.suggestions
    items = getattr(out, "suggestions", None) if out is not None else None
    if not items:
        return None
    findings = {
        f.id: f
        for f in (
            inputs.results.get("findings").findings
            if inputs.results.get("findings") is not None
            else []
        )
    }
    blocks: list[Any] = [Paragraph(text=AI_NOTE, style="note")]
    for s in items:
        ref = []
        if s.finding_id:
            f = findings.get(s.finding_id)
            ref.append(
                f"finding '{clean(f.title, 100)}'" if f else f"finding {clean(s.finding_id, 60)}"
            )
        if s.log_entry_id is not None:
            ref.append(f"pipeline step {s.log_entry_id}")
        blocks += [
            Heading(text=clean(s.title, 120)),
            Paragraph(text=clean(f"{s.rationale} {s.action}".strip())),
            Paragraph(text="Refers to " + " and ".join(ref) + ".", style="note"),
        ]
    return Section(id="suggestions", title=SECTION_TITLES["suggestions"], blocks=blocks)


def _limitations(inputs: ReportInputs, version) -> Section:
    notes = [
        "Findings and statistics describe associations in the data. They do not show what caused "
        "anything.",
        "Forecasts and what-if scenarios are estimates, not measurements or guarantees.",
        f"All results describe dataset version {version.version_id}; other versions may differ.",
        "Anomaly and change-point detection depend on the chosen thresholds and sensitivity; "
        "p-values reported with change points are indicative only.",
        "Cleaning choices (such as filling short gaps) change the data; the pipeline log lists "
        "every change.",
    ]
    if inputs.suggestions is not None and getattr(inputs.suggestions, "suggestions", None):
        notes.append(
            "AI-generated suggestions come from a language model working only from "
            "computed findings; they are hypotheses to check."
        )
    if inputs.results.get("weather_link") is not None:
        notes.append(
            "Weather associations depend on the weather data supplied and its alignment "
            "with the readings."
        )
    return Section(
        id="limitations", title=SECTION_TITLES["limitations"], blocks=[Bullets(items=notes)]
    )


def recipe_hash(store: VersionStore, version_id: str, source: Mapping[str, Any]) -> str:
    """A hash of the recipe's content (operations, fingerprints, source), not of its timestamps."""
    recipe = store.to_recipe(version_id, source=dict(source))
    payload = {
        "raw_fingerprint": recipe.raw_fingerprint,
        "head_fingerprint": recipe.head_fingerprint,
        "steps": [
            {
                "version": s.version_id,
                "op": s.op.model_dump(),
                "fingerprint": s.fingerprint,
                "rows": s.rows,
                "columns": s.columns,
            }
            for s in recipe.steps
        ],
        "source": dict(source),
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _reproducibility(inputs: ReportInputs, version_id: str) -> Section:
    store = inputs.store
    chain = store.chain(version_id)
    env = library_versions()
    rows = [
        [
            v.version_id,
            clean(v.op.operation, 40) if v.op else "raw data",
            clean(json.dumps(v.op.params, default=str, sort_keys=True), 100) if v.op else "",
            v.fingerprint[:12],
            v.rows,
            v.columns,
        ]
        for v in chain
    ]
    blocks: list[Any] = [
        Paragraph(
            text="Replaying this recipe on the original file must reproduce each fingerprint "
            "below. The hash covers the operations, fingerprints and source, not dates.",
            style="note",
        ),
        _kv(
            ("Recipe content hash (SHA-256)", recipe_hash(store, version_id, inputs.source)),
            ("Raw data fingerprint", chain[0].fingerprint),
            ("This version's fingerprint", chain[-1].fingerprint),
            ("Steps", str(len(chain) - 1)),
        ),
        _table(
            "Operation chain",
            ["Version", "Operation", "Parameters", "Fingerprint", "Rows", "Columns"],
            rows,
        ),
        _kv(*[(k, v) for k, v in sorted(env.items())]),
    ]
    return Section(id="reproducibility", title=SECTION_TITLES["reproducibility"], blocks=blocks)


# -------------------------------------------------------------------------------- assemble


def _all_sections(inputs: ReportInputs, version_id: str, df: pd.DataFrame, charts):
    version = inputs.store.version(version_id)
    prof = inputs.profile if inputs.profile is not None else profile(df)
    char = inputs.characterization
    if char is None and inputs.column in df.columns:
        try:
            char = characterize(df, inputs.column)
        except DatanautError:
            char = None
    r = inputs.results
    return version, {
        "summary": _summary(inputs, version, df, prof, char),
        "quality": _quality(prof),
        "pipeline": _pipeline(inputs, version_id),
        "validation": _validation(inputs.validation),
        "findings": _findings(r, charts),
        "patterns": _patterns(inputs, charts),
        "anomalies": _anomalies(inputs, charts),
        "forecast": _forecast(inputs, charts),
        "whatif": _whatif(inputs, charts),
        "investigation": _investigation(inputs),
        "suggestions": _suggestions(inputs),
    }


def _assemble(inputs: ReportInputs, version_id: str | None):
    store = inputs.store
    version_id = version_id or store.head
    df = store.get(version_id)
    charts = _charts(inputs, df)
    version, found = _all_sections(inputs, version_id, df, charts)
    found["limitations"] = _limitations(inputs, version)
    found["reproducibility"] = _reproducibility(inputs, version_id)
    return version_id, version, found, charts


def available_sections(inputs: ReportInputs, version_id: str | None = None) -> dict[str, int]:
    """The sections that have something to show for this version, with their block counts."""
    _, _, found, _ = _assemble(inputs, version_id)
    return {sid: len(sec.blocks) for sid, sec in found.items() if sec is not None}


def build_report_data(
    inputs: ReportInputs,
    version_id: str | None = None,
    selected_sections: Collection[str] | None = None,
    *,
    generated: datetime | None = None,
) -> ReportData:
    """Assemble the report for one dataset version (default: the working version).

    `selected_sections` chooses which sections to include (default: every one that has data).
    Sections with nothing to show are omitted. `generated` fixes the date for reproducible output.
    """
    version_id, version, found, charts = _assemble(inputs, version_id)
    wanted = set(SECTION_TITLES) if selected_sections is None else set(selected_sections)
    sections = [
        found[sid] for sid in SECTION_TITLES if sid in wanted and found.get(sid) is not None
    ]
    used = {b.chart_id for s in sections for b in s.blocks if isinstance(b, ChartBlock)}
    return ReportData(
        cover=Cover(
            dataset_name=clean(inputs.dataset_name, MAX_NAME),
            generated=now_utc_text(generated),
            version_id=version.version_id,
            fingerprint=version.fingerprint,
        ),
        sections=sections,
        charts={k: v for k, v in charts.items() if k in used},
    )
