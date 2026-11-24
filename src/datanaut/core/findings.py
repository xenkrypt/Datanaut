# ruff: noqa: E501  (finding wording templates are long on purpose)
"""Findings: ranked, plain-language statements built from computed results by fixed rules.

Every finding says what the numbers show, carries its evidence, gets a 0 to 1 score based on
effect size, and names the chart that would display it (a name and parameters only; nothing is
drawn here). Wording describes association and pattern, never cause.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

from datanaut.core.anomalies import STLAnomalies
from datanaut.core.changepoints import ChangePoints
from datanaut.core.evidence import Evidence, analysis_log
from datanaut.core.forecast import Backtest, recommend_model
from datanaut.core.log import StepLog
from datanaut.core.patterns import EnergyMetrics, LoadProfile, PeakPeriods
from datanaut.core.weather import WeatherLink

CHANCE_TOP_HOURS = 3 / 24
CHANCE_TOP_WEEKDAYS = 2 / 7
MIN_PEAK_CONCENTRATION = 0.4
MIN_WEEKDAY_CONCENTRATION = 0.5
MIN_WEEKEND_DIFFERENCE_PCT = 5.0
MIN_WEATHER_ASSOCIATION = 0.3
FORBIDDEN_WORDS = ("caused", "because of", "due to", "leads to", "results in")


class ChartSpec(BaseModel):
    """Which chart would show a finding. A name and parameters only."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    """One ranked finding with its evidence."""

    id: str
    title: str
    statement: str
    category: str
    score: float = Field(ge=0, le=1)
    source: str
    numbers: dict[str, Any] = Field(default_factory=dict)
    evidence: Evidence
    chart: ChartSpec


class FindingsReport(BaseModel):
    """The ranked findings and what they were built from."""

    findings: list[Finding]
    n_candidates: int
    max_findings: int
    sources: list[str]
    evidence: Evidence
    log_entry: StepLog


def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _peak_findings(peaks: PeakPeriods) -> list[Finding]:
    out: list[Finding] = []
    if peaks.top_hours_share >= MIN_PEAK_CONCENTRATION:
        hours = ", ".join(peaks.top_hours)
        out.append(
            Finding(
                id="peak_hours",
                title=f"Peaks concentrate around {peaks.top_hours[0]}",
                statement=(
                    f"The highest {100 - peaks.percentile:g}% of readings of '{peaks.column}' fall mostly "
                    f"in {hours}: {peaks.top_hours_share:.0%} of them, against about "
                    f"{CHANCE_TOP_HOURS:.0%} if peaks were spread evenly through the day."
                ),
                category="peaks",
                score=_clip((peaks.top_hours_share - CHANCE_TOP_HOURS) / (1 - CHANCE_TOP_HOURS)),
                source="peak_periods",
                numbers={
                    "top_hours": peaks.top_hours,
                    "share": peaks.top_hours_share,
                    "chance": CHANCE_TOP_HOURS,
                },
                evidence=peaks.evidence,
                chart=ChartSpec(name="hour_of_day_peak_shares", params={"column": peaks.column}),
            )
        )
    if peaks.top_weekdays_share >= MIN_WEEKDAY_CONCENTRATION:
        days = " and ".join(peaks.top_weekdays)
        out.append(
            Finding(
                id="peak_weekdays",
                title=f"Peaks are more common on {days}",
                statement=(
                    f"{peaks.top_weekdays_share:.0%} of the highest readings of '{peaks.column}' occur on "
                    f"{days}, against about {CHANCE_TOP_WEEKDAYS:.0%} for two days if evenly spread."
                ),
                category="peaks",
                score=_clip(
                    (peaks.top_weekdays_share - CHANCE_TOP_WEEKDAYS) / (1 - CHANCE_TOP_WEEKDAYS)
                ),
                source="peak_periods",
                numbers={"top_weekdays": peaks.top_weekdays, "share": peaks.top_weekdays_share},
                evidence=peaks.evidence,
                chart=ChartSpec(name="weekday_peak_shares", params={"column": peaks.column}),
            )
        )
    return out


def _load_factor_finding(metrics: EnergyMetrics) -> list[Finding]:
    if not metrics.applicable or metrics.load_factor is None:
        return []
    lf = metrics.load_factor
    shape = (
        "peaky: demand is far above its average at the peak"
        if lf < 0.5
        else ("moderately peaky" if lf < 0.75 else "fairly flat: demand stays near its average")
    )
    return [
        Finding(
            id="load_factor",
            title=f"Load factor is {lf:.2f}",
            statement=(
                f"The load factor of '{metrics.column}' (average divided by peak) is {lf:.2f}, so "
                f"the profile is {shape}. Peak {metrics.peak:.4g} at {metrics.peak_time}; baseload "
                f"{metrics.baseload:.4g}."
            ),
            category="metrics",
            score=_clip(1 - lf),
            source="energy_metrics",
            numbers={
                "load_factor": lf,
                "peak": metrics.peak,
                "baseload": metrics.baseload,
                "mean": metrics.mean,
            },
            evidence=metrics.evidence,
            chart=ChartSpec(name="load_duration_curve", params={"column": metrics.column}),
        )
    ]


def _weekend_finding(weekday: LoadProfile) -> list[Finding]:
    def mean_of(days: range) -> float | None:
        buckets = [b for b in weekday.buckets if b.weekday in days and b.mean is not None and b.n]
        total = sum(b.n for b in buckets)
        return sum(b.mean * b.n for b in buckets) / total if total else None  # type: ignore[operator]

    weekdays, weekend = mean_of(range(0, 5)), mean_of(range(5, 7))
    if not weekdays or weekend is None:
        return []
    pct = 100 * (weekend - weekdays) / abs(weekdays)
    if abs(pct) < MIN_WEEKEND_DIFFERENCE_PCT:
        return []
    direction = "higher" if pct > 0 else "lower"
    return [
        Finding(
            id="weekday_vs_weekend",
            title=f"Weekends are {abs(pct):.0f}% {direction} than weekdays",
            statement=(
                f"Average '{weekday.column}' is {weekend:.4g} on weekends and {weekdays:.4g} on "
                f"weekdays: {abs(pct):.0f}% {direction} on weekends."
            ),
            category="patterns",
            score=_clip(abs(pct) / 50),
            source="load_profile",
            numbers={"weekday_mean": weekdays, "weekend_mean": weekend, "difference_pct": pct},
            evidence=weekday.evidence,
            chart=ChartSpec(name="weekday_profile", params={"column": weekday.column}),
        )
    ]


def _weather_finding(link: WeatherLink) -> list[Finding]:
    candidates = [r for r in (link.pearson, link.best_lag_pearson) if r is not None]
    if not candidates:
        return []
    r = max(candidates, key=abs)
    if abs(r) < MIN_WEATHER_ASSOCIATION:
        return []
    lagged = (
        link.best_lag_pearson is not None and r == link.best_lag_pearson and link.best_lag_steps
    )
    sign = "higher when it is warmer" if r > 0 else "higher when it is colder"
    when = f", strongest when the weather leads by {link.best_lag_hours:g} h" if lagged else ""
    return [
        Finding(
            id=f"weather:{link.variable}",
            title=f"Load is associated with {link.variable.replace('_', ' ')}",
            statement=(
                f"'{link.column}' is associated with {link.variable} (correlation {r:.2f}{when}): "
                f"{link.column} tends to be {sign}. This is an association in the data, not a "
                "demonstration that one drives the other."
            ),
            category="weather",
            score=_clip(abs(r)),
            source="weather_link",
            numbers={
                "pearson": link.pearson,
                "spearman": link.spearman,
                "best_lag_hours": link.best_lag_hours,
                "best_lag_pearson": link.best_lag_pearson,
            },
            evidence=link.evidence,
            chart=ChartSpec(
                name="scatter_with_weather",
                params={"column": link.column, "variable": link.variable},
            ),
        )
    ]


def _anomaly_finding(a: STLAnomalies) -> list[Finding]:
    if a.n_anomalies == 0:
        return []
    worst = a.points[0]
    day = a.days[0] if a.days else None
    where = f"; the day with the most is {day.date} ({day.count})" if day else ""
    return [
        Finding(
            id=f"anomalies:{a.column}",
            title=f"{a.n_anomalies} unusual readings in {a.column}",
            statement=(
                f"{a.n_anomalies} readings of '{a.column}' are unusual compared with its usual daily "
                f"pattern; the largest, {worst.value:.4g} at {worst.timestamp}, is "
                f"{worst.score:.1f} robust deviations {'above' if worst.direction == 'high' else 'below'} "
                f"what the pattern predicts{where}."
            ),
            category="anomalies",
            score=_clip(worst.score / 15),
            source="stl_anomalies",
            numbers={
                "n": a.n_anomalies,
                "worst_timestamp": worst.timestamp,
                "worst_score": worst.score,
                "days": [d.date for d in a.days[:3]],
            },
            evidence=a.evidence,
            chart=ChartSpec(name="anomaly_timeline", params={"column": a.column}),
        )
    ]


def _change_findings(cp: ChangePoints, limit: int = 2) -> list[Finding]:
    scored: list[tuple[float, Any]] = []
    for b in cp.breaks:
        if b.kind == "minor":
            continue
        mean_part = _clip(abs(b.mean_change_sd or 0) / 5) if b.kind in ("mean", "both") else 0.0
        var_part = (
            _clip(abs(math.log2(b.variance_ratio)) / 3)
            if b.variance_ratio and b.kind in ("variance", "both")
            else 0.0
        )
        scored.append((max(mean_part, var_part), b))
    out: list[Finding] = []
    for score, b in sorted(scored, key=lambda t: -t[0])[:limit]:
        if b.kind in ("mean", "both"):
            pct = 100 * b.mean_change / abs(b.mean_before) if b.mean_before else None
            title = f"Level {'rose' if b.mean_change > 0 else 'fell'} around {b.timestamp[:10]}"
            statement = (
                f"The typical level of '{cp.column}' changed around {b.timestamp}: the average moved from "
                f"{b.mean_before:.4g} before to {b.mean_after:.4g} after"
                + (f" ({pct:+.0f}%)" if pct is not None else "")
                + f", about {abs(b.mean_change_sd or 0):.1f} times the usual reading-to-reading noise."
            )
        else:
            title = f"Variability changed around {b.timestamp[:10]}"
            statement = (
                f"The spread of '{cp.column}' changed around {b.timestamp}: its variance became "
                f"{b.variance_ratio:.1f} times what it was before."
            )
        out.append(
            Finding(
                id=f"change_point:{b.timestamp}",
                title=title,
                statement=statement,
                category="change_points",
                score=score,
                source="change_points",
                numbers={
                    "timestamp": b.timestamp,
                    "mean_before": b.mean_before,
                    "mean_after": b.mean_after,
                    "mean_change_sd": b.mean_change_sd,
                    "variance_ratio": b.variance_ratio,
                },
                evidence=cp.evidence,
                chart=ChartSpec(name="change_point_timeline", params={"column": cp.column}),
            )
        )
    return out


def _forecast_finding(bt: Backtest) -> list[Finding]:
    rec = recommend_model(bt)
    if rec.model is None:
        return []
    score = next(s for s in bt.scores if s.model == rec.model)
    return [
        Finding(
            id="forecast_model",
            title=f"{rec.model.replace('_', ' ').title()} forecasts best",
            statement=(
                f"In {bt.n_folds} chronological backtests of {bt.horizon} steps, {rec.model} "
                f"averaged an error of {score.mae:.4g} (sMAPE {score.smape:.1f}%). {rec.reason} "
                "Forecasts are estimates, not guarantees."
            ),
            category="forecast",
            score=_clip(0.5 * (1 - score.smape / 100)),
            source="forecast_backtest",
            numbers={
                "model": rec.model,
                "mae": score.mae,
                "rmse": score.rmse,
                "smape": score.smape,
            },
            evidence=bt.evidence,
            chart=ChartSpec(
                name="forecast_with_interval", params={"column": bt.column, "model": rec.model}
            ),
        )
    ]


def generate_findings(
    *,
    peaks: PeakPeriods | None = None,
    metrics: EnergyMetrics | None = None,
    weekday_profile: LoadProfile | None = None,
    weather: WeatherLink | None = None,
    anomalies: STLAnomalies | None = None,
    change_points: ChangePoints | None = None,
    backtest: Backtest | None = None,
    max_findings: int = 5,
) -> FindingsReport:
    """Build findings from the results given, rank them by score and keep the top ones."""
    candidates: list[Finding] = []
    sources: list[str] = []
    for result, builder, name in (
        (peaks, _peak_findings, "peak_periods"),
        (metrics, _load_factor_finding, "energy_metrics"),
        (weekday_profile, _weekend_finding, "load_profile"),
        (weather, _weather_finding, "weather_link"),
        (anomalies, _anomaly_finding, "stl_anomalies"),
        (change_points, _change_findings, "change_points"),
        (backtest, _forecast_finding, "forecast_backtest"),
    ):
        if result is not None:
            sources.append(name)
            candidates += builder(result)  # type: ignore[operator]
    unique: dict[str, Finding] = {}
    for f in candidates:
        if f.id not in unique or f.score > unique[f.id].score:
            unique[f.id] = f
    ranked = sorted(unique.values(), key=lambda f: (-f.score, f.id))[:max_findings]
    columns = list(dict.fromkeys(c for f in ranked for c in f.evidence.columns))
    starts = [f.evidence.start for f in ranked if f.evidence.start]
    ends = [f.evidence.end for f in ranked if f.evidence.end]
    evidence = Evidence(
        columns=columns,
        start=min(starts) if starts else None,
        end=max(ends) if ends else None,
        n_observations=max((f.evidence.n_observations for f in ranked), default=0),
        method="findings_rules",
        parameters={"max_findings": max_findings, "sources": sources},
    )
    summary = (
        f"Built {len(candidates)} candidate finding(s) from {len(sources)} result(s); kept the top "
        f"{len(ranked)}" + (f": {ranked[0].title}." if ranked else ".")
    )
    return FindingsReport(
        findings=ranked,
        n_candidates=len(candidates),
        max_findings=max_findings,
        sources=sources,
        evidence=evidence,
        log_entry=analysis_log(
            "findings",
            summary,
            evidence,
            counts={"candidates": len(candidates), "kept": len(ranked)},
        ),
    )
