"""Investigation: what is associated with a change in a chosen period?

Given a period (or a change point someone clicked), `investigate` compares it with a baseline
and runs the same fixed set of checks every time: the change in level, weekday versus weekend
mix, a shift in the daily peak hour, a nearby change point, and the association with weather
and with other columns. Each check is a plain function that returns an effect size and a
verdict, so everything runs without any AI. The conclusion states association in the data, never
cause, and only mentions factors that have computed support.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from datanaut.core.changepoints import ChangePoints
from datanaut.core.evidence import Evidence, analysis_log, make_evidence, numeric_series
from datanaut.core.log import StepLog
from datanaut.core.weather import align_weather
from datanaut.errors import AnalysisError

Baseline = Literal["previous", "same_weekdays"]
Verdict = Literal["supported", "weak", "no evidence"]

MIN_ROWS = 10
BASELINE_WEEKS = 4
LEVEL_SUPPORTED_D, LEVEL_WEAK_D = 0.5, 0.2
"""Standardized mean difference (Cohen's d) for a medium and a small effect."""
MIX_SUPPORTED, MIX_WEAK = 0.25, 0.10
"""Change in the weekend share of observations."""
PEAK_SHIFT_SUPPORTED, PEAK_SHIFT_WEAK = 2.0, 1.0
"""Shift of the typical daily peak hour, in hours."""
CORR_SUPPORTED, CORR_WEAK = 0.5, 0.3
"""Absolute correlation within the period."""
MIN_PEAK_CONSISTENCY = 0.5
"""Daily peak hours must agree this much (0 to 1) in both periods before a shift is judged."""
NO_CAUSE = "This describes association in the data; it does not show what caused the change."


class ChangeSummary(BaseModel):
    """How the period differs from the baseline."""

    mean_period: float
    mean_baseline: float
    difference: float
    percent: float | None = None
    effect_size_d: float | None = None
    direction: Literal["higher", "lower", "unchanged"]
    n_period: int
    n_baseline: int


class Factor(BaseModel):
    """One candidate factor: its effect size, a verdict, and the numbers behind it."""

    name: str
    verdict: Verdict
    effect_size: float | None = None
    effect_measure: str
    numbers: dict[str, Any] = Field(default_factory=dict)
    statement: str


class BaselineInfo(BaseModel):
    """Which data the period was compared with."""

    kind: Baseline
    description: str
    start: str
    end: str
    n: int


class InvestigationResult(BaseModel):
    """The period, the change, every checked factor, and a conclusion about association."""

    column: str
    period_start: str
    period_end: str
    baseline: BaselineInfo
    change: ChangeSummary
    factors: list[Factor]
    supported: list[str]
    weak: list[str]
    conclusion: str
    evidence: Evidence
    log_entry: StepLog


# ----------------------------------------------------------------------- helpers


def _verdict(effect: float | None, supported: float, weak: float) -> Verdict:
    if effect is None or not math.isfinite(effect):
        return "no evidence"
    effect = round(effect, 9)
    return "supported" if effect >= supported else "weak" if effect >= weak else "no evidence"


def _pearson(a: pd.Series, b: pd.Series) -> tuple[float | None, int]:
    both = pd.concat([a, b], axis=1).dropna()
    n = len(both)
    if n < MIN_ROWS or both.iloc[:, 0].nunique() < 2 or both.iloc[:, 1].nunique() < 2:
        return None, n
    return float(stats.pearsonr(both.iloc[:, 0], both.iloc[:, 1])[0]), n


def _circular_hour(hours: np.ndarray) -> tuple[float, float]:
    """Mean hour of day (circular) and how consistent the hours are (0 scattered, 1 identical)."""
    angle = 2 * np.pi * hours / 24
    x, y = np.cos(angle).mean(), np.sin(angle).mean()
    return float((np.arctan2(y, x) * 24 / (2 * np.pi)) % 24), float(np.hypot(x, y))


def _daily_peak_hours(s: pd.Series) -> np.ndarray:
    peaks = s.groupby(s.index.normalize()).idxmax()
    return np.array([ts.hour for ts in peaks.dropna()], dtype=float)


# ------------------------------------------------------------------- the checks


def check_level_change(period: pd.Series, baseline: pd.Series) -> ChangeSummary:
    """Difference in mean level, as an absolute change, a percentage and Cohen's d."""
    mp, mb = float(period.mean()), float(baseline.mean())
    pooled = math.sqrt(
        ((len(period) - 1) * period.var(ddof=1) + (len(baseline) - 1) * baseline.var(ddof=1))
        / max(len(period) + len(baseline) - 2, 1)
    )
    d = (mp - mb) / pooled if pooled > 0 else None
    direction = "unchanged" if mp == mb else ("higher" if mp > mb else "lower")
    return ChangeSummary(
        mean_period=mp,
        mean_baseline=mb,
        difference=mp - mb,
        percent=100 * (mp - mb) / abs(mb) if mb else None,
        effect_size_d=d,
        direction=direction,  # type: ignore[arg-type]
        n_period=len(period),
        n_baseline=len(baseline),
    )


def check_weekday_mix(period: pd.Series, baseline: pd.Series) -> Factor:
    """Did the share of weekend observations differ between period and baseline?"""
    share_p = float((period.index.day_of_week >= 5).mean())
    share_b = float((baseline.index.day_of_week >= 5).mean())
    effect = abs(share_p - share_b)
    verdict = _verdict(effect, MIX_SUPPORTED, MIX_WEAK)
    text = (
        f"Weekend readings make up {share_p:.0%} of the period and {share_b:.0%} of the baseline."
    )
    return Factor(
        name="weekday_weekend_mix",
        verdict=verdict,
        effect_size=effect,
        effect_measure="change in weekend share (0 to 1)",
        numbers={"weekend_share_period": share_p, "weekend_share_baseline": share_b},
        statement=text,
    )


def check_peak_hour_shift(period: pd.Series, baseline: pd.Series) -> Factor:
    """Did the typical hour of the daily maximum move?"""
    hp, hb = _daily_peak_hours(period), _daily_peak_hours(baseline)
    if len(hp) < 2 or len(hb) < 2:
        return Factor(
            name="peak_hour_shift",
            verdict="no evidence",
            effect_measure="shift of the daily peak hour (hours)",
            numbers={"days_period": len(hp), "days_baseline": len(hb)},
            statement="Too few days to compare the daily peak hour.",
        )
    (mp, rp), (mb, rb) = _circular_hour(hp), _circular_hour(hb)
    shift = abs((mp - mb + 12) % 24 - 12)
    numbers = {
        "typical_peak_hour_period": mp,
        "typical_peak_hour_baseline": mb,
        "consistency_period": rp,
        "consistency_baseline": rb,
    }
    if min(rp, rb) < MIN_PEAK_CONSISTENCY:
        return Factor(
            name="peak_hour_shift",
            verdict="no evidence",
            effect_measure="shift of the daily peak hour (hours)",
            numbers=numbers,
            statement=(
                "The daily peak hour is not consistent from day to day, so a shift in it cannot "
                "be judged."
            ),
        )
    return Factor(
        name="peak_hour_shift",
        verdict=_verdict(shift, PEAK_SHIFT_SUPPORTED, PEAK_SHIFT_WEAK),
        effect_size=float(shift),
        effect_measure="shift of the daily peak hour (hours)",
        numbers=numbers,
        statement=(
            f"The typical daily peak is around {mp:04.1f}h in the period and {mb:04.1f}h in the "
            f"baseline (a shift of {shift:.1f} h)."
        ),
    )


def check_change_point(
    change_points: ChangePoints, start: pd.Timestamp, end: pd.Timestamp
) -> Factor:
    """Does a detected change point fall inside, or close to, the period?"""
    margin = (end - start) / 2
    best: tuple[float, Any, str] | None = None
    for b in change_points.breaks:
        t = pd.Timestamp(b.timestamp)
        if start <= t <= end:
            where = "inside"
        elif start - margin <= t <= end + margin:
            where = "near"
        else:
            continue
        size = abs(b.mean_change_sd or 0.0)
        if best is None or (where == "inside", size) > (best[2] == "inside", best[0]):
            best = (size, b, where)
    if best is None:
        return Factor(
            name="change_point",
            verdict="no evidence",
            effect_measure="mean change at the break (noise sd)",
            numbers={"breaks_checked": change_points.n_breaks},
            statement="No change point falls in or near the period.",
        )
    size, b, where = best
    return Factor(
        name="change_point",
        verdict="supported" if where == "inside" else "weak",
        effect_size=float(size),
        effect_measure="mean change at the break (noise sd)",
        numbers={
            "timestamp": b.timestamp,
            "location": where,
            "mean_before": b.mean_before,
            "mean_after": b.mean_after,
        },
        statement=(
            f"A change point at {b.timestamp} falls {where} the period: the average moved from "
            f"{b.mean_before:.4g} to {b.mean_after:.4g}."
        ),
    )


def check_association(
    name: str,
    label: str,
    value_p: pd.Series,
    value_b: pd.Series,
    other_p: pd.Series,
    other_b: pd.Series,
) -> Factor:
    """Correlation of the column with another series, in the period and in the baseline."""
    r_p, n_p = _pearson(value_p, other_p)
    r_b, n_b = _pearson(value_b, other_b)
    effect = abs(r_p) if r_p is not None else None
    shift = (r_p - r_b) if r_p is not None and r_b is not None else None
    mean_change = (
        float(other_p.mean() - other_b.mean())
        if other_p.notna().any() and other_b.notna().any()
        else None
    )
    if r_p is None:
        text = f"Too little variation or overlap to correlate with {label} in the period."
    else:
        text = (
            f"In the period the column is associated with {label} (correlation {r_p:.2f}"
            + (f"; baseline {r_b:.2f}, change {shift:+.2f}" if shift is not None else "")
            + ")."
        )
    return Factor(
        name=name,
        verdict=_verdict(effect, CORR_SUPPORTED, CORR_WEAK),
        effect_size=effect,
        effect_measure="absolute correlation in the period",
        numbers={
            "correlation_period": r_p,
            "correlation_baseline": r_b,
            "correlation_change": shift,
            "pairs_period": n_p,
            "pairs_baseline": n_b,
            "mean_change_of_other": mean_change,
        },
        statement=text,
    )


# ------------------------------------------------------------------- the workflow


def _slices(
    s: pd.Series, start: pd.Timestamp, end: pd.Timestamp, baseline: Baseline
) -> tuple[pd.Series, pd.Series, BaselineInfo]:
    period = s.loc[start:end]
    if baseline == "previous":
        before = s.loc[: start - pd.Timedelta(microseconds=1)]
        base = before.iloc[-len(period) :] if len(period) else before.iloc[:0]
        desc = "the previous period (the same number of observations)"
    elif baseline == "same_weekdays":
        window = s.loc[
            start - pd.Timedelta(days=7 * BASELINE_WEEKS) : start - pd.Timedelta(microseconds=1)
        ]
        base = window[window.index.day_of_week.isin(sorted(set(period.index.day_of_week)))]
        desc = f"the same weekdays in the {BASELINE_WEEKS} weeks before the period"
    else:
        raise AnalysisError("baseline must be 'previous' or 'same_weekdays'.")
    info = BaselineInfo(
        kind=baseline,
        description=desc,
        start=str(base.index.min()) if len(base) else "",
        end=str(base.index.max()) if len(base) else "",
        n=len(base),
    )
    return period, base, info


def investigate(
    df: pd.DataFrame,
    column: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    baseline: Baseline = "previous",
    other_columns: list[str] | None = None,
    weather_df: pd.DataFrame | None = None,
    change_points: ChangePoints | None = None,
    *,
    weather_variable: str | None = None,
) -> InvestigationResult:
    """Compare a period with a baseline and report which factors are associated with the change.

    Runs the fixed checks (level change, weekend mix, daily peak hour, change point, weather,
    other columns). A factor is "supported", "weak" or "no evidence" by stated effect-size
    thresholds; the conclusion names only supported and weak factors and always states
    association, not cause.
    """
    s = numeric_series(df, column)
    s = s[np.isfinite(s.to_numpy())]
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if t0 >= t1:
        raise AnalysisError("The period must start before it ends.")
    period, base, info = _slices(s, t0, t1, baseline)
    if len(period) < MIN_ROWS or len(base) < MIN_ROWS:
        raise AnalysisError(
            f"Not enough data to compare: {len(period)} observation(s) in the period and "
            f"{len(base)} in the baseline; at least {MIN_ROWS} each are needed."
        )
    change = check_level_change(period, base)
    factors = [check_weekday_mix(period, base), check_peak_hour_shift(period, base)]
    if change_points is not None:
        factors.append(check_change_point(change_points, t0, t1))
    used_columns = [column]
    if weather_df is not None:
        variable = weather_variable or (
            "temperature_2m"
            if "temperature_2m" in weather_df.columns
            else str(weather_df.columns[0])
        )
        aligned = align_weather(df[[column]], weather_df[[variable]]).df[variable]
        factors.append(
            check_association(
                f"weather:{variable}",
                f"weather '{variable}'",
                period,
                base,
                aligned.reindex(period.index),
                aligned.reindex(base.index),
            )
        )
        used_columns.append(variable)
    for other in other_columns or []:
        o = numeric_series(df, other)
        factors.append(
            check_association(
                f"column:{other}",
                f"column '{other}'",
                period,
                base,
                o.reindex(period.index),
                o.reindex(base.index),
            )
        )
        used_columns.append(other)

    supported = [f.name for f in factors if f.verdict == "supported"]
    weak = [f.name for f in factors if f.verdict == "weak"]
    pct = f" ({change.percent:+.1f}%)" if change.percent is not None else ""
    size = (
        f", standardized difference {change.effect_size_d:+.2f}"
        if change.effect_size_d is not None
        else ""
    )
    parts = [
        f"Compared with {info.description}, '{column}' averaged {change.mean_period:.4g} between "
        f"{t0} and {t1}, {change.direction} than the baseline average of {change.mean_baseline:.4g}"
        f"{pct}{size}."
    ]
    by_name = {f.name: f for f in factors}
    if supported:
        parts.append(
            "Associated with this change in the data: "
            + "; ".join(by_name[n].statement for n in supported)
        )
    if weak:
        parts.append("Weakly associated: " + "; ".join(by_name[n].statement for n in weak))
    if not supported and not weak:
        parts.append("None of the checked factors shows evidence of an association with it.")
    parts.append(NO_CAUSE)
    conclusion = " ".join(parts)

    evidence = make_evidence(
        period.index,
        used_columns,
        len(period),
        "investigation_fixed_checks",
        {
            "baseline": baseline,
            "baseline_n": len(base),
            "checks": [f.name for f in factors],
            "thresholds": {
                "level_d": [LEVEL_SUPPORTED_D, LEVEL_WEAK_D],
                "weekend_share": [MIX_SUPPORTED, MIX_WEAK],
                "peak_hour_shift": [PEAK_SHIFT_SUPPORTED, PEAK_SHIFT_WEAK],
                "correlation": [CORR_SUPPORTED, CORR_WEAK],
            },
        },
    )
    summary = (
        f"Investigated '{column}' {t0} to {t1} against {info.description}: level {change.direction}"
        f"{pct}; {len(supported)} factor(s) supported, {len(weak)} weak. {NO_CAUSE}"
    )
    return InvestigationResult(
        column=column,
        period_start=str(t0),
        period_end=str(t1),
        baseline=info,
        change=change,
        factors=factors,
        supported=supported,
        weak=weak,
        conclusion=conclusion,
        evidence=evidence,
        log_entry=analysis_log(
            "investigate",
            summary,
            evidence,
            counts={
                "period_observations": len(period),
                "baseline_observations": len(base),
                "supported": len(supported),
                "weak": len(weak),
            },
            rows=len(df),
        ),
    )
