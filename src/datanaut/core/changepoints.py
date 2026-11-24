"""Change points: when the typical level or variability of a series shifts.

Anomaly detection finds unusual points; this finds the moments after which the series behaves
differently (a new level, or a calmer or noisier regime). It runs the PELT algorithm from
ruptures on a robustly standardized copy of the series. Effect sizes come first; the p-values
are labelled indicative because consecutive readings are correlated, which makes them too
optimistic. A change point marks when behavior changed, not why.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import ruptures as rpt
from pydantic import BaseModel, Field
from scipy import stats

from datanaut.core.cleaner import MAD_TO_SIGMA
from datanaut.core.evidence import (
    Evidence,
    analysis_log,
    finite_part,
    make_evidence,
    median_step,
    numeric_series,
)
from datanaut.core.log import StepLog
from datanaut.errors import AnalysisError

Sensitivity = Literal["low", "medium", "high"]
Cost = Literal["rbf", "l2"]

PENALTY_BASE: dict[str, float] = {"rbf": 3.0, "l2": 3.0}
"""Penalty = base x log(n) on the standardized signal, then scaled by the sensitivity factor.
Calibrated on simulated white noise (no false breaks) and on mean and variance shifts."""

SENSITIVITY_FACTOR: dict[str, float] = {"low": 2.0, "medium": 1.0, "high": 0.6}
MAX_POINTS = {"rbf": 500, "l2": 20000}
MEAN_EFFECT_SD = 1.0
"""A break counts as a change of level if the mean moves by at least this many noise sd."""
VARIANCE_RATIO = 2.0
"""A break counts as a change of spread if the variance ratio is at least this (or its inverse)."""
P_VALUE_NOTE = (
    "Indicative only: consecutive readings are correlated, so these p-values are too optimistic. "
    "Rely on the effect sizes."
)


class Segment(BaseModel):
    """A stretch of the series between two change points."""

    start: str
    end: str
    mean: float
    std: float | None = None
    n: int


class Break(BaseModel):
    """One change point and how big the change is."""

    timestamp: str
    mean_before: float
    mean_after: float
    mean_change: float
    mean_change_sd: float | None = None
    variance_ratio: float | None = None
    direction: Literal["increase", "decrease", "none"]
    kind: Literal["mean", "variance", "both", "minor"]
    p_welch: float | None = None
    p_levene: float | None = None
    p_ks: float | None = None
    p_values_indicative: bool = True


class ChangePoints(BaseModel):
    """Where a series changed level or variability, with the size of each change."""

    column: str
    cost: Cost
    sensitivity: Sensitivity
    penalty: float | None = None
    min_size: int
    n_breaks: int
    segments: list[Segment]
    breaks: list[Break]
    downsample_factor: int = 1
    downsampled: bool = False
    seasonally_adjusted: bool = False
    noise_sd: float | None = None
    rows_total: int
    excluded_missing: int
    notes: list[str] = Field(default_factory=list)
    p_value_note: str = P_VALUE_NOTE
    evidence: Evidence
    log_entry: StepLog


def _noise_sd(values: np.ndarray) -> float:
    """Noise spread from first differences, so level shifts do not inflate it."""
    d = np.diff(values)
    if len(d) == 0:
        return 0.0
    return float(MAD_TO_SIGMA * np.median(np.abs(d - np.median(d))) / math.sqrt(2))


def _remove_cycle(
    values: np.ndarray, stamps: pd.DatetimeIndex, period: int, step: pd.Timedelta
) -> np.ndarray:
    """Subtract the average value at each position in the cycle (positions come from time)."""
    phase = np.rint((stamps - stamps[0]) / step).astype(np.int64) % period
    sums = np.bincount(phase, weights=values, minlength=period)
    counts = np.bincount(phase, minlength=period)
    pattern = np.divide(sums, counts, out=np.zeros(period), where=counts > 0)
    return values - pattern[phase] + pattern[counts > 0].mean()


def default_penalty(n: int, cost: str = "rbf", sensitivity: str = "medium") -> float:
    """The penalty used when none is given: base x log(n) x sensitivity factor."""
    return PENALTY_BASE[cost] * math.log(max(n, 2)) * SENSITIVITY_FACTOR[sensitivity]


def _empty(
    column: str,
    cost: Cost,
    sensitivity: Sensitivity,
    min_size: int,
    values: pd.Series,
    rows_total: int,
    excluded: int,
    note: str,
) -> ChangePoints:
    evidence = make_evidence(
        values.index,
        [column],
        len(values),
        f"pelt_{cost}",
        {"cost": cost, "sensitivity": sensitivity, "min_size": min_size},
    )
    segments = []
    if len(values):
        segments = [
            Segment(
                start=str(values.index[0]),
                end=str(values.index[-1]),
                mean=float(values.mean()),
                std=float(values.std()) if len(values) > 1 else None,
                n=len(values),
            )
        ]
    return ChangePoints(
        column=column,
        cost=cost,
        sensitivity=sensitivity,
        min_size=min_size,
        n_breaks=0,
        segments=segments,
        breaks=[],
        rows_total=rows_total,
        excluded_missing=excluded,
        notes=[note],
        evidence=evidence,
        log_entry=analysis_log(
            "change_points",
            f"Change points in '{column}': none looked for. {note}",
            evidence,
            counts={"breaks": 0, "observations": len(values), "excluded_missing": excluded},
            rows=rows_total,
        ),
    )


def detect_change_points(
    df: pd.DataFrame,
    column: str,
    penalty: float | None = None,
    min_size: int | None = None,
    *,
    sensitivity: Sensitivity = "medium",
    cost: Cost = "rbf",
    period: int | None = None,
    max_points: int | None = None,
) -> ChangePoints:
    """Find moments when the level or variability of `column` shifts (PELT, via ruptures).

    The series is standardized (median and a noise sd taken from first differences), so the
    default penalty depends only on the number of points: penalty = 3 x log(n) x a sensitivity
    factor (low 2.0, medium 1.0, high 0.6). Pass `penalty` to set it directly, in those
    standardized units. The default `rbf` cost reacts to changes in level and in spread; `l2`
    sees level only but is much faster. `min_size` is the shortest segment, in observations
    (default max(10, n/50)). With `period`, the repeating pattern is removed first so that a
    daily cycle is not mistaken for changes. Long series are averaged in blocks before the
    search (at most `max_points`, default 500 for rbf and 20000 for l2) and the notes say so.
    Missing values are left out and counted.
    """
    if sensitivity not in SENSITIVITY_FACTOR:
        raise AnalysisError("sensitivity must be 'low', 'medium' or 'high'.")
    if cost not in PENALTY_BASE:
        raise AnalysisError("cost must be 'rbf' or 'l2'.")
    if penalty is not None and penalty <= 0:
        raise AnalysisError("penalty must be positive.")
    if period is not None and period < 2:
        raise AnalysisError("period must be at least 2 readings per cycle.")
    raw = numeric_series(df, column)
    used, excluded = finite_part(raw)
    used = used.groupby(level=0).mean().sort_index()
    n = len(used)
    min_orig = int(min_size) if min_size is not None else max(10, n // 50)
    if min_orig < 2:
        raise AnalysisError("min_size must be at least 2 observations.")
    if n < 2 * min_orig + 1:
        return _empty(
            column,
            cost,
            sensitivity,
            min_orig,
            used,
            len(raw),
            excluded,
            f"Only {n} observation(s); at least {2 * min_orig + 1} are needed to look for changes "
            f"with a minimum segment of {min_orig}.",
        )
    values = used.to_numpy(dtype=float)
    stamps = used.index
    notes: list[str] = []
    adjusted = values
    if period is not None:
        step = median_step(stamps)
        if step is None or step <= pd.Timedelta(0):
            raise AnalysisError("A period needs at least two distinct timestamps.")
        adjusted = _remove_cycle(values, stamps, period, step)
        notes.append(f"The repeating pattern (period {period}) was removed before the search.")
    sigma = _noise_sd(adjusted)
    if sigma <= 0 or not np.isfinite(sigma):
        return _empty(
            column,
            cost,
            sensitivity,
            min_orig,
            used,
            len(raw),
            excluded,
            "The series is constant, so there is nothing to detect.",
        )
    signal = (adjusted - np.median(adjusted)) / sigma

    limit = max_points or MAX_POINTS[cost]
    factor = max(1, math.ceil(n / limit))
    if factor > 1:
        usable = (n // factor) * factor
        work = signal[:usable].reshape(-1, factor).mean(axis=1)
        if usable < n:  # the leftover observations form one last, shorter block
            work = np.append(work, signal[usable:].mean())
        notes.append(
            f"The series was averaged in blocks of {factor} observations "
            f"({n} -> {len(work)} points) "
            "for speed; break locations are accurate to about one block."
        )
    else:
        work = signal
    min_work = max(2, math.ceil(min_orig / factor))
    pen = float(penalty) if penalty is not None else default_penalty(len(work), cost, sensitivity)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ends = (
            rpt.Pelt(model=cost, min_size=min_work, jump=1)
            .fit(work.reshape(-1, 1))
            .predict(pen=pen)
        )
    cut_work = [e for e in ends if e < len(work)]
    cuts = [min(c * factor, n - 1) for c in cut_work]  # first observation of each new segment

    bounds = [0, *cuts, n]
    segments = [
        Segment(
            start=str(stamps[lo]),
            end=str(stamps[hi - 1]),
            mean=float(values[lo:hi].mean()),
            std=float(values[lo:hi].std(ddof=1)) if hi - lo > 1 else None,
            n=hi - lo,
        )
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)
    ]
    breaks: list[Break] = []
    for i, cut in enumerate(cuts):
        lo, hi = bounds[i], bounds[i + 2]
        before, after = adjusted[lo:cut], adjusted[cut:hi]
        raw_before, raw_after = values[lo:cut], values[cut:hi]
        change = float(raw_after.mean() - raw_before.mean())
        change_sd = change / sigma
        v_before, v_after = float(before.var(ddof=1)), float(after.var(ddof=1))
        ratio = v_after / v_before if v_before > 0 else None
        mean_moved = abs(change_sd) >= MEAN_EFFECT_SD
        spread_moved = ratio is not None and (
            ratio >= VARIANCE_RATIO or ratio <= 1 / VARIANCE_RATIO
        )
        kind = (
            "both"
            if mean_moved and spread_moved
            else "mean"
            if mean_moved
            else "variance"
            if spread_moved
            else "minor"
        )
        p_welch = p_levene = p_ks = None
        if len(before) >= 3 and len(after) >= 3:
            p_welch = _p(stats.ttest_ind(after, before, equal_var=False).pvalue)
            p_levene = _p(stats.levene(before, after).pvalue)
            p_ks = _p(stats.ks_2samp(before, after).pvalue)
        breaks.append(
            Break(
                timestamp=str(stamps[cut]),
                mean_before=float(raw_before.mean()),
                mean_after=float(raw_after.mean()),
                mean_change=change,
                mean_change_sd=float(change_sd),
                variance_ratio=float(ratio) if ratio is not None else None,
                direction="increase"
                if change > 0 and mean_moved
                else "decrease"
                if change < 0 and mean_moved
                else "none",
                kind=kind,
                p_welch=p_welch,
                p_levene=p_levene,
                p_ks=p_ks,
            )
        )
    evidence = make_evidence(
        stamps,
        [column],
        n,
        f"pelt_{cost}",
        {
            "cost": cost,
            "penalty": pen,
            "penalty_given": penalty is not None,
            "sensitivity": sensitivity,
            "min_size": min_orig,
            "period": period,
            "downsample_factor": factor,
        },
    )
    summary = (
        f"Change points in '{column}' ({cost}, {sensitivity} sensitivity): {len(breaks)} found"
        + (f", at {', '.join(b.timestamp for b in breaks[:3])}" if breaks else "")
        + f"; {n} observation(s) used, {excluded} excluded as missing."
    )
    return ChangePoints(
        column=column,
        cost=cost,
        sensitivity=sensitivity,
        penalty=pen,
        min_size=min_orig,
        n_breaks=len(breaks),
        segments=segments,
        breaks=breaks,
        downsample_factor=factor,
        downsampled=factor > 1,
        seasonally_adjusted=period is not None,
        noise_sd=float(sigma),
        rows_total=len(raw),
        excluded_missing=excluded,
        notes=notes,
        evidence=evidence,
        log_entry=analysis_log(
            "change_points",
            summary,
            evidence,
            counts={"breaks": len(breaks), "observations": n, "excluded_missing": excluded},
            rows=len(raw),
        ),
    )


def _p(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
