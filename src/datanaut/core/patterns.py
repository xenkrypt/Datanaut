"""Patterns and energy metrics: when demand happens, how peaky it is, and where peaks cluster.

Every result carries an evidence block (columns, time range, observation count, method and
parameters) and a read-only log entry. Missing and infinite values are excluded and counted;
nothing here changes the data, and nothing claims a cause.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from datanaut.core.evidence import (
    Evidence,
    analysis_log,
    finite_part,
    make_evidence,
    numeric_series,
)
from datanaut.core.log import StepLog
from datanaut.errors import AnalysisError

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
TOP_HOURS = 3
TOP_WEEKDAYS = 2

ProfileBy = Literal["hour", "weekday", "hour_x_weekday"]


# --------------------------------------------------------------------------- load profile


class ProfileBucket(BaseModel):
    """One bucket of a load profile: its label, count and spread of values."""

    label: str
    hour: int | None = None
    weekday: int | None = None
    n: int
    mean: float | None = None
    median: float | None = None
    p25: float | None = None
    p75: float | None = None


class LoadProfile(BaseModel):
    """Average load by hour of day, weekday, or both."""

    column: str
    by: ProfileBy
    buckets: list[ProfileBucket]
    rows_total: int
    excluded: int
    used: int
    highest_bucket: str | None = None
    lowest_bucket: str | None = None
    evidence: Evidence
    log_entry: StepLog


def _bucket_keys(
    index: pd.DatetimeIndex, by: ProfileBy
) -> tuple[np.ndarray, list[tuple[int | None, int | None]]]:
    """A key per observation, and every possible bucket as (hour, weekday)."""
    hours, days = index.hour.to_numpy(), index.day_of_week.to_numpy()
    if by == "hour":
        return hours, [(h, None) for h in range(24)]
    if by == "weekday":
        return days, [(None, d) for d in range(7)]
    if by == "hour_x_weekday":
        return days * 24 + hours, [(h, d) for d in range(7) for h in range(24)]
    raise AnalysisError(f"Unknown profile grouping '{by}'. Use hour, weekday or hour_x_weekday.")


def _bucket_label(hour: int | None, weekday: int | None) -> str:
    if hour is not None and weekday is not None:
        return f"{WEEKDAYS[weekday]} {hour:02d}:00"
    if hour is not None:
        return f"{hour:02d}:00"
    return WEEKDAYS[weekday]  # type: ignore[index]


def load_profile(df: pd.DataFrame, column: str, by: ProfileBy = "hour") -> LoadProfile:
    """Mean, median and the 25th to 75th percentile range of `column` per time-of-use bucket.

    Buckets with no observations are listed with n=0, so gaps in coverage stay visible.
    """
    values = numeric_series(df, column)
    used, excluded = finite_part(values)
    keys, buckets = _bucket_keys(used.index, by)
    grouped = used.groupby(keys)
    stats = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "median": grouped.median(),
            "p25": grouped.quantile(0.25),
            "p75": grouped.quantile(0.75),
        }
    )
    out: list[ProfileBucket] = []
    for hour, weekday in buckets:
        key = (
            (weekday * 24 + hour) if by == "hour_x_weekday" else (hour if by == "hour" else weekday)
        )
        if key in stats.index:
            row = stats.loc[key]
            out.append(
                ProfileBucket(
                    label=_bucket_label(hour, weekday),
                    hour=hour,
                    weekday=weekday,
                    n=int(row["n"]),
                    mean=float(row["mean"]),
                    median=float(row["median"]),
                    p25=float(row["p25"]),
                    p75=float(row["p75"]),
                )
            )
        else:
            out.append(
                ProfileBucket(label=_bucket_label(hour, weekday), hour=hour, weekday=weekday, n=0)
            )
    seen = [b for b in out if b.mean is not None]
    highest = max(seen, key=lambda b: b.mean).label if seen else None  # type: ignore[arg-type,return-value]
    lowest = min(seen, key=lambda b: b.mean).label if seen else None  # type: ignore[arg-type,return-value]
    evidence = make_evidence(used.index, [column], len(used), "load_profile", {"by": by})
    summary = (
        f"Load profile of '{column}' by {by.replace('_', ' ')}: "
        + (f"highest {highest}, lowest {lowest}; " if seen else "no usable observations; ")
        + f"{len(used)} observation(s) used, {excluded} excluded as missing or not finite."
    )
    return LoadProfile(
        column=column,
        by=by,
        buckets=out,
        rows_total=len(values),
        excluded=excluded,
        used=len(used),
        highest_bucket=highest,
        lowest_bucket=lowest,
        evidence=evidence,
        log_entry=analysis_log(
            "load_profile",
            summary,
            evidence,
            counts={"observations": len(used), "excluded": excluded, "buckets": len(out)},
            rows=len(values),
        ),
    )


# ------------------------------------------------------------------------- energy metrics


class EnergyMetrics(BaseModel):
    """Peak demand, baseload and load factor of a consumption column."""

    column: str
    applicable: bool
    reason: str | None = None
    peak: float | None = None
    peak_time: str | None = None
    baseload: float | None = None
    baseload_percentile: float = 5.0
    mean: float | None = None
    load_factor: float | None = None
    negative_values: int = 0
    rows_total: int
    excluded: int = 0
    used: int = 0
    notes: list[str] = Field(default_factory=list)
    evidence: Evidence
    log_entry: StepLog


def energy_metrics(
    df: pd.DataFrame,
    column: str,
    *,
    consumption: bool = False,
    baseload_percentile: float = 5.0,
) -> EnergyMetrics:
    """Peak, baseload (a low percentile), mean and load factor (mean / peak).

    Only runs for a column the caller marks as consumption (`consumption=True`); otherwise the
    result says it is not applicable instead of guessing.
    """
    if not 0 < baseload_percentile < 50:
        raise AnalysisError("baseload_percentile must be between 0 and 50.")
    values = numeric_series(df, column)
    used, excluded = finite_part(values)
    params = {"baseload_percentile": baseload_percentile, "consumption": consumption}
    if not consumption:
        reason = (
            f"'{column}' is not marked as a consumption column, so energy metrics were skipped."
        )
        evidence = make_evidence(None, [column], 0, "energy_metrics", params)
        return EnergyMetrics(
            column=column,
            applicable=False,
            reason=reason,
            baseload_percentile=baseload_percentile,
            rows_total=len(values),
            evidence=evidence,
            log_entry=analysis_log(
                "energy_metrics", reason, evidence, counts={"applicable": 0}, rows=len(values)
            ),
        )
    if used.empty:
        raise AnalysisError(f"Column '{column}' has no usable values for energy metrics.")
    peak = float(used.max())
    peak_time = str(used.idxmax())
    baseload = float(np.percentile(used.to_numpy(), baseload_percentile))
    mean = float(used.mean())
    load_factor = mean / peak if peak > 0 else None
    negative = int((used < 0).sum())
    notes = [f"{negative} negative value(s) are included; check the data."] if negative else []
    if load_factor is None:
        notes.append("Peak is not positive, so the load factor is undefined.")
    evidence = make_evidence(used.index, [column], len(used), "energy_metrics", params)
    summary = (
        f"Energy metrics of '{column}': peak {peak:.4g} at {peak_time}, baseload {baseload:.4g} "
        f"({baseload_percentile:g}th percentile), mean {mean:.4g}"
        + (f", load factor {load_factor:.3f}" if load_factor is not None else "")
        + f"; {len(used)} observation(s) used, {excluded} excluded."
    )
    return EnergyMetrics(
        column=column,
        applicable=True,
        peak=peak,
        peak_time=peak_time,
        baseload=baseload,
        baseload_percentile=baseload_percentile,
        mean=mean,
        load_factor=load_factor,
        negative_values=negative,
        rows_total=len(values),
        excluded=excluded,
        used=len(used),
        notes=notes,
        evidence=evidence,
        log_entry=analysis_log(
            "energy_metrics",
            summary,
            evidence,
            counts={"observations": len(used), "excluded": excluded, "negative_values": negative},
            rows=len(values),
        ),
    )


# ---------------------------------------------------------------------------- peak periods


class ShareEntry(BaseModel):
    """Share of the peak observations that fall in one hour or weekday."""

    label: str
    count: int
    share: float


class PeakPeriods(BaseModel):
    """Where the highest readings concentrate in the day and the week."""

    column: str
    percentile: float
    threshold: float | None = None
    n_peak_observations: int
    hours: list[ShareEntry]
    weekdays: list[ShareEntry]
    top_hours: list[str] = Field(default_factory=list)
    top_hours_share: float = 0.0
    top_weekdays: list[str] = Field(default_factory=list)
    top_weekdays_share: float = 0.0
    rows_total: int
    excluded: int
    evidence: Evidence
    log_entry: StepLog


def peak_periods(df: pd.DataFrame, column: str, percentile: float = 95.0) -> PeakPeriods:
    """Hours of day and weekdays where observations at or above the percentile concentrate.

    `top_hours_share` is the share of peaks in the three busiest hours; by chance alone it
    would be 3/24 = 12.5%, so a much larger share means the peaks are concentrated.
    """
    if not 0 < percentile < 100:
        raise AnalysisError("percentile must be between 0 and 100 (exclusive).")
    values = numeric_series(df, column)
    used, excluded = finite_part(values)
    params = {"percentile": percentile}
    if used.empty:
        raise AnalysisError(f"Column '{column}' has no usable values.")
    threshold = float(np.percentile(used.to_numpy(), percentile))
    peaks = used[used >= threshold]
    hour_counts = peaks.groupby(peaks.index.hour).size().reindex(range(24), fill_value=0)
    day_counts = peaks.groupby(peaks.index.day_of_week).size().reindex(range(7), fill_value=0)
    total = int(len(peaks))

    def entries(counts: pd.Series, label) -> list[ShareEntry]:
        return [
            ShareEntry(label=label(k), count=int(c), share=float(c) / total)
            for k, c in counts.items()
        ]

    hours = entries(hour_counts, lambda h: f"{h:02d}:00")
    days = entries(day_counts, lambda d: WEEKDAYS[d])
    top_h = sorted(hours, key=lambda e: -e.count)[:TOP_HOURS]  # ties stay in calendar order
    top_d = sorted(days, key=lambda e: -e.count)[:TOP_WEEKDAYS]
    evidence = make_evidence(used.index, [column], len(used), "peak_periods", params)
    summary = (
        f"Peak periods of '{column}' (at or above the {percentile:g}th percentile, "
        f"{threshold:.4g}): "
        f"{total} peak observation(s); busiest hours {', '.join(e.label for e in top_h)} hold "
        f"{sum(e.share for e in top_h):.0%}; {excluded} observation(s) excluded."
    )
    return PeakPeriods(
        column=column,
        percentile=percentile,
        threshold=threshold,
        n_peak_observations=total,
        hours=hours,
        weekdays=days,
        top_hours=[e.label for e in top_h],
        top_hours_share=float(sum(e.share for e in top_h)),
        top_weekdays=[e.label for e in top_d],
        top_weekdays_share=float(sum(e.share for e in top_d)),
        rows_total=len(values),
        excluded=excluded,
        evidence=evidence,
        log_entry=analysis_log(
            "peak_periods",
            summary,
            evidence,
            counts={"peak_observations": total, "excluded": excluded},
            rows=len(values),
        ),
    )
