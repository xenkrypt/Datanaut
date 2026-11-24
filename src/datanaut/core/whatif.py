"""What-if: how much would moving peak load to quiet hours change the peak and load factor?

`peak_shift` takes the part of each day's load that sits above a percentile threshold and moves
a chosen fraction of it into the lowest-load hours of the same day. Total energy is preserved
exactly, no hour is raised above the threshold, and the peak can only go down. This is a
scenario estimate under stated assumptions, not a prediction. A cost figure is produced only
when the caller supplies tariff rates; no price is ever assumed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from datanaut.core.evidence import (
    Evidence,
    analysis_log,
    make_evidence,
    median_step,
    numeric_series,
)
from datanaut.core.log import StepLog
from datanaut.errors import AnalysisError


class Tariff(BaseModel):
    """Prices per unit of energy, in the caller's currency. Both are required."""

    model_config = ConfigDict(extra="forbid")

    peak_rate: float = Field(ge=0)
    offpeak_rate: float = Field(ge=0)
    currency: str = ""


class CostEstimate(BaseModel):
    """Estimated saving from moving energy, from the caller's tariff only."""

    energy_moved: float
    peak_rate: float
    offpeak_rate: float
    currency: str
    estimated_saving: float
    note: str = "An estimate from the rates you supplied; it assumes the shift is possible."


class PeakShift(BaseModel):
    """The effect of moving some peak load to quiet hours of the same day."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    column: str
    shift_fraction: float
    percentile: float
    window: Literal["same_day"]
    threshold: float | None = None
    new_series: pd.Series = Field(exclude=True)
    peak_before: float | None = None
    peak_after: float | None = None
    peak_change: float | None = None
    peak_change_pct: float | None = None
    load_factor_before: float | None = None
    load_factor_after: float | None = None
    load_factor_change: float | None = None
    total_before: float
    total_after: float
    energy_moved: float
    energy_not_placed: float
    days_affected: int
    excluded_missing: int
    cost: CostEstimate | None = None
    assumptions: list[str]
    label: Literal["scenario estimate"] = "scenario estimate"
    evidence: Evidence
    log_entry: StepLog


def _fill_lowest(
    values: np.ndarray, candidates: np.ndarray, amount: float, ceiling: float
) -> np.ndarray:
    """Add `amount` to the lowest candidates by raising them to a common level (water-filling).

    The level never exceeds `ceiling`; the caller guarantees the room is large enough.
    """
    low = np.sort(values[candidates])
    count = len(low)
    prefix = np.cumsum(low)
    level = None
    for k in range(1, count + 1):  # raise the k lowest values to a common level
        candidate = (prefix[k - 1] + amount) / k
        if k == count or candidate <= low[k]:
            level = candidate
            break
    out = values.copy()
    raised = candidates & (values < level)  # type: ignore[operator]
    out[raised] = level
    return out


def peak_shift(
    df: pd.DataFrame,
    column: str,
    shift_fraction: float,
    percentile: float = 95.0,
    window: Literal["same_day"] = "same_day",
    *,
    tariff: Tariff | dict | None = None,
) -> PeakShift:
    """Move `shift_fraction` of the load above the percentile threshold to the quietest hours.

    Each day, the excess above the threshold is reduced by `shift_fraction` and the removed
    energy is added to that day's lowest readings (never raising any above the threshold). If
    a day has too little room, only what fits is moved and the rest is reported as not placed,
    so total energy stays equal. Missing readings are untouched and counted.
    """
    if not 0 <= shift_fraction <= 1:
        raise AnalysisError("shift_fraction must be between 0 and 1.")
    if not 0 < percentile < 100:
        raise AnalysisError("percentile must be between 0 and 100 (exclusive).")
    if window != "same_day":
        raise AnalysisError("Only the 'same_day' window is supported.")
    series = numeric_series(df, column)
    series = series.replace([np.inf, -np.inf], np.nan)
    present = series.notna().to_numpy()
    if present.sum() < 2:
        raise AnalysisError(f"'{column}' has too few usable values for a what-if estimate.")
    values = series.to_numpy(dtype=float)
    threshold = float(np.percentile(values[present], percentile))
    out = values.copy()
    moved_total = not_placed = 0.0
    days_affected = 0
    if shift_fraction > 0:
        day_keys = series.index.normalize()
        for day in pd.unique(day_keys):
            in_day = np.asarray(day_keys == day) & present
            peak_mask = in_day & (values > threshold)
            if not peak_mask.any():
                continue
            excess = values[peak_mask] - threshold
            wanted = shift_fraction * excess.sum()
            quiet = in_day & ~peak_mask
            room = float((threshold - values[quiet]).clip(min=0).sum()) if quiet.any() else 0.0
            movable = min(wanted, room)
            not_placed += wanted - movable
            if movable <= 0:
                continue
            scale = movable / excess.sum()  # take the same share of every peak's excess
            out[peak_mask] = values[peak_mask] - excess * scale
            out[:] = np.where(in_day, _fill_lowest(out, quiet, movable, threshold), out)
            moved_total += movable
            days_affected += 1
    new = pd.Series(out, index=series.index, name=column)
    before, after = values[present], out[present]
    step = median_step(series.index)
    hours = step.total_seconds() / 3600 if step is not None else 1.0
    peak_b, peak_a = float(before.max()), float(after.max())
    lf_b = float(before.mean() / peak_b) if peak_b > 0 else None
    lf_a = float(after.mean() / peak_a) if peak_a > 0 else None
    tariff_model = Tariff.model_validate(tariff) if isinstance(tariff, dict) else tariff
    cost = None
    if tariff_model is not None:
        energy = moved_total * hours
        cost = CostEstimate(
            energy_moved=energy,
            peak_rate=tariff_model.peak_rate,
            offpeak_rate=tariff_model.offpeak_rate,
            currency=tariff_model.currency,
            estimated_saving=energy * (tariff_model.peak_rate - tariff_model.offpeak_rate),
        )
    assumptions = [
        "Readings are treated as average load over each interval: energy = load x interval length.",
        f"Only load above the {percentile:g}th percentile ({threshold:.4g}) is moved.",
        "Moved load goes to the lowest-load hours of the same day and never above that threshold.",
        "Total energy is unchanged; the shift is assumed to be operationally possible.",
        "Prices are never assumed: a cost estimate appears only if a tariff is supplied.",
    ]
    evidence = make_evidence(
        series.index[present],
        [column],
        int(present.sum()),
        "peak_shift",
        {"shift_fraction": shift_fraction, "percentile": percentile, "window": window},
    )
    summary = (
        f"Scenario estimate: moving {shift_fraction:.0%} of the load above the {percentile:g}th "
        f"percentile of '{column}' to quiet hours changes the peak from {peak_b:.4g} to "
        f"{peak_a:.4g}"
        + (
            f" and the load factor from {lf_b:.3f} to {lf_a:.3f}"
            if lf_b is not None and lf_a is not None
            else ""
        )
        + f"; {days_affected} day(s) affected."
    )
    return PeakShift(
        column=column,
        shift_fraction=shift_fraction,
        percentile=percentile,
        window=window,
        threshold=threshold,
        new_series=new,
        peak_before=peak_b,
        peak_after=peak_a,
        peak_change=peak_a - peak_b,
        peak_change_pct=100 * (peak_a - peak_b) / peak_b if peak_b else None,
        load_factor_before=lf_b,
        load_factor_after=lf_a,
        load_factor_change=(lf_a - lf_b) if lf_a is not None and lf_b is not None else None,
        total_before=float(before.sum()),
        total_after=float(after.sum()),
        energy_moved=moved_total,
        energy_not_placed=not_placed,
        days_affected=days_affected,
        excluded_missing=int((~present).sum()),
        cost=cost,
        assumptions=assumptions,
        evidence=evidence,
        log_entry=analysis_log(
            "peak_shift",
            summary,
            evidence,
            counts={
                "energy_moved": moved_total,
                "energy_not_placed": not_placed,
                "days_affected": days_affected,
                "excluded_missing": int((~present).sum()),
            },
            rows=len(series),
        ),
    )
