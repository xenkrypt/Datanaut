"""Generate the demo dataset: about 12 months of hourly household energy with planted events.

Seeded and deterministic; nothing is downloaded. Every Datanaut feature has something to find:
daily and weekly seasonality, a rising trend, a weekday/weekend difference, a temperature column
linked to consumption (including a heatwave), one clear mean shift on a known date, a handful of
spikes, multi-hour gaps, duplicate rows, "?" markers and a few rows whose unit is mislabelled
(watts instead of kilowatts). The planted events are printed so a demo script can point at them.

Usage:  python scripts/make_demo_dataset.py [--out-dir PATH] [--seed N]

Writes demo_energy_12_months.csv and demo_energy_12_months.xlsx (same readings) to sample_data/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

START = "2024-01-01 00:00"
HOURS = 365 * 24
SEED = 7

MEAN_SHIFT_AT = "2024-09-09 00:00"
MEAN_SHIFT_KW = 0.50
HEATWAVE = ("2024-07-15", "2024-07-28")
HEATWAVE_EXTRA_C = 8.0
SPIKES = [
    ("2024-02-14 08:00", 6.0),
    ("2024-04-27 22:00", 7.5),
    ("2024-06-30 03:00", 5.5),
    ("2024-08-19 13:00", 6.5),
    ("2024-10-23 18:00", 8.0),
    ("2024-11-18 11:00", 6.0),
]
GAPS = [("2024-03-12 02:00", 7), ("2024-06-05 14:00", 5), ("2024-10-21 00:00", 9)]
WATT_ROWS = ["2024-01-20 12:00", "2024-05-08 16:00", "2024-07-04 09:00", "2024-12-02 20:00"]
N_MARKERS = 25
N_DUPLICATES = 12


@dataclass(frozen=True)
class Demo:
    raw: pd.DataFrame  # the messy table as it is written to the file
    events: list[dict[str, Any]]  # what was planted


def _clean_series(rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range(START, periods=HOURS, freq="h")
    t = np.arange(HOURS)
    hour = idx.hour.to_numpy()
    weekend = np.asarray(idx.day_of_week >= 5)
    doy = np.asarray(idx.day_of_year)

    season = 11 + 10 * np.sin(2 * np.pi * (doy - 110) / 365)
    wander = np.cumsum(rng.normal(0, 0.15, HOURS))
    wander -= pd.Series(wander).rolling(24 * 14, min_periods=1, center=True).mean().to_numpy()
    temp = season + 3.5 * np.sin(2 * np.pi * (hour - 9) / 24) + wander
    hot = np.asarray(
        (idx >= pd.Timestamp(HEATWAVE[0]))
        & (idx < pd.Timestamp(HEATWAVE[1]) + pd.Timedelta(days=1))
    )
    ramp = np.clip(np.minimum(np.cumsum(hot), np.cumsum(hot[::-1])[::-1]) / 24.0, 0, 1)
    temp = temp + HEATWAVE_EXTRA_C * ramp

    morning = np.where(weekend, 0.25, 0.45) * np.exp(-(((hour - 7.5) / 1.5) ** 2))
    evening = 1.5 * np.exp(-(((hour - 19.5) / 2.0) ** 2))
    night = -0.25 * np.exp(-(((hour - 3) / 2.0) ** 2))
    daytime_weekend = np.where(weekend, 0.30, 0.0) * np.exp(-(((hour - 12) / 4.0) ** 2))
    heating = 0.05 * np.maximum(0, 12 - temp)
    cooling = 0.11 * np.maximum(0, temp - 22)
    trend = 0.40 * t / HOURS
    power = 1.05 + trend + morning + evening + night + daytime_weekend + heating + cooling
    power = power + rng.normal(0, 0.10, HOURS)
    power = np.maximum(power, 0.12)
    power = power + np.where(idx >= pd.Timestamp(MEAN_SHIFT_AT), MEAN_SHIFT_KW, 0.0)
    frame = pd.DataFrame({"power_kw": power, "temperature_c": temp}, index=idx)
    for when, size in SPIKES:
        frame.loc[pd.Timestamp(when), "power_kw"] += size
    return frame


def build(seed: int = SEED) -> Demo:
    """The messy table and the list of planted events."""
    rng = np.random.default_rng(seed)
    clean = _clean_series(rng)
    protected = {pd.Timestamp(w) for w, _ in SPIKES} | {pd.Timestamp(w) for w in WATT_ROWS}
    frame = clean.copy()
    frame["unit"] = "kW"

    for when in WATT_ROWS:  # mislabelled units: the value is in watts and the unit says so
        frame.loc[pd.Timestamp(when), "power_kw"] *= 1000
        frame.loc[pd.Timestamp(when), "unit"] = "W"

    gap_rows: list[pd.Timestamp] = []
    for when, hours in GAPS:
        start = pd.Timestamp(when)
        gap_rows += list(pd.date_range(start, periods=hours, freq="h"))
    frame = frame.drop(index=gap_rows)

    eligible = [t for t in frame.index if t not in protected and t not in set(gap_rows)]
    marker_rows = sorted(rng.choice(len(eligible), size=N_MARKERS, replace=False))
    marker_times = [eligible[i] for i in marker_rows]

    text = pd.DataFrame(
        {
            "Date": frame.index.strftime("%d/%m/%Y"),
            "Time": frame.index.strftime("%H:%M:%S"),
            "power_kw": [f"{v:.3f}" for v in frame["power_kw"]],
            "unit": frame["unit"].to_numpy(),
            "temperature_c": [f"{v:.1f}" for v in frame["temperature_c"]],
        },
        index=frame.index,
    )
    text.loc[marker_times, "power_kw"] = "?"

    dup_candidates = [t for t in text.index if t not in protected]
    dup_times = sorted(rng.choice(len(dup_candidates), size=N_DUPLICATES, replace=False))
    dup_stamps = [dup_candidates[i] for i in dup_times]
    text = pd.concat([text, text.loc[dup_stamps]]).sort_index(kind="stable").reset_index(drop=True)

    events: list[dict[str, Any]] = [
        {
            "kind": "mean_shift",
            "when": MEAN_SHIFT_AT,
            "size_kw": MEAN_SHIFT_KW,
            "note": "consumption steps up and stays up (a new appliance)",
        },
        {
            "kind": "heatwave",
            "from": HEATWAVE[0],
            "to": HEATWAVE[1],
            "extra_celsius": HEATWAVE_EXTRA_C,
            "note": "temperature rises and cooling load rises with it",
        },
    ]
    events += [{"kind": "spike", "when": w, "size_kw": s} for w, s in SPIKES]
    events += [{"kind": "gap", "from": w, "hours": h} for w, h in GAPS]
    events += [
        {"kind": "watts_not_kilowatts", "when": w, "note": "value is x1000 and unit says W"}
        for w in WATT_ROWS
    ]
    events.append({"kind": "missing_markers", "count": N_MARKERS, "marker": "?"})
    events.append(
        {"kind": "duplicate_rows", "count": N_DUPLICATES, "at": [str(t) for t in dup_stamps]}
    )
    return Demo(raw=text, events=events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "sample_data"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    demo = build(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv = args.out_dir / "demo_energy_12_months.csv"
    xlsx = args.out_dir / "demo_energy_12_months.xlsx"
    demo.raw.to_csv(csv, index=False, lineterminator="\n")
    demo.raw.to_excel(xlsx, index=False, sheet_name="readings")
    print(f"wrote {csv.name} ({len(demo.raw):,} rows) and {xlsx.name}")
    print("planted events:")
    for event in demo.events:
        print("  " + json.dumps(event))


if __name__ == "__main__":
    main()
