"""Generate a small, seeded, synthetic household-energy file with realistic mess.

Nothing is downloaded. The output is deterministic: the same seed always writes the same bytes.

Usage:  python scripts/make_sample.py [--out PATH] [--seed N] [--days N]

The file has separate dd/mm/yyyy Date and Time columns, "?" missing markers, a 10-hour outage
(missing rows), an 8-hour sensor dropout, a few spikes and negative readings, duplicate rows
and one block of rows in reverse order.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "sample_data" / "household_energy_sample.csv"
HEADER = ["Date", "Time", "global_active_power", "voltage", "global_intensity", "outdoor_temp"]


def generate(seed: int = 2024, days: int = 120) -> bytes:
    """Return the sample file as UTF-8 CSV bytes."""
    rng = np.random.default_rng(seed)
    n = days * 24
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    hour = ts.hour.to_numpy().astype(float)
    weekend = (ts.day_of_week.to_numpy() >= 5).astype(float)
    day = np.arange(n) / 24

    temp = 5 + 0.04 * day + 4 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.8, n)
    evening = np.exp(-(((hour - 19) / 2.5) ** 2))
    morning = np.exp(-(((hour - 7.5) / 1.5) ** 2))
    power = (
        0.6
        + 0.9 * evening
        + 0.4 * morning
        + 0.2 * weekend
        + 0.02 * (10 - temp)
        + rng.normal(0, 0.06, n)
    )
    voltage = 240 + 3 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.2, n)
    intensity = power * 1000 / voltage * 1.05 + rng.normal(0, 0.1, n)

    rows: list[list[str | None]] = [
        [
            t.strftime("%d/%m/%Y"),
            t.strftime("%H:%M:%S"),
            f"{p:.3f}",
            f"{v:.2f}",
            f"{i:.2f}",
            f"{c:.1f}",
        ]
        for t, p, v, i, c in zip(ts, power, voltage, intensity, temp, strict=True)
    ]

    for k in rng.choice(n, size=5, replace=False):  # sensor spikes
        rows[int(k)][2] = f"{power[int(k)] * rng.uniform(5, 8):.3f}"
    for k in rng.choice(n, size=2, replace=False):  # impossible negative consumption
        rows[int(k)][2] = f"{-abs(power[int(k)]):.3f}"
    for k in rng.choice(n, size=int(n * 0.02), replace=False):  # scattered "?" readings
        rows[int(k)][2] = "?"
    for k in rng.choice(n, size=int(n * 0.01), replace=False):
        rows[int(k)][3] = "?"
    dropout = int(n * 0.7)
    for k in range(dropout, dropout + 8):  # an 8-hour dropout: every sensor reads "?"
        rows[k][2:] = ["?"] * 4

    outage = int(n * 0.35)
    del rows[outage : outage + 10]  # a 10-hour outage: the rows are simply absent
    dup_at = sorted(
        int(k) for k in rng.choice(np.arange(10, len(rows) - 10), size=6, replace=False)
    )
    for k in reversed(dup_at):  # exact duplicate rows
        rows.insert(k + 1, list(rows[k]))
    reverse = int(len(rows) * 0.84)
    rows[reverse : reverse + 30] = rows[reverse : reverse + 30][::-1]  # logged in reverse order

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(HEADER)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic sample dataset.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args(argv)
    data = generate(args.seed, args.days)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"Wrote {args.out} ({len(data):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
