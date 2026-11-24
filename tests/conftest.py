"""Shared fixtures: a seeded synthetic hourly series and messy file variants of it."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import openpyxl
import pandas as pd
import pytest

SEED = 42
HOURS = 24 * 14
NUMERIC_COLS = ["power_kw", "temperature", "energy_kwh"]
_DECIMALS = {"power_kw": 3, "temperature": 2, "energy_kwh": 2}

VARIANT_NAMES = [
    "baseline",
    "semicolon",
    "decimal_comma",
    "thousands_us",
    "latin1",
    "question_marks",
    "date_time_columns",
    "duplicates",
    "gap_6h",
    "shuffled",
    "xlsx",
]


@dataclass(frozen=True)
class Variant:
    """A messy file plus the ground truth the loader should recover."""

    name: str
    filename: str
    data: bytes
    truth: pd.DataFrame  # numeric columns (NaN where the file has a missing marker), file order
    timestamps: pd.Series  # timestamps in file order
    expect: dict = field(default_factory=dict)


def make_clean(seed: int = SEED, hours: int = HOURS) -> pd.DataFrame:
    """Hourly series with daily and weekly seasonality, a trend and noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    power = (
        120
        + 0.02 * t
        + 10 * np.sin(2 * np.pi * (t % 24 - 6) / 24)
        + 4 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 1.5, hours)
    )
    temperature = 12 + 6 * np.sin(2 * np.pi * (t % 24 - 9) / 24) + rng.normal(0, 0.5, hours)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-03-01", periods=hours, freq="h"),
            "power_kw": power.round(3),
            "temperature": temperature.round(2),
            "energy_kwh": (power * 50).round(2),
        }
    )


def _fmt_plain(v: float, d: int) -> str:
    return f"{v:.{d}f}"


def _fmt_us(v: float, d: int) -> str:
    return f"{v:,.{d}f}"


def _fmt_eu(v: float, d: int) -> str:
    return f"{v:,.{d}f}".translate({ord(","): ".", ord("."): ","})


def _render(
    frame: pd.DataFrame,
    *,
    sep: str = ",",
    encoding: str = "utf-8",
    fmt: Callable[[float, int], str] = _fmt_plain,
    split: bool = False,
    markers: dict[tuple[int, str], str] | None = None,
    temp_name: str = "temperature",
    site: str | None = None,
) -> bytes:
    markers = markers or {}
    header = ["Date", "Time"] if split else ["timestamp"]
    header += ["power_kw", temp_name, "energy_kwh"] + (["site"] if site else [])
    out = io.StringIO()
    writer = csv.writer(out, delimiter=sep, lineterminator="\n")
    writer.writerow(header)
    for i, row in enumerate(frame.itertuples(index=False)):
        ts = row.timestamp
        cells = (
            [ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M:%S")]
            if split
            else [ts.strftime("%Y-%m-%d %H:%M:%S")]
        )
        for col in NUMERIC_COLS:
            cells.append(
                markers[(i, col)] if (i, col) in markers else fmt(getattr(row, col), _DECIMALS[col])
            )
        if site:
            cells.append(site)
        writer.writerow(cells)
    return out.getvalue().encode(encoding)


def _variant(name: str, filename: str, data: bytes, frame: pd.DataFrame, **expect) -> Variant:
    frame = frame.reset_index(drop=True)
    return Variant(
        name=name,
        filename=filename,
        data=data,
        truth=frame[NUMERIC_COLS].copy(),
        timestamps=frame["timestamp"].copy(),
        expect={"rows": len(frame), **expect},
    )


def build_variants(seed: int = SEED) -> dict[str, Variant]:
    """Build every messy variant deterministically."""
    clean = make_clean(seed)
    n = len(clean)
    rng = np.random.default_rng(seed + 1)
    out: dict[str, Variant] = {}

    out["baseline"] = _variant(
        "baseline", "baseline.csv", _render(clean), clean, format="csv", delimiter=","
    )
    out["semicolon"] = _variant(
        "semicolon", "semicolon.csv", _render(clean, sep=";"), clean, format="csv", delimiter=";"
    )
    out["decimal_comma"] = _variant(
        "decimal_comma",
        "decimal_comma.csv",
        _render(clean, sep=";", fmt=_fmt_eu),
        clean,
        format="csv",
        delimiter=";",
    )
    out["thousands_us"] = _variant(
        "thousands_us",
        "thousands_us.csv",
        _render(clean, sep=",", fmt=_fmt_us),
        clean,
        format="csv",
        delimiter=",",
    )
    out["latin1"] = _variant(
        "latin1",
        "latin1.csv",
        _render(
            clean, sep=";", fmt=_fmt_eu, encoding="latin-1", temp_name="température", site="Zürich"
        ),
        clean,
        format="csv",
        delimiter=";",
        temp_name="température",
        site="Zürich",
    )

    picks = rng.choice(n, size=46, replace=False)
    power_rows = sorted(int(i) for i in picks[:30])
    temp_rows = sorted(int(i) for i in picks[30:])
    temp_markers = ["NA", "N/A", "na", "null", "NULL", "-", "--", ""]
    markers: dict[tuple[int, str], str] = {(i, "power_kw"): "?" for i in power_rows}
    for k, i in enumerate(temp_rows):
        markers[(i, "temperature")] = temp_markers[k % len(temp_markers)]
    with_nan = clean.copy()
    with_nan.loc[power_rows, "power_kw"] = np.nan
    with_nan.loc[temp_rows, "temperature"] = np.nan
    out["question_marks"] = _variant(
        "question_marks",
        "question_marks.csv",
        _render(with_nan, markers=markers),
        with_nan,
        format="csv",
        delimiter=",",
        missing={"power_kw": 30, "temperature": 16},
    )

    out["date_time_columns"] = _variant(
        "date_time_columns",
        "date_time_columns.csv",
        _render(clean, split=True),
        clean,
        format="csv",
        delimiter=",",
        split=True,
    )

    dup_positions = sorted(int(i) for i in rng.choice(n, size=5, replace=False))
    dup = pd.concat([clean, clean.iloc[dup_positions]]).sort_index(kind="stable")
    out["duplicates"] = _variant(
        "duplicates", "duplicates.csv", _render(dup), dup, format="csv", delimiter=",", n_dups=5
    )

    gap = clean.drop(index=range(100, 106))
    out["gap_6h"] = _variant(
        "gap_6h", "gap_6h.csv", _render(gap), gap, format="csv", delimiter=",", gap_hours=6
    )

    shuffled = clean.iloc[rng.permutation(n)]
    out["shuffled"] = _variant(
        "shuffled", "shuffled.csv", _render(shuffled), shuffled, format="csv", delimiter=","
    )

    q_rows = sorted(int(i) for i in rng.choice(n, size=10, replace=False))
    xl_truth = clean.copy()
    xl_truth.loc[q_rows, "power_kw"] = np.nan
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append(["timestamp", "power_kw", "temperature", "energy_kwh"])
    for i, row in enumerate(clean.itertuples(index=False)):
        power = "?" if i in q_rows else float(row.power_kw)
        ws.append(
            [row.timestamp.to_pydatetime(), power, f"{row.temperature:.2f}", float(row.energy_kwh)]
        )
    notes = wb.create_sheet("notes")
    notes.append(["note"])
    notes.append(["second sheet"])
    buf = io.BytesIO()
    wb.save(buf)
    out["xlsx"] = _variant(
        "xlsx",
        "xlsx.xlsx",
        buf.getvalue(),
        xl_truth,
        format="xlsx",
        sheet_names=["data", "notes"],
        missing={"power_kw": 10},
    )
    return out


@pytest.fixture(scope="session")
def variants() -> dict[str, Variant]:
    return build_variants()


@pytest.fixture(params=VARIANT_NAMES)
def variant(request: pytest.FixtureRequest, variants: dict[str, Variant]) -> Variant:
    return variants[request.param]


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    return make_clean()


# --------------------------------------------------------------------------- Phase 3 dataset

PHASE3_DAYS = 48
PHASE3_SPIKE_AT = 20 * 24 + 7  # position of the injected spike
PHASE3_SHIFT_AT = 30 * 24  # position where the level rises
PHASE3_SPIKE_SIZE = 80.0
PHASE3_SHIFT_SIZE = 15.0
PHASE3_TEMP_SLOPE = -2.0  # load falls by 2.0 per degree: a known, clear temperature link


@dataclass(frozen=True)
class Phase3Data:
    """A seeded hourly series with a known daily pattern, spike, mean shift and weather link."""

    df: pd.DataFrame  # columns: kw
    weather: pd.DataFrame  # column: temperature_2m
    csv_bytes: bytes
    weather_csv: bytes
    spike_time: pd.Timestamp
    shift_time: pd.Timestamp


def make_phase3(seed: int = 11) -> Phase3Data:
    rng = np.random.default_rng(seed)
    n = PHASE3_DAYS * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n)
    hour = idx.hour.to_numpy()
    wander = np.cumsum(rng.normal(0, 0.2, n))
    wander -= np.linspace(wander[0], wander[-1], n)
    temp = 10 + 8 * np.sin(2 * np.pi * (t - 9) / 24) + wander
    kw = (
        60
        + 25 * np.exp(-(((hour - 19) / 2.0) ** 2))  # an evening peak around 19:00
        + PHASE3_TEMP_SLOPE * (temp - 10)
        + rng.normal(0, 1.0, n)
    )
    kw[PHASE3_SHIFT_AT:] += PHASE3_SHIFT_SIZE
    kw[PHASE3_SPIKE_AT] += PHASE3_SPIKE_SIZE
    df = pd.DataFrame({"kw": kw.round(3)}, index=idx)
    weather = pd.DataFrame({"temperature_2m": temp.round(2)}, index=idx.rename("time"))
    lines = ["timestamp,kw"] + [f"{ts:%Y-%m-%d %H:%M:%S},{v}" for ts, v in df["kw"].items()]
    wlines = ["time,temperature_2m"] + [
        f"{ts:%Y-%m-%dT%H:%M},{v}" for ts, v in weather["temperature_2m"].items()
    ]
    return Phase3Data(
        df=df,
        weather=weather,
        csv_bytes=("\n".join(lines) + "\n").encode("utf-8"),
        weather_csv=("\n".join(wlines) + "\n").encode("utf-8"),
        spike_time=idx[PHASE3_SPIKE_AT],
        shift_time=idx[PHASE3_SHIFT_AT],
    )


@pytest.fixture(scope="session")
def phase3() -> Phase3Data:
    return make_phase3()
