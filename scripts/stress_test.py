# ruff: noqa: E501  (a few printed lines are long)
"""Stress test: run the whole pipeline on a dataset near the configured upload limit.

Run by hand (it is slow and not part of CI):  python scripts/stress_test.py [--fraction 0.9]

It writes a synthetic 5-minute dataset in memory (never to disk) whose size is `--fraction` of the
smaller of the row limit and the upload limit in `src/datanaut/limits.py`, then runs: load, time
index, profile, characterize, default cleaning, and the analysis modules the app uses. It prints
wall time and peak memory for each stage and PASS or FAIL against the thresholds below. On FAIL
it suggests lower limits. Peak memory is the process's own peak resident memory (Windows: peak working set; Linux and
macOS: ru_maxrss), read after each stage, so each figure is the highest value so far.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datanaut import limits  # noqa: E402
from datanaut.core.anomalies import stl_anomalies  # noqa: E402
from datanaut.core.changepoints import detect_change_points  # noqa: E402
from datanaut.core.characterize import characterize  # noqa: E402
from datanaut.core.cleaner import propose_fixes  # noqa: E402
from datanaut.core.findings import generate_findings  # noqa: E402
from datanaut.core.forecast import backtest  # noqa: E402
from datanaut.core.loader import load_file  # noqa: E402
from datanaut.core.patterns import energy_metrics, load_profile, peak_periods  # noqa: E402
from datanaut.core.profiler import profile  # noqa: E402
from datanaut.core.timeindex import build_time_index, detect_time_columns  # noqa: E402
from datanaut.core.versioning import VersionStore  # noqa: E402
from datanaut.errors import AnalysisError  # noqa: E402

# --- thresholds: change them here ----------------------------------------------------------
PEAK_MEMORY_MB = 800.0
TOTAL_SECONDS = 600.0
COLUMNS = ["power_kw", "voltage", "current", "temperature_c"]


def make_csv(rows: int, seed: int = 3) -> bytes:
    """A seeded 5-minute dataset as CSV bytes (built in memory)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=rows, freq="5min")
    minute_of_day = (idx.hour * 60 + idx.minute).to_numpy()
    daily = np.sin(2 * np.pi * (minute_of_day - 360) / 1440)
    frame = pd.DataFrame(
        {
            "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
            "power_kw": 1.2 + 0.6 * daily + rng.normal(0, 0.1, rows),
            "voltage": 230 + 2 * daily + rng.normal(0, 0.8, rows),
            "current": 5 + 2.5 * daily + rng.normal(0, 0.4, rows),
            "temperature_c": 12 + 6 * daily + rng.normal(0, 0.5, rows),
        }
    )
    return frame.round(3).to_csv(index=False).encode("utf-8")


def bytes_per_row() -> float:
    return len(make_csv(2000)) / 2000


def planned_rows(fraction: float) -> int:
    by_size = limits.MAX_UPLOAD_BYTES / bytes_per_row()
    return int(min(limits.MAX_ROWS, by_size) * fraction)


def peak_memory_mb() -> float:
    """The highest memory this process has used so far, in MB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]  # fmt: skip

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.K32GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not kernel32.K32GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError("could not read this process's memory use")
        return counters.PeakWorkingSetSize / 1e6
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


class Stages:
    """Times each stage and records the process's peak memory after it."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, float, float]] = []
        self.peak_mb = 0.0

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        try:
            return fn()
        finally:
            seconds = time.perf_counter() - started
            peak = peak_memory_mb()
            self.rows.append((name, seconds, peak))
            self.peak_mb = max(self.peak_mb, peak)
            print(f"  {name:<22}{seconds:>9.1f} s{peak:>12.0f} MB peak so far", flush=True)


def run(rows: int) -> Stages:
    stages = Stages()
    data = stages.run("generate the file", lambda: make_csv(rows))
    print(f"  (file size: {len(data) / 1e6:.1f} MB, {rows:,} rows)")
    loaded = stages.run(
        "load", lambda raw=data: load_file(raw, "stress.csv", max_bytes=limits.MAX_UPLOAD_BYTES)
    )
    del data
    cols = detect_time_columns(loaded.df)[0].columns
    indexed = stages.run("time index", lambda table=loaded.df: build_time_index(table, cols))
    del loaded
    df = indexed.df
    prof = stages.run("profile", lambda: profile(df))
    char = stages.run("characterize", lambda: characterize(df, "power_kw"))
    period = char.seasonal_period

    def cleaning() -> pd.DataFrame:
        store = VersionStore(df)
        for p in propose_fixes(store.current(), prof, char):
            if p.default_selected:
                store.apply(p.spec)
        return store.current()

    clean = stages.run("default cleaning", cleaning)
    hour = stages.run(
        "patterns",
        lambda: (
            load_profile(clean, "power_kw", "hour"),
            peak_periods(clean, "power_kw"),
            energy_metrics(clean, "power_kw", consumption=True),
        ),
    )
    anomalies = None
    if period:
        try:
            limits.check_seasonal_work(int(clean["power_kw"].notna().sum()), period)
            anomalies = stages.run(
                "stl anomalies", lambda: stl_anomalies(clean, "power_kw", period)
            )
        except AnalysisError as exc:  # the app refuses this too; show why
            print(f"  stl anomalies        refused, as the app does: {str(exc)[:110]}...")
    changes = stages.run(
        "change points", lambda: detect_change_points(clean, "power_kw", cost="l2", period=period)
    )
    bt = stages.run(
        "forecast backtest",
        lambda: backtest(clean, "power_kw", ["seasonal_naive", "holt_winters"], 24, period=period),
    )
    stages.run(
        "findings",
        lambda: generate_findings(
            peaks=hour[1], metrics=hour[2], anomalies=anomalies, change_points=changes, backtest=bt
        ),
    )
    return stages


def verdict(stages: Stages, rows: int, max_mb: float, max_seconds: float) -> bool:
    total = sum(s for _, s, _ in stages.rows)
    ok_memory, ok_time = stages.peak_mb <= max_mb, total <= max_seconds
    print()
    print(
        f"Peak memory: {stages.peak_mb:,.0f} MB (limit {max_mb:,.0f} MB): "
        f"{'PASS' if ok_memory else 'FAIL'}"
    )
    print(
        f"Total time:  {total:,.0f} s (limit {max_seconds:,.0f} s): {'PASS' if ok_time else 'FAIL'}"
    )
    passed = ok_memory and ok_time
    print("RESULT: " + ("PASS" if passed else "FAIL"))
    if not passed:
        scale = min(
            max_mb / stages.peak_mb if stages.peak_mb else 1, max_seconds / total if total else 1
        )
        safe_rows = int(rows * scale * 0.8)
        safe_bytes = int(safe_rows * bytes_per_row())
        print(
            "Suggested lower limits in src/datanaut/limits.py (80% of what passed the thresholds):"
        )
        print(f"  MAX_ROWS = {safe_rows:,}")
        print(
            f"  MAX_UPLOAD_BYTES = {safe_bytes:,}  (and maxUploadSize = {max(1, safe_bytes // 1_000_000)} "
            "in .streamlit/config.toml)"
        )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fraction", type=float, default=0.9, help="share of the limits to fill")
    parser.add_argument("--max-memory-mb", type=float, default=PEAK_MEMORY_MB)
    parser.add_argument("--max-seconds", type=float, default=TOTAL_SECONDS)
    args = parser.parse_args()
    rows = planned_rows(args.fraction)
    print(
        f"Stress test: {rows:,} rows ({args.fraction:.0%} of the limits: "
        f"{limits.MAX_ROWS:,} rows, {limits.MAX_UPLOAD_BYTES / 1e6:g} MB)"
    )
    stages = run(rows)
    return 0 if verdict(stages, rows, args.max_memory_mb, args.max_seconds) else 1


if __name__ == "__main__":
    sys.exit(main())
