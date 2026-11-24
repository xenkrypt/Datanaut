"""Command-line interface: `python -m datanaut profile <file>`.

This module only wires the core functions together and prints their results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import datanaut
from datanaut.core.characterize import SeriesCharacterization, characterize
from datanaut.core.loader import LoadResult, load_file
from datanaut.core.log import PipelineLog
from datanaut.core.profiler import DataProfile, profile
from datanaut.core.timeindex import (
    TimeIndexResult,
    build_time_index,
    detect_time_columns,
    detect_value_columns,
)
from datanaut.errors import DatanautError

# The project title, kept on one line so it can be found with a plain text search.
_DESCRIPTION = "An AI-Powered Intelligent Platform for Time-Series Data Analysis, Analytics and Insight Generation"  # noqa: E501


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="datanaut",
        description=_DESCRIPTION,
        epilog="Phase 1: load, time index, profile and series characterization.",
    )
    parser.add_argument("--version", action="version", version=f"datanaut {datanaut.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "profile",
        help="load a CSV or Excel file and print a data-quality and characterization summary",
        description="Load a file, build its time index, profile it and characterize one series.",
    )
    p.add_argument("file", type=Path, help="path to a CSV or Excel (.xlsx) file")
    p.add_argument("--sheet", help="Excel sheet name or number (default: the first sheet)")
    p.add_argument(
        "--time-col",
        action="append",
        metavar="COL",
        help="timestamp column; give twice for separate date and time columns (default: detected)",
    )
    p.add_argument("--value-col", help="numeric column to characterize (default: first detected)")
    order = p.add_mutually_exclusive_group()
    order.add_argument("--dayfirst", action="store_true", help="read 01/02/2024 as 1 February")
    order.add_argument("--monthfirst", action="store_true", help="read 01/02/2024 as 2 January")
    p.add_argument("--log-json", type=Path, metavar="PATH", help="also write the pipeline log here")
    return parser


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}g}"


def _section(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


def _render_source(loaded: LoadResult) -> list[str]:
    info = loaded.source_info
    lines = _section("Source")
    lines.append(f"file: {info.filename}   format: {info.format}   size: {info.size_bytes:,} bytes")
    if info.format == "csv":
        lines.append(f"encoding: {info.encoding}   delimiter: {info.delimiter!r}")
    else:
        lines.append(f"sheet: {info.sheet}   sheets: {', '.join(info.sheet_names)}")
    lines.append(f"loaded: {len(loaded.df):,} rows x {loaded.df.shape[1]} columns")
    return lines


def _render_index(indexed: TimeIndexResult) -> list[str]:
    lines = _section("Time index")
    c, f = indexed.choices, indexed.flags
    lines.append(f"time column(s): {' + '.join(c.time_columns)}   reading: {c.dayfirst_source}")
    lines.append(
        f"timezone: {c.timezone or 'naive'}   rows moved by sorting: {f.rows_moved_by_sort}"
    )
    if f.missing_timestamp_rows or f.unparseable_rows:
        lines.append(
            f"set aside: {f.missing_timestamp_rows} missing, {f.unparseable_rows} unparseable "
            f"(examples: {', '.join(f.unparseable_examples) or '-'})"
        )
    if len(indexed.df):
        lines.append(f"range: {indexed.df.index.min()} -> {indexed.df.index.max()}")
    return lines


def _render_profile(p: DataProfile) -> list[str]:
    lines = _section("Data quality")
    lines.append(
        f"{p.rows:,} rows x {p.n_columns} columns, {p.memory_bytes / 1e6:.2f} MB in memory"
    )
    if p.sampling:
        lines.append(
            f"sampling: every {p.sampling.median_interval_label}, "
            f"{p.sampling.regular_share:.1%} regular, {p.sampling.irregular_count} irregular"
        )
    if p.gaps:
        lines.append(
            f"gaps: {p.gaps.missing_timestamps} missing timestamp(s) in {p.gaps.gap_count} gap(s)"
        )
        for g in p.gaps.largest:
            lines.append(f"  {g.start} -> {g.end}  ({g.missing_timestamps} missing)")
    d = p.duplicates
    lines.append(f"duplicate rows: {d.full_row_duplicates}")
    if d.duplicate_timestamps is not None:
        lines.append(
            f"duplicate timestamps: {d.duplicate_timestamps} "
            f"({d.duplicate_timestamps_identical} identical, "
            f"{d.duplicate_timestamps_conflicting} conflicting)"
        )
    lines.append("")
    header = (
        f"{'column':<24}{'type':<12}{'missing%':>9}{'unique':>8}{'min':>10}{'mean':>10}{'max':>10}"
    )
    lines += [header, "-" * len(header)]
    for c in p.columns:
        lines.append(
            f"{c.name[:23]:<24}{c.inferred_type:<12}{c.missing_pct:>9.2f}{c.unique_count:>8}"
            f"{_fmt(c.min):>10}{_fmt(c.mean):>10}{_fmt(c.max):>10}"
        )
    if p.warnings:
        lines += _section("Warnings")
        lines += [f"! {w}" for w in p.warnings]
    return lines


def _render_characterization(c: SeriesCharacterization) -> list[str]:
    lines = _section(f"Characterization of '{c.column}'")
    lines.append(f"frequency: {c.frequency_label}   observations: {c.n_observations:,}")
    if c.seasonal_candidates:
        cands = ", ".join(
            f"{s.period} ({s.label}) r={s.strength:.2f}{'*' if s.significant else ''}"
            for s in c.seasonal_candidates
        )
        lines.append(f"seasonal candidates: {cands}   (* = significant)")
    lines.append(f"seasonal period: {c.seasonal_period if c.seasonal_period else 'none detected'}")
    t = c.trend
    lines.append(
        f"trend: {t.direction}"
        + (
            f" ({t.total_change:+.3g} total, p={t.p_value:.3g})"
            if t.total_change is not None
            else ""
        )
    )
    s = c.stationarity
    lines.append(
        f"stationarity: {s.label}" + (f" (ADF p={s.p_value:.3g})" if s.p_value is not None else "")
    )
    lines.append(f"recommendation: {c.recommendation}")
    lines += [f"note: {n}" for n in c.notes]
    return lines


def _render_log(log: PipelineLog) -> list[str]:
    lines = _section("Pipeline log")
    for e in log:
        mark = "read-only" if e.read_only else "changes data"
        lines.append(f"{e.step_id:>2}. [{e.operation}] ({mark}) {e.summary}")
    return lines


def _run_profile(args: argparse.Namespace) -> int:
    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.lstrip("-").isdigit():
        sheet = int(sheet)
    dayfirst = True if args.dayfirst else (False if args.monthfirst else None)

    loaded = load_file(args.file, sheet=sheet)
    log = PipelineLog()
    log.extend(loaded.log)
    out = _render_source(loaded)

    time_cols = args.time_col
    if not time_cols:
        candidates = detect_time_columns(loaded.df)
        if not candidates:
            print("No timestamp column was found. Use --time-col to choose one.", file=sys.stderr)
            return 1
        time_cols = candidates[0].columns
    indexed = build_time_index(loaded.df, time_cols, dayfirst)
    log.extend(indexed.log)
    if indexed.flags.ambiguous:
        print("\n".join(out + _render_log(log)))
        print("\nThe dates are ambiguous: day-first and month-first both parse but differ.")
        for sample in indexed.flags.ambiguous_preview:
            print(
                f"  {sample.raw!r}: day-first -> {sample.dayfirst}, "
                f"month-first -> {sample.monthfirst}"
            )
        print("Re-run with --dayfirst or --monthfirst.")
        _write_log(args, log)
        return 2
    out += _render_index(indexed)

    prof = profile(indexed.df)
    log.extend([prof.log_entry])
    out += _render_profile(prof)

    value_col = args.value_col
    if value_col is None:
        values = detect_value_columns(indexed.df)
        value_col = values[0] if values else None
    if value_col is None:
        out += _section("Characterization") + ["no numeric value column found; use --value-col"]
    else:
        char = characterize(indexed.df, value_col)
        log.extend([char.log_entry])
        out += _render_characterization(char)

    out += _render_log(log)
    print("\n".join(out))
    _write_log(args, log)
    return 0


def _write_log(args: argparse.Namespace, log: PipelineLog) -> None:
    if args.log_json:
        args.log_json.write_text(log.to_json(), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return _run_profile(args)
    except DatanautError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
