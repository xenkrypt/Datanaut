"""Command-line interface: `python -m datanaut profile|clean|replay|analyze`.

This module only wires the core functions together and prints their results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import datanaut
from datanaut.core.anomalies import stl_anomalies
from datanaut.core.changepoints import detect_change_points
from datanaut.core.characterize import SeriesCharacterization, characterize
from datanaut.core.cleaner import propose_fixes
from datanaut.core.export import write_safe_csv
from datanaut.core.findings import generate_findings
from datanaut.core.forecast import backtest, forecast, recommend_model
from datanaut.core.loader import LoadResult, load_file
from datanaut.core.log import PipelineLog
from datanaut.core.patterns import energy_metrics, load_profile, peak_periods
from datanaut.core.profiler import DataProfile, profile
from datanaut.core.timeindex import (
    TimeIndexResult,
    build_time_index,
    detect_time_columns,
    detect_value_columns,
)
from datanaut.core.validation import RuleSet, ValidationReport, validate
from datanaut.core.versioning import Recipe, VersionStore, replay_recipe
from datanaut.core.weather import read_weather_csv, weather_link
from datanaut.core.whatif import peak_shift
from datanaut.errors import AnalysisError, DatanautError, ParseError, VersioningError

# The project title, kept on one line so it can be found with a plain text search.
_DESCRIPTION = "An AI-Powered Intelligent Platform for Time-Series Data Analysis, Analytics and Insight Generation"  # noqa: E501


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="datanaut",
        description=_DESCRIPTION,
        epilog=(
            "Commands: profile, clean (record a recipe), replay (verify), analyze (findings), "
            "eval-ai (check the AI analyst)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"datanaut {datanaut.__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "profile",
        help="load a CSV or Excel file and print a data-quality and characterization summary",
        description="Load a file, build its time index, profile it and characterize one series.",
    )
    _add_source_options(p)
    p.add_argument("--log-json", type=Path, metavar="PATH", help="also write the pipeline log here")

    c = sub.add_parser(
        "clean",
        help="apply the default-selected fixes, then write a recipe and the cleaned CSV",
        description=(
            "Load a file, propose fixes, apply only the conservative default-selected ones "
            "(exact duplicates and short-gap interpolation) and write a replayable recipe."
        ),
    )
    _add_source_options(c)
    c.add_argument(
        "--recipe-out", type=Path, required=True, metavar="PATH", help="write the recipe JSON here"
    )
    c.add_argument(
        "--rules",
        type=Path,
        metavar="PATH",
        help="validate the cleaned data against this rule file",
    )
    c.add_argument(
        "--cleaned-out", type=Path, metavar="PATH", help="write the cleaned data as CSV here"
    )
    c.add_argument(
        "--strict",
        action="store_true",
        help="with --rules: exit with code 3 if an error-severity rule is not satisfied",
    )

    r = sub.add_parser(
        "replay",
        help="rebuild a cleaned dataset from a recipe and verify every fingerprint",
        description="Re-run a recipe on the original file; any difference in the data is reported.",
    )
    r.add_argument("file", type=Path, help="the original CSV or Excel (.xlsx) file")
    r.add_argument("recipe", type=Path, help="recipe JSON written by the clean command")
    r.add_argument(
        "--cleaned-out", type=Path, metavar="PATH", help="write the rebuilt data as CSV here"
    )

    a = sub.add_parser(
        "analyze",
        help="clean with the default fixes, then find patterns, anomalies, changes and a forecast",
        description=(
            "Load a file, apply the default-selected fixes, run the analysis modules on one "
            "column and print the ranked findings. Findings describe association, not cause."
        ),
    )
    a.add_argument("file", type=Path, help="path to a CSV or Excel (.xlsx) file")
    a.add_argument("--column", required=True, help="numeric column to analyze")
    a.add_argument("--consumption", action="store_true", help="the column is energy consumption")
    a.add_argument("--sheet", help="Excel sheet name or number (default: the first sheet)")
    a.add_argument(
        "--time-col",
        action="append",
        metavar="COL",
        help="timestamp column (give twice for date + time)",
    )
    order = a.add_mutually_exclusive_group()
    order.add_argument("--dayfirst", action="store_true", help="read 01/02/2024 as 1 February")
    order.add_argument("--monthfirst", action="store_true", help="read 01/02/2024 as 2 January")
    a.add_argument("--lat", type=float, help="site latitude (recorded; nothing is downloaded)")
    a.add_argument("--lon", type=float, help="site longitude (recorded; nothing is downloaded)")
    a.add_argument(
        "--weather-file",
        type=Path,
        metavar="CSV",
        help="local weather CSV: first column time, other columns numeric",
    )

    ev = sub.add_parser(
        "eval-ai",
        help="measure the AI analyst against ground truth (calls a real provider)",
        description=(
            "Ask the evaluation questions about the sample dataset through a real provider "
            "and score the answers against answers computed in pandas. Needs GEMINI_API_KEY or "
            "GROQ_API_KEY in the environment. It makes network calls and is not run in tests."
        ),
    )
    ev.add_argument("--provider", required=True, choices=["gemini", "groq"])
    ev.add_argument("--questions", type=Path, metavar="JSON", help="question file to use")
    ev.add_argument("--file", type=Path, metavar="PATH", help="dataset to ask about")
    ev.add_argument("--mode", choices=["beginner", "analyst"], default="analyst")
    ev.add_argument("--pause", type=float, default=2.0, help="seconds to wait between questions")
    return parser


def _add_source_options(p: argparse.ArgumentParser) -> None:
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


def _sheet_arg(value: str | None) -> str | int | None:
    return int(value) if isinstance(value, str) and value.lstrip("-").isdigit() else value


def _dayfirst_arg(args: argparse.Namespace) -> bool | None:
    return True if args.dayfirst else (False if args.monthfirst else None)


def _pick_time_columns(chosen: list[str] | None, loaded: LoadResult) -> list[str]:
    if chosen:
        return chosen
    candidates = detect_time_columns(loaded.df)
    if not candidates:
        raise ParseError("No timestamp column was found. Use --time-col to choose one.")
    return candidates[0].columns


def _print_ambiguity(indexed: TimeIndexResult) -> None:
    print("\nThe dates are ambiguous: day-first and month-first both parse but differ.")
    for sample in indexed.flags.ambiguous_preview:
        print(
            f"  {sample.raw!r}: day-first -> {sample.dayfirst}, month-first -> {sample.monthfirst}"
        )
    print("Re-run with --dayfirst or --monthfirst.")


def _run_profile(args: argparse.Namespace) -> int:
    sheet = _sheet_arg(args.sheet)
    dayfirst = _dayfirst_arg(args)

    loaded = load_file(args.file, sheet=sheet)
    log = PipelineLog()
    log.extend(loaded.log)
    out = _render_source(loaded)

    time_cols = _pick_time_columns(args.time_col, loaded)
    indexed = build_time_index(loaded.df, time_cols, dayfirst)
    log.extend(indexed.log)
    if indexed.flags.ambiguous:
        print("\n".join(out + _render_log(log)))
        _print_ambiguity(indexed)
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


def _render_versions(store: VersionStore) -> list[str]:
    lines = _section("Versions")
    for node in store.history():
        v = node.version
        what = v.op.operation if v.op else "raw data"
        head = "  <- head" if node.is_head else ""
        lines.append(
            f"{'  ' * node.depth}{v.version_id}  {what}  {v.fingerprint[:12]}  "
            f"{v.rows:,} rows x {v.columns} columns{head}"
        )
    return lines


def _render_validation(report: ValidationReport) -> list[str]:
    lines = _section(f"Validation against '{report.ruleset_name}' v{report.ruleset_version}")
    for r in report.results:
        mark = {"pass": "PASS", "fail": "FAIL", "not_evaluated": "SKIP"}[r.status]
        detail = "" if r.status == "pass" else f": {r.violations} violation(s). {r.message}"
        lines.append(f"[{mark}] ({r.severity}) {r.name}{detail}")
    lines.append(
        "all error-severity rules passed" if report.ok else "some error-severity rules did not pass"
    )
    return lines


def _run_clean(args: argparse.Namespace) -> int:
    if args.strict and not args.rules:
        raise ParseError("--strict needs a rule file: add --rules PATH.")
    ruleset = RuleSet.load_json(args.rules) if args.rules else None  # fail early on a bad file
    loaded = load_file(args.file, sheet=_sheet_arg(args.sheet))
    log = PipelineLog()
    log.extend(loaded.log)
    indexed = build_time_index(
        loaded.df, _pick_time_columns(args.time_col, loaded), _dayfirst_arg(args)
    )
    log.extend(indexed.log)
    if indexed.flags.ambiguous:
        _print_ambiguity(indexed)
        return 2
    out = _render_source(loaded) + _render_index(indexed)

    prof = profile(indexed.df)
    log.extend([prof.log_entry])
    value_col = args.value_col
    if value_col is None:
        values = detect_value_columns(indexed.df)
        value_col = values[0] if values else None
    char = characterize(indexed.df, value_col) if value_col else None
    if char is not None:
        log.extend([char.log_entry])

    store = VersionStore(indexed.df, log)
    proposals = propose_fixes(indexed.df, prof, char)
    out += _section("Proposed fixes")
    failed = 0
    for proposal in proposals:
        if not proposal.default_selected:
            out.append(f"[skipped] {proposal.description} (not selected by default)")
            continue
        try:
            version = store.apply(proposal.spec)
        except DatanautError as exc:
            failed += 1
            out.append(f"[failed]  {proposal.description} ({exc})")
            continue
        out.append(f"[applied] {proposal.description} -> {version.version_id}")
    if not proposals:
        out.append("No fixes proposed: nothing to clean.")
    out += _render_versions(store)

    rules_ok = True
    if ruleset is not None:
        report = validate(store.current(), ruleset)
        rules_ok = report.ok
        log.extend([report.log_entry])
        out += _render_validation(report)

    source = {
        "filename": loaded.source_info.filename,
        "sheet": loaded.source_info.sheet,
        "time_columns": indexed.choices.time_columns,
        "dayfirst": indexed.choices.dayfirst,
    }
    recipe = store.to_recipe(source=source)
    args.recipe_out.write_text(recipe.to_json(), encoding="utf-8")
    out += _section("Output")
    out.append(f"recipe: {args.recipe_out} ({len(recipe.steps)} step(s))")
    if args.cleaned_out:
        export = write_safe_csv(store.current(), args.cleaned_out)
        log.extend([export.log_entry])
        out.append(f"cleaned data: {args.cleaned_out} ({export.rows:,} rows)")
        if export.cells_neutralized:
            out.append(
                f"note: {export.cells_neutralized} cell(s) starting with =, +, - or @ were "
                "prefixed with a quote so spreadsheets read them as text"
            )
    print("\n".join(out))
    if failed:
        return 1
    return 3 if args.strict and not rules_ok else 0


def _run_replay(args: argparse.Namespace) -> int:
    try:
        text = args.recipe.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise VersioningError(f"The recipe file '{args.recipe.name}' could not be opened.") from exc
    recipe = Recipe.from_json(text)
    source = recipe.source
    loaded = load_file(args.file, sheet=source.get("sheet"))
    time_cols = source.get("time_columns") or _pick_time_columns(None, loaded)
    indexed = build_time_index(loaded.df, time_cols, source.get("dayfirst"))
    if indexed.flags.ambiguous:
        _print_ambiguity(indexed)
        return 2
    store = replay_recipe(indexed.df, recipe)
    out = _render_source(loaded) + _render_index(indexed) + _section("Replay")
    out.append(f"raw data matches the recipe ({recipe.raw_fingerprint[:12]})")
    for step in recipe.steps:
        out.append(
            f"[ok] {step.version_id} {step.op.operation}: "
            f"fingerprint {step.fingerprint[:12]} verified"
        )
    out.append(f"all {len(recipe.steps)} step(s) verified; result {recipe.head_fingerprint[:12]}")
    if args.cleaned_out:
        export = write_safe_csv(store.current(), args.cleaned_out)
        out.append(f"cleaned data: {args.cleaned_out} ({export.rows:,} rows)")
    print("\n".join(out))
    return 0


def _attempt(label: str, notes: list[str], call):  # type: ignore[no-untyped-def]
    """Run one analysis; if it cannot run on this data, say why instead of stopping."""
    try:
        return call()
    except AnalysisError as exc:
        notes.append(f"[skipped] {label}: {exc}")
        return None


def _run_analyze(args: argparse.Namespace) -> int:
    if (args.lat is None) != (args.lon is None):
        raise ParseError("Give both --lat and --lon, or neither.")
    loaded = load_file(args.file, sheet=_sheet_arg(args.sheet))
    log = PipelineLog()
    log.extend(loaded.log)
    indexed = build_time_index(
        loaded.df, _pick_time_columns(args.time_col, loaded), _dayfirst_arg(args)
    )
    log.extend(indexed.log)
    if indexed.flags.ambiguous:
        _print_ambiguity(indexed)
        return 2
    prof = profile(indexed.df)
    char = characterize(indexed.df, args.column)
    log.extend([prof.log_entry, char.log_entry])
    out = _render_source(loaded) + _render_index(indexed)

    store = VersionStore(indexed.df, log)
    out += _section("Default cleaning")
    for proposal in propose_fixes(indexed.df, prof, char):
        if proposal.default_selected:
            try:
                out.append(
                    f"[applied] {proposal.description} -> {store.apply(proposal.spec).version_id}"
                )
            except DatanautError as exc:
                out.append(f"[failed]  {proposal.description} ({exc})")
    df, period = store.current(), char.seasonal_period
    out.append(f"analyzing {len(df):,} rows; seasonal period: {period or 'none detected'}")

    notes: list[str] = []
    column = args.column
    by_hour = _attempt("load profile", notes, lambda: load_profile(df, column, "hour"))
    by_day = _attempt("weekday profile", notes, lambda: load_profile(df, column, "weekday"))
    peaks = _attempt("peak periods", notes, lambda: peak_periods(df, column))
    metrics = _attempt(
        "energy metrics", notes, lambda: energy_metrics(df, column, consumption=args.consumption)
    )
    anomalies = (
        _attempt("anomalies", notes, lambda: stl_anomalies(df, column, period)) if period else None
    )
    if not period:
        notes.append("[skipped] anomalies: no seasonal period was detected to model.")
    changes = _attempt(
        "change points", notes, lambda: detect_change_points(df, column, period=period)
    )
    horizon = period or 24
    bt = _attempt(
        "forecast benchmark",
        notes,
        lambda: backtest(
            df, column, ["seasonal_naive", "holt_winters"], horizon=horizon, period=period
        ),
    )
    fc = None
    if bt is not None and (rec := recommend_model(bt)).model:
        fc = _attempt(
            "forecast",
            notes,
            lambda: forecast(df, column, rec.model, horizon, period, backtest_result=bt),
        )
    shift = (
        _attempt("what-if", notes, lambda: peak_shift(df, column, 0.2))
        if args.consumption
        else None
    )
    link = None
    if args.weather_file:
        weather_df = read_weather_csv(args.weather_file)
        link = _attempt("weather link", notes, lambda: weather_link(df, column, weather_df))
    elif args.lat is not None:
        notes.append("[note] location recorded; no weather is downloaded here. Add --weather-file.")

    results = [by_hour, by_day, peaks, metrics, anomalies, changes, bt, fc, shift, link]
    out += _section("Analyses")
    out += [f"- {r.log_entry.summary}" for r in results if r is not None]
    out += notes
    report = generate_findings(
        peaks=peaks,
        metrics=metrics,
        weekday_profile=by_day,
        weather=link,
        anomalies=anomalies,
        change_points=changes,
        backtest=bt,
    )
    log.extend([r.log_entry for r in results if r is not None] + [report.log_entry])
    out += _section("Findings (ranked; associations in the data, not causes)")
    for i, f in enumerate(report.findings, 1):
        out += [f"{i}. [{f.score:.2f}] {f.title}", f"   {f.statement}"]
    if not report.findings:
        out.append("No findings: the data shows no pattern strong enough to report.")
    print("\n".join(out))
    return 0


def _run_eval_ai(args: argparse.Namespace) -> int:
    import os

    from datanaut.ai import config
    from datanaut.ai.evaluate import (
        default_questions_path,
        format_report,
        load_eval_frame,
        load_questions,
        report_json,
        run_eval,
        summarize,
    )
    from datanaut.ai.llm import GeminiClient, GroqClient, urllib_transport
    from datanaut.errors import AIConfigError

    name = config.KEY_NAMES[args.provider]
    key = os.environ.get(name, "")
    if not key:
        raise AIConfigError(f"Set {name} in your environment to run the evaluation.")
    builder = GeminiClient if args.provider == "gemini" else GroqClient
    client = builder(key, urllib_transport)
    questions = load_questions(args.questions or default_questions_path())
    df = load_eval_frame(args.file)
    results = run_eval(questions, df, client, mode=args.mode, pause=args.pause)
    summary = summarize(results)
    print(format_report(summary, results))
    print()
    print(report_json(summary, results))
    return 0


def _write_log(args: argparse.Namespace, log: PipelineLog) -> None:
    if args.log_json:
        args.log_json.write_text(log.to_json(), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    handlers = {
        "profile": _run_profile,
        "clean": _run_clean,
        "replay": _run_replay,
        "analyze": _run_analyze,
        "eval-ai": _run_eval_ai,
    }
    try:
        return handlers[args.command](args)
    except DatanautError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
