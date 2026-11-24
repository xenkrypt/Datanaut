import hashlib
import json

import pandas as pd
import pytest

from datanaut.cli import main
from datanaut.core.anomalies import stl_anomalies
from datanaut.core.changepoints import detect_change_points
from datanaut.core.characterize import characterize
from datanaut.core.cleaner import propose_fixes
from datanaut.core.findings import FORBIDDEN_WORDS, generate_findings
from datanaut.core.forecast import backtest, forecast, recommend_model
from datanaut.core.investigate import investigate
from datanaut.core.loader import load_file
from datanaut.core.log import PipelineLog
from datanaut.core.patterns import energy_metrics, load_profile, peak_periods
from datanaut.core.profiler import profile
from datanaut.core.timeindex import build_time_index, detect_time_columns
from datanaut.core.versioning import VersionStore
from datanaut.core.weather import read_weather_csv, weather_link
from datanaut.core.whatif import peak_shift


@pytest.fixture(scope="module")
def run(tmp_path_factory, phase3):
    """The whole Phase 3 pipeline on the seeded series, once."""
    path = tmp_path_factory.mktemp("p3") / "series.csv"
    path.write_bytes(phase3.csv_bytes)
    wpath = path.with_name("weather.csv")
    wpath.write_bytes(phase3.weather_csv)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    log = PipelineLog()
    loaded = load_file(path)
    log.extend(loaded.log)
    indexed = build_time_index(loaded.df, detect_time_columns(loaded.df)[0].columns)
    log.extend(indexed.log)
    prof = profile(indexed.df)
    char = characterize(indexed.df, "kw")
    log.extend([prof.log_entry, char.log_entry])
    store = VersionStore(indexed.df, log)
    for proposal in propose_fixes(indexed.df, prof, char):
        if proposal.default_selected:
            store.apply(proposal.spec)
    df = store.current()
    weather = read_weather_csv(wpath)
    period = char.seasonal_period

    hour = load_profile(df, "kw", "hour")
    weekday = load_profile(df, "kw", "weekday")
    peaks = peak_periods(df, "kw")
    metrics = energy_metrics(df, "kw", consumption=True)
    anomalies = stl_anomalies(df, "kw", period)
    changes = detect_change_points(df, "kw", period=period)
    link = weather_link(df, "kw", weather)
    bt = backtest(df, "kw", ["seasonal_naive", "holt_winters"], horizon=24, period=period)
    fc = forecast(df, "kw", recommend_model(bt).model, 24, period, backtest_result=bt)
    shift = peak_shift(df, "kw", 0.3)
    report = generate_findings(
        peaks=peaks,
        metrics=metrics,
        weekday_profile=weekday,
        weather=link,
        anomalies=anomalies,
        change_points=changes,
        backtest=bt,
    )
    start = phase3.shift_time - pd.Timedelta(days=1)
    inv = investigate(
        df, "kw", start, start + pd.Timedelta(days=10), weather_df=weather, change_points=changes
    )
    results = [hour, weekday, peaks, metrics, anomalies, changes, link, bt, fc, shift, report, inv]
    log.extend([r.log_entry for r in results])
    return {
        "path": path,
        "wpath": wpath,
        "digest": digest,
        "log": log,
        "period": period,
        "results": results,
        "peaks": peaks,
        "anomalies": anomalies,
        "changes": changes,
        "link": link,
        "bt": bt,
        "fc": fc,
        "shift": shift,
        "metrics": metrics,
        "report": report,
        "inv": inv,
    }


def test_the_daily_pattern_is_recovered(run):
    assert run["period"] == 24
    peaks = run["peaks"]
    # the evening peak (19:00) leads; the cold early hours also peak, because load is linked to
    # temperature. Together the three busiest hours hold far more than the 12.5% chance level.
    assert peaks.top_hours[0] == "19:00" and peaks.top_hours_share > 0.45
    assert {"02:00", "03:00"} & set(peaks.top_hours)


def test_the_injected_spike_is_found(run, phase3):
    worst = run["anomalies"].points[0]
    assert pd.Timestamp(worst.timestamp) == phase3.spike_time and worst.direction == "high"
    assert worst.residual == pytest.approx(80, abs=8) and str(phase3.spike_time.date()) in {
        d.date for d in run["anomalies"].days
    }


def test_the_mean_shift_is_found_near_the_true_time_with_the_right_size(run, phase3):
    near = [
        b
        for b in run["changes"].breaks
        if abs(pd.Timestamp(b.timestamp) - phase3.shift_time) <= pd.Timedelta(hours=12)
    ]
    assert len(near) == 1 and near[0].direction == "increase"
    assert near[0].mean_change == pytest.approx(15, abs=3)


def test_the_temperature_link_has_the_known_sign(run):
    assert run["link"].pearson < -0.2 and run["link"].spearman < -0.2
    assert run["link"].heating.pearson > 0.2


def test_the_forecast_and_what_if_are_labelled_estimates(run):
    assert run["fc"].label == "estimate" and len(run["fc"].points) == 24
    assert {s.model for s in run["bt"].scores} == {"seasonal_naive", "holt_winters"}
    shift = run["shift"]
    assert shift.total_after == pytest.approx(shift.total_before) and shift.peak_change < 0
    assert shift.cost is None and shift.label == "scenario estimate"


def test_the_top_findings_mention_the_pattern_spike_shift_and_weather(run, phase3):
    findings = run["report"].findings
    assert 3 <= len(findings) <= 5
    text = " ".join(f"{f.title}. {f.statement}" for f in findings)
    assert any(h in text for h in run["peaks"].top_hours)  # the peak hours
    assert str(phase3.spike_time) in text  # the spike
    assert any(f.id.startswith("change_point:") and "rose" in f.title for f in findings)
    assert any(f.id.startswith("weather:") and "associated with" in f.statement for f in findings)
    scores = [f.score for f in findings]
    assert scores == sorted(scores, reverse=True)
    for f in findings:
        assert not any(w in text.lower() for w in FORBIDDEN_WORDS), f.id


def test_investigating_the_shifted_period_reports_the_change_and_never_claims_cause(run):
    inv = run["inv"]
    assert (
        inv.change.direction == "higher"
        and inv.change.percent > 10
        and inv.change.effect_size_d > 1
    )
    by = {f.name: f for f in inv.factors}
    assert by["change_point"].verdict == "supported"
    assert "change_point" in inv.supported
    assert by["change_point"].numbers["location"] == "inside"
    text = inv.conclusion.lower()
    assert "association" in text and "does not show what caused" in text
    assert "caused by" not in text and "because" not in text and "due to" not in text
    assert inv.log_entry.read_only and inv.evidence.method == "investigation_fixed_checks"


def test_every_result_has_evidence_and_serializes_to_json(run):
    for result in run["results"]:
        ev = result.evidence
        assert ev.columns and ev.method and ev.n_observations > 0
        assert ev.start is not None and ev.end is not None and ev.start <= ev.end
        assert result.log_entry.read_only
        json.loads(result.model_dump_json())


def test_every_step_is_in_the_pipeline_log_and_the_input_file_is_untouched(run, phase3):
    ops = [e.operation for e in run["log"]]
    for operation in (
        "load",
        "build_time_index",
        "profile",
        "characterize",
        "load_profile",
        "peak_periods",
        "energy_metrics",
        "stl_anomalies",
        "change_points",
        "weather_link",
        "forecast_backtest",
        "forecast",
        "peak_shift",
        "findings",
        "investigate",
    ):
        assert operation in ops, operation
    assert [e.step_id for e in run["log"]] == list(range(1, len(run["log"]) + 1))
    assert PipelineLog.from_json(run["log"].to_json()).entries == run["log"].entries
    assert hashlib.sha256(run["path"].read_bytes()).hexdigest() == run["digest"]
    assert run["path"].read_bytes() == phase3.csv_bytes


# ------------------------------------------------------------------ the analyze command


def test_analyze_prints_ranked_findings_from_a_local_weather_file(run, capsys):
    code = main(
        [
            "analyze",
            str(run["path"]),
            "--column",
            "kw",
            "--consumption",
            "--weather-file",
            str(run["wpath"]),
            "--lat",
            "52.5",
            "--lon",
            "13.4",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    for heading in ("Source", "Default cleaning", "Analyses", "Findings (ranked"):
        assert heading in out
    assert "seasonal period: 24" in out
    assert "1. [" in out and "Level rose" in out
    assert "associated with" in out
    assert not any(w in out.lower() for w in FORBIDDEN_WORDS)
    assert hashlib.sha256(run["path"].read_bytes()).hexdigest() == run["digest"]


def test_analyze_without_weather_or_consumption_skips_those_parts(run, capsys):
    assert main(["analyze", str(run["path"]), "--column", "kw"]) == 0
    out = capsys.readouterr().out
    assert (
        "Peaks concentrate" in out
        and "Load factor" not in out
        and "weather" not in out.lower().split("analyses")[1]
    )


def test_analyze_reports_skipped_analyses_and_bad_input_cleanly(capsys, tmp_path):
    short = tmp_path / "short.csv"
    short.write_text(
        "ts,kw\n"
        + "\n".join(f"2024-01-01 {h:02d}:00:00,{(h * 7) % 11 + 1.5}" for h in range(24))
        + "\n",
        encoding="utf-8",
    )
    assert main(["analyze", str(short), "--column", "kw"]) == 0
    assert "[skipped]" in capsys.readouterr().out
    assert main(["analyze", str(short), "--column", "nope"]) == 1
    assert "not found" in capsys.readouterr().err
    assert main(["analyze", str(short), "--column", "kw", "--lat", "52.5"]) == 1
    assert "both --lat and --lon" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["analyze", str(short)])  # --column is required
