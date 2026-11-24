import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from datanaut.cli import main
from datanaut.core.log import PipelineLog

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "household_energy_sample.csv"


def _title() -> str:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["project"]["description"]


def test_help_uses_the_exact_project_title(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    flat = " ".join(capsys.readouterr().out.split())
    assert _title() in flat


def test_module_entry_point_runs():
    out = subprocess.run(
        [sys.executable, "-m", "datanaut", "profile", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for flag in ("--sheet", "--time-col", "--value-col", "--dayfirst", "--monthfirst"):
        assert flag in out


def test_title_appears_in_exactly_three_places():
    title = _title()
    hits = [
        p.relative_to(ROOT).as_posix()
        for p in [ROOT / "pyproject.toml", *sorted((ROOT / "src").rglob("*.py"))]
        if title in " ".join(p.read_text(encoding="utf-8").split())
    ]
    # the three places from the start, plus the report's cover page (allowed in the report phase)
    assert hits == [
        "pyproject.toml",
        "src/datanaut/__init__.py",
        "src/datanaut/cli.py",
        "src/datanaut/report/model.py",
    ]


def test_profile_sample_prints_summary_and_log(capsys, tmp_path):
    log_path = tmp_path / "log.json"
    assert main(["profile", str(SAMPLE), "--log-json", str(log_path)]) == 0
    out = capsys.readouterr().out
    for heading in ("Source", "Time index", "Data quality", "Characterization", "Pipeline log"):
        assert heading in out
    assert "global_active_power" in out
    assert "seasonal period: 24" in out
    assert "expected timestamp" in out or "missing timestamp" in out
    log = PipelineLog.from_json(log_path.read_text(encoding="utf-8"))
    ops = [e.operation for e in log]
    assert ops[:3] == ["load", "normalize_missing", "convert_numeric"]
    assert ops[-2:] == ["profile", "characterize"]
    assert json.loads(log_path.read_text(encoding="utf-8"))["metadata"]["datanaut"]


def test_overrides_time_cols_value_col_and_dayfirst(capsys):
    code = main(
        [
            "profile",
            str(SAMPLE),
            "--time-col",
            "Date",
            "--time-col",
            "Time",
            "--dayfirst",
            "--value-col",
            "voltage",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Characterization of 'voltage'" in out and "reading: user" in out


def test_ambiguous_dates_exit_code_and_message(capsys, tmp_path):
    f = tmp_path / "amb.csv"
    f.write_text("when,v\n01/02/2024,1\n03/04/2024,2\n05/06/2024,3\n", encoding="utf-8")
    assert main(["profile", str(f)]) == 2
    out = capsys.readouterr().out
    assert "ambiguous" in out and "--dayfirst or --monthfirst" in out
    assert main(["profile", str(f), "--monthfirst"]) == 0
    assert main(["profile", str(f), "--dayfirst"]) == 0


def test_dayfirst_and_monthfirst_are_exclusive(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["profile", str(SAMPLE), "--dayfirst", "--monthfirst"])
    assert exc.value.code == 2


def test_excel_sheet_option(capsys, variants, tmp_path):
    f = tmp_path / "book.xlsx"
    f.write_bytes(variants["xlsx"].data)
    assert main(["profile", str(f), "--sheet", "data"]) == 0
    assert "sheet: data" in capsys.readouterr().out
    assert main(["profile", str(f), "--sheet", "nope"]) == 1
    assert "not found" in capsys.readouterr().err


def test_friendly_errors_not_tracebacks(capsys, tmp_path):
    assert main(["profile", str(tmp_path / "missing.csv")]) == 1
    assert "could not be opened" in capsys.readouterr().err
    pdf = tmp_path / "x.csv"
    pdf.write_bytes(b"%PDF-1.4 binary")
    assert main(["profile", str(pdf)]) == 1
    assert "not supported" in capsys.readouterr().err
    nots = tmp_path / "notimes.csv"
    nots.write_text("a,b\nx,1\ny,2\n", encoding="utf-8")
    assert main(["profile", str(nots)]) == 1
    assert "--time-col" in capsys.readouterr().err


def test_no_value_column(capsys, tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("ts,label\n2024-01-01,a\n2024-01-02,b\n", encoding="utf-8")
    assert main(["profile", str(f)]) == 0
    assert "no numeric value column" in capsys.readouterr().out


def test_messy_variant_through_cli(capsys, variants, tmp_path):
    f = tmp_path / "latin.csv"
    f.write_bytes(variants["latin1"].data)
    assert main(["profile", str(f)]) == 0
    out = capsys.readouterr().out
    assert "température" in out and "seasonal period: 24" in out
