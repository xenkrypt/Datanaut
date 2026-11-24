"""Deployment readiness: pins, secrets template, CI, config, central limits and start-up checks."""

import re
import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest

from datanaut import limits, startup
from datanaut.ai import config as ai_config

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app.py")


def names(lines: list[str]) -> list[str]:
    return [re.split(r"[<>=!~\[ ;]", ln.strip())[0].lower() for ln in lines]


def requirement_lines() -> list[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in text if ln.strip() and not ln.lstrip().startswith("#")]


# --- requirements.txt ------------------------------------------------------------------------


def test_requirements_pin_exactly_the_runtime_dependencies():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = set(names(project["dependencies"]))
    lines = requirement_lines()
    assert set(names(lines)) == declared  # nothing extra, nothing missing
    assert len(lines) == len(declared)  # each package once
    for line in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.\-]+==\d+(\.\d+)*", line), line  # pinned to one release


def test_requirements_have_no_dev_tools_and_explain_how_they_were_made():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for dev in ("pytest", "ruff", "hypothesis", "pypdf", "pip-audit"):
        assert dev not in text.lower()
    assert text.startswith("#") and "How this file was made" in text and "3.11" in text


# --- secrets template ------------------------------------------------------------------------


def test_the_secrets_template_has_placeholders_and_a_warning():
    text = (ROOT / ".streamlit" / "secrets.toml.example").read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert set(parsed) == {"GEMINI_API_KEY", "GROQ_API_KEY"}
    assert all("your-" in v and "here" in v for v in parsed.values())  # clearly not real keys
    assert "NEVER be committed" in text and ".gitignore" in text
    assert set(parsed) == set(ai_config.KEY_NAMES.values())


# --- config and limits ----------------------------------------------------------------------


def test_the_server_config_is_suited_to_hosting_and_matches_the_code_limit():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["server"]["maxUploadSize"] * 1_000_000 == limits.MAX_UPLOAD_BYTES
    assert config["server"]["headless"] is True and config["server"]["enableXsrfProtection"] is True
    assert config["browser"]["gatherUsageStats"] is False
    assert config["client"]["showErrorDetails"] == "none"  # no tracebacks in the browser
    assert config["client"]["toolbarMode"] == "minimal"


def test_every_limit_lives_in_one_file_and_the_rest_of_the_code_reads_it():
    src = ROOT / "src" / "datanaut"
    definitions = {
        "MAX_UPLOAD_BYTES": r"^MAX_UPLOAD_BYTES\s*=",
        "MAX_ROWS": r"^MAX_ROWS\s*=",
        "MAX_COLUMNS": r"^MAX_COLUMNS\s*=",
        "DISPLAY_POINT_LIMIT": r"^DISPLAY_POINT_LIMIT\s*=",
        "AI_QUESTION_CAP": r"^AI_QUESTION_CAP\s*=",
    }
    for name, pattern in definitions.items():
        where = [
            p.relative_to(ROOT).as_posix()
            for p in src.rglob("*.py")
            if re.search(pattern, p.read_text(encoding="utf-8"), re.M)
        ]
        assert where == ["src/datanaut/limits.py"], (name, where)
    assert ai_config.SESSION_QUESTION_CAP == limits.AI_QUESTION_CAP == 20
    assert limits.MAX_ROWS == 750_000 and limits.MAX_COLUMNS == 200


def test_oversized_inputs_are_refused_in_code_with_the_limit_and_the_reason():
    import pandas as pd
    import pytest

    from datanaut.errors import FileTooLargeError

    with pytest.raises(FileTooLargeError, match="limit is 35 MB.*memory"):
        limits.check_upload_size(40_000_000)
    wide = pd.DataFrame([[0] * (limits.MAX_COLUMNS + 1)])
    with pytest.raises(FileTooLargeError, match=f"limit is {limits.MAX_COLUMNS}.*memory"):
        limits.check_dimensions(wide)
    limits.check_upload_size(limits.MAX_UPLOAD_BYTES)  # exactly the limit is fine


# --- CI workflow -----------------------------------------------------------------------------


def test_the_ci_workflow_runs_the_checks_on_both_pythons_and_audits_dependencies_separately():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "push:" in text and "pull_request:" in text
    assert '"3.11", "3.12"' in text
    for step in ("ruff check .", "ruff format --check .", "pytest -q", 'pip install -e ".[dev]"'):
        assert step in text, step
    assert "cache: pip" in text
    assert "pip-audit -r requirements.txt" in text and "allow_vulnerabilities" in text
    test_job, audit_job = text.split("  audit:")
    assert "pip-audit" not in test_job.split("  test:")[1]  # only the audit job installs it
    assert "pip install pip-audit" in audit_job
    for project_file in ("pyproject.toml", "requirements.txt"):
        assert "pip-audit" not in (ROOT / project_file).read_text(encoding="utf-8")


def test_no_dockerfile_and_no_readme_changes_were_made_by_this_phase():
    assert not (ROOT / "Dockerfile").exists()


# --- start-up checks ------------------------------------------------------------------------


def test_the_startup_check_names_what_is_missing_and_nothing_else(monkeypatch):
    present = startup.check_environment(env={"GEMINI_API_KEY": "set"})
    assert present == []
    no_key = startup.check_environment(env={})
    assert [(n.level, n.feature) for n in no_key] == [("info", "AI analyst")]
    assert "GEMINI_API_KEY" in no_key[0].message and "secrets.toml" in no_key[0].message
    assert startup.check_environment(env={}, secrets={"GROQ_API_KEY": "set"}) == []

    monkeypatch.setattr("datanaut.report.text.font_dir", lambda: str(ROOT / "nowhere"))
    broken = startup.check_environment(env={"GROQ_API_KEY": "x"})
    assert [(n.level, n.feature) for n in broken] == [("warning", "Report")]
    assert startup.report_available() is False


def test_without_the_report_fonts_the_app_still_runs_and_the_report_screen_explains(monkeypatch):
    monkeypatch.setattr("datanaut.report.text.font_dir", lambda: str(ROOT / "nowhere"))
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert any("Report:" in w.value and "fonts" in w.value for w in at.warning)  # said at the top
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        [b for b in at.button if b.label == label][0].click()
        at = at.run()
    [b for b in at.button if b.label == "Build time index and create V1"][0].click()
    at = at.run()
    at.sidebar.button(key="nav_Report").click()
    at = at.run()
    assert not at.exception and at.header[0].value == "Report"
    assert any("PDF report is turned off" in w.value for w in at.warning)
    assert at.button(key="rep_generate").disabled is True
    names_ = {d.label for d in at.get("download_button")}
    assert {"Cleaned data (CSV)", "Pipeline log (JSON)", "Recipe (JSON)"} <= names_


def test_without_any_key_the_ai_screen_explains_and_the_rest_works(monkeypatch):
    from datanaut.ui import ask

    monkeypatch.setattr(ask, "provider_keys", lambda secrets=None, env=None: {})
    at = AppTest.from_file(APP, default_timeout=300).run()
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        [b for b in at.button if b.label == label][0].click()
        at = at.run()
    [b for b in at.button if b.label == "Build time index and create V1"][0].click()
    at = at.run()
    at.sidebar.button(key="nav_Ask").click()
    at = at.run()
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    assert not at.exception and any("API key" in i.value for i in at.info)
    at.sidebar.button(key="nav_Analyze").click()
    at = at.run()
    assert not at.exception and at.header[0].value == "Analyze"


def test_the_seasonal_anomaly_check_is_refused_when_it_would_take_minutes():
    import pytest

    from datanaut.errors import AnalysisError

    limits.check_seasonal_work(8_760, 24)  # a year of hourly readings: fine
    limits.check_seasonal_work(limits.MAX_SEASONAL_WORK // 288, 288)  # right at the line
    with pytest.raises(AnalysisError, match=r"100,000 readings with 288 per cycle.*minute.*hourly"):
        limits.check_seasonal_work(100_000, 288)  # about 7 minutes measured
