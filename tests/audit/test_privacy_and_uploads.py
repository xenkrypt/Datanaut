"""Privacy and upload audit: nothing is written, bad files are refused kindly, nothing leaks."""

import builtins
import csv
import io
import logging
import os
import tempfile
import zipfile
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from datanaut import limits
from datanaut.core.export import LEADING_BLANKS, neutralize_formulas
from datanaut.errors import DatanautError, FileTooLargeError
from datanaut.ui import actions, analysis, cleaning
from datanaut.ui.state import WizardState

ROOT = Path(__file__).resolve().parents[2]
APP = str(ROOT / "app.py")
KEY = "sk-test-SECRET-key-0123456789"
MARKER = "PRIVATE-CELL-8842"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Empty working and temp folders, a record of every file opened for writing, and a tripwire
    on the temp-file helpers."""
    cwd, tmp = tmp_path / "cwd", tmp_path / "tmp"
    cwd.mkdir()
    tmp.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    writes: list[str] = []
    real_open = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        if isinstance(file, str | os.PathLike) and any(c in mode for c in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    return cwd, tmp, writes


def click(at: AppTest, label: str) -> AppTest:
    [b for b in at.button if b.label == label][0].click()
    return at.run()


# --- uploaded bytes are never written -------------------------------------------------------


def test_the_whole_app_flow_writes_nothing_to_disk(sandbox):
    cwd, tmp, writes = sandbox
    at = AppTest.from_file(APP, default_timeout=900).run()
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        at = click(at, label)
    at = click(at, "Build time index and create V1")
    at.button(key="apply_sel").click()
    at = at.run()
    at.sidebar.button(key="nav_Analyze").click()
    at = at.run()
    at.button(key="run_anom").click()
    at = at.run()
    at.sidebar.button(key="nav_Insights").click()
    at = at.run()
    at.button(key="gen_findings").click()
    at = at.run()
    at.sidebar.button(key="nav_Ask").click()
    at = at.run()
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    at.sidebar.button(key="nav_Report").click()
    at = at.run()
    at.button(key="rep_generate").click()
    at = at.run()
    assert not at.exception
    assert at.session_state["wizard"].report_pdf is not None  # the report really was built
    assert list(cwd.iterdir()) == [] and list(tmp.iterdir()) == []
    assert writes == [], writes


def test_a_uploaded_files_content_stays_in_memory_through_every_stage(sandbox):
    cwd, tmp, writes = sandbox
    rows = "\n".join(
        f"2024-01-{d:02d} {h:02d}:00,{20 + h + d / 10},{MARKER}"
        for d in range(1, 9)
        for h in range(24)
    )
    data = f"timestamp,kw,note\n{rows}\n".encode()
    s = WizardState()
    actions.load_upload(s, data, "private.csv")
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {"kw": True})
    analysis.characterization(s)
    analysis.peaks(s)
    analysis.findings(s)
    cleaning.cleaned_csv(s)
    cleaning.recipe_json(s)
    from datanaut.ui import reporting

    reporting.generate_pdf(s, "V1", ["summary", "pipeline", "reproducibility"])
    assert list(cwd.iterdir()) == [] and list(tmp.iterdir()) == [] and writes == []
    for folder in (cwd, tmp):
        assert not any(MARKER.encode() in p.read_bytes() for p in folder.rglob("*") if p.is_file())


# --- formula-safe export --------------------------------------------------------------------

DANGEROUS = ["=1+1", "+cmd|' /C calc'!A0", "-2+3", "@SUM(A1:A9)", "  =HYPERLINK(\"http://x\")",
             "\t=1", "\r+1", " @x", "\n=1+1", "  \t -5+5"]  # fmt: skip


def test_csv_export_neutralizes_text_that_a_spreadsheet_would_run():
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=len(DANGEROUS) + 2, freq="h")
    notes = [*DANGEROUS, "plain text", "a = b"]
    s = WizardState()
    frame = pd.DataFrame({"note": notes, "=evil": notes, "kw": range(len(notes))}, index=idx)
    frame.index.name = "timestamp"
    actions.load_upload(s, frame.to_csv().encode("utf-8"), "x.csv")
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {})
    text = cleaning.cleaned_csv(s).decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    header, body = rows[0], rows[1:]
    assert "=evil" not in header and "'=evil" in header  # the column name was neutralized too
    for row in body:
        for cell in row[1:3]:  # the two text columns
            assert not cell.lstrip(LEADING_BLANKS).startswith(("=", "+", "-", "@")), repr(cell)
            assert not cell.startswith(("\t", "\r")), repr(cell)
    assert [r[1] for r in body][-2:] == ["plain text", "a = b"]  # harmless text is untouched


def test_neutralize_formulas_counts_what_it_changed():
    import pandas as pd

    df = pd.DataFrame({"t": ["=1", "  @x", "ok", None], "n": [1, 2, 3, 4]})
    safe, count = neutralize_formulas(df)
    assert count == 2 and list(safe["t"][:3]) == ["'=1", "'  @x", "ok"]
    assert list(safe["n"]) == [1, 2, 3, 4]


# --- upload validation ----------------------------------------------------------------------


def zip_bytes(**members: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in members.items():
            z.writestr(name.replace("__", "/"), content)
    return buffer.getvalue()


def bad_uploads() -> dict[str, tuple[bytes, str]]:
    return {
        "a zip that is not a spreadsheet, named .csv": (zip_bytes(readme=b"hello"), "data.csv"),
        "binary data renamed .csv": (b"\x89PNG\r\n\x1a\n" + os.urandom(4000), "photo.csv"),
        "a program renamed .csv": (b"MZ\x90\x00" + os.urandom(2000), "tool.csv"),
        "a pdf renamed .csv": (b"%PDF-1.7\n" + b"x" * 500, "report.csv"),
        "an empty file": (b"", "empty.csv"),
        "only blanks": (b"   \n\n  ", "blank.csv"),
    }


@pytest.mark.parametrize("case", list(bad_uploads()))
def test_a_bad_file_is_refused_with_a_friendly_message(case):
    data, name = bad_uploads()[case]
    s = WizardState()
    with pytest.raises(DatanautError) as err:
        actions.load_upload(s, data, name)
    assert err.value.message and "Traceback" not in err.value.message
    again = WizardState()
    assert actions.guard(again, "Upload", actions.load_upload, again, data, name) is None
    message = again.messages[-1]
    assert message.level == "error" and "Upload" in message.text
    assert "Traceback" not in message.text and "Error(" not in message.text
    assert again.loaded is None


def test_an_oversized_file_is_refused_and_says_the_limit_and_why(monkeypatch):
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 10_000)
    s = WizardState()
    with pytest.raises(FileTooLargeError) as err:
        actions.load_upload(s, b"a,b\n" + b"1,2\n" * 5000, "big.csv")
    text = err.value.message
    assert "limit is 0.01 MB" in text and "memory" in text
    assert s.loaded is None


def test_a_zip_bomb_spreadsheet_is_refused_by_its_declared_unpacked_size(monkeypatch):
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 50_000)
    bomb = zip_bytes(
        **{
            "[Content_Types].xml": b"<Types/>",
            "xl__workbook.xml": b"<workbook/>",
            "xl__worksheets__sheet1.xml": b"0" * 5_000_000,  # 5 MB unpacked, a few KB packed
        }
    )
    assert len(bomb) < 50_000  # it passes the size check: only its unpacked size gives it away
    s = WizardState()
    with pytest.raises(DatanautError) as err:
        actions.load_upload(s, bomb, "bomb.xlsx")
    assert "Traceback" not in err.value.message
    assert (
        actions.guard(WizardState(), "Upload", actions.load_upload, WizardState(), bomb, "b.xlsx")
        is None
    )


def test_a_real_spreadsheet_with_the_wrong_extension_is_read_by_its_content():
    import pandas as pd

    buffer = io.BytesIO()
    pd.DataFrame(
        {"timestamp": ["2024-01-01 00:00", "2024-01-01 01:00"], "kw": [1.0, 2.0]}
    ).to_excel(buffer, index=False)
    s = WizardState()
    assert actions.load_upload(s, buffer.getvalue(), "actually_excel.csv")
    assert s.loaded.source_info.format == "xlsx"


# --- errors and logs never leak -------------------------------------------------------------


def test_messages_and_logs_never_contain_cell_values_or_keys(caplog, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    caplog.set_level(logging.DEBUG)
    rows = "\n".join(
        f"2024-01-{d:02d} {h:02d}:00,{20 + h},{MARKER},{MARKER}-{d}"
        for d in range(1, 6)
        for h in range(24)
    )
    data = f"timestamp,kw,label,secret_note\n{rows}\n".encode()
    s = WizardState()
    actions.load_upload(s, data, "private.csv")
    # deliberately failing steps
    assert (
        actions.guard(
            s, "Map columns", actions.confirm_mapping, s, ["timestamp"], None, ["label"], {}
        )
        is None
    )  # a text column cannot be analyzed
    assert (
        actions.guard(s, "Map columns", actions.confirm_mapping, s, ["nope"], None, ["kw"], {})
        is None
    )
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {})
    assert actions.guard(s, "Analyze", analysis.fetch_weather, s, 999.0, 0.0) is None

    def offline(url):
        raise OSError(f"cannot reach {url} with {KEY} and {MARKER}")

    assert actions.guard(s, "Analyze", analysis.fetch_weather, s, 1.0, 1.0, offline) is None
    assert actions.guard(s, "Analyze", lambda: 1 / 0) is None
    assert (
        actions.guard(
            s, "Upload", actions.load_upload, s, b"\x00\x01binary" + MARKER.encode(), "x.csv"
        )
        is None
    )
    texts = "\n".join(m.text for m in s.messages)
    assert len(s.messages) >= 5 and all(m.level == "error" for m in s.messages)
    for blob in (texts, caplog.text):
        assert MARKER not in blob and KEY not in blob
        assert "Traceback" not in blob


def test_the_ai_page_error_path_does_not_expose_the_key_or_the_data(caplog, monkeypatch):
    from datanaut.errors import RateLimitError

    class Failing:
        name = "failing"

        def generate(self, *a, **k):
            raise RateLimitError()

    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    caplog.set_level(logging.DEBUG)
    s = WizardState()
    actions.load_upload(
        s,
        b"timestamp,kw\n" + "\n".join(f"2024-01-01 {h:02d}:00,{h}.5" for h in range(24)).encode(),
        "p.csv",
    )
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {})
    from datanaut.ui import ask

    s.ai_session.guard.min_interval = 0.0
    assert ask.ask_question(s, "What is the average?", Failing()) is None
    blob = "\n".join(m.text for m in s.messages) + caplog.text
    assert KEY not in blob and "12.5" not in blob
