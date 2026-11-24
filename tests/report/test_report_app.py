import builtins
import io
import os
import tempfile
from pathlib import Path

import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from datanaut.ui.state import STEPS

APP = str(Path(__file__).resolve().parents[2] / "app.py")


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Empty working and temp folders, plus a record of every file opened for writing."""
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


def test_report_is_the_last_step():
    assert STEPS[-1] == "Report"


def test_the_report_screen_builds_a_pdf_from_the_sample_without_writing_a_file(sandbox):
    cwd, tmp, writes = sandbox
    at = AppTest.from_file(APP, default_timeout=300).run()
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        at = click(at, label)
    at = click(at, "Build time index and create V1")
    at.button(key="apply_sel").click()  # apply the default fixes: V1 and a later version exist
    at = at.run()
    at.sidebar.button(key="nav_Report").click()
    at = at.run()
    assert not at.exception and at.header[0].value == "Report"
    sentence = next(m.value for m in at.main.markdown if "dn-board" not in m.value)
    assert sentence.endswith(".") and "PDF" in sentence

    # sections with data are offered, each with a count; empty ones are not listed
    labels = [c.label for c in at.checkbox if c.key.startswith("rep_sec_")]
    assert any(label.startswith("Dataset summary (") for label in labels)
    assert any(label.startswith("Reproducibility appendix (") for label in labels)
    assert not any(label.startswith("Key findings") for label in labels)
    assert [d for d in at.get("download_button") if "PDF" in d.label] == []  # not yet

    at.button(key="rep_generate").click()
    at = at.run()
    assert not at.exception
    state = at.session_state["wizard"]
    version, pdf = state.report_pdf[0].split("|")[0], state.report_pdf[1]
    assert version == state.store.head
    reader = PdfReader(io.BytesIO(pdf))
    assert len(reader.pages) >= 5
    assert state.store.version(version).fingerprint[:12] in reader.pages[2].extract_text()
    assert any(d.label == "Download the PDF report" for d in at.get("download_button"))
    names = {d.label for d in at.get("download_button")}
    assert {"Cleaned data (CSV)", "Pipeline log (JSON)", "Recipe (JSON)"} <= names

    assert list(cwd.iterdir()) == [] and list(tmp.iterdir()) == []
    assert writes == [], writes


def test_the_version_picker_switches_the_report_to_an_earlier_version():
    at = AppTest.from_file(APP, default_timeout=300).run()
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        at = click(at, label)
    at = click(at, "Build time index and create V1")
    at.button(key="apply_sel").click()
    at = at.run()
    at.sidebar.button(key="nav_Report").click()
    at = at.run()
    state = at.session_state["wizard"]
    assert state.store.head != "V1" and at.selectbox(key="rep_version").value == state.store.head
    at.selectbox(key="rep_version").select("V1")
    at = at.run()
    at.button(key="rep_generate").click()
    at = at.run()
    assert not at.exception
    assert state.report_pdf[0].startswith("V1|")
    text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(state.report_pdf[1])).pages)
    assert state.store.version("V1").fingerprint[:12] in text
    rows_v1 = f"{state.store.version('V1').rows:,}"
    assert rows_v1 in " ".join(text.split())
