import io
import re

import numpy as np
import pandas as pd
import pytest
from pypdf import PdfReader
from reporthelpers import FIXED_DATE

from datanaut.report.model import SECTION_TITLES, build_report_data
from datanaut.report.pdf import build_pdf, printable
from datanaut.ui import actions, analysis, reporting, rules
from datanaut.ui.state import WizardState


def pdf_for(state, version="V1", sections=None) -> bytes:
    data = build_report_data(
        reporting.inputs_for(state, version), version, sections, generated=FIXED_DATE
    )
    return build_pdf(data)


def pages(pdf: bytes) -> list[str]:
    return [p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages]


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


@pytest.fixture(scope="module")
def full_pdf(state) -> bytes:
    return pdf_for(state)


# --- a valid, complete document -----------------------------------------------------------


def test_the_pdf_opens_and_has_a_reasonable_length(full_pdf):
    assert full_pdf.startswith(b"%PDF-")
    reader = PdfReader(io.BytesIO(full_pdf))
    assert 8 <= len(reader.pages) <= 25
    assert reader.metadata.title.startswith("Datanaut report")
    assert reader.metadata.author == "Datanaut"
    assert "2026-01-15" in str(reader.metadata.creation_date)


def test_the_cover_contents_and_footers_carry_the_dataset_identity(state, full_pdf):
    fingerprint = state.store.version("V1").fingerprint
    texts = pages(full_pdf)
    cover = squash(texts[0])
    assert "Datanaut" in texts[0] and "Analysisreport" in cover
    assert "An AI-Powered Intelligent Platform for Time-Series Data Analysis" in " ".join(
        texts[0].split()
    )
    assert fingerprint in cover and "household.csv" in cover and "2026-01-1509:30UTC" in cover
    assert "Contents" in texts[1]
    for text in texts[1:]:
        assert f"Dataset fingerprint {fingerprint[:12]}" in text.replace("\n", " ")
    assert all(f"Page {i}" in texts[i - 1] for i in range(2, len(texts) + 1))


def test_section_titles_appear_in_order_in_the_contents_and_in_the_body(state, full_pdf):
    texts = pages(full_pdf)
    ids = ["summary", "quality", "pipeline", "findings", "patterns", "anomalies", "forecast",
           "whatif", "limitations", "reproducibility"]  # fmt: skip
    titles = [SECTION_TITLES[i] for i in ids]
    for scope in (texts[1], "\n".join(texts[2:])):
        flat = " ".join(scope.split())
        positions = [flat.index(t) for t in titles]
        assert positions == sorted(positions), scope[:80]
    assert len({t for t in titles if t in " ".join(texts[1].split())}) == len(titles)


def test_the_pdf_contains_a_known_finding_the_log_and_the_appendix(state, full_pdf):
    findings = analysis.peek(state, "findings", weather=state.weather_id).result.findings
    body = squash("\n".join(pages(full_pdf)))
    assert squash(findings[0].statement) in body
    assert squash("Evidence:") in body
    assert "Reproducibilityappendix" in body and "Operationchain" in body
    assert re.search(r"[0-9a-f]{64}", body)  # the recipe content hash
    assert "load_file" in body or "Loaded" in body or "load" in body
    assert "Replayingthisrecipe" in body
    assert "datanaut" in body.lower() and "pandas" in body


def test_estimates_are_labelled_and_causal_wording_is_absent(full_pdf):
    text = " ".join("\n".join(pages(full_pdf)).split()).lower()
    assert "caused by" not in text and "because of" not in text
    assert "estimate" in text and "scenario estimate" in text
    assert "associated with" in text or "association" in text


def test_charts_are_embedded_as_images(state, full_pdf):
    reader = PdfReader(io.BytesIO(full_pdf))
    images = sum(len(p.images) for p in reader.pages)
    report = build_report_data(reporting.inputs_for(state, "V1"), "V1")
    placed = sum(1 for s in report.sections for b in s.blocks if b.type == "chart")
    assert placed >= 5 and images == placed


# --- awkward text -------------------------------------------------------------------------


def csv_with(*names: str) -> bytes:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=24 * 12, freq="h")
    frame = pd.DataFrame(
        {
            n: 50 + 10 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 1, len(idx))
            for n in names
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    return frame.round(3).to_csv().encode("utf-8")


def test_unicode_names_render_without_errors_and_odd_characters_become_question_marks():
    names = ["Température (°C)", "電力、使用量", "Zoë's <b>load</b> $5"]
    s = WizardState()
    actions.load_upload(s, csv_with(*names), "énergie.csv")
    actions.confirm_mapping(s, ["timestamp"], None, names, {n: True for n in names})
    for name in names:
        s.analysis_column = name
        analysis.hour_profile(s)
        analysis.peaks(s)
        analysis.stl(s)
    s.analysis_column = "電力、使用量"
    pdf = pdf_for(s)
    text = " ".join("\n".join(pages(pdf)).split())
    assert "énergie.csv" in text and "Température (°C)" in text and "Zoë's" in text
    assert "<b>load</b>" in text and "$5" in text  # shown as typed, not interpreted
    assert "?" in text  # kanji the font lacks are shown as ?, never as boxes or errors
    assert PdfReader(io.BytesIO(pdf)).pages


def test_printable_replaces_missing_glyphs_and_drops_control_characters():
    assert printable("Température °C") == "Température °C"
    assert printable("電") == "?" and printable("a\x00b\x1bc") == "abc"
    assert printable("tab\there") == "tab here" and printable("line\nbreak") == "line\nbreak"


def test_a_300_character_column_name_is_truncated():
    long_name = "A" * 300
    s = WizardState()
    actions.load_upload(s, csv_with(long_name, "kw"), "wide.csv")
    actions.confirm_mapping(s, ["timestamp"], None, [long_name, "kw"], {})
    s.analysis_column = long_name
    analysis.hour_profile(s)
    text = squash("\n".join(pages(pdf_for(s))))
    assert "A" * 79 + "…" in text and "A" * 100 not in text


def test_a_very_long_log_paginates_without_errors(phase3):
    s = WizardState()
    actions.load_upload(s, phase3.csv_bytes, "household.csv")
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {})
    for i in range(1, 501):
        s.log.append("synthetic", f"synthetic entry {i}", counts={"n": i}, read_only=True)
    pdf = pdf_for(s, sections=["pipeline", "reproducibility"])
    texts = pages(pdf)
    joined = "\n".join(texts)
    assert len(texts) > 10
    for i in (1, 250, 500):
        assert f"synthetic entry {i}" in " ".join(joined.split())
    assert "Reproducibility appendix" in joined
    # a header row is repeated when a table continues on the next page
    assert sum("What happened" in t for t in texts) >= 5


# --- repeatability and memory -------------------------------------------------------------


def test_the_same_inputs_and_date_give_the_same_text(state):
    one, two = pdf_for(state), pdf_for(state)
    assert pages(one) == pages(two)
    other = pdf_for(state, sections=["summary"])
    assert pages(other) != pages(one)


def test_generating_reports_does_not_leave_figures_open(state):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    pdf_for(state)
    pdf_for(state)
    assert plt.get_fignums() == before == []


def test_the_state_keeps_the_pdf_only_in_memory_and_follows_the_selection(
    state, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    selected = ["summary", "reproducibility"]
    pdf = reporting.generate_pdf(state, "V1", selected)
    assert state.report_pdf == (reporting.pdf_key("V1", selected), pdf)
    assert reporting.pdf_key("V1", ["b", "a"]) == reporting.pdf_key("V1", ["a", "b"])
    assert list(tmp_path.iterdir()) == []


def test_a_report_for_an_earlier_version_has_that_versions_numbers_in_the_pdf(phase3):
    s = WizardState()
    actions.load_upload(s, phase3.csv_bytes, "household.csv")
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {})
    v1 = s.store.version("V1")
    rules.add_rule(s, rules.build_rule(s, "range", column="kw", min=0.0, max=120.0, name="cap"))
    rules.run_validation(s)
    rules.apply_rule_action(s, s.rules[0].id, "drop")
    v2 = s.store.version("V2")
    old, new = (
        squash("\n".join(pages(pdf_for(s, "V1")))),
        squash("\n".join(pages(pdf_for(s, "V2")))),
    )
    assert v1.fingerprint in old and v2.fingerprint not in old
    assert v2.fingerprint in new and v1.fingerprint in new  # V2's chain starts from V1's data
    assert f"{v1.rows:,}" in old and f"{v2.rows:,}" in new
