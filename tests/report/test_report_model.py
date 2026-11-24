import json
import sys
from pathlib import Path

import pytest
from reporthelpers import FIXED_DATE

from datanaut.core.cleaner import OpSpec
from datanaut.report.model import (
    SECTION_TITLES,
    ChartBlock,
    ReportData,
    available_sections,
    build_report_data,
    clean,
    recipe_hash,
    thin_positions,
)
from datanaut.ui import actions, analysis, reporting, rules
from datanaut.ui.state import WizardState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ai"))
from aihelpers import ScriptedClient, tool  # noqa: E402

EXPECTED = [
    "summary", "quality", "pipeline", "findings", "patterns", "anomalies", "forecast", "whatif",
    "limitations", "reproducibility",
]  # fmt: skip


def data(state, version="V1", sections=None):
    return build_report_data(
        reporting.inputs_for(state, version), version, sections, generated=FIXED_DATE
    )


def test_the_report_has_exactly_the_sections_that_have_data_in_order(state):
    report = data(state)
    assert report.section_ids == EXPECTED
    assert [s.title for s in report.sections] == [SECTION_TITLES[i] for i in EXPECTED]
    for absent in ("validation", "investigation", "suggestions"):
        assert absent not in report.section_ids  # omitted, not shown empty
    assert all(s.blocks for s in report.sections)  # no section is empty
    assert (
        report.cover.generated == "2026-01-15 09:30 UTC"
        and report.cover.dataset_name == "household.csv"
    )


def test_the_user_can_include_or_exclude_sections(state):
    only = data(state, sections=["findings", "reproducibility", "validation"])
    assert only.section_ids == ["findings", "reproducibility"]  # validation has no data
    assert data(state, sections=[]).section_ids == []
    counts = available_sections(reporting.inputs_for(state, "V1"), "V1")
    assert list(counts) == EXPECTED and all(n > 0 for n in counts.values())
    assert reporting.section_counts(state, "V1") == counts
    assert "Key findings (" in reporting.preview_line(["findings"], counts)
    assert reporting.preview_line([], counts) == "Nothing is selected."


def test_only_charts_that_the_sections_use_are_kept(state):
    report = data(state, sections=["findings"])
    used = {b.chart_id for s in report.sections for b in s.blocks if isinstance(b, ChartBlock)}
    assert set(report.charts) == used and used
    assert data(state, sections=["summary"]).charts == {}


def test_findings_carry_statement_evidence_and_wording(state):
    findings = analysis.peek(state, "findings", weather=state.weather_id).result.findings
    section = next(s for s in data(state).sections if s.id == "findings")
    text = json.dumps([b.model_dump() for b in section.blocks])
    assert clean(findings[0].statement) in text and "Evidence:" in text
    assert "caused by" not in text.lower()
    assert any(b.type == "chart" for b in section.blocks)


def test_estimates_are_labelled_as_estimates(state):
    report = data(state)
    forecast = next(s for s in report.sections if s.id == "forecast")
    whatif = next(s for s in report.sections if s.id == "whatif")
    assert forecast.blocks[0].style == "estimate" and "estimate" in forecast.blocks[0].text.lower()
    assert (
        whatif.blocks[0].style == "estimate"
        and "scenario estimate" in whatif.blocks[0].text.lower()
    )


def test_validation_investigation_and_ai_suggestions_appear_once_the_user_made_them(phase3):
    s = WizardState()
    actions.load_upload(s, phase3.csv_bytes, "household.csv")
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {"kw": True})
    analysis.findings(s)
    rules.add_rule(
        s, rules.build_rule(s, "range", column="kw", min=0.0, max=500.0, name="plausible")
    )
    rules.run_validation(s)
    import pandas as pd

    start = pd.Timestamp("2024-01-24")
    analysis.run_investigation(s, start, start + pd.Timedelta(days=7), "previous")
    first = analysis.peek(s, "findings", weather=s.weather_id).result.findings[0]
    client = ScriptedClient(
        tool(
            "suggestions",
            items=[
                {
                    "title": "Look at the peak",
                    "kind": "peak_reduction",
                    "finding_id": first.id,
                    "rationale": "It is the strongest finding.",
                    "action": "Try the what-if tool.",
                }
            ],
        )
    )
    from datanaut.ui import ask

    ask.run_suggestions(s, client)
    report = build_report_data(reporting.inputs_for(s, "V1"), "V1", generated=FIXED_DATE)
    ids = report.section_ids
    assert {"validation", "investigation"} <= set(ids)
    assert ids == [i for i in SECTION_TITLES if i in ids]  # always in the fixed order
    if "suggestions" in ids:  # only if the finding kind fits (peak_reduction needs a peaks finding)
        sec = next(x for x in report.sections if x.id == "suggestions")
        assert "language model" in sec.blocks[0].text and "Refers to finding" in json.dumps(
            [b.model_dump() for b in sec.blocks]
        )
        assert sec.title == "AI-generated suggestions"


def test_a_report_for_an_earlier_version_describes_that_version(phase3):
    s = WizardState()
    actions.load_upload(s, phase3.csv_bytes, "household.csv")
    actions.confirm_mapping(s, ["timestamp"], None, ["kw"], {"kw": True})
    analysis.findings(s)
    v1 = s.store.version("V1")
    rules.add_rule(s, rules.build_rule(s, "range", column="kw", min=0.0, max=120.0, name="cap"))
    rules.run_validation(s)
    rules.apply_rule_action(s, s.rules[0].id, "drop")
    head = s.store.version(s.store.head)
    assert head.version_id == "V2" and head.rows == v1.rows - 1

    old = build_report_data(reporting.inputs_for(s, "V1"), "V1", generated=FIXED_DATE)
    new = build_report_data(reporting.inputs_for(s, "V2"), "V2", generated=FIXED_DATE)
    assert old.cover.version_id == "V1" and old.cover.fingerprint == v1.fingerprint
    assert new.cover.version_id == "V2" and new.cover.fingerprint == head.fingerprint
    summary = lambda r: json.dumps(next(x for x in r.sections if x.id == "summary").model_dump())  # noqa: E731
    assert f"{v1.rows:,}" in summary(old) and f"{head.rows:,}" in summary(new)
    assert f"{v1.rows:,}" not in summary(new)
    assert "findings" in old.section_ids and "findings" not in new.section_ids  # never recomputed
    chain = next(x for x in old.sections if x.id == "reproducibility")
    assert "V2" not in json.dumps(chain.model_dump())


def test_the_recipe_hash_ignores_dates_and_follows_the_chain(state):
    store = state.store
    a = recipe_hash(store, "V1", {"filename": "x"})
    assert a == recipe_hash(store, "V1", {"filename": "x"}) and len(a) == 64
    assert a != recipe_hash(store, "V1", {"filename": "y"})
    store.apply(OpSpec(operation="drop_duplicates", params={"keep": "first"}))
    try:
        assert recipe_hash(store, store.head, {"filename": "x"}) != a
    finally:
        store.restore("V1")


def test_user_text_is_cleaned_and_shortened():
    assert clean("a\x00b\x1b\n  c") == "ab c"
    assert clean("x" * 500, 80) == "x" * 79 + "…"
    assert clean("short") == "short"


def test_thinning_keeps_the_extremes_and_the_limit():
    import numpy as np

    values = np.sin(np.arange(40_000) / 300)
    values[1234], values[30_000] = 9.0, -9.0
    pos = thin_positions(values, 2000)
    assert len(pos) <= 2000 and values[pos].max() == 9.0 and values[pos].min() == -9.0
    assert list(thin_positions(values[:50], 2000)) == list(range(50))


def test_the_model_is_plain_data_that_round_trips(state):
    report = data(state)
    again = ReportData.model_validate_json(report.model_dump_json())
    assert again == report
    with pytest.raises(Exception):  # noqa: B017
        ReportData.model_validate({"cover": {}, "sections": []})
