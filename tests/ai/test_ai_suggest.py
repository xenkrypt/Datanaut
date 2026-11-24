import numpy as np
import pandas as pd
import pytest
from aihelpers import ScriptedClient, text, tool

from datanaut.ai.context import build_context
from datanaut.ai.suggest import (
    suggest_next_steps,
    suggest_with_status,
    summarize_investigation,
    summarize_with_status,
    validate_suggestions,
)
from datanaut.core.evidence import Evidence
from datanaut.core.findings import ChartSpec, Finding
from datanaut.core.investigate import investigate
from datanaut.core.log import PipelineLog
from datanaut.core.profiler import profile
from datanaut.errors import RateLimitError


def finding(fid, category, title="A finding", statement="Something computed.", **numbers):
    return Finding(
        id=fid,
        title=title,
        statement=statement,
        category=category,
        score=0.5,
        source="x",
        numbers=numbers,
        evidence=Evidence(columns=["kw"], n_observations=10, method="m"),
        chart=ChartSpec(name="x"),
    )


@pytest.fixture
def findings():
    return [
        finding(
            "peak_hours",
            "peaks",
            "Peaks around 19:00",
            "Peaks fall around 19:00 (45% of them).",
            share=0.45,
        ),
        finding(
            "anomalies:kw",
            "anomalies",
            "Unusual readings on 2024-01-21",
            "3 readings differ from the usual pattern on 2024-01-21.",
        ),
        finding("weekday_vs_weekend", "patterns", "Weekends differ", "Weekend load is 8% higher."),
    ]


@pytest.fixture
def log():
    out = PipelineLog()
    out.append("load", "Loaded 672 rows", counts={"rows": 672})
    out.append("profile", "Profiled 672 rows", counts={"rows": 672}, read_only=True)
    return out


@pytest.fixture
def messy_profile(frame):
    messy = frame.copy()
    messy.loc[messy.index[:30], "kw"] = np.nan
    return profile(messy)


def item(**kw):
    base = {
        "title": "Do a thing",
        "rationale": "Because it helps.",
        "kind": "cleaning",
        "log_entry_id": 2,
    }
    return base | kw


# --- references must be real --------------------------------------------------------------


def test_suggestions_that_refer_to_nothing_real_are_dropped(findings, log):
    raw = [
        item(),  # real log step: kept
        item(title="Peaks", kind="peak_reduction", finding_id="peak_hours", log_entry_id=None),
        item(title="Ghost finding", finding_id="does_not_exist", log_entry_id=None),
        item(title="Ghost step", log_entry_id=99),
        item(title="No reference", log_entry_id=None),
        item(title="Half real", finding_id="peak_hours", log_entry_id=99),
        {"title": "bad shape"},
        "not even a dict",
    ]
    kept = validate_suggestions(raw, findings, log)
    assert [s.title for s in kept] == ["Do a thing", "Peaks"]


def test_the_kind_must_fit_the_finding_it_points_at(findings, log):
    raw = [
        item(
            title="Anomaly idea",
            kind="anomaly_hypothesis",
            finding_id="peak_hours",
            log_entry_id=None,
        ),
        item(
            title="Peak idea", kind="peak_reduction", finding_id="anomalies:kw", log_entry_id=None
        ),
        item(
            title="Good anomaly",
            kind="anomaly_hypothesis",
            finding_id="anomalies:kw",
            log_entry_id=None,
        ),
    ]
    assert [s.title for s in validate_suggestions(raw, findings, log)] == [
        "Hypothesis: Good anomaly"
    ]


def test_anomaly_explanations_are_always_marked_as_hypotheses(findings, log):
    marked = item(
        title="Hypothesis: a sensor fault",
        kind="anomaly_hypothesis",
        finding_id="anomalies:kw",
        log_entry_id=None,
    )
    plain = item(
        title="A meter fault",
        kind="anomaly_hypothesis",
        finding_id="anomalies:kw",
        log_entry_id=None,
        rationale="Worth checking the meter.",
    )
    kept = validate_suggestions([marked, plain], findings, log)
    assert [s.title for s in kept] == ["Hypothesis: a sensor fault", "Hypothesis: A meter fault"]


def test_invented_numbers_and_causal_wording_are_dropped(findings, log):
    raw = [
        item(title="Fix 987 rows", rationale="That would repair them."),
        item(title="Two things", rationale="Peaks near 19:00 are associated with cooking."),
        item(title="Cause", rationale="The cold caused the peak."),
        item(title="Significant", rationale="This is statistically significant."),
    ]
    assert [s.title for s in validate_suggestions(raw, findings, log)] == ["Two things"]


def test_duplicates_are_removed_and_the_list_is_capped(findings, log):
    dupes = [item(), item()] + [item(title=f"Step {i}") for i in range(10)]
    kept = validate_suggestions(dupes, findings, log)
    assert len(kept) == 6 and len({s.title for s in kept}) == 6


def test_long_text_is_cut_down(findings, log):
    kept = validate_suggestions(
        [item(title="t" * 500, rationale="r" * 2000, action="a" * 900)], findings, log
    )
    assert (
        len(kept[0].title) <= 120 and len(kept[0].rationale) <= 400 and len(kept[0].action) <= 200
    )


# --- the model call and the fallback ------------------------------------------------------


def test_good_suggestions_come_from_the_model_with_the_real_ids_in_the_prompt(frame, findings, log):
    ctx = build_context(frame, findings=findings, log=log)
    client = ScriptedClient(
        tool(
            "suggestions",
            items=[
                item(),
                item(
                    title="Try the what-if",
                    kind="peak_reduction",
                    finding_id="peak_hours",
                    log_entry_id=None,
                ),
            ],
        )
    )
    out = suggest_with_status(ctx, findings, None, log, client)
    assert [s.kind for s in out.suggestions] == ["cleaning", "peak_reduction"]
    assert not out.templated and out.provider == "scripted" and out.error is None
    call = client.calls[0]
    assert call["mode"] == "tool" and [t.name for t in call["tools"]] == ["suggestions"]
    sent = client.prompts
    assert "peak_hours" in sent and "anomalies:kw" in sent and "<<DATA:findings>>" in sent


def test_when_nothing_valid_comes_back_python_writes_the_suggestions(
    frame, findings, log, messy_profile
):
    ctx = build_context(frame, findings=findings, log=log)
    client = ScriptedClient(tool("suggestions", items=[item(title="Ghost", log_entry_id=99)]))
    out = suggest_with_status(ctx, findings, messy_profile, log, client)
    assert out.templated and out.error
    kinds = {s.kind for s in out.suggestions}
    assert (
        kinds == {"cleaning", "anomaly_hypothesis", "peak_reduction"}
        or {"anomaly_hypothesis", "peak_reduction"} <= kinds
    )
    ids = {f.id for f in findings}
    steps = {e.step_id for e in log}
    for s in out.suggestions:  # a fallback suggestion also refers to something real
        assert (s.finding_id in ids) or (s.log_entry_id in steps)
    assert all(
        s.title.startswith("Hypothesis") for s in out.suggestions if s.kind == "anomaly_hypothesis"
    )


def test_provider_failure_and_text_only_replies_also_fall_back(frame, findings, log):
    ctx = build_context(frame, findings=findings, log=log)
    failed = suggest_with_status(ctx, findings, None, log, ScriptedClient(RateLimitError()))
    assert failed.templated and "busy" in failed.error and failed.suggestions
    chatty = suggest_with_status(
        ctx, findings, None, log, ScriptedClient(text("Sure: do 42 things"))
    )
    assert chatty.templated and all("42" not in s.title for s in chatty.suggestions)


def test_suggest_next_steps_returns_the_list(frame, findings, log):
    ctx = build_context(frame, findings=findings, log=log)
    got = suggest_next_steps(
        ctx, findings, None, log, ScriptedClient(tool("suggestions", items=[item()]))
    )
    assert [s.title for s in got] == ["Do a thing"]


# --- investigation summaries --------------------------------------------------------------


@pytest.fixture(scope="module")
def result(phase3):
    # a quiet week: the weather is associated with the level, the weekday mix and peak hour are not
    start = pd.Timestamp("2024-01-24")
    return investigate(
        phase3.df, "kw", start, start + pd.Timedelta(days=7), weather_df=phase3.weather
    )


def good(result) -> str:
    c = result.change
    return (
        f"Average {result.column} was {c.mean_period:.1f} in the period against "
        f"{c.mean_baseline:.1f} before it. This is an association in the data and does not "
        "show what produced the change."
    )


def test_a_good_summary_is_used_as_written(result):
    client = ScriptedClient(text(good(result)))
    out = summarize_with_status(result, "analyst", client)
    assert out.text == good(result) and not (out.rewritten or out.templated)
    assert summarize_investigation(result, "beginner", ScriptedClient(text(good(result)))) == good(
        result
    )
    assert client.calls[0]["tools"] == [] and client.calls[0]["mode"] == "text"


def test_only_factors_with_computed_support_are_shown_to_the_model(result):
    unsupported = [f.name for f in result.factors if f.verdict == "no evidence"]
    supported = [f.name for f in result.factors if f.verdict != "no evidence"]
    assert unsupported and supported
    client = ScriptedClient(text(good(result)))
    summarize_with_status(result, "analyst", client)
    sent = client.prompts
    block_start = sent.index("factors_with_computed_support")
    listed = sent[block_start : sent.index("conclusion", block_start)]
    assert all(f'"name": "{n}"' in listed for n in supported)
    assert all(f'"name": "{n}"' not in listed for n in unsupported)


def test_an_invented_number_is_rewritten_then_replaced_by_the_templated_text(result):
    client = ScriptedClient(
        text("The load rose by 987.65 units. An association."),
        text("Honestly it went up 555.5. An association."),
    )
    out = summarize_with_status(result, "beginner", client)
    assert out.templated and out.rewritten and "987" not in out.text and "555" not in out.text
    assert result.conclusion in out.text and len(client.calls) == 2
    fixed = ScriptedClient(text("It rose by 987.65. An association."), text(good(result)))
    again = summarize_with_status(result, "beginner", fixed)
    assert again.rewritten and not again.templated and again.text == good(result)


def test_factors_with_no_support_are_not_mentioned(result):
    name = next(f.name for f in result.factors if f.verdict == "no evidence")
    mention = f"{good(result)} The {name.replace('_', ' ')} also mattered."
    client = ScriptedClient(text(mention), text(good(result)))
    out = summarize_with_status(result, "analyst", client)
    assert out.rewritten and out.text == good(result)
    assert "do not mention" in client.calls[1]["messages"][-1].content


def test_the_summary_must_say_association_and_never_cause(result):
    c = result.change
    no_word = (
        f"Average {result.column} was {c.mean_period:.1f} then and {c.mean_baseline:.1f} before."
    )
    out = summarize_with_status(
        result, "analyst", ScriptedClient(text(no_word), text(good(result)))
    )
    assert out.rewritten and "association" in out.text
    caused = good(result) + " The change was caused by the weather."
    out2 = summarize_with_status(
        result, "analyst", ScriptedClient(text(caused), text(good(result)))
    )
    assert out2.rewritten and "caused" not in out2.text


def test_a_provider_error_gives_the_templated_summary(result):
    out = summarize_with_status(result, "beginner", ScriptedClient(RateLimitError()))
    assert out.templated and out.error and result.conclusion in out.text
    assert "association" in out.text.lower()
