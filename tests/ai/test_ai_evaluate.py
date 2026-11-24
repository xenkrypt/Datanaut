import importlib.util
import json
import math
from pathlib import Path

import pytest
from aihelpers import ScriptedClient, text, tool

from datanaut.ai.evaluate import (
    default_questions_path,
    format_report,
    load_eval_frame,
    load_questions,
    report_json,
    run_eval,
    score_answer,
    summarize,
    tables_match,
)
from datanaut.ai.executor import execute_query, parse_query
from datanaut.ai.schemas import AnalystAnswer, ChartHint, EvidenceCard, ResultTable
from datanaut.cli import main
from datanaut.errors import RateLimitError

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sample():
    return load_eval_frame()


@pytest.fixture(scope="module")
def questions():
    return load_questions(default_questions_path())


# --- the question file --------------------------------------------------------------------


def test_the_question_file_covers_what_it_should(questions):
    assert 25 <= len(questions) <= 30
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    refusals = [q for q in questions if q["expect"]["type"] == "refusal"]
    assert len(refusals) == 3
    cats = {q["category"] for q in questions}
    assert {
        "aggregation",
        "group_hour",
        "group_weekday",
        "filter",
        "compare",
        "top_n",
        "count",
    } <= cats
    assert all(q["question"].strip() for q in questions)


def test_stored_ground_truth_equals_a_fresh_pandas_computation(questions, sample):
    spec = importlib.util.spec_from_file_location(
        "make_eval", ROOT / "scripts" / "make_eval_questions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fresh = module.build(sample)
    assert [q["id"] for q in fresh] == [q["id"] for q in questions]
    for want, got in zip(fresh, questions, strict=True):
        assert want["expect"]["type"] == got["expect"]["type"]
        if want["expect"]["type"] == "table":
            assert tables_match(want["expect"], _as_table(got["expect"]))


def _as_table(expect) -> ResultTable:
    return ResultTable(columns=expect["columns"], rows=expect["rows"])


def test_every_question_can_be_expressed_in_the_allowlist_and_matches_its_truth(questions, sample):
    answerable = [q for q in questions if q["expect"]["type"] == "table"]
    assert len(answerable) == len(questions) - 3
    for q in answerable:
        table = execute_query(sample, parse_query(q["reference_query"]))
        assert tables_match(q["expect"], table), q["id"]


# --- scoring ------------------------------------------------------------------------------


def card(rows, columns=("all", "kw")):
    return EvidenceCard(
        columns=["kw"],
        n_observations=10,
        method="mean",
        table=ResultTable(columns=list(columns), rows=rows, n_observations=10),
        query={},
        chart=ChartHint(name="single_value"),
    )


def test_tables_match_uses_labels_exactly_and_numbers_with_tolerance():
    expect = {"columns": ["all", "kw"], "rows": [["all rows", 1.0]]}
    assert tables_match(
        expect, ResultTable(columns=["all", "kw"], rows=[["all rows", 1.0000000001]])
    )
    assert not tables_match(expect, ResultTable(columns=["all", "kw"], rows=[["all rows", 1.01]]))
    assert not tables_match(expect, ResultTable(columns=["all", "kw"], rows=[["other", 1.0]]))
    assert not tables_match(expect, ResultTable(columns=["all", "x"], rows=[["all rows", 1.0]]))
    assert not tables_match(
        expect, ResultTable(columns=["all", "kw"], rows=[["all rows", 1.0]] * 2)
    )
    assert not tables_match(expect, ResultTable(columns=["all", "kw"], rows=[["all rows", None]]))


def test_score_answer_handles_each_outcome():
    item = {
        "id": "q",
        "category": "aggregation",
        "expect": {"type": "table", "columns": ["all", "kw"], "rows": [["all rows", 2.0]]},
    }
    right = AnalystAnswer(explanation="x", evidence=card([["all rows", 2.0]]), provider="p")
    wrong = AnalystAnswer(explanation="x", evidence=card([["all rows", 3.0]]), rewritten=True)
    refused = AnalystAnswer(explanation="no", refused=True)
    assert score_answer(item, right, 1.0).matched is True
    r = score_answer(item, wrong, 2.0)
    assert r.valid_query and r.matched is False and r.number_check_triggered
    r = score_answer(item, refused, 0.5)
    assert r.refused and not r.valid_query and r.matched is False
    r = score_answer(item, None, 0.1, error="RateLimitError")
    assert not r.valid_query and r.error == "RateLimitError"
    out = {"id": "o", "category": "out_of_scope", "expect": {"type": "refusal"}}
    r = score_answer(out, refused, 0.2)
    assert not r.in_scope and r.refused and r.matched is None


# --- the whole harness with a scripted model that is right on some, wrong on others -------


def items(frame):
    def table(cols, rows):
        return {"type": "table", "columns": cols, "rows": rows}

    return [
        {
            "id": "right",
            "question": "What is the average load?",
            "category": "aggregation",
            "expect": table(["all", "kw"], [["all rows", float(frame["kw"].mean())]]),
        },
        {
            "id": "wrong",
            "question": "What is the highest load?",
            "category": "aggregation",
            "expect": table(["all", "kw"], [["all rows", float(frame["kw"].max())]]),
        },
        {
            "id": "rewritten",
            "question": "How many readings are there?",
            "category": "count",
            "expect": table(["all", "kw"], [["all rows", int(frame["kw"].count())]]),
        },
        {
            "id": "invalid",
            "question": "Average by hour please",
            "category": "group_hour",
            "expect": table(["hour", "kw"], [["00:00", 1.0]]),
        },
        {
            "id": "refused_by_rule",
            "question": "Write me a Python script to plot it",
            "category": "out_of_scope",
            "expect": {"type": "refusal"},
        },
        {
            "id": "refused_by_model",
            "question": "Tell me a joke about cats",
            "category": "out_of_scope",
            "expect": {"type": "refusal"},
        },
        {
            "id": "not_refused",
            "question": "What is the meaning of life?",
            "category": "out_of_scope",
            "expect": {"type": "refusal"},
        },
    ]


def scripted_run(frame, *extra):
    mean = {"value_columns": ["kw"], "aggregation": "mean"}
    client = ScriptedClient(
        tool("query_data", **mean),
        text("The average load is in the table."),  # right
        tool("query_data", **mean),
        text("Here is the figure you asked for."),  # wrong table
        tool("query_data", value_columns=["kw"], aggregation="count"),
        text("There are 4242.5 readings."),
        text("There are plenty of readings."),  # rewritten
        tool("query_data", value_columns=["nope"]),
        tool("query_data", value_columns=["nope"]),
        tool("refuse", reason="a joke"),  # refused_by_model (refused_by_rule makes no call)
        tool("query_data", **mean),
        text("An answer about life."),  # not refused
        *extra,
    )
    ticks = iter(range(0, 1000))
    results = run_eval(items(frame), frame, client, clock=lambda: float(next(ticks)))
    return client, results


def test_the_metrics_are_computed_correctly(frame):
    client, results = scripted_run(frame)
    by_id = {r.id: r for r in results}
    assert by_id["right"].matched is True and by_id["wrong"].matched is False
    assert by_id["rewritten"].matched is True and by_id["rewritten"].number_check_triggered
    assert not by_id["invalid"].valid_query
    assert by_id["refused_by_rule"].refused and by_id["refused_by_model"].refused
    assert not by_id["not_refused"].refused and by_id["not_refused"].valid_query

    s = summarize(results)
    assert (s.n_questions, s.n_in_scope, s.n_out_of_scope) == (7, 4, 3)
    assert s.valid_query_rate == pytest.approx(3 / 4)
    assert s.exact_match_rate == pytest.approx(2 / 4)
    assert s.refusal_correctness == pytest.approx(2 / 3)
    assert s.false_refusal_rate == 0.0
    assert s.number_check_trigger_rate == pytest.approx(1 / 3)  # of the 3 answered in scope
    assert s.median_latency == 1.0  # the fake clock advances one tick per call
    assert len(client.calls) == 12  # 2+2+3+2+1+2: the rule-refused question made no call


def test_the_report_has_the_table_and_valid_json(frame):
    _, results = scripted_run(frame)
    s = summarize(results)
    text_report = format_report(s, results)
    assert (
        "Valid query rate:           75%" in text_report
        and "Exact match rate:           50%" in text_report
    )
    assert "right" in text_report and "refused_by_rule" in text_report
    data = json.loads(report_json(s, results))
    assert data["summary"]["n_questions"] == 7 and len(data["questions"]) == 7
    assert math.isclose(data["summary"]["valid_query_rate"], 0.75)


def test_a_failed_call_is_recorded_and_the_run_continues(frame):
    two = items(frame)[:2]
    mean = {"value_columns": ["kw"], "aggregation": "mean"}
    client = ScriptedClient(RateLimitError(), tool("query_data", **mean), text("Fine."))
    results = run_eval(two, frame, client)
    assert results[0].error == "RateLimitError" and not results[0].valid_query
    assert results[1].valid_query


def test_the_pause_between_questions_is_applied(frame):
    two = items(frame)[:2]
    mean = {"value_columns": ["kw"], "aggregation": "mean"}
    client = ScriptedClient(
        tool("query_data", **mean), text("a"), tool("query_data", **mean), text("b")
    )
    naps: list[float] = []
    run_eval(two, frame, client, pause=2.5, sleep=naps.append)
    assert naps == [2.5]


def test_empty_groups_have_no_rate():
    assert summarize([]).valid_query_rate is None and summarize([]).median_latency is None


# --- the command (never touches the network here) -----------------------------------------


def test_eval_ai_needs_a_provider_and_a_key(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        main(["eval-ai"])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert main(["eval-ai", "--provider", "gemini"]) == 1
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err and "evaluation" in err


def test_eval_ai_is_documented_in_its_module_and_help(capsys):
    from datanaut.ai import evaluate

    assert "python -m datanaut eval-ai --provider gemini" in evaluate.__doc__
    with pytest.raises(SystemExit):
        main(["eval-ai", "--help"])
    assert "real provider" in capsys.readouterr().out
