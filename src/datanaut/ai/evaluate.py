"""Accuracy evaluation of the AI analyst against answers computed directly in pandas.

Run it against a real provider (keys come from the environment; it is NOT run in tests or CI)::

    python -m datanaut eval-ai --provider gemini      # needs GEMINI_API_KEY
    python -m datanaut eval-ai --provider groq        # needs GROQ_API_KEY

It asks each question in `tests/ai/eval_questions.json` about the bundled sample dataset and
prints a table and a JSON report with five measures:

* valid query rate: in-scope questions for which a query was formed and ran;
* exact match rate: in-scope questions whose executed table equals the pandas ground truth
  (labels exactly, numbers within a small tolerance, row order included);
* refusal correctness: out-of-scope questions that were refused (and, separately, the share of
  in-scope questions wrongly refused);
* number check trigger rate: answers where the model's text had to be rewritten;
* median latency per question.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

import datanaut
from datanaut.ai.analyst import AnswerMode, ask
from datanaut.ai.context import build_context
from datanaut.ai.llm import AISession, LLMClient, SessionGuard
from datanaut.ai.schemas import AnalystAnswer, ResultTable
from datanaut.core.loader import load_file
from datanaut.core.timeindex import build_time_index, detect_time_columns
from datanaut.errors import AIError

REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-9


def repo_root() -> Path:
    """The folder that holds `sample_data` and `tests` in a source checkout."""
    return Path(datanaut.__file__).resolve().parents[2]


def default_questions_path() -> Path:
    return repo_root() / "tests" / "ai" / "eval_questions.json"


def default_sample_path() -> Path:
    return repo_root() / "sample_data" / "household_energy_sample.csv"


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Read the question file: a list of {id, question, category, expect, ...}."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise AIError("The question file must hold a non-empty list.")
    return items


def load_eval_frame(path: str | Path | None = None) -> pd.DataFrame:
    """The sample dataset with its time index (no cleaning, so ground truth is easy to check)."""
    loaded = load_file(path or default_sample_path())
    candidates = detect_time_columns(loaded.df)
    return build_time_index(loaded.df, candidates[0].columns).df


# ------------------------------------------------------------------------------------ scoring


@dataclass(frozen=True)
class QuestionResult:
    id: str
    category: str
    in_scope: bool
    valid_query: bool
    matched: bool | None
    refused: bool
    number_check_triggered: bool
    latency: float
    provider: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class EvalSummary:
    n_questions: int
    n_in_scope: int
    n_out_of_scope: int
    valid_query_rate: float | None
    exact_match_rate: float | None
    refusal_correctness: float | None
    false_refusal_rate: float | None
    number_check_trigger_rate: float | None
    median_latency: float | None


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)
    return a == b


def tables_match(expected: dict[str, Any], table: ResultTable) -> bool:
    """Does an executed table equal the ground truth? Labels exactly, numbers within tolerance."""
    if list(expected["columns"]) != list(table.columns) or len(expected["rows"]) != len(table.rows):
        return False
    for want, got in zip(expected["rows"], table.rows, strict=True):
        if len(want) != len(got) or not all(_same(w, g) for w, g in zip(want, got, strict=True)):
            return False
    return True


def score_answer(
    item: dict[str, Any], answer: AnalystAnswer | None, latency: float, error: str | None = None
) -> QuestionResult:
    """Score one answer against its ground truth (`answer` is None if the call failed)."""
    expect = item["expect"]
    in_scope = expect["type"] == "table"
    evidence = answer.evidence if answer is not None else None
    refused = bool(answer and answer.refused)
    matched = None
    if in_scope:
        matched = bool(evidence is not None and tables_match(expect, evidence.table))
    return QuestionResult(
        id=item["id"],
        category=item.get("category", ""),
        in_scope=in_scope,
        valid_query=evidence is not None,
        matched=matched,
        refused=refused,
        number_check_triggered=bool(answer and answer.rewritten),
        latency=latency,
        provider=answer.provider if answer else None,
        error=error or (answer.error if answer else None),
    )


def _rate(hits: int, total: int) -> float | None:
    return hits / total if total else None


def summarize(results: list[QuestionResult]) -> EvalSummary:
    """The five measures over a set of scored questions."""
    inside = [r for r in results if r.in_scope]
    outside = [r for r in results if not r.in_scope]
    answered = [r for r in inside if r.valid_query]
    return EvalSummary(
        n_questions=len(results),
        n_in_scope=len(inside),
        n_out_of_scope=len(outside),
        valid_query_rate=_rate(sum(r.valid_query for r in inside), len(inside)),
        exact_match_rate=_rate(sum(bool(r.matched) for r in inside), len(inside)),
        refusal_correctness=_rate(sum(r.refused for r in outside), len(outside)),
        false_refusal_rate=_rate(sum(r.refused for r in inside), len(inside)),
        number_check_trigger_rate=_rate(
            sum(r.number_check_triggered for r in answered), len(answered)
        ),
        median_latency=statistics.median(r.latency for r in results) if results else None,
    )


# -------------------------------------------------------------------------------------- run


def run_eval(
    questions: list[dict[str, Any]],
    df: pd.DataFrame,
    client: LLMClient,
    *,
    mode: AnswerMode = "analyst",
    pause: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.perf_counter,
) -> list[QuestionResult]:
    """Ask every question and score the answers. A failed call is recorded, not raised."""
    context = build_context(df)
    session = AISession(guard=SessionGuard(question_cap=len(questions) + 1, min_interval=0.0))
    results: list[QuestionResult] = []
    for i, item in enumerate(questions):
        if i and pause:
            sleep(pause)
        started = clock()
        try:
            answer = ask(item["question"], mode, context, df, client, session=session)
            error = None
        except AIError as exc:
            answer, error = None, type(exc).__name__
        results.append(score_answer(item, answer, clock() - started, error))
    return results


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_report(summary: EvalSummary, results: list[QuestionResult]) -> str:
    """A readable table of the measures and of each question."""
    lines = [
        f"Questions: {summary.n_questions} ({summary.n_in_scope} in scope, "
        f"{summary.n_out_of_scope} out of scope)",
        f"Valid query rate:           {_pct(summary.valid_query_rate)}",
        f"Exact match rate:           {_pct(summary.exact_match_rate)}",
        f"Refusal correctness:        {_pct(summary.refusal_correctness)}",
        f"False refusal rate:         {_pct(summary.false_refusal_rate)}",
        f"Number check trigger rate:  {_pct(summary.number_check_trigger_rate)}",
        "Median latency:             "
        + ("n/a" if summary.median_latency is None else f"{summary.median_latency:.2f} s"),
        "",
        f"{'id':<28}{'scope':<8}{'valid':<7}{'match':<7}{'refused':<9}{'rewritten':<10}{'s':>6}",
    ]
    for r in results:
        lines.append(
            f"{r.id:<28}{'in' if r.in_scope else 'out':<8}{'yes' if r.valid_query else 'no':<7}"
            f"{'-' if r.matched is None else ('yes' if r.matched else 'no'):<7}"
            f"{'yes' if r.refused else 'no':<9}{'yes' if r.number_check_triggered else 'no':<10}"
            f"{r.latency:>6.2f}"
        )
    return "\n".join(lines)


def report_json(summary: EvalSummary, results: list[QuestionResult]) -> str:
    return json.dumps(
        {"summary": asdict(summary), "questions": [asdict(r) for r in results]}, indent=2
    )
