import json

import numpy as np
import pandas as pd
import pytest

from datanaut.core.cleaner import OpSpec
from datanaut.core.versioning import Recipe
from datanaut.errors import RuleSetError
from datanaut.ui import actions, charts, cleaning, display, rules
from datanaut.ui.state import WizardState


@pytest.fixture
def mapped() -> WizardState:
    s = WizardState()
    actions.load_sample(s)
    time_cols, values = actions.suggest_mapping(s.loaded.df)
    actions.confirm_mapping(s, time_cols, None, values, {})
    s.take_messages()
    return s


# --- which proposals are applied, and in what order ---------------------------------------


def test_default_selection_applies_only_the_default_proposals_in_proposal_order(mapped):
    proposals = cleaning.get_proposals(mapped)
    specs = cleaning.chosen_specs(proposals, cleaning.default_selection(proposals))
    expected = [p.spec for p in proposals if p.default_selected]
    assert specs == expected and specs[0].operation == "drop_duplicates"
    assert [p.id for p in proposals if p.default_selected] == [
        p.id for p in proposals if p.spec in specs
    ]


def test_selection_honours_unchecking_and_the_chosen_alternative(mapped):
    proposals = cleaning.get_proposals(mapped)
    selected = {p.id: False for p in proposals}
    first_missing = next(p for p in proposals if p.category == "missing")
    selected[first_missing.id] = True
    specs = cleaning.chosen_specs(proposals, selected, {first_missing.id: 1})
    assert specs == [first_missing.alternatives[1].spec]
    assert specs[0] is not first_missing.alternatives[1].spec  # a copy, not shared
    assert cleaning.chosen_specs(proposals, {p.id: False for p in proposals}) == []
    assert cleaning.chosen_specs(proposals, selected, {first_missing.id: 99}) == [
        first_missing.spec
    ]  # an out-of-range choice falls back to the suggestion


def test_proposals_are_cached_per_version(mapped):
    first = cleaning.get_proposals(mapped)
    assert cleaning.get_proposals(mapped) is first


# --- applying, undo, restore --------------------------------------------------------------


def test_applying_default_fixes_makes_versions_and_log_entries_and_undo_restores(mapped):
    before_log, v1 = len(mapped.log), mapped.fingerprint
    proposals = cleaning.get_proposals(mapped)
    out = cleaning.apply_specs(mapped, cleaning.chosen_specs(proposals, {}))
    n = sum(p.default_selected for p in proposals)
    assert out.applied == n and out.error is None
    assert mapped.store.head == f"V{n + 1}" and len(mapped.log) == before_log + n
    assert mapped.fingerprint != v1
    assert [m.level for m in mapped.take_messages()] == ["success"]

    assert cleaning.undo(mapped) is True
    assert mapped.store.head == f"V{n}" and len(mapped.store.history()) == n + 1  # nothing deleted
    assert cleaning.restore(mapped, "V1") and mapped.fingerprint == v1
    assert cleaning.undo(mapped) is False and "nothing to undo" in mapped.messages[-1].text.lower()
    assert cleaning.restore(mapped, "V1") is False  # already there
    assert cleaning.restore(mapped, "V99") is False and mapped.messages[-1].level == "error"


def test_a_failing_fix_keeps_earlier_ones_and_names_the_operation(mapped):
    good = OpSpec(operation="drop_duplicates", params={"keep": "first"})
    bad = OpSpec(operation="handle_missing", params={"column": "nope", "method": "ffill"})
    out = cleaning.apply_specs(mapped, [good, bad])
    assert out.applied == 1 and out.error and "handle_missing" in out.error
    assert "earlier fix" in out.error and mapped.store.head == "V2"
    assert mapped.messages[-1].level == "error"


def test_applying_nothing_changes_nothing(mapped):
    out = cleaning.apply_specs(mapped, [])
    assert out.applied == 0 and mapped.store.head == "V1"
    assert mapped.messages[-1].level == "info"


# --- preview, compare, downloads ----------------------------------------------------------


def test_preview_does_not_change_the_data_and_shows_before_and_after(mapped):
    spec = OpSpec(operation="handle_missing", params={"column": "voltage", "method": "ffill"})
    head = mapped.fingerprint
    pv = cleaning.preview_spec(mapped, spec)
    assert mapped.fingerprint == head and mapped.store.head == "V1"
    assert cleaning.preview_spec(mapped, spec) is pv  # cached
    table = cleaning.preview_table(pv)
    assert table.loc[0, "Column"] == "voltage"
    assert table.loc[0, "Missing after"] <= table.loc[0, "Missing before"]
    assert cleaning.preview_column(mapped, spec, pv) == "voltage"
    fig = charts.before_after(mapped.store.current()["voltage"], pv.df["voltage"], "voltage")
    assert fig.layout.title.text and [t.name for t in fig.data] == ["Before", "After"]
    assert fig.layout.xaxis.title.text and fig.layout.yaxis.title.text


def test_compare_summarizes_the_difference_between_versions(mapped):
    cleaning.apply_specs(mapped, cleaning.chosen_specs(cleaning.get_proposals(mapped), {}))
    diff = cleaning.compare(mapped, "V1", mapped.store.head)
    lines = cleaning.diff_summary(diff)
    assert not diff.identical and any("cell(s) changed" in x or "row(s)" in x for x in lines)
    assert list(cleaning.diff_examples(diff).columns) == ["Row", "Column", "Before", "After"]
    assert cleaning.diff_summary(cleaning.compare(mapped, "V1", "V1")) == [
        "V1 and V1 are identical."
    ]


def test_downloads_are_built_in_memory_and_are_formula_safe(mapped, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cleaning.apply_specs(mapped, cleaning.chosen_specs(cleaning.get_proposals(mapped), {}))
    csv = cleaning.cleaned_csv(mapped).decode("utf-8")
    assert csv.splitlines()[0].startswith("timestamp") and len(csv.splitlines()) > 100
    assert len(json.loads(cleaning.log_json(mapped))["entries"]) == len(mapped.log)
    recipe = Recipe.from_json(cleaning.recipe_json(mapped))
    assert recipe.head_fingerprint == mapped.fingerprint
    assert recipe.source["time_columns"] == ["Date", "Time"]
    assert list(tmp_path.iterdir()) == []  # nothing was written


def test_cleaned_csv_neutralizes_text_that_looks_like_a_formula():
    s = WizardState()
    actions.load_upload(s, b"t,note,v\n2024-01-01,=SUM(A1),1\n2024-01-02,ok,2\n", "x.csv")
    actions.confirm_mapping(s, ["t"], None, ["v"], {})
    assert "'=SUM(A1)" in cleaning.cleaned_csv(s).decode()


# --- validation rules ---------------------------------------------------------------------


def test_suggested_rules_can_be_accepted_or_deleted(mapped):
    suggestions = rules.get_suggestions(mapped)
    first, second = suggestions[0].id, suggestions[1].id
    rules.accept_suggestion(mapped, first)
    rules.delete_suggestion(mapped, second)
    assert [r.id for r in mapped.rules] == [first] and mapped.rules[0].suggested is False
    assert first not in [r.id for r in mapped.suggestions]
    assert second not in [r.id for r in mapped.suggestions] and second not in [
        r.id for r in mapped.rules
    ]


def test_built_rules_are_checked_and_get_unique_ids(mapped):
    a = rules.build_rule(mapped, "range", column="voltage", min=200.0, max=250.0)
    rules.add_rule(mapped, a)
    b = rules.build_rule(mapped, "range", column="voltage", min=200.0, max=250.0)
    assert a.id != b.id
    with pytest.raises(RuleSetError, match="min cannot be greater"):
        rules.build_rule(mapped, "range", column="voltage", min=5.0, max=1.0)
    with pytest.raises(RuleSetError, match="not valid"):
        rules.build_rule(mapped, "frequency", expected_interval="soon")
    with pytest.raises(RuleSetError, match="Unknown"):
        rules.build_rule(mapped, "nonsense")


def test_save_then_load_round_trips_and_bad_files_give_friendly_errors(mapped):
    rules.add_rule(mapped, rules.build_rule(mapped, "range", column="voltage", min=1.0, max=300.0))
    rules.add_rule(mapped, rules.build_rule(mapped, "unique_timestamp"))
    text = rules.rules_json(mapped)
    other = WizardState()
    other.store = mapped.store
    assert rules.load_rules(other, text, "f.json") == 2
    assert [r.model_dump() for r in other.rules] == [r.model_dump() for r in mapped.rules]
    with pytest.raises(RuleSetError, match="not valid JSON"):
        rules.load_rules(other, "{oops", "f.json")
    with pytest.raises(RuleSetError, match="problems"):
        rules.load_rules(other, '{"name": "x", "rules": [{"type": "range"}]}', "f.json")
    assert len(other.rules) == 2  # a bad file leaves the rules alone


def test_validation_runs_logs_and_lists_the_affected_rows(mapped):
    with pytest.raises(RuleSetError, match="at least one rule"):
        rules.run_validation(mapped)
    rules.add_rule(
        mapped, rules.build_rule(mapped, "range", column="voltage", min=0.0, max=1.0, name="tiny")
    )
    n = len(mapped.log)
    report = rules.run_validation(mapped)
    assert len(mapped.log) == n + 1 and mapped.log.entries[-1].read_only
    result = report.results[0]
    assert result.status == "fail" and result.violations > 0
    table = rules.result_table(report)
    assert table.loc[0, "Status"] == "✗ fail" and table.loc[0, "Violations"] == result.violations
    rows, total = rules.violating_rows(mapped, result.rule_id, cap=5)
    assert len(rows) == 5 and total == result.violations
    assert rules.report_is_current(mapped)


def test_a_rule_action_goes_through_the_version_flow(mapped):
    rules.add_rule(
        mapped, rules.build_rule(mapped, "range", column="voltage", min=0.0, max=1.0, name="tiny")
    )
    report = rules.run_validation(mapped)
    rule_id = report.results[0].rule_id
    assert rules.actions_for(mapped.rules[0]) == ["flag", "set_missing", "drop"]
    out = rules.apply_rule_action(mapped, rule_id, "set_missing")
    assert out.applied == 1 and mapped.store.head == "V2"
    assert mapped.store.current()["voltage"].isna().all()
    assert not rules.report_is_current(mapped)  # the data moved on; the old report is stale
    assert np.isclose(rules.run_validation(mapped).results[0].violations, 0)
    with pytest.raises(RuleSetError, match="no rule"):
        rules.apply_rule_action(mapped, "ghost", "flag")


def test_only_sensible_actions_are_offered(mapped):
    freq = rules.build_rule(mapped, "frequency", expected_interval="1h")
    uniq = rules.build_rule(mapped, "unique_timestamp")
    assert rules.actions_for(freq) == ["flag"] and rules.actions_for(uniq) == ["flag", "drop"]


# --- display.downsample_minmax ------------------------------------------------------------


def test_downsampling_keeps_the_extremes_of_every_bucket_and_respects_the_limit():
    rng = np.random.default_rng(0)
    values = rng.normal(0, 1, 20_000)
    values[777], values[15_001] = 50.0, -50.0
    s = pd.Series(values, index=pd.date_range("2024-01-01", periods=20_000, freq="min"))
    out = display.downsample_minmax(s, 1000)
    assert len(out) <= 1000 and out.max() == 50.0 and out.min() == -50.0
    assert out.index.is_monotonic_increasing
    edges = np.linspace(0, len(s), 501).astype(int)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        chunk = s.iloc[lo:hi]
        assert chunk.max() in out.iloc[:].to_numpy() and chunk.min() in out.to_numpy()
    small = s.iloc[:100]
    assert display.downsample_minmax(small, 1000) is small
    gappy = s.copy()
    gappy.iloc[:5000] = np.nan
    assert len(display.downsample_minmax(gappy, 200)) <= 200
