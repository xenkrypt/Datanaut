import copy
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from datanaut.core.cleaner import OpSpec, apply_op
from datanaut.core.fingerprint import fingerprint
from datanaut.core.log import PipelineLog
from datanaut.core.validation import RangeRule, RuleSet, rule_action_spec
from datanaut.core.versioning import (
    Recipe,
    VersionStore,
    compare_frames,
    replay_recipe,
)
from datanaut.errors import CleaningError, RecipeMismatchError, VersioningError

NaN = np.nan


def raw_frame():
    idx = pd.date_range("2024-01-01", periods=12, freq="h")
    df = pd.DataFrame(
        {
            "v": [1.0, 2.0, NaN, 4.0, 5.0, 500.0, 7.0, NaN, NaN, 10.0, 11.0, 12.0],
            "w": np.arange(12.0),
        },
        index=idx,
    )
    return pd.concat([df, df.iloc[[2]]]).sort_index(kind="stable")  # one exact duplicate row


DEDUP = OpSpec(operation="drop_duplicates")
FILL = OpSpec(operation="handle_missing", params={"column": "v", "method": "interpolate_time"})
FLAG = OpSpec(operation="outliers", params={"column": "v"})
BLANK = OpSpec(operation="outliers", params={"column": "v", "action": "set_missing"})


def direct(df, *specs):
    for spec in specs:
        df = apply_op(df, spec).df
    return df


def store_with_chain(**kw):
    store = VersionStore(raw_frame(), **kw)
    for spec in (DEDUP, FILL, FLAG):
        store.apply(spec)
    return store


# ------------------------------------------------------------------ creation and chains


def test_v1_is_the_raw_data():
    raw = raw_frame()
    store = VersionStore(raw)
    v1 = store.version("V1")
    assert (v1.version_id, v1.parent_id, v1.op) == ("V1", None, None)
    assert v1.fingerprint == fingerprint(raw)
    assert (v1.rows, v1.columns) == (13, 2)
    assert store.head == "V1" and set(v1.environment) == {"datanaut", "python", "pandas", "numpy"}
    pd.testing.assert_frame_equal(store.current(), raw)


def test_apply_builds_a_linked_chain_with_fingerprints():
    store = store_with_chain()
    versions = [store.version(f"V{i}") for i in (1, 2, 3, 4)]
    assert [v.parent_id for v in versions] == [None, "V1", "V2", "V3"]
    assert [v.op.operation if v.op else None for v in versions] == [
        None,
        "drop_duplicates",
        "handle_missing",
        "outliers",
    ]
    assert store.head == "V4"
    expected = raw_frame()
    for v, spec in zip(versions[1:], (DEDUP, FILL, FLAG), strict=True):
        expected = apply_op(expected, spec).df
        assert v.fingerprint == fingerprint(expected)
        assert (v.rows, v.columns) == (len(expected), expected.shape[1])
    assert versions[3].columns == 3  # the flag column was added


def test_replay_equals_direct_application_for_every_version():
    store = store_with_chain()
    raw = raw_frame()
    chain = [DEDUP, FILL, FLAG]
    for i in range(4):
        pd.testing.assert_frame_equal(store.get(f"V{i + 1}"), direct(raw, *chain[:i]))


def test_failed_apply_changes_nothing():
    store = store_with_chain()
    versions, head, entries = len(store.history()), store.head, len(store.log)
    with pytest.raises(CleaningError):
        store.apply(OpSpec(operation="handle_missing", params={"column": "nope", "method": "drop"}))
    with pytest.raises(CleaningError):
        store.apply(OpSpec(operation="not_an_operation"))
    assert (len(store.history()), store.head, len(store.log)) == (versions, head, entries)


def test_unknown_version_is_a_clear_error():
    store = VersionStore(raw_frame())
    for call in (store.get, store.version, store.restore):
        with pytest.raises(VersioningError, match="no version 'V9'"):
            call("V9")
    with pytest.raises(VersioningError):
        store.compare("V1", "V9")


# ------------------------------------------------------------------ cache


def test_cache_keeps_the_three_most_recently_used_and_never_changes_results():
    cached = store_with_chain()
    uncached = store_with_chain(cache_size=0)
    for vid in ("V1", "V2", "V3", "V4"):
        pd.testing.assert_frame_equal(cached.get(vid), uncached.get(vid))
    assert uncached.cached_ids() == []
    cached.get("V1")
    cached.get("V2")
    cached.get("V3")
    cached.get("V4")
    assert cached.cached_ids() == ["V2", "V3", "V4"]
    cached.get("V2")  # touching V2 makes it the most recent
    assert cached.cached_ids() == ["V3", "V4", "V2"]
    for vid in ("V1", "V3", "V4", "V2", "V1"):  # many cache hits and misses, same answers
        pd.testing.assert_frame_equal(cached.get(vid), uncached.get(vid))


def test_returned_frames_are_copies_so_mutation_cannot_poison_the_cache_or_raw():
    store = store_with_chain()
    snapshot = {v: store.get(v) for v in ("V1", "V2", "V3", "V4")}
    for v in snapshot:
        frame = store.get(v)
        frame.iloc[:, 0] = -999.0
        frame["extra"] = 1
    raw_copy, current_copy = store.raw(), store.current()
    raw_copy.iloc[:, 0] = -999.0
    current_copy.iloc[:, 0] = -999.0
    for v, expected in snapshot.items():
        pd.testing.assert_frame_equal(store.get(v), expected)
    pd.testing.assert_frame_equal(store.raw(), raw_frame())


def test_mutating_the_original_input_after_creation_does_not_affect_the_store():
    raw = raw_frame()
    store = VersionStore(raw)
    raw.iloc[:, 0] = -1.0
    pd.testing.assert_frame_equal(store.get("V1"), raw_frame())
    assert fingerprint(store.get("V1")) == store.version("V1").fingerprint


def test_replay_detects_a_changed_result_instead_of_returning_wrong_data():
    store = store_with_chain(cache_size=0)
    store._versions["V3"] = store._versions["V3"].model_copy(update={"fingerprint": "0" * 64})
    with pytest.raises(VersioningError, match="Rebuilding V3 did not reproduce"):
        store.get("V4")


# ------------------------------------------------------------------ restore and branching


def test_restore_then_branch_keeps_all_history():
    store = store_with_chain()
    v3_before = store.get("V3")
    store.restore("V2")
    assert store.head == "V2"
    branch = store.apply(BLANK)
    assert (branch.version_id, branch.parent_id) == ("V5", "V2")
    assert store.head == "V5"
    assert [n.version.version_id for n in store.history()] == ["V1", "V2", "V3", "V4", "V5"]
    pd.testing.assert_frame_equal(store.get("V3"), v3_before)  # the old branch still rebuilds
    pd.testing.assert_frame_equal(store.get("V5"), direct(raw_frame(), DEDUP, BLANK))
    assert [v.version_id for v in store.chain("V5")] == ["V1", "V2", "V5"]
    assert [v.version_id for v in store.chain("V4")] == ["V1", "V2", "V3", "V4"]


def test_history_is_a_tree_in_display_order():
    store = store_with_chain()
    store.restore("V2")
    store.apply(BLANK)
    nodes = store.history()
    assert [(n.version.version_id, n.depth) for n in nodes] == [
        ("V1", 0),
        ("V2", 1),
        ("V3", 2),
        ("V4", 3),
        ("V5", 2),
    ]
    by_id = {n.version.version_id: n for n in nodes}
    assert by_id["V2"].children == ["V3", "V5"] and by_id["V4"].children == []
    assert [n.is_head for n in nodes] == [False, False, False, False, True]


def test_restore_to_the_current_head_is_harmless_and_still_logged():
    store = VersionStore(raw_frame())
    store.restore("V1")
    assert store.head == "V1" and len(store.log) == 1


# ------------------------------------------------------------------ pipeline log sync


def test_every_apply_and_restore_is_logged_in_step_with_the_versions():
    log = PipelineLog()
    log.append("load", "Loaded.")
    store = VersionStore(raw_frame(), log)
    assert store.log is log
    for spec in (DEDUP, FILL):
        store.apply(spec)
    store.restore("V2")
    store.apply(FLAG)
    ops = [e.operation for e in log]
    assert ops == ["load", "drop_duplicates", "handle_missing", "restore_version", "outliers"]
    assert [e.step_id for e in log] == [1, 2, 3, 4, 5]
    linked = log.entries[1].params["dataset_version"]
    assert linked == {
        "version_id": "V2",
        "parent_id": "V1",
        "fingerprint": store.version("V2").fingerprint,
    }
    assert log.entries[4].params["dataset_version"]["parent_id"] == "V2"
    restore = log.entries[3]
    assert restore.params == {"from_version": "V3", "to_version": "V2"}
    assert restore.rows_before == store.version("V3").rows
    assert PipelineLog.from_json(log.to_json()).entries == log.entries


# ------------------------------------------------------------------ compare


def test_compare_detects_specific_cell_changes():
    store = store_with_chain()
    diff = store.compare("V2", "V3")  # interpolation filled V's gaps
    assert diff.cells_changed == {"v": 3} and diff.cells_changed_total == 3
    first = diff.examples[0]
    assert (first.row, first.column, first.before, first.after) == (
        "2024-01-01 02:00:00",
        "v",
        None,
        3.0,
    )
    assert [e.after for e in diff.examples] == [3.0, 8.0, 9.0]  # linear in time between 7 and 10
    assert [e.before for e in diff.examples] == [None, None, None]
    assert [e.row for e in diff.examples][1:] == ["2024-01-01 07:00:00", "2024-01-01 08:00:00"]
    assert diff.rows_added == diff.rows_removed == 0 and not diff.identical
    assert diff.columns_added == [] and diff.dtype_changes == {}


def test_compare_reports_rows_columns_and_dtypes():
    store = store_with_chain()
    rows = store.compare("V1", "V2")
    assert (rows.rows_removed, rows.rows_added, rows.removed_sample) == (
        1,
        0,
        ["2024-01-01 02:00:00"],
    )
    cols = store.compare("V3", "V4")
    assert cols.columns_added == ["v__outlier"] and cols.cells_changed == {}
    back = store.compare("V4", "V3")
    assert back.columns_removed == ["v__outlier"]
    ints = compare_frames(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1.0, 2.0]}))
    assert (
        ints.dtype_changes == {"a": "int64 -> float64"}
        and not ints.identical
        and ints.cells_changed_total == 0
    )


def test_compare_identical_versions_and_a_version_with_itself():
    store = store_with_chain()
    same = store.compare("V3", "V3")
    assert same.identical and same.cells_changed_total == 0 and same.examples == []
    assert compare_frames(raw_frame(), raw_frame()).identical


def test_compare_handles_duplicate_index_labels_and_nan_equality():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02"])
    a = pd.DataFrame({"v": [1.0, 2.0, NaN, 4.0]}, index=idx)
    b = a.copy()
    assert compare_frames(a, b).identical  # NaN equals NaN, repeated labels match by order
    b.iloc[1, 0] = 99.0
    diff = compare_frames(a, b)
    assert (
        diff.cells_changed == {"v": 1}
        and diff.examples[0].before == 2.0
        and diff.examples[0].after == 99.0
    )
    c = a.iloc[[0, 1, 3]]  # one of the three repeated rows removed
    removed = compare_frames(a, c)
    assert removed.rows_removed == 1 and removed.rows_added == 0
    longer = pd.concat([a, a.iloc[[0]]])
    assert compare_frames(a, longer).rows_added == 1


def test_compare_caps_examples_but_counts_everything():
    a = pd.DataFrame({"v": np.zeros(100), "w": np.zeros(100)})
    b = a + 1
    diff = compare_frames(a, b)
    assert diff.cells_changed_total == 200 and diff.cells_changed == {"v": 100, "w": 100}
    assert len(diff.examples) == 20
    assert compare_frames(a, pd.DataFrame({"v": np.zeros(30)})).rows_removed == 70


def test_compare_text_columns_and_mixed_types():
    a = pd.DataFrame({"t": ["x", "y", None], "m": [1, "a", 3]})
    b = pd.DataFrame({"t": ["x", "z", None], "m": [1, "b", 3.0]})
    diff = compare_frames(a, b)
    assert diff.cells_changed == {"t": 1, "m": 1}


# ------------------------------------------------------------------ recipes


def test_recipe_json_round_trip_and_replay_to_the_same_fingerprint():
    store = store_with_chain()
    recipe = store.to_recipe()
    assert [s.op.operation for s in recipe.steps] == [
        "drop_duplicates",
        "handle_missing",
        "outliers",
    ]
    assert (
        recipe.head_fingerprint == store.version("V4").fingerprint == recipe.steps[-1].fingerprint
    )
    text = recipe.to_json()
    assert Recipe.from_json(text) == recipe and json.loads(text)["format_version"] == 1
    for form in (recipe, json.loads(text), text):
        rebuilt = replay_recipe(raw_frame(), form)
        assert fingerprint(rebuilt.current()) == recipe.head_fingerprint
        pd.testing.assert_frame_equal(rebuilt.current(), store.current())
        assert rebuilt.head == "V4" and len(rebuilt.history()) == 4


def test_recipe_for_an_older_version_and_for_a_branch_follows_only_its_ancestry():
    store = store_with_chain()
    assert [s.version_id for s in store.to_recipe("V3").steps] == ["V2", "V3"]
    store.restore("V2")
    store.apply(BLANK)
    recipe = store.to_recipe()
    assert [s.op.operation for s in recipe.steps] == ["drop_duplicates", "outliers"]
    rebuilt = replay_recipe(raw_frame(), recipe)
    pd.testing.assert_frame_equal(rebuilt.current(), store.current())
    assert [s.version_id for s in recipe.steps] == ["V2", "V5"]  # original ids are kept as labels


def test_replay_with_different_raw_data_is_a_clear_mismatch():
    recipe = store_with_chain().to_recipe()
    other = raw_frame()
    other.iloc[0, 0] = 123.0
    with pytest.raises(RecipeMismatchError, match="raw data does not match"):
        replay_recipe(other, recipe)


def test_replay_names_the_first_step_whose_result_differs():
    recipe = store_with_chain().to_recipe()
    tampered = recipe.model_copy(deep=True)
    tampered.steps[1] = tampered.steps[1].model_copy(update={"fingerprint": "f" * 64})
    with pytest.raises(RecipeMismatchError, match=r"Step V3 \(handle_missing\)"):
        replay_recipe(raw_frame(), tampered)
    swapped = recipe.model_copy(deep=True)
    swapped.steps[1] = swapped.steps[1].model_copy(
        update={"op": OpSpec(operation="handle_missing", params={"column": "v", "method": "ffill"})}
    )
    with pytest.raises(RecipeMismatchError, match="Step V3"):
        replay_recipe(raw_frame(), swapped)


def test_mismatch_message_mentions_library_drift_when_the_environment_differs():
    recipe = store_with_chain().to_recipe()
    old = recipe.model_copy(update={"environment": {**recipe.environment, "pandas": "1.0.0"}})
    other = raw_frame()
    other.iloc[0, 0] = 5.0
    with pytest.raises(RecipeMismatchError, match="made with pandas 1.0.0"):
        replay_recipe(other, old)


def test_malformed_recipes_are_rejected_clearly():
    with pytest.raises(VersioningError, match="not valid"):
        replay_recipe(raw_frame(), "{not json")
    with pytest.raises(VersioningError, match="not valid"):
        replay_recipe(raw_frame(), {"steps": []})
    bad = store_with_chain().to_recipe().model_copy(update={"format_version": 99})
    with pytest.raises(VersioningError, match="format 99"):
        replay_recipe(raw_frame(), bad)


def test_replay_appends_to_a_given_log_and_survives_rule_based_steps():
    store = VersionStore(raw_frame())
    store.apply(DEDUP)
    store.apply(
        rule_action_spec(RangeRule(id="cap", name="cap", column="v", max=100), "set_missing")
    )
    store.apply(FILL)
    recipe = store.to_recipe()
    log = PipelineLog()
    rebuilt = replay_recipe(raw_frame(), recipe.to_json(), log=log)
    assert [e.operation for e in log] == ["drop_duplicates", "apply_rule_action", "handle_missing"]
    assert fingerprint(rebuilt.current()) == recipe.head_fingerprint
    assert rebuilt.current()["v"].isna().sum() == 0 and rebuilt.current()["v"].max() < 100


def test_input_to_the_store_and_recipe_functions_is_never_modified():
    raw = raw_frame()
    snapshot = copy.deepcopy(raw)
    store = VersionStore(raw)
    store.apply(DEDUP)
    store.apply(FILL)
    replay_recipe(raw, store.to_recipe())
    pd.testing.assert_frame_equal(raw, snapshot)


# ------------------------------------------------------------------ property: any chain replays

_MENU = [
    OpSpec(operation="drop_duplicates", params={"conflicts": "mean"}),
    OpSpec(operation="handle_missing", params={"column": "v", "method": "ffill", "limit": 2}),
    OpSpec(
        operation="handle_missing",
        params={"column": "v", "method": "interpolate_time", "max_gap": 3},
    ),
    OpSpec(operation="outliers", params={"column": "v"}),
    OpSpec(operation="outliers", params={"column": "v", "action": "set_missing", "window": 5}),
    OpSpec(operation="outliers", params={"column": "v", "action": "clip", "window": 5}),
    OpSpec(operation="check_warnings"),
]


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.sampled_from(range(len(_MENU))), min_size=1, max_size=6),
    st.lists(
        st.one_of(st.floats(-50, 50, allow_nan=False), st.just(NaN)), min_size=12, max_size=30
    ),
)
def test_property_any_chain_rebuilds_and_replays_to_the_same_fingerprints(picks, values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h")
    raw = pd.DataFrame({"v": values}, index=idx)
    store = VersionStore(raw, cache_size=1)
    for i in picks:
        store.apply(_MENU[i])
    expected = raw
    for version in store.chain()[1:]:
        expected = apply_op(expected, version.op).df
        assert (
            fingerprint(store.get(version.version_id))
            == version.fingerprint
            == fingerprint(expected)
        )
    rebuilt = replay_recipe(raw, store.to_recipe())
    assert fingerprint(rebuilt.current()) == store.version(store.head).fingerprint


# ------------------------------------------------------------------ review fixes


def test_the_store_keeps_its_own_copy_of_every_spec():
    store = VersionStore(raw_frame())
    spec = OpSpec(operation="handle_missing", params={"column": "v", "method": "interpolate_time"})
    returned = store.apply(spec)
    spec.params["method"] = "drop"  # the caller changes its object afterwards
    assert store.version("V2").op.params["method"] == "interpolate_time"
    returned.op.params["method"] = "drop"  # ...or the object the store handed back
    store.version("V2").op.params["column"] = "other"
    store.chain()[1].op.params["method"] = "ffill"
    store.history()[1].version.op.params["method"] = "ffill"
    assert store.version("V2").op.params == {"column": "v", "method": "interpolate_time"}
    recipe = store.to_recipe()
    recipe.steps[0].op.params["method"] = "drop"
    assert store.to_recipe().steps[0].op.params["method"] == "interpolate_time"
    rebuilt = VersionStore(raw_frame(), cache_size=0)
    rebuilt.apply(
        OpSpec(operation="handle_missing", params={"column": "v", "method": "interpolate_time"})
    )
    pd.testing.assert_frame_equal(store.get("V2"), rebuilt.get("V2"))  # still rebuilds correctly


def test_restore_and_get_do_not_expose_internal_metadata():
    store = store_with_chain()
    store.restore("V2").op.params["method"] = "tampered"
    assert store.version("V2").op.operation == "drop_duplicates"
    assert len(store.version("V2").fingerprint) == 64  # reading is fine
    pd.testing.assert_frame_equal(store.get("V4"), direct(raw_frame(), DEDUP, FILL, FLAG))


def test_rules_given_by_file_are_inlined_so_recipes_do_not_depend_on_the_file(tmp_path):
    path = tmp_path / "rules.json"
    RuleSet(name="r", rules=[RangeRule(id="cap", name="cap", column="v", max=100)]).save_json(path)
    ref = OpSpec(
        operation="apply_rule_action",
        params={"action": "set_missing", "ruleset_path": str(path), "rule_id": "cap"},
    )
    store = VersionStore(raw_frame())
    store.apply(DEDUP)
    version = store.apply(ref)
    assert "rule" in version.op.params and "ruleset_path" not in version.op.params
    recipe = store.to_recipe()
    expected = fingerprint(store.current())
    path.write_text("changed or deleted later", encoding="utf-8")  # the file no longer matters
    path.unlink()
    assert fingerprint(replay_recipe(raw_frame(), recipe).current()) == expected
    assert store.current()["v"].max() < 100


def test_history_and_chain_handle_very_long_chains_without_recursion():
    raw = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=pd.date_range("2024", periods=3, freq="h"))
    store = VersionStore(raw)
    warn = OpSpec(operation="check_warnings")
    for _ in range(1500):
        store.apply(warn)
    nodes = store.history()
    assert len(nodes) == 1501 and nodes[-1].depth == 1500 and nodes[-1].is_head
    assert [n.version.version_id for n in nodes[:3]] == ["V1", "V2", "V3"]
    assert len(store.chain()) == 1501


def test_a_step_that_cannot_run_during_replay_names_the_step():
    from datanaut.core.versioning import RecipeStep

    good = store_with_chain().to_recipe()
    broken = good.model_copy(
        update={
            "steps": [
                RecipeStep(
                    version_id="V2",
                    op=OpSpec(
                        operation="handle_missing", params={"column": "nope", "method": "drop"}
                    ),
                    fingerprint="0" * 64,
                    rows=1,
                    columns=1,
                )
            ]
        }
    )
    with pytest.raises(
        RecipeMismatchError, match=r"Step V2 \(handle_missing\) could not be applied.*nope"
    ):
        replay_recipe(raw_frame(), broken)


def test_recipe_text_with_a_byte_order_mark_loads():
    recipe = store_with_chain().to_recipe()
    assert Recipe.from_json("\ufeff" + recipe.to_json()) == recipe
