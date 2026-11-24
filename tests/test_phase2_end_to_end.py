import hashlib
from pathlib import Path

import pandas as pd
import pytest

from datanaut.core.characterize import characterize
from datanaut.core.cleaner import propose_fixes
from datanaut.core.export import write_safe_csv
from datanaut.core.fingerprint import fingerprint
from datanaut.core.loader import load_file
from datanaut.core.log import PipelineLog
from datanaut.core.profiler import profile
from datanaut.core.timeindex import build_time_index, detect_time_columns, detect_value_columns
from datanaut.core.validation import RangeRule, RuleSet, rule_action_spec, validate
from datanaut.core.versioning import Recipe, VersionStore, replay_recipe
from datanaut.errors import RecipeMismatchError

ROOT = Path(__file__).resolve().parent.parent


def run_phase2(source):
    """Phase 1 chain, then proposals, versioned fixes, validation and one rule action."""
    log = PipelineLog()
    loaded = load_file(source)
    log.extend(loaded.log)
    indexed = build_time_index(loaded.df, detect_time_columns(loaded.df)[0].columns)
    log.extend(indexed.log)
    prof = profile(indexed.df)
    value_col = detect_value_columns(indexed.df)[0]
    char = characterize(indexed.df, value_col)
    log.extend([prof.log_entry, char.log_entry])
    proposals = propose_fixes(indexed.df, prof, char)

    store = VersionStore(indexed.df, log)
    selected = [p for p in proposals if p.default_selected]
    for proposal in selected:
        store.apply(proposal.spec)

    cap = float(store.current()[value_col].quantile(0.9))
    rule = RangeRule(id="cap", name="Upper cap", column=value_col, max=cap)
    ruleset = RuleSet(name="e2e", rules=[rule])
    before = validate(store.current(), ruleset)
    log.extend([before.log_entry])
    action = rule_action_spec(rule, "set_missing")
    store.apply(action)
    after = validate(store.current(), ruleset)
    return {
        "loaded": loaded,
        "indexed": indexed,
        "proposals": proposals,
        "selected": selected,
        "store": store,
        "log": log,
        "before": before,
        "after": after,
        "action": action,
        "value_col": value_col,
    }


def test_every_variant_goes_through_clean_validate_version_and_replay(variant, tmp_path):
    path = tmp_path / variant.filename
    path.write_bytes(variant.data)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    r = run_phase2(path)
    store, log = r["store"], r["log"]

    # the original file bytes were never modified
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert path.read_bytes() == variant.data

    # the version chain is exactly: raw, each selected proposal, then the rule action
    chain = store.chain()
    expected_ops = [p.spec for p in r["selected"]] + [r["action"]]
    assert [v.version_id for v in chain] == [f"V{i}" for i in range(1, len(expected_ops) + 2)]
    assert [v.parent_id for v in chain] == [None, *[v.version_id for v in chain[:-1]]]
    assert [v.op for v in chain[1:]] == expected_ops
    assert chain[0].fingerprint == fingerprint(r["indexed"].df)
    assert store.head == chain[-1].version_id
    assert fingerprint(store.current()) == chain[-1].fingerprint

    # validation found something to handle and the rule action resolved it
    assert r["before"].results[0].violations > 0
    assert r["after"].results[0].status == "pass"

    # the final recipe replays to the same fingerprint, from an object or from JSON
    recipe = store.to_recipe()
    for form in (recipe, recipe.to_json()):
        rebuilt = replay_recipe(r["indexed"].df, form)
        assert fingerprint(rebuilt.current()) == chain[-1].fingerprint
    assert [s.op for s in recipe.steps] == expected_ops

    # the log has an entry for every step, in order, linked to the versions
    ops = [e.operation for e in log]
    for operation in (
        "load",
        "normalize_missing",
        "convert_numeric",
        "build_time_index",
        "profile",
        "characterize",
        "validate",
    ):
        assert operation in ops, operation
    linked = [e for e in log if "dataset_version" in e.params]
    assert [e.params["dataset_version"]["version_id"] for e in linked] == [
        v.version_id for v in chain[1:]
    ]
    assert [e.operation for e in linked] == [op.operation for op in expected_ops]
    assert [e.step_id for e in log] == list(range(1, len(log) + 1))
    assert PipelineLog.from_json(log.to_json()).entries == log.entries

    # the cleaned data can be exported safely
    out = tmp_path / "cleaned.csv"
    export = write_safe_csv(store.current(), out)
    assert len(pd.read_csv(out, index_col=0)) == export.rows == len(store.current())


def test_replay_on_changed_raw_data_is_refused_for_every_variant(variant):
    r = run_phase2(variant.data)
    recipe = r["store"].to_recipe()
    changed = r["indexed"].df.copy()
    changed.iloc[0, changed.columns.get_loc(r["value_col"])] += 1.0
    with pytest.raises(RecipeMismatchError, match="raw data does not match"):
        replay_recipe(changed, recipe)


def test_sample_dataset_produces_the_expected_version_chain():
    r = run_phase2(ROOT / "sample_data" / "household_energy_sample.csv")
    chain = r["store"].chain()
    ops = [
        (v.op.operation, v.op.params.get("column", v.op.params.get("method"))) for v in chain[1:]
    ]
    assert ops == [
        ("drop_duplicates", None),
        ("handle_missing", "global_active_power"),
        ("handle_missing", "voltage"),
        ("apply_rule_action", None),
    ]
    assert [v.rows for v in chain] == [2876, 2870, 2870, 2870, 2870]
    power_gaps = r["store"].compare("V2", "V3").cells_changed
    assert power_gaps == {"global_active_power": 57}  # 57 short-gap values filled, 8 left alone
    assert r["store"].compare("V3", "V4").cells_changed == {"voltage": 28}
    assert r["store"].current()["global_active_power"].isna().sum() > 8  # plus the capped values
    recipe = Recipe.from_json(r["store"].to_recipe().to_json())
    assert replay_recipe(r["indexed"].df, recipe).head == "V5"


def test_restore_and_branch_inside_the_pipeline_still_replays():
    r = run_phase2(ROOT / "sample_data" / "household_energy_sample.csv")
    store = r["store"]
    store.restore("V2")
    store.apply(r["selected"][1].spec)  # a different branch from V2
    assert store.chain()[-1].parent_id == "V2"
    rebuilt = replay_recipe(r["indexed"].df, store.to_recipe())
    assert fingerprint(rebuilt.current()) == fingerprint(store.current())
    assert any(e.operation == "restore_version" for e in r["log"])
