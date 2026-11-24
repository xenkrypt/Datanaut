import copy
import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from datanaut.core.cleaner import OpSpec, apply_op, available_operations
from datanaut.core.fingerprint import fingerprint
from datanaut.core.profiler import profile
from datanaut.core.validation import (
    MAX_SAMPLE,
    FrequencyRule,
    RangeRule,
    RelationRule,
    RequiredRule,
    RuleSet,
    UniqueTimestampRule,
    default_ruleset,
    evaluate_rule,
    rule_action_spec,
    validate,
)
from datanaut.errors import CleaningError, RuleSetError

NaN = np.nan


def hourly(values, name="v", **cols):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h")
    return pd.DataFrame({name: np.asarray(values, dtype=float), **cols}, index=idx)


def one(rule):
    return RuleSet(name="t", rules=[rule])


def result_of(df, rule):
    return validate(df, one(rule)).results[0]


def range_rule(**kw):
    return RangeRule(id="r", name="range", column="v", **kw)


# ------------------------------------------------------------------ range


def test_range_rule_pass_and_fail_with_exact_rows():
    df = hourly([1, 5, 10, -1, 11, NaN])
    res = result_of(df, range_rule(min=0, max=10))
    assert (res.status, res.violations, res.rows_checked) == ("fail", 2, 5)
    assert res.sample == ["2024-01-01 03:00:00", "2024-01-01 04:00:00"]
    assert (
        "1 below and 1 above [0, 10]" in res.message
        and "1 missing value(s) not checked" in res.message
    )
    assert result_of(hourly([0, 5, 10]), range_rule(min=0, max=10)).status == "pass"


def test_range_inclusive_versus_exclusive_boundaries():
    df = hourly([0, 5, 10])
    assert result_of(df, range_rule(min=0, max=10, inclusive=True)).violations == 0
    assert result_of(df, range_rule(min=0, max=10, inclusive=False)).violations == 2


def test_range_with_only_one_bound_and_validation_of_bounds():
    df = hourly([-5, 0, 5])
    assert result_of(df, range_rule(min=0)).violations == 1
    assert result_of(df, range_rule(max=0)).violations == 1
    with pytest.raises(ValidationError, match="min, a max or both"):
        range_rule()
    with pytest.raises(ValidationError, match="greater"):
        range_rule(min=5, max=1)


def test_range_on_missing_or_text_column_is_not_evaluated_not_passed():
    assert (
        result_of(hourly([1]), RangeRule(id="r", name="n", column="zz", min=0)).status
        == "not_evaluated"
    )
    res = result_of(hourly([1], t=["a"]), RangeRule(id="r", name="n", column="t", min=0))
    assert res.status == "not_evaluated" and "not numeric" in res.message


# ------------------------------------------------------------------ required


def test_required_rule_lists_rows_and_counts_per_column():
    df = hourly([1, NaN, 3, NaN], other=[1, 2, NaN, 4])
    rule = RequiredRule(id="req", name="complete", columns=["v", "other"])
    res = result_of(df, rule)
    assert (res.status, res.violations) == ("fail", 3)
    assert "v: 2" in res.message and "other: 1" in res.message
    assert result_of(hourly([1, 2]), RequiredRule(id="r", name="n", columns=["v"])).status == "pass"
    res = result_of(df, RequiredRule(id="r", name="n", columns=["v", "nope"]))
    assert res.status == "not_evaluated" and "nope" in res.message
    with pytest.raises(ValidationError):
        RequiredRule(id="r", name="n", columns=[])


# ------------------------------------------------------------------ frequency


def test_frequency_rule_flags_the_reading_after_each_long_gap():
    df = hourly(range(10)).drop(hourly(range(10)).index[[3, 4, 5, 8]])  # gaps of 4h and 2h
    tight = result_of(df, FrequencyRule(id="f", name="f", expected_interval="1h"))
    assert tight.violations == 2 and tight.sample == ["2024-01-01 06:00:00", "2024-01-01 09:00:00"]
    assert "about 4 expected reading" in tight.message and "longer than 1h;" in tight.message
    loose = result_of(df, FrequencyRule(id="f", name="f", expected_interval="1h", max_gap="3h"))
    assert loose.violations == 1 and loose.sample == ["2024-01-01 06:00:00"]
    exact = result_of(df, FrequencyRule(id="f", name="f", expected_interval="1h", max_gap="4h"))
    assert exact.status == "pass"  # a gap equal to the limit is allowed


def test_frequency_rule_is_order_independent_and_checks_edges():
    df = hourly(range(6)).drop(hourly(range(6)).index[2])
    shuffled = df.iloc[[4, 0, 3, 1]]
    rule = FrequencyRule(id="f", name="f", expected_interval="1h")
    rep = validate(shuffled, one(rule))
    assert rep.results[0].violations == 1 and rep.violation_mask("f").iloc[2]  # label 03:00
    assert result_of(hourly([1]), rule).status == "pass"


def test_frequency_rule_validation_and_no_time_index():
    with pytest.raises(ValidationError, match="not a valid time interval"):
        FrequencyRule(id="f", name="f", expected_interval="banana")
    with pytest.raises(ValidationError, match="positive"):
        FrequencyRule(id="f", name="f", expected_interval="0min")
    res = result_of(
        pd.DataFrame({"v": [1.0, 2.0]}), FrequencyRule(id="f", name="f", expected_interval="1h")
    )
    assert res.status == "not_evaluated" and "time index" in res.message


# ------------------------------------------------------------------ relation


@pytest.mark.parametrize(
    "op, violations",
    [
        ("<", [True, True, False]),
        ("<=", [False, True, False]),
        (">", [False, True, True]),
        (">=", [False, True, True]),
        ("==", [True, False, True]),
        ("!=", [False, True, False]),
    ],
)
def test_relation_operators(op, violations):
    # a = [1, 2, 3], b = [2, 2, 2]  ->  the rule says "a op b"; violations where it is false
    df = hourly([1, 2, 3], name="a", b=[2.0, 2.0, 2.0])
    rule = RelationRule(id="rel", name="rel", column_a="a", operator=op, column_b="b")
    mask = evaluate_rule(df, rule).mask
    expected = {
        "<": [False, True, True],
        "<=": [False, False, True],
        ">": [True, True, False],
        ">=": [True, False, False],
        "==": [True, False, True],
        "!=": [False, True, False],
    }[op]
    assert mask.tolist() == expected
    assert result_of(df, rule).violations == sum(expected)


def test_relation_skips_missing_values_and_reports_it():
    df = hourly([1, NaN, 5], name="a", b=[2.0, 2.0, NaN])
    rule = RelationRule(id="rel", name="rel", column_a="a", operator="<=", column_b="b")
    res = result_of(df, rule)
    assert (res.status, res.rows_checked) == ("pass", 1)
    assert "2 row(s) with a missing value not checked" in res.message


def test_relation_errors_and_operator_whitelist():
    with pytest.raises(ValidationError):
        RelationRule(id="r", name="n", column_a="a", operator="__import__('os')", column_b="b")
    with pytest.raises(ValidationError):
        RelationRule(id="r", name="n", column_a="a", operator="=", column_b="b")
    df = hourly([1, 2], name="a", t=["x", "y"])
    res = result_of(df, RelationRule(id="r", name="n", column_a="a", operator="<", column_b="t"))
    assert res.status == "not_evaluated" and "cannot be compared" in res.message
    res = result_of(df, RelationRule(id="r", name="n", column_a="a", operator="<", column_b="zz"))
    assert res.status == "not_evaluated"


# ------------------------------------------------------------------ unique timestamps


def test_unique_timestamp_rule():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-01"])
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    res = result_of(df, UniqueTimestampRule(id="u", name="unique"))
    assert (res.status, res.violations) == ("fail", 2)
    assert result_of(hourly([1, 2]), UniqueTimestampRule(id="u", name="u")).status == "pass"
    assert (
        result_of(pd.DataFrame({"v": [1.0]}), UniqueTimestampRule(id="u", name="u")).status
        == "not_evaluated"
    )


# ------------------------------------------------------------------ rule sets and JSON


def _ruleset():
    return RuleSet(
        name="plant",
        version="2.1",
        description="demo",
        rules=[
            range_rule(min=0, max=100),
            RequiredRule(id="req", name="complete", columns=["v"], severity="warning"),
            FrequencyRule(id="freq", name="hourly", expected_interval="1h", max_gap="2h"),
            RelationRule(id="rel", name="a<b", column_a="a", operator="<", column_b="b"),
            UniqueTimestampRule(id="uniq", name="unique"),
        ],
    )


def test_ruleset_json_round_trip_through_a_file(tmp_path):
    rs = _ruleset()
    path = tmp_path / "rules.json"
    rs.save_json(path)
    assert RuleSet.load_json(path) == rs
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [r["type"] for r in data["rules"]] == [
        "range",
        "required",
        "frequency",
        "relation",
        "unique_timestamp",
    ]
    assert data["version"] == "2.1"


def test_malformed_rule_files_give_clear_errors(tmp_path):
    def load(text):
        f = tmp_path / "bad.json"
        f.write_text(text, encoding="utf-8")
        return RuleSet.load_json(f)

    with pytest.raises(RuleSetError, match=r"not valid JSON \(line 1, column 2\)"):
        load("{,}")
    with pytest.raises(RuleSetError, match="JSON object"):
        load("[1, 2]")
    with pytest.raises(RuleSetError, match="name"):
        load('{"rules": []}')
    with pytest.raises(RuleSetError, match="does not match any of the expected tags"):
        load('{"name": "x", "rules": [{"type": "rnage", "id": "a", "name": "a"}]}')
    with pytest.raises(RuleSetError, match=r"rules\.0\.range\.column"):
        load('{"name": "x", "rules": [{"type": "range", "id": "a", "name": "a", "min": 1}]}')
    with pytest.raises(RuleSetError, match="not permitted"):
        load(
            '{"name": "x", "rules": [{"type": "unique_timestamp", "id": "a",'
            ' "name": "a", "evil": 1}]}'
        )
    with pytest.raises(RuleSetError, match="more than once"):
        load(
            '{"name":"x","rules":[{"type":"unique_timestamp","id":"a","name":"a"},'
            '{"type":"unique_timestamp","id":"a","name":"b"}]}'
        )
    with pytest.raises(RuleSetError, match="could not be opened"):
        RuleSet.load_json(tmp_path / "missing.json")


def test_ruleset_get_and_error_lists_known_ids():
    rs = _ruleset()
    assert rs.get("req").type == "required"
    with pytest.raises(RuleSetError, match="No rule with id 'zzz'.*Rules: r, req"):
        rs.get("zzz")


# ------------------------------------------------------------------ validate


def test_validate_report_structure_log_and_severity():
    df = hourly([1, 200, 3], a=[1.0, 5.0, 3.0], b=[2.0, 2.0, 2.0])
    rs = RuleSet(
        name="mix",
        rules=[
            range_rule(min=0, max=100, severity="warning"),
            RelationRule(id="rel", name="a<b", column_a="a", operator="<", column_b="b"),
            RequiredRule(id="req", name="complete", columns=["v"]),
        ],
    )
    rep = validate(df, rs)
    assert [r.status for r in rep.results] == ["fail", "fail", "pass"]
    assert rep.ok is False  # the relation (error severity) failed
    only_warning = validate(df, RuleSet(name="w", rules=[rs.rules[0]]))
    assert only_warning.ok is True  # a failing warning does not fail the report
    e = rep.log_entry
    assert e.operation == "validate" and e.read_only
    assert e.counts == {"rules": 3, "passed": 1, "failed": 2, "not_evaluated": 0, "violations": 3}
    assert json.loads(rep.model_dump_json())["results"][0]["rule_id"] == "r"


def test_not_evaluated_error_rule_keeps_report_from_passing():
    rep = validate(hourly([1]), one(RequiredRule(id="req", name="n", columns=["gone"])))
    assert rep.results[0].status == "not_evaluated" and rep.ok is False
    with pytest.raises(RuleSetError, match="was not evaluated"):
        rep.violation_mask("req")
    with pytest.raises(RuleSetError, match="No result for rule"):
        rep.violation_mask("nope")


def test_violation_mask_is_aligned_and_a_copy():
    df = hourly([1, 200, 3, 300])
    rep = validate(df, one(range_rule(max=100)))
    mask = rep.violation_mask("r")
    assert mask.index.equals(df.index) and mask.tolist() == [False, True, False, True]
    mask[:] = False
    assert rep.violation_mask("r").sum() == 2  # callers cannot change the report


def test_large_violation_lists_are_truncated_but_counted_in_full():
    n = MAX_SAMPLE * 2 + 500
    df = hourly(np.full(n, 500.0))
    rep = validate(df, one(range_rule(max=100)))
    res = rep.results[0]
    assert res.violations == n and len(res.sample) == MAX_SAMPLE and res.truncated
    assert rep.violation_mask("r").sum() == n  # the mask is complete
    assert (
        not validate(hourly(np.full(MAX_SAMPLE, 500.0)), one(range_rule(max=100)))
        .results[0]
        .truncated
    )


def test_validate_never_modifies_the_data():
    df = hourly([1, 200, NaN], a=[1.0, 5.0, 3.0], b=[2.0, 2.0, 2.0])
    snapshot, before = copy.deepcopy(df), fingerprint(df)
    validate(df, _ruleset())
    pd.testing.assert_frame_equal(df, snapshot)
    assert fingerprint(df) == before


def test_duplicate_index_labels_do_not_break_masks():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"v": [1.0, 500.0, 2.0]}, index=idx)
    rep = validate(df, one(range_rule(max=100)))
    assert rep.violation_mask("r").tolist() == [False, True, False]


# ------------------------------------------------------------------ apply_rule_action


def _range_spec(action, **extra):
    return rule_action_spec(range_rule(min=0, max=10), action, **extra)


def test_apply_rule_action_is_a_registered_cleaner_operation():
    assert "apply_rule_action" in available_operations()


def test_flag_set_missing_and_drop_for_a_range_rule():
    df = hourly([1, 50, 3, -4, 5])
    flagged = apply_op(df, _range_spec("flag"))
    assert flagged.df["r__violation"].tolist() == [False, True, False, True, False]
    pd.testing.assert_series_equal(flagged.df["v"], df["v"])
    assert flagged.log_entry.counts["violations"] == 2

    blanked = apply_op(df, _range_spec("set_missing"))
    assert blanked.df["v"].isna().tolist() == [False, True, False, True, False]
    assert blanked.log_entry.counts["cells_set_missing"] == 2

    dropped = apply_op(df, _range_spec("drop"))
    assert dropped.df["v"].tolist() == [1, 3, 5]
    assert dropped.log_entry.counts["rows_dropped"] == 2
    assert dropped.log_entry.rows_before == 5 and dropped.log_entry.rows_after == 3


def test_actions_leave_input_unchanged_and_replay_identically():
    df = hourly([1, 50, 3, -4, 5], other=[1, 2, 3, 4, 5])
    snapshot = copy.deepcopy(df)
    spec = _range_spec("set_missing")
    first, second = apply_op(df, spec), apply_op(df, spec)
    pd.testing.assert_frame_equal(df, snapshot)
    pd.testing.assert_frame_equal(first.df, second.df)
    assert OpSpec.model_validate_json(spec.model_dump_json()) == spec  # replayable as JSON


def test_set_missing_on_integer_columns_upcasts_instead_of_failing():
    df = pd.DataFrame({"n": [1, 500, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))
    res = apply_op(
        df, rule_action_spec(RangeRule(id="r", name="n", column="n", max=10), "set_missing")
    )
    assert res.df["n"].isna().tolist() == [False, True, False]


def test_relation_set_missing_needs_explicit_columns():
    df = hourly([1, 2, 3], name="a", b=[2.0, 2.0, 2.0])
    rule = RelationRule(id="rel", name="rel", column_a="a", operator="<=", column_b="b")
    with pytest.raises(CleaningError, match="say which to blank"):
        apply_op(df, rule_action_spec(rule, "set_missing"))
    res = apply_op(df, rule_action_spec(rule, "set_missing", columns=["a"]))
    assert res.df["a"].isna().tolist() == [False, False, True] and res.df["b"].notna().all()


def test_actions_that_make_no_sense_are_refused_clearly():
    df = hourly([1, 2, 3])
    freq = FrequencyRule(id="f", name="f", expected_interval="30min")
    with pytest.raises(CleaningError, match="delete valid data"):
        apply_op(df, rule_action_spec(freq, "drop"))
    assert apply_op(df, rule_action_spec(freq, "flag")).df["f__violation"].sum() == 2
    with pytest.raises(CleaningError, match="does not apply to a 'required' rule"):
        apply_op(df, rule_action_spec(RequiredRule(id="q", name="q", columns=["v"]), "set_missing"))
    with pytest.raises(CleaningError, match="could not be evaluated"):
        apply_op(df, rule_action_spec(RangeRule(id="r", name="n", column="zz", min=0), "flag"))


def test_required_and_unique_rules_can_drive_drops():
    df = hourly([1, NaN, 3])
    res = apply_op(df, rule_action_spec(RequiredRule(id="q", name="q", columns=["v"]), "drop"))
    assert res.df["v"].tolist() == [1, 3]
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    dup = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
    res = apply_op(dup, rule_action_spec(UniqueTimestampRule(id="u", name="u"), "drop"))
    assert res.df["v"].tolist() == [1, 3]


def test_rule_by_reference_to_a_saved_ruleset(tmp_path):
    path = tmp_path / "rules.json"
    one(range_rule(min=0, max=10)).save_json(path)
    df = hourly([1, 50, 3])
    by_ref = apply_op(
        df,
        OpSpec(
            operation="apply_rule_action",
            params={"action": "drop", "ruleset_path": str(path), "rule_id": "r"},
        ),
    )
    assert by_ref.df["v"].tolist() == [1, 3]
    assert by_ref.df.equals(apply_op(df, _range_spec("drop")).df)
    with pytest.raises(CleaningError, match="No rule with id 'x'"):
        apply_op(
            df,
            OpSpec(
                operation="apply_rule_action",
                params={"action": "drop", "ruleset_path": str(path), "rule_id": "x"},
            ),
        )
    with pytest.raises(CleaningError, match="could not be opened"):
        apply_op(
            df,
            OpSpec(
                operation="apply_rule_action",
                params={
                    "action": "drop",
                    "ruleset_path": str(tmp_path / "no.json"),
                    "rule_id": "r",
                },
            ),
        )


def test_apply_rule_action_param_validation():
    df = hourly([1])
    rule = range_rule(min=0).model_dump()
    for params in (
        {"action": "flag"},
        {"action": "flag", "rule": rule, "rule_id": "r", "ruleset_path": "x.json"},
        {"action": "flag", "ruleset_path": "x.json"},
        {"action": "explode", "rule": rule},
        {"action": "flag", "rule": {**rule, "type": "nonsense"}},
    ):
        with pytest.raises(CleaningError):
            apply_op(df, OpSpec(operation="apply_rule_action", params=params))


# ------------------------------------------------------------------ default_ruleset


def _plant():
    rng = np.random.default_rng(7)
    n = 400
    df = hourly(
        rng.normal(50, 5, n),
        name="power",
        temp=rng.normal(10, 3, n),
        gappy=rng.normal(0, 1, n),
        label=["x"] * n,
    )
    df.loc[df.index[5:20], "gappy"] = NaN
    df.iloc[100, 0] = 500.0  # an obvious spike
    return df


def test_default_ruleset_is_marked_as_suggestions_and_changes_nothing():
    df = _plant()
    snapshot = copy.deepcopy(df)
    suggested = default_ruleset(df, profile(df))
    pd.testing.assert_frame_equal(df, snapshot)
    assert suggested.suggested and suggested.name == "suggested"
    assert all(
        r.suggested and r.severity == "warning" and r.name.startswith("Suggested:")
        for r in suggested.rules
    )


def test_default_ruleset_contents():
    df = _plant()
    rs = default_ruleset(df, profile(df))
    required = rs.get("required_complete")
    assert set(required.columns) == {"power", "temp", "label"}  # 'gappy' has missing values
    ids = {r.id for r in rs.rules}
    assert {"range_power", "range_temp", "range_gappy"} <= ids and "range_label" not in ids
    power = rs.get("range_power")
    assert (
        20 < power.min < 40 and 60 < power.max < 90
    )  # percentile based: the spike did not inflate it
    report = validate(df, rs)
    assert report.result("range_power").violations == 1  # and the spike is caught
    assert report.result("required_complete").status == "pass"
    assert RuleSet.from_json(rs.model_dump_json()) == rs


def test_default_ruleset_uses_zero_floor_for_non_negative_data():
    rng = np.random.default_rng(1)
    df = hourly(rng.gamma(5, 2, 300), name="kwh")
    assert default_ruleset(df, profile(df)).get("range_kwh").min == 0
    df["kwh"] -= 20
    assert default_ruleset(df, profile(df)).get("range_kwh").min < 0


def test_zero_floor_only_for_columns_that_cannot_go_negative():
    rng = np.random.default_rng(3)
    # temperature around 4 degrees: its 0.5th percentile is positive, yet below zero is normal
    df = hourly(rng.normal(4, 1.5, 400), name="outdoor_temp", cooling_power=rng.normal(2, 0.3, 400))
    df["outdoor_temp"] = df["outdoor_temp"].clip(lower=0.2)
    df.iloc[7, 0] = -0.5
    rs = default_ruleset(df, profile(df))
    assert rs.get("range_outdoor_temp").min < 0  # not clamped: the data itself goes below zero
    assert rs.get("range_cooling_power").min >= 0  # a consumption-like name keeps a sensible floor
    assert validate(df, rs).result("range_outdoor_temp").violations == 0


def test_default_ruleset_skips_tiny_and_constant_columns():
    df = hourly(np.arange(10), flat=[1.0] * 10)
    rs = default_ruleset(df, profile(df))
    assert [r.id for r in rs.rules] == ["required_complete"]


# ------------------------------------------------------------------ properties


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.one_of(st.floats(-1e3, 1e3, allow_nan=False), st.just(NaN)), min_size=1, max_size=50
    ),
    st.floats(-1e3, 1e3, allow_nan=False),
    st.floats(0, 500),
    st.booleans(),
)
def test_property_range_violations_match_numpy(values, low, width, inclusive):
    high = low + width
    df = hourly(values)
    rule = range_rule(min=low, max=high, inclusive=inclusive)
    x = np.array(values, dtype=float)
    present = ~np.isnan(x)
    with np.errstate(invalid="ignore"):
        bad = ((x < low) | (x > high)) if inclusive else ((x <= low) | (x >= high))
    expected = bad & present
    mask = evaluate_rule(df, rule).mask.to_numpy()
    assert np.array_equal(mask, expected)
    assert result_of(df, rule).violations == int(expected.sum())
    assert apply_op(df, rule_action_spec(rule, "drop")).df["v"].shape[0] == len(values) - int(
        expected.sum()
    )


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.tuples(st.integers(-5, 5), st.integers(-5, 5)), min_size=1, max_size=40),
    st.sampled_from(["<", "<=", ">", ">=", "==", "!="]),
)
def test_property_relation_matches_python_semantics(pairs, op):
    df = hourly([a for a, _ in pairs], name="a", b=[float(b) for _, b in pairs])
    rule = RelationRule(id="rel", name="rel", column_a="a", operator=op, column_b="b")
    holds = {
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }[op]
    expected = [not holds(a, b) for a, b in pairs]
    assert evaluate_rule(df, rule).mask.tolist() == expected


# ------------------------------------------------------------------ review fixes


def test_rule_files_and_text_with_a_byte_order_mark_load(tmp_path):
    rs = one(range_rule(min=0, max=10))
    path = tmp_path / "rules.json"
    path.write_bytes(b"\xef\xbb\xbf" + rs.model_dump_json().encode("utf-8"))  # as Notepad saves it
    assert RuleSet.load_json(path) == rs
    assert RuleSet.from_json("\ufeff" + rs.model_dump_json()) == rs


def test_set_missing_is_refused_for_rules_whose_violations_are_valid_readings():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 05:00", "2024-01-01 06:00"])
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=idx)
    for rule in (
        FrequencyRule(id="f", name="f", expected_interval="1h"),
        RequiredRule(id="q", name="q", columns=["v"]),
        UniqueTimestampRule(id="u", name="u"),
    ):
        with pytest.raises(CleaningError, match="valid readings"):
            apply_op(df, rule_action_spec(rule, "set_missing", columns=["v"]))
    ok = RangeRule(id="r", name="r", column="v", max=2.5)
    assert (
        apply_op(df, rule_action_spec(ok, "set_missing", columns=["v"])).df["v"].isna().sum() == 1
    )


def test_a_rule_read_from_a_file_is_recorded_in_full_in_the_log(tmp_path):
    path = tmp_path / "rules.json"
    one(range_rule(min=0, max=10)).save_json(path)
    spec = OpSpec(
        operation="apply_rule_action",
        params={"action": "flag", "ruleset_path": str(path), "rule_id": "r"},
    )
    entry = apply_op(hourly([1, 50]), spec).log_entry
    assert (
        entry.params["resolved_rule"]["max"] == 10.0 and entry.params["resolved_rule"]["id"] == "r"
    )
    assert entry.params["resolved_ruleset_path"] == str(path.resolve())
    inline = apply_op(hourly([1, 50]), _range_spec("flag")).log_entry
    assert "resolved_rule" not in inline.params


def test_inline_rule_reference_only_touches_path_references(tmp_path):
    from datanaut.core.validation import inline_rule_reference

    path = tmp_path / "rules.json"
    one(range_rule(min=0, max=10)).save_json(path)
    ref = OpSpec(
        operation="apply_rule_action",
        params={"action": "drop", "ruleset_path": str(path), "rule_id": "r", "columns": ["v"]},
    )
    inlined = inline_rule_reference(ref)
    assert inlined.params["rule"]["max"] == 10.0 and inlined.params["columns"] == ["v"]
    assert "ruleset_path" not in inlined.params and inline_rule_reference(inlined) == inlined
    other = OpSpec(operation="drop_duplicates")
    assert inline_rule_reference(other) is other
    broken = OpSpec(operation="apply_rule_action", params={"action": "nonsense"})
    assert inline_rule_reference(broken) is broken  # apply_op reports the real problem
    gone = OpSpec(
        operation="apply_rule_action",
        params={"action": "drop", "ruleset_path": str(tmp_path / "nope.json"), "rule_id": "r"},
    )
    with pytest.raises(CleaningError, match="could not be opened"):
        inline_rule_reference(gone)
