"""Validation-rule logic for the Clean screen: build, accept, load, save, run and act on rules.

No streamlit import.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError

from datanaut.core.profiler import profile
from datanaut.core.validation import (
    AnyRule,
    FrequencyRule,
    RangeRule,
    RelationRule,
    RequiredRule,
    RuleSet,
    UniqueTimestampRule,
    ValidationReport,
    default_ruleset,
    rule_action_spec,
    validate,
)
from datanaut.errors import RuleSetError
from datanaut.ui import cleaning
from datanaut.ui.state import WizardState, cache_key

RULE_KINDS = ("range", "required", "frequency", "relation", "unique_timestamp")
ACTIONS = ("flag", "set_missing", "drop")
ROW_CAP = 50
_NAME = "my rules"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "rule"


def unique_id(state: WizardState, base: str) -> str:
    taken = {r.id for r in state.rules} | {r.id for r in state.suggestions or []}
    candidate, n = base, 2
    while candidate in taken:
        candidate, n = f"{base}_{n}", n + 1
    return candidate


def build_rule(
    state: WizardState,
    kind: str,
    *,
    severity: Literal["error", "warning"] = "error",
    **fields: Any,
) -> AnyRule:
    """Make a rule from form values. Raises RuleSetError with a readable message."""
    classes = {
        "range": RangeRule,
        "required": RequiredRule,
        "frequency": FrequencyRule,
        "relation": RelationRule,
        "unique_timestamp": UniqueTimestampRule,
    }
    if kind not in classes:
        raise RuleSetError(f"Unknown rule type '{kind}'.")
    name = fields.pop("name", None) or _default_name(kind, fields)
    rule_id = unique_id(state, _slug(f"{kind}_{name}")[:40])
    try:
        return classes[kind](id=rule_id, name=name, severity=severity, **fields)
    except ValidationError as exc:
        err = exc.errors()[0]
        raise RuleSetError(
            f"This rule is not valid: {err['msg'].removeprefix('Value error, ')}."
        ) from exc


def _default_name(kind: str, f: dict[str, Any]) -> str:
    if kind == "range":
        return f"{f.get('column', '?')} between {f.get('min')} and {f.get('max')}"
    if kind == "required":
        return "No missing values in " + ", ".join(f.get("columns") or ["?"])
    if kind == "frequency":
        return f"A reading at least every {f.get('max_gap') or f.get('expected_interval')}"
    if kind == "relation":
        return f"{f.get('column_a')} {f.get('operator')} {f.get('column_b')}"
    return "Each timestamp appears once"


def add_rule(state: WizardState, rule: AnyRule) -> None:
    state.rules.append(rule)
    state.report = None


def delete_rule(state: WizardState, rule_id: str) -> None:
    state.rules = [r for r in state.rules if r.id != rule_id]
    state.report = None


def get_suggestions(state: WizardState) -> list[AnyRule]:
    """Suggested rules from the data (generated once; accepted or deleted ones disappear)."""
    assert state.store is not None
    if state.suggestions is None:
        store = state.store

        def compute() -> list[AnyRule]:
            df = store.current()
            return list(default_ruleset(df, profile(df)).rules)

        found = state.get_or_compute(cache_key(state.fingerprint, "suggested_rules"), compute)
        state.suggestions = [r for r in found if r.id not in {x.id for x in state.rules}]
    return state.suggestions


def accept_suggestion(state: WizardState, rule_id: str) -> None:
    keep = []
    for r in state.suggestions or []:
        if r.id == rule_id:
            add_rule(state, r.model_copy(update={"suggested": False}))
        else:
            keep.append(r)
    state.suggestions = keep


def delete_suggestion(state: WizardState, rule_id: str) -> None:
    state.suggestions = [r for r in state.suggestions or [] if r.id != rule_id]


def ruleset(state: WizardState) -> RuleSet:
    return RuleSet(name=_NAME, rules=list(state.rules))


def rules_json(state: WizardState) -> str:
    return ruleset(state).model_dump_json(indent=2)


def load_rules(state: WizardState, text: str, filename: str = "the rule file") -> int:
    """Replace the rules with the contents of an uploaded JSON file. Raises RuleSetError."""
    loaded = RuleSet.from_json(text, source=filename)
    state.rules = list(loaded.rules)
    state.suggestions = [
        r for r in state.suggestions or [] if r.id not in {x.id for x in state.rules}
    ]
    state.report = None
    return len(loaded.rules)


def rules_table(rules: list[AnyRule]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"ID": r.id, "Rule": r.name, "Type": r.type, "Severity": r.severity} for r in rules],
        columns=["ID", "Rule", "Type", "Severity"],
    )


# --- running ------------------------------------------------------------------------------


def run_validation(state: WizardState) -> ValidationReport:
    """Check the working version against the rules and log the (read-only) check."""
    assert state.store is not None
    if not state.rules:
        raise RuleSetError("Add or accept at least one rule first.")
    report = validate(state.store.current(), ruleset(state))
    state.log.extend([report.log_entry])
    state.report, state.report_fingerprint = report, state.fingerprint
    return report


def report_is_current(state: WizardState) -> bool:
    return state.report is not None and state.report_fingerprint == state.fingerprint


def result_table(report: ValidationReport) -> pd.DataFrame:
    labels = {"pass": "✓ pass", "fail": "✗ fail", "not_evaluated": "– not checked"}
    return pd.DataFrame(
        [
            {
                "ID": r.rule_id,
                "Rule": r.name,
                "Severity": r.severity,
                "Status": labels[r.status],
                "Violations": r.violations,
                "Note": r.message,
            }
            for r in report.results
        ],
        columns=["ID", "Rule", "Severity", "Status", "Violations", "Note"],
    )


def violating_rows(
    state: WizardState, rule_id: str, cap: int = ROW_CAP
) -> tuple[pd.DataFrame, int]:
    """The rows that break a rule (at most `cap`) and how many there are in total."""
    assert state.store is not None and state.report is not None
    mask = state.report.violation_mask(rule_id)
    df = state.store.get(state.store.head)
    total = int(mask.sum())
    return df.loc[mask.to_numpy()].head(cap), total


def actions_for(rule: AnyRule) -> list[str]:
    """Which actions make sense for a rule (the core refuses the others)."""
    if isinstance(rule, FrequencyRule):
        return ["flag"]
    if isinstance(rule, RangeRule | RelationRule):
        return list(ACTIONS)
    return ["flag", "drop"]


def apply_rule_action(
    state: WizardState,
    rule_id: str,
    action: Literal["flag", "set_missing", "drop"],
    columns: list[str] | None = None,
) -> cleaning.ApplyOutcome:
    """Apply an action to a rule's violations through the same version flow as the fixes."""
    rule = next((r for r in state.rules if r.id == rule_id), None)
    if rule is None:
        raise RuleSetError(f"There is no rule '{rule_id}'.")
    return cleaning.apply_one(state, rule_action_spec(rule, action, columns))
