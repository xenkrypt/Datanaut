"""Validation rules: user-defined expectations about the data, checked with plain pandas.

Rules are pydantic models (a discriminated union on `type`), saved and loaded as JSON, and
evaluated with a fixed set of comparison operators: no eval, no exec. `validate` reports
violations without touching the data; the `apply_rule_action` cleaner operation (registered
here) can then flag, blank or drop the violating rows as a replayable fix.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

from datanaut.core.cleaner import CONSUMPTION_HINTS, OpSpec, Outcome
from datanaut.core.log import PipelineLog, StepLog
from datanaut.core.profiler import DataProfile
from datanaut.errors import CleaningError, RuleSetError

MAX_SAMPLE = 1000
"""At most this many affected row labels are stored per rule."""

VIOLATION_SUFFIX = "__violation"
"""Suffix of the boolean column added by apply_rule_action(action='flag')."""

_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


# ----------------------------------------------------------------------------------- rules


class _RuleBase(BaseModel):
    """Fields shared by every rule."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    severity: Literal["error", "warning"] = "error"
    suggested: bool = False  # True for rules proposed by default_ruleset, not chosen by a user


class RangeRule(_RuleBase):
    """Values of `column` must lie within [min, max] (strictly inside if inclusive is False)."""

    type: Literal["range"] = "range"
    column: str
    min: float | None = None
    max: float | None = None
    inclusive: bool = True

    @model_validator(mode="after")
    def _has_a_bound(self) -> RangeRule:
        if self.min is None and self.max is None:
            raise ValueError("a range rule needs a min, a max or both")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("min cannot be greater than max")
        return self


class RequiredRule(_RuleBase):
    """These columns must have no missing values."""

    type: Literal["required"] = "required"
    columns: list[str] = Field(min_length=1)


class FrequencyRule(_RuleBase):
    """Readings must arrive at least every `max_gap` (default: the expected interval).

    Intervals are pandas time deltas such as '5min', '1h' or '1D'.
    """

    type: Literal["frequency"] = "frequency"
    expected_interval: str
    max_gap: str | None = None

    @model_validator(mode="after")
    def _intervals_parse(self) -> FrequencyRule:
        for label, value in (
            ("expected_interval", self.expected_interval),
            ("max_gap", self.max_gap),
        ):
            if value is None:
                continue
            try:
                delta = pd.Timedelta(value)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{label} '{value}' is not a valid time interval") from exc
            if delta <= pd.Timedelta(0):
                raise ValueError(f"{label} must be positive")
        return self


class RelationRule(_RuleBase):
    """`column_a <operator> column_b` must hold on every row where both are present."""

    type: Literal["relation"] = "relation"
    column_a: str
    operator: Literal["<", "<=", ">", ">=", "==", "!="]
    column_b: str


class UniqueTimestampRule(_RuleBase):
    """Each timestamp may appear only once."""

    type: Literal["unique_timestamp"] = "unique_timestamp"


Rule = Annotated[
    RangeRule | RequiredRule | FrequencyRule | RelationRule | UniqueTimestampRule,
    Field(discriminator="type"),
]


class RuleSet(BaseModel):
    """A named, versioned list of rules that can be saved and shared as JSON."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = "1"
    description: str = ""
    suggested: bool = False
    rules: list[Rule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> RuleSet:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"rule id '{rule.id}' is used more than once")
            seen.add(rule.id)
        return self

    def get(self, rule_id: str) -> AnyRule:
        """Find a rule by id."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        known = ", ".join(r.id for r in self.rules) or "none"
        raise RuleSetError(f"No rule with id '{rule_id}' in '{self.name}'. Rules: {known}.")

    def save_json(self, path: str | Path) -> None:
        """Write the rule set as JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, text: str, source: str = "the rule file") -> RuleSet:
        """Parse JSON text, raising RuleSetError with a readable message if it is malformed."""
        try:
            data = json.loads(text.lstrip("\ufeff"))
        except json.JSONDecodeError as exc:
            raise RuleSetError(
                f"{source} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}."
            ) from exc
        if not isinstance(data, dict):
            raise RuleSetError(f"{source} must contain a JSON object with 'name' and 'rules'.")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            problems = []
            for err in exc.errors()[:5]:
                where = ".".join(str(p) for p in err["loc"]) or "rule set"
                problems.append(f"{where}: {err['msg']}")
            more = f" (and {len(exc.errors()) - 5} more)" if len(exc.errors()) > 5 else ""
            raise RuleSetError(
                f"{source} has problems: " + "; ".join(problems) + more + "."
            ) from exc

    @classmethod
    def load_json(cls, path: str | Path) -> RuleSet:
        """Read a rule set from a JSON file."""
        file = Path(path)
        try:
            text = file.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise RuleSetError(f"The rule file '{file.name}' could not be opened.") from exc
        return cls.from_json(text, source=f"The rule file '{file.name}'")


AnyRule = RangeRule | RequiredRule | FrequencyRule | RelationRule | UniqueTimestampRule


# ------------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class RuleEvaluation:
    """The outcome of one rule on one frame. `mask` is None if the rule could not be run."""

    mask: pd.Series | None
    rows_checked: int
    message: str


def _no_mask(df: pd.DataFrame, message: str) -> RuleEvaluation:
    return RuleEvaluation(mask=None, rows_checked=0, message=message)


def _missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c not in df.columns]


def _series_mask(df: pd.DataFrame, values: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=bool), index=df.index)


def evaluate_rule(df: pd.DataFrame, rule: AnyRule) -> RuleEvaluation:
    """Check one rule. The returned mask is True on violating rows."""
    if isinstance(rule, RangeRule):
        return _eval_range(df, rule)
    if isinstance(rule, RequiredRule):
        return _eval_required(df, rule)
    if isinstance(rule, FrequencyRule):
        return _eval_frequency(df, rule)
    if isinstance(rule, RelationRule):
        return _eval_relation(df, rule)
    return _eval_unique(df, rule)


def _eval_range(df: pd.DataFrame, rule: RangeRule) -> RuleEvaluation:
    if rule.column not in df.columns:
        return _no_mask(df, f"Column '{rule.column}' was not found.")
    s = df[rule.column]
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return _no_mask(df, f"Column '{rule.column}' is not numeric.")
    values = s.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        low = np.zeros(len(values), bool)
        high = np.zeros(len(values), bool)
        if rule.min is not None:
            low = values < rule.min if rule.inclusive else values <= rule.min
        if rule.max is not None:
            high = values > rule.max if rule.inclusive else values >= rule.max
    present = ~np.isnan(values)
    mask = (low | high) & present
    n_low, n_high = int((low & present).sum()), int((high & present).sum())
    lo_text = f"{rule.min:g}" if rule.min is not None else "-inf"
    hi_text = f"{rule.max:g}" if rule.max is not None else "inf"
    left, right = ("[", "]") if rule.inclusive else ("(", ")")
    return RuleEvaluation(
        _series_mask(df, mask),
        int(present.sum()),
        f"{n_low} below and {n_high} above {left}{lo_text}, {hi_text}{right} "
        f"({int((~present).sum())} missing value(s) not checked).",
    )


def _eval_required(df: pd.DataFrame, rule: RequiredRule) -> RuleEvaluation:
    absent = _missing_columns(df, rule.columns)
    if absent:
        return _no_mask(df, f"Column(s) not found: {', '.join(absent)}.")
    missing = df[rule.columns].isna()
    per_column = {c: int(missing[c].sum()) for c in rule.columns if missing[c].any()}
    detail = ", ".join(f"{c}: {n}" for c, n in per_column.items()) or "none missing"
    return RuleEvaluation(
        _series_mask(df, missing.any(axis=1).to_numpy()),
        len(df),
        f"Missing values by column: {detail}.",
    )


def _eval_frequency(df: pd.DataFrame, rule: FrequencyRule) -> RuleEvaluation:
    if not isinstance(df.index, pd.DatetimeIndex):
        return _no_mask(df, "The data has no time index.")
    expected = pd.Timedelta(rule.expected_interval)
    limit = pd.Timedelta(rule.max_gap) if rule.max_gap else expected
    n = len(df)
    if n < 2:
        return RuleEvaluation(
            _series_mask(df, np.zeros(n)), n, "Fewer than two rows: no gaps to check."
        )
    order = np.argsort(df.index.asi8, kind="stable")
    stamps = df.index[order]
    gaps = stamps[1:] - stamps[:-1]
    too_long = np.asarray(gaps > limit)
    mask = np.zeros(n, bool)
    mask[order[1:]] = too_long  # the reading that ends each long gap
    steps = np.asarray(gaps[too_long] / expected, dtype=float)
    missing = int(np.maximum(np.rint(steps) - 1, 0).sum())
    return RuleEvaluation(
        _series_mask(df, mask),
        n - 1,
        f"{int(too_long.sum())} gap(s) longer than {rule.max_gap or rule.expected_interval}; "
        f"about {missing} expected reading(s) every {rule.expected_interval} are missing.",
    )


def _eval_relation(df: pd.DataFrame, rule: RelationRule) -> RuleEvaluation:
    absent = _missing_columns(df, [rule.column_a, rule.column_b])
    if absent:
        return _no_mask(df, f"Column(s) not found: {', '.join(absent)}.")
    a, b = df[rule.column_a], df[rule.column_b]
    both = (a.notna() & b.notna()).to_numpy()
    try:
        holds = np.asarray(_OPERATORS[rule.operator](a, b), dtype=bool)
    except TypeError:
        return _no_mask(df, f"'{rule.column_a}' and '{rule.column_b}' cannot be compared.")
    return RuleEvaluation(
        _series_mask(df, both & ~holds),
        int(both.sum()),
        f"Rows where '{rule.column_a} {rule.operator} {rule.column_b}' is false; "
        f"{len(df) - int(both.sum())} row(s) with a missing value not checked.",
    )


def _eval_unique(df: pd.DataFrame, rule: UniqueTimestampRule) -> RuleEvaluation:
    if not isinstance(df.index, pd.DatetimeIndex):
        return _no_mask(df, "The data has no time index.")
    repeated = df.index.duplicated(keep="first")
    return RuleEvaluation(
        _series_mask(df, repeated),
        len(df),
        f"{int(repeated.sum())} row(s) repeat an earlier timestamp.",
    )


# --------------------------------------------------------------------------------- report


class RuleResult(BaseModel):
    """What one rule found."""

    rule_id: str
    name: str
    severity: Literal["error", "warning"]
    status: Literal["pass", "fail", "not_evaluated"]
    violations: int = 0
    rows_checked: int = 0
    sample: list[str] = Field(default_factory=list)
    truncated: bool = False
    message: str = ""


class ValidationReport(BaseModel):
    """Per-rule results for one dataset. Masks are kept for use by the cleaner."""

    ruleset_name: str
    ruleset_version: str
    rows: int
    results: list[RuleResult]
    log_entry: StepLog
    _masks: dict[str, pd.Series] = PrivateAttr(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True if no error-severity rule failed or was left unevaluated."""
        return not any(r.severity == "error" and r.status != "pass" for r in self.results)

    def result(self, rule_id: str) -> RuleResult:
        """Look up one rule's result."""
        for r in self.results:
            if r.rule_id == rule_id:
                return r
        raise RuleSetError(f"No result for rule '{rule_id}' in this report.")

    def violation_mask(self, rule_id: str) -> pd.Series:
        """Boolean Series, aligned with the validated frame, True where the rule was violated."""
        res = self.result(rule_id)
        if rule_id not in self._masks:
            raise RuleSetError(f"Rule '{rule_id}' was not evaluated: {res.message}")
        return self._masks[rule_id].copy()


def validate(df: pd.DataFrame, ruleset: RuleSet) -> ValidationReport:
    """Run every rule on `df` (which is never modified) and report the violations."""
    results: list[RuleResult] = []
    masks: dict[str, pd.Series] = {}
    for rule in ruleset.rules:
        ev = evaluate_rule(df, rule)
        if ev.mask is None:
            results.append(
                RuleResult(
                    rule_id=rule.id,
                    name=rule.name,
                    severity=rule.severity,
                    status="not_evaluated",
                    message=ev.message,
                )
            )
            continue
        masks[rule.id] = ev.mask
        positions = np.flatnonzero(ev.mask.to_numpy())
        results.append(
            RuleResult(
                rule_id=rule.id,
                name=rule.name,
                severity=rule.severity,
                status="fail" if len(positions) else "pass",
                violations=int(len(positions)),
                rows_checked=ev.rows_checked,
                sample=[str(v) for v in df.index[positions[:MAX_SAMPLE]]],
                truncated=len(positions) > MAX_SAMPLE,
                message=ev.message,
            )
        )
    counts = {
        "rules": len(results),
        "passed": sum(r.status == "pass" for r in results),
        "failed": sum(r.status == "fail" for r in results),
        "not_evaluated": sum(r.status == "not_evaluated" for r in results),
        "violations": sum(r.violations for r in results),
    }
    entry = PipelineLog().append(
        "validate",
        f"Validated against '{ruleset.name}' v{ruleset.version}: {counts['passed']} of "
        f"{counts['rules']} rule(s) passed, {counts['failed']} failed, "
        f"{counts['not_evaluated']} could not be evaluated. No data was changed.",
        params={
            "ruleset": ruleset.name,
            "version": ruleset.version,
            "rules": [r.id for r in ruleset.rules],
        },
        counts=counts,
        rows_before=len(df),
        rows_after=len(df),
        read_only=True,
    )
    report = ValidationReport(
        ruleset_name=ruleset.name,
        ruleset_version=ruleset.version,
        rows=len(df),
        results=results,
        log_entry=entry,
    )
    report._masks = masks
    return report


# ----------------------------------------------------------------------- suggested rules


def _round(value: float) -> float:
    return float(f"{value:.4g}")


def default_ruleset(df: pd.DataFrame, prof: DataProfile) -> RuleSet:
    """Suggest a starting rule set. Every rule is marked `suggested`; nothing is applied.

    Columns with no missing values become one required-columns rule. Numeric columns get a
    plausible range: the 0.5th to 99.5th percentile widened by half its span, and not below
    zero for columns that look like consumption or never go negative.
    """
    rules: list[RangeRule | RequiredRule] = []
    complete = [c.name for c in prof.columns if c.missing_count == 0 and c.name in df.columns]
    if complete:
        rules.append(
            RequiredRule(
                id="required_complete",
                name="Suggested: columns that are complete today must stay complete",
                severity="warning",
                suggested=True,
                columns=complete,
            )
        )
    for col in prof.columns:
        if col.inferred_type != "numeric" or col.name not in df.columns:
            continue
        values = df[col.name].dropna().to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 20 or col.unique_count < 3:
            continue
        low, high = np.percentile(values, [0.5, 99.5])
        margin = 0.5 * (high - low)
        non_negative = float(values.min()) >= 0 or any(
            hint in col.name.lower() for hint in CONSUMPTION_HINTS
        )
        lower = max(low - margin, 0.0) if non_negative and low >= 0 else low - margin
        upper = high + margin
        rules.append(
            RangeRule(
                id=f"range_{col.name}",
                name=f"Suggested: {col.name} stays within its usual range",
                severity="warning",
                suggested=True,
                column=col.name,
                min=_round(lower),
                max=_round(upper),
            )
        )
    return RuleSet(
        name="suggested",
        version="1",
        description="Suggested starting rules derived from the data. Review them before use.",
        suggested=True,
        rules=list(rules),
    )


# ------------------------------------------------------------------------- apply_rule_action


class ApplyRuleActionParams(BaseModel):
    """Handle the rows that violate a rule.

    Give the rule inline (`rule`) or by reference (`ruleset_path` + `rule_id`). Actions:
    flag adds a boolean '<rule id>__violation' column; set_missing blanks the offending cells
    (`columns` names them; range rules default to their column); drop removes the rows.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["flag", "set_missing", "drop"]
    rule: Rule | None = None
    rule_id: str | None = None
    ruleset_path: str | None = None
    columns: list[str] | None = None

    @model_validator(mode="after")
    def _one_source(self) -> ApplyRuleActionParams:
        by_reference = self.ruleset_path is not None or self.rule_id is not None
        if self.rule is not None and by_reference:
            raise ValueError("give either an inline rule or ruleset_path + rule_id, not both")
        if self.rule is None and (self.ruleset_path is None or self.rule_id is None):
            raise ValueError("give an inline rule, or both ruleset_path and rule_id")
        return self


def rule_action_spec(
    rule: AnyRule, action: Literal["flag", "set_missing", "drop"], columns: list[str] | None = None
) -> OpSpec:
    """Build an OpSpec that applies `action` to the violations of an inline rule."""
    params: dict[str, Any] = {"action": action, "rule": rule.model_dump()}
    if columns:
        params["columns"] = columns
    return OpSpec(operation="apply_rule_action", params=params)


def _resolve_rule(p: ApplyRuleActionParams) -> AnyRule:
    if p.rule is not None:
        return p.rule
    try:
        return RuleSet.load_json(str(p.ruleset_path)).get(str(p.rule_id))
    except RuleSetError as exc:
        raise CleaningError(str(exc)) from exc


def _target_columns(rule: AnyRule, p: ApplyRuleActionParams, df: pd.DataFrame) -> list[str]:
    if not isinstance(rule, RangeRule | RelationRule):
        raise CleaningError(
            f"set_missing does not apply to a '{rule.type}' rule: its violating rows are valid "
            "readings. Use flag or drop."
        )
    if p.columns:
        targets = list(p.columns)
    elif isinstance(rule, RangeRule):
        targets = [rule.column]
    else:
        raise CleaningError(
            f"Rule '{rule.id}' compares two columns: say which to blank with columns=["
            f"'{rule.column_a}'] or ['{rule.column_b}']."
        )
    absent = _missing_columns(df, targets)
    if absent:
        raise CleaningError(f"Column '{absent[0]}' was not found in the data.")
    return targets


def _apply_rule_action(df: pd.DataFrame, p: ApplyRuleActionParams) -> Outcome:
    rule = _resolve_rule(p)
    ev = evaluate_rule(df, rule)
    if ev.mask is None:
        raise CleaningError(f"Rule '{rule.id}' could not be evaluated: {ev.message}")
    mask = ev.mask.to_numpy()
    n = int(mask.sum())
    out = df.copy()
    counts: dict[str, float] = {"violations": n, "rows_checked": ev.rows_checked}
    label = f"Rule '{rule.name}' ({rule.id}): {n} violation(s)"
    if p.action == "flag":
        flag = rule.id + VIOLATION_SUFFIX
        counts["replaced_existing_flag"] = float(flag in out.columns)
        out[flag] = pd.Series(mask, index=out.index, dtype=bool)
        summary = f"{label}; flagged in '{flag}', data unchanged."
    elif p.action == "drop":
        if isinstance(rule, FrequencyRule):
            raise CleaningError(
                "drop does not apply to a frequency rule: it would delete valid data."
            )
        out = out.loc[~mask]
        counts["rows_dropped"] = n
        summary = f"{label}; dropped {n} row(s)."
    else:
        targets = _target_columns(rule, p, df)
        cells = 0
        for col in targets:
            if not pd.api.types.is_float_dtype(out[col]) and pd.api.types.is_integer_dtype(
                out[col]
            ):
                out[col] = out[col].astype(float)
            was_present = out[col].notna().to_numpy() & mask
            cells += int(was_present.sum())
            out.loc[mask, col] = np.nan
        counts["cells_set_missing"] = cells
        summary = f"{label}; set {cells} cell(s) in {', '.join(targets)} to missing."
    extra: dict[str, Any] = {}
    if p.rule is None:  # looked up in a file: keep what that file said at this moment
        extra = {
            "resolved_rule": rule.model_dump(),
            "resolved_ruleset_path": str(Path(str(p.ruleset_path)).resolve()),
        }
    return Outcome(df=out, summary=summary, counts=counts, params_extra=extra)


def inline_rule_reference(spec: OpSpec) -> OpSpec:
    """Replace a rule given as ruleset_path + rule_id with the rule itself.

    Recipes call this so that replaying them never depends on a rules file that may have moved
    or changed. Any other spec is returned unchanged; invalid params are left for apply_op to
    report.
    """
    if spec.operation != "apply_rule_action":
        return spec
    try:
        params = ApplyRuleActionParams.model_validate(spec.params)
    except ValidationError:
        return spec
    if params.rule is not None:
        return spec
    rule = _resolve_rule(params)
    inline: dict[str, Any] = {"action": params.action, "rule": rule.model_dump()}
    if params.columns:
        inline["columns"] = params.columns
    return OpSpec(operation="apply_rule_action", params=inline)


# The operation table in cleaner.py reads this: (name, params model, handler).
RULE_ACTION = ("apply_rule_action", ApplyRuleActionParams, _apply_rule_action)
