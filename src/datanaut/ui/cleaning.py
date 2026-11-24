"""Cleaning logic for the Clean screen: proposals, selection, apply, undo, preview, downloads.

No streamlit import. Every function only calls the core; nothing here changes data by itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datanaut.core.characterize import characterize
from datanaut.core.cleaner import OpSpec, PreviewResult, Proposal, preview, propose_fixes
from datanaut.core.export import neutralize_formulas
from datanaut.core.profiler import profile
from datanaut.core.versioning import VersionDiff
from datanaut.errors import DatanautError, VersioningError
from datanaut.ui.state import WizardState, cache_key

PREVIEW_COLUMN_CAP = 6


def get_proposals(state: WizardState) -> list[Proposal]:
    """The suggested fixes for the working version, in the order they should be applied."""
    assert state.store is not None
    store = state.store

    def compute() -> list[Proposal]:
        df = store.current()
        char = None
        if state.value_columns:
            try:
                char = characterize(df, state.value_columns[0])
            except DatanautError:
                char = None
        return propose_fixes(df, profile(df), char)

    return state.get_or_compute(cache_key(state.fingerprint, "proposals"), compute)


def default_selection(proposals: list[Proposal]) -> dict[str, bool]:
    return {p.id: p.default_selected for p in proposals}


def chosen_specs(
    proposals: list[Proposal],
    selected: dict[str, bool],
    alternative: dict[str, int | None] | None = None,
) -> list[OpSpec]:
    """The operations to apply: selected proposals only, in proposal order.

    `alternative[id]` is an index into the proposal's alternatives (None: the suggested one).
    """
    alternative = alternative or {}
    specs: list[OpSpec] = []
    for p in proposals:
        if not selected.get(p.id, p.default_selected):
            continue
        idx = alternative.get(p.id)
        if idx is not None and 0 <= idx < len(p.alternatives):
            specs.append(p.alternatives[idx].spec.model_copy(deep=True))
        else:
            specs.append(p.spec.model_copy(deep=True))
    return specs


@dataclass(frozen=True)
class ApplyOutcome:
    applied: int
    error: str | None = None


def apply_specs(state: WizardState, specs: list[OpSpec]) -> ApplyOutcome:
    """Apply operations in order through the VersionStore. Stops at the first failure; the
    versions made before it are kept (each one is a complete, valid version)."""
    assert state.store is not None
    applied = 0
    for spec in specs:
        try:
            state.store.apply(spec)
        except DatanautError as exc:
            where = f"'{spec.operation}'"
            msg = f"{where} could not be applied: {exc.message}"
            if applied:
                msg += f" The {applied} earlier fix(es) were kept."
            state.add_message("error", msg)
            return ApplyOutcome(applied, msg)
        applied += 1
    if applied:
        state.add_message(
            "success", f"Applied {applied} fix(es). The working version is {state.store.head}."
        )
    else:
        state.add_message("info", "Nothing was selected, so nothing was changed.")
    return ApplyOutcome(applied)


def apply_one(state: WizardState, spec: OpSpec) -> ApplyOutcome:
    return apply_specs(state, [spec])


def undo(state: WizardState) -> bool:
    """Go back to the version the working version was made from."""
    assert state.store is not None
    parent = state.store.version(state.store.head).parent_id
    if parent is None:
        state.add_message("info", "There is nothing to undo: you are on the raw data (V1).")
        return False
    state.store.restore(parent)
    state.add_message("success", f"Restored {parent}. No version was deleted.")
    return True


def restore(state: WizardState, version_id: str) -> bool:
    assert state.store is not None
    if version_id == state.store.head:
        return False
    try:
        state.store.restore(version_id)
    except VersioningError as exc:
        state.add_message("error", exc.message)
        return False
    state.add_message("success", f"Restored {version_id}. Later changes will branch from it.")
    return True


# --- preview ------------------------------------------------------------------------------


def preview_spec(state: WizardState, spec: OpSpec) -> PreviewResult:
    """What an operation would do to the working version (cached; nothing is changed)."""
    assert state.store is not None
    key = cache_key(state.fingerprint, "preview", spec=spec.model_dump_json())
    return state.get_or_compute(key, lambda: preview(state.store.current(), spec))


def preview_table(pv: PreviewResult) -> pd.DataFrame:
    """Before and after summary statistics, one row per column."""
    rows = []
    for name, b in list(pv.before.items())[:PREVIEW_COLUMN_CAP]:
        a = pv.after.get(name)
        rows.append(
            {
                "Column": name,
                "Missing before": b.missing,
                "Missing after": a.missing if a else None,
                "Mean before": b.mean,
                "Mean after": a.mean if a else None,
                "Min before": b.min,
                "Min after": a.min if a else None,
                "Max before": b.max,
                "Max after": a.max if a else None,
            }
        )
    return pd.DataFrame(rows)


def preview_column(state: WizardState, spec: OpSpec, pv: PreviewResult) -> str | None:
    """The column to draw for a preview: the one the operation names, else the first value."""
    column = spec.params.get("column")
    if isinstance(column, str) and column in pv.df.columns:
        return column
    for name in state.value_columns:
        if name in pv.df.columns:
            return name
    return None


# --- compare and downloads ----------------------------------------------------------------


def compare(state: WizardState, a: str, b: str) -> VersionDiff:
    assert state.store is not None
    return state.store.compare(a, b)


def diff_summary(diff: VersionDiff) -> list[str]:
    """Plain sentences describing a VersionDiff."""
    if diff.identical:
        return [f"{diff.a} and {diff.b} are identical."]
    lines = []
    if diff.rows_added or diff.rows_removed:
        lines.append(f"{diff.rows_added:,} row(s) added and {diff.rows_removed:,} removed.")
    if diff.columns_added:
        lines.append("Columns added: " + ", ".join(diff.columns_added))
    if diff.columns_removed:
        lines.append("Columns removed: " + ", ".join(diff.columns_removed))
    if diff.cells_changed_total:
        per = ", ".join(f"{c}: {n:,}" for c, n in diff.cells_changed.items())
        lines.append(f"{diff.cells_changed_total:,} cell(s) changed ({per}).")
    return lines or ["The versions differ only in data types."]


def diff_examples(diff: VersionDiff) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Row": c.row, "Column": c.column, "Before": c.before, "After": c.after}
            for c in diff.examples
        ],
        columns=["Row", "Column", "Before", "After"],
    )


def cleaned_csv(state: WizardState, version_id: str | None = None) -> bytes:
    """A version (default: the working one) as formula-safe CSV bytes; nothing is written."""
    assert state.store is not None
    safe, _ = neutralize_formulas(state.store.get(version_id or state.store.head))
    return safe.to_csv(lineterminator="\n").encode("utf-8")


def log_json(state: WizardState) -> str:
    return state.log.to_json()


def recipe_json(state: WizardState, version_id: str | None = None) -> str:
    assert state.store is not None and state.indexed is not None and state.loaded is not None
    source = {
        "filename": state.loaded.source_info.filename,
        "sheet": state.loaded.source_info.sheet,
        "time_columns": state.indexed.choices.time_columns,
        "dayfirst": state.indexed.choices.dayfirst,
    }
    return state.store.to_recipe(version_id, source=source).to_json()
