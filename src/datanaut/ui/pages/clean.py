"""Clean screen: review suggested fixes, apply them, validate against rules, download results."""

from __future__ import annotations

import streamlit as st

from datanaut.errors import DatanautError
from datanaut.ui import actions, charts, cleaning, rules
from datanaut.ui.state import WizardState, cache_key


def render(state: WizardState) -> None:
    st.header("Clean")
    st.write(
        "Review the suggested fixes, preview each one, then apply the ones you want. "
        "Every change makes a new version you can undo."
    )
    if state.store is None:
        st.info("Map your columns first.")
        return
    fixes, validation, versions = st.tabs(["Fixes", "Validation", "Versions and downloads"])
    with fixes:
        _fixes(state)
    with validation:
        _validation(state)
    with versions:
        _versions(state)
    st.button("Continue to Analyze", type="primary", on_click=_continue, args=(state,))


def _continue(state: WizardState) -> None:
    state.mark_done("Clean")
    state.go("Analyze")


# --- fixes --------------------------------------------------------------------------------


def _fixes(state: WizardState) -> None:
    store = state.store
    assert store is not None
    proposals = actions.guard(state, "Clean", cleaning.get_proposals, state)
    if proposals is None:
        return
    st.caption(f"Working on {store.head}. Fixes are listed in the order they will be applied.")
    if not proposals:
        st.success("No fixes are suggested for this version.")
    selected: dict[str, bool] = {}
    alternative: dict[str, int | None] = {}
    for p in proposals:
        with st.container(border=True):
            selected[p.id] = st.checkbox(
                p.description, value=p.default_selected, key=f"sel_{store.head}_{p.id}"
            )
            labels = ["Suggested method"] + [a.label for a in p.alternatives]
            if p.alternatives:
                choice = st.selectbox(
                    "Method", labels, key=f"alt_{store.head}_{p.id}", label_visibility="collapsed"
                )
                alternative[p.id] = labels.index(choice) - 1 if choice != labels[0] else None
            else:
                alternative[p.id] = None
            if st.button("Preview", key=f"prev_{store.head}_{p.id}", help="Show the effect only"):
                st.session_state["clean_preview"] = p.id
            if st.session_state.get("clean_preview") == p.id:
                _preview(state, cleaning.chosen_specs([p], {p.id: True}, alternative)[0])

    specs = cleaning.chosen_specs(proposals, selected, alternative)
    left, right = st.columns(2)
    if left.button(
        f"Apply {len(specs)} selected fix(es)", type="primary", disabled=not specs, key="apply_sel"
    ):
        cleaning.apply_specs(state, specs)
        st.session_state.pop("clean_preview", None)
        st.rerun()
    parent = store.version(store.head).parent_id
    if right.button(
        "Undo the last change",
        disabled=parent is None,
        help="Go back to the previous version (nothing is deleted)",
        key="undo",
    ):
        cleaning.undo(state)
        st.rerun()


def _preview(state: WizardState, spec) -> None:
    pv = actions.guard(state, "Clean", cleaning.preview_spec, state, spec)
    if pv is None:
        return
    st.caption(f"Before: {pv.rows_before:,} rows. After: {pv.rows_after:,} rows. Not applied yet.")
    st.dataframe(cleaning.preview_table(pv), hide_index=True, use_container_width=True)
    column = cleaning.preview_column(state, spec, pv)
    if column is not None and column in state.store.current().columns:
        before = state.store.current()[column]
        st.plotly_chart(
            charts.before_after(before, pv.df[column], column), use_container_width=True
        )


# --- validation ---------------------------------------------------------------------------


def _validation(state: WizardState) -> None:
    st.write("Rules describe what valid data looks like. Suggested rules are only suggestions.")
    suggestions = actions.guard(state, "Validation", rules.get_suggestions, state) or []
    if suggestions:
        st.subheader("Suggested rules")
        for r in list(suggestions):
            a, b, c = st.columns([6, 1, 1])
            a.write(f"{r.name} ({r.severity})")
            if b.button("Accept", key=f"acc_{r.id}"):
                rules.accept_suggestion(state, r.id)
                st.rerun()
            if c.button("Delete", key=f"delsug_{r.id}"):
                rules.delete_suggestion(state, r.id)
                st.rerun()

    st.subheader("Your rules")
    if state.rules:
        st.dataframe(rules.rules_table(state.rules), hide_index=True, use_container_width=True)
        gone = st.selectbox("Remove a rule", [r.id for r in state.rules], key="rule_remove")
        if st.button("Remove this rule", key="rule_remove_btn"):
            rules.delete_rule(state, gone)
            st.rerun()
    else:
        st.caption("No rules yet. Accept a suggestion or add one below.")
    _editor(state)
    _save_load(state)

    if st.button("Run validation", type="primary", disabled=not state.rules, key="run_val"):
        actions.guard(state, "Validation", rules.run_validation, state)
    if state.report is not None:
        _results(state)


def _editor(state: WizardState) -> None:
    st.subheader("Add a rule")
    df_cols = list(state.store.current().columns)
    kind = st.selectbox("Rule type", list(rules.RULE_KINDS), key="rule_kind")
    with st.form(f"rule_form_{kind}", clear_on_submit=True):
        severity = st.radio("If broken, this is an", ["error", "warning"], horizontal=True)
        fields: dict = {}
        if kind == "range":
            fields["column"] = st.selectbox("Column", df_cols)
            lo = st.text_input("Minimum (optional)")
            hi = st.text_input("Maximum (optional)")
            fields["min"], fields["max"] = _num(lo), _num(hi)
        elif kind == "required":
            fields["columns"] = st.multiselect("Columns that must have no missing values", df_cols)
        elif kind == "frequency":
            fields["expected_interval"] = st.text_input("Expected interval, e.g. 1h", "1h")
            gap = st.text_input("Longest allowed gap (optional), e.g. 3h")
            fields["max_gap"] = gap or None
        elif kind == "relation":
            fields["column_a"] = st.selectbox("Column A", df_cols)
            fields["operator"] = st.selectbox("Must be", ["<", "<=", ">", ">=", "==", "!="])
            fields["column_b"] = st.selectbox("Column B", df_cols, index=min(1, len(df_cols) - 1))
        else:
            st.caption("Each timestamp may appear only once.")
        if st.form_submit_button("Add rule"):
            try:
                rules.add_rule(state, rules.build_rule(state, kind, severity=severity, **fields))
            except DatanautError as exc:
                state.add_message("error", exc.message)
            st.rerun()


def _num(text: str) -> float | None:
    try:
        return float(text) if text.strip() else None
    except ValueError:
        return None


def _save_load(state: WizardState) -> None:
    left, right = st.columns(2)
    left.download_button(
        "Save rules (JSON)",
        data=rules.rules_json(state),
        file_name="datanaut_rules.json",
        mime="application/json",
        disabled=not state.rules,
    )
    upload = right.file_uploader("Load rules (JSON)", type=["json"], key="rules_upload")
    if upload is not None and right.button("Load these rules", key="rules_load"):
        try:
            text = upload.getvalue().decode("utf-8-sig")
            count = rules.load_rules(state, text, f"The rule file '{upload.name}'")
            state.add_message("success", f"Loaded {count} rule(s).")
        except (DatanautError, UnicodeDecodeError) as exc:
            state.add_message(
                "error", getattr(exc, "message", "The rule file is not readable text.")
            )
        st.rerun()


def _results(state: WizardState) -> None:
    report = state.report
    assert report is not None
    st.subheader("Validation results")
    if not rules.report_is_current(state):
        st.warning("The data changed since this check. Run validation again for current results.")
    (st.success if report.ok else st.error)(
        "All error-level rules are satisfied."
        if report.ok
        else "Some error-level rules are broken."
    )
    st.dataframe(rules.result_table(report), hide_index=True, use_container_width=True)
    for r in report.results:
        if r.status != "fail":
            continue
        rule = next((x for x in state.rules if x.id == r.rule_id), None)
        with st.expander(f"{r.name}: {r.violations:,} violation(s)"):
            rows, total = rules.violating_rows(state, r.rule_id)
            st.caption(f"Showing {len(rows)} of {total:,} affected row(s).")
            st.dataframe(rows, use_container_width=True)
            if rule is None:
                continue
            action = st.selectbox("What to do", rules.actions_for(rule), key=f"act_{r.rule_id}")
            if st.button(f"Apply '{action}' to these rows", key=f"actbtn_{r.rule_id}"):
                columns = None
                if action == "set_missing" and rule.type == "relation":
                    columns = [rule.column_a]
                actions.guard(
                    state, "Validation", rules.apply_rule_action, state, r.rule_id, action, columns
                )
                st.rerun()


# --- versions and downloads ---------------------------------------------------------------


def _versions(state: WizardState) -> None:
    store = state.store
    assert store is not None
    ids = [n.version.version_id for n in store.history()]
    st.subheader("Restore a version")
    choice = st.selectbox("Version", ids, index=ids.index(store.head), key="ver_pick")
    if st.button("Restore this version", disabled=choice == store.head, key="ver_restore"):
        cleaning.restore(state, choice)
        st.rerun()
    with st.expander("Compare versions"):
        a = st.selectbox("From", ids, key="cmp_a")
        b = st.selectbox("To", ids, index=len(ids) - 1, key="cmp_b")
        diff = actions.guard(state, "Clean", cleaning.compare, state, a, b)
        if diff is not None:
            for line in cleaning.diff_summary(diff):
                st.write(line)
            if diff.examples:
                st.dataframe(cleaning.diff_examples(diff), hide_index=True)

    st.subheader("Downloads")
    fp = state.fingerprint
    csv = state.get_or_compute(cache_key(fp, "cleaned_csv"), lambda: cleaning.cleaned_csv(state))
    st.download_button("Cleaned data (CSV)", csv, "cleaned.csv", "text/csv")
    st.download_button(
        "Pipeline log (JSON)", cleaning.log_json(state), "pipeline_log.json", "application/json"
    )
    st.download_button(
        "Recipe (JSON)", cleaning.recipe_json(state), "recipe.json", "application/json"
    )
