"""Profile screen: a read-only picture of the data's quality."""

from __future__ import annotations

import streamlit as st

from datanaut.ui import actions, charts, display
from datanaut.ui.state import WizardState


def render(state: WizardState) -> None:
    st.header("Profile")
    st.write("A read-only check of your data: size, missing values, duplicates and gaps in time.")
    prof = actions.guard(state, "Profile", actions.ensure_profile, state)
    if prof is None:
        return
    state.mark_done("Profile")

    for col, (label, value) in zip(st.columns(5), display.profile_cards(prof), strict=True):
        col.metric(label, value)
    st.caption(display.sampling_line(prof))

    if prof.warnings:
        st.subheader("Things to know")
        for warning in prof.warnings:
            st.warning(warning)

    st.subheader("Columns")
    st.dataframe(display.column_table(prof), use_container_width=True, hide_index=True)

    st.subheader("Missing values")
    matrix = prof.missingness_matrix
    if matrix.empty or float(matrix.to_numpy(dtype=float).sum()) == 0.0:
        st.success("No missing values were found.")
    else:
        st.plotly_chart(charts.missingness_heatmap(matrix), use_container_width=True)

    st.subheader("Gaps in time")
    gaps = display.gap_table(prof)
    if gaps.empty:
        st.write("No gaps were found in the time index.")
    else:
        st.dataframe(gaps, use_container_width=True, hide_index=True)
    st.button(
        "Continue to Map columns",
        type="primary",
        on_click=state.go,
        args=("Map columns",),
        key="prof_next",
    )
