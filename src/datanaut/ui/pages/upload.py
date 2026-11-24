"""Upload screen: choose a file (or the sample) and see a preview."""

from __future__ import annotations

import streamlit as st

from datanaut.ui import actions
from datanaut.ui.display import format_bytes
from datanaut.ui.state import WizardState


def render(state: WizardState) -> None:
    st.header("Upload")
    st.write(
        "Choose a CSV or Excel file with a time column and numeric readings, or try the sample."
    )

    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"], key="uploader")
    if uploaded is not None:
        actions.guard(
            state, "Upload", actions.load_upload, state, uploaded.getvalue(), uploaded.name
        )
    if st.button("Use sample data", help="Load the bundled household energy example"):
        actions.guard(state, "Upload", actions.load_sample, state)

    loaded = state.loaded
    if loaded is None:
        st.info("No file loaded yet. Nothing leaves this session.")
        return
    info = loaded.source_info
    if info.format == "xlsx" and len(info.sheet_names) > 1:
        names = list(info.sheet_names)
        choice = st.selectbox("Excel sheet", names, index=names.index(info.sheet or names[0]))
        if choice != info.sheet and uploaded is not None:
            actions.guard(
                state,
                "Upload",
                actions.load_upload,
                state,
                uploaded.getvalue(),
                uploaded.name,
                choice,
            )
            loaded = state.loaded
            assert loaded is not None

    df = loaded.df
    cols = st.columns(4)
    cols[0].metric("File", loaded.source_info.filename or "uploaded file")
    cols[1].metric("Rows", f"{len(df):,}")
    cols[2].metric("Columns", f"{df.shape[1]:,}")
    cols[3].metric("Size", format_bytes(loaded.source_info.size_bytes))
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.button(
        "Continue to Profile", type="primary", on_click=state.go, args=("Profile",), key="up_next"
    )
