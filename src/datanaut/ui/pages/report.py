"""Report screen: choose a version and sections, build the PDF and download the files."""

from __future__ import annotations

import streamlit as st

from datanaut import startup
from datanaut.report.model import SECTION_TITLES
from datanaut.ui import actions, cleaning, reporting
from datanaut.ui.state import WizardState

STEP = "Report"


def render(state: WizardState) -> None:
    st.header("Report")
    st.write(
        "Build a PDF of everything you have done and found, for any version of your data, "
        "and download the cleaned data and its recipe."
    )
    if state.store is None:
        st.info("Map your columns first.")
        return
    if not startup.report_available():
        st.warning(
            "The PDF report is turned off because its fonts were not found (they ship with "
            "matplotlib). You can still download the cleaned data, the log and the recipe."
        )
    store = state.store
    ids = reporting.version_ids(state)
    version_id = st.selectbox(
        "Dataset version",
        ids,
        index=ids.index(store.head),
        key="rep_version",
        help="The report describes this version only.",
    )
    counts = actions.guard(state, STEP, reporting.section_counts, state, version_id)
    if counts is None:
        return
    st.subheader("What to include")
    st.caption(
        "Only sections that have something to show are listed. Run analyses on the Analyze, "
        "Insights and Ask screens first to fill more of them."
    )
    selected = [
        sid
        for sid in SECTION_TITLES
        if sid in counts
        and st.checkbox(
            f"{SECTION_TITLES[sid]} ({counts[sid]})", value=True, key=f"rep_sec_{version_id}_{sid}"
        )
    ]
    st.caption(reporting.preview_line(selected, counts))

    if st.button(
        "Generate PDF",
        type="primary",
        key="rep_generate",
        disabled=not selected or not startup.report_available(),
        help="Builds the report in memory; nothing is saved on the server",
    ):
        with st.spinner("Building the report..."):
            actions.guard(state, STEP, reporting.generate_pdf, state, version_id, selected)
    ready = state.report_pdf
    if ready is not None and ready[0] == reporting.pdf_key(version_id, selected):
        st.success(f"The report is ready ({len(ready[1]) / 1024:,.0f} KB).")
        st.download_button(
            "Download the PDF report",
            ready[1],
            "datanaut_report.pdf",
            "application/pdf",
            key="rep_download",
        )

    st.subheader("Data and records for this version")
    csv = state.get_or_compute(
        (state.fingerprint, "rep_csv", version_id), lambda: cleaning.cleaned_csv(state, version_id)
    )
    st.download_button(
        "Cleaned data (CSV)", csv, f"cleaned_{version_id}.csv", "text/csv", key="rep_csv"
    )
    st.download_button(
        "Pipeline log (JSON)",
        cleaning.log_json(state),
        "pipeline_log.json",
        "application/json",
        key="rep_log",
    )
    st.download_button(
        "Recipe (JSON)",
        cleaning.recipe_json(state, version_id),
        f"recipe_{version_id}.json",
        "application/json",
        key="rep_recipe",
    )
