# Run with:  streamlit run app.py
"""Datanaut web app entry point and router."""

from __future__ import annotations

import streamlit as st

from datanaut.ui import actions
from datanaut.ui.pages import analyze, clean, insights, map_columns, profile, sidebar, upload
from datanaut.ui.state import STEPS, WizardState

st.set_page_config(
    page_title="Datanaut - An AI-Powered Intelligent Platform for Time-Series Data Analysis, Analytics and Insight Generation",  # noqa: E501
    layout="wide",
)

_PAGES = {
    "Upload": upload,
    "Profile": profile,
    "Map columns": map_columns,
    "Clean": clean,
    "Analyze": analyze,
    "Insights": insights,
}


def _state() -> WizardState:
    """One WizardState per browser session; nothing is shared between sessions."""
    if "wizard" not in st.session_state:
        st.session_state["wizard"] = WizardState()
    return st.session_state["wizard"]


def _start_over() -> None:
    st.session_state.pop("wizard", None)


def _recovery(state: WizardState) -> None:
    """After an unexpected error: ways to carry on, never a traceback."""
    back = STEPS[max(0, STEPS.index(state.step) - 1)]
    left, right = st.columns(2)
    left.button(f"Go back to {back}", on_click=state.go, args=(back,), key="err_back")
    right.button("Start over with a new file", on_click=_start_over, key="err_reset")


def main() -> None:
    state = _state()
    st.title("Datanaut")
    st.caption(
        "An AI-Powered Intelligent Platform for Time-Series Data Analysis, Analytics and Insight Generation"  # noqa: E501
    )
    messages = st.container()
    page = _PAGES.get(state.step)
    if page is None:  # an unknown step can only come from a corrupted session
        state.step = "Upload"
        st.rerun()
    ok = actions.guarded_render(state, state.step, page.render)
    with messages:  # drawn after the page ran, so messages from this run are included
        for message in state.take_messages():
            getattr(st, message.level)(message.text)
    if not ok:
        _recovery(state)
    actions.guarded_render(state, "Sidebar", sidebar.render)


main()
