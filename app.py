# Run with:  streamlit run app.py
"""Datanaut web app entry point and router."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

try:
    import datanaut  # noqa: F401
except ImportError:  # an unpackaged checkout, as on Streamlit Community Cloud: use ./src
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from datanaut import startup  # noqa: E402
from datanaut.ai import config as ai_config  # noqa: E402
from datanaut.ui import actions, overview, theme  # noqa: E402
from datanaut.ui.pages import (  # noqa: E402
    analyze,
    ask,
    clean,
    insights,
    map_columns,
    profile,
    report,
    sidebar,
    upload,
)
from datanaut.ui.state import STEPS, WizardState  # noqa: E402

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
    "Ask": ask,
    "Report": report,
}


def _state() -> WizardState:
    """One WizardState per browser session; nothing is shared between sessions."""
    if "wizard" not in st.session_state:
        st.session_state["wizard"] = WizardState()
    return st.session_state["wizard"]


def _notices() -> list[startup.Notice]:
    """What is missing at start-up, checked once per session (keys are only looked for)."""
    if "startup_notices" not in st.session_state:
        present: dict[str, str] = {}
        try:
            present = {n: "set" for n in ai_config.KEY_NAMES.values() if n in st.secrets}
        except Exception:  # no secrets file is the normal case
            present = {}
        st.session_state["startup_notices"] = startup.check_environment(secrets=present)
    return st.session_state["startup_notices"]


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
    with st.sidebar:  # the stylesheet lives here so the main column keeps only content
        st.markdown(theme.style_tag(), unsafe_allow_html=True)
    st.title("Datanaut")
    st.caption(
        "An AI-Powered Intelligent Platform for Time-Series Data Analysis, Analytics and Insight Generation"  # noqa: E501
    )
    board = overview.overview_html(state)
    if board is not None:
        st.markdown(board, unsafe_allow_html=True)
    messages = st.container()
    for notice in _notices():
        if notice.level == "warning":
            messages.warning(f"{notice.feature}: {notice.message}")
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
