"""Small widgets shared by the Analyze and Insights screens."""

from __future__ import annotations

import streamlit as st

from datanaut.ui import actions
from datanaut.ui.analysis import Run
from datanaut.ui.state import WizardState


def evidence_box(result) -> None:
    """The evidence behind a result: columns, time range, observations and method."""
    ev = result.evidence
    with st.expander("Evidence"):
        st.write(f"**Columns:** {', '.join(ev.columns)}")
        st.write(f"**Time range:** {ev.start} to {ev.end}")
        st.write(f"**Observations:** {ev.n_observations:,}")
        st.write(f"**Method:** {ev.method}")
        if ev.parameters:
            st.json(ev.parameters, expanded=False)


def run_button(
    state: WizardState, step: str, label: str, key: str, fn, *args, **kwargs
) -> Run | None:
    """A button that runs `fn` behind a spinner and shows how long it took."""
    if st.button(label, key=key, type="primary"):
        with st.spinner("Working..."):
            return actions.guard(state, step, fn, state, *args, **kwargs)
    return None


def timing(run: Run | None) -> None:
    if run is not None:
        st.caption(run.caption())
