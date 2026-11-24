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


def ai_client():
    """The AI client for this session: a test override, or one built from keys on hand.

    Raises AIConfigError if no key is configured. Keys are looked up each time and never kept.
    """
    from datanaut.errors import AIConfigError
    from datanaut.ui import ask

    factory = st.session_state.get("ai_client_factory")
    if factory is not None:
        return factory()
    secrets: dict[str, str] = {}
    try:
        from datanaut.ai import config

        for name in config.KEY_NAMES.values():
            if name in st.secrets:
                secrets[name] = st.secrets[name]
    except Exception:  # no secrets file is the normal case
        secrets = {}
    keys = ask.provider_keys(secrets)
    if not keys:
        raise AIConfigError(ask.NO_KEY_HELP)
    return ask.real_client(keys)


def ai_configured() -> bool:
    """Is any provider available (without building a client)?"""
    from datanaut.errors import AIConfigError

    try:
        ai_client()
    except AIConfigError:
        return False
    return True
