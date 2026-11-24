"""Sidebar: the step list with status, the pipeline log, the version list and the privacy note."""

from __future__ import annotations

import streamlit as st

from datanaut.ui import display
from datanaut.ui.state import STEPS, WizardState

PRIVACY_NOTE = "Your file is processed in this session only and is not stored."
_ICONS = {"done": "✓", "current": "▶", "available": "○", "locked": "🔒"}


def render(state: WizardState) -> None:
    with st.sidebar:
        st.title("Datanaut")
        st.subheader("Steps")
        for number, step in enumerate(STEPS, start=1):
            status = state.status(step)
            st.button(
                f"{_ICONS[status]} {number}. {step}",
                key=f"nav_{step}",
                disabled=status == "locked",
                use_container_width=True,
                on_click=state.go,
                args=(step,),
                help=f"Go to {step}" if status != "locked" else "Finish the earlier steps first",
            )
        _versions(state)
        _log(state)
        st.caption(PRIVACY_NOTE)


def _versions(state: WizardState) -> None:
    rows = display.version_rows(state.store)
    st.subheader("Versions")
    if not rows:
        st.caption("Versions appear once columns are mapped.")
        return
    for row in rows:
        mark = "  ← current" if row["is_head"] else ""
        st.markdown(
            f"{'&nbsp;' * 4 * row['depth']}**{row['id']}** {row['label']} "
            f"({row['rows']:,} rows){mark}"
        )


def _log(state: WizardState) -> None:
    rows = display.log_rows(state.log)
    with st.expander(f"Pipeline log ({len(rows)})"):
        if not rows:
            st.caption("Nothing has been done yet.")
        for row in rows:  # expanders cannot nest, so each entry is a short block
            st.markdown(f"**{row['step']}. {row['operation']}**")
            st.caption(row["summary"])
            details = ["Read-only" if row["read_only"] else "Changed the data"]
            if row["rows_before"] is not None:
                details.append(f"rows {row['rows_before']:,} → {row['rows_after']:,}")
            details += [f"{k}: {v:g}" for k, v in row["counts"].items()]
            st.caption(" · ".join(details))
