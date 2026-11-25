"""Ask screen: opt-in AI questions about the data, with the evidence behind every answer."""

from __future__ import annotations

import streamlit as st

from datanaut.ai import config
from datanaut.ai.schemas import AnalystAnswer
from datanaut.ui import actions, ask, charts
from datanaut.ui.pages.common import ai_client, ai_configured, params_markdown
from datanaut.ui.state import WizardState

STEP = "Ask"
_KIND = {
    "cleaning": "Cleaning",
    "anomaly_hypothesis": "Hypothesis to check",
    "peak_reduction": "Reducing peaks",
}


def render(state: WizardState) -> None:
    st.header("Ask")
    st.write(
        "Ask questions about your data in plain English: the AI chooses a query, Python computes "
        "every number, and each answer shows the evidence behind it."
    )
    if state.store is None:
        st.info("Map your columns first.")
        return

    state.ai_enabled = st.toggle(
        "Turn on the AI analyst for this session",
        value=state.ai_enabled,
        key="ai_toggle",
        help="Off by default. Nothing is sent anywhere until you turn this on and ask something.",
    )
    st.caption(
        "When on, a summary of your data (column names and types, summary statistics, the "
        "pipeline log and the ranked findings), your question and the computed result table are "
        "sent to the AI provider. The full dataset is never sent."
    )
    if not state.ai_enabled:
        st.info("The AI analyst is off. The rest of the app works without it.")
        return

    state.ai_sample_rows = st.checkbox(
        f"Also send up to {config.DEFAULT_SAMPLE_ROWS} sample rows",
        value=state.ai_sample_rows,
        key="ai_sample",
        help="A second, separate opt-in. Off by default.",
    )
    with st.expander("Exactly what will be sent to the AI"):
        st.caption("Your question and the table Python computes for it are added to this.")
        st.code(ask.payload_preview(state), language="text")

    ready = ai_configured()
    if not ready:
        st.info(ask.NO_KEY_HELP)
    state.ai_mode = st.radio(
        "Answer style",
        ["beginner", "analyst"],
        index=["beginner", "analyst"].index(state.ai_mode),
        horizontal=True,
        key="ai_mode_radio",
        format_func=lambda m: (
            "Beginner: plain language" if m == "beginner" else "Analyst: method and caveats"
        ),
    )
    counter = st.empty()  # filled in after the question below is handled

    prompt = st.chat_input("Ask a question about your data", disabled=not ready, key="ai_input")
    if prompt and ready:
        with st.spinner("Thinking..."):
            client = actions.guard(state, STEP, ai_client)
            if client is not None:
                actions.guard(state, STEP, ask.ask_question, state, prompt, client)

    counter.caption(ask.questions_line(state))
    for question, answer in state.ai_history:
        _turn(question, answer)
    if not state.ai_history:
        st.caption("Try: What is the average on weekdays compared with weekends?")

    st.subheader("Suggested next steps")
    st.caption("Ideas that refer to your real findings and pipeline steps. Hypotheses are marked.")
    if st.button("Suggest next steps", key="ai_suggest", disabled=not ready):
        with st.spinner("Thinking..."):
            client = actions.guard(state, STEP, ai_client)
            if client is not None:
                actions.guard(state, STEP, ask.run_suggestions, state, client)
    _suggestions(state)


def _turn(question: str, answer: AnalystAnswer) -> None:
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        st.write(answer.explanation)
        if answer.provider:
            note = f"Answered by {answer.provider}"
            if answer.fell_back:
                note += " (the first provider was busy, so the second one answered)"
            st.caption(note)
        if answer.cached:
            st.caption("Repeated question: shown from this session's memory, nothing was sent.")
        if answer.templated:
            st.warning(
                "The AI's wording did not pass the number check twice, so this summary was "
                "written by the app from the result table."
            )
        elif answer.rewritten:
            st.warning(
                "The number check found a figure that was not in the data, so the answer was "
                "rewritten."
            )
        if answer.evidence is not None:
            _evidence(answer)


def _evidence(answer: AnalystAnswer) -> None:
    card = answer.evidence
    assert card is not None
    with st.expander("Evidence"):
        st.write(f"**Columns:** {', '.join(card.columns)}")
        st.write(f"**Time range:** {card.start} to {card.end}")
        st.write(f"**Observations:** {card.n_observations:,}")
        st.write(f"**Method:** {card.method}")
        st.caption("The query the AI chose (checked against an allowlist, then run by Python):")
        st.markdown(params_markdown(card.query))
        st.dataframe(
            [dict(zip(card.table.columns, row, strict=True)) for row in card.table.rows],
            hide_index=True,
            use_container_width=True,
        )
        if card.table.truncated:
            st.caption("Only the first rows are shown.")
        fig = charts.ask_chart(card)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


def _suggestions(state: WizardState) -> None:
    outcome = state.ai_suggestions
    if outcome is None:
        return
    if outcome.templated:
        st.caption(
            "These suggestions were written by the app from your findings"
            + (f" ({outcome.error})" if outcome.error else "")
            + "."
        )
    if not outcome.suggestions:
        st.info("No suggestion refers to a real finding yet. Generate findings on Insights first.")
    for s in outcome.suggestions:
        with st.container(border=True):
            st.markdown(f"**{s.title}**")
            st.caption(_KIND.get(s.kind, s.kind))
            st.write(s.rationale)
            if s.action:
                st.write(f"What to do: {s.action}")
            ref = []
            if s.finding_id:
                ref.append(f"finding {s.finding_id}")
            if s.log_entry_id is not None:
                ref.append(f"pipeline step {s.log_entry_id}")
            st.caption("Refers to " + " and ".join(ref))
