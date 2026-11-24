"""Logic for the Ask screen: keys, the payload preview, questions, suggestions and summaries.

No streamlit import. Keys are read from the environment or Streamlit secrets handed in by the
page and are never stored in the session state. Conversation history lives in the WizardState
only; nothing is written anywhere.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from datanaut.ai import config
from datanaut.ai.analyst import ask
from datanaut.ai.context import ContextPayload, build_context
from datanaut.ai.llm import LLMClient, make_client, urllib_transport
from datanaut.ai.schemas import AnalystAnswer
from datanaut.ai.suggest import (
    SuggestionOutcome,
    SummaryOutcome,
    suggest_with_status,
    summarize_with_status,
)
from datanaut.core.findings import Finding
from datanaut.core.investigate import InvestigationResult
from datanaut.errors import AIError
from datanaut.ui import analysis
from datanaut.ui.state import WizardState

NO_KEY_HELP = (
    "No API key was found. Add GEMINI_API_KEY and/or GROQ_API_KEY to .streamlit/secrets.toml "
    "(that file is kept out of git) or to your environment, then restart the app. The rest of "
    "the app works without it."
)


def provider_keys(
    secrets: Mapping[str, str] | None = None, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The keys that are configured, by provider. Secrets win over the environment."""
    env = os.environ if env is None else env
    found: dict[str, str] = {}
    for provider, name in config.KEY_NAMES.items():
        value = (secrets or {}).get(name) or env.get(name) or ""
        if value:
            found[provider] = str(value)
    return found


def real_client(keys: Mapping[str, str]) -> LLMClient:
    """A client over the real network. Raises AIConfigError if there is no key."""
    return make_client(keys, urllib_transport)


# --- what is sent -------------------------------------------------------------------------


def known_findings(state: WizardState) -> list[Finding]:
    """The ranked findings, if they were generated on the Insights screen."""
    cached = analysis.peek(state, "findings", weather=state.weather_id)
    return [] if cached is None else list(cached.result.findings)


def build_payload(state: WizardState) -> ContextPayload:
    """Exactly the background that would be sent about the data (no sample rows unless opted in)."""
    assert state.store is not None
    return build_context(
        state.store.current(),
        findings=known_findings(state),
        log=state.log,
        sample_rows=config.DEFAULT_SAMPLE_ROWS if state.ai_sample_rows else 0,
    )


def payload_preview(state: WizardState) -> str:
    return build_payload(state).to_prompt()


# --- questions ----------------------------------------------------------------------------


def questions_line(state: WizardState) -> str:
    guard = state.ai_session.guard
    return f"Questions used: {guard.questions_used} of {guard.question_cap}"


def ask_question(state: WizardState, question: str, client: LLMClient) -> AnalystAnswer | None:
    """Ask one question. Provider and session errors become a message, not an exception."""
    assert state.store is not None
    try:
        answer = ask(
            question,
            state.ai_mode,  # type: ignore[arg-type]
            build_payload(state),
            state.store.current(),
            client,
            session=state.ai_session,
            fingerprint=state.fingerprint,
        )
    except AIError as exc:
        state.add_message("error", exc.message)
        return None
    state.ai_history.append((question, answer))
    return answer


def run_suggestions(state: WizardState, client: LLMClient) -> SuggestionOutcome:
    """Suggested next steps that refer to real findings and log entries."""
    outcome = suggest_with_status(
        build_payload(state), known_findings(state), state.profile, state.log, client
    )
    state.ai_suggestions = outcome
    return outcome


def summary_key(state: WizardState, result: InvestigationResult) -> str:
    return "|".join(
        [
            state.fingerprint,
            result.column,
            result.period_start,
            result.period_end,
            result.baseline.kind,
            state.ai_mode,
        ]
    )


def explain_investigation(
    state: WizardState, result: InvestigationResult, client: LLMClient
) -> SummaryOutcome:
    """A plain-language summary of an investigation, kept for the rest of the session."""
    key = summary_key(state, result)
    if key not in state.ai_summaries:
        state.ai_summaries[key] = summarize_with_status(
            result,
            state.ai_mode,
            client,  # type: ignore[arg-type]
        )
    return state.ai_summaries[key]
