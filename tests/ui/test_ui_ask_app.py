import logging
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "ai"))
from aihelpers import KEY, ScriptedClient, text, tool  # noqa: E402

from datanaut.errors import RateLimitError  # noqa: E402
from datanaut.ui import actions, ask  # noqa: E402
from datanaut.ui.state import STEPS, WizardState  # noqa: E402

APP = str(ROOT / "app.py")
FIRST_VOLTAGE = "238.82"  # the first voltage reading in the sample file
MEAN = {"value_columns": ["global_active_power"], "aggregation": "mean"}


def answering_client(n: int = 1) -> ScriptedClient:
    script = []
    for _ in range(n):
        script += [tool("query_data", **MEAN), text("The average is shown in the table.")]
    return ScriptedClient(*script)


def mapped() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300).run()
    for label in ("Use sample data", "Continue to Profile", "Continue to Map columns"):
        [b for b in at.button if b.label == label][0].click()
        at = at.run()
    [b for b in at.button if b.label == "Build time index and create V1"][0].click()
    return at.run()


def on_ask(client: ScriptedClient) -> AppTest:
    at = mapped()
    at.session_state["ai_client_factory"] = lambda: client
    at.sidebar.button(key="nav_Ask").click()
    return at.run()


def page_text(at: AppTest) -> str:
    parts: list[str] = []
    for kind in (
        "markdown",
        "text",
        "code",
        "caption",
        "error",
        "info",
        "warning",
        "success",
        "subheader",
    ):
        parts += [str(e.value) for e in at.get(kind)]
    return "\n".join(parts)


# --- the screen ---------------------------------------------------------------------------


def test_ask_is_a_step_after_insights_and_needs_the_mapping_first():
    assert STEPS[-2:] == ("Insights", "Ask")
    s = WizardState()
    assert not s.is_reachable("Ask")
    s.mark_done("Upload")
    assert not s.is_reachable("Ask")
    s.mark_done("Map columns")
    assert s.is_reachable("Ask")


def test_with_the_opt_in_off_nothing_is_sent_and_the_page_says_so():
    client = answering_client()
    at = on_ask(client)
    assert not at.exception and at.header[0].value == "Ask"
    assert at.toggle(key="ai_toggle").value is False
    assert any("analyst is off" in i.value for i in at.info)
    assert len(at.chat_input) == 0 and client.calls == []
    at = at.run()
    assert client.calls == []  # reruns send nothing either


def test_turning_it_on_shows_exactly_what_will_be_sent_and_still_sends_nothing():
    client = answering_client()
    at = on_ask(client)
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    assert not at.exception and client.calls == []
    assert any("Exactly what will be sent to the AI" in e.label for e in at.expander)
    preview = " ".join(c.value for c in at.code)
    assert "<<DATA:schema>>" in preview and "global_active_power" in preview
    assert "sample_rows" not in preview and FIRST_VOLTAGE not in preview
    assert at.checkbox(key="ai_sample").value is False

    at.checkbox(key="ai_sample").check()
    at = at.run()
    preview = " ".join(c.value for c in at.code)
    assert "sample_rows" in preview and FIRST_VOLTAGE in preview  # only after the separate opt-in
    assert client.calls == []


def test_an_answer_renders_with_its_evidence_and_provider_badge():
    client = answering_client()
    at = on_ask(client)
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    at.chat_input(key="ai_input").set_value("What is the average power?").run()
    at = at.run() if False else at
    state = at.session_state["wizard"]
    assert len(client.calls) == 2 and len(state.ai_history) == 1
    question, answer = state.ai_history[0]
    assert answer.evidence is not None and answer.provider == "scripted"
    assert not at.exception


def test_the_screen_shows_the_answer_text_evidence_and_counter():
    client = answering_client()
    at = on_ask(client)
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    at.chat_input(key="ai_input").set_value("What is the average power?")
    at = at.run()
    assert not at.exception
    shown = page_text(at)
    assert "What is the average power?" in shown
    assert "The average is shown in the table." in shown
    assert "Answered by scripted" in shown
    assert any(e.label == "Evidence" for e in at.expander)
    assert "Questions used: 1 of 20" in shown
    assert any("global_active_power" in str(j.value) for j in at.json)  # the query is visible


def test_the_cap_message_appears_after_the_cap_and_the_model_is_not_called_again():
    client = answering_client(2)
    at = on_ask(client)
    guard = at.session_state["wizard"].ai_session.guard
    guard.question_cap, guard.min_interval = 1, 0.0
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    at.chat_input(key="ai_input").set_value("What is the average power?")
    at = at.run()
    calls_after_first = len(client.calls)
    at.chat_input(key="ai_input").set_value("And the maximum power?")
    at = at.run()
    assert any("all 1 AI questions" in e.value for e in at.error)
    assert len(client.calls) == calls_after_first and not at.exception


def test_a_rate_limit_is_shown_clearly_and_the_session_carries_on():
    client = ScriptedClient(RateLimitError())
    at = on_ask(client)
    at.session_state["wizard"].ai_session.guard.min_interval = 0.0
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    at.chat_input(key="ai_input").set_value("What is the average power?")
    at = at.run()
    assert any("busy" in e.value for e in at.error) and not at.exception
    assert at.session_state["wizard"].ai_history == []


def test_no_key_means_a_helpful_note_and_no_chat_while_the_rest_works(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    at = mapped()
    at.sidebar.button(key="nav_Ask").click()
    at = at.run()
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in i.value and "secrets.toml" in i.value for i in at.info)
    assert at.chat_input(key="ai_input").disabled is True
    at.sidebar.button(key="nav_Analyze").click()
    at = at.run()
    assert not at.exception and at.header[0].value == "Analyze"


def test_suggestions_come_from_a_button_and_point_at_real_steps():
    client = ScriptedClient(
        tool(
            "suggestions",
            items=[
                {
                    "title": "Review how the file was loaded",
                    "kind": "cleaning",
                    "log_entry_id": 1,
                    "rationale": "The load step is the first record in the pipeline log.",
                },
                {"title": "Ghost", "kind": "cleaning", "log_entry_id": 999, "rationale": "None."},
            ],
        )
    )
    at = on_ask(client)
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    assert client.calls == []
    at.button(key="ai_suggest").click()
    at = at.run()
    assert not at.exception and len(client.calls) == 1
    shown = page_text(at)
    assert "Review how the file was loaded" in shown and "Ghost" not in shown
    assert "Refers to pipeline step 1" in shown


def test_no_key_or_data_value_reaches_the_page_or_the_log(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    caplog.set_level(logging.DEBUG)
    client = answering_client()
    at = on_ask(client)
    at.toggle(key="ai_toggle").set_value(True)
    at = at.run()
    at.chat_input(key="ai_input").set_value("What is the average power?")
    at = at.run()
    everything = page_text(at) + "\n" + caplog.text
    assert KEY not in everything and FIRST_VOLTAGE not in everything
    assert KEY not in repr(at.session_state["wizard"].ai_history)


# --- the explain button on Insights -------------------------------------------------------


def test_the_investigation_can_be_explained_once_the_ai_is_on():
    client = ScriptedClient(
        text("This is an association in the data and does not show what produced the change.")
    )
    at = mapped()
    at.button(key="apply_sel").click()  # the default fixes remove the repeated timestamps
    at = at.run()
    at.session_state["ai_client_factory"] = lambda: client
    at.sidebar.button(key="nav_Insights").click()
    at = at.run()
    at.button(key="run_inv").click()
    at = at.run()
    assert not at.exception
    assert any("Turn on the AI analyst" in c.value for c in at.caption)  # off: no button
    assert [b for b in at.button if b.key == "ai_explain"] == [] and client.calls == []
    at.session_state["wizard"].ai_enabled = True
    at = at.run()
    assert not at.exception and client.calls == []
    at.button(key="ai_explain").click()
    at = at.run()
    assert not at.exception and len(client.calls) == 1
    assert "This is an association in the data" in page_text(at)
    again = at.run()
    assert len(client.calls) == 1 and not again.exception  # kept for the session, not re-asked


# --- the logic without the page -----------------------------------------------------------


def test_keys_come_from_secrets_or_the_environment_and_secrets_win():
    env = {"GEMINI_API_KEY": "env-g", "GROQ_API_KEY": "env-r"}
    assert ask.provider_keys({"GEMINI_API_KEY": "sec-g"}, env) == {
        "gemini": "sec-g",
        "groq": "env-r",
    }
    assert ask.provider_keys(None, {}) == {}
    assert ask.provider_keys({"GEMINI_API_KEY": ""}, {"GROQ_API_KEY": "x"}) == {"groq": "x"}


@pytest.fixture
def state():
    s = WizardState()
    actions.load_sample(s)
    time_cols, values = actions.suggest_mapping(s.loaded.df)
    actions.confirm_mapping(s, time_cols, None, values, {})
    s.take_messages()
    return s


def test_history_is_kept_in_the_session_and_cleared_for_a_new_file(state):
    state.ai_session.guard.min_interval = 0.0
    answer = ask.ask_question(state, "What is the average power?", answering_client())
    assert answer is not None and len(state.ai_history) == 1
    assert ask.questions_line(state) == "Questions used: 1 of 20"
    state.invalidate_mapping("changed")
    assert state.ai_history == [] and len(state.ai_session.cache) == 0
    assert state.ai_session.guard.questions_used == 1  # the cap belongs to the session


def test_ask_question_turns_errors_into_messages(state):
    state.ai_session.guard.min_interval = 0.0
    assert ask.ask_question(state, "q", ScriptedClient(RateLimitError())) is None
    assert state.messages[-1].level == "error" and state.ai_history == []


def test_the_payload_follows_the_two_opt_ins(state):
    assert "sample_rows" not in ask.payload_preview(state)
    state.ai_sample_rows = True
    assert "sample_rows" in ask.payload_preview(state)
    assert len(ask.build_payload(state).sample_rows) == 20


def test_investigating_uncleaned_data_with_repeated_timestamps_says_what_to_do(state):
    import pandas as pd

    from datanaut.errors import AnalysisError
    from datanaut.ui import analysis

    last = analysis.series(state).index.max()
    with pytest.raises(AnalysisError, match="repeated timestamps"):
        analysis.run_investigation(state, last - pd.Timedelta(days=7), last, "previous")
