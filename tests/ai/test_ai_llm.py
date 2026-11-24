import json
import logging

import pytest
from aihelpers import KEY, FakeTransport, ScriptedClient, text

from datanaut.ai import config
from datanaut.ai.llm import (
    AISession,
    AnswerCache,
    BudgetedClient,
    CallBudget,
    FallbackClient,
    GeminiClient,
    GroqClient,
    Message,
    SessionGuard,
    ToolSpec,
    make_client,
)
from datanaut.errors import (
    AIConfigError,
    MalformedResponseError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    SafetyBlockError,
    SessionLimitError,
)

TOOL = ToolSpec("query_data", "d", {"type": "object", "properties": {"x": {"type": "string"}}})
MSGS = [Message("user", "hello")]


def gemini_ok(**part):
    return (
        200,
        {
            "candidates": [{"content": {"parts": [part]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3},
        },
    )


def groq_ok(message):
    return (
        200,
        {
            "choices": [{"message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        },
    )


# --- Gemini -------------------------------------------------------------------------------


def test_gemini_request_and_response_are_translated():
    t = FakeTransport(gemini_ok(functionCall={"name": "query_data", "args": {"x": "1"}}))
    r = GeminiClient(KEY, t).generate("SYS", MSGS, [TOOL], "tool")
    call = t.calls[0]
    assert config.GEMINI_MODEL in call["url"] and call["url"].endswith(":generateContent")
    assert call["headers"]["x-goog-api-key"] == KEY and KEY not in json.dumps(call["body"])
    body = call["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "SYS"
    assert body["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    decl = body["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "query_data" and decl["parameters"] == TOOL.parameters
    assert body["toolConfig"]["functionCallingConfig"] == {
        "mode": "ANY",
        "allowedFunctionNames": ["query_data"],
    }
    assert [(c.name, c.arguments) for c in r.tool_calls] == [("query_data", {"x": "1"})]
    assert r.provider == "gemini" and (r.prompt_tokens, r.completion_tokens) == (11, 3)


def test_gemini_text_mode_offers_no_tools_and_maps_assistant_to_model():
    t = FakeTransport(gemini_ok(text="hi there"))
    msgs = [Message("user", "a"), Message("assistant", "b"), Message("user", "c")]
    r = GeminiClient(KEY, t).generate("S", msgs, [TOOL], "text")
    body = t.calls[0]["body"]
    assert "tools" not in body and "toolConfig" not in body
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert r.text == "hi there" and r.tool_calls == []


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((429, {}), RateLimitError),
        ((504, {}), ProviderTimeoutError),
        ((408, {}), ProviderTimeoutError),
        ((401, {}), ProviderError),
        ((500, {}), ProviderError),
        ((200, "not json"), MalformedResponseError),
        ((200, {"candidates": []}), MalformedResponseError),
        ((200, {"candidates": [{"content": {"parts": "x"}}]}), MalformedResponseError),
        (
            (200, {"candidates": [{"content": {"parts": [{"functionCall": {"args": {}}}]}}]}),
            MalformedResponseError,
        ),
        ((200, {"promptFeedback": {"blockReason": "SAFETY"}}), SafetyBlockError),
        (
            (200, {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}),
            SafetyBlockError,
        ),
        (TimeoutError("slow"), ProviderTimeoutError),
        (ConnectionError("down"), ProviderError),
    ],
)
def test_gemini_errors_are_typed(reply, error):
    with pytest.raises(error):
        GeminiClient(KEY, FakeTransport(reply)).generate("S", MSGS, [TOOL], "tool")


# --- Groq ---------------------------------------------------------------------------------


def test_groq_request_and_response_are_translated():
    msg = {
        "content": None,
        "tool_calls": [{"function": {"name": "query_data", "arguments": '{"x": "2"}'}}],
    }
    t = FakeTransport(groq_ok(msg))
    r = GroqClient(KEY, t).generate("SYS", MSGS, [TOOL], "tool")
    call = t.calls[0]
    assert call["url"] == config.GROQ_URL and call["headers"]["Authorization"] == f"Bearer {KEY}"
    body = call["body"]
    assert KEY not in json.dumps(body) and body["model"] == config.GROQ_MODEL
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert (
        body["tools"][0]["function"]["name"] == "query_data" and body["tool_choice"] == "required"
    )
    assert r.tool_calls[0].arguments == {"x": "2"} and r.provider == "groq"
    assert (r.prompt_tokens, r.completion_tokens) == (7, 2)


def test_groq_text_mode_has_no_tools():
    t = FakeTransport(groq_ok({"content": "plain"}))
    r = GroqClient(KEY, t).generate("S", MSGS, [TOOL], "text")
    assert "tools" not in t.calls[0]["body"] and "tool_choice" not in t.calls[0]["body"]
    assert r.text == "plain"


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ((429, {}), RateLimitError),
        ((504, {}), ProviderTimeoutError),
        ((403, {}), ProviderError),
        ((200, {"choices": []}), MalformedResponseError),
        (
            (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [{"function": {"name": "q", "arguments": "{oops"}}]
                            }
                        }
                    ]
                },
            ),
            MalformedResponseError,
        ),
        (
            (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [{"function": {"name": "q", "arguments": "[1]"}}]
                            }
                        }
                    ]
                },
            ),
            MalformedResponseError,
        ),
        ((200, {"error": {"code": "tool_use_failed"}}), MalformedResponseError),
        ((200, {"error": {"code": "other"}}), ProviderError),
        (
            (200, {"choices": [{"finish_reason": "content_filter", "message": {}}]}),
            SafetyBlockError,
        ),
        (OSError("reset"), ProviderError),
    ],
)
def test_groq_errors_are_typed(reply, error):
    with pytest.raises(error):
        GroqClient(KEY, FakeTransport(reply)).generate("S", MSGS, [TOOL], "tool")


def test_an_empty_key_is_a_config_error():
    for cls in (GeminiClient, GroqClient):
        with pytest.raises(AIConfigError):
            cls("", FakeTransport())


# --- fallback and construction ------------------------------------------------------------


def test_fallback_uses_the_secondary_on_a_rate_limit_and_says_so():
    primary = ScriptedClient(RateLimitError())
    primary.name = "gemini"
    secondary = ScriptedClient(text("from groq"))
    secondary.name = "groq"
    r = FallbackClient(primary, secondary).generate("S", MSGS, [], "text")
    assert r.text == "from groq" and r.fell_back is True and len(primary.calls) == 1


def test_fallback_also_covers_timeouts_but_not_other_errors():
    a, b = ScriptedClient(ProviderTimeoutError()), ScriptedClient(text("ok"))
    assert FallbackClient(a, b).generate("S", MSGS, [], "text").fell_back
    c, d = ScriptedClient(MalformedResponseError()), ScriptedClient(text("never"))
    with pytest.raises(MalformedResponseError):
        FallbackClient(c, d).generate("S", MSGS, [], "text")
    assert d.calls == []


def test_fallback_without_a_secondary_and_when_both_fail_raises():
    with pytest.raises(RateLimitError):
        FallbackClient(ScriptedClient(RateLimitError())).generate("S", MSGS, [], "text")
    both = FallbackClient(ScriptedClient(RateLimitError()), ScriptedClient(RateLimitError()))
    with pytest.raises(RateLimitError):
        both.generate("S", MSGS, [], "text")


def test_a_primary_that_answers_is_not_marked_as_a_fallback():
    r = FallbackClient(ScriptedClient(text("ok")), ScriptedClient()).generate("S", MSGS, [], "text")
    assert r.fell_back is False


def test_make_client_orders_providers_and_needs_a_key():
    t = FakeTransport(gemini_ok(text="g"))
    only_groq = make_client({"groq": KEY}, FakeTransport(groq_ok({"content": "r"})))
    assert only_groq.generate("S", MSGS, [], "text").provider == "groq"
    both = make_client({"gemini": KEY, "groq": KEY}, t)
    assert both.generate("S", MSGS, [], "text").provider == "gemini"
    with pytest.raises(AIConfigError):
        make_client({}, t)


# --- keys never leak ----------------------------------------------------------------------


def test_keys_never_appear_in_logs_errors_or_reprs(caplog):
    caplog.set_level(logging.DEBUG)
    boom = RuntimeError(f"connection used header {KEY}")
    for client in (GeminiClient(KEY, FakeTransport(boom)), GroqClient(KEY, FakeTransport(boom))):
        with pytest.raises(ProviderError) as err:
            client.generate("S", MSGS, [TOOL], "tool")
        assert KEY not in str(err.value) and KEY not in repr(err.value)
        assert KEY not in repr(client) and KEY not in str(client)
    for reply in ((429, {"echo": KEY}), (500, {"echo": KEY}), (200, {"echo": KEY})):
        for cls in (GeminiClient, GroqClient):
            with pytest.raises(Exception) as err:  # noqa: B017
                cls(KEY, FakeTransport(reply)).generate("S", MSGS, [TOOL], "tool")
            assert KEY not in str(err.value) and KEY not in repr(err.value)
    ok = GeminiClient(KEY, FakeTransport(gemini_ok(text="fine")))
    ok.generate("S", [Message("user", "PRIVATE-PROMPT-TEXT")], [], "text")
    assert KEY not in caplog.text and "PRIVATE-PROMPT-TEXT" not in caplog.text
    assert "provider=gemini" in caplog.text  # metadata is logged


# --- limits and the cache -----------------------------------------------------------------


def test_the_question_cap_and_the_minimum_gap_are_enforced():
    now = [100.0]
    guard = SessionGuard(question_cap=2, min_interval=5.0, clock=lambda: now[0])
    guard.begin_question()
    assert guard.remaining == 1
    with pytest.raises(SessionLimitError, match="wait"):
        guard.begin_question()
    now[0] += 5
    guard.begin_question()
    now[0] += 60
    with pytest.raises(SessionLimitError, match="all 2 AI questions"):
        guard.begin_question()
    assert guard.questions_used == 2 and guard.remaining == 0


def test_the_call_budget_stops_runaway_retries():
    client = ScriptedClient(*[text("x")] * 10)
    wrapped = BudgetedClient(client, CallBudget(limit=3))
    for _ in range(3):
        wrapped.generate("S", MSGS, [], "text")
    with pytest.raises(SessionLimitError, match="too many model calls"):
        wrapped.generate("S", MSGS, [], "text")
    assert len(client.calls) == 3


def test_the_cache_is_bounded_and_normalizes_the_question():
    cache = AnswerCache(size=2)
    k1 = cache.key("fp", "  What IS the   peak? ", "beginner", "ctx")
    assert k1 == cache.key("fp", "what is the peak?", "beginner", "ctx")
    assert k1 != cache.key("fp", "what is the peak?", "analyst", "ctx")
    assert k1 != cache.key("other", "what is the peak?", "beginner", "ctx")
    assert k1 != cache.key("fp", "what is the peak?", "beginner", "ctx2")
    cache.put(k1, "a")
    cache.put(("b",) * 4, "b")
    cache.get(k1)
    cache.put(("c",) * 4, "c")
    assert len(cache) == 2 and cache.get(k1) == "a" and cache.get(("b",) * 4) is None


def test_a_session_starts_with_the_configured_limits():
    s = AISession()
    assert s.guard.question_cap == config.SESSION_QUESTION_CAP == 20
    assert config.MAX_REPAIR_RETRIES == 1 and len(s.cache) == 0
