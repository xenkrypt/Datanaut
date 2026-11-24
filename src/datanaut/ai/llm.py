"""Provider adapters, fallback, per-session limits and the answer cache.

Both adapters talk to their provider's REST API through an INJECTED transport function, so the
network is never touched in tests. Keys are only ever placed in a request header; they are never
logged, put in an error message or stored. Only metadata (provider, latency, token counts, error
type) is logged.

`generate(system, messages, tools, mode)`: `mode` is "tool" (the model must call one of the tools)
or "text" (the model answers in plain text and no tools are offered).
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from datanaut.ai import config
from datanaut.errors import (
    AIConfigError,
    MalformedResponseError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    SafetyBlockError,
    SessionLimitError,
)

_LOG = logging.getLogger("datanaut.ai")

Mode = Literal["tool", "text"]
Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, Any]]
"""transport(url, headers, json_body, timeout) -> (http_status, parsed_json_or_text)."""

_ALLOWED_HOSTS = frozenset({"generativelanguage.googleapis.com", "api.groq.com"})


# ------------------------------------------------------------------------------------- types


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. `parameters` is a plain JSON schema (object/string/array...)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """What a provider said: text and/or tool calls, and which provider said it."""

    text: str
    tool_calls: list[ToolCall]
    provider: str
    fell_back: bool = False
    latency: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    name: str

    def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec], mode: Mode
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------------- transport


def urllib_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
) -> tuple[int, Any]:
    """The real transport (standard library only). Not used in tests."""
    parts = urlparse(url)
    if parts.scheme != "https" or parts.hostname not in _ALLOWED_HOSTS:
        raise ProviderError("The request was blocked: only the configured providers are allowed.")
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            status, raw = response.status, response.read(2_000_000)
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read(200_000)
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise ProviderTimeoutError() from exc
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


def _check_status(status: int) -> None:
    if status == 429:
        raise RateLimitError()
    if status in (408, 504):
        raise ProviderTimeoutError()
    if status in (401, 403):
        raise ProviderError("The AI provider rejected the API key or the request.")
    if status >= 400:
        raise ProviderError(f"The AI provider reported an error (HTTP {status}).")


def _send(
    transport: Transport, url: str, headers: dict[str, str], body: dict[str, Any], provider: str
) -> tuple[Any, float]:
    started = time.perf_counter()
    try:
        status, payload = transport(url, headers, body, config.TIMEOUT_SECONDS)
    except (RateLimitError, ProviderTimeoutError, ProviderError):
        _LOG.warning("provider=%s result=error", provider)
        raise
    except TimeoutError as exc:
        _LOG.warning("provider=%s result=timeout", provider)
        raise ProviderTimeoutError() from exc
    except Exception as exc:  # any transport failure; the message may hold request details
        _LOG.warning("provider=%s result=%s", provider, type(exc).__name__)
        raise ProviderError("The AI provider could not be reached.") from exc
    latency = time.perf_counter() - started
    try:
        _check_status(status)
    except (RateLimitError, ProviderTimeoutError, ProviderError) as exc:
        _LOG.warning("provider=%s status=%s result=%s", provider, status, type(exc).__name__)
        raise
    if not isinstance(payload, dict):
        raise MalformedResponseError()
    return payload, latency


# ---------------------------------------------------------------------------------- Gemini


class GeminiClient:
    """Google's generateContent REST API, with function calling."""

    name = "gemini"

    def __init__(
        self, api_key: str, transport: Transport, model: str = config.GEMINI_MODEL
    ) -> None:
        if not api_key:
            raise AIConfigError()
        self._key = api_key
        self._transport = transport
        self._model = model

    def __repr__(self) -> str:  # never show the key
        return f"GeminiClient(model={self._model!r})"

    def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec], mode: Mode
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {"role": "user" if m.role == "user" else "model", "parts": [{"text": m.content}]}
                for m in messages
            ],
            "generationConfig": {
                "temperature": config.TEMPERATURE,
                "maxOutputTokens": config.MAX_OUTPUT_TOKENS,
            },
        }
        if mode == "tool" and tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        {"name": t.name, "description": t.description, "parameters": t.parameters}
                        for t in tools
                    ]
                }
            ]
            body["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [t.name for t in tools],
                }
            }
        url = config.GEMINI_URL.format(model=self._model)
        headers = {"Content-Type": "application/json", "x-goog-api-key": self._key}
        payload, latency = _send(self._transport, url, headers, body, self.name)
        return self._parse(payload, latency)

    def _parse(self, payload: dict[str, Any], latency: float) -> LLMResponse:
        feedback = payload.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise SafetyBlockError()
        candidates = payload.get("candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not isinstance(candidates[0], dict)
        ):
            raise MalformedResponseError()
        cand = candidates[0]
        if cand.get("finishReason") in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}:
            raise SafetyBlockError()
        parts = (cand.get("content") or {}).get("parts")
        if not isinstance(parts, list):
            raise MalformedResponseError()
        text, calls = [], []
        for part in parts:
            if not isinstance(part, dict):
                raise MalformedResponseError()
            if isinstance(part.get("text"), str):
                text.append(part["text"])
            call = part.get("functionCall")
            if call is not None:
                if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                    raise MalformedResponseError()
                args = call.get("args") or {}
                if not isinstance(args, dict):
                    raise MalformedResponseError()
                calls.append(ToolCall(call["name"], args))
        usage = payload.get("usageMetadata") or {}
        _LOG.info("provider=gemini latency=%.2f", latency)
        return LLMResponse(
            "".join(text),
            calls,
            self.name,
            latency=latency,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )


# ------------------------------------------------------------------------------------ Groq


class GroqClient:
    """Groq's OpenAI-compatible chat completions API, with tool calling."""

    name = "groq"

    def __init__(self, api_key: str, transport: Transport, model: str = config.GROQ_MODEL) -> None:
        if not api_key:
            raise AIConfigError()
        self._key = api_key
        self._transport = transport
        self._model = model

    def __repr__(self) -> str:  # never show the key
        return f"GroqClient(model={self._model!r})"

    def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec], mode: Mode
    ) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "temperature": config.TEMPERATURE,
            "max_tokens": config.MAX_OUTPUT_TOKENS,
        }
        if mode == "tool" and tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            body["tool_choice"] = "required"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        payload, latency = _send(self._transport, config.GROQ_URL, headers, body, self.name)
        return self._parse(payload, latency)

    def _parse(self, payload: dict[str, Any], latency: float) -> LLMResponse:
        error = payload.get("error")
        if isinstance(error, dict):
            if error.get("code") == "tool_use_failed":
                raise MalformedResponseError()
            raise ProviderError()
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise MalformedResponseError()
        choice = choices[0]
        if choice.get("finish_reason") == "content_filter":
            raise SafetyBlockError()
        message = choice.get("message")
        if not isinstance(message, dict):
            raise MalformedResponseError()
        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
                raise MalformedResponseError()
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError) as exc:
                raise MalformedResponseError() from exc
            if not isinstance(args, dict):
                raise MalformedResponseError()
            calls.append(ToolCall(fn["name"], args))
        usage = payload.get("usage") or {}
        _LOG.info("provider=groq latency=%.2f", latency)
        return LLMResponse(
            message.get("content") or "",
            calls,
            self.name,
            latency=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


# ----------------------------------------------------------------------------------- fallback


class FallbackClient:
    """Try the primary provider; on a rate limit or timeout use the secondary (once)."""

    def __init__(self, primary: LLMClient, secondary: LLMClient | None = None) -> None:
        self._primary = primary
        self._secondary = secondary
        self.name = primary.name

    def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec], mode: Mode
    ) -> LLMResponse:
        try:
            return self._primary.generate(system, messages, tools, mode)
        except (RateLimitError, ProviderTimeoutError) as exc:
            if self._secondary is None:
                raise
            _LOG.warning(
                "provider=%s result=%s fallback=%s",
                self._primary.name,
                type(exc).__name__,
                self._secondary.name,
            )
        response = self._secondary.generate(system, messages, tools, mode)
        return replace(response, fell_back=True)


def make_client(
    keys: Mapping[str, str],
    transport: Transport,
    primary: str = "gemini",
) -> LLMClient:
    """A client from the keys that are present: the chosen provider first, the other as fallback."""
    builders = {"gemini": GeminiClient, "groq": GroqClient}
    order = [primary] + [p for p in config.PROVIDERS if p != primary]
    clients = [builders[p](keys[p], transport) for p in order if keys.get(p)]
    if not clients:
        raise AIConfigError()
    return FallbackClient(clients[0], clients[1] if len(clients) > 1 else None)


# ------------------------------------------------------------------------ limits and the cache


class SessionGuard:
    """Question cap and a minimum gap between questions, for one session."""

    def __init__(
        self,
        question_cap: int = config.SESSION_QUESTION_CAP,
        min_interval: float = config.MIN_SECONDS_BETWEEN_CALLS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.question_cap = question_cap
        self.min_interval = min_interval
        self._clock = clock
        self.questions_used = 0
        self._last = float("-inf")

    @property
    def remaining(self) -> int:
        return max(0, self.question_cap - self.questions_used)

    def begin_question(self) -> None:
        if self.questions_used >= self.question_cap:
            raise SessionLimitError(
                f"You have used all {self.question_cap} AI questions for this session."
            )
        wait = self.min_interval - (self._clock() - self._last)
        if wait > 0:
            raise SessionLimitError(f"Please wait {wait:.0f} more second(s) before asking again.")
        self.questions_used += 1
        self._last = self._clock()


class CallBudget:
    """At most `limit` model calls for one question (a repair and a rewrite are included)."""

    def __init__(self, limit: int = config.MAX_CALLS_PER_QUESTION) -> None:
        self.limit = limit
        self.used = 0

    def spend(self) -> None:
        if self.used >= self.limit:
            raise SessionLimitError("This question needed too many model calls and was stopped.")
        self.used += 1


class BudgetedClient:
    """Wraps a client so every call is counted against one question's budget."""

    def __init__(self, client: LLMClient, budget: CallBudget) -> None:
        self._client = client
        self._budget = budget
        self.name = client.name

    def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec], mode: Mode
    ) -> LLMResponse:
        self._budget.spend()
        return self._client.generate(system, messages, tools, mode)


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class AnswerCache:
    """A small bounded cache of answers, keyed by data, question, mode and context."""

    def __init__(self, size: int = config.CACHE_SIZE) -> None:
        self._size = max(1, size)
        self._items: OrderedDict[tuple[str, str, str, str], Any] = OrderedDict()

    @staticmethod
    def key(
        fingerprint: str, question: str, mode: str, context_hash: str
    ) -> tuple[str, str, str, str]:
        return (fingerprint, normalize_question(question), mode, context_hash)

    def get(self, key: tuple[str, str, str, str]) -> Any | None:
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        return None

    def put(self, key: tuple[str, str, str, str], value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._size:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class AISession:
    """The per-session state: the question guard and the answer cache."""

    guard: SessionGuard = field(default_factory=SessionGuard)
    cache: AnswerCache = field(default_factory=AnswerCache)
