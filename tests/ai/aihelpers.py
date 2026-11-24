"""Helpers for the AI tests: a scripted model and a fake network transport."""

from __future__ import annotations

from typing import Any

from datanaut.ai.llm import LLMResponse, Message, ToolCall, ToolSpec

KEY = "sk-test-SECRET-key-0123456789"


def tool(name: str, **arguments: Any) -> LLMResponse:
    return LLMResponse("", [ToolCall(name, arguments)], "scripted")


def text(content: str) -> LLMResponse:
    return LLMResponse(content, [], "scripted")


class ScriptedClient:
    """A stand-in model that returns prepared responses and records what it was sent."""

    name = "scripted"

    def __init__(self, *responses: LLMResponse | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec], mode: str
    ) -> LLMResponse:
        self.calls.append(
            {"system": system, "messages": list(messages), "tools": tools, "mode": mode}
        )
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def prompts(self) -> str:
        """Everything sent so far, as one string."""
        return "\n".join(
            c["system"] + "\n" + "\n".join(m.content for m in c["messages"]) for c in self.calls
        )


class FakeTransport:
    """A stand-in for the network: scripted (status, payload) replies or exceptions."""

    def __init__(self, *replies: tuple[int, Any] | Exception) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
    ) -> tuple[int, Any]:
        self.calls.append({"url": url, "headers": dict(headers), "body": body, "timeout": timeout})
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
