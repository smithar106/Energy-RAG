"""LLM provider interface + DeepSeek implementation.

DeepSeek exposes an OpenAI-compatible API. We use the ``openai`` Python SDK
purely as a *client* for that endpoint — ``base_url`` is hard-wired to
``https://api.deepseek.com`` and no request ever reaches OpenAI.

The provider is abstracted so a future provider (any OpenAI-compatible host, or
a native client) can be added without touching the agent, synthesis, or
orchestration layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from openai import OpenAI

from app.config import get_settings


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[Any] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class LLMProvider(Protocol):
    """Minimal contract the rest of the app depends on."""

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Return the raw assistant response dict (content + optional tool_calls)."""
        ...


class DeepSeekProvider:
    """OpenAI-compatible client pointed explicitly at DeepSeek."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        self.model = settings.deepseek_model
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> dict:
        payload: list[dict[str, Any]] = []
        for m in messages:
            entry: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.tool_calls:
                entry["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                entry["tool_call_id"] = m.tool_call_id
            if m.name:
                entry["name"] = m.name
            payload.append(entry)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        return {
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in (message.tool_calls or [])
            ],
            "finish_reason": response.choices[0].finish_reason,
        }


def get_llm_provider() -> LLMProvider:
    return DeepSeekProvider()
