from __future__ import annotations

from typing import Any, Optional
from openai import OpenAI


class ArkChatClient:
    """OpenAI-compatible chat.completions client (same style as teacher)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout: Optional[float] = None,
    ) -> None:
        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def chat_text(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str | dict[str, Any] = "auto",
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Any:
        # If Ark doesn't support tools, runtime will fallback automatically.
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
        )