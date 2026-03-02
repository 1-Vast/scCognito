from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    type: str
    function: _Fn


class _Msg:
    def __init__(self, content: str, tool_calls: list[_ToolCall] | None):
        self.content = content
        self.tool_calls = tool_calls or None

    def model_dump(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return d


class _Choice:
    def __init__(self, msg: _Msg):
        self.message = msg


class _Resp:
    def __init__(self, msg: _Msg):
        self.choices = [_Choice(msg)]


class ArkChatClient:
    """
    OpenAI-compatible ChatCompletions client (agent side).
    Supports optional streaming with tool-call aggregation.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        timeout: Optional[float] = None,
        enable_streaming: bool = True,
        stream_print: bool = True,
        stream_print_reasoning: bool = True,
    ) -> None:
        self.model_id = model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.enable_streaming = bool(enable_streaming)
        self.stream_print = bool(stream_print)
        self.stream_print_reasoning = bool(stream_print_reasoning)

    def chat_text(self, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 4096) -> str:
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
        if not self.enable_streaming:
            return self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        # Streaming path
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        full_text_parts: list[str] = []
        tool_calls_buf: dict[int, dict[str, Any]] = {}

        if self.stream_print:
            sys.stdout.write("\n[scAgent] streaming started...\n")
            sys.stdout.flush()

        for chunk in stream:
            delta = chunk.choices[0].delta

            # Optional: provider-specific reasoning channel
            reasoning = getattr(delta, "reasoning_content", None)
            if self.stream_print and self.stream_print_reasoning and reasoning:
                sys.stdout.write(str(reasoning))
                sys.stdout.flush()
                full_text_parts.append(str(reasoning))

            if getattr(delta, "content", None):
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                full_text_parts.append(delta.content)

            # Tool calls can arrive incrementally; aggregate by index
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = int(tc.index)
                    buf = tool_calls_buf.setdefault(
                        idx,
                        {
                            "id": tc.id or f"call_{idx}",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            buf["function"]["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            buf["function"]["arguments"] += fn.arguments

        if self.stream_print:
            sys.stdout.write("\n\n[scAgent] streaming finished.\n")
            sys.stdout.flush()

        full_content = "".join(full_text_parts).strip()

        tool_calls: list[_ToolCall] = []
        for idx in sorted(tool_calls_buf.keys()):
            item = tool_calls_buf[idx]
            tool_calls.append(
                _ToolCall(
                    id=str(item["id"]),
                    type=str(item["type"]),
                    function=_Fn(name=str(item["function"]["name"]), arguments=str(item["function"]["arguments"])),
                )
            )

        msg = _Msg(content=full_content, tool_calls=tool_calls if tool_calls else None)
        return _Resp(msg)
