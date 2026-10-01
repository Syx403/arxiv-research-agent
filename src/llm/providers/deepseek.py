from __future__ import annotations

import inspect
import json
import os
from typing import Any

import httpx

from src.core.config import get_settings, require_secret
from src.core.credentials import read_deepseek_key
from src.core.trace import record_event
from src.llm.stages import current_stage
from src.core.types import ChatResponse, LLM_REQUEST_TIMEOUT_SECONDS, Message, TokenUsage, ToolCall
from src.llm.errors import error_for_status
from src.llm.budget import admit_request
from src.llm.retry import with_retry


class DeepSeekChatClient:
    provider = "deepseek_native"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or read_deepseek_key() or require_secret(
            settings.deepseek_api_key,
            env_var="DEEPSEEK_API_KEY",
            provider=self.provider,
        )
        self._client = http_client or httpx.AsyncClient(
            base_url="https://api.deepseek.com/v1",
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        raise error_for_status(self.provider, response.status_code, response.text, headers=response.headers)

    @with_retry()
    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        reasoning_effort: str | None = None,
        thinking: bool | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_wire_message(message) for message in messages],
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if thinking is False:
            payload["thinking"] = {"type": "disabled"}
        elif reasoning_effort is not None:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = reasoning_effort
            payload.pop("temperature", None)  # Ignored in DeepSeek thinking mode.

        reservation = await admit_request(self.provider, model, payload, max_tokens or 32768)
        try:
            response = await self._client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
            )
            self._raise_for_status(response)
            data = response.json()
            if reservation and data.get("usage"):
                reservation.settle(data["usage"])
            choice = (data.get("choices") or [{}])[0]
            record_event("model_response", model=model, returned_model=data.get("model"),
                         stage=current_stage(), reasoning_effort=reasoning_effort,
                         max_tokens=max_tokens, usage=data.get("usage"),
                         finish_reason=choice.get("finish_reason"),
                         thinking=payload.get("thinking", {}).get("type"),
                         content_length=len((choice.get("message") or {}).get("content") or ""))
            return _parse_chat_response(data, fallback_model=model)
        finally:
            if reservation:
                reservation.uncertain()

    async def aclose(self) -> None:
        result = self._client.aclose()
        if inspect.isawaitable(result):
            await result


def _parse_chat_response(data: dict[str, Any], *, fallback_model: str) -> ChatResponse:
    choice = (data.get("choices") or [{}])[0]
    raw_message = choice.get("message") or {}
    raw_tool_calls = raw_message.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            id=item.get("id", ""),
            name=(item.get("function") or {}).get("name", item.get("name", "")),
            arguments=_tool_arguments((item.get("function") or {}).get("arguments", item.get("arguments", {}))),
        )
        for item in raw_tool_calls
    ]
    usage = data.get("usage")
    parsed_usage = TokenUsage(**usage) if usage else None
    return ChatResponse(
        content=raw_message.get("content"),
        tool_calls=tool_calls,
        usage=parsed_usage,
        model=data.get("model") or fallback_model,
        finish_reason=choice.get("finish_reason") or "",
        reasoning_content=raw_message.get("reasoning_content"),
    )


def _tool_arguments(value) -> dict:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must decode to an object.")
    return parsed


def _wire_message(message: Message) -> dict:
    result = message.model_dump(exclude_none=True)
    if message.tool_calls:
        result["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {
                "name": call.name, "arguments": json.dumps(call.arguments),
            }} for call in message.tool_calls
        ]
    return result
