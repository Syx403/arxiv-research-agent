from __future__ import annotations

import inspect
from typing import Any

import httpx

from src.core.config import get_settings, require_secret
from src.core.types import ChatResponse, LLM_REQUEST_TIMEOUT_SECONDS, Message, TokenUsage, ToolCall
from src.llm.errors import error_for_status
from src.llm.retry import with_retry


class OpenRouterChatClient:
    provider = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or require_secret(
            settings.openrouter_api_key,
            env_var="OPENROUTER_API_KEY",
            provider=self.provider,
        )
        self._http_referer = settings.openrouter_http_referer
        self._x_title = settings.openrouter_x_title
        self._client = http_client or httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
        )

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    def _headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._x_title:
            headers["X-Title"] = self._x_title
        return headers

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
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_dict(message) for message in messages],
            "temperature": temperature,
        }
        if tools is not None:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        response = await self._client.post(
            "/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        self._raise_for_status(response)
        return _parse_chat_response(response.json(), fallback_model=model)

    async def aclose(self) -> None:
        result = self._client.aclose()
        if inspect.isawaitable(result):
            await result


def _message_to_dict(message: Message) -> dict[str, Any]:
    return message.model_dump(exclude_none=True)


def _parse_chat_response(data: dict[str, Any], *, fallback_model: str) -> ChatResponse:
    choice = (data.get("choices") or [{}])[0]
    raw_message = choice.get("message") or {}
    raw_tool_calls = raw_message.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            id=item.get("id", ""),
            name=(item.get("function") or {}).get("name", item.get("name", "")),
            arguments=(item.get("function") or {}).get("arguments", item.get("arguments", {})),
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
    )
