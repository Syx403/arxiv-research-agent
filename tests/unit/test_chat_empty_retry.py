from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Message
from src.llm.client import EmptyProviderResponseError, _RoutedChatClient
from src.llm.registry import ProviderRoute


@pytest.mark.asyncio
async def test_chat_retries_on_empty_non_length_response() -> None:
    provider = _FakeProvider(
        [
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="valid", model="fake", finish_reason="stop"),
        ]
    )
    client = _RoutedChatClient(
        role="main",
        route=ProviderRoute("deepseek_native", "fake-model", "chat"),
        provider_client=provider,
    )

    response = await client.chat([Message(role="user", content="hello")], temperature=0.0)

    assert provider.calls == 2
    assert provider.temperatures == [0.0, 0.0]
    assert response.content == "valid"


@pytest.mark.asyncio
async def test_chat_raises_after_three_empty_responses() -> None:
    provider = _FakeProvider(
        [
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="", model="fake", finish_reason="stop"),
        ]
    )
    client = _RoutedChatClient(
        role="main",
        route=ProviderRoute("deepseek_native", "fake-model", "chat"),
        provider_client=provider,
    )

    with pytest.raises(EmptyProviderResponseError):
        await client.chat([Message(role="user", content="hello")])

    assert provider.calls == 3


class _FakeProvider:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = responses
        self.calls = 0
        self.temperatures: list[float] = []

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        self.temperatures.append(kwargs["temperature"])
        return self._responses[self.calls - 1]

    async def aclose(self) -> None:
        return None
