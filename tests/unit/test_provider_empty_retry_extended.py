from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Message
from src.llm import usage
from src.llm.client import EmptyProviderResponseError, _RoutedChatClient
from src.llm.registry import ProviderRoute


@pytest.fixture(autouse=True)
def reset_usage() -> None:
    usage.reset()


@pytest.mark.asyncio
async def test_empty_then_success_on_second_attempt_uses_original_temperature() -> None:
    provider = _FakeProvider(
        [
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="valid", model="fake", finish_reason="stop"),
        ]
    )

    response = await _client(provider).chat([Message(role="user", content="hello")], temperature=0.2)

    assert response.content == "valid"
    assert provider.temperatures == [0.2, 0.2]
    assert usage.snapshot()["main"]["provider_empty_retry_count"] == 1


@pytest.mark.asyncio
async def test_third_empty_attempt_adds_nudge_and_temperature_floor() -> None:
    provider = _FakeProvider(
        [
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="valid", model="fake", finish_reason="stop"),
        ]
    )

    response = await _client(provider).chat([Message(role="user", content="hello")], temperature=0.0)

    assert response.content == "valid"
    assert provider.temperatures == [0.0, 0.0, 0.3]
    assert "Your previous response was empty" in provider.messages[-1][-1].content
    assert usage.snapshot()["main"]["provider_empty_retry_count"] == 2


@pytest.mark.asyncio
async def test_empty_three_times_raises_and_counts_all_empty_responses() -> None:
    provider = _FakeProvider([ChatResponse(content="", model="fake", finish_reason="stop") for _ in range(3)])

    with pytest.raises(EmptyProviderResponseError):
        await _client(provider).chat([Message(role="user", content="hello")])

    assert provider.calls == 3
    assert usage.snapshot()["main"]["provider_empty_retry_count"] == 3


def _client(provider: "_FakeProvider") -> _RoutedChatClient:
    return _RoutedChatClient(
        role="main",
        route=ProviderRoute("deepseek_native", "fake-model", "chat"),
        provider_client=provider,
    )


class _FakeProvider:
    def __init__(self, responses: list[ChatResponse]) -> None:
        self.responses = responses
        self.calls = 0
        self.temperatures: list[float] = []
        self.messages: list[list[Message]] = []

    async def chat(self, messages: list[Message], *args, **kwargs) -> ChatResponse:
        self.messages.append(messages)
        self.temperatures.append(kwargs["temperature"])
        response = self.responses[self.calls]
        self.calls += 1
        return response

    async def aclose(self) -> None:
        return None
