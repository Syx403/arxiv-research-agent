from __future__ import annotations

import httpx
import pytest

from src.core.types import ChatResponse, Message
from src.llm.client import _RoutedChatClient
from src.llm.registry import ProviderRoute


def _speed_up_network_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_INITIAL_SECONDS", 0.01)
    monkeypatch.setattr("src.llm.retry.RETRY_WAIT_MAX_SECONDS", 0.05)


@pytest.mark.asyncio
async def test_chat_retries_read_timeout_twice_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_network_retry(monkeypatch)
    provider = _FlakyProvider(
        [
            httpx.ReadTimeout("read timed out"),
            httpx.ReadTimeout("read timed out"),
            ChatResponse(content="ok", model="fake", finish_reason="stop"),
        ]
    )
    client = _client(provider)

    response = await client.chat([Message(role="user", content="hello")])

    assert response.content == "ok"
    assert provider.calls == 3


@pytest.mark.asyncio
async def test_chat_raises_when_read_timeout_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_network_retry(monkeypatch)
    provider = _FlakyProvider([httpx.ReadTimeout("read timed out") for _ in range(4)])
    client = _client(provider)

    with pytest.raises(httpx.ReadTimeout):
        await client.chat([Message(role="user", content="hello")])

    assert provider.calls == 3


@pytest.mark.asyncio
async def test_chat_retries_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _speed_up_network_retry(monkeypatch)
    provider = _FlakyProvider(
        [
            httpx.ConnectError("connection failed"),
            ChatResponse(content="ok", model="fake", finish_reason="stop"),
        ]
    )
    client = _client(provider)

    response = await client.chat([Message(role="user", content="hello")])

    assert response.content == "ok"
    assert provider.calls == 2


def _client(provider) -> _RoutedChatClient:
    return _RoutedChatClient(
        role="main",
        route=ProviderRoute("deepseek_native", "fake-model", "chat"),
        provider_client=provider,
    )


class _FlakyProvider:
    def __init__(self, outcomes: list[ChatResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        return None
