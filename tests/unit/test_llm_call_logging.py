from __future__ import annotations

import logging

import pytest

from src.core.types import ChatResponse, Message, TokenUsage
from src.llm.client import _RoutedChatClient
from src.llm.registry import ProviderRoute


@pytest.mark.asyncio
async def test_chat_logs_llm_call_with_role_tokens_and_finish_reason(caplog) -> None:
    client = _RoutedChatClient(
        role="main",
        route=ProviderRoute("deepseek_native", "fake-model", "chat"),
        provider_client=_FakeProvider(),
    )

    with caplog.at_level(logging.INFO, logger="src.llm.client"):
        await client.chat([Message(role="user", content="hello")])

    records = [record for record in caplog.records if record.message == "llm_call"]
    assert len(records) == 1
    assert records[0].role == "main"
    assert records[0].prompt_tokens == 12
    assert records[0].completion_tokens == 5
    assert records[0].finish_reason == "stop"


class _FakeProvider:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(
            content="valid",
            model="fake",
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=12, completion_tokens=5, total_tokens=17),
        )

    async def aclose(self) -> None:
        return None
