from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.retrieval.query import decomposer


@pytest.mark.asyncio
async def test_invalid_json_twice_on_non_comparison_returns_singleton_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeChatClient(["not json", "still not json"])
    monkeypatch.setattr(decomposer, "get_chat_client", lambda name: client)

    result = await decomposer.decompose("Survey how agentic memory systems use retrieval over time.", max_subq=3)

    assert client.calls == 2
    assert [subq.text for subq in result.sub_questions] == [
        "Survey how agentic memory systems use retrieval over time."
    ]


@pytest.mark.asyncio
async def test_invalid_json_twice_on_comparison_uses_comparison_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _FakeChatClient(["not json", "still not json", "bad retry", "bad retry repair"])
    monkeypatch.setattr(decomposer, "get_chat_client", lambda name: client)

    result = await decomposer.decompose("Compare ReAct and Reflexion", max_subq=3)

    assert [subq.text for subq in result.sub_questions] == [
        "What is ReAct?",
        "What is Reflexion?",
    ]


class _FakeChatClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResponse(content=response, model="fake", finish_reason="stop")
