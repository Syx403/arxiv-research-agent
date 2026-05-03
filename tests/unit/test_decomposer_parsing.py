from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.retrieval.query import decomposer


class _FakeChatClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return ChatResponse(content=response, model="fake", finish_reason="stop")


@pytest.mark.asyncio
async def test_decompose_repairs_invalid_json_once(monkeypatch) -> None:
    client = _FakeChatClient(
        [
            "not json",
            '{"sub_questions":[{"text":"Find ReAct mechanism","depends_on":[]},{"text":"Compare Reflexion","depends_on":[0]}]}',
        ]
    )
    monkeypatch.setattr(decomposer, "get_chat_client", lambda name: client)

    result = await decomposer.decompose(
        "Explain how ReAct works. Then compare it with Reflexion.",
        max_subq=2,
    )

    assert client.calls == 2
    assert [subq.text for subq in result.sub_questions] == ["Find ReAct mechanism", "Compare Reflexion"]
    assert result.sub_questions[1].depends_on == [0]


@pytest.mark.asyncio
async def test_decompose_bypasses_llm_for_trivial_question(monkeypatch) -> None:
    def fail_get_client(name: str):
        raise AssertionError("trivial questions should not call the LLM")

    monkeypatch.setattr(decomposer, "get_chat_client", fail_get_client)

    result = await decomposer.decompose("What is ReAct?")

    assert len(result.sub_questions) == 1
    assert result.sub_questions[0].text == "What is ReAct?"
