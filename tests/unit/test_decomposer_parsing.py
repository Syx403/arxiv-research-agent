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


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    "What is ReAct? Explain its core mechanism with sources.",
    "Explain how agentic memory systems use retrieval over time.",
    "请解释 ReAct 的核心机制，并给出原论文出处。",
    "What are the limitations of MemGPT's virtual context management? Separate paper evidence from unknowns.",
])
async def test_focused_questions_keep_scope_regardless_of_length(monkeypatch, question):
    def no_planner(role):
        raise AssertionError("A focused request needs no extra planning call")
    monkeypatch.setattr(decomposer, "get_chat_client", no_planner)
    result = await decomposer.decompose(question)
    assert [q.text for q in result.sub_questions] == [question]


def test_short_complex_requests_do_not_take_the_simple_path():
    assert not decomposer._is_trivial_question("Survey agent memory")
    assert not decomposer._is_trivial_question("比较 ReAct 和 Reflexion")
    assert not decomposer._is_trivial_question("调研 Agent 最新进展")
    assert not decomposer._is_trivial_question("What are the differences between ReAct and Reflexion?")
    assert not decomposer._is_trivial_question("How is ReAct different from Reflexion?")
    assert decomposer._detect_comparison_entities(
        "Compare ReAct and Reflexion: what changes in the agent loop?"
    ) == ["ReAct", "Reflexion"]
