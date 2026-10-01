from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.retrieval.query import decomposer


def test_detects_common_comparison_entities() -> None:
    assert decomposer._detect_comparison_entities("How does Toolformer differ from ToolLLM for tool use?") == [
        "Toolformer",
        "ToolLLM",
    ]
    assert decomposer._detect_comparison_entities("Compare ReAct and Reflexion") == ["ReAct", "Reflexion"]
    assert decomposer._detect_comparison_entities("How do RAG and Self-RAG differ in how they use retrieval?") == [
        "RAG",
        "Self-RAG",
    ]
    assert decomposer._detect_comparison_entities("What is ReAct?") == []


@pytest.mark.asyncio
async def test_comparison_decomposition_retries_then_falls_back_when_entity_missing(monkeypatch) -> None:
    client = _MissingEntityClient()
    monkeypatch.setattr(decomposer, "get_chat_client", lambda name: client)

    result = await decomposer.decompose("How does Toolformer differ from ToolLLM for tool use?")

    assert client.calls == 2
    texts = [subq.text for subq in result.sub_questions]
    assert texts == ["What is Toolformer? Focus on: tool use?", "What is ToolLLM? Focus on: tool use?"]
    assert all(not subq.depends_on for subq in result.sub_questions)


class _MissingEntityClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            content='{"sub_questions":[{"text":"What is Toolformer?","depends_on":[]}]}',
            model="fake-main",
            finish_reason="stop",
        )
