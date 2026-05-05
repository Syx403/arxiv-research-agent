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
    assert decomposer._detect_comparison_entities("What is ReAct?") == []


@pytest.mark.asyncio
async def test_comparison_decomposition_retries_then_falls_back_when_entity_missing(monkeypatch) -> None:
    client = _MissingEntityClient()
    monkeypatch.setattr(decomposer, "get_chat_client", lambda name: client)

    result = await decomposer.decompose("How does Toolformer differ from ToolLLM for tool use?")

    assert client.calls == 2
    texts = [subq.text for subq in result.sub_questions]
    assert "What is Toolformer?" in texts
    assert "What is ToolLLM?" in texts
    assert "How do Toolformer and ToolLLM compare?" in texts
    assert result.sub_questions[-1].depends_on == [0, 1]


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
