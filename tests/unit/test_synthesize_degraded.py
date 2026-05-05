from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.graph.nodes import synthesize
from src.retrieval.index.types import Hit


class _NoCitationClient:
    def __init__(self) -> None:
        self.calls = 0
        self.messages = []

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        self.messages.append(args[0])
        return ChatResponse(
            content="This answer has prose but no machine-readable citation.",
            model="fake-main",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_synthesize_degrades_without_raising_after_two_no_citation_outputs(monkeypatch) -> None:
    client = _NoCitationClient()
    monkeypatch.setattr(synthesize, "get_chat_client", lambda name: client)

    updates = await synthesize.synthesize_node(
        {
            "question": "What is ReAct?",
            "evidence": [_hit()],
            "messages": [],
        }
    )

    assert client.calls == 2
    assert "MANDATORY CITATION RULE" in client.messages[0][0].content
    assert "previous answer is missing required" in client.messages[1][0].content
    assert updates["answer"] == "This answer has prose but no machine-readable citation."
    assert updates["citations"] == []
    assert updates["synthesis_format_degraded"] is True


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:2210.03629",
        section="Introduction",
        text="ReAct interleaves reasoning and acting.",
        score=1.0,
    )
