from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.graph.nodes import synthesize
from src.retrieval.index.types import Hit


class _NoCitationClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
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
    assert updates["answer"]
    assert updates["citations"] == []
    assert updates["synthesis_format_degraded"] is True


@pytest.mark.asyncio
async def test_synthesize_allows_bracketed_prose_when_valid_citation_exists(monkeypatch) -> None:
    class BracketedProseClient:
        async def chat(self, *args, **kwargs) -> ChatResponse:
            return ChatResponse(
                content="ReAct [Wei et al.] interleaves reasoning and acting [arxiv:2210.03629#1].",
                model="fake-main",
                finish_reason="stop",
            )

    monkeypatch.setattr(synthesize, "get_chat_client", lambda name: BracketedProseClient())

    updates = await synthesize.synthesize_node({"question": "What is ReAct?", "evidence": [_hit()], "messages": []})

    assert len(updates["citations"]) == 1
    assert updates["synthesis_format_degraded"] is False


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:2210.03629",
        section="Introduction",
        text="ReAct interleaves reasoning and acting.",
        score=1.0,
    )
