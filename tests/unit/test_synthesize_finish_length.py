from __future__ import annotations

import pytest

from src.core.types import ChatResponse
from src.graph.nodes import synthesize
from src.retrieval.index.types import Hit


class _LengthClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            content=f"Truncated answer attempt {self.calls} [arxiv",
            model="fake-main",
            finish_reason="length",
        )


@pytest.mark.asyncio
async def test_synthesize_degrades_when_compact_retry_still_hits_length(monkeypatch) -> None:
    client = _LengthClient()
    monkeypatch.setattr(synthesize, "get_chat_client", lambda name: client)

    updates = await synthesize.synthesize_node(
        {
            "question": "What is MemGPT?",
            "evidence": [_hit()],
            "messages": [],
        }
    )

    assert client.calls == 2
    assert updates["answer"] == "Truncated answer attempt 2 [arxiv"
    assert updates["citations"] == []
    assert updates["synthesis_format_degraded"] is True
    assert updates["synthesis_finish_reason"] == "length"


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:2310.08560",
        section="Intro",
        text="MemGPT uses virtual context management.",
        score=1.0,
    )
