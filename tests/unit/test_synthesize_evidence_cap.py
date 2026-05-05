from __future__ import annotations

import pytest

from src.core.types import (
    SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK,
    SYNTHESIS_EVIDENCE_MAX_CHUNKS,
    ChatResponse,
)
from src.graph.nodes import synthesize
from src.retrieval.index.types import Hit


class _CaptureClient:
    def __init__(self) -> None:
        self.messages = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.messages.append(messages)
        return ChatResponse(
            content="The strongest evidence supports the claim [arxiv:top#19].",
            model="fake-main",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_synthesize_caps_and_truncates_prompt_evidence(monkeypatch) -> None:
    client = _CaptureClient()
    monkeypatch.setattr(synthesize, "get_chat_client", lambda name: client)

    await synthesize.synthesize_node(
        {
            "question": "What matters?",
            "evidence": [_hit(index) for index in range(20)],
            "messages": [],
        }
    )

    user_prompt = client.messages[0][1].content
    assert user_prompt.count("(section=Intro)") == SYNTHESIS_EVIDENCE_MAX_CHUNKS
    assert "[arxiv:top#19]" in user_prompt
    assert "[arxiv:top#11]" not in user_prompt
    assert "...[truncated from " in user_prompt
    assert len(synthesize._format_evidence(synthesize._select_synthesis_evidence([_hit(i) for i in range(20)]))) <= (
        SYNTHESIS_EVIDENCE_MAX_CHUNKS * (SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK + 180)
    )


def _hit(index: int) -> Hit:
    return Hit(
        chunk_id=index,
        paper_id="arxiv:top",
        section="Intro",
        text=f"chunk-{index} " + ("x" * (SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK + 200)),
        score=float(index),
    )
