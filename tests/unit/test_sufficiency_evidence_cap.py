from __future__ import annotations

import pytest

from src.core.types import (
    SUFFICIENCY_EVIDENCE_CHARS_PER_CHUNK,
    SUFFICIENCY_EVIDENCE_MAX_CHUNKS,
    ChatResponse,
)
from src.retrieval.index.types import Hit
from src.retrieval.self_rag import sufficiency_check


class _CaptureClient:
    def __init__(self) -> None:
        self.messages = []

    async def chat(self, messages, **kwargs) -> ChatResponse:
        self.messages.append(messages)
        return ChatResponse(
            content='{"sufficient": true, "missing_aspects": []}',
            model="fake-main",
            finish_reason="stop",
        )


class _LengthClient:
    async def chat(self, messages, **kwargs) -> ChatResponse:
        return ChatResponse(
            content='{"sufficient": true',
            model="fake-main",
            finish_reason="length",
        )


@pytest.mark.asyncio
async def test_sufficiency_caps_to_top_five_and_truncates(monkeypatch) -> None:
    client = _CaptureClient()
    monkeypatch.setattr(sufficiency_check, "get_chat_client", lambda name: client)

    verdict = await sufficiency_check.is_sufficient("question", [_hit(index) for index in range(15)])

    assert verdict.sufficient is True
    user_prompt = client.messages[0][1].content
    assert user_prompt.count("[arxiv:test#") == SUFFICIENCY_EVIDENCE_MAX_CHUNKS
    assert "[arxiv:test#14]" in user_prompt
    assert "[arxiv:test#9]" not in user_prompt
    assert "...[truncated from " in user_prompt
    assert len(sufficiency_check._format_evidence(sufficiency_check._select_evidence([_hit(i) for i in range(15)]))) <= (
        SUFFICIENCY_EVIDENCE_MAX_CHUNKS * (SUFFICIENCY_EVIDENCE_CHARS_PER_CHUNK + 160)
    )


@pytest.mark.asyncio
async def test_sufficiency_finish_length_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(sufficiency_check, "get_chat_client", lambda name: _LengthClient())

    verdict = await sufficiency_check.is_sufficient("question", [_hit(1)])

    assert verdict.sufficient is False
    assert verdict.missing_aspects[0].startswith("PARSE_FAILURE")


def _hit(index: int) -> Hit:
    return Hit(
        chunk_id=index,
        paper_id="arxiv:test",
        section="Intro",
        text=f"chunk-{index} " + ("x" * (SUFFICIENCY_EVIDENCE_CHARS_PER_CHUNK + 200)),
        score=float(index),
    )
