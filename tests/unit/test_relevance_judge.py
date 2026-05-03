from __future__ import annotations

import asyncio

import pytest

from src.core.types import RelevanceVerdict
from src.core.types import ChatResponse
from src.retrieval.index.types import Hit
from src.retrieval.self_rag import relevance_judge


@pytest.mark.asyncio
async def test_judge_batch_is_semaphore_bounded(monkeypatch) -> None:
    active = 0
    max_active = 0

    async def fake_judge(subq: str, hit: Hit) -> RelevanceVerdict:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return RelevanceVerdict(relevant=True, rationale=str(hit.chunk_id))

    monkeypatch.setattr(relevance_judge, "judge_relevance", fake_judge)
    hits = [
        Hit(chunk_id=idx, paper_id="p", section="S", text="text long enough", score=1.0)
        for idx in range(20)
    ]

    verdicts = await relevance_judge.judge_batch("subq", hits)

    assert len(verdicts) == 20
    assert max_active <= 8


@pytest.mark.asyncio
async def test_judge_relevance_repairs_invalid_json(monkeypatch) -> None:
    client = _RepairingChatClient()
    monkeypatch.setattr(relevance_judge, "get_chat_client", lambda name: client)

    verdict = await relevance_judge.judge_relevance(
        "What is ReAct?",
        Hit(chunk_id=1, paper_id="p", section="S", text="ReAct text", score=1.0),
    )

    assert verdict.relevant is True
    assert client.calls == 2


@pytest.mark.asyncio
async def test_judge_relevance_keeps_hit_when_repair_is_invalid(monkeypatch) -> None:
    client = _BrokenRepairChatClient()
    monkeypatch.setattr(relevance_judge, "get_chat_client", lambda name: client)

    verdict = await relevance_judge.judge_relevance(
        "What is ReAct?",
        Hit(chunk_id=1, paper_id="p", section="S", text="ReAct text", score=1.0),
    )

    assert verdict.relevant is True
    assert "invalid JSON twice" in verdict.rationale


class _RepairingChatClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(content='{"relevant": true, "rationale": "truncated', model="fake", finish_reason="length")
        return ChatResponse(content='{"relevant": true, "rationale": "fixed"}', model="fake", finish_reason="stop")


class _BrokenRepairChatClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(content="", model="fake", finish_reason="length")
