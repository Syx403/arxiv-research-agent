from __future__ import annotations

import asyncio

import pytest

from src.core.types import RelevanceVerdict
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
