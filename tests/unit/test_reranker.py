from __future__ import annotations

from dataclasses import dataclass

import pytest
import json

from src.retrieval.index.types import Hit
from src.retrieval.postprocess import reranker


@dataclass(frozen=True)
class _RerankResult:
    index: int
    score: float


class _FakeRerankClient:
    async def rerank(self, query: str, documents: list[str], *, top_n: int):
        assert query == "react"
        assert [json.loads(doc)["content"] for doc in documents] == ["first", "second"]
        assert all(json.loads(doc)["section"] == "S" for doc in documents)
        assert top_n == 2
        return [_RerankResult(index=1, score=0.91), _RerankResult(index=0, score=0.42)]


@pytest.mark.asyncio
async def test_reranker_overwrites_score_with_relevance_score(monkeypatch) -> None:
    monkeypatch.setattr(reranker, "get_rerank_client", lambda name: _FakeRerankClient())
    hits = [
        Hit(chunk_id=1, paper_id="p1", section="S", text="first", score=0.01),
        Hit(chunk_id=2, paper_id="p2", section="S", text="second", score=0.02),
    ]

    reranked = await reranker.rerank("react", hits, top_n=2)

    assert [hit.chunk_id for hit in reranked] == [2, 1]
    assert [hit.score for hit in reranked] == [0.91, 0.42]
