from __future__ import annotations

import pytest

from src.core.types import RelevanceVerdict, RouteDecision, SufficiencyVerdict
from src.graph.nodes import retrieve
from src.retrieval.index.types import Hit


@pytest.mark.asyncio
async def test_retrieve_retry_accumulates_relevant_hits_by_chunk_id(monkeypatch) -> None:
    hybrid_calls = 0
    sufficiency_calls = 0
    judged = []

    async def fake_route(_subq: str) -> RouteDecision:
        return RouteDecision()

    async def fake_rewrite(subq: str) -> str:
        return subq

    async def fake_hybrid_search(*args, **kwargs) -> list[Hit]:
        nonlocal hybrid_calls
        hybrid_calls += 1
        if hybrid_calls == 1:
            return [_hit(1), _hit(2)]
        if hybrid_calls == 2:
            return [_hit(2), _hit(3)]
        return [_hit(3), _hit(4)]

    async def fake_rerank(query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return hits

    async def fake_judge_batch(subq: str, hits: list[Hit]) -> list[RelevanceVerdict]:
        judged.extend(hit.chunk_id for hit in hits)
        return [
            RelevanceVerdict(relevant=hit.chunk_id != 2, rationale="ok") for hit in hits
        ]

    async def fake_sufficient(subq: str, hits: list[Hit], **kwargs) -> SufficiencyVerdict:
        nonlocal sufficiency_calls
        sufficiency_calls += 1
        return SufficiencyVerdict(sufficient=sufficiency_calls == 3, missing_aspects=["more"])

    monkeypatch.setattr(retrieve.rewriter, "rewrite_for_retrieval", fake_rewrite)
    monkeypatch.setattr(retrieve, "get_embedding_client", lambda name: _FakeEmbeddingClient())
    monkeypatch.setattr(retrieve, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(retrieve.deduplicator, "dedupe", lambda hits: hits)
    monkeypatch.setattr(retrieve.reranker, "rerank", fake_rerank)
    monkeypatch.setattr(retrieve.relevance_judge, "judge_batch", fake_judge_batch)
    monkeypatch.setattr(retrieve.sufficiency_check, "is_sufficient", fake_sufficient)

    result, _graph_expansion = await retrieve._run_subquestion_pipeline(
        "How does Toolformer differ from ToolLLM?",
        subq_index=0,
        multi_hop_used=False,
        question_type="comparison",
    )

    assert [hit.chunk_id for hit in result.hits] == [1, 3, 4]
    assert judged == [1, 2, 2, 3, 4]  # Reassess when the focus changes; reuse within the same focus.
    assert result.sufficient is True
    assert result.retries_used == 2


def _hit(chunk_id: int) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        paper_id=f"arxiv:paper-{chunk_id}",
        section="Results",
        text=f"chunk {chunk_id}",
        score=1.0 / chunk_id,
    )


class _FakeEmbeddingClient:
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.01] * 1536 for _ in texts]


@pytest.mark.asyncio
async def test_empty_followup_retrieval_preserves_existing_evidence(monkeypatch):
    calls = 0
    original = _hit(1)

    async def rewrite(question):
        return question

    async def search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [original] if calls == 1 else []

    async def rerank(query, hits, **kwargs):
        return hits

    async def judge(query, hits):
        return [RelevanceVerdict(relevant=True, rationale="Supported.") for hit in hits]

    async def sufficient(query, hits, **kwargs):
        return SufficiencyVerdict(sufficient=False, missing_aspects=["missing detail"])

    monkeypatch.setattr(retrieve.rewriter, "rewrite_for_retrieval", rewrite)
    monkeypatch.setattr(retrieve, "get_embedding_client", lambda _: _FakeEmbeddingClient())
    monkeypatch.setattr(retrieve, "hybrid_search", search)
    monkeypatch.setattr(retrieve.reranker, "rerank", rerank)
    monkeypatch.setattr(retrieve.relevance_judge, "judge_batch", judge)
    monkeypatch.setattr(retrieve.sufficiency_check, "is_sufficient", sufficient)
    result, _ = await retrieve._run_subquestion_pipeline(
        "Question", subq_index=0, multi_hop_used=False, max_retries=1)
    assert calls == 2
    assert [hit.chunk_id for hit in result.hits] == [original.chunk_id]
    assert result.sufficient is False
