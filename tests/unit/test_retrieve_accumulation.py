from __future__ import annotations

import pytest

from src.core.types import RelevanceVerdict, RouteDecision, SufficiencyVerdict
from src.graph.nodes import retrieve
from src.retrieval.index.types import Hit


@pytest.mark.asyncio
async def test_retrieve_retry_accumulates_relevant_hits_by_chunk_id(monkeypatch) -> None:
    hybrid_calls = 0
    sufficiency_calls = 0

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
        verdicts_by_call = {
            1: [True, False],
            2: [True, True],
            3: [False, True],
        }
        return [
            RelevanceVerdict(relevant=relevant, rationale="ok")
            for relevant in verdicts_by_call[hybrid_calls]
        ]

    async def fake_sufficient(subq: str, hits: list[Hit]) -> SufficiencyVerdict:
        nonlocal sufficiency_calls
        sufficiency_calls += 1
        return SufficiencyVerdict(sufficient=sufficiency_calls == 3, missing_aspects=["more"])

    monkeypatch.setattr(retrieve.router, "route", fake_route)
    monkeypatch.setattr(retrieve.rewriter, "rewrite_for_retrieval", fake_rewrite)
    monkeypatch.setattr(retrieve.rewriter, "should_use_hyde", lambda subq: False)
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
    )

    assert [hit.chunk_id for hit in result.hits] == [1, 2, 3, 4]
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
async def test_multihop_reretrieval_keeps_previously_relevant_evidence(monkeypatch):
    from src.core.types import Decomposition, GraphExpansion, SubQuestion, SubQResult

    original = _hit(1)
    route = RouteDecision(may_use_semantic_scholar_live=True)

    async def pipeline(subq, *, subq_index, multi_hop_used, initial_hits):
        assert multi_hop_used is True
        assert list(initial_hits) == [original]
        return SubQResult(
            subq_index=0,
            hits=list(initial_hits),
            sufficient=False,
            route_decision=route,
            multi_hop_used=True,
        ), GraphExpansion()

    monkeypatch.setattr(retrieve, "_run_subquestion_pipeline", pipeline)
    updates = await retrieve.retrieve_node(
        {
            "decomposition": Decomposition(sub_questions=[SubQuestion(text="Question")]),
            "subq_results": {
                0: SubQResult(
                    subq_index=0,
                    hits=[original],
                    sufficient=False,
                    route_decision=route,
                    multi_hop_used=False,
                )
            },
            "active_subq_index": 0,
            "multi_hop_calls": 1,
        }
    )
    assert updates["evidence"] == [original]
