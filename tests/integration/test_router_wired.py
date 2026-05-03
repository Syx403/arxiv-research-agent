from __future__ import annotations

import os

import pytest

from src.core.types import Decomposition, GraphExpansion, RelevanceVerdict, RouteDecision, SubQuestion, SufficiencyVerdict
from src.graph.nodes import retrieve
from src.retrieval.index.types import Hit


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="router wiring integration test requires INTEGRATION_TESTS=1",
)


@pytest.mark.asyncio
async def test_retrieve_skips_graph_retriever_when_router_disables_it(monkeypatch) -> None:
    calls = await _run_retrieve_with_route(monkeypatch, RouteDecision(use_concept_graph=False))
    assert calls == 0


@pytest.mark.asyncio
async def test_retrieve_calls_graph_retriever_when_router_enables_it(monkeypatch) -> None:
    calls = await _run_retrieve_with_route(monkeypatch, RouteDecision(use_concept_graph=True))
    assert calls >= 1


async def _run_retrieve_with_route(monkeypatch, route_decision: RouteDecision) -> int:
    graph_calls = 0

    async def fake_route(_subq: str) -> RouteDecision:
        return route_decision

    async def fake_expand(_subq: str) -> GraphExpansion:
        nonlocal graph_calls
        graph_calls += 1
        return GraphExpansion(extra_paper_hints=["arxiv:2210.03629"])

    async def fake_rewrite(subq: str) -> str:
        return subq

    async def fake_hyde(subq: str) -> str:
        return subq

    async def fake_hybrid_search(*args, **kwargs) -> list[Hit]:
        return [_hit(score=1.0)]

    async def fake_rerank(query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return hits[:top_n]

    async def fake_judge_batch(subq: str, hits: list[Hit]) -> list[RelevanceVerdict]:
        return [RelevanceVerdict(relevant=True, rationale="matched") for _ in hits]

    async def fake_sufficient(subq: str, hits: list[Hit]) -> SufficiencyVerdict:
        return SufficiencyVerdict(sufficient=True, missing_aspects=[])

    monkeypatch.setattr(retrieve.router, "route", fake_route)
    monkeypatch.setattr(retrieve.graph_retriever, "expand_query_via_graph", fake_expand)
    monkeypatch.setattr(retrieve.rewriter, "rewrite_for_retrieval", fake_rewrite)
    monkeypatch.setattr(retrieve.rewriter, "should_use_hyde", lambda subq: False)
    monkeypatch.setattr(retrieve.rewriter, "hyde", fake_hyde)
    monkeypatch.setattr(retrieve, "get_embedding_client", lambda name: _FakeEmbeddingClient())
    monkeypatch.setattr(retrieve, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(retrieve.deduplicator, "dedupe", lambda hits: hits)
    monkeypatch.setattr(retrieve.reranker, "rerank", fake_rerank)
    monkeypatch.setattr(retrieve.relevance_judge, "judge_batch", fake_judge_batch)
    monkeypatch.setattr(retrieve.sufficiency_check, "is_sufficient", fake_sufficient)

    state = {
        "decomposition": Decomposition(sub_questions=[SubQuestion(text="What is ReAct?")]),
        "subq_results": {},
        "multi_hop_calls": 0,
    }
    await retrieve.retrieve_node(state)
    return graph_calls


class _FakeEmbeddingClient:
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.01] * 1536 for _ in texts]


def _hit(*, score: float) -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:2210.03629",
        section="Introduction",
        text="ReAct interleaves reasoning and acting.",
        score=score,
    )
