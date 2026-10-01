from __future__ import annotations

import logging

import pytest

from src.core.types import GraphExpansion, RelevanceVerdict, RouteDecision, SufficiencyVerdict
from src.graph.nodes import retrieve
from src.retrieval.index.types import Hit


@pytest.mark.asyncio
async def test_retrieve_pipeline_logs_each_stage(monkeypatch, caplog) -> None:
    async def fake_route(_subq: str) -> RouteDecision:
        return RouteDecision(use_concept_graph=True)

    async def fake_expand(_subq: str) -> GraphExpansion:
        return GraphExpansion(seed_concept_ids=[1], neighbor_concept_ids=[2], extra_paper_hints=["arxiv:paper"])

    async def fake_rewrite(subq: str) -> str:
        return subq

    async def fake_hybrid_search(*args, **kwargs) -> list[Hit]:
        return [_hit()]

    async def fake_rerank(query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return hits[:top_n]

    async def fake_judge_batch(subq: str, hits: list[Hit]) -> list[RelevanceVerdict]:
        return [RelevanceVerdict(relevant=True, rationale="ok") for _ in hits]

    async def fake_sufficient(subq: str, hits: list[Hit], **kwargs) -> SufficiencyVerdict:
        return SufficiencyVerdict(sufficient=True, missing_aspects=[])

    monkeypatch.setattr(retrieve.rewriter, "rewrite_for_retrieval", fake_rewrite)
    monkeypatch.setattr(retrieve, "get_embedding_client", lambda name: _FakeEmbeddingClient())
    monkeypatch.setattr(retrieve, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(retrieve.deduplicator, "dedupe", lambda hits: hits)
    monkeypatch.setattr(retrieve.reranker, "rerank", fake_rerank)
    monkeypatch.setattr(retrieve.relevance_judge, "judge_batch", fake_judge_batch)
    monkeypatch.setattr(retrieve.sufficiency_check, "is_sufficient", fake_sufficient)

    with caplog.at_level(logging.INFO, logger=retrieve.__name__):
        result, graph_expansion = await retrieve._run_subquestion_pipeline(
            "What is ReAct?",
            subq_index=0,
            multi_hop_used=False,
        )

    assert result.hits
    assert graph_expansion.extra_paper_hints == []
    logged_stages = {record.stage for record in caplog.records if record.message == "retrieve_stage"}
    assert {
        "rewrite",
        "embed",
        "hybrid_search",
        "dedupe",
        "rerank",
        "filter_relevant",
        "accumulate_evidence",
        "check_sufficiency",
    } <= logged_stages


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:paper",
        section="Introduction",
        text="chunk text",
        score=1.0,
    )


class _FakeEmbeddingClient:
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.01] * 1536 for _ in texts]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["definition", "comparison"])
async def test_focused_retrieval_skips_speculative_hyde(monkeypatch, kind):
    async def rewrite(question):
        return "ReAct reasoning acting"
    async def forbidden_hyde(question):
        raise AssertionError("Named-method questions can search the rewritten query directly")
    monkeypatch.setattr(retrieve.rewriter, "rewrite_for_retrieval", rewrite)
    ctx = retrieve.RetrievalContext(subq="How does ReAct work with LLM reasoning and actions?", subq_index=0,
                                   question_type=kind, question_id="q", multi_hop_used=False)
    await retrieve._stage_rewrite(ctx)
    assert ctx.embedding_text == ctx.rewritten_query == "ReAct reasoning acting"


def test_ui_progress_reports_the_final_evidence_pack(monkeypatch):
    events = []
    monkeypatch.setattr(retrieve, 'record_event', lambda name, **values: events.append(values))
    ctx = retrieve.RetrievalContext(subq='What is ReAct?', subq_index=0, question_type='definition',
                                   question_id='q', multi_hop_used=False, hits=[_hit()],
                                   accumulated_hits={1: _hit(), 2: Hit(2, 'other', 'Related work', 'A passing mention.', .1)})
    retrieve._log_stage(ctx, 'check_sufficiency')
    assert events[0]['accepted_chunks'] == 1
    assert events[0]['papers'] == 1
