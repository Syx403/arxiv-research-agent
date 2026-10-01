from __future__ import annotations

import pytest

from src.core.types import (
    Decomposition,
    RelevanceVerdict,
    RouteDecision,
    SubQuestion,
    SufficiencyVerdict,
)
from src.graph.nodes import retrieve
from src.retrieval.index.types import Hit


@pytest.mark.asyncio
async def test_no_new_evidence_stops_early_without_claiming_sufficiency(monkeypatch) -> None:
    _patch_common_retrieve_dependencies(monkeypatch)

    async def fake_sufficient(subq: str, hits: list[Hit], **kwargs) -> SufficiencyVerdict:
        return SufficiencyVerdict(
            sufficient=False,
            missing_aspects=["PARSE_FAILURE: sufficiency check returned invalid JSON; treating as insufficient."],
        )

    monkeypatch.setattr(retrieve.sufficiency_check, "is_sufficient", fake_sufficient)
    state = _state()

    updates = await retrieve.retrieve_node(state)
    next_state = {**state, **updates}
    result = next_state["subq_results"][0]

    assert result.sufficient is False
    assert result.retries_used == 1
    assert result.stop_reason == "no_new_evidence"


@pytest.mark.asyncio
async def test_fail_closed_relevance_drops_hit_from_kept_hits(monkeypatch) -> None:
    _patch_common_retrieve_dependencies(monkeypatch, patch_judge=False)

    async def fake_sufficient(subq: str, hits: list[Hit], **kwargs) -> SufficiencyVerdict:
        return SufficiencyVerdict(sufficient=True, missing_aspects=[])

    monkeypatch.setattr(retrieve.relevance_judge, "get_chat_client", lambda name: _BrokenChatClient())
    monkeypatch.setattr(retrieve.sufficiency_check, "is_sufficient", fake_sufficient)

    updates = await retrieve.retrieve_node(_state())
    result = updates["subq_results"][0]

    assert result.hits == []
    assert result.sufficient is False
    assert result.stop_reason == "relevance_judgment_failed"
    assert result.judgment_failures == [1]
    assert result.unassessed_hits[0].chunk_id == 1
    assert result.retries_used == 0


def _patch_common_retrieve_dependencies(monkeypatch, *, patch_judge: bool = True) -> None:
    async def fake_route(_subq: str) -> RouteDecision:
        return RouteDecision(may_use_semantic_scholar_live=True)

    async def fake_rewrite(subq: str) -> str:
        return subq

    async def fake_hybrid_search(*args, **kwargs) -> list[Hit]:
        return [_hit()]

    async def fake_rerank(query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return hits[:top_n]

    async def fake_judge_batch(subq: str, hits: list[Hit]) -> list[RelevanceVerdict]:
        return [RelevanceVerdict(relevant=True, rationale="ok") for _ in hits]

    monkeypatch.setattr(retrieve.rewriter, "rewrite_for_retrieval", fake_rewrite)
    monkeypatch.setattr(retrieve, "get_embedding_client", lambda name: _FakeEmbeddingClient())
    monkeypatch.setattr(retrieve, "hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(retrieve.deduplicator, "dedupe", lambda hits: hits)
    monkeypatch.setattr(retrieve.reranker, "rerank", fake_rerank)
    if patch_judge:
        monkeypatch.setattr(retrieve.relevance_judge, "judge_batch", fake_judge_batch)


def _state() -> dict:
    return {
        "question": "How does Reflexion extend ReAct?",
        "thread_id": "test-thread", "retrieval_paper_ids": ["arxiv:2210.03629v1"],
        "question_type": "multi_hop",
        "decomposition": Decomposition(sub_questions=[SubQuestion(text="How does Reflexion extend ReAct?")]),
        "subq_results": {},
        "multi_hop_attempts": 0,
        "multi_hop_expansions": 0,
        "multi_hop_preemptive_requested": False,
        "multi_hop_preemptive_subqs": [0],
        "active_subq_index": None,
    }


def _hit() -> Hit:
    return Hit(
        chunk_id=1,
        paper_id="arxiv:2210.03629",
        section="Introduction",
        text="ReAct interleaves reasoning and acting.",
        score=1.0,
    )


class _FakeEmbeddingClient:
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.01] * 1536 for _ in texts]


class _BrokenChatClient:
    async def chat(self, *args, **kwargs):
        from src.core.types import ChatResponse

        return ChatResponse(content="", model="fake-fast", finish_reason="length")
