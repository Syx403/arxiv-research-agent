from __future__ import annotations

from dataclasses import replace

from src.core.types import (
    MULTI_HOP_MAX_CALLS,
    PAPER_HINT_BOOST_FACTOR,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K_AFTER_FUSION,
    SELF_RAG_MAX_RETRIES,
    GraphExpansion,
    SubQResult,
)
from src.graph.state import AgentState
from src.llm.client import get_embedding_client
from src.memory.semantic import graph_retriever
from src.retrieval.index.hybrid_search import hybrid_search
from src.retrieval.index.types import Hit
from src.retrieval.postprocess import deduplicator, reranker
from src.retrieval.query import rewriter, router
from src.retrieval.self_rag import relevance_judge, sufficiency_check


async def retrieve_node(state: AgentState) -> dict:
    decomposition = state.get("decomposition")
    if decomposition is None:
        return {}

    subq_results = dict(state.get("subq_results", {}))
    active_index = _next_subq_index(state)
    if active_index is None:
        return {
            "evidence": _assemble_evidence(subq_results),
            "active_subq_index": None,
        }

    graph_expansion = state.get("graph_expansion")
    while active_index is not None:
        subq = decomposition.sub_questions[active_index].text
        multi_hop_used = _should_mark_multi_hop_used(state, subq_results, active_index)
        result, graph_expansion = await _run_subquestion_pipeline(
            subq,
            subq_index=active_index,
            multi_hop_used=multi_hop_used,
        )
        subq_results[active_index] = result

        if _should_route_to_multi_hop(state, result):
            break
        active_index = _first_missing_index(decomposition, subq_results)

    return {
        "subq_results": subq_results,
        "evidence": _assemble_evidence(subq_results),
        "graph_expansion": graph_expansion,
        "active_subq_index": active_index,
    }


async def _run_subquestion_pipeline(
    subq: str,
    *,
    subq_index: int,
    multi_hop_used: bool,
) -> tuple[SubQResult, GraphExpansion]:
    route_decision = await router.route(subq)
    graph_expansion = (
        await graph_retriever.expand_query_via_graph(subq)
        if route_decision.use_concept_graph
        else GraphExpansion()
    )
    rewritten = await rewriter.rewrite_for_retrieval(subq)
    embedding_text = await rewriter.hyde(subq) if rewriter.should_use_hyde(subq) else rewritten

    accumulated_hits: dict[int, Hit] = {}
    kept_hits: list[Hit] = []
    retries_used = 0
    sufficient = False
    query_text = rewritten
    while True:
        embedding = (await get_embedding_client("embed-small").embed([embedding_text]))[0]
        hits = await hybrid_search(query_text, embedding, top_k_final=RETRIEVAL_TOP_K_AFTER_FUSION)
        hits = _apply_paper_hint_boost(hits, graph_expansion.extra_paper_hints)
        hits = deduplicator.dedupe(hits)
        hits = await reranker.rerank(query_text, hits, top_n=RERANK_TOP_K)
        verdicts = await relevance_judge.judge_batch(subq, hits)
        new_kept = [hit for hit, verdict in zip(hits, verdicts, strict=True) if verdict.relevant]
        for hit in new_kept:
            if hit.chunk_id not in accumulated_hits:
                accumulated_hits[hit.chunk_id] = hit
        kept_hits = list(accumulated_hits.values())
        sufficiency = await sufficiency_check.is_sufficient(subq, kept_hits)
        sufficient = sufficiency.sufficient
        if sufficient or retries_used >= SELF_RAG_MAX_RETRIES:
            break
        retries_used += 1
        missing = " ".join(sufficiency.missing_aspects).strip()
        query_text = f"{rewritten} specifically {missing or subq}".strip()
        embedding_text = query_text

    return (
        SubQResult(
            subq_index=subq_index,
            hits=kept_hits,
            sufficient=sufficient,
            route_decision=route_decision,
            retries_used=retries_used,
            multi_hop_used=multi_hop_used,
        ),
        graph_expansion,
    )


def _apply_paper_hint_boost(hits: list[Hit], paper_hints: list[str]) -> list[Hit]:
    hinted = set(paper_hints)
    if not hinted:
        return hits
    boosted = [
        replace(hit, score=hit.score * PAPER_HINT_BOOST_FACTOR) if hit.paper_id in hinted else hit
        for hit in hits
    ]
    return sorted(boosted, key=lambda hit: hit.score, reverse=True)


def _next_subq_index(state: AgentState) -> int | None:
    decomposition = state.get("decomposition")
    if decomposition is None:
        return None
    subq_results = state.get("subq_results", {})
    active_index = state.get("active_subq_index")
    if active_index is not None:
        current = subq_results.get(active_index)
        if current is not None and not current.sufficient and not current.multi_hop_used:
            if state.get("multi_hop_calls", 0) > 0:
                return active_index
    return _first_missing_index(decomposition, subq_results)


def _first_missing_index(decomposition, subq_results: dict[int, SubQResult]) -> int | None:
    for idx in range(len(decomposition.sub_questions)):
        if idx not in subq_results:
            return idx
    return None


def _should_mark_multi_hop_used(
    state: AgentState,
    subq_results: dict[int, SubQResult],
    active_index: int,
) -> bool:
    current = subq_results.get(active_index)
    return bool(
        current is not None
        and not current.sufficient
        and not current.multi_hop_used
        and state.get("multi_hop_calls", 0) > 0
    )


def _should_route_to_multi_hop(state: AgentState, result: SubQResult) -> bool:
    return (
        not result.sufficient
        and not result.multi_hop_used
        and result.route_decision.may_use_semantic_scholar_live
        and state.get("multi_hop_calls", 0) < MULTI_HOP_MAX_CALLS
    )


def _assemble_evidence(subq_results: dict[int, SubQResult]) -> list[Hit]:
    evidence: list[Hit] = []
    seen_chunk_ids: set[int] = set()
    for idx in sorted(subq_results):
        for hit in subq_results[idx].hits:
            if hit.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(hit.chunk_id)
            evidence.append(hit)
    return evidence
