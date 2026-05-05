from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from src.core.types import (
    MULTI_HOP_MAX_CALLS,
    PAPER_HINT_BOOST_FACTOR,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K_AFTER_FUSION,
    SELF_RAG_MAX_RETRIES,
    GraphExpansion,
    RouteDecision,
    SubQResult,
    SufficiencyVerdict,
)
from src.graph.state import AgentState
from src.llm.client import get_embedding_client
from src.memory.semantic import graph_retriever
from src.retrieval.index.hybrid_search import hybrid_search
from src.retrieval.index.types import Hit
from src.retrieval.postprocess import deduplicator, reranker
from src.retrieval.query import rewriter, router
from src.retrieval.self_rag import relevance_judge, sufficiency_check


logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    subq: str
    subq_index: int
    multi_hop_used: bool
    route_decision: RouteDecision | None = None
    graph_expansion: GraphExpansion | None = None
    rewritten_query: str = ""
    embedding_text: str = ""
    embedding: list[float] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    accumulated_hits: dict[int, Hit] = field(default_factory=dict)
    sufficiency: SufficiencyVerdict | None = None
    retries_used: int = 0


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
    ctx = RetrievalContext(subq=subq, subq_index=subq_index, multi_hop_used=multi_hop_used)
    ctx = await _stage_route(ctx)
    if ctx.route_decision is not None and ctx.route_decision.use_concept_graph:
        ctx = await _stage_graph_expand(ctx)
    else:
        ctx.graph_expansion = GraphExpansion()
    ctx = await _stage_rewrite(ctx)

    while True:
        ctx = await _stage_embed(ctx)
        ctx = await _stage_hybrid_search(ctx)
        ctx = _stage_apply_paper_boost(ctx)
        ctx = _stage_dedupe(ctx)
        ctx = await _stage_rerank(ctx)
        ctx = await _stage_filter_relevant(ctx)
        ctx = _stage_accumulate_evidence(ctx)
        ctx = await _stage_check_sufficiency(ctx)
        if ctx.sufficiency is not None and (ctx.sufficiency.sufficient or ctx.retries_used >= SELF_RAG_MAX_RETRIES):
            break
        ctx = _stage_prepare_retry(ctx)

    return _to_subq_result(ctx), ctx.graph_expansion or GraphExpansion()


async def _stage_route(ctx: RetrievalContext) -> RetrievalContext:
    ctx.route_decision = await router.route(ctx.subq)
    _log_stage(
        ctx,
        "route",
        use_concept_graph=ctx.route_decision.use_concept_graph,
        may_use_semantic_scholar_live=ctx.route_decision.may_use_semantic_scholar_live,
    )
    return ctx


async def _stage_graph_expand(ctx: RetrievalContext) -> RetrievalContext:
    ctx.graph_expansion = await graph_retriever.expand_query_via_graph(ctx.subq)
    _log_stage(
        ctx,
        "graph_expand",
        concept_count=len(ctx.graph_expansion.seed_concept_ids),
        neighbor_count=len(ctx.graph_expansion.neighbor_concept_ids),
        paper_hints=ctx.graph_expansion.extra_paper_hints,
    )
    return ctx


async def _stage_rewrite(ctx: RetrievalContext) -> RetrievalContext:
    ctx.rewritten_query = await rewriter.rewrite_for_retrieval(ctx.subq)
    ctx.embedding_text = await rewriter.hyde(ctx.subq) if rewriter.should_use_hyde(ctx.subq) else ctx.rewritten_query
    _log_stage(ctx, "rewrite", used_hyde=ctx.embedding_text != ctx.rewritten_query)
    return ctx


async def _stage_embed(ctx: RetrievalContext) -> RetrievalContext:
    ctx.embedding = (await get_embedding_client("embed-small").embed([ctx.embedding_text]))[0]
    _log_stage(ctx, "embed", embedding_dim=len(ctx.embedding))
    return ctx


async def _stage_hybrid_search(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    ctx.hits = await hybrid_search(ctx.rewritten_query, ctx.embedding, top_k_final=RETRIEVAL_TOP_K_AFTER_FUSION)
    _log_stage(ctx, "hybrid_search", input_count=before, output_count=len(ctx.hits))
    return ctx


def _stage_apply_paper_boost(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    paper_hints = (ctx.graph_expansion or GraphExpansion()).extra_paper_hints
    ctx.hits = _apply_paper_hint_boost(ctx.hits, paper_hints)
    _log_stage(ctx, "apply_paper_boost", input_count=before, output_count=len(ctx.hits), paper_hint_count=len(paper_hints))
    return ctx


def _stage_dedupe(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    ctx.hits = deduplicator.dedupe(ctx.hits)
    _log_stage(ctx, "dedupe", input_count=before, output_count=len(ctx.hits))
    return ctx


async def _stage_rerank(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    ctx.hits = await reranker.rerank(ctx.rewritten_query, ctx.hits, top_n=RERANK_TOP_K)
    _log_stage(ctx, "rerank", input_count=before, output_count=len(ctx.hits), rerank_top_n=RERANK_TOP_K)
    return ctx


async def _stage_filter_relevant(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    verdicts = await relevance_judge.judge_batch(ctx.subq, ctx.hits)
    parse_failures = sum(1 for verdict in verdicts if verdict.rationale.startswith("PARSE_FAILURE"))
    ctx.hits = [hit for hit, verdict in zip(ctx.hits, verdicts, strict=True) if verdict.relevant]
    _log_stage(
        ctx,
        "filter_relevant",
        input_count=before,
        output_count=len(ctx.hits),
        filtered_out=before - len(ctx.hits),
        parse_failures=parse_failures,
    )
    return ctx


def _stage_accumulate_evidence(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.accumulated_hits)
    for hit in ctx.hits:
        if hit.chunk_id not in ctx.accumulated_hits:
            ctx.accumulated_hits[hit.chunk_id] = hit
    _log_stage(ctx, "accumulate_evidence", input_count=before, output_count=len(ctx.accumulated_hits))
    return ctx


async def _stage_check_sufficiency(ctx: RetrievalContext) -> RetrievalContext:
    hits = list(ctx.accumulated_hits.values())
    ctx.sufficiency = await sufficiency_check.is_sufficient(ctx.subq, hits)
    _log_stage(
        ctx,
        "check_sufficiency",
        input_count=len(hits),
        output_count=len(hits),
        sufficient=ctx.sufficiency.sufficient,
        missing_aspects=ctx.sufficiency.missing_aspects,
    )
    return ctx


def _stage_prepare_retry(ctx: RetrievalContext) -> RetrievalContext:
    ctx.retries_used += 1
    missing = " ".join(ctx.sufficiency.missing_aspects if ctx.sufficiency else []).strip()
    base_query = ctx.subq if not ctx.rewritten_query else ctx.rewritten_query.split(" specifically ", maxsplit=1)[0]
    ctx.rewritten_query = f"{base_query} specifically {missing or ctx.subq}".strip()
    ctx.embedding_text = ctx.rewritten_query
    _log_stage(ctx, "prepare_retry", retries_used=ctx.retries_used)
    return ctx


def _to_subq_result(ctx: RetrievalContext) -> SubQResult:
    if ctx.route_decision is None:
        raise RuntimeError("Retrieve pipeline ended without a route decision")
    return SubQResult(
        subq_index=ctx.subq_index,
        hits=list(ctx.accumulated_hits.values()),
        sufficient=bool(ctx.sufficiency and ctx.sufficiency.sufficient),
        route_decision=ctx.route_decision,
        retries_used=ctx.retries_used,
        multi_hop_used=ctx.multi_hop_used,
    )


def _log_stage(ctx: RetrievalContext, stage: str, **fields) -> None:
    extra = {
        "stage": stage,
        "subq_index": ctx.subq_index,
        "input_count": fields.pop("input_count", len(ctx.hits)),
        "output_count": fields.pop("output_count", len(ctx.hits)),
    }
    extra.update(fields)
    logger.info("retrieve_stage", extra=extra)


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
