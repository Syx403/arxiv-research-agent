from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass, field, replace

from src.core.trace import record_event

from src.core.types import (
    RERANK_TOP_K,
    RelevanceVerdict,
    RETRIEVAL_TOP_K_AFTER_FUSION,
    SELF_RAG_MAX_RETRIES,
    GraphExpansion,
    RouteDecision,
    SubQResult,
    SufficiencyVerdict,
)
from src.graph.state import AgentState
from src.llm.client import get_embedding_client
from src.retrieval.index.hybrid_search import hybrid_search
from src.retrieval.index.types import Hit
from src.retrieval.evidence import (
    EVIDENCE_TOTAL_TOKENS, EVIDENCE_SUBQUESTION_TOKENS, evidence_text, is_reference_list,
    merge_spans, pack_evidence, select_sentences, token_count,
)
from src.retrieval.postprocess import deduplicator, reranker
from src.retrieval.query import rewriter
from src.retrieval.self_rag import relevance_judge, sufficiency_check


logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    subq: str
    subq_index: int
    question_type: str
    question_id: str
    multi_hop_used: bool
    preemptive_already_done: bool = False
    multi_hop_attempts: int = 0
    multi_hop_expansions: int = 0
    route_decision: RouteDecision | None = None
    graph_expansion: GraphExpansion | None = None
    rewritten_query: str = ""
    embedding_text: str = ""
    embedding: list[float] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    accumulated_hits: dict[int, Hit] = field(default_factory=dict)
    sufficiency: SufficiencyVerdict | None = None
    retries_used: int = 0
    preemptive_multi_hop_requested: bool = False
    next_subqueries: list[str] = field(default_factory=list)
    relevance_cache: dict[tuple[str, int, str, str], RelevanceVerdict] = field(default_factory=dict)
    stop_reason: str = ""
    judgment_failures: list[int] = field(default_factory=list)
    unassessed_hits: list[Hit] = field(default_factory=list)
    evidence_budget: int = EVIDENCE_SUBQUESTION_TOKENS
    filter_paper_ids: list[str] | None = None
    sufficiency_question: str | None = None
    last_pack_signature: tuple = ()


async def retrieve_node(state: AgentState) -> dict:
    if not state.get("decomposition"):
        return {}
    if state.get("retrieval_paper_ids") is None:
        raise ValueError("Full-text retrieval requires an explicit paper/version scope")
    update = await _retrieve_independent(state)
    if state.get("paper_plan", {}).get("reading_scope") == "verify_claim":
        from src.retrieval.self_rag.claim_audit import audit_claim

        review, hits = await audit_claim(state.get("research_question") or state["question"],
                                        state["retrieval_paper_ids"])
        # A missing theorem is not automatically a failed task. Completion is
        # bounded by the actual follow-up scan and subsequently verified claims.
        groups = update["subq_results"]
        for index, result in groups.items():
            groups[index] = result.model_copy(update={
                "hits": hits or result.hits, "sufficient": review["status"] == "complete",
                "missing_aspects": [] if review["status"] == "complete" else [review["rationale"]],
                "next_subqueries": [], "stop_reason": "claim_audit_" + review["status"],
            })
        update.update(claim_review=review, evidence=_assemble_evidence(groups))
    return update


async def _retrieve_independent(state: AgentState) -> dict:
    decomposition = state["decomposition"]
    results = dict(state.get("subq_results", {}))
    pending = set(range(len(decomposition.sub_questions))) - results.keys()
    graph_expansion = state.get("graph_expansion")

    async def retrieve_one(index):
        paper_scope = decomposition.sub_questions[index].paper_ids
        if paper_scope is not None and not set(paper_scope) <= set(state.get("retrieval_paper_ids") or []):
            raise ValueError("Subquestion source scope exceeds acquired papers")
        if state.get("retrieval_paper_ids") == []:
            # An empty acquisition scope must not query unrelated indexed papers.
            return SubQResult(subq_index=index, sufficient=False, route_decision=RouteDecision(),
                              missing_aspects=["No full-text evidence acquired for the requested papers or dates."],
                              stop_reason="external_evidence_unavailable"), GraphExpansion()
        return await _run_subquestion_pipeline(
            decomposition.sub_questions[index].text, subq_index=index,
            question_type=state.get("question_type", "unknown"),
            question_id=state.get("thread_id", ""),
            multi_hop_used=bool(state.get("external_discovery", {}).get("attempted")),
            preemptive_already_done=True,
            evidence_budget=min(EVIDENCE_SUBQUESTION_TOKENS,
                                EVIDENCE_TOTAL_TOKENS // max(1, len(decomposition.sub_questions))),
            filter_paper_ids=paper_scope if paper_scope is not None else state.get("retrieval_paper_ids"),
            sufficiency_question=decomposition.sub_questions[index].sufficiency_question,
            **({"max_retries": 0} if state.get("paper_plan", {}).get("reading_scope") == "verify_claim" else {}),
        )

    running = {}
    try:
        while pending or running:
            ready = sorted(i for i in pending if set(decomposition.sub_questions[i].depends_on) <= results.keys())
            for index in ready[:max(0, 2 - len(running))]:
                running[asyncio.create_task(retrieve_one(index))] = index
                pending.remove(index)
            if not running:
                raise ValueError("Invalid subquestion dependency graph")
            # Refill a free slot immediately; a slow sibling must not block
            # independent papers or a task whose own dependencies are complete.
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=running.__getitem__):
                index = running.pop(task)
                result, expansion = task.result()
                results[index] = result
                graph_expansion = expansion
    finally:
        for task in running:
            if not task.done():
                task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
    return {"subq_results": results, "evidence": _assemble_evidence(results),
            "graph_expansion": graph_expansion, "active_subq_index": None,
            "multi_hop_preemptive_requested": False}


async def _run_subquestion_pipeline(
    subq: str,
    *,
    subq_index: int,
    multi_hop_used: bool,
    question_type: str = "unknown",
    question_id: str = "",
    preemptive_already_done: bool = False,
    multi_hop_attempts: int = 0,
    multi_hop_expansions: int = 0,
    evidence_budget: int = EVIDENCE_SUBQUESTION_TOKENS,
    filter_paper_ids: list[str] | None = None,
    sufficiency_question: str | None = None,
    max_retries: int = SELF_RAG_MAX_RETRIES,
) -> tuple[SubQResult, GraphExpansion]:
    ctx = RetrievalContext(
        subq=subq,
        subq_index=subq_index,
        question_type=question_type,
        question_id=question_id,
        multi_hop_used=multi_hop_used,
        preemptive_already_done=preemptive_already_done,
        multi_hop_attempts=multi_hop_attempts,
        multi_hop_expansions=multi_hop_expansions,
        evidence_budget=evidence_budget,
        filter_paper_ids=filter_paper_ids,
        sufficiency_question=sufficiency_question,
    )
    ctx.route_decision = RouteDecision()
    ctx.graph_expansion = GraphExpansion()
    ctx = await _stage_rewrite(ctx)

    while True:
        ctx = await _stage_embed(ctx)
        ctx = await _stage_hybrid_search(ctx)
        ctx = _stage_dedupe(ctx)
        ctx = await _stage_rerank(ctx)
        ctx = await _stage_filter_relevant(ctx)
        ctx = _stage_accumulate_evidence(ctx)
        if ctx.judgment_failures and not ctx.accumulated_hits:
            ctx.sufficiency = SufficiencyVerdict(sufficient=False, status="error",
                                                 error_code="relevance_judgment_failed")
            ctx.stop_reason = "relevance_judgment_failed"
            break
        signature = tuple((h.paper_id, h.chunk_id, evidence_text(h)) for h in
                          pack_evidence(list(ctx.accumulated_hits.values()), ctx.subq, token_budget=ctx.evidence_budget))
        if ctx.retries_used and signature == ctx.last_pack_signature:
            ctx.stop_reason = "relevance_judgment_failed" if ctx.judgment_failures else "no_new_evidence"
            _log_stage(ctx, "no_new_evidence")
            break
        ctx = await _stage_check_sufficiency(ctx)
        ctx.last_pack_signature = signature
        if ctx.sufficiency and ctx.sufficiency.status == "error":
            ctx.stop_reason = "sufficiency_judgment_failed"
            break
        if ctx.judgment_failures and ctx.sufficiency and not ctx.sufficiency.sufficient:
            ctx.stop_reason = "relevance_judgment_failed"
            break
        if ctx.sufficiency is not None and (ctx.sufficiency.sufficient or ctx.retries_used >= max_retries):
            break
        ctx = _stage_prepare_retry(ctx)

    return _to_subq_result(ctx), ctx.graph_expansion or GraphExpansion()


async def _stage_rewrite(ctx: RetrievalContext) -> RetrievalContext:
    ctx.rewritten_query = await rewriter.rewrite_for_retrieval(ctx.subq)
    ctx.embedding_text = ctx.rewritten_query
    _log_stage(ctx, "rewrite", used_hyde=ctx.embedding_text != ctx.rewritten_query)
    record_event("retrieval_query", subq_index=ctx.subq_index, original_query=ctx.subq,
                 rewritten_query=ctx.rewritten_query, embedding_query=ctx.embedding_text)
    return ctx


async def _stage_embed(ctx: RetrievalContext) -> RetrievalContext:
    ctx.embedding = (await get_embedding_client("embed-small").embed([ctx.embedding_text]))[0]
    _log_stage(ctx, "embed", embedding_dim=len(ctx.embedding))
    return ctx


async def _stage_hybrid_search(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    ctx.hits = await hybrid_search(ctx.rewritten_query, ctx.embedding,
                                  top_k_final=RETRIEVAL_TOP_K_AFTER_FUSION, original_query=ctx.subq,
                                  **({"filter_paper_ids": ctx.filter_paper_ids} if ctx.filter_paper_ids is not None else {}))
    _log_stage(ctx, "hybrid_search", input_count=before, output_count=len(ctx.hits))
    return ctx


def _stage_dedupe(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    ctx.hits = deduplicator.dedupe([h for h in ctx.hits if not is_reference_list(h, ctx.subq)])
    _log_stage(ctx, "dedupe", input_count=before, output_count=len(ctx.hits))
    return ctx


async def _stage_rerank(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    focus = ctx.rewritten_query if ctx.retries_used else ctx.subq
    ctx.hits = await reranker.rerank(focus, ctx.hits, top_n=RERANK_TOP_K)
    record_event("retrieval_candidates", subq_index=ctx.subq_index, retry=ctx.retries_used,
                 chunks=[{"chunk_id": h.chunk_id, "paper_id": h.paper_id, "section": h.section,
                          "title": h.title, "score": h.score} for h in ctx.hits])
    _log_stage(ctx, "rerank", input_count=before, output_count=len(ctx.hits), rerank_top_n=RERANK_TOP_K)
    return ctx


async def _stage_filter_relevant(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.hits)
    def key(hit):
        return (hit.paper_id, hit.chunk_id, hit.text, focus)
    focus = ctx.subq if not ctx.retries_used else ctx.rewritten_query
    fresh = [hit for hit in ctx.hits if key(hit) not in ctx.relevance_cache]
    fresh_verdicts = await relevance_judge.judge_batch(focus, fresh) if fresh else []
    current = dict(ctx.relevance_cache)
    for hit, verdict in zip(fresh, fresh_verdicts, strict=True):
        current[key(hit)] = verdict
        if verdict.status == "ok":
            ctx.relevance_cache[key(hit)] = verdict
    verdicts = [current[key(hit)] for hit in ctx.hits]
    ctx.judgment_failures = [h.chunk_id for h, v in zip(ctx.hits, verdicts, strict=True) if v.status == "error"]
    ctx.unassessed_hits = [h for h in ctx.hits if h.chunk_id in ctx.judgment_failures]
    parse_failures = len(ctx.judgment_failures)
    ctx.hits = [select_sentences(hit, verdict.sentence_ids, role=verdict.evidence_role)
                if verdict.sentence_ids else hit
                for hit, verdict in zip(ctx.hits, verdicts, strict=True) if verdict.relevant is True]
    _log_stage(
        ctx,
        "filter_relevant",
        input_count=before,
        output_count=len(ctx.hits),
        filtered_out=before - len(ctx.hits),
        parse_failures=parse_failures,
        reused_verdicts=before - len(fresh),
    )
    return ctx


def _stage_accumulate_evidence(ctx: RetrievalContext) -> RetrievalContext:
    before = len(ctx.accumulated_hits)
    for hit in ctx.hits:
        ctx.accumulated_hits[hit.chunk_id] = hit
    _log_stage(ctx, "accumulate_evidence", input_count=before, output_count=len(ctx.accumulated_hits))
    return ctx


async def _stage_check_sufficiency(ctx: RetrievalContext) -> RetrievalContext:
    hits = pack_evidence(list(ctx.accumulated_hits.values()), ctx.subq, token_budget=ctx.evidence_budget)
    record_event("evidence_pack", subq_index=ctx.subq_index, token_count=sum(token_count(evidence_text(h)) for h in hits),
                 chunks=[{"chunk_id": h.chunk_id, "paper_id": h.paper_id, "role": h.evidence_role,
                          "source_spans": [[s.start, s.end] for s in h.evidence_spans],
                          "text": evidence_text(h)} for h in hits])
    # Subsequent synthesis receives this exact checked pack, not a newly truncated ranking.
    ctx.hits = hits
    ctx.sufficiency = await sufficiency_check.is_sufficient(
        ctx.sufficiency_question or ctx.subq,
        hits,
        question_type=ctx.question_type,
        question_id=ctx.question_id or ctx.subq,
        requery_round=ctx.retries_used,
    )
    ctx.next_subqueries = list(ctx.sufficiency.missing_aspects) if not ctx.sufficiency.sufficient else []
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
        hits=pack_evidence(list(ctx.accumulated_hits.values()), ctx.subq, token_budget=ctx.evidence_budget),
        sufficient=bool(ctx.sufficiency and ctx.sufficiency.sufficient),
        route_decision=ctx.route_decision,
        retries_used=ctx.retries_used,
        multi_hop_used=ctx.multi_hop_used,
        preemptive_multi_hop_requested=ctx.preemptive_multi_hop_requested,
        next_subqueries=ctx.next_subqueries,
        missing_aspects=list(ctx.sufficiency.missing_aspects) if ctx.sufficiency else [],
        stop_reason=ctx.stop_reason or (
            "evidence_sufficient" if ctx.sufficiency and ctx.sufficiency.sufficient
            else "expand_citations" if ctx.preemptive_multi_hop_requested
            else "retrieval_retry_limit"
        ),
        judgment_failures=ctx.judgment_failures,
        unassessed_hits=ctx.unassessed_hits,
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
    if stage in {"rerank", "filter_relevant", "accumulate_evidence", "check_sufficiency", "no_new_evidence"}:
        # After packing, report the evidence actually supplied to the model,
        # rather than all earlier semantically accepted candidates.
        retained = list(ctx.hits) if stage == "check_sufficiency" else list(ctx.accumulated_hits.values())
        record_event("retrieval_progress", subq_index=ctx.subq_index, stage=stage,
                     candidate_chunks=len(ctx.hits), accepted_chunks=len(retained),
                     papers=len({h.paper_id for h in retained}),
                     retry=ctx.retries_used, reused_verdicts=fields.get("reused_verdicts", 0))


def _assemble_evidence(subq_results: dict[int, SubQResult]) -> list[Hit]:
    evidence: dict[int, Hit] = {}
    for idx in sorted(subq_results):
        for hit in subq_results[idx].hits:
            prior = evidence.get(hit.chunk_id)
            if prior is not None:
                if prior.paper_id != hit.paper_id or prior.text != hit.text:
                    raise ValueError("Source identity changed between subquestions")
                evidence[hit.chunk_id] = replace(prior, evidence_spans=merge_spans([
                    *prior.evidence_spans, *hit.evidence_spans]))
            else:
                evidence[hit.chunk_id] = hit
    return list(evidence.values())
