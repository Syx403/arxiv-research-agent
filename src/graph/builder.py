from __future__ import annotations

import asyncio
import httpx

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph

from src.core import db
from src.core.run_context import current_run
from src.core.research_policy import current_policy
from src.core.trace import traced_node
from src.graph.controller import control_research_node
from src.graph.nodes.finalize import finalize_node
from src.graph.nodes.retrieve import retrieve_node
from src.graph.nodes.self_rag import self_rag_node, recover_evidence_node
from src.graph.nodes.synthesize import synthesize_node
from src.graph.nodes.papers import (
    assess_papers_node, plan_papers_node, expand_reading_node,
    present_papers_node, read_papers_node, search_papers_node,
)
from src.graph.state import AgentState
from src.llm.client import close_llm_clients
from src.llm.budget import budget_stop_reason, BudgetExceeded
from src.llm.errors import EmptyProviderResponseError, LLMError
from src.memory.episodic import session_store


_graph = None
_checkpointer: AsyncPostgresSaver | None = None
_setup_done = False
_init_lock = asyncio.Lock()


async def run(question: str, *, thread_id: str, skip_reflection: bool = False) -> AgentState:
    await session_store.get_or_create_session(thread_id)
    graph = await _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = _initial_state(question, thread_id, skip_reflection=skip_reflection) if question.strip() else None
    try:
        return await graph.ainvoke(initial_state, config=config)
    except (BudgetExceeded, LLMError, EmptyProviderResponseError, httpx.HTTPError, TimeoutError) as exc:
        snapshot = await graph.aget_state(config)
        last = dict(snapshot.values)
        last["forced_stop_reason"] = budget_stop_reason(exc) if isinstance(exc, BudgetExceeded) else "provider_unavailable"
        name, operation = ("present_papers", present_papers_node) if last.get("result_kind") != "evidence" else ("finalize", finalize_node)
        updates = await traced_node(name, operation)(last)
        await graph.aupdate_state(config, updates, as_node=name)
        return {**last, **updates}


async def shutdown() -> None:
    global _graph, _checkpointer, _setup_done
    await close_llm_clients()
    await db.close_pools()
    _graph = None
    _checkpointer = None
    _setup_done = False


async def finalize_interrupted_turn(*, thread_id: str, turn_id: str, reason: str) -> AgentState:
    """Project this turn's checkpoint without resuming any model/tool work."""
    if _graph is None:
        return {}
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await _graph.aget_state(config)
    last = dict(snapshot.values)
    if not turn_id or last.get("turn_id") != turn_id:
        return {}  # Never deliver the previous turn's evidence after an early failure.
    last["forced_stop_reason"] = reason
    name, operation = ("finalize", finalize_node) if last.get("result_kind") == "evidence" else ("present_papers", present_papers_node)
    updates = await traced_node(name, operation)(last)
    await _graph.aupdate_state(config, updates, as_node=name)
    return {**last, **updates}


async def _get_graph():
    global _graph, _checkpointer, _setup_done
    async with _init_lock:
        if _graph is not None:
            return _graph
        pool = await db.get_psycopg_pool()
        _checkpointer = AsyncPostgresSaver(pool, serde=checkpoint_serializer())
        if not _setup_done:
            await _checkpointer.setup()
            _setup_done = True
        _graph = build_graph(checkpointer=_checkpointer)
        return _graph


def checkpoint_serializer():
    types = [("src.retrieval.index.types", name) for name in ("Hit", "SourceSpan")]
    types.extend(("src.core.types", name) for name in (
        "Citation", "CitationVerdict", "Decomposition", "EvidenceChunk", "GraphExpansion",
        "ReflectionReport", "RouteDecision", "SubQResult", "SubQuestion", "VerificationReport",
    ))
    return JsonPlusSerializer(allowed_msgpack_modules=types)


def build_graph(*, checkpointer=None):
    """The same graph supports durable runs and deterministic offline checks."""
    builder = StateGraph(AgentState)
    for name, operation in (
        ("plan_papers", plan_papers_node), ("control_research", control_research_node),
        ("search_papers", search_papers_node), ("assess_papers", assess_papers_node),
        ("read_papers", read_papers_node), ("expand_reading", expand_reading_node),
        ("present_papers", present_papers_node), ("retrieve", retrieve_node),
        ("synthesize", synthesize_node), ("self_rag", self_rag_node), ("recover_evidence", recover_evidence_node), ("finalize", finalize_node),
    ):
        builder.add_node(name, traced_node(name, operation))
    builder.set_entry_point("plan_papers")
    for name in ("plan_papers", "search_papers", "assess_papers", "read_papers", "retrieve", "self_rag", "recover_evidence", "expand_reading"):
        builder.add_edge(name, "control_research")
    builder.add_conditional_edges("control_research", lambda state: state["next_node"],
        {name: name for name in ("search_papers", "assess_papers", "read_papers", "retrieve",
                                 "expand_reading", "synthesize", "self_rag", "recover_evidence", "present_papers", "finalize")})
    builder.add_edge("synthesize", "self_rag")
    builder.add_edge("present_papers", END)
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=checkpointer).with_config(recursion_limit=64)


def _initial_state(question: str, thread_id: str, *, skip_reflection: bool = False) -> AgentState:
    return {
        "messages": [HumanMessage(content=question)],
        "thread_id": thread_id,
        "turn_id": current_run().turn_id,
        "question": question,
        "decomposition": None,
        "subq_results": {},
        "evidence": [],
        "graph_expansion": None,
        "multi_hop_visited": [],
        "multi_hop_attempts": 0,
        "multi_hop_expansions": 0,
        "multi_hop_preemptive_requested": False,
        "multi_hop_preemptive_subqs": [],
        "next_subqueries": [],
        "question_type": "unknown",
        "answer": None,
        "answer_fields": [],
        "response_fields": [f.model_dump(mode="json") for f in current_run().response_fields],
        "citations": [],
        "verification": None,
        "reflection": None,
        "tool_iters": 0,
        "active_subq_index": None,
        "evidence_lookup": {},
        "skip_reflection": skip_reflection,
        "synthesis_format_degraded": False,
        "synthesis_finish_reason": "",
        "draft_verification": None,
        "verification_notes": [],
        "research_question": question,
        "conversation_context": [],
        "status": "running",
        "stop_reason": "",
        "missing_aspects": [],
        "verified_answer": "",
        "forced_stop_reason": "",
        "research_policy": current_policy().model_dump(),
        "external_discovery": {},
        "retrieval_paper_ids": None,
        "best_verified": {},
        "last_verification_evidence": {},
        "delivery_from_prior_draft": False,
        "verification_recovery_attempted": False,
        "verification_citation_repair_attempted": False,
        "verification_recovery_changed": False,
        "verification_recovery_requires_rewrite": False,
        "claim_review": {},
        "verification_stalled": False,
        "result_kind": "papers",
        "completed_node": "", "next_node": "", "assessed_fingerprint": "", "rejected_candidate_fingerprints": {},
        "node_timings_s": {},
        "related_expansion_attempted": False, "related_search": False, "reading_attempted_ids": [], "reading_added_count": 0,
        "paper_plan": {}, "paper_search": {}, "paper_candidates": [], "paper_results": [],
        "paper_intro": "", "pending_paper_queries": [],
    }
