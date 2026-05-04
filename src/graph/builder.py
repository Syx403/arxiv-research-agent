from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from src.core import db
from src.graph.edges import route_after_retrieve, route_after_verify
from src.graph.nodes.decompose import decompose_node
from src.graph.nodes.multi_hop import multi_hop_node
from src.graph.nodes.reflect import reflect_node
from src.graph.nodes.retrieve import retrieve_node
from src.graph.nodes.self_rag import self_rag_node
from src.graph.nodes.synthesize import synthesize_node
from src.graph.state import AgentState
from src.llm.client import close_llm_clients
from src.memory.episodic import session_store


_graph = None
_checkpointer: AsyncPostgresSaver | None = None
_setup_done = False
_init_lock = asyncio.Lock()


async def run(question: str, *, thread_id: str, skip_reflection: bool = False) -> AgentState:
    await session_store.get_or_create_session(thread_id)
    graph = await _get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    if not question.strip():
        return await graph.ainvoke(None, config=config)
    initial_state = _initial_state(question, thread_id, skip_reflection=skip_reflection)
    return await graph.ainvoke(initial_state, config=config)


async def shutdown() -> None:
    global _graph, _checkpointer, _setup_done
    await close_llm_clients()
    await db.close_pools()
    _graph = None
    _checkpointer = None
    _setup_done = False


async def _get_graph():
    global _graph, _checkpointer, _setup_done
    async with _init_lock:
        if _graph is not None:
            return _graph
        pool = await db.get_psycopg_pool()
        _checkpointer = AsyncPostgresSaver(pool)
        if not _setup_done:
            await _checkpointer.setup()
            _setup_done = True
        builder = StateGraph(AgentState)
        builder.add_node("decompose", decompose_node)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("multi_hop", multi_hop_node)
        builder.add_node("synthesize", synthesize_node)
        builder.add_node("self_rag", self_rag_node)
        builder.add_node("reflect", reflect_node)
        builder.set_entry_point("decompose")
        builder.add_edge("decompose", "retrieve")
        builder.add_conditional_edges(
            "retrieve",
            route_after_retrieve,
            {"synthesize": "synthesize", "multi_hop": "multi_hop"},
        )
        builder.add_edge("multi_hop", "retrieve")
        builder.add_edge("synthesize", "self_rag")
        builder.add_conditional_edges(
            "self_rag",
            route_after_verify,
            {"synthesize": "synthesize", "reflect": "reflect"},
        )
        builder.add_edge("reflect", END)
        _graph = builder.compile(checkpointer=_checkpointer)
        return _graph


def _initial_state(question: str, thread_id: str, *, skip_reflection: bool = False) -> AgentState:
    return {
        "messages": [HumanMessage(content=question)],
        "thread_id": thread_id,
        "question": question,
        "decomposition": None,
        "subq_results": {},
        "evidence": [],
        "graph_expansion": None,
        "multi_hop_visited": [],
        "multi_hop_calls": 0,
        "answer": None,
        "citations": [],
        "verification": None,
        "reflection": None,
        "tool_iters": 0,
        "active_subq_index": None,
        "evidence_lookup": {},
        "skip_reflection": skip_reflection,
        "synthesis_format_degraded": False,
    }
