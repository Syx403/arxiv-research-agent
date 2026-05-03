from __future__ import annotations

from src.graph.state import AgentState
from src.memory import reflection
from src.memory.episodic import session_store


async def reflect_node(state: AgentState) -> dict:
    thread_id = state.get("thread_id")
    if not thread_id:
        raise RuntimeError("reflect_node requires state.thread_id")
    answer = state.get("answer") or ""
    citations = state.get("citations", [])
    await session_store.append_message(
        thread_id,
        "assistant",
        answer,
        metadata={"citations": [citation.model_dump() for citation in citations]},
    )
    report = await reflection.reflect_session(thread_id)
    return {"reflection": report}
