from __future__ import annotations

from src.core.types import ReflectionReport
from src.graph.state import AgentState
from src.memory import reflection
from src.memory.episodic import session_store


async def reflect_node(state: AgentState) -> dict:
    if state.get("skip_reflection"):
        return {"reflection": ReflectionReport()}
    thread_id = state.get("thread_id")
    if not thread_id:
        raise RuntimeError("reflect_node requires state.thread_id")
    answer = state.get("answer") or ""
    citations = state.get("citations", [])
    verification = state.get("verification")
    verified = bool(verification is not None and verification.passed)
    await session_store.append_message(
        thread_id,
        "assistant",
        answer,
        metadata={
            "citations": [citation.model_dump() for citation in citations],
            "verification_passed": verified,
        },
    )
    if not verified:
        return {"reflection": ReflectionReport()}
    report = await reflection.reflect_session(thread_id)
    return {"reflection": report}
