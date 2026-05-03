from __future__ import annotations

from src.core.types import MULTI_HOP_MAX_CALLS, SELF_RAG_MAX_RETRIES
from src.graph.state import AgentState


def route_after_retrieve(state: AgentState) -> str:
    if _all_subquestions_complete(state):
        return "synthesize"

    current = _current_result(state)
    if (
        current is not None
        and not current.sufficient
        and not current.multi_hop_used
        and current.route_decision.may_use_semantic_scholar_live
        and state.get("multi_hop_calls", 0) < MULTI_HOP_MAX_CALLS
    ):
        return "multi_hop"

    return "synthesize"


def route_after_verify(state: AgentState) -> str:
    verification = state.get("verification")
    if (
        verification is not None
        and not verification.passed
        and state.get("tool_iters", 0) < SELF_RAG_MAX_RETRIES
    ):
        return "synthesize"
    return "reflect"


def _all_subquestions_complete(state: AgentState) -> bool:
    decomposition = state.get("decomposition")
    if decomposition is None:
        return True
    return len(state.get("subq_results", {})) >= len(decomposition.sub_questions)


def _current_result(state: AgentState):
    current_index = state.get("active_subq_index")
    if current_index is None:
        return None
    return state.get("subq_results", {}).get(current_index)
