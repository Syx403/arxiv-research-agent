from __future__ import annotations

from src.graph.state import AgentState
from src.retrieval.query import decomposer


async def decompose_node(state: AgentState) -> dict:
    question = state.get("question", "").strip()
    if state.get("decomposition") is not None:
        return {}
    decomposition = await decomposer.decompose(question)
    return {
        "decomposition": decomposition,
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
    }
