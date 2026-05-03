from __future__ import annotations

from src.core.types import MULTI_HOP_FRONTIER_LIMIT, MULTI_HOP_MAX_DEPTH
from src.graph.state import AgentState
from src.retrieval.multi_hop.citation_walker import walk_citations


async def multi_hop_node(state: AgentState) -> dict:
    active_index = state.get("active_subq_index")
    if active_index is None:
        return {}
    decomposition = state.get("decomposition")
    if decomposition is None:
        return {}
    current_result = state.get("subq_results", {}).get(active_index)
    if current_result is None:
        return {}

    seed_paper_ids = sorted({hit.paper_id for hit in current_result.hits})
    if not seed_paper_ids:
        return {"multi_hop_calls": state.get("multi_hop_calls", 0) + 1}

    result = await walk_citations(
        seed_paper_ids,
        decomposition.sub_questions[active_index].text,
        frontier_limit=MULTI_HOP_FRONTIER_LIMIT,
        max_depth=MULTI_HOP_MAX_DEPTH,
    )
    return {
        "multi_hop_calls": state.get("multi_hop_calls", 0) + 1,
        "multi_hop_visited": result.visited_paper_ids,
    }
