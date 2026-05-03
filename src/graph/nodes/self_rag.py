from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.core.types import SELF_RAG_MAX_RETRIES
from src.graph.state import AgentState
from src.retrieval.self_rag.verifier import verify_citations


async def self_rag_node(state: AgentState) -> dict:
    answer = state.get("answer") or ""
    citations = state.get("citations", [])
    verification = await verify_citations(answer, citations, state.get("evidence_lookup", {}))
    if verification.passed:
        return {"verification": verification}

    failed = [verdict for verdict in verification.verdicts if not verdict.supports]
    kept_citations = [verdict.citation for verdict in verification.verdicts if verdict.supports]
    updates = {
        "verification": verification,
        "citations": kept_citations,
        "messages": [
            SystemMessage(
                content=(
                    f"Citation {verdict.citation.paper_id}#{verdict.citation.chunk_id} "
                    f"failed verification: {verdict.rationale}"
                )
            )
            for verdict in failed
        ],
    }
    if state.get("tool_iters", 0) < SELF_RAG_MAX_RETRIES:
        updates["tool_iters"] = state.get("tool_iters", 0) + 1
    return updates
