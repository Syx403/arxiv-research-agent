from __future__ import annotations

from src.core.types import (
    Citation,
    CitationVerdict,
    Decomposition,
    MULTI_HOP_MAX_CALLS,
    RouteDecision,
    SELF_RAG_MAX_RETRIES,
    SubQResult,
    SubQuestion,
    VerificationReport,
)
from src.graph.edges import route_after_retrieve, route_after_verify


def test_route_after_retrieve_hard_stops_at_multi_hop_budget() -> None:
    state = {
        "decomposition": Decomposition(sub_questions=[SubQuestion(text="What is ReAct?")]),
        "subq_results": {
            0: SubQResult(
                subq_index=0,
                hits=[],
                sufficient=False,
                route_decision=RouteDecision(may_use_semantic_scholar_live=True),
                retries_used=SELF_RAG_MAX_RETRIES,
                multi_hop_used=False,
            )
        },
        "active_subq_index": 0,
        "multi_hop_calls": MULTI_HOP_MAX_CALLS,
    }

    assert route_after_retrieve(state) == "synthesize"


def test_route_after_verify_hard_stops_at_retry_budget() -> None:
    citation = Citation(paper_id="arxiv:2210.03629", chunk_id=1, claim_span=(0, 10))
    state = {
        "verification": VerificationReport(
            verdicts=[CitationVerdict(citation=citation, supports=False, rationale="unsupported")],
            passed=False,
        ),
        "tool_iters": SELF_RAG_MAX_RETRIES,
    }

    assert route_after_verify(state) == "reflect"
