from __future__ import annotations

from src.core.types import (
    Citation,
    CitationVerdict,
    SELF_RAG_MAX_RETRIES,
    VerificationReport,
)
from src.graph.edges import route_after_verify




def test_route_after_verify_hard_stops_at_retry_budget() -> None:
    citation = Citation(paper_id="arxiv:2210.03629", chunk_id=1, claim_text="ReAct claim", claim_span=(0, 10))
    state = {
        "verification": VerificationReport(
            verdicts=[CitationVerdict(citation=citation, supports=False, rationale="unsupported")],
            passed=False,
        ),
        "tool_iters": SELF_RAG_MAX_RETRIES,
    }

    assert route_after_verify(state) == "finalize"
