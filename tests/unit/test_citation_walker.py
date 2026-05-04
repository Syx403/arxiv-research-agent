from __future__ import annotations

import pytest

from src.retrieval.multi_hop import citation_walker
from src.retrieval.multi_hop.citation_walker import CitationCandidate


@pytest.mark.asyncio
async def test_walk_returns_partial_when_budget_exceeded_before_expansion(monkeypatch) -> None:
    monkeypatch.setattr(citation_walker, "_budget_exceeded", lambda started_at, budget: True)

    result = await citation_walker.walk_citations(
        ["arxiv:2210.03629"],
        "question",
        wall_clock_budget_s=1,
    )

    assert result.visited_paper_ids == ["arxiv:2210.03629"]
    assert result.newly_ingested_paper_ids == []


def test_heuristic_score_uses_generic_lexical_overlap_without_domain_boosts() -> None:
    candidate = CitationCandidate(
        paper_id="arxiv:2303.11366",
        title="Reflexion and ReAct",
        abstract="Reflection feedback for agents.",
        depth=1,
        edge_type="outbound",
        is_complete=False,
    )

    score = citation_walker._heuristic_score("How does Reflexion extend ReAct with feedback?", candidate)

    assert score == pytest.approx(5.0)
