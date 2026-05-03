from __future__ import annotations

import pytest

from src.retrieval.multi_hop import citation_walker


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
