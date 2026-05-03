from __future__ import annotations

import os
import time

import pytest

from src.core import db
from src.core.types import MULTI_HOP_WALL_CLOCK_BUDGET_S
from src.llm.client import close_llm_clients
from src.retrieval.multi_hop.citation_walker import walk_citations


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="multi-hop integration test requires INTEGRATION_TESTS=1, seeded Postgres, and provider keys",
)


@pytest.mark.asyncio
async def test_multi_hop_walk_reaches_reflexion_and_lazy_ingests() -> None:
    before = await _paper_ids()
    result = None
    started_at = time.monotonic()
    try:
        result = await walk_citations(
            ["arxiv:2210.03629"],
            "How is reflection added on top of ReAct in Reflexion?",
            frontier_limit=2,
            max_depth=1,
        )
        elapsed = time.monotonic() - started_at
        print(f"multi_hop_elapsed_s={elapsed:.2f}")
        print(f"visited={result.visited_paper_ids}")
        print(f"newly_ingested={result.newly_ingested_paper_ids}")

        assert "arxiv:2303.11366" in result.visited_paper_ids
        assert len(result.newly_ingested_paper_ids) >= 1
        assert elapsed < MULTI_HOP_WALL_CLOCK_BUDGET_S
    finally:
        after = await _paper_ids()
        created = set(result.newly_ingested_paper_ids if result is not None else []) | (after - before)
        if created:
            await _delete_papers(sorted(created))
        await close_llm_clients()
        await db.close_pools()


async def _paper_ids() -> set[str]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch("SELECT paper_id FROM papers")
    return {row["paper_id"] for row in rows}


async def _delete_papers(paper_ids: list[str]) -> None:
    async with db.acquire_app() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM citations WHERE citing_paper_id = ANY($1::text[])", paper_ids)
            await conn.execute("DELETE FROM citations WHERE cited_paper_id = ANY($1::text[])", paper_ids)
            await conn.execute("DELETE FROM chunks WHERE paper_id = ANY($1::text[])", paper_ids)
            await conn.execute("DELETE FROM papers WHERE paper_id = ANY($1::text[])", paper_ids)
