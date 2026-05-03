from __future__ import annotations

import os

import pytest

from src.core import db
from src.corpus.ingestion import ingest_paper
from src.llm.client import close_llm_clients


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="ingestion integration tests require INTEGRATION_TESTS=1 and live services",
)


@pytest.mark.asyncio
async def test_ingest_one_small_paper_end_to_end() -> None:
    arxiv_id = "2210.03629"
    paper_id = f"arxiv:{arxiv_id}"
    try:
        async with db.acquire_app() as conn:
            await conn.execute("DELETE FROM papers WHERE paper_id = $1", paper_id)

        assert await ingest_paper(arxiv_id, allow_redownload=True) == paper_id

        async with db.acquire_app() as conn:
            status = await conn.fetchval(
                "SELECT ingestion_status FROM papers WHERE paper_id = $1",
                paper_id,
            )
            chunk_count = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE paper_id = $1",
                paper_id,
            )
            citation_count = await conn.fetchval(
                "SELECT count(*) FROM citations WHERE citing_paper_id = $1",
                paper_id,
            )

        assert status == "complete"
        assert chunk_count > 0
        assert citation_count >= 1
    finally:
        await close_llm_clients()
        await db.close_pools()
