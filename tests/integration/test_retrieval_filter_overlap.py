from __future__ import annotations

import os

import pytest

from src.core import db
from src.llm.client import close_llm_clients, get_embedding_client
from src.retrieval.index.hybrid_search import hybrid_search


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="retrieval integration tests require INTEGRATION_TESTS=1, running Postgres, and seeded corpus",
)


@pytest.mark.asyncio
async def test_hybrid_search_deduplicates_and_honors_filter() -> None:
    query = "ReAct reasoning and acting interleaved"
    filter_paper_id = "arxiv:2210.03629"
    try:
        embedding = (await get_embedding_client("embed-small").embed([query]))[0]
        all_hits = await hybrid_search(
            query,
            embedding,
            top_k_vector=2000,
            top_k_lexical=2000,
            top_k_final=2000,
        )
        filtered_hits = await hybrid_search(
            query,
            embedding,
            top_k_vector=2000,
            top_k_lexical=2000,
            top_k_final=2000,
            filter_paper_ids=[filter_paper_id],
        )

        all_chunk_ids = {hit.chunk_id for hit in all_hits}
        filtered_chunk_ids = {hit.chunk_id for hit in filtered_hits}

        assert len(all_hits) == len(all_chunk_ids)
        assert len(filtered_hits) == len(filtered_chunk_ids)
        assert filtered_hits
        assert all(hit.paper_id == filter_paper_id for hit in filtered_hits)
        assert filtered_chunk_ids < all_chunk_ids
    finally:
        await close_llm_clients()
        await db.close_pools()
