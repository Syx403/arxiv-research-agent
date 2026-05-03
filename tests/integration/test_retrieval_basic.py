from __future__ import annotations

import os
import time

import pytest

from src.core import db
from src.llm.client import close_llm_clients, get_embedding_client
from src.retrieval.index.hybrid_search import hybrid_search


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="retrieval integration tests require INTEGRATION_TESTS=1, running Postgres, and seeded corpus",
)


@pytest.mark.asyncio
async def test_hybrid_search_finds_react_top_five_and_is_fast() -> None:
    query = "ReAct reasoning and acting interleaved"
    try:
        embedding = (await get_embedding_client("embed-small").embed([query]))[0]
        await hybrid_search(query, embedding, top_k_final=5)

        start = time.perf_counter()
        hits = await hybrid_search(query, embedding, top_k_final=5)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"hybrid_search_latency_ms={elapsed_ms:.2f}")
        print("top5=", [(hit.chunk_id, hit.paper_id, hit.section) for hit in hits])

        assert elapsed_ms < 200
        assert any(hit.paper_id == "arxiv:2210.03629" for hit in hits)
    finally:
        await close_llm_clients()
        await db.close_pools()
