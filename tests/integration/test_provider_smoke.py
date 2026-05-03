from __future__ import annotations

import os

import pytest

from src.core.types import EMBEDDING_DIM, Message
from src.llm.client import close_llm_clients, get_chat_client, get_embedding_client, get_rerank_client


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="provider smoke tests require INTEGRATION_TESTS=1 and live provider keys",
)


@pytest.mark.asyncio
async def test_cost_minimized_provider_smoke() -> None:
    try:
        chat = get_chat_client("fast")
        chat_response = await chat.chat([Message(role="user", content="Hi")], max_tokens=1)
        assert chat_response.model

        embedder = get_embedding_client("embed-small")
        vectors = await embedder.embed(["hello"])
        assert len(vectors) == 1
        assert len(vectors[0]) == EMBEDDING_DIM

        reranker = get_rerank_client("rerank")
        results = await reranker.rerank("hello", ["hello world"], top_n=1)
        assert len(results) == 1
    finally:
        await close_llm_clients()
