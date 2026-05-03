from __future__ import annotations

import pytest

from src.retrieval.query import rewriter


@pytest.mark.asyncio
async def test_hyde_skips_short_query_without_llm(monkeypatch) -> None:
    def fail_get_client(name: str):
        raise AssertionError("short queries should skip HyDE")

    monkeypatch.setattr(rewriter, "get_chat_client", fail_get_client)

    assert await rewriter.hyde("ReAct basics") == "ReAct basics"
