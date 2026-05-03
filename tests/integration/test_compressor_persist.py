from __future__ import annotations

import os
from uuid import uuid4

import pytest

from src.core import db
from src.core.types import ChatResponse
from src.memory.episodic import compressor
from src.memory.episodic.session_store import append_message, get_or_create_session


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="compressor persistence test requires INTEGRATION_TESTS=1 and running Postgres",
)


class _FakeChatClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            content=f"persisted summary {self.calls}",
            model="fake-fast",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_compressor_persists_incremental_summaries(monkeypatch) -> None:
    session_id = f"phase7-{uuid4()}"
    client = _FakeChatClient()
    monkeypatch.setattr(compressor, "get_chat_client", lambda name: client)

    try:
        await get_or_create_session(session_id, title="Phase 7 integration")
        for idx in range(30):
            role = "user" if idx % 2 == 0 else "assistant"
            await append_message(session_id, role, f"message {idx}", metadata={"idx": idx})

        first = await compressor.compressed_history(session_id, verbatim_turns=6)
        assert len(first) == 7
        assert first[0].role == "system"
        assert await _summary_count(session_id) == 1

        for idx in range(30, 34):
            role = "user" if idx % 2 == 0 else "assistant"
            await append_message(session_id, role, f"message {idx}", metadata={"idx": idx})

        second = await compressor.compressed_history(session_id, verbatim_turns=6)
        assert len(second) == 7
        assert first[-1].content == "message 29"
        assert second[-1].content == "message 33"
        assert await _summary_count(session_id) == 2
        assert client.calls == 2
    finally:
        async with db.acquire_app() as conn:
            await conn.execute("DELETE FROM sessions WHERE session_id = $1", session_id)
        await db.close_pools()


async def _summary_count(session_id: str) -> int:
    async with db.acquire_app() as conn:
        return int(
            await conn.fetchval(
                "SELECT count(*) FROM session_summaries WHERE session_id = $1",
                session_id,
            )
        )
