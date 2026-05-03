from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.types import ChatResponse, StoredMessage
from src.memory.episodic import compressor


class _CountingChatClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args, **kwargs) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            content=f"summary call {self.calls}",
            model="fake-fast",
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_compressed_history_uses_cache_and_incremental_summary(monkeypatch) -> None:
    session_id = "unit-session"
    messages = [_message(idx, session_id=session_id) for idx in range(1, 11)]
    summaries: dict[int, str] = {}
    client = _CountingChatClient()

    async def fake_fetch_all_messages(_session_id: str) -> list[StoredMessage]:
        return list(messages)

    async def fake_get_cached_summary(_session_id: str, cutoff_id: int) -> compressor.CachedSummary | None:
        eligible = [up_to for up_to in summaries if up_to <= cutoff_id]
        if not eligible:
            return None
        up_to = max(eligible)
        return compressor.CachedSummary(summary=summaries[up_to], up_to_message_id=up_to)

    async def fake_upsert_summary(_session_id: str, cutoff_id: int, summary: str) -> None:
        summaries.setdefault(cutoff_id, summary)

    monkeypatch.setattr(compressor, "_fetch_all_messages", fake_fetch_all_messages)
    monkeypatch.setattr(compressor, "_get_cached_summary", fake_get_cached_summary)
    monkeypatch.setattr(compressor, "_upsert_summary", fake_upsert_summary)
    monkeypatch.setattr(compressor, "get_chat_client", lambda name: client)

    first = await compressor.compressed_history(session_id, verbatim_turns=6)
    second = await compressor.compressed_history(session_id, verbatim_turns=6)

    assert client.calls == 1
    assert len(first) == 7
    assert len(second) == 7
    assert set(summaries) == {4}

    messages.extend(_message(idx, session_id=session_id) for idx in range(11, 15))
    third = await compressor.compressed_history(session_id, verbatim_turns=6)

    assert client.calls == 2
    assert len(third) == 7
    assert set(summaries) == {4, 8}


def _message(message_id: int, *, session_id: str) -> StoredMessage:
    return StoredMessage(
        message_id=message_id,
        session_id=session_id,
        role="user" if message_id % 2 else "assistant",
        content=f"message {message_id}",
        metadata={},
        created_at=datetime.now(timezone.utc),
    )
