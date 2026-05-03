from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

from src.core import db
from src.core.types import ChatResponse
from src.memory import reflection
from src.memory.episodic.session_store import append_message, get_or_create_session
from src.memory.semantic import extractor, linker
from src.memory.semantic.graph_store import _canonical


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="reflection integration test requires INTEGRATION_TESTS=1, seeded Postgres, and running DB",
)


class _FakeExtractorChatClient:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    async def chat(self, *args, **kwargs) -> ChatResponse:
        concepts = [
            {
                "display_name": f"{self.prefix} concept {idx}",
                "definition": f"Definition for {self.prefix} concept {idx}.",
                "aliases": [f"{self.prefix} alias {idx}"],
                "evidence_chunk_ids": [self.chunk_id],
                "relations": [],
            }
            for idx in range(5)
        ]
        concepts[0]["relations"] = [
            {
                "target_display": f"{self.prefix} concept 1",
                "type": "uses",
                "evidence_chunk_ids": [self.chunk_id],
            }
        ]
        return ChatResponse(
            content=json.dumps({"concepts": concepts}),
            model="fake-main",
            finish_reason="stop",
        )


class _FakeEmbeddingClient:
    async def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]:
        return [[0.03] * 1536 for _ in texts]


@pytest.mark.asyncio
async def test_reflection_loop_is_idempotent(monkeypatch) -> None:
    prefix = f"phase8-{uuid4()}"
    session_id = f"{prefix}-session"
    chunk_id, paper_id = await _sample_chunk()
    fake_chat = _FakeExtractorChatClient(prefix)
    fake_chat.chunk_id = chunk_id

    monkeypatch.setattr(extractor, "get_chat_client", lambda name: fake_chat)
    monkeypatch.setattr(linker, "get_embedding_client", lambda name: _FakeEmbeddingClient())
    monkeypatch.setattr(linker, "_nearest_by_embedding", _no_nearest_match)

    try:
        await get_or_create_session(session_id, title="Phase 8 reflection")
        metadata = {"citations": [{"chunk_id": chunk_id, "paper_id": paper_id, "claim_span": [0, 10]}]}
        await append_message(session_id, "assistant", "Answer one with citations.", metadata=metadata)
        await append_message(session_id, "assistant", "Answer two with citations.", metadata=metadata)

        first = await reflection.reflect_session(session_id)
        second = await reflection.reflect_session(session_id)

        assert first.concepts_created >= 5
        assert first.relations_created >= 1
        assert await _new_concepts_have_embeddings(prefix)
        assert second.concepts_created == 0
        assert second.relations_created == 0
        assert await _relation_evidence_has_no_duplicates(prefix)
    finally:
        await _cleanup(session_id, prefix)
        await db.close_pools()


async def _no_nearest_match(*args, **kwargs):
    return None


async def _sample_chunk() -> tuple[int, str]:
    async with db.acquire_app() as conn:
        row = await conn.fetchrow(
            """
            SELECT chunk_id, paper_id
            FROM chunks
            ORDER BY chunk_id
            LIMIT 1
            """
        )
    assert row is not None
    return int(row["chunk_id"]), row["paper_id"]


async def _relation_evidence_has_no_duplicates(prefix: str) -> bool:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            SELECT r.evidence_chunks
            FROM concept_relations r
            JOIN concepts c ON c.concept_id = r.src_id
            WHERE c.canonical LIKE $1
            """,
            f"{_canonical(prefix)}%",
        )
    assert rows
    for row in rows:
        evidence = row["evidence_chunks"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        if len(evidence) != len(set(evidence)):
            return False
    return True


async def _new_concepts_have_embeddings(prefix: str) -> bool:
    async with db.acquire_app() as conn:
        missing = await conn.fetchval(
            """
            SELECT count(*)
            FROM concepts
            WHERE canonical LIKE $1
              AND embedding IS NULL
            """,
            f"{_canonical(prefix)}%",
        )
    return int(missing) == 0


async def _cleanup(session_id: str, prefix: str) -> None:
    async with db.acquire_app() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM sessions WHERE session_id = $1", session_id)
            await conn.execute("DELETE FROM concepts WHERE canonical LIKE $1", f"{_canonical(prefix)}%")
