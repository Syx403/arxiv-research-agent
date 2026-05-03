from __future__ import annotations

import os
from uuid import uuid4

import pytest

from src.core import db
from src.core.types import ChatResponse
from src.memory.semantic import graph_retriever, graph_store


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="graph retriever integration test requires INTEGRATION_TESTS=1, seeded Postgres, and running DB",
)


class _FakeMentionClient:
    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases

    async def chat(self, *args, **kwargs) -> ChatResponse:
        phrase_json = ", ".join(f'"{phrase}"' for phrase in self.phrases)
        return ChatResponse(content=f'{{"phrases":[{phrase_json}]}}', model="fake-fast", finish_reason="stop")


@pytest.mark.asyncio
async def test_graph_retriever_returns_paper_hints_from_relation_evidence(monkeypatch) -> None:
    token = f"phase8-{uuid4()}"
    phrase_a = f"{token} chain-of-thought prompting"
    phrase_b = f"{token} in-context learning"
    concept_ids: list[int] = []
    monkeypatch.setattr(graph_retriever, "get_chat_client", lambda name: _FakeMentionClient([phrase_a, phrase_b]))

    try:
        chunks = await _seed_chunks()
        concept_a = await graph_store.upsert_concept(
            f"{token} Chain-of-thought prompting",
            definition="Reasoning trace prompting.",
            evidence_chunk_ids=[chunks[0][0]],
            embedding=_vector(0.01),
            aliases=[phrase_a],
        )
        concept_b = await graph_store.upsert_concept(
            f"{token} In-context learning",
            definition="Learning behavior induced by context examples.",
            evidence_chunk_ids=[chunks[1][0]],
            embedding=_vector(0.02),
            aliases=[phrase_b],
        )
        concept_ids.extend([concept_a.concept_id, concept_b.concept_id])
        await graph_store.upsert_relation(
            concept_a.concept_id,
            concept_b.concept_id,
            "compares",
            [chunks[0][0], chunks[1][0]],
        )

        expansion = await graph_retriever.expand_query_via_graph(
            "How does chain-of-thought relate to in-context learning?"
        )

        expected_papers = {chunks[0][1], chunks[1][1]}
        assert len(expansion.extra_paper_hints) >= 2
        assert expected_papers.issubset(set(expansion.extra_paper_hints))
        assert concept_a.concept_id in expansion.seed_concept_ids
        assert concept_b.concept_id in expansion.seed_concept_ids
    finally:
        if concept_ids:
            await _delete_concepts(concept_ids)
        await db.close_pools()


async def _seed_chunks() -> list[tuple[int, str]]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT ON (paper_id) chunk_id, paper_id
            FROM chunks
            WHERE paper_id = ANY($1::text[])
            ORDER BY paper_id, chunk_id
            """,
            ["arxiv:2201.11903", "arxiv:2210.03629"],
        )
    chunks = [(int(row["chunk_id"]), row["paper_id"]) for row in rows]
    assert len(chunks) == 2
    return chunks


async def _delete_concepts(concept_ids: list[int]) -> None:
    async with db.acquire_app() as conn:
        await conn.execute("DELETE FROM concepts WHERE concept_id = ANY($1::bigint[])", concept_ids)


def _vector(value: float) -> list[float]:
    return [value] * 1536
