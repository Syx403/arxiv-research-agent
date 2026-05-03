from __future__ import annotations

import os
from uuid import uuid4

import pytest

from src.core import db
from src.graph.builder import run, shutdown


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="graph e2e test requires INTEGRATION_TESTS=1, running Postgres, seeded corpus, and provider keys",
)


@pytest.mark.asyncio
async def test_graph_e2e_answers_with_citations_and_reflection() -> None:
    thread_id = f"e2e-{uuid4()}"
    before_concepts = await _concept_ids()
    try:
        result = await run("What is ReAct?", thread_id=thread_id)
        print("answer=", result["answer"])
        print("citations=", [citation.model_dump() for citation in result["citations"]])
        print("verification=", result["verification"].model_dump() if result["verification"] else None)
        print("reflection=", result["reflection"].model_dump() if result["reflection"] else None)

        assert result["answer"]
        assert len(result["citations"]) >= 1
        assert await _paper_exists(result["citations"][0].paper_id)
        assert result["verification"] is not None
        assert result["verification"].passed is True
        assert result["reflection"] is not None
        assert result["reflection"].concepts_created + result["reflection"].concepts_reused >= 1
    finally:
        await _cleanup(thread_id, before_concepts)
        await shutdown()


async def _paper_exists(paper_id: str) -> bool:
    async with db.acquire_app() as conn:
        return bool(await conn.fetchval("SELECT EXISTS (SELECT 1 FROM papers WHERE paper_id = $1)", paper_id))


async def _concept_ids() -> set[int]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch("SELECT concept_id FROM concepts")
    return {int(row["concept_id"]) for row in rows}


async def _cleanup(thread_id: str, before_concepts: set[int]) -> None:
    after_concepts = await _concept_ids()
    created = sorted(after_concepts - before_concepts)
    async with db.acquire_app() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM sessions WHERE session_id = $1", thread_id)
            if created:
                await conn.execute("DELETE FROM concepts WHERE concept_id = ANY($1::bigint[])", created)
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = $1", thread_id)
            await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = $1", thread_id)
