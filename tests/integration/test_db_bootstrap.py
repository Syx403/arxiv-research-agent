from __future__ import annotations

import os

import pytest

from scripts.bootstrap_db import bootstrap
from src.core import db
from src.core.types import EMBEDDING_DIM


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="database integration tests require INTEGRATION_TESTS=1 and running Postgres",
)


EXPECTED_TABLES = {
    "papers",
    "chunks",
    "citations",
    "concepts",
    "concept_aliases",
    "concept_relations",
    "sessions",
    "session_messages",
    "session_summaries",
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}


@pytest.mark.asyncio
async def test_bootstrap_creates_schema_extensions_indexes_trigger_and_vector_roundtrip() -> None:
    try:
        await bootstrap()

        async with db.acquire_app() as conn:
            table_rows = await conn.fetch(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                """
            )
            assert EXPECTED_TABLES.issubset({row["tablename"] for row in table_rows})

            extension_rows = await conn.fetch("SELECT extname FROM pg_extension")
            assert {"vector", "pg_trgm"}.issubset({row["extname"] for row in extension_rows})

            index_rows = await conn.fetch(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE schemaname = 'public'
                """
            )
            index_names = {row["indexname"] for row in index_rows}
            assert "chunks_embedding_hnsw" in index_names
            assert "chunks_tsv_idx" in index_names
            assert "concept_relations_dst_idx" in index_names

            await conn.execute(
                """
                INSERT INTO concepts (canonical, display, definition)
                VALUES ('phase2-test', 'Phase 2 Test', 'trigger check')
                ON CONFLICT (canonical) DO UPDATE SET definition = EXCLUDED.definition
                """
            )
            before = await conn.fetchval(
                "SELECT updated_at FROM concepts WHERE canonical = 'phase2-test'"
            )
            await conn.execute(
                "UPDATE concepts SET definition = 'trigger check updated' WHERE canonical = 'phase2-test'"
            )
            after = await conn.fetchval(
                "SELECT updated_at FROM concepts WHERE canonical = 'phase2-test'"
            )
            assert after >= before

            # pgvector stores values as float4 (single precision); compare
            # with pytest.approx because exact double equality cannot survive
            # the round-trip through 32-bit storage.
            vector = [0.001] * EMBEDDING_DIM
            await conn.execute(
                """
                INSERT INTO papers (paper_id, source, title)
                VALUES ('test:vector-roundtrip', 'arxiv', 'Vector Roundtrip')
                ON CONFLICT (paper_id) DO UPDATE SET title = EXCLUDED.title
                """
            )
            chunk_id = await conn.fetchval(
                """
                INSERT INTO chunks (paper_id, section, ord, text, token_count, embedding)
                VALUES ('test:vector-roundtrip', 'Test', 1, 'Vector roundtrip text', 3, $1)
                ON CONFLICT (paper_id, index_key, ord) DO UPDATE SET embedding = EXCLUDED.embedding
                RETURNING chunk_id
                """,
                vector,
            )
            selected = await conn.fetchval(
                "SELECT embedding FROM chunks WHERE chunk_id = $1",
                chunk_id,
            )
            assert list(selected) == pytest.approx(vector)
    finally:
        await db.close_pools()
