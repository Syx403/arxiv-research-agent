from __future__ import annotations

from pathlib import Path


SQL_DIR = Path("src/core/sql")
EXPECTED_SQL_FILES = [
    "001_papers.sql",
    "002_chunks.sql",
    "003_citations.sql",
    "004_concepts.sql",
    "005_sessions.sql",
    "006_session_summaries.sql",
    "007_triggers.sql",
]


def test_sql_files_present_in_lexical_order() -> None:
    files = [path.name for path in sorted(SQL_DIR.glob("*.sql"))]
    assert files == EXPECTED_SQL_FILES


def test_sql_files_are_idempotent_or_replace_safe() -> None:
    for path in SQL_DIR.glob("*.sql"):
        text = path.read_text()
        assert "CREATE TABLE IF NOT EXISTS" in text or "CREATE OR REPLACE FUNCTION" in text
        for statement in ("CREATE INDEX", "DROP TRIGGER"):
            if statement in text:
                assert f"{statement} IF " in text


def test_chunks_schema_contains_vector_tsv_and_indexes() -> None:
    text = (SQL_DIR / "002_chunks.sql").read_text()
    assert "embedding    vector(1536) NOT NULL" in text
    assert "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED" in text
    assert "USING hnsw (embedding vector_cosine_ops)" in text
    assert "USING gin (tsv)" in text
