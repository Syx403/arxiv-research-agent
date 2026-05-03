from __future__ import annotations

from typing import Any

from src.core import db
from src.retrieval.index.types import Hit


VECTOR_SEARCH_SQL = """
SELECT chunk_id, paper_id, section, text, 1 - (embedding <=> $1::vector) AS score
FROM chunks
WHERE ($2::text[] IS NULL OR paper_id = ANY($2))
ORDER BY embedding <=> $1::vector
LIMIT $3;
"""

LEXICAL_SEARCH_SQL = """
SELECT chunk_id,
       paper_id,
       section,
       text,
       ts_rank_cd(tsv, plainto_tsquery('english', $1)) AS score
FROM chunks
WHERE tsv @@ plainto_tsquery('english', $1)
  AND ($2::text[] IS NULL OR paper_id = ANY($2))
ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $1)) DESC
LIMIT $3;
"""


async def search_vector(
    query_embedding: list[float],
    top_k: int,
    *,
    filter_paper_ids: list[str] | None = None,
) -> list[Hit]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(VECTOR_SEARCH_SQL, query_embedding, filter_paper_ids, top_k)
    return [_row_to_hit(row) for row in rows]


async def search_lexical(
    query_text: str,
    top_k: int,
    *,
    filter_paper_ids: list[str] | None = None,
) -> list[Hit]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(LEXICAL_SEARCH_SQL, query_text, filter_paper_ids, top_k)
    return [_row_to_hit(row) for row in rows]


def _row_to_hit(row: Any) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"],
        paper_id=row["paper_id"],
        section=row["section"],
        text=row["text"],
        score=float(row["score"]),
    )
