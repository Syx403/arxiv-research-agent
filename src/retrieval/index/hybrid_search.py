from __future__ import annotations

from typing import Any

from src.core import db
from src.retrieval.index.types import Hit


HYBRID_SEARCH_SQL = """
WITH vec AS (
  SELECT chunk_id,
         ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rnk
  FROM chunks
  WHERE ($2::text[] IS NULL OR paper_id = ANY($2))
  ORDER BY embedding <=> $1::vector
  LIMIT $3
),
lex AS (
  SELECT chunk_id,
         ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $4)) DESC) AS rnk
  FROM chunks
  WHERE tsv @@ plainto_tsquery('english', $4)
    AND ($2::text[] IS NULL OR paper_id = ANY($2))
  ORDER BY ts_rank_cd(tsv, plainto_tsquery('english', $4)) DESC
  LIMIT $5
),
fused AS (
  SELECT chunk_id, SUM(1.0 / ($6::float + rnk)) AS score
  FROM (
    SELECT chunk_id, rnk FROM vec
    UNION ALL
    SELECT chunk_id, rnk FROM lex
  ) AS combined
  GROUP BY chunk_id
)
SELECT c.chunk_id, c.paper_id, c.section, c.text, f.score
FROM fused f
JOIN chunks c ON c.chunk_id = f.chunk_id
ORDER BY f.score DESC
LIMIT $7;
"""


async def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    *,
    top_k_vector: int = 30,
    top_k_lexical: int = 30,
    rrf_k: int = 60,
    top_k_final: int = 30,
    filter_paper_ids: list[str] | None = None,
) -> list[Hit]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            HYBRID_SEARCH_SQL,
            query_embedding,
            filter_paper_ids,
            top_k_vector,
            query_text,
            top_k_lexical,
            rrf_k,
            top_k_final,
        )
    return [_row_to_hit(row) for row in rows]


def _row_to_hit(row: Any) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"],
        paper_id=row["paper_id"],
        section=row["section"],
        text=row["text"],
        score=float(row["score"]),
    )
