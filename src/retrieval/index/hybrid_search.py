from __future__ import annotations

from typing import Any
import re

from src.core import db
from src.retrieval.index.types import Hit


HYBRID_SEARCH_SQL = """
WITH vec AS (
  SELECT chunk_id,
         ROW_NUMBER() OVER (ORDER BY embedding <=> $1::vector) AS rnk
  FROM chunks
  WHERE index_key = COALESCE((SELECT raw_metadata->>'index_key' FROM papers WHERE papers.paper_id=chunks.paper_id), 'legacy') AND ($2::text[] IS NULL OR paper_id = ANY($2))
  ORDER BY embedding <=> $1::vector
  LIMIT $3
),
lex AS (
  SELECT chunk_id,
         ROW_NUMBER() OVER (ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $4)) DESC) AS rnk
  FROM chunks
  WHERE index_key = COALESCE((SELECT raw_metadata->>'index_key' FROM papers WHERE papers.paper_id=chunks.paper_id), 'legacy') AND tsv @@ websearch_to_tsquery('english', $4)
    AND ($2::text[] IS NULL OR paper_id = ANY($2))
  ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $4)) DESC
  LIMIT $5
),
named AS (
  SELECT c.chunk_id, ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector) AS rnk
  FROM chunks c JOIN papers p ON p.paper_id = c.paper_id
  WHERE p.ingestion_status = 'complete' AND c.index_key = COALESCE(p.raw_metadata->>'index_key', 'legacy')
    AND ($2::text[] IS NULL OR c.paper_id = ANY($2))
    AND length(split_part(p.title, ':', 1)) BETWEEN 3 AND 100
    AND position(
      ' ' || trim(regexp_replace(lower(split_part(p.title, ':', 1)), '[^a-z0-9]+', ' ', 'g')) || ' '
      IN ' ' || trim(regexp_replace(lower($8::text), '[^a-z0-9]+', ' ', 'g')) || ' '
    ) > 0
  ORDER BY c.embedding <=> $1::vector
  LIMIT 10
),
fused AS (
  SELECT chunk_id, SUM(1.0 / ($6::float + rnk)) AS score
  FROM (
    SELECT chunk_id, rnk FROM vec
    UNION ALL
    SELECT chunk_id, rnk FROM lex
    UNION ALL
    SELECT chunk_id, rnk FROM named
  ) AS combined
  GROUP BY chunk_id
)
SELECT c.chunk_id, c.paper_id, c.section, c.text, f.score, p.title, p.published_at, p.url
FROM fused f
JOIN chunks c ON c.chunk_id = f.chunk_id
JOIN papers p ON p.paper_id = c.paper_id
WHERE p.ingestion_status = 'complete' AND c.index_key = COALESCE(p.raw_metadata->>'index_key', 'legacy')
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
    original_query: str | None = None,
) -> list[Hit]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            HYBRID_SEARCH_SQL,
            query_embedding,
            filter_paper_ids,
            top_k_vector,
            lexical_query(query_text),
            top_k_lexical,
            rrf_k,
            top_k_final,
            original_query or query_text,
        )
    return [_row_to_hit(row) for row in rows]


def _row_to_hit(row: Any) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"],
        paper_id=row["paper_id"],
        section=row["section"],
        text=row["text"],
        score=float(row["score"]),
        title=row.get("title", ""),
        published_at=str(row["published_at"]) if row.get("published_at") else None,
        url=row.get("url") or "",
    )


def lexical_query(query: str) -> str:
    """Broad lexical recall; semantic relevance is decided by the reranker.

    Only parsed words enter query syntax. Adding a query term cannot impose a new
    mandatory AND constraint on all the other terms.
    """
    ignored = {"what", "how", "does", "the", "a", "an", "and", "or", "of", "is", "are", "in", "for",
               "with", "its", "core", "explain", "compare", "comparison", "specifically"}
    terms = dict.fromkeys(word for word in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", query.lower())
                          if word not in ignored)
    return " OR ".join('"' + word + '"' for word in list(terms)[:16])
