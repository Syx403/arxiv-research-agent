"""Search primitives (DESIGN §5), each scoped to a set of documents and returning chunk ids, best
first: BM25 (two tokenizers, D15), native full-text search, exact dense search, and reciprocal rank
fusion. Composing them into the product pipeline happens in the read subgraph (M2)."""

from collections.abc import Hashable, Mapping, Sequence
from typing import LiteralString

from ara.db.pool import Connection
from ara.rag.embed import Vector

POOL = 50  # candidates taken from each list before fusion
RRF_K = 60

# `|||` matches any query token, tokenised like the field; pdb.score is the BM25 score. Every
# ranking breaks ties by id, so a run is reproducible.
BM25_PLAIN = """
SELECT id FROM chunks
WHERE search_text ||| %(query)s AND document_id = ANY(%(documents)s)
ORDER BY pdb.score(id) DESC, id LIMIT %(limit)s
"""
BM25_STEMMED = """
SELECT id FROM chunks
WHERE search_text::pdb.alias('search_text_stemmed') ||| %(query)s
  AND document_id = ANY(%(documents)s)
ORDER BY pdb.score(id) DESC, id LIMIT %(limit)s
"""
# plainto_tsquery joins the terms with AND; turning them into OR matches BM25's any-token query.
FTS = """
SELECT id FROM chunks,
     CAST(replace(plainto_tsquery('english', %(query)s)::text, '&', '|') AS tsquery) AS query
WHERE tsv @@ query AND document_id = ANY(%(documents)s)
ORDER BY ts_rank_cd(tsv, query) DESC, id LIMIT %(limit)s
"""
# The MATERIALIZED scope keeps the planner off any vector index: within a few papers the scan is
# exact and fast.
DENSE = """
WITH scoped AS MATERIALIZED (
    SELECT id, embedding FROM chunks WHERE document_id = ANY(%(documents)s)
)
SELECT id FROM scoped ORDER BY embedding <=> %(vector)s, id LIMIT %(limit)s
"""


async def bm25(
    conn: Connection, query: str, documents: Sequence[int], *, stemmed: bool, limit: int = POOL
) -> list[int]:
    sql = BM25_STEMMED if stemmed else BM25_PLAIN
    return await _ids(conn, sql, {"query": query, "documents": list(documents), "limit": limit})


async def fts(
    conn: Connection, query: str, documents: Sequence[int], limit: int = POOL
) -> list[int]:
    return await _ids(conn, FTS, {"query": query, "documents": list(documents), "limit": limit})


async def dense(
    conn: Connection, vector: Vector, documents: Sequence[int], limit: int = POOL
) -> list[int]:
    return await _ids(conn, DENSE, {"vector": vector, "documents": list(documents), "limit": limit})


def rrf[T: Hashable](rankings: Sequence[Sequence[T]], k: int = RRF_K) -> list[T]:
    """Reciprocal rank fusion: score(d) = Σ 1 / (k + rank); ties keep first appearance."""
    scores: dict[T, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1 / (k + rank)
    return sorted(scores, key=lambda item: -scores[item])


async def _ids(conn: Connection, sql: LiteralString, params: Mapping[str, object]) -> list[int]:
    cursor = await conn.execute(sql, params)
    return [int(row["id"]) for row in await cursor.fetchall()]
