"""The user's library (DESIGN §7, §8): papers read, in SQL, and the library-wide search that picks
which of them a library question should read again (D27)."""

from ara.db.pool import Connection
from ara.rag.embed import Vector
from ara.rag.ingest import PIPELINE_VERSION

CANDIDATES = 50  # nearest chunks looked at when choosing papers

RECORD = """
INSERT INTO library_items (user_id, paper_id)
SELECT %s, paper_id FROM documents WHERE id = ANY(%s)
ON CONFLICT (user_id, paper_id) DO UPDATE SET last_read_at = now()
"""
# HNSW over every chunk, filtered to the user's papers; the iterative scan keeps searching the
# index until enough rows pass the filter (pgvector >= 0.8). Ordered by distance alone: a second
# sort key would stop the planner from using the index.
NEAREST = """
SELECT d.paper_id FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.pipeline_version = %(pipeline)s
  AND d.paper_id IN (SELECT paper_id FROM library_items WHERE user_id = %(user)s)
ORDER BY c.embedding <=> %(vector)s LIMIT %(limit)s
"""


async def record_read(conn: Connection, user: str, documents: list[int]) -> None:
    """The papers behind the documents just read join the user's library (again: a newer date)."""
    await conn.execute(RECORD, (user, documents))


async def closest_papers(conn: Connection, user: str, vector: Vector, limit: int) -> list[str]:
    """The user's papers whose passages lie nearest the question, best first, at most `limit`."""
    async with conn.transaction():
        await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cursor = await conn.execute(
            NEAREST,
            {"pipeline": PIPELINE_VERSION, "user": user, "vector": vector, "limit": CANDIDATES},
        )
        ranked = [row["paper_id"] for row in await cursor.fetchall()]
    return list(dict.fromkeys(ranked))[:limit]
