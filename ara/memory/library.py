"""The user's library (DESIGN §7, §8): papers read, in SQL; the library-wide search that ranks
them for a question or a need (D27, D30); and the passages a screen sees with each (D30)."""

from ara.db.pool import Connection
from ara.graph.state import PaperCard
from ara.rag.embed import Vector
from ara.rag.ingest import PIPELINE_VERSION

CANDIDATES = 50  # nearest chunks looked at when ranking papers
PASSAGE_CHARS = 1_000  # of each passage shown to the screen

RECORD = """
INSERT INTO library_items (user_id, paper_id)
SELECT %s, paper_id FROM documents WHERE id = ANY(%s)
ON CONFLICT (user_id, paper_id) DO UPDATE SET last_read_at = now()
"""
PAPERS = """
SELECT p.arxiv_id, p.version, p.title, p.abstract, p.published FROM library_items l
JOIN papers p ON p.id = l.paper_id
WHERE l.user_id = %s AND p.source = 'arxiv' ORDER BY l.first_read_at
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
# Exact scan over a few papers' chunks: the k nearest of each, for the screen (D30).
NEAREST_EACH = """
SELECT paper_id, heading_path, text FROM (
    SELECT d.paper_id, c.heading_path, c.text,
           row_number() OVER (PARTITION BY d.paper_id ORDER BY c.embedding <=> %(vector)s) AS n
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE d.pipeline_version = %(pipeline)s AND d.paper_id = ANY(%(papers)s)
) nearest WHERE n <= %(k)s ORDER BY paper_id, n
"""


async def record_read(conn: Connection, user: str, documents: list[int]) -> None:
    """The papers behind the documents just read join the user's library (again: a newer date)."""
    await conn.execute(RECORD, (user, documents))


async def papers(conn: Connection, user: str) -> list[PaperCard]:
    """The user's arXiv papers, in the order first read; `published` is "" for a paper stored
    before migration 0005 and not yet backfilled."""
    cursor = await conn.execute(PAPERS, (user,))
    return [
        PaperCard(
            arxiv_id=row["arxiv_id"],
            version=row["version"],
            title=row["title"],
            abstract=row["abstract"],
            published=row["published"].isoformat() if row["published"] else "",
        )
        for row in await cursor.fetchall()
    ]


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


async def passages_near(
    conn: Connection, papers: list[str], vector: Vector, k: int
) -> dict[str, list[str]]:
    """The k passages of each paper nearest `vector`, as "heading: text", nearest first."""
    cursor = await conn.execute(
        NEAREST_EACH, {"vector": vector, "pipeline": PIPELINE_VERSION, "papers": papers, "k": k}
    )
    found: dict[str, list[str]] = {p: [] for p in papers}
    for row in await cursor.fetchall():
        found[row["paper_id"]].append(f"{row['heading_path']}: {row['text'][:PASSAGE_CHARS]}")
    return found
