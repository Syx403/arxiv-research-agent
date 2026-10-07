"""Ingestion (DESIGN §5, §8): parse → chunk → embed → store one document per (paper, pipeline).
Idempotent: a paper already ingested with this PIPELINE_VERSION is returned without any work."""

from ara.db.pool import Connection, Pool, fetch_one
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope
from ara.llm.stages import EMBEDDING_MODEL
from ara.rag.chunking import MAX_TOKENS, chunk
from ara.rag.embed import embed
from ara.rag.sources import ParsedPaper

# Parser, chunker and embedding model. Bump it when any of them changes: chunks of different
# versions are never searched together, because a search is scoped to document ids.
PIPELINE_VERSION = f"parse1-chunk{MAX_TOKENS}-{EMBEDDING_MODEL}"

UPSERT_PAPER = """
INSERT INTO papers (id, source, arxiv_id, version, title, abstract)
VALUES (%(id)s, %(source)s, %(arxiv_id)s, %(version)s, %(title)s, %(abstract)s)
ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, abstract = EXCLUDED.abstract
"""
INSERT_DOCUMENT = """
INSERT INTO documents (paper_id, pipeline_version, format)
VALUES (%(paper_id)s, %(pipeline_version)s, %(format)s)
RETURNING id
"""
INSERT_CHUNK = """
INSERT INTO chunks (document_id, ord, paragraph, heading_path, text, search_text, sentences,
                    embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""


async def ingest(paper: ParsedPaper, *, pool: Pool, gateway: Gateway, scope: Scope) -> int:
    """Store `paper` and return its document id. Embeddings are computed before the transaction;
    the write itself runs under a per-paper advisory lock, so concurrent ingests store it once."""
    async with pool.connection() as conn:
        if (existing := await document_id(conn, paper.id)) is not None:
            return existing
    chunks = chunk(paper.paragraphs)
    vectors = await embed([c.search_text for c in chunks], pool=pool, gateway=gateway, scope=scope)
    async with pool.connection() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (paper.id,))
        if (existing := await document_id(conn, paper.id)) is not None:
            return existing
        await conn.execute(UPSERT_PAPER, _paper_row(paper))
        row = {"paper_id": paper.id, "pipeline_version": PIPELINE_VERSION, "format": paper.format}
        document = int((await fetch_one(conn, INSERT_DOCUMENT, row))["id"])
        await conn.cursor().executemany(
            INSERT_CHUNK,
            [
                (
                    document,
                    c.ord,
                    c.paragraph,
                    c.heading_path,
                    c.text,
                    c.search_text,
                    list(c.sentences),
                    vector,
                )
                for c, vector in zip(chunks, vectors, strict=True)
            ],
        )
        return document


async def document_id(conn: Connection, paper_id: str) -> int | None:
    """The stored document of `paper_id` under this PIPELINE_VERSION, if any."""
    cursor = await conn.execute(
        "SELECT id FROM documents WHERE paper_id = %s AND pipeline_version = %s",
        (paper_id, PIPELINE_VERSION),
    )
    row = await cursor.fetchone()
    return None if row is None else int(row["id"])


def _paper_row(paper: ParsedPaper) -> dict[str, object]:
    return {
        "id": paper.id,
        "source": paper.source,
        "arxiv_id": paper.arxiv_id,
        "version": paper.version,
        "title": paper.title,
        "abstract": paper.abstract,
    }
