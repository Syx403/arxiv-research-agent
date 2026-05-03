from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.core import db
from src.corpus.arxiv_client import GLOBAL_ARXIV_CLIENT, PaperMetadata, normalize_arxiv_id
from src.corpus.chunking import Chunk, chunk_sections
from src.corpus.pdf_loader import load_pdf
from src.corpus.semantic_scholar import GLOBAL_SEMANTIC_SCHOLAR_CLIENT, S2Reference
from src.llm.client import get_embedding_client


logger = logging.getLogger(__name__)
PDF_DIR = Path("data/pdfs")
EMBED_BATCH_SIZE = 256


@dataclass(frozen=True)
class IngestResult:
    paper_id: str
    status: str
    chunks_inserted: int = 0
    citations_inserted: int = 0


arxiv_client = GLOBAL_ARXIV_CLIENT
semantic_scholar = GLOBAL_SEMANTIC_SCHOLAR_CLIENT


async def ingest_paper(arxiv_id: str, *, allow_redownload: bool = False) -> str:
    result = await ingest_paper_with_result(arxiv_id, allow_redownload=allow_redownload)
    return result.paper_id


async def ingest_paper_with_result(
    arxiv_id: str,
    *,
    allow_redownload: bool = False,
) -> IngestResult:
    normalized_id = normalize_arxiv_id(arxiv_id)
    paper_id = f"arxiv:{normalized_id}"
    status = await _get_ingestion_status(paper_id)
    if status == "complete":
        return IngestResult(paper_id=paper_id, status="skipped")

    retrying_failed = status in {"failed", "in_progress"}
    try:
        metadata = await arxiv_client.fetch_metadata(normalized_id)
        pdf_path = PDF_DIR / f"{normalized_id.replace('/', '_')}.pdf"
        if allow_redownload or retrying_failed or not pdf_path.exists():
            pdf_path = await arxiv_client.download_pdf(normalized_id, PDF_DIR)
        sections = load_pdf(pdf_path)
        chunks = chunk_sections(sections)
        if not chunks:
            raise ValueError(f"No chunks extracted from {paper_id}")
        embeddings = await _embed_chunks(chunks)
        try:
            references = await semantic_scholar.fetch_references(normalized_id, limit=50)
        except Exception as exc:  # Semantic Scholar is best effort.
            logger.warning("Semantic Scholar references unavailable for %s: %s", paper_id, exc)
            references = []
    except Exception:
        await _mark_failed(paper_id)
        raise

    try:
        chunks_inserted, citations_inserted = await _write_ingestion(
            metadata=metadata,
            pdf_path=pdf_path,
            chunks=chunks,
            embeddings=embeddings,
            references=references,
            clean_existing=retrying_failed,
        )
    except Exception:
        await _mark_failed(paper_id, metadata=metadata)
        raise

    return IngestResult(
        paper_id=paper_id,
        status="ingested",
        chunks_inserted=chunks_inserted,
        citations_inserted=citations_inserted,
    )


async def _get_ingestion_status(paper_id: str) -> str | None:
    async with db.acquire_app() as conn:
        return await conn.fetchval(
            "SELECT ingestion_status FROM papers WHERE paper_id = $1",
            paper_id,
        )


async def _embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    embedder = get_embedding_client("embed-small")
    embeddings: list[list[float]] = []
    texts = [chunk.text for chunk in chunks]
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        embeddings.extend(await embedder.embed(texts[start : start + EMBED_BATCH_SIZE]))
    return embeddings


async def _write_ingestion(
    *,
    metadata: PaperMetadata,
    pdf_path: Path,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    references: list[S2Reference],
    clean_existing: bool,
) -> tuple[int, int]:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts differ")

    async with db.acquire_app() as conn:
        async with conn.transaction():
            if clean_existing:
                await conn.execute("DELETE FROM citations WHERE citing_paper_id = $1", metadata.paper_id)
                await conn.execute("DELETE FROM chunks WHERE paper_id = $1", metadata.paper_id)
            await conn.execute(
                """
                INSERT INTO papers (
                  paper_id, source, title, authors, abstract, published_at,
                  url, pdf_path, ingestion_status, raw_metadata
                )
                VALUES ($1, 'arxiv', $2, $3::jsonb, $4, $5, $6, $7, 'in_progress', $8::jsonb)
                ON CONFLICT (paper_id) DO UPDATE SET
                  source = EXCLUDED.source,
                  title = EXCLUDED.title,
                  authors = EXCLUDED.authors,
                  abstract = EXCLUDED.abstract,
                  published_at = EXCLUDED.published_at,
                  url = EXCLUDED.url,
                  pdf_path = EXCLUDED.pdf_path,
                  ingestion_status = 'in_progress',
                  raw_metadata = EXCLUDED.raw_metadata
                """,
                metadata.paper_id,
                metadata.title,
                json.dumps(metadata.authors),
                metadata.abstract,
                metadata.published_at,
                metadata.url,
                str(pdf_path),
                json.dumps(metadata.raw),
            )
            chunk_rows = [
                (metadata.paper_id, chunk.section, idx, chunk.text, chunk.token_count, embedding)
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
            ]
            await conn.executemany(
                """
                INSERT INTO chunks (paper_id, section, ord, text, token_count, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (paper_id, ord) DO UPDATE SET
                  section = EXCLUDED.section,
                  text = EXCLUDED.text,
                  token_count = EXCLUDED.token_count,
                  embedding = EXCLUDED.embedding
                """,
                chunk_rows,
            )
            citation_rows = [
                (metadata.paper_id, reference.paper_id, reference.title)
                for reference in references
                if reference.paper_id
            ]
            if citation_rows:
                await conn.executemany(
                    """
                    INSERT INTO citations (citing_paper_id, cited_paper_id, cited_title)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (citing_paper_id, cited_paper_id) DO UPDATE SET
                      cited_title = EXCLUDED.cited_title
                    """,
                    citation_rows,
                )
            await conn.execute(
                "UPDATE papers SET ingestion_status = 'complete' WHERE paper_id = $1",
                metadata.paper_id,
            )
    return len(chunks), len({row[1] for row in citation_rows})


async def _mark_failed(paper_id: str, metadata: PaperMetadata | None = None) -> None:
    title = metadata.title if metadata else paper_id
    authors = json.dumps(metadata.authors if metadata else [])
    abstract = metadata.abstract if metadata else None
    published_at = metadata.published_at if metadata else None
    url = metadata.url if metadata else None
    raw_metadata = json.dumps(metadata.raw if metadata else {})
    async with db.acquire_app() as conn:
        async with conn.transaction():
            await conn.execute("DELETE FROM citations WHERE citing_paper_id = $1", paper_id)
            await conn.execute("DELETE FROM chunks WHERE paper_id = $1", paper_id)
            await conn.execute(
                """
                INSERT INTO papers (
                  paper_id, source, title, authors, abstract, published_at,
                  url, ingestion_status, raw_metadata
                )
                VALUES ($1, 'arxiv', $2, $3::jsonb, $4, $5, $6, 'failed', $7::jsonb)
                ON CONFLICT (paper_id) DO UPDATE SET
                  title = EXCLUDED.title,
                  authors = EXCLUDED.authors,
                  abstract = EXCLUDED.abstract,
                  published_at = EXCLUDED.published_at,
                  url = EXCLUDED.url,
                  ingestion_status = 'failed',
                  raw_metadata = EXCLUDED.raw_metadata
                """,
                paper_id,
                title,
                authors,
                abstract,
                published_at,
                url,
                raw_metadata,
            )
