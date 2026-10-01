from __future__ import annotations

import json
import logging
import asyncio
import hashlib
import httpx
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from src.core import db
from src.core.trace import record_event
from src.corpus.arxiv_client import GLOBAL_ARXIV_CLIENT, PaperMetadata, normalize_arxiv_id
from src.corpus.chunking import Chunk
from src.corpus.parse_worker import parse_chunks
from src.llm.client import get_embedding_client
from src.llm.registry import get_route


logger = logging.getLogger(__name__)
PDF_DIR = Path("data/pdfs")
EMBED_BATCH_SIZE = 256


@dataclass(frozen=True)
class IngestResult:
    paper_id: str
    status: str
    chunks_inserted: int = 0
    citations_inserted: int = 0
    warnings: tuple[str, ...] = ()


arxiv_client = GLOBAL_ARXIV_CLIENT


async def ingest_paper(arxiv_id: str, *, allow_redownload: bool = False) -> str:
    result = await ingest_paper_with_result(arxiv_id, allow_redownload=allow_redownload)
    return result.paper_id


async def ingest_paper_with_result(
    arxiv_id: str,
    *,
    allow_redownload: bool = False,
    metadata: PaperMetadata | None = None,
    max_chunks: int | None = None,
) -> IngestResult:
    base_id = normalize_arxiv_id(arxiv_id)
    metadata = metadata or await arxiv_client.fetch_metadata(
        arxiv_id, fresh=(normalize_arxiv_id(arxiv_id, keep_version=True) == base_id)
    )
    if normalize_arxiv_id(metadata.paper_id) != base_id:
        raise ValueError("Ingestion metadata identity does not match the requested paper")
    normalized_id = metadata.raw.get("version_id")
    if not normalized_id or not metadata.raw.get("version"):
        raise ValueError("Cannot cache full text without a confirmed arXiv version")
    requested = normalize_arxiv_id(arxiv_id, keep_version=True)
    if requested != base_id and requested != normalized_id:
        raise ValueError("Requested paper version does not match metadata")
    paper_id = f"arxiv:{normalized_id}"
    metadata = replace(
        metadata,
        paper_id=paper_id,
        url=f"https://arxiv.org/abs/{normalized_id}",
        pdf_url=f"https://arxiv.org/pdf/{normalized_id}",
        raw={**metadata.raw, "canonical_id": "arxiv:" + base_id, "index_key": index_key()},
    )
    async with _paper_lock(paper_id):
        return await _ingest_locked(
            normalized_id,
            paper_id,
            allow_redownload=allow_redownload,
            metadata=metadata,
            max_chunks=max_chunks,
        )


@asynccontextmanager
async def _paper_lock(paper_id: str):
    # Session-level advisory lock also coordinates independent CLI/UI processes.
    lock_id = int.from_bytes(hashlib.sha256(paper_id.encode()).digest()[:8], "big", signed=True)
    async with db.acquire_app() as conn:
        acquired = False
        try:
            while not acquired:
                acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1::bigint)", lock_id)
                if not acquired:
                    await asyncio.sleep(0.1)
            yield
        finally:
            if acquired:
                await asyncio.shield(conn.execute("SELECT pg_advisory_unlock($1::bigint)", lock_id))


async def _ingest_locked(
    normalized_id, paper_id, *, allow_redownload, metadata, max_chunks
) -> IngestResult:
    status = await _get_ingestion_status(paper_id)
    async with db.acquire_app() as conn:
        stored_key = await conn.fetchval(
            "SELECT raw_metadata->>'index_key' FROM papers WHERE paper_id=$1", paper_id
        )
    if status == "complete" and stored_key == (metadata.raw.get("index_key") if metadata else None):
        return IngestResult(paper_id=paper_id, status="skipped")

    retrying_failed = status in {"failed", "in_progress"}
    warnings = []
    try:
        metadata = metadata or await arxiv_client.fetch_metadata(normalized_id)
        try:
            pdf_path = await arxiv_client.download_html(normalized_id, PDF_DIR.parent / "html")
            chunks = await parse_chunks(pdf_path)
            if not chunks:
                raise ValueError("No usable HTML chunks")
            material_format = "arxiv_html"
        except (httpx.HTTPError, ValueError, TimeoutError) as exc:
            warnings.append("html_unavailable_pdf_fallback")
            record_event("original_fallback", from_format="arxiv_html", to_format="pdf",
                         reason=type(exc).__name__, paper_id=paper_id)
            pdf_path = PDF_DIR / f"{normalized_id.replace('/', '_')}.pdf"
            if allow_redownload or retrying_failed or not pdf_path.exists():
                pdf_path = await arxiv_client.download_pdf(normalized_id, PDF_DIR)
            chunks = await parse_chunks(pdf_path)
            material_format = "pdf"
        metadata = replace(
            metadata,
            raw={
                **metadata.raw,
                "material_format": material_format,
                "material_url": f"https://arxiv.org/{'html' if material_format == 'arxiv_html' else 'pdf'}/{normalized_id}",
                "material_sha256": hashlib.sha256(
                    await asyncio.to_thread(pdf_path.read_bytes)
                ).hexdigest(),
            },
        )
        if not chunks:
            raise ValueError(f"No chunks extracted from {paper_id}")
        if max_chunks is not None and len(chunks) > max_chunks:
            raise ValueError("Paper exceeds this run's full-text indexing limit")
        record_event("original_material", paper_id=paper_id, format=material_format,
                     material_sha256=metadata.raw["material_sha256"], chunks=len(chunks))
        embeddings = await _embed_chunks(chunks)
    except Exception:
        await _mark_failed(paper_id, metadata=metadata)
        raise

    try:
        chunks_inserted, citations_inserted = await _write_ingestion(
            metadata=metadata,
            pdf_path=pdf_path,
            chunks=chunks,
            embeddings=embeddings,
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
        warnings=tuple(warnings),
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
    clean_existing: bool,
) -> tuple[int, int]:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts differ")

    async with db.acquire_app() as conn:
        async with conn.transaction():
            if clean_existing:
                await conn.execute(
                    "DELETE FROM citations WHERE citing_paper_id = $1", metadata.paper_id
                )
                await conn.execute("DELETE FROM chunks WHERE paper_id = $1", metadata.paper_id)
            await conn.execute(
                """
                INSERT INTO papers (
                  paper_id, source, title, authors, abstract, published_at,
                  url, pdf_path, ingestion_status, raw_metadata
                )
                VALUES ($1, $9, $2, $3::jsonb, $4, $5, $6, $7, 'in_progress', $8::jsonb)
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
                str(pdf_path) if pdf_path.suffix == ".pdf" else None,
                json.dumps(metadata.raw),
                metadata.source,
            )
            chunk_rows = [
                (
                    metadata.paper_id,
                    chunk.section,
                    idx,
                    chunk.text,
                    chunk.token_count,
                    embedding,
                    metadata.raw.get("index_key", "legacy"),
                )
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
            ]
            await conn.executemany(
                """
                INSERT INTO chunks (paper_id, section, ord, text, token_count, embedding, index_key)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (paper_id, index_key, ord) DO UPDATE SET
                  section = EXCLUDED.section,
                  text = EXCLUDED.text,
                  token_count = EXCLUDED.token_count,
                  embedding = EXCLUDED.embedding
                """,
                chunk_rows,
            )
            await conn.execute(
                "UPDATE papers SET ingestion_status = 'complete' WHERE paper_id = $1",
                metadata.paper_id,
            )
    return len(chunks), 0


def index_key():
    """Content pipeline and model identity, independent of API keys/settings secrets."""
    root = Path(__file__).parent
    parser = b"".join(
        (root / name).read_bytes()
        for name in ("pdf_loader.py", "html_loader.py", "arxiv_html.py", "chunking.py", "parse_worker.py")
    )
    route = get_route("embed-small")
    return hashlib.sha256(
        parser + f"{route.provider}:{route.vendor_model}:1536".encode()
    ).hexdigest()[:20]


async def _mark_failed(paper_id: str, metadata: PaperMetadata | None = None) -> None:
    title = metadata.title if metadata else paper_id
    authors = json.dumps(metadata.authors if metadata else [])
    abstract = metadata.abstract if metadata else None
    published_at = metadata.published_at if metadata else None
    url = metadata.url if metadata else None
    raw_metadata = json.dumps(metadata.raw if metadata else {})
    async with db.acquire_app() as conn:
        async with conn.transaction():
            # A failed attempt must never erase an already published paper.
            status = await conn.fetchval(
                "SELECT ingestion_status FROM papers WHERE paper_id = $1 FOR UPDATE", paper_id
            )
            if status == "complete":
                return
            await conn.execute("DELETE FROM citations WHERE citing_paper_id = $1", paper_id)
            await conn.execute("DELETE FROM chunks WHERE paper_id = $1", paper_id)
            await conn.execute(
                """
                INSERT INTO papers (
                  paper_id, source, title, authors, abstract, published_at,
                  url, ingestion_status, raw_metadata
                )
                VALUES ($1, $8, $2, $3::jsonb, $4, $5, $6, 'failed', $7::jsonb)
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
                metadata.source if metadata else "arxiv",
            )
