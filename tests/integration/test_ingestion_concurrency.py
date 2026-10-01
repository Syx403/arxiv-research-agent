"""Real PostgreSQL transactions/locks, synthetic PDF and embeddings; no paid API."""

import asyncio
from contextlib import asynccontextmanager
import os

import pytest

from src.core import db
from src.corpus import ingestion
from src.corpus.arxiv_client import PaperMetadata
from src.corpus.chunking import Chunk

pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1", reason="Requires local PostgreSQL"
)


@asynccontextmanager
async def synthetic_paper(monkeypatch, tmp_path):
    identifier = "9999.00002"
    pid = "arxiv:" + identifier + "v1"
    async with db.acquire_app() as conn:
        assert not await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM papers WHERE paper_id=$1)", pid
        )
    path = tmp_path / "synthetic.pdf"
    path.write_bytes(b"%PDF-1.4 synthetic")
    metadata = PaperMetadata(
        identifier,
        pid,
        "Concurrency test",
        [],
        "synthetic",
        None,
        "",
        None,
        {"version": 1, "version_id": identifier + "v1"},
    )

    async def fetch(*args, **kwargs):
        return metadata

    async def download(*args):
        return path

    async def no_html(*args):
        raise ValueError("Synthetic fixture tests PDF fallback")

    async def parse(*args):
        return [Chunk(section="Methods", text="Synthetic full-text evidence.", token_count=6)]

    monkeypatch.setattr(ingestion.arxiv_client, "fetch_metadata", fetch)
    monkeypatch.setattr(ingestion.arxiv_client, "download_pdf", download)
    monkeypatch.setattr(ingestion.arxiv_client, "download_html", no_html)
    monkeypatch.setattr(ingestion, "parse_chunks", parse)
    try:
        yield identifier, pid
    finally:
        async with db.acquire_app() as conn:
            await conn.execute("DELETE FROM papers WHERE paper_id=$1", pid)
        await db.close_pools()


@pytest.mark.asyncio
async def test_duplicate_acquisitions_embed_once_and_failure_cannot_erase_complete_paper(
    monkeypatch, tmp_path
):
    async with synthetic_paper(monkeypatch, tmp_path) as (identifier, pid):
        calls = []

        async def embed(chunks):
            calls.append(True)
            await asyncio.sleep(0.03)
            return [[0.0] * 1536 for chunk in chunks]

        monkeypatch.setattr(ingestion, "_embed_chunks", embed)
        results = await asyncio.gather(
            *[ingestion.ingest_paper_with_result(identifier) for _ in range(2)]
        )
        assert sorted(result.status for result in results) == ["ingested", "skipped"]
        assert len(calls) == 1
        await ingestion._mark_failed(pid)
        async with db.acquire_app() as conn:
            assert (
                await conn.fetchval("SELECT ingestion_status FROM papers WHERE paper_id=$1", pid)
                == "complete"
            )
            assert await conn.fetchval("SELECT count(*) FROM chunks WHERE paper_id=$1", pid) == 1


async def test_html_ingestion_keeps_math_and_records_material_provenance(monkeypatch, tmp_path):
    import json
    from src.corpus.parse_worker import parse_chunks

    async with synthetic_paper(monkeypatch, tmp_path) as (identifier, pid):
        path = tmp_path / 'original.html'
        path.write_text('<article class="ltx_document"><h2>Method</h2><p>'
                        + r'Load error <math alttext="e_i=\overline{c_i}-c_i">ambiguous</math>. '
                        + 'Supporting original text. ' * 20 + '</p></article>')

        async def download(*args):
            return path

        async def embed(chunks):
            return [[0.0] * 1536 for _ in chunks]

        monkeypatch.setattr(ingestion.arxiv_client, 'download_html', download)
        monkeypatch.setattr(ingestion, 'parse_chunks', parse_chunks)
        monkeypatch.setattr(ingestion, '_embed_chunks', embed)
        await ingestion.ingest_paper_with_result(identifier)
        async with db.acquire_app() as conn:
            row = await conn.fetchrow('SELECT raw_metadata,pdf_path FROM papers WHERE paper_id=$1', pid)
            text = await conn.fetchval('SELECT text FROM chunks WHERE paper_id=$1 LIMIT 1', pid)
        assert json.loads(row['raw_metadata'])['material_format'] == 'arxiv_html'
        assert row['pdf_path'] is None
        assert r'e_i=\overline{c_i}-c_i' in text


@pytest.mark.asyncio
async def test_cancelled_ingestion_releases_lock_and_allows_clean_retry(monkeypatch, tmp_path):
    async with synthetic_paper(monkeypatch, tmp_path) as (identifier, pid):

        async def slow(chunks):
            await asyncio.sleep(10)

        monkeypatch.setattr(ingestion, "_embed_chunks", slow)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.03):
                await ingestion.ingest_paper_with_result(identifier)

        async def embed(chunks):
            return [[0.0] * 1536 for chunk in chunks]

        monkeypatch.setattr(ingestion, "_embed_chunks", embed)
        async with asyncio.timeout(1):
            result = await ingestion.ingest_paper_with_result(identifier)
        assert result.status == "ingested"


@pytest.mark.asyncio
async def test_versions_and_parser_revisions_preserve_historical_chunk_ids(monkeypatch, tmp_path):
    from dataclasses import replace

    async with synthetic_paper(monkeypatch, tmp_path) as (identifier, pid):

        async def embed(chunks):
            return [[0.01] * 1536 for _ in chunks]

        monkeypatch.setattr(ingestion, "_embed_chunks", embed)
        first = await ingestion.ingest_paper_with_result(identifier)
        async with db.acquire_app() as conn:
            old = await conn.fetchrow(
                "SELECT chunk_id,text,index_key FROM chunks WHERE paper_id=$1", pid
            )
        monkeypatch.setattr(ingestion, "index_key", lambda: "new-parser-pipeline")
        await ingestion.ingest_paper_with_result(identifier)
        async with db.acquire_app() as conn:
            rows = await conn.fetch(
                "SELECT chunk_id,text,index_key FROM chunks WHERE paper_id=$1 ORDER BY chunk_id",
                pid,
            )
            assert len(rows) == 2 and rows[0] == old and rows[1]["chunk_id"] != old["chunk_id"]
            assert (
                await conn.fetchval(
                    "SELECT raw_metadata->>'index_key' FROM papers WHERE paper_id=$1", pid
                )
                == "new-parser-pipeline"
            )
        pinned = await ingestion.arxiv_client.fetch_metadata(identifier)
        next_metadata = replace(pinned, raw={"version": 2, "version_id": identifier + "v2"})
        try:
            second = await ingestion.ingest_paper_with_result(identifier, metadata=next_metadata)
            assert first.paper_id == pid and second.paper_id == pid[:-1] + "2"
            async with db.acquire_app() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT text FROM chunks WHERE chunk_id=$1", old["chunk_id"]
                    )
                    == old["text"]
                )
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM chunks WHERE paper_id=$1", second.paper_id
                    )
                    == 1
                )
        finally:
            async with db.acquire_app() as conn:
                await conn.execute("DELETE FROM papers WHERE paper_id=$1", pid[:-1] + "2")
