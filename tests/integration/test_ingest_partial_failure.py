from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.core import db
from src.corpus import ingestion
from src.corpus.arxiv_client import PaperMetadata
from src.corpus.chunking import Chunk
from src.llm.client import close_llm_clients
from src.llm.providers.openai_embed import OpenAIEmbeddingClient


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="ingestion integration tests require INTEGRATION_TESTS=1 and running Postgres",
)


@pytest.mark.asyncio
async def test_ingest_partial_failure_marks_failed_and_rolls_back_chunks(monkeypatch, tmp_path: Path) -> None:
    arxiv_id = "9999.00001"
    paper_id = f"arxiv:{arxiv_id}v1"
    pdf_path = tmp_path / "fake.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    async def fake_fetch_metadata(_arxiv_id: str, **kwargs) -> PaperMetadata:
        return PaperMetadata(
            arxiv_id=arxiv_id,
            paper_id=paper_id,
            title="Partial Failure Test",
            authors=["Tester"],
            abstract="Synthetic abstract.",
            published_at=None,
            url="https://arxiv.org/abs/9999.00001",
            pdf_url=None,
            raw={"version":1,"version_id":arxiv_id+"v1"},
        )

    async def fake_download_pdf(_arxiv_id: str, _dest_dir: Path) -> Path:
        return pdf_path

    async def no_html(*args):
        raise ValueError("Synthetic fixture tests PDF fallback")

    async def fake_references(*args, **kwargs):
        return []

    calls = 0

    async def fake_embed(self, texts: list[str], *, model: str = "text-embedding-3-small"):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic embedding batch failure")
        return [[0.0] * 1536 for _ in texts]

    monkeypatch.setattr(ingestion.arxiv_client, "fetch_metadata", fake_fetch_metadata)
    monkeypatch.setattr(ingestion.arxiv_client, "download_pdf", fake_download_pdf)
    monkeypatch.setattr(ingestion.arxiv_client, "download_html", no_html)
    async def fake_chunks(path):
        return [Chunk(section="Test", text=f"chunk {idx}", token_count=2) for idx in range(300)]
    monkeypatch.setattr(ingestion, "parse_chunks", fake_chunks)
    monkeypatch.setattr(OpenAIEmbeddingClient, "embed", fake_embed)

    try:
        async with db.acquire_app() as conn:
            await conn.execute("DELETE FROM papers WHERE paper_id = $1", paper_id)

        with pytest.raises(RuntimeError, match="synthetic embedding batch failure"):
            await ingestion.ingest_paper(arxiv_id)

        async with db.acquire_app() as conn:
            status = await conn.fetchval(
                "SELECT ingestion_status FROM papers WHERE paper_id = $1",
                paper_id,
            )
            chunk_count = await conn.fetchval(
                "SELECT count(*) FROM chunks WHERE paper_id = $1",
                paper_id,
            )

        assert status == "failed"
        assert chunk_count == 0
    finally:
        await close_llm_clients()
        await db.close_pools()
