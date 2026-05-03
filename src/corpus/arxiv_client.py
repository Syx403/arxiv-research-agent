from __future__ import annotations

import asyncio
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arxiv


@dataclass(frozen=True)
class PaperMetadata:
    arxiv_id: str
    paper_id: str
    title: str
    authors: list[str]
    abstract: str | None
    published_at: Any
    url: str
    pdf_url: str | None
    raw: dict


class ArxivClient:
    def __init__(self) -> None:
        self._client = arxiv.Client(page_size=10, delay_seconds=3)
        self._lock = asyncio.Lock()

    async def fetch_metadata(self, arxiv_id: str) -> PaperMetadata:
        normalized_id = normalize_arxiv_id(arxiv_id)
        async with self._lock:
            return await asyncio.to_thread(self._fetch_metadata_sync, normalized_id)

    async def download_pdf(self, arxiv_id: str, dest_dir: Path) -> Path:
        normalized_id = normalize_arxiv_id(arxiv_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            return await asyncio.to_thread(self._download_pdf_sync, normalized_id, dest_dir)

    def _fetch_result(self, arxiv_id: str):
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        results = list(self._client.results(search))
        if not results:
            raise LookupError(f"arXiv paper not found: {arxiv_id}")
        return results[0]

    def _fetch_metadata_sync(self, arxiv_id: str) -> PaperMetadata:
        result = self._fetch_result(arxiv_id)
        return PaperMetadata(
            arxiv_id=arxiv_id,
            paper_id=f"arxiv:{arxiv_id}",
            title=result.title,
            authors=[str(author) for author in result.authors],
            abstract=result.summary,
            published_at=result.published.date() if result.published else None,
            url=result.entry_id,
            pdf_url=getattr(result, "pdf_url", None),
            raw={
                "entry_id": result.entry_id,
                "primary_category": getattr(result, "primary_category", None),
                "categories": list(getattr(result, "categories", []) or []),
            },
        )

    def _download_pdf_sync(self, arxiv_id: str, dest_dir: Path) -> Path:
        result = self._fetch_result(arxiv_id)
        filename = f"{arxiv_id.replace('/', '_')}.pdf"
        pdf_url = getattr(result, "pdf_url", None)
        if not pdf_url:
            raise LookupError(f"arXiv paper has no PDF URL: {arxiv_id}")
        path = dest_dir / filename
        headers = {"User-Agent": "arxiv-research-agent/0.1"}
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(pdf_url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected_length = response.headers.get("Content-Length")
                    written = 0
                    with path.open("wb") as file:
                        while True:
                            chunk = response.read(1024 * 256)
                            if not chunk:
                                break
                            file.write(chunk)
                            written += len(chunk)
                if expected_length is not None and written < int(expected_length):
                    raise urllib.error.ContentTooShortError(
                        f"retrieval incomplete: got only {written} out of {expected_length} bytes",
                        None,
                    )
                return path
            except (urllib.error.ContentTooShortError, urllib.error.HTTPError, urllib.error.URLError):
                if path.exists():
                    path.unlink()
                if attempt == 3:
                    raise
                time.sleep(3 * attempt)
        raise RuntimeError(f"Failed to download arXiv PDF: {arxiv_id}")


def normalize_arxiv_id(arxiv_id: str) -> str:
    cleaned = arxiv_id.strip()
    if cleaned.lower().startswith("arxiv:"):
        cleaned = cleaned.split(":", 1)[1]
    return cleaned


GLOBAL_ARXIV_CLIENT = ArxivClient()
