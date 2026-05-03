from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from src.core.config import get_settings


class SemanticScholarRateLimitError(Exception):
    pass


@dataclass(frozen=True)
class S2Author:
    author_id: str | None
    name: str


@dataclass(frozen=True)
class S2Paper:
    paper_id: str | None
    external_ids: dict[str, Any]
    title: str | None
    abstract: str | None
    authors: list[S2Author]
    year: int | None
    reference_count: int | None
    citation_count: int | None


@dataclass(frozen=True)
class S2Reference:
    paper_id: str
    external_ids: dict[str, Any]
    title: str | None
    abstract: str | None


@dataclass(frozen=True)
class S2Citation:
    paper_id: str
    external_ids: dict[str, Any]
    title: str | None
    abstract: str | None


class SemanticScholarClient:
    def __init__(self, *, http_client: httpx.AsyncClient | None = None) -> None:
        settings = get_settings()
        self._api_key = (
            settings.semantic_scholar_api_key.get_secret_value()
            if settings.semantic_scholar_api_key
            else None
        )
        self._rate = 10 if self._api_key else 1
        self._semaphore = asyncio.Semaphore(self._rate)
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0
        self._client = http_client or httpx.AsyncClient(
            base_url="https://api.semanticscholar.org/graph/v1",
            timeout=30,
        )

    async def fetch_paper(self, s2_id_or_arxiv: str) -> S2Paper:
        data = await self._get(
            f"/paper/{quote(_s2_lookup_id(s2_id_or_arxiv), safe=':')}",
            params={
                "fields": (
                    "paperId,externalIds,title,abstract,authors,year,"
                    "referenceCount,citationCount"
                )
            },
        )
        return _parse_paper(data)

    async def fetch_references(self, paper_id: str, limit: int = 50) -> list[S2Reference]:
        data = await self._get(
            f"/paper/{quote(_s2_lookup_id(paper_id), safe=':')}/references",
            params={
                "limit": str(limit),
                "fields": "citedPaper.paperId,citedPaper.externalIds,citedPaper.title,citedPaper.abstract",
            },
        )
        references: list[S2Reference] = []
        for item in data.get("data", []) or []:
            paper = item.get("citedPaper") or {}
            parsed_id = _paper_identifier(paper)
            if parsed_id:
                references.append(
                    S2Reference(
                        paper_id=parsed_id,
                        external_ids=paper.get("externalIds") or {},
                        title=paper.get("title"),
                        abstract=paper.get("abstract"),
                    )
                )
        return references

    async def fetch_citations(self, paper_id: str, limit: int = 50) -> list[S2Citation]:
        data = await self._get(
            f"/paper/{quote(_s2_lookup_id(paper_id), safe=':')}/citations",
            params={
                "limit": str(limit),
                "fields": "citingPaper.paperId,citingPaper.externalIds,citingPaper.title,citingPaper.abstract",
            },
        )
        citations: list[S2Citation] = []
        for item in data.get("data", []) or []:
            paper = item.get("citingPaper") or {}
            parsed_id = _paper_identifier(paper)
            if parsed_id:
                citations.append(
                    S2Citation(
                        paper_id=parsed_id,
                        external_ids=paper.get("externalIds") or {},
                        title=paper.get("title"),
                        abstract=paper.get("abstract"),
                    )
                )
        return citations

    @retry(
        retry=retry_if_exception_type(SemanticScholarRateLimitError),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def _get(self, path: str, *, params: dict[str, str]) -> dict[str, Any]:
        async with self._semaphore:
            await self._pace()
            headers = {"x-api-key": self._api_key} if self._api_key else None
            response = await self._client.get(path, params=params, headers=headers)
        if response.status_code == 429:
            raise SemanticScholarRateLimitError("Semantic Scholar rate limit")
        response.raise_for_status()
        return response.json()

    async def _pace(self) -> None:
        async with self._lock:
            min_interval = 1.0 / self._rate
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_request_at = time.monotonic()

    async def aclose(self) -> None:
        await self._client.aclose()


def _s2_lookup_id(paper_id: str) -> str:
    cleaned = paper_id.strip()
    if cleaned.lower().startswith("arxiv:"):
        return f"arXiv:{cleaned.split(':', 1)[1]}"
    if ":" in cleaned:
        return cleaned
    return f"arXiv:{cleaned}"


def _parse_paper(data: dict[str, Any]) -> S2Paper:
    return S2Paper(
        paper_id=data.get("paperId"),
        external_ids=data.get("externalIds") or {},
        title=data.get("title"),
        abstract=data.get("abstract"),
        authors=[
            S2Author(author_id=author.get("authorId"), name=author.get("name") or "")
            for author in data.get("authors", []) or []
        ],
        year=data.get("year"),
        reference_count=data.get("referenceCount"),
        citation_count=data.get("citationCount"),
    )


def _paper_identifier(paper: dict[str, Any]) -> str | None:
    external_ids = paper.get("externalIds") or {}
    if external_ids.get("ArXiv"):
        return f"arxiv:{external_ids['ArXiv']}"
    if external_ids.get("CorpusId"):
        return f"s2:{external_ids['CorpusId']}"
    if paper.get("paperId"):
        return f"s2:{paper['paperId']}"
    if paper.get("title"):
        return "title:" + paper["title"].strip().lower().replace(" ", "-")[:120]
    return None


GLOBAL_SEMANTIC_SCHOLAR_CLIENT = SemanticScholarClient()
