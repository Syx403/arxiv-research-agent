from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
import time
from typing import Any
from uuid import uuid4
import xml.etree.ElementTree as ET

import httpx

from src.core.trace import record_event
from src.corpus.http import Pacer, ResponseCache, scholarly_get, retry_after
from src.corpus.arxiv_html import parse_search, parse_abstract

ARXIV_ID = r"(?:\d{4}\.\d{4,5}|[a-z][a-z.\-]+/\d{7})(?:v\d+)?"
NS = {"a": "http://www.w3.org/2005/Atom"}


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
    source: str = "arxiv"


class ArxivClient:
    def __init__(self, *, http_client: httpx.AsyncClient | None = None,
                 cache: ResponseCache | None = None) -> None:
        self._client = http_client or httpx.AsyncClient(timeout=20, follow_redirects=True,
                                                       headers={"User-Agent": "arxiv-research-agent/0.1"})
        self._pacer = Pacer(3)
        self._cache = cache or ResponseCache()
        self._api_retry_at = 0.0
        self._api_lock = asyncio.Lock()

    async def _query(self, params: dict, *, fresh: bool = False) -> list[PaperMetadata]:
        key = "arxiv:" + json.dumps(params, sort_keys=True)
        content = None if fresh else self._cache.get(key, max_age_s=3600 if params.get("sortBy") == "submittedDate" else None)
        cached = content is not None
        if content is None:
            response = await scholarly_get(self._client, self._pacer,
                                           "https://export.arxiv.org/api/query", params=params, timeout=5,
                                           retry_rate_limits=False)
            content = response.text
        papers = parse_feed(content)
        if not cached:
            self._cache.put(key, content)
        record_event("external_metadata", provider="arxiv", cache_hit=cached,
                     query=params, result_count=len(papers))
        return papers

    async def search(self, query: str, *, limit: int = 12, latest: bool = False,
                     date_from: str | None = None, date_to: str | None = None,
                     offset: int = 0, fresh: bool = True) -> list[PaperMetadata]:
        if not 0 <= offset <= 40 or not 1 <= limit <= 24:
            raise ValueError("Search page exceeds the bounded discovery window")
        terms = list(dict.fromkeys(re.findall(r'(?:ti|abs):"([^"\n]+)"', query)))
        title_only = bool(terms) and not re.search(r'\b(?:abs|all):', query)
        web_field = 'title' if title_only else 'all'
        web_url = "https://arxiv.org/search/"
        web_params = {"query": " AND ".join(f'"{term}"' for term in terms), "searchtype": web_field, "abstracts": "show",
                      "order": "-announced_date_first" if latest else "", "size": "50"}
        if date_from and date_to:
            from datetime import date
            start, end = date.fromisoformat(date_from), date.fromisoformat(date_to)
            if start > end:
                raise ValueError("Invalid publication date range")
            query += f' AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]'
            web_url = "https://arxiv.org/search/advanced"
            web_params = {"advanced": "", "abstracts": "show", "size": "50",
                          "order": "-announced_date_first" if latest else "",
                          "date-filter_by": "date_range", "date-from_date": date_from,
                          "date-to_date": date_to, "date-date_type": "submitted_date_first"}
            for i, term in enumerate(terms):
                web_params.update({f"terms-{i}-operator": "AND", f"terms-{i}-term": f'"{term}"',
                                   f"terms-{i}-field": web_field})
        web_cached = not fresh and terms and self._cache.get("arxiv_html:" + web_url +
                                              json.dumps(web_params, sort_keys=True), max_age_s=3600 if latest else None) is not None
        api_params = {"search_query": query, "start": offset, "max_results": limit,
                      "sortBy": "submittedDate" if latest else "relevance", "sortOrder": "descending"}
        # A provider cooldown prevents new HTTP, not reuse of successful cached feeds.
        if not fresh and self._cache.get("arxiv:" + json.dumps(api_params, sort_keys=True), max_age_s=3600 if latest else None) is not None:
            return await self._query(api_params)
        async with self._api_lock:
            if not web_cached and time.monotonic() >= self._api_retry_at:
                try:
                    return await self._query(api_params, fresh=fresh)
                except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
                    if not terms:
                        raise
                    self._backoff(exc)
        # Preserve title-only constraints, conjunction and phrases. HTML's all-field
        # alternative is broader than API title/abstract; selection checks abstracts.
        if not terms:
            raise ValueError("This arXiv query has no supported website fallback")
        rows = await self._web(web_url, web_params, parse_search, fresh=fresh)
        papers = []
        for row in rows[offset:offset + limit]:
            identifier = normalize_arxiv_id(row["url"])
            papers.append(PaperMetadata(
                arxiv_id=identifier, paper_id=f"arxiv:{identifier}", title=row["title"],
                authors=row["authors"], abstract=row["abstract"], published_at=row["published_at"],
                url=row["url"], pdf_url=f"https://arxiv.org/pdf/{identifier}",
                raw={"metadata_source": "arxiv_search_html", "publication_date_unconfirmed": True},
            ))
        return papers

    async def fetch_metadata(self, arxiv_id: str, *, fresh: bool = True) -> PaperMetadata:
        normalized = normalize_arxiv_id(arxiv_id, keep_version=True)
        if (time.monotonic() < self._api_retry_at or
            (not fresh and self._cache.get(f"arxiv_html:https://arxiv.org/abs/{normalized}" + "{}") is not None)):
            return await self._web_metadata(normalized, fresh=fresh)
        try:
            papers = await self._query({"id_list": normalized, "max_results": 1}, fresh=fresh)
        except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
            self._backoff(exc)
            return await self._web_metadata(normalized, fresh=fresh)
        if not papers or papers[0].arxiv_id != normalize_arxiv_id(normalized):
            raise LookupError(f"arXiv paper not found: {normalized}")
        if re.search(r'v\d+$', normalized) and papers[0].raw.get('version_id') != normalized:
            raise ValueError("arXiv returned a different paper version")
        return papers[0]

    def _backoff(self, exc):
        delay = max(300, retry_after(exc.response, 0)) if isinstance(exc, httpx.HTTPStatusError) else 300
        self._api_retry_at = time.monotonic() + delay
        record_event("external_fallback", provider="arxiv", source="official_html",
                     reason=type(exc).__name__, api_cooldown_s=delay)

    async def _web(self, url, params, parser, *, fresh=False):
        key = "arxiv_html:" + url + json.dumps(params, sort_keys=True)
        content = None if fresh else self._cache.get(key, max_age_s=3600 if params.get("order") == "-announced_date_first" else None)
        cached = content is not None
        if content is None:
            response = await scholarly_get(self._client, self._pacer, url, params=params,
                                            timeout=25 if "/search/" in url else 12)
            content = response.text
        parsed = parser(content)
        if not cached:
            self._cache.put(key, content)
        record_event("external_metadata", provider="arxiv_html", cache_hit=cached, query=params)
        return parsed

    async def _web_metadata(self, identifier, *, fresh=True):
        row = await self._web(f"https://arxiv.org/abs/{identifier}", {}, parse_abstract, fresh=fresh)
        base = normalize_arxiv_id(identifier)
        if normalize_arxiv_id(row["arxiv_id"]) != base:
            raise ValueError("arXiv abstract identity mismatch")
        if re.search(r'v\d+$', identifier) and row.get('version_id') != identifier:
            raise ValueError("arXiv abstract version mismatch")
        return PaperMetadata(arxiv_id=base, paper_id=f"arxiv:{base}", title=row["title"],
                             authors=row["authors"], abstract=row["abstract"],
                             published_at=row["published_at"], url=f"https://arxiv.org/abs/{identifier}",
                             pdf_url=f"https://arxiv.org/pdf/{identifier}",
                             raw={"metadata_source": "arxiv_abstract_html", "entry_id": row["arxiv_id"],
                                  "version_id": row.get('version_id'), "version": row.get('version'),
                                  "updated": row.get('updated')})

    async def download_html(self, arxiv_id: str, dest_dir: Path) -> Path:
        from src.corpus.html_loader import load_html

        normalized = normalize_arxiv_id(arxiv_id, keep_version=True)
        if not re.search(r"v\d+$", normalized):
            raise ValueError("Full text requires a pinned arXiv version")

        def validate(content):
            if len(content.encode()) > 15 * 1024 * 1024:
                raise ValueError("HTML exceeds acquisition limit")
            load_html(content, expected_id=normalized)
            return content

        content = await self._web(f"https://arxiv.org/html/{normalized}", {}, validate)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{normalized.replace('/', '_')}.html"
        temporary = path.with_suffix(f".{uuid4().hex}.part")
        try:
            temporary.write_text(content)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    async def download_pdf(self, arxiv_id: str, dest_dir: Path) -> Path:
        normalized = normalize_arxiv_id(arxiv_id, keep_version=True)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{normalized.replace('/', '_')}.pdf"
        temporary = path.with_suffix(f".{uuid4().hex}.part")
        try:
            await self._pacer.wait()
            async with self._client.stream("GET", f"https://arxiv.org/pdf/{normalized}") as response:
                response.raise_for_status()
                written = 0
                with temporary.open("wb") as file:
                    async for block in response.aiter_bytes(256 * 1024):
                        written += len(block)
                        if written > 25 * 1024 * 1024:
                            raise ValueError("PDF exceeds the 25 MiB acquisition limit")
                        file.write(block)
                with temporary.open("rb") as file:
                    if b"%PDF-" not in file.read(1024):
                        raise ValueError("arXiv returned a non-PDF response")
                temporary.replace(path)
                return path
        finally:
            temporary.unlink(missing_ok=True)

    async def aclose(self):
        await self._client.aclose()


def normalize_arxiv_id(arxiv_id: str, *, keep_version: bool = False) -> str:
    cleaned = re.sub(r"^(?:arxiv:|https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/)", "",
                     arxiv_id.strip(), flags=re.I).removesuffix(".pdf")
    if not re.fullmatch(ARXIV_ID, cleaned, flags=re.I):
        raise ValueError("Invalid arXiv identifier")
    # Versions share one corpus identity; returned version remains in raw metadata.
    return cleaned if keep_version else re.sub(r"v\d+$", "", cleaned)


def parse_feed(content: str) -> list[PaperMetadata]:
    root = ET.fromstring(content)
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ValueError("Invalid arXiv Atom feed")
    papers = []
    for entry in root.findall("a:entry", NS):
        entry_id = entry.findtext("a:id", "", NS)
        paper_id = normalize_arxiv_id(entry_id)
        published = datetime.fromisoformat(entry.findtext("a:published", "", NS).replace("Z", "+00:00"))
        title = " ".join(entry.findtext("a:title", "", NS).split())
        if not title:
            raise ValueError("Missing paper title")
        papers.append(PaperMetadata(
            arxiv_id=paper_id, paper_id=f"arxiv:{paper_id}", title=title,
            authors=[a.findtext("a:name", "", NS) for a in entry.findall("a:author", NS)],
            abstract=entry.findtext("a:summary", "", NS).strip(), published_at=published.date(),
            url=f"https://arxiv.org/abs/{paper_id}", pdf_url=f"https://arxiv.org/pdf/{paper_id}",
            raw={"entry_id": entry_id, "updated": entry.findtext("a:updated", "", NS),
                 "version_id": normalize_arxiv_id(entry_id, keep_version=True),
                 "version": int(re.search(r'v(\d+)$', entry_id)[1]) if re.search(r'v(\d+)$', entry_id) else None,
                 "search_total": int(root.findtext('{http://a9.com/-/spec/opensearch/1.1/}totalResults', '0')),
                 "categories": [a.get("term") for a in entry.findall("a:category", NS)]},
        ))
    return papers


GLOBAL_ARXIV_CLIENT = ArxivClient()
