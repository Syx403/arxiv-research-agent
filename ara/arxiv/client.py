"""arXiv client: search and metadata from the export API, full text as HTML with PDF as fallback.
Every request, to any arXiv host, waits its turn: one connection, one request per three seconds."""

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import date
from functools import partial
from types import TracebackType

import httpx

from ara import faults
from ara.rag.sources import ParsedPaper, arxiv_html_paper, arxiv_pdf_paper

API_URL = "https://export.arxiv.org/api/query"
INTERVAL_S = 3.0
MAX_PAUSE_S = 20.0  # longest Retry-After honoured: well inside the node timeout (D38)
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "ara-v2/0.1 (+https://github.com/Syx403/arxiv-research-agent)"
ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v(\d+))?$")
FIRST_DAY = date(1991, 1, 1)  # arXiv rejects a submittedDate range that starts earlier


@dataclass(frozen=True)
class Metadata:
    arxiv_id: str
    version: int  # the latest version
    title: str
    abstract: str
    published: date  # first version


class SearchError(Exception):
    """arXiv answered a search with an error entry (usually a malformed query)."""


class ArxivClient:
    def __init__(self, timeout_s: float = 20) -> None:
        self.http = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=1),
            timeout=timeout_s,
            follow_redirects=True,
        )
        self._turn = asyncio.Lock()
        self._last = 0.0

    async def __aenter__(self) -> "ArxivClient":
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        """One request at a time, 3 s apart. After a 429 the next request also waits out arXiv's
        Retry-After (at most MAX_PAUSE_S); the caller's node retry sends it again (M5a, D38)."""
        async with self._turn:
            await asyncio.sleep(max(0.0, self._last + INTERVAL_S - time.monotonic()))
            pause = 0.0
            try:
                faults.trip("arxiv")
                response = await self.http.get(url, params=params)
                if response.status_code == 429:
                    pause = min(_seconds(response.headers.get("Retry-After")), MAX_PAUSE_S)
                return response
            finally:
                self._last = time.monotonic() + pause

    async def search(
        self,
        query: str,
        *,
        after: date | None = None,
        before: date | None = None,
        newest_first: bool = False,
        max_results: int = 20,
    ) -> list[Metadata]:
        """Papers matching an arXiv query ("ti:..", "abs:..", AND / OR), by relevance or newest
        first. The date window is added here, so a model-written query cannot widen it."""
        if after or before:
            start, end = after or FIRST_DAY, before or date.today()
            query = f"({query}) AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
        params = {
            "search_query": query,
            "max_results": str(max_results),
            "sortBy": "submittedDate" if newest_first else "relevance",
            "sortOrder": "descending",
        }
        return await self._feed(params)

    async def lookup(self, arxiv_ids: list[str]) -> list[Metadata]:
        """Metadata for several ids in one request; unknown ids are left out."""
        return await self._feed(
            {"id_list": ",".join(arxiv_ids), "max_results": str(len(arxiv_ids))}
        )

    async def metadata(self, arxiv_id: str) -> Metadata:
        found = await self.lookup([arxiv_id])
        if not found:
            raise LookupError(f"arXiv has no paper {arxiv_id}")
        return found[0]

    async def _feed(self, params: dict[str, str]) -> list[Metadata]:
        response = await self.get(API_URL, params)
        if response.status_code == 400:
            raise SearchError(f"arXiv rejected the query: {params.get('search_query', '')}")
        response.raise_for_status()
        entries = ET.fromstring(response.content).findall("atom:entry", ATOM)
        if any(e.findtext("atom:id", "", ATOM).endswith("/api/errors") for e in entries):
            raise SearchError(entries[0].findtext("atom:summary", "", ATOM).strip())
        return [m for e in entries if (m := _metadata(e)) is not None]

    async def paper(self, reference: str) -> ParsedPaper:
        """Full text of "2210.03629" (latest version) or "2210.03629v2": HTML when arXiv has it,
        otherwise the PDF."""
        parsed = ARXIV_ID.match(reference)
        if parsed is None:
            raise ValueError(f"not an arXiv id: {reference!r}")
        arxiv_id = parsed[1]
        faults.trip("fetch")
        meta = await self.metadata(arxiv_id)
        version = int(parsed[2]) if parsed[2] else meta.version
        html = await self.get(f"https://arxiv.org/html/{arxiv_id}v{version}")
        # parsing takes seconds of CPU: off the event loop, so other turns keep streaming (D39)
        if html.status_code != 404:
            html.raise_for_status()
            parsed_paper = await asyncio.to_thread(arxiv_html_paper, html.text, arxiv_id, version)
        else:
            pdf = await self.get(f"https://arxiv.org/pdf/{arxiv_id}v{version}")
            pdf.raise_for_status()
            parsed_paper = await asyncio.to_thread(
                partial(arxiv_pdf_paper, title=meta.title, abstract=meta.abstract),
                pdf.content,
                arxiv_id,
                version,
            )
        return replace(parsed_paper, published=meta.published)


def _metadata(entry: ET.Element) -> Metadata | None:
    found = ARXIV_ID.match(entry.findtext("atom:id", "", ATOM).rsplit("/", 1)[-1])
    if found is None or found[2] is None:
        return None
    return Metadata(
        arxiv_id=found[1],
        version=int(found[2]),
        title=" ".join(entry.findtext("atom:title", "", ATOM).split()),
        abstract=" ".join(entry.findtext("atom:summary", "", ATOM).split()),
        published=date.fromisoformat(entry.findtext("atom:published", "", ATOM)[:10]),
    )


def _seconds(retry_after: str | None) -> float:
    """Retry-After in seconds; arXiv sends seconds, and a date or nothing counts as the spacing."""
    return float(retry_after) if retry_after and retry_after.isdigit() else INTERVAL_S
