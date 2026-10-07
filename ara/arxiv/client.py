"""arXiv client: paper metadata from the export API, full text as HTML with PDF as fallback.
Every request, to any arXiv host, waits its turn: one connection, one request per three seconds."""

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from types import TracebackType

import httpx

from ara.rag.sources import ParsedPaper, arxiv_html_paper, arxiv_pdf_paper

API_URL = "https://export.arxiv.org/api/query"
INTERVAL_S = 3.0
ATOM = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "ara-v2/0.1 (+https://github.com/Syx403/arxiv-research-agent)"
ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5})(?:v(\d+))?$")


@dataclass(frozen=True)
class Metadata:
    arxiv_id: str
    version: int  # the latest version
    title: str
    abstract: str


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
        await self.http.aclose()

    async def get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        async with self._turn:
            await asyncio.sleep(max(0.0, self._last + INTERVAL_S - time.monotonic()))
            try:
                return await self.http.get(url, params=params)
            finally:
                self._last = time.monotonic()

    async def metadata(self, arxiv_id: str) -> Metadata:
        response = await self.get(API_URL, {"id_list": arxiv_id})
        response.raise_for_status()
        entry = ET.fromstring(response.content).find("atom:entry", ATOM)
        abs_url = entry.findtext("atom:id", "", ATOM) if entry is not None else ""
        found = ARXIV_ID.match(abs_url.rsplit("/", 1)[-1])
        if entry is None or found is None or found[2] is None:
            raise LookupError(f"arXiv has no paper {arxiv_id}")
        return Metadata(
            arxiv_id=found[1],
            version=int(found[2]),
            title=" ".join(entry.findtext("atom:title", "", ATOM).split()),
            abstract=" ".join(entry.findtext("atom:summary", "", ATOM).split()),
        )

    async def paper(self, reference: str) -> ParsedPaper:
        """Full text of "2210.03629" (latest version) or "2210.03629v2": HTML when arXiv has it,
        otherwise the PDF."""
        parsed = ARXIV_ID.match(reference)
        if parsed is None:
            raise ValueError(f"not an arXiv id: {reference!r}")
        arxiv_id = parsed[1]
        meta = await self.metadata(arxiv_id)
        version = int(parsed[2]) if parsed[2] else meta.version
        html = await self.get(f"https://arxiv.org/html/{arxiv_id}v{version}")
        if html.status_code != 404:
            html.raise_for_status()
            return arxiv_html_paper(html.text, arxiv_id, version)
        pdf = await self.get(f"https://arxiv.org/pdf/{arxiv_id}v{version}")
        pdf.raise_for_status()
        return arxiv_pdf_paper(
            pdf.content, arxiv_id, version, title=meta.title, abstract=meta.abstract
        )
