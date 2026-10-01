from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
import subprocess
import sys

import httpx
import pytest

from src.corpus.arxiv_client import ArxivClient, normalize_arxiv_id
from src.corpus.http import Pacer, ResponseCache, scholarly_get
from src.llm.budget import RequestBudget

FEED = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2608.12345v2</id><title>Agent memory</title>
<summary>Mechanism and evidence</summary><published>2026-08-20T00:00:00Z</published>
<updated>2026-08-25T00:00:00Z</updated><author><name>Author</name></author>
</entry></feed>'''


@pytest.mark.asyncio
async def test_real_adapter_parses_versions_and_refreshes_every_new_search(tmp_path):
    requests = []
    async def handler(request):
        requests.append(request)
        return httpx.Response(200, text=FEED)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ArxivClient(http_client=http, cache=ResponseCache(tmp_path))
        first = await client.search('ti:"memory"', latest=True)
        second = await client.search('ti:"memory"', latest=True)
    assert first == second and len(requests) == 2
    assert first[0].paper_id == "arxiv:2608.12345"
    assert first[0].published_at == date(2026, 8, 20)
    assert first[0].raw["entry_id"].endswith("v2")
    assert requests[0].url.params["sortBy"] == "submittedDate"


@pytest.mark.asyncio
async def test_bad_atom_response_is_not_cached_as_no_results(tmp_path):
    calls = []
    async def handler(request):
        calls.append(True)
        return httpx.Response(200, text="<html>service unavailable</html>" if len(calls) == 1 else FEED)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ArxivClient(http_client=http, cache=ResponseCache(tmp_path))
        client._pacer = Pacer(0)
        with pytest.raises(ValueError):
            await client.search("agent")
        assert len(await client.search("agent")) == 1
    assert len(calls) == 2




@pytest.mark.asyncio
async def test_retry_after_short_interval_recovers_once():
    calls = []
    async def handler(request):
        calls.append(True)
        return httpx.Response(429, headers={"Retry-After": "0.01"}) if len(calls) == 1 else httpx.Response(200, json={})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        response = await scholarly_get(http, Pacer(0), "https://example.test")
    assert response.status_code == 200 and len(calls) == 2


@pytest.mark.asyncio
async def test_cancelled_pdf_download_removes_partial_file(tmp_path):
    class SlowPDF(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"%PDF-1.4" + b"a" * (256 * 1024)
            await asyncio.sleep(10)
    async def handler(request):
        return httpx.Response(200, stream=SlowPDF())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ArxivClient(http_client=http, cache=ResponseCache(tmp_path / "cache"))
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(.02):
                await client.download_pdf("2608.12345", tmp_path)
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob("*.pdf"))






def test_identifiers_cannot_escape_pdf_directory():
    for value in ["../../secret", "https://other.test/1234.56789", "2210.03629?x=1"]:
        with pytest.raises(ValueError):
            normalize_arxiv_id(value)


def test_independent_processes_share_one_budget_without_lost_writes(tmp_path):
    path = tmp_path / "budget.json"
    stale = RequestBudget(.025, path=path)
    script = '''import sys
from pathlib import Path
from src.llm.budget import RequestBudget, BudgetExceeded
budget = RequestBudget(.025, path=Path(sys.argv[1]))
try:
    budget.reserve("deepseek_native", "deepseek-flash", 100, 10000)
except BudgetExceeded:
    pass
'''
    processes = [subprocess.Popen([sys.executable, "-c", script, str(path)], cwd=Path(__file__).resolve().parents[2]) for _ in range(5)]
    for process in processes:
        assert process.wait(timeout=20) == 0
    snapshot = stale.snapshot()
    assert len(snapshot["requests"]) == 2
    assert snapshot["committed_usd"] <= .025


@pytest.mark.asyncio
async def test_official_html_fallback_obeys_api_cooldown_and_reuses_metadata_cache(tmp_path):
    calls = []
    html = '''<html><meta name="citation_title" content="Agent memory">
<meta name="citation_arxiv_id" content="2608.12345"><meta name="citation_date" content="2026/08/20">
<blockquote class="abstract"><span>Abstract:</span> Verified abstract text.</blockquote></html>'''
    async def handler(request):
        calls.append(str(request.url))
        if "/api/" in str(request.url):
            return httpx.Response(429, headers={"Retry-After": "60"})
        return httpx.Response(200, text=html)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ArxivClient(http_client=http, cache=ResponseCache(tmp_path))
        client._pacer = Pacer(0)
        first = await client.fetch_metadata("2608.12345", fresh=False)
        second = await client.fetch_metadata("2608.12345", fresh=False)
    assert first == second
    assert first.raw["metadata_source"] == "arxiv_abstract_html"
    assert first.abstract == "Verified abstract text."
    assert len(calls) == 2


def test_html_search_parser_rejects_changed_markup_instead_of_empty_results():
    from src.corpus.arxiv_html import parse_search
    with pytest.raises(ValueError):
        parse_search("<html>Temporarily unavailable</html>")
    html = '''<li class="arxiv-result"><a href="https://arxiv.org/abs/2608.12345">ID</a>
<p class="title is-5">Memory <span>Agent</span></p><span class="abstract-full">Abstract text.<a>△ Less</a></span>
<p>Submitted 20 August, 2026; originally announced August 2026.</p></li>'''
    rows = parse_search(html)
    assert len(rows) == 1 and rows[0]["title"] == "Memory Agent"
    assert rows[0]["abstract"] == "Abstract text."
    assert rows[0]["published_at"] == date(2026, 8, 20)
