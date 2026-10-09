"""M5a reliability against real code: injected faults (ARA_FAULTS) make the gateway and the arXiv
client fail before anything is sent, so each degraded path runs without mocks and without
billable calls. Two tests run a node's retries in full (about 4 s each)."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any
from uuid import uuid4

import httpx
import httpx2
import openai
import pytest
from langchain_core.messages import HumanMessage
from langgraph.errors import NodeError, NodeTimeoutError
from langgraph.runtime import Runtime

from ara import faults
from ara.arxiv.client import ArxivClient, _seconds
from ara.db.pool import Pool
from ara.graph import answer, app, discover, read, reliability
from ara.graph.state import Claim, Evidence
from ara.llm.gateway import InvalidOutput
from ara.llm.ledger import BudgetExceeded
from ara.llm.prompt import ToolCall
from ara.rag.retrieve import Passage
from tests.unit.test_app import base_state, card, context, request
from tests.unit.test_search import setup


@pytest.fixture
def fault(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Set ARA_FAULTS for one test (the environment, not the code, is patched)."""
    faults.reset()

    def set_faults(spec: str) -> None:
        monkeypatch.setenv("ARA_FAULTS", spec)

    yield set_faults
    faults.reset()


def status(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.invalid")
    return httpx.HTTPStatusError(
        "x", request=request, response=httpx.Response(code, request=request)
    )


def test_only_failures_another_attempt_can_fix_are_retried() -> None:
    request = httpx.Request("POST", "https://example.invalid")
    retried = [
        status(429),
        status(503),
        httpx.ReadTimeout("slow", request=request),
        openai.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid")),
        NodeTimeoutError("node", 45.0, kind="run", run_timeout=45.0),
        InvalidOutput("empty answer"),
    ]
    refused = [status(400), status(404), BudgetExceeded("cap"), ValueError("bad input")]
    assert all(reliability.transient(e) for e in retried)
    assert not any(reliability.transient(e) for e in refused)
    assert reliability.attempts(InvalidOutput("x")) == 2 and reliability.attempts(status(500)) == 3


def test_a_fault_can_fail_only_the_first_calls(fault: Any) -> None:
    fault("arxiv=429x1, synthesize=empty")
    with pytest.raises(httpx.HTTPStatusError):
        faults.trip("arxiv")
    faults.trip("arxiv")  # the second call goes through
    with pytest.raises(InvalidOutput):
        faults.trip("synthesize")
    faults.trip("verify")  # no fault configured
    assert _seconds("7") == 7 and _seconds(None) == 3.0 and _seconds("Wed, 21 Oct") == 3.0


async def test_a_line_that_cannot_be_verified_is_not_delivered(pool: Pool, fault: Any) -> None:
    fault("verify=refused")
    line = Claim(index=1, text="It is fast.", citations=["E1"])
    task = answer.VerifyTask(question="q", claim=line, evidence=[])
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        result = await answer.verify(task, Runtime(context=ctx))
        await ctx.gateway.aclose()
    assert not result["verdicts"][line.key].supported
    assert result["problems"] == ["verification failed (HTTPStatusError)"]


async def test_a_paper_that_cannot_be_fetched_is_skipped(pool: Pool, fault: Any) -> None:
    fault("fetch=500")
    async with ArxivClient() as arxiv:

        async def fetch(reference: str) -> Any:
            return await arxiv.paper(reference.removeprefix("arxiv:"))

        ctx = replace(context(pool, arxiv), fetch=fetch)
        result = await read.ingest_paper({"reference": "arxiv:2401.09999v1"}, Runtime(context=ctx))
        await ctx.gateway.aclose()
    assert result == {
        "documents": [],
        "problems": ["reading arxiv:2401.09999v1 failed (HTTPStatusError)"],
    }


async def test_failed_searches_and_selections_add_no_evidence(pool: Pool, fault: Any) -> None:
    gateway, first, second = await setup(pool)
    fault("embed=500, select_evidence=timeout")
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        searched = await read.search(
            read.SearchTask(
                question="q", query="uncached query", documents=[first, second], round=0
            ),
            Runtime(context=ctx),
        )
        passage = Passage(1, first, "arxiv:2401.00001v1", 0, "h", "One. Two.", "One. Two.", (0, 5))
        task = read.SelectTask(question="q", query="q", document=first, round=0, passages=[passage])
        selected = await read.select(task, Runtime(context=ctx))
        await ctx.gateway.aclose()
    assert [(t["document"], t["passages"]) for t in searched["staged"]] == [
        (first, []),
        (second, []),
    ]
    assert [f.sentences for f in selected["found"]] == [[]]
    assert selected["problems"] == ["selecting evidence failed (ReadTimeout)"]
    empty: Any = {"question": "q", "documents": [], "requeried": True}
    collected: Any = read.collect(empty)
    assert collected["evidence"] == [] and collected["missing"] == []
    await gateway.aclose()


async def test_an_arxiv_tool_failure_goes_back_to_the_researcher(pool: Pool, fault: Any) -> None:
    fault("arxiv=429")
    call = ToolCall("c1", "search_arxiv", '{"query": "ti:ReWOO"}')
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        text = await discover.run_tool(call, request(), {}, ctx)
        await ctx.gateway.aclose()
    assert text.startswith("arXiv did not answer (an arXiv request failed (HTTPStatusError))")


async def test_a_failed_screen_batch_leaves_its_papers_unjudged(pool: Pool, fault: Any) -> None:
    fault("screen=refused")
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        task = discover.ScreenTask(request=request(), batch=[card(1)])
        result = await discover.screen(task, Runtime(context=ctx))
        await ctx.gateway.aclose()
    assert result == {"judged": [], "problems": ["screening a batch failed (HTTPStatusError)"]}


def test_a_failed_part_becomes_a_note_and_the_turn_is_partial() -> None:
    handler = app.give_up("the arXiv search", "respond", papers=[])
    state = base_state(request(intent="discover_read"), papers=[card(1)])
    command = handler(state, NodeError("discover", status(503)))
    assert command.goto == "respond"
    assert command.update == {
        "papers": [],
        "problems": ["the arXiv search failed (HTTPStatusError)"],
    }
    failed: Any = {**state, **command.update}
    result: Any = app.respond(failed)
    text = result["messages"][0].text
    assert "I found no arXiv papers" not in text and result["status"] == "partial"
    assert text.endswith(
        "Note: the arXiv search failed (HTTPStatusError). The rest of this reply is unaffected."
    )
    library: Any = {
        **base_state(request(intent="library")),
        "problems": ["searching your library failed (X)"],
    }
    assert app.reply(library) != app.NOT_DISCUSSED, "a failure is not a missing history"


async def test_a_synthesis_that_keeps_failing_ends_in_an_abstention(pool: Pool, fault: Any) -> None:
    """The node is retried (three attempts, about 4 s), then its handler finalizes."""
    fault("synthesize=500")
    evidence = [
        Evidence(id="E1", paper_id="p", chunk_id=1, paragraph=0, heading_path="h", text="t")
    ]
    inputs: Any = {
        "question": "q",
        "evidence": evidence,
        "missing": [],
        "priorities": [],
        "context": "",
    }
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        result = await answer.build().ainvoke(inputs, context=ctx)
        await ctx.gateway.aclose()
    assert result["answer"].abstained
    assert result["problems"] == ["writing the answer failed (HTTPStatusError)"]
    assert faults._tripped["synthesize"] == 3, "retried twice before giving up"


async def test_a_failed_search_still_replies_with_a_note(pool: Pool, fault: Any) -> None:
    """The whole turn after understand: the researcher keeps failing (retried, about 4 s), the
    discover subgraph ranks nothing, and the reply says the search failed."""
    fault("researcher=503")
    graph = app.build(await app.checkpointer(pool))
    config: Any = {"configurable": {"thread_id": str(uuid4())}}
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage("Find papers on KV-cache eviction")],
            **app.fresh_turn(),
            "request": request(intent="discover"),
        },
        as_node="understand",
    )
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        state = await graph.ainvoke(None, config, context=ctx)
        await ctx.gateway.aclose()
    assert state["status"] == "partial"
    assert state["messages"][-1].text.endswith(
        "Note: the arXiv search failed (HTTPStatusError). The rest of this reply is unaffected."
    )
