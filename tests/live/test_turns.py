"""M3 live check (DESIGN §14: a live discover → read turn and a clarify turn), through the top-level
graph with the Postgres checkpointer, real arXiv and real models.

Approved by Ewan on 2026-10-08 with the M3 debugging runs: about 40 requests, US$0.05 in all.
- clarify: a vague need stops with a question; the answer resumes the turn, which discovers papers
  and keeps the user's constraint (2 understand calls, then discovery: ≤ 8 researcher steps, 1
  embedding request, up to 3 screen batches);
- discover → read: "find 2 papers ... and compare" lists papers, reads the top two and answers
  with verified citations (discovery as above, then ingestion, evidence selection, synthesize and
  verification).
Run one with `uv run pytest -m live tests/live/test_turns.py -k <name> -s`.
"""

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from ara.arxiv.client import ArxivClient
from ara.graph import app
from ara.graph.state import Context
from ara.llm.gateway import Gateway
from ara.llm.ledger import Scope

pytestmark = pytest.mark.live

RUN_ID = f"m3-live-{datetime.now(UTC):%Y%m%dT%H%M%S}"
RUN_CAP = Decimal("0.03")  # per module run; the M3 approval is US$0.05 over all runs


@pytest.fixture
async def ctx(gateway: Gateway) -> AsyncIterator[tuple[Any, Callable[[], Context]]]:
    async with ArxivClient() as arxiv:
        graph = app.build(await app.checkpointer(gateway.ledger.pool))

        async def fetch(reference: str) -> Any:
            return await arxiv.paper(reference.removeprefix("arxiv:"))

        def turn() -> Context:
            scope = Scope(run_id=RUN_ID, turn_id=str(uuid4()), run_cap_usd=RUN_CAP)
            return Context(gateway.ledger.pool, gateway, scope, fetch, arxiv)

        yield graph, turn


def show(turn: app.Turn) -> None:
    state = turn.state
    request = state.get("request")
    print("\nrequest:", request.model_dump_json(indent=1) if request else None)
    for p in state.get("papers", []):
        print(f"  {p.arxiv_id} r={p.relevance} s={p.similarity:.3f} {p.title[:70]}")
    print("selected:", state.get("selected"))
    print("waiting:", turn.waiting)
    print("reply:\n", turn.reply)


async def test_a_vague_need_is_clarified_then_discovered(ctx: Any) -> None:
    graph, turn = ctx
    thread = str(uuid4())
    asked = await app.send(
        graph, thread, turn(), message="Find the best paper to cut my agent's cost"
    )
    show(asked)
    assert asked.waiting is not None and asked.waiting["kind"] == "clarify"
    answered = await app.send(
        graph,
        thread,
        turn(),
        resume="API spend: my agent makes many tool calls through hosted LLM APIs."
        " I only use hosted APIs, no fine-tuning.",
    )
    show(answered)
    state = answered.state
    assert state["status"] == "complete" and 1 <= len(state["papers"]) <= 5
    assert state["request"].constraints, "the no-fine-tuning constraint was dropped"
    assert not any(p.violated for p in state["papers"])


async def test_find_two_papers_then_read_and_compare(ctx: Any) -> None:
    graph, turn = ctx
    done = await app.send(
        graph,
        str(uuid4()),
        turn(),
        message="Find 2 papers on scheduling LLM tool calls in parallel and compare their"
        " mechanisms.",
    )
    show(done)
    state = done.state
    assert done.waiting is None and len(state["selected"]) == 2
    answer = state["answer"]
    assert answer is not None and not answer.abstained and answer.sentences
