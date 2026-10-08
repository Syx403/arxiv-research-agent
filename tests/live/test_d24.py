"""D24 live check, approved by Ewan on 2026-10-08: about 40 requests, US$0.02 in all.
- understand on five representative S4 items (5 Luna requests);
- a "latest papers" topic search: prefer_recent on, newest first within each relevance grade;
- a read by ids, then "Which approach should I use?", which must point at the two papers just read.
Run one with `uv run pytest -m live tests/live/test_d24.py -k <name> -s`.
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
from evals.suites import s4

pytestmark = pytest.mark.live

RUN_ID = f"d24-live-{datetime.now(UTC):%Y%m%dT%H%M%S}"
RUN_CAP = Decimal("0.02")  # the whole approval; the module never needs more
ITEMS = ("f-which", "t-two-years", "p-mixed", "v1-c12", "d-broad")


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


async def test_understand_on_representative_items(gateway: Gateway) -> None:
    items = {i.id: i for i in s4.load()}
    scope = Scope(run_id=RUN_ID, run_cap_usd=RUN_CAP)
    failed = []
    for name in ITEMS:
        item = items[name]
        request = await s4.understand(gateway, item, scope)
        metrics = s4.score(request, item)
        wrong = [k for k, v in metrics.items() if k.startswith("field_") and v == 0.0]
        print(f"\n{name}: intent {request.intent}, wrong fields {wrong}")
        print(request.model_dump_json(indent=1))
        if request.intent != item.expected["intent"]:
            failed.append(name)
    # clarifications and fields are what S4 measures; this check only needs the new inputs to work
    assert not failed, f"intent wrong on {failed}"


async def test_latest_papers_come_newest_first(ctx: Any) -> None:
    graph, turn = ctx
    done = await app.send(
        graph, str(uuid4()), turn(), message="What are the latest papers on agentic RAG?"
    )
    state = done.state
    print("\n", done.reply)
    assert state["request"].prefer_recent and 1 <= len(state["papers"]) <= 5
    for a, b in zip(state["papers"], state["papers"][1:], strict=False):
        assert (a.relevance, a.published) >= (b.relevance, b.published), "not newest first"


async def test_a_follow_up_points_at_the_papers_just_read(ctx: Any) -> None:
    graph, turn = ctx
    thread = str(uuid4())
    first = await app.send(
        graph,
        thread,
        turn(),
        message="Compare 2305.18323 and 2312.04511 on how they handle tool outputs",
    )
    print("\n", first.reply)
    assert [p.arxiv_id for p in first.state["shown"]] == ["2305.18323", "2312.04511"]
    follow = await app.send(graph, thread, turn(), message="Which approach should I use?")
    print("\n", follow.reply)
    request = follow.state["request"]
    assert request.intent == "read" and sorted(request.listed) == [1, 2]
    assert sorted(follow.state["selected"]) == ["arxiv:2305.18323v1", "arxiv:2312.04511v3"]
