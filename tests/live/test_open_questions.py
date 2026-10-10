"""D40 live check: open questions are answered, in the user's language, and a remembered fact does
not bend a request it does not concern. Through the top-level graph with the Postgres checkpointer,
the Store, real arXiv and real models, as the simulated user "live-check" (never the app's own
user, so the app's memory stays the user's own; `ara db prune` removes these threads and memories).

Approved by Ewan on 2026-10-10 with the D40 fixes: four research turns and one memory turn, about
120 requests, estimated US$0.035, capped at US$0.06.
- a Chinese open question about a model ("详细介绍一下kimi k3的架构"): find, read, a verified answer
  in Chinese, even if its one-line answer is not verified;
- an English open question about a named paper: a verified explanation, with at most one rerank
  call (requeries keep the hybrid order);
- "I only use hosted model APIs" remembered, then a Chinese search for Kimi K3 papers: no
  constraint or priority comes from the fact, the K3 report is among the first two, and the
  reasons are in Chinese.
Run one with `uv run pytest -m live tests/live/test_open_questions.py -k <name> -s`.
"""

import re
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
from ara.memory import store

pytestmark = pytest.mark.live

RUN_ID = f"d40-live-{datetime.now(UTC):%Y%m%dT%H%M%S}"
RUN_CAP = Decimal("0.04")  # per run: the first run spent US$0.0162 of the US$0.06 approval
USER = "live-check"
CJK = re.compile(r"[一-鿿]")
K3 = "2607.24653"


@pytest.fixture
async def ctx(gateway: Gateway) -> AsyncIterator[tuple[Any, Callable[[], Context]]]:
    async with ArxivClient() as arxiv:
        memory = await store.open_store(gateway.ledger.pool, gateway)
        for key in await _facts(memory):
            await store.forget(memory, USER, key)
        graph = app.build(await app.checkpointer(gateway.ledger.pool), memory)

        async def fetch(reference: str) -> Any:
            return await arxiv.paper(reference.removeprefix("arxiv:"))

        def turn() -> Context:
            scope = Scope(run_id=RUN_ID, turn_id=f"{RUN_ID}:{uuid4().hex[:8]}", run_cap_usd=RUN_CAP)
            return Context(gateway.ledger.pool, gateway, scope, fetch, arxiv, user_id=USER)

        yield graph, turn


async def _facts(memory: Any) -> list[str]:
    return [i.key for i in await memory.asearch(store.profile_namespace(USER), limit=50)]


def show(turn: app.Turn) -> None:
    state = turn.state
    request = state.get("request")
    print("\nrequest:", request.model_dump_json(indent=1) if request else None)
    answer = state.get("answer")
    if answer is not None:
        print(f"answer: abstained={answer.abstained} withheld={answer.withheld}")
        print(f"  checked={answer.checked} delivered={len(answer.sentences)}")
        print(f"  dropped={len(answer.dropped)} rejected={[c.text for c in answer.rejected]}")
    print("problems:", state.get("problems"))
    print("reply:\n", turn.reply)


async def calls(ctx: Context, stage: str) -> int:
    async with ctx.pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT count(*) AS n FROM llm_calls WHERE turn_id = %s AND stage = %s",
            (ctx.scope.turn_id, stage),
        )
        row = await cursor.fetchone()
    return int(row["n"]) if row else 0


async def test_a_chinese_open_question_about_a_model_is_answered_in_chinese(ctx: Any) -> None:
    graph, turn = ctx
    asked = await app.send(graph, str(uuid4()), turn(), message="详细介绍一下kimi k3的架构")
    show(asked)
    answer = asked.state["answer"]
    assert asked.state["request"].language == "Chinese"
    assert answer is not None and len(answer.sentences) >= 4, "the verified explanation stands"
    assert all(CJK.search(c.text) for c in answer.sentences)


async def test_an_open_question_on_a_named_paper_reranks_once(ctx: Any) -> None:
    graph, turn = ctx
    context = turn()
    asked = await app.send(
        graph,
        str(uuid4()),
        context,
        message="Read 2305.18323 and explain how its planner, workers and solver work together.",
    )
    show(asked)
    answer = asked.state["answer"]
    assert asked.state["request"].language == "English"
    assert answer is not None and len(answer.sentences) >= 4
    assert await calls(context, "rerank") <= 1, "requeries keep the hybrid order (D40)"


async def test_a_remembered_fact_does_not_bend_an_unrelated_search(ctx: Any) -> None:
    graph, turn = ctx
    noted = await app.send(
        graph, str(uuid4()), turn(), message="Remember that I only use hosted model APIs."
    )
    show(noted)
    found = await app.send(graph, str(uuid4()), turn(), message="找kimi k3相关论文")
    show(found)
    request, papers = found.state["request"], found.state["papers"]
    assert found.state["profile"], "the fact was remembered and loaded"
    assert (request.constraints, request.priorities) == ([], [])
    assert K3 in [p.arxiv_id for p in papers[:2]]
    assert all(CJK.search(p.reason) for p in papers)
