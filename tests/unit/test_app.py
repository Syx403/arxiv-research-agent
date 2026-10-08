"""The conversation graph's deterministic parts: the understand trust boundary, routing, paper
choice, rendering, and interrupts with resume on a real Postgres checkpointer. Nodes that call
models are exercised by the live check (DESIGN §12: no fake LLMs)."""

from decimal import Decimal
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from ara.arxiv.client import ArxivClient
from ara.db.pool import Pool
from ara.graph import app
from ara.graph.state import Constraint, Context, PaperCard, ResearchRequest
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.settings import Settings

EMPTY: Any = {}


def request(**fields: Any) -> ResearchRequest:
    base: dict[str, Any] = {
        "intent": "discover",
        "clarification": None,
        "need": "KV-cache eviction",
        "question": None,
        "paper_ids": [],
        "listed": [],
        "count": None,
        "constraints": [],
        "published_after": None,
        "published_before": None,
    }
    return ResearchRequest(**(base | fields))


def card(n: int, relevance: int = 3) -> PaperCard:
    return PaperCard(
        arxiv_id=f"2401.0000{n}",
        version=1,
        title=f"Paper {n}",
        abstract="",
        published="2024-01-01",
        relevance=relevance,
        reason=f"reason {n}",
    )


def test_understand_keeps_only_literal_constraints_and_valid_references() -> None:
    messages: Any = [HumanMessage("Read 2305.18323v1, 只用现成模型 API, before 2024")]
    raw = request(
        constraints=[
            Constraint(quote="只用现成模型  API", meaning="hosted models only"),
            Constraint(quote="no training at all", meaning="invented by the model"),
        ],
        paper_ids=["2305.18323v1", "ReWOO"],
        listed=[1, 4],
        count=0,
        published_before="2024-13-01",
    )
    checked = app.checked(raw, messages, [card(1)])
    assert [c.meaning for c in checked.constraints] == ["hosted models only"]
    assert checked.paper_ids == ["2305.18323v1"]
    assert checked.listed == [1]
    assert (checked.count, checked.published_before) == (None, None)


def test_an_id_the_conversation_never_wrote_is_dropped() -> None:
    messages: Any = [
        HumanMessage("Find the ReWOO paper"),
        AIMessage("1. ReWOO (arXiv 2305.18323v1, 2023-05-23)"),
        HumanMessage("Now read 2312.04511 and that one"),
    ]
    raw = request(intent="read", paper_ids=["2312.04511", "2305.18323", "2210.03629", "2312.0451"])
    assert app.checked(raw, messages, []).paper_ids == ["2312.04511", "2305.18323"]
    recalled = request(intent="read", paper_ids=["2305.18323"])
    alone: Any = [HumanMessage("Read the ReWOO paper")]
    assert app.checked(recalled, alone, []).intent == "discover_read"


def test_a_read_request_without_a_paper_becomes_find_then_read() -> None:
    messages: Any = [HumanMessage("Read the ReWOO paper and explain its planner")]
    assert app.checked(request(intent="read"), messages, []).intent == "discover_read"


def test_routing_after_understand_bounds_clarification() -> None:
    asking = request(clarification="Which cost?")
    assert app.after_understand(base_state(asking)) == "clarify"
    assert app.after_understand(base_state(asking, clarifications=2)) == "discover"
    assert app.after_understand(base_state(request(intent="read", paper_ids=["1"]))) == "resolve"
    assert app.after_understand(base_state(request(intent="library"))) == "respond"


def base_state(r: ResearchRequest, **fields: Any) -> Any:
    state: dict[str, Any] = {
        "messages": [],
        "shown": [],
        **app.load_context(EMPTY),
        "request": r,
    }
    return state | fields


def test_paper_choice_is_deterministic_when_the_count_or_direct_matches_decide() -> None:
    papers = [card(1), card(2), card(3, relevance=2)]
    counted = base_state(request(intent="discover_read", count=3), papers=papers)
    assert app.choose_papers(counted)["selected"] == [p.reference for p in papers]
    direct = base_state(request(intent="discover_read"), papers=papers)
    assert app.choose_papers(direct)["selected"] == ["arxiv:2401.00001v1", "arxiv:2401.00002v1"]


def test_a_read_resolves_named_ids_and_shown_positions_to_stored_ids() -> None:
    state = base_state(
        request(intent="read", paper_ids=["2305.18323v1"], listed=[2]), shown=[card(1), card(2)]
    )
    assert app.resolve(state)["selected"] == ["arxiv:2305.18323v1", "arxiv:2401.00002v1"]


def test_a_paper_named_and_pointed_at_is_read_once() -> None:
    named = request(intent="read", paper_ids=["2401.00002", "2210.03629"], listed=[2])
    state = base_state(named, shown=[card(1), card(2)])
    assert app.resolve(state)["selected"] == ["arxiv:2401.00002v1", "arxiv:2210.03629"]


def test_the_reply_lists_papers_and_remembers_them_for_the_next_turn() -> None:
    state = base_state(request(), papers=[card(1)], shown=[card(9)])
    result: Any = app.respond(state)
    assert result["messages"][0].text.startswith("1. Paper 1 (arXiv 2401.00001v1, 2024-01-01)")
    assert result["shown"] == [card(1)]
    kept: Any = app.respond(base_state(request(intent="other"), shown=[card(9)]))
    assert kept["shown"] == [card(9)] and kept["messages"][0].text == app.OTHER


def test_the_understand_prompt_extends_the_previous_turn() -> None:
    first: Any = [HumanMessage("Find ReWOO")]
    second: Any = [*first, AIMessage("1. ReWOO"), HumanMessage("read the first one")]
    a, b = app.understand_prompt(first, []), app.understand_prompt(second, [card(1)])
    assert b.shared[: len(a.shared) + len(a.item)] == a.shared + a.item
    assert [x.role for x in b.shared] == ["user", "assistant"]
    papers, latest = b.item
    assert papers.text.startswith("Papers shown last:") and '"id": "2401.00001v1"' in papers.text
    assert latest.text == "read the first one"


def test_resume_positions_accept_a_list_or_text() -> None:
    assert app._positions([1, 3]) == [1, 3]
    assert app._positions("1 and 3") == [1, 3]


def context(pool: Pool, arxiv: ArxivClient) -> Context:
    """A real gateway with unused keys: the paths below never reach a model."""
    settings = Settings(
        openai_api_key=SecretStr("unused"),
        deepseek_api_key=SecretStr("unused"),
        cohere_api_key=SecretStr("unused"),
    )
    ledger = Ledger(pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=Decimal(1))
    gateway = Gateway(settings, ledger)

    async def fetch(reference: str) -> Any:
        raise AssertionError(f"fetched {reference}")

    return Context(pool, gateway, Scope(), fetch, arxiv)


async def test_an_ambiguous_choice_waits_for_the_user_and_resumes(pool: Pool) -> None:
    graph = app.build(await app.checkpointer(pool))
    thread = str(uuid4())
    config: Any = {"configurable": {"thread_id": thread}}
    papers = [card(n) for n in range(1, 5)]  # four direct matches, more than can be read
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage("find and read")],
            **app.load_context(EMPTY),
            "request": request(intent="discover_read"),
            "papers": papers,
        },
        as_node="discover",
    )
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        waiting = await graph.ainvoke(None, config, context=ctx)
        assert waiting["__interrupt__"][0].value["kind"] == "choose_papers"
        assert len(waiting["__interrupt__"][0].value["papers"]) == 4
        turn = await app.send(graph, thread, ctx, resume="none of them")
    assert turn.waiting is None and turn.reply is not None
    assert "Nothing was chosen to read." in turn.reply
    assert turn.state["status"] == "complete"
    await ctx.gateway.aclose()


async def test_clarify_waits_with_the_question(pool: Pool) -> None:
    graph = app.build(await app.checkpointer(pool))
    thread = str(uuid4())
    config: Any = {"configurable": {"thread_id": thread}}
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage("find the best paper to cut my agent's cost")],
            **app.load_context(EMPTY),
            "request": request(clarification="Which cost: API spend or latency?"),
        },
        as_node="understand",
    )
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        state = await graph.ainvoke(None, config, context=ctx)
    assert state["__interrupt__"][0].value == {
        "kind": "clarify",
        "question": "Which cost: API spend or latency?",
    }
    await ctx.gateway.aclose()
