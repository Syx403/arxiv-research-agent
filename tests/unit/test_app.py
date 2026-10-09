"""The conversation graph's deterministic parts: the understand trust boundary, routing, paper
choice, rendering, and interrupts with resume on a real Postgres checkpointer. Nodes that call
models are exercised by the live check (DESIGN §12: no fake LLMs)."""

from dataclasses import replace
from datetime import date
from decimal import Decimal
from typing import Any, TypeAliasType, get_args, get_type_hints, is_typeddict
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.runtime import Runtime
from pydantic import SecretStr

from ara.arxiv.client import ArxivClient
from ara.db.pool import Pool
from ara.graph import answer, app, discover, read
from ara.graph.state import (
    Answer,
    Constraint,
    Context,
    Evidence,
    PaperCard,
    Priority,
    ResearchRequest,
)
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.memory import library
from ara.settings import Settings
from tests.unit.test_search import setup


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
        "priorities": [],
        "titles": [],
        "prefer_recent": False,
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
    messages: Any = [
        HumanMessage("Read 2305.18323v1, hosted APIs only, latency matters, before 2024")
    ]
    raw = request(
        constraints=[
            Constraint(quote="Hosted  APIs only", meaning="hosted models only"),
            Constraint(quote="no training at all", meaning="invented by the model"),
        ],
        priorities=[
            Priority(quote="latency matters", meaning="low latency"),
            Priority(quote="cheap to run", meaning="invented by the model"),
        ],
        titles=[" ReWOO ", "ReWOO", ""],
        paper_ids=["2305.18323v1", "ReWOO"],
        listed=[1, 4],
        count=0,
        published_before="2024-13-01",
    )
    checked = app.checked(raw, messages, [card(1)])
    assert [c.meaning for c in checked.constraints] == ["hosted models only"]
    assert [p.meaning for p in checked.priorities] == ["low latency"]
    assert checked.titles == ["ReWOO"]
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
    assert app.after_understand(base_state(request(intent="library"))) == "library"


def base_state(r: ResearchRequest, **fields: Any) -> Any:
    state: dict[str, Any] = {
        "messages": [],
        "shown": [],
        **app.fresh_turn(),
        "request": r,
    }
    return state | fields


def test_paper_choice_never_asks() -> None:
    """D30: the count, else the direct matches, else the top of the list; at most three."""
    papers = [card(1, relevance=2), card(2), card(3), card(4), card(5, relevance=2)]
    counted = base_state(request(intent="discover_read", count=2), papers=papers)
    assert app.choose_papers(counted)["selected"] == [p.reference for p in papers[:2]]
    direct = base_state(request(intent="discover_read"), papers=papers)
    assert app.choose_papers(direct)["selected"] == [p.reference for p in papers[1:4]]
    weak = [card(n, relevance=2) for n in range(1, 6)]
    top = base_state(request(intent="discover_read"), papers=weak)
    assert app.choose_papers(top)["selected"] == [p.reference for p in weak[:3]]
    many = base_state(request(intent="discover_read", count=9), papers=papers)
    chosen: Any = app.choose_papers(many)["selected"]
    assert len(chosen) == 3


async def resolved(pool: Pool, state: Any) -> Any:
    """resolve with the user "u"; a title in the user's library is matched without arXiv."""
    async with ArxivClient() as arxiv:  # sends nothing on these paths
        ctx = context(pool, arxiv)
        runtime = Runtime(context=replace(ctx, user_id="u"))
        result = await app.resolve(state, runtime)
        await ctx.gateway.aclose()
    return result


async def test_a_read_resolves_named_ids_and_shown_positions_to_stored_ids(pool: Pool) -> None:
    state = base_state(
        request(intent="read", paper_ids=["2305.18323v1"], listed=[2]), shown=[card(1), card(2)]
    )
    found = await resolved(pool, state)
    assert found["selected"] == ["arxiv:2305.18323v1", "arxiv:2401.00002v1"]


async def test_a_paper_named_and_pointed_at_is_read_once(pool: Pool) -> None:
    named = request(intent="read", paper_ids=["2401.00002", "2210.03629"], listed=[2])
    state = base_state(named, shown=[card(1), card(2)])
    found = await resolved(pool, state)
    assert found["selected"] == ["arxiv:2401.00002v1", "arxiv:2210.03629"]


def test_the_reply_lists_papers_and_remembers_them_for_the_next_turn() -> None:
    state = base_state(request(), papers=[card(1)], shown=[card(9)])
    result: Any = app.respond(state)
    assert result["messages"][0].text.startswith("1. Paper 1 (arXiv 2401.00001v1, 2024-01-01)")
    assert result["shown"] == [card(1)]
    kept: Any = app.respond(base_state(request(intent="other"), shown=[card(9)]))
    assert kept["shown"] == [card(9)] and kept["messages"][0].text == app.OTHER
    read: Any = app.respond(base_state(request(intent="read"), read=[card(2)], shown=[card(9)]))
    assert read["shown"] == [card(2)], "a follow-up points at the papers just read"


def test_the_understand_prompt_extends_the_previous_turn() -> None:
    first: Any = [HumanMessage("Find ReWOO")]
    second: Any = [*first, AIMessage("1. ReWOO"), HumanMessage("read the first one")]
    today = date(2026, 10, 8)
    a = app.understand_prompt(first, [], today)
    b = app.understand_prompt(second, [card(1)], today)
    assert b.shared[: len(a.shared)] == a.shared and b.shared[len(a.shared)] == a.item[-1]
    assert [x.role for x in b.shared] == ["user", "assistant"]
    when, papers, latest = b.item
    assert when.text == "Today: 2026-10-08"
    assert papers.text.startswith("Papers shown last:") and '"id": "2401.00001v1"' in papers.text
    assert latest.text == "read the first one"


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


async def test_clarify_waits_with_the_question(pool: Pool) -> None:
    graph = app.build(await app.checkpointer(pool))
    thread = str(uuid4())
    config: Any = {"configurable": {"thread_id": thread}}
    await graph.aupdate_state(
        config,
        {
            "messages": [HumanMessage("find the best paper to cut my agent's cost")],
            **app.fresh_turn(),
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


def test_named_papers_are_read_without_asking_and_a_missing_one_is_reported() -> None:
    named = card(1, relevance=2).model_copy(update={"named": "ReWOO"})
    state = base_state(
        request(intent="discover_read", titles=["ReWOO", "LLMCompiler"]), papers=[named, card(2)]
    )
    assert app.choose_papers(state)["selected"] == ["arxiv:2401.00001v1"]
    assert "I could not find on arXiv: LLMCompiler." in app.reply(state)


def test_a_request_that_names_papers_reads_exactly_those() -> None:
    messages: Any = [HumanMessage("Of ReWOO and LLMCompiler, which is more advanced? ''")]
    titled = request(intent="discover_read", titles=["ReWOO", "LLMCompiler"])
    assert app.checked(titled, messages, []).intent == "read"
    blank = request(constraints=[Constraint(quote=" ", meaning="invented")])
    assert app.checked(blank, messages, []).constraints == [], "an empty quote is no quote"
    library = base_state(request(intent="library", titles=["ReWOO"]))
    assert app.after_understand(library) == "resolve", "a named paper is read, not searched for"


def test_titles_resolve_cannot_find_go_to_discovery_with_only_those() -> None:
    both = request(intent="read", titles=["ReWOO", "LLMCompiler"])
    assert app.after_resolve(base_state(both, identified=["ReWOO"])) == "discover"
    done = base_state(both, identified=["ReWOO", "LLMCompiler"], selected=["arxiv:2305.18323v1"])
    assert app.after_resolve(done) == "read"
    found = base_state(both, selected=["arxiv:2305.18323v1"], papers=[])
    assert app.after_discover(found) == "choose_papers", "what resolve found is still read"
    named = card(2).model_copy(update={"named": "LLMCompiler"})
    merged = base_state(both, selected=["arxiv:2401.00002v1"], papers=[named, card(3)])
    assert app.choose_papers(merged)["selected"] == ["arxiv:2401.00002v1"]


async def test_resolve_finds_titles_in_the_library_and_says_what_was_not_read(pool: Pool) -> None:
    gateway, _, second = await setup(pool)  # two papers titled "Cache"; the user read the second
    async with pool.connection() as conn:
        await library.record_read(conn, "u", [second])
    named = request(intent="library", titles=["cache"], paper_ids=["2305.18323"])
    found = await resolved(pool, base_state(named))
    assert found["selected"] == ["arxiv:2305.18323", "arxiv:2401.00002v1"]
    assert found["identified"] == ["cache"]
    assert found["notes"] == ["We had not read arXiv 2305.18323 before; I read it from arXiv now."]
    many = request(intent="read", paper_ids=["2305.18323", "2312.04511", "2210.03629", "2401.1"])
    messages: Any = [HumanMessage("2305.18323 2312.04511 2210.03629 2401.1")]
    capped = await resolved(pool, base_state(many, messages=messages))
    assert len(capped["selected"]) == 3 and capped["notes"][0].startswith("I read at most 3")
    await gateway.aclose()


def test_the_reply_says_when_a_named_paper_does_not_fit_or_breaks_a_constraint() -> None:
    fits = card(1)
    off = card(2).model_copy(update={"abstract": "We study cache eviction. It is fast."})
    breaks = card(3).model_copy(update={"violated": ["never fine-tune"]})
    by_screen = card(4).model_copy(update={"named": "LLMCompiler"})
    evidence = [
        Evidence(id="E1", paper_id=p.reference, chunk_id=1, paragraph=0, heading_path="", text="")
        for p in (fits, breaks, by_screen)
    ]
    state = base_state(
        request(intent="read", paper_ids=["2401.00001"]),
        read=[fits, off, breaks, by_screen],
        evidence=evidence,
        notes=["I took a note."],
    )
    text = app.reply(state)
    assert text.startswith("I took a note.")
    assert '"Paper 2" (arXiv 2401.00002v1) does not seem to discuss this' in text
    assert 'Its abstract begins: "We study cache eviction."' in text
    assert 'Paper 1" (arXiv 2401.00001v1) does not' not in text
    assert 'Note: "Paper 3" may break what you asked for ("never fine-tune")' in text
    assert 'I took "LLMCompiler" to be "Paper 4"' in text


def answer_citing(*papers: PaperCard, abstained: bool = False) -> Answer:
    return Answer(
        question="q",
        short="Not stated in the provided papers." if abstained else "yes",
        abstained=abstained,
        sentences=[],
        dropped=[],
        checked=1,
        rejected=[],
        evidence=[
            Evidence(
                id=f"E{n}",
                paper_id=p.reference,
                chunk_id=n,
                paragraph=0,
                heading_path=f"{p.title} › Method",
                text="",
            )
            for n, p in enumerate(papers, 1)
        ],
    )


def test_a_question_about_our_history_lists_relevant_papers_or_says_we_never_discussed_it() -> None:
    """D30: nothing relevant in the library → NOT_DISCUSSED; relevant papers → all listed, the
    answer cited per sentence, and the record links only what the answer cites."""
    asked = request(intent="library")
    nothing = base_state(asked, papers=[], selected=[])
    assert app.after_library(nothing) == "respond" and app.reply(nothing) == app.NOT_DISCUSSED
    assert app.episode(nothing, date(2026, 10, 9)) is None, "a miss links no paper to the topic"
    first, second = card(1, relevance=2), card(2)
    state = base_state(
        asked, papers=[second, first], read=[second, first], answer=answer_citing(second)
    )
    assert app.after_answer(state) == "respond"
    result: Any = app.respond(state)
    text = result["messages"][0].text
    assert text.startswith("From your library:\n1. Paper 2 (arXiv 2401.00002v1, 2024-01-01)")
    assert "\n2. Paper 1" in text and result["shown"] == [second, first]
    assert "\n[E1] Paper 2 › Method (arXiv 2401.00002v1)" in text
    record = app.episode(state, date(2026, 10, 9))
    assert record is not None and record.read == ["2401.00002v1 Paper 2"]
    assert record.listed == ["2401.00001v1 Paper 1"]
    assert app.relevant_first(
        [card(5, relevance=1), card(6, relevance=2), card(7, relevance=3), card(8, relevance=2)]
    ) == [card(7), card(6, relevance=2), card(8, relevance=2)]


def test_library_papers_that_do_not_answer_lead_to_one_arxiv_search() -> None:
    read_first = [card(1)]
    unanswered = base_state(
        request(intent="library"),
        papers=read_first,
        read=read_first,
        answer=answer_citing(abstained=True),
        missing=["token use per task"],
    )
    assert app.after_answer(unanswered) == "search_instead"
    named: Any = {**unanswered, "request": request(intent="library", paper_ids=["2401.00001v1"])}
    assert app.after_answer(named) == "search_instead", "also when a record named the paper"
    unnamed: Any = app.search_instead(named)["request"]
    assert not app.names(unnamed) and unnamed.intent == "library"
    moved: Any = {**unanswered, **app.search_instead(unanswered)}
    assert moved["earlier"] == read_first and moved["read"] == [] and moved["answer"] is None
    assert moved["gaps"] == ["token use per task"] and moved["missing"] == []
    two = app.searched_instead_note([card(1), card(2)], [])
    assert (
        two == 'We read "Paper 1" and "Paper 2" before, but they do not cover what you asked,'
        " so I searched arXiv:"
    )
    found: Any = {**moved, "papers": [card(3)]}
    assert app.after_discover(found) == "choose_papers"
    searched: Any = {
        **moved,
        "papers": [card(3)],
        "read": [card(3)],
        "answer": answer_citing(card(3)),
    }
    assert app.after_answer(searched) == "respond", "only once"
    text = app.reply(searched)
    assert text.startswith('We read "Paper 1" before, but it does not cover token use per task,')
    assert (
        "From your library" not in text and "\n[E1] Paper 3 › Method (arXiv 2401.00003v1)" in text
    )
    worded: Any = {
        **searched,
        "answer": answer_citing(card(3)).model_copy(update={"context": "Mine."}),
    }
    assert app.reply(worded).startswith("Mine."), "the model's opening replaces the facts (D31)"
    nothing: Any = {**moved, "answer": answer_citing(abstained=True)}
    assert "I found no arXiv papers that match this request." in app.reply(nothing)


def test_discovery_marks_papers_read_before_and_keeps_library_papers_in_the_dates() -> None:
    papers = app.marked([card(1), card(2)], {"2401.00002"})
    assert [p.read_before for p in papers] == [False, True]
    assert "(arXiv 2401.00002v1, 2024-01-01, read before)" in app._listing(2, papers[1])
    async_ctx: Any = type("Ctx", (), {"today": date(2026, 10, 9)})()
    windowed = request(published_after="2024-01-01", published_before="2025-12-31")
    old = card(3).model_copy(update={"published": "2023-12-31"})
    undated = card(4).model_copy(update={"published": ""})
    assert app._within(windowed, card(1), async_ctx) and not app._within(windowed, old, async_ctx)
    assert not app._within(request(), undated, async_ctx), "an undated paper cannot be placed"
    late = card(5).model_copy(update={"published": "2026-10-10"})
    assert not app._within(request(), late, async_ctx), "nothing after today"


def test_every_type_a_checkpoint_holds_is_registered() -> None:
    """The serializer loads an unregistered type as a plain dict, so every class reachable from a
    state schema or a Send payload must be in CHECKPOINTED (D29)."""
    found: set[tuple[str, str]] = set()

    def walk(tp: Any) -> None:
        if isinstance(tp, TypeAliasType):
            return walk(tp.__value__)
        for arg in get_args(tp):
            walk(arg)
        if not isinstance(tp, type) or not tp.__module__.startswith("ara."):
            return
        if not is_typeddict(tp):
            if (tp.__module__, tp.__name__) in found:
                return
            found.add((tp.__module__, tp.__name__))
        for hint in get_type_hints(tp).values():
            walk(hint)

    for schema in (
        app.ConversationState,
        discover.DiscoverState,
        discover.ScreenTask,
        read.ReadState,
        read.SearchTask,
        read.SelectTask,
        read.IngestTask,
        answer.AnswerState,
        answer.VerifyTask,
    ):
        walk(schema)
    assert found - set(app.CHECKPOINTED) == set()
    serde = JsonPlusSerializer(allowed_msgpack_modules=app.CHECKPOINTED)
    sample = read.Found(round=0, document=1, query="q", sentences=[], missing=[])
    assert serde.loads_typed(serde.dumps_typed(sample)) == sample
