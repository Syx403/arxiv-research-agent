"""Memory (D27) against the real Store and library tables: profile facts, research records, the
remember trust boundary, and the library search. Texts the Store embeds are put in the embedding
cache first, so no request is sent (as in test_search)."""

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import date
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import SecretStr

from ara.arxiv.client import ArxivClient
from ara.db.pool import Pool
from ara.graph import app
from ara.graph.state import Constraint
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger
from ara.memory import extract, library, store
from ara.memory.extract import MemoryUpdate
from ara.memory.store import Episode, Fact, Remembered
from ara.rag.embed import cache_key
from ara.rag.ingest import set_published
from ara.settings import Settings
from tests.unit.test_app import base_state, card, context, request
from tests.unit.test_search import setup, unit

KV = Episode(
    day="2026-10-01", need="KV-cache eviction", read=["2306.14048v3 H2O"], listed=[], answer=""
)
MOE = Episode(
    day="2026-10-02", need="MoE routing", read=[], listed=["2401.04088v1 Mixtral"], answer="top-2"
)


async def cache(pool: Pool, vectors: dict[str, int]) -> None:
    """Text → the i-th unit vector, as if it had been embedded before."""
    async with pool.connection() as conn:
        await conn.cursor().executemany(
            "INSERT INTO embedding_cache (key, model, embedding) VALUES (%s, %s, %s)",
            [(cache_key(text), "test", unit(i)) for text, i in vectors.items()],
        )


@pytest.fixture
async def memory(pool: Pool) -> AsyncIterator[Any]:
    settings = Settings(
        openai_api_key=SecretStr("unused"),
        deepseek_api_key=SecretStr("unused"),
        cohere_api_key=SecretStr("unused"),
    )
    ledger = Ledger(
        pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
    )
    gateway = Gateway(settings, ledger)
    opened = await store.open_store(pool, gateway)
    async with pool.connection() as conn:
        await conn.execute("TRUNCATE store, store_vectors")
    yield opened
    await gateway.aclose()


def fact(key: str, quote: str) -> Remembered:
    said = Fact(key=key, quote=quote, statement=f"The user said {quote}.")
    return Remembered(**said.model_dump(), thread="t", turn=0, day="2026-10-09")


async def test_a_newer_fact_replaces_the_old_one_and_forget_deletes(memory: Any) -> None:
    await store.remember(memory, "u", fact("access", "hosted APIs only"))
    await store.remember(memory, "u", fact("access", "I can run local models now"))
    await store.remember(memory, "u", fact("goal", "latency matters most"))
    assert [f.quote for f in await store.profile(memory, "u")] == [
        "I can run local models now",
        "latency matters most",
    ]
    await store.forget(memory, "u", "goal")
    assert [f.key for f in await store.profile(memory, "u")] == ["access"]
    assert await store.profile(memory, "someone else") == []


async def test_research_records_are_recalled_by_meaning(pool: Pool, memory: Any) -> None:
    await cache(pool, {"KV-cache eviction": 1, "MoE routing": 2, "how do experts get picked?": 2})
    await store.record(memory, "u", "a", KV)
    await store.record(memory, "u", "b", MOE)
    assert (await store.recall(memory, "u", "how do experts get picked?"))[0] == MOE


def test_remember_keeps_only_quotes_from_this_turn_and_known_keys() -> None:
    update = MemoryUpdate(
        facts=[
            fact("access", "Hosted  APIs only"),
            fact("goal", "I never said this"),
            fact("empty", "  "),
        ],
        forget=["access", "unknown", "goal"],
    )
    profile = [fact("goal", "latency"), fact("access", "old")]
    kept = extract.checked(update, profile, ["I use hosted APIs only."])
    assert [f.key for f in kept.facts] == ["access"], "an empty quote is no quote"
    assert kept.forget == ["goal"], "a key kept this turn is not also forgotten"


async def test_the_library_returns_the_users_papers_nearest_the_question(pool: Pool) -> None:
    gateway, _, second = await setup(pool)
    async with pool.connection() as conn:
        await library.record_read(conn, "u", [second])
        await library.record_read(conn, "u", [second])  # read again: one row
        rows = await (await conn.execute("SELECT paper_id FROM library_items")).fetchall()
        assert [r["paper_id"] for r in rows] == ["arxiv:2401.00002v1"]
        near = await library.closest_papers(conn, "u", unit(1), limit=3)
        assert near == ["arxiv:2401.00002v1"], "only the user's papers"
        assert await library.closest_papers(conn, "nobody", unit(1), limit=3) == []
    await gateway.aclose()


def test_understand_sees_the_profile_before_the_conversation_and_research_in_the_item() -> None:
    messages: Any = [HumanMessage("what did the H2O paper find?")]
    prompt = app.understand_prompt(
        messages, [], date(2026, 10, 9), [fact("access", "hosted APIs only")], [KV]
    )
    assert prompt.shared[0].text.startswith("About the user:")
    assert '"quote": "hosted APIs only"' in prompt.shared[0].text
    assert any(
        b.text.startswith("Earlier research:") and "2306.14048v3" in b.text for b in prompt.item
    )


def test_remembered_quotes_and_recorded_ids_pass_the_trust_boundary() -> None:
    messages: Any = [HumanMessage("Find papers on agent memory, and re-read the H2O one")]
    raw = request(
        intent="read",
        paper_ids=["2306.14048v3"],
        constraints=[Constraint(quote="hosted APIs only", meaning="hosted")],
    )
    alone = app.checked(raw, messages, [])
    assert alone.paper_ids == [] and alone.constraints == []
    remembered = app.checked(raw, messages, [], [fact("access", "hosted APIs only")], [KV])
    assert remembered.paper_ids == ["2306.14048v3"] and len(remembered.constraints) == 1


def test_memory_turns_remember_first_and_other_turns_after_replying() -> None:
    memory_turn = base_state(request(intent="memory"))
    assert app.after_understand(memory_turn) == "remember"
    assert (
        app.after_remember(memory_turn) == "respond" and app.after_respond(memory_turn) == "__end__"
    )
    research = base_state(request(intent="discover"))
    assert app.after_respond(research) == "remember" and app.after_remember(research) == "__end__"
    assert app.after_understand(base_state(request(intent="library"))) == "library"
    unsure = request(intent="library", clarification="Which papers do you mean?")
    assert app.after_understand(base_state(unsure)) == "library", "the library search answers"


def test_a_memory_turn_says_what_changed() -> None:
    state = base_state(
        request(intent="memory"),
        profile=[fact("goal", "latency")],
        memory=MemoryUpdate(facts=[fact("access", "hosted APIs only")], forget=["goal"]),
    )
    assert app.memory_reply(state) == (
        "Noted: The user said hosted APIs only.\nForgotten: The user said latency."
    )
    assert "nothing to remember" in app.memory_reply(base_state(request(intent="memory")))


def test_a_library_question_nothing_read_covers_offers_a_search() -> None:
    nothing = base_state(request(intent="library"), selected=["arxiv:2210.03629v3"], answer=None)
    assert app.reply(nothing) == app.NOT_DISCUSSED
    assert "search arXiv" in app.NOT_DISCUSSED


async def test_remember_calls_no_model_when_the_constraints_came_from_the_profile(
    pool: Pool, memory: Any
) -> None:
    """The gateway has unused keys: a model call here would fail the test."""
    await cache(pool, {"KV-cache eviction": 1})  # the record is indexed by its need
    remembered = fact("training", "I never fine-tune models")
    raw = request(constraints=[Constraint(quote="I never fine-tune models", meaning="no tuning")])
    state = base_state(
        raw,
        messages=[HumanMessage("Find papers on tool latency")],
        profile=[remembered],
        papers=[card(1)],
    )
    async with ArxivClient() as arxiv:
        ctx = replace(context(pool, arxiv), user_id="u")
        result = await app.remember(state, Runtime(context=ctx, store=memory))
        await ctx.gateway.aclose()
    assert result == {"memory": MemoryUpdate(facts=[], forget=[])}
    [record] = await memory.asearch(("users", "u", "episodes"))
    assert record.value["listed"] == ["2401.00001v1 Paper 1"]


async def test_library_candidates_fuse_passage_distance_with_research_records(
    pool: Pool, memory: Any
) -> None:
    gateway, first, second = await setup(pool)  # identical papers: passages tie
    await cache(pool, {"what about cache eviction?": 1})
    async with pool.connection() as conn:
        await library.record_read(conn, "u", [first, second])
        await set_published(conn, {"2401.00001": date(2024, 1, 1)})
        dated = await library.papers(conn, "u")
        assert [(p.arxiv_id, p.published) for p in dated] == [
            ("2401.00001", "2024-01-01"),
            ("2401.00002", ""),
        ]
        nearest = await library.passages_near(conn, [p.reference for p in dated], unit(1), 2)
        assert [len(v) for v in nearest.values()] == [2, 2]
        assert (
            nearest["arxiv:2401.00001v1"][0]
            == "Cache › Method: Evicting keys by attention score reduces memory."
        )
    record = Episode(
        day="d", need="n", read=["2401.00002v1 Cache", "2999.00001v1 X"], listed=[], answer=""
    )
    async with ArxivClient() as arxiv:
        ctx = replace(context(pool, arxiv), user_id="u")
        found, owned = await app.library_candidates(ctx, "what about cache eviction?", [record], 5)
        alone = replace(ctx, user_id="nobody")
        assert await app.library_candidates(alone, "never embedded", [record], 5) == ([], set())
        empty: Any = base_state(request(intent="library", question="never embedded"))
        assert await app.find_in_library(empty, Runtime(context=alone)) == {
            "papers": [],
            "selected": [],
        }, "an empty library answers without a request"
        await ctx.gateway.aclose()
    assert [p.arxiv_id for p in found] == ["2401.00002", "2401.00001"], "the record decides the tie"
    assert owned == {"2401.00001", "2401.00002"}
    await gateway.aclose()


def test_a_question_about_our_history_names_only_ids_the_user_wrote() -> None:
    """D32: an id taken from a research record does not make a library question a named read;
    the library search, which ranks that record's papers, decides."""
    messages: Any = [HumanMessage("What did the papers we read about KV caches find?")]
    raw = request(intent="library", paper_ids=["2306.14048v3"])
    assert app.checked(raw, messages, [], [], [KV]).paper_ids == []
    written: Any = [HumanMessage("What did 2306.14048 we read find?")]
    assert app.checked(raw, written, [], [], [KV]).paper_ids == ["2306.14048v3"]


async def test_a_shared_name_resolves_to_the_paper_closest_to_the_need(pool: Pool) -> None:
    """D32: "Gorilla" names a consensus protocol and an API-calling LLM; the need decides."""
    consensus = card(1).model_copy(update={"title": "Gorilla: consensus", "abstract": "a"})
    apis = card(2).model_copy(update={"title": "Gorilla: APIs", "abstract": "a"})
    await cache(pool, {"API hallucination": 1, "Gorilla: consensus. a": 2, "Gorilla: APIs. a": 1})
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        ranked = await app.closest_to("API hallucination", [consensus, apis], ctx)
        await ctx.gateway.aclose()
    assert ranked == [apis, consensus]
