"""Discovery's deterministic parts: the tool budget, what the researcher sees, ranking, and how a
tool turn is sent back to DeepSeek. The researcher and screen calls are checked live."""

from datetime import date
from typing import Any

from langgraph.runtime import Runtime

from ara.arxiv.client import ArxivClient, Metadata
from ara.db.pool import Pool
from ara.graph import discover
from ara.graph.state import MAX_LISTED, Constraint, PaperCard, ResearchRequest
from ara.llm.prompt import Block, Instructions, Prompt, ToolCall, deepseek_messages
from tests.unit.test_app import context
from tests.unit.test_memory import cache


def meta(n: int, published: date = date(2024, 1, 1)) -> Metadata:
    return Metadata(f"2401.0000{n}", 2, f"Paper {n}", "A long abstract. " * 40, published)


def request(**fields: Any) -> ResearchRequest:
    base: dict[str, Any] = {
        "intent": "discover",
        "language": "English",
        "clarification": None,
        "need": "need",
        "question": None,
        "paper_ids": [],
        "listed": [],
        "count": None,
        "constraints": [],
        "priorities": [],
        "titles": [],
        "names": [],
        "prefer_recent": False,
        "published_after": None,
        "published_before": None,
        "history": None,
    }
    return ResearchRequest(**(base | fields))


def test_a_paper_seen_before_is_shown_by_title_only() -> None:
    found: dict[str, PaperCard] = {}
    first = discover.describe([meta(1)], found)
    again = discover.describe([meta(1), meta(2)], found)
    assert first.count("\n") == 1 and len(first) < 400
    assert again.splitlines()[0] == "2401.00001 (2024) Paper 1 [already found]"
    assert list(found) == ["2401.00001", "2401.00002"]


def test_the_loop_stops_when_the_model_stops_or_the_budget_is_spent() -> None:
    calling = Block("assistant", "", calls=(ToolCall("c1", "search_arxiv", "{}"),))
    state: Any = {"transcript": [calling], "tool_calls": 3}
    assert discover.after_researcher(state) == "arxiv_tools"
    stopped: Any = {**state, "transcript": [Block("assistant", "done")]}
    spent: Any = {**state, "tool_calls": discover.MAX_TOOL_CALLS}
    assert discover.after_researcher(stopped) == "prerank"
    assert discover.after_tools(spent) == "prerank"


def judge(
    paper_id: str, relevance: Any, violated: list[str] | None = None, named: str | None = None
) -> discover.Judgement:
    return discover.Judgement(
        id=paper_id, relevance=relevance, reason="", violated=violated or [], named=named
    )


def card(n: int, similarity: float) -> PaperCard:
    m = meta(n)
    return discover.card(m).model_copy(update={"similarity": similarity})


def test_rank_keeps_relevant_papers_that_break_no_quoted_constraint() -> None:
    constraint = Constraint(quote="no fine-tuning", meaning="no training")
    shortlist = [card(1, 0.9), card(2, 0.8), card(3, 0.7), card(4, 0.95)]
    judged = [
        judge("2401.00001", 2, []),
        judge("2401.00002", 3, []),
        judge("2401.00003", 3, ["No  Fine-tuning"]),
        judge("2401.00004", 1, ["made up"]),
    ]
    state: Any = {
        "request": request(constraints=[constraint], count=5),
        "shortlist": [*shortlist, card(5, 0.99)],
        "judged": judged,
    }
    ranked: Any = discover.rank(state)
    assert [p.arxiv_id for p in ranked["papers"]] == ["2401.00002", "2401.00001"]
    assert ranked["unjudged"] == ["2401.00005"]


def test_screen_judgements_are_matched_by_bare_id_within_the_batch() -> None:
    def judgement(paper_id: str) -> discover.Judgement:
        return judge(paper_id, 3)

    batch = [card(1, 0.0), card(2, 0.0)]
    kept = discover.in_batch(
        [judgement("2401.00001v2"), judgement("arXiv:2401.00002"), judgement("2401.00009")], batch
    )
    assert [j.id for j in kept] == ["2401.00001", "2401.00002"]


def test_screening_batches_share_the_request_prefix() -> None:
    shortlist = [card(n, 0.0) for n in range(1, 10)]
    groups = discover.batches(shortlist)
    assert [len(g) for g in groups] == [8, 1]
    a, b = (discover.screen_prompt(request(), g) for g in groups)
    assert a.shared == b.shared and a.item != b.item


def test_a_tool_turn_is_sent_back_with_its_reasoning_and_call_ids() -> None:
    call = ToolCall("call_1", "search_arxiv", '{"query": "abs:eviction"}')
    prompt = Prompt(
        Instructions("researcher", "Search."),
        item=(
            Block("assistant", "", reasoning="think", calls=(call,)),
            Block("tool", "2401.00001 (2024) Paper 1", call_id="call_1"),
        ),
    )
    assistant, tool = deepseek_messages(prompt)[1:]
    assert assistant == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "think",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "search_arxiv", "arguments": '{"query": "abs:eviction"}'},
            }
        ],
    }
    assert tool == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "2401.00001 (2024) Paper 1",
    }


def test_recent_work_comes_first_within_a_relevance_grade() -> None:
    old, new, best = card(1, 0.9), card(2, 0.5), card(3, 0.1)
    new = new.model_copy(update={"published": "2026-01-01"})
    judged = [
        judge("2401.00001", 2, []),
        judge("2401.00002", 2, []),
        judge("2401.00003", 3, []),
    ]
    state: Any = {"request": request(), "shortlist": [old, new, best], "judged": judged}
    by_similarity: Any = discover.rank(state)
    assert [p.arxiv_id for p in by_similarity["papers"]] == [
        "2401.00003",
        "2401.00001",
        "2401.00002",
    ]
    state["request"] = request(prefer_recent=True)
    by_date: Any = discover.rank(state)
    assert [p.arxiv_id for p in by_date["papers"]] == ["2401.00003", "2401.00002", "2401.00001"]


def test_the_request_block_carries_priorities_titles_and_recency() -> None:
    block = discover.request_block(request(titles=["ReWOO"], prefer_recent=True))
    assert '"titles": ["ReWOO"]' in block.text and '"prefer_recent": true' in block.text


def test_a_named_paper_is_listed_first_whatever_screening_thought_of_it() -> None:
    rewoo = card(1, 0.2).model_copy(update={"title": "ReWOO: Decoupling Reasoning"})
    compiler = card(2, 0.1).model_copy(update={"title": "An LLM Compiler for Parallel Calls"})
    other = card(3, 0.9)
    state: Any = {
        "request": request(titles=["ReWOO", "LLMCompiler"], constraints=[]),
        "shortlist": [other, rewoo, compiler],
        "judged": [
            judge("2401.00003", 3),
            judge("2401.00001", 1),  # off-topic for the need, but named by its title
            judge("2401.00002", 2, named="LLMCompiler"),  # named, recognised by the screen
        ],
    }
    ranked: Any = discover.rank(state)
    assert [(p.arxiv_id, p.named) for p in ranked["papers"]] == [
        ("2401.00001", "ReWOO"),
        ("2401.00002", "LLMCompiler"),
        ("2401.00003", None),
    ]


def test_a_title_names_a_paper_when_equal_or_its_prefix() -> None:
    assert discover.name(["ReWOO"], "ReWOO: Decoupling Reasoning", None) == "ReWOO"
    assert discover.name(["Attention Is All You Need"], "Attention is all you need", None)
    assert discover.name(["ReWOO"], "Beyond ReWOO: planning", None) is None
    assert discover.name(["GQA"], "Grouped queries", "gqa") == "GQA"


def test_named_papers_never_push_the_list_past_its_limit() -> None:
    titles = [f"Paper {n}" for n in range(1, 8)]
    shortlist = [card(n, 0.5).model_copy(update={"title": f"Paper {n}"}) for n in range(1, 8)]
    judged = [judge(p.arxiv_id, 1) for p in shortlist]
    state: Any = {"request": request(titles=titles), "shortlist": shortlist, "judged": judged}
    ranked: Any = discover.rank(state)
    assert len(ranked["papers"]) == MAX_LISTED
    assert not discover.titled("", "Anything"), "an empty name names nothing"


async def test_prerank_always_shortlists_a_paper_named_by_title(pool: Pool) -> None:
    others = [
        card(n, 0.0).model_copy(update={"title": f"T{n}", "abstract": "a"}) for n in range(1, 26)
    ]
    named = card(99, 0.0).model_copy(update={"title": "ReWOO: Decoupling", "abstract": "a"})
    found = {c.arxiv_id: c for c in [*others, named]}
    vectors = {"need": 1, **{f"T{n}. a": 1 for n in range(1, 26)}, "ReWOO: Decoupling. a": 2}
    await cache(pool, vectors)
    library = card(77, 0.0).model_copy(update={"title": "T77", "abstract": "a"})
    await cache(pool, {"T77. a": 3})
    state: Any = {
        "request": request(need="need", titles=["ReWOO"]),
        "found": found,
        "library": [library, named],
    }
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        result: Any = await discover.prerank(state, Runtime(context=ctx))
        await ctx.gateway.aclose()
    assert result["shortlisted"][0] == named.arxiv_id and len(result["shortlisted"]) == 24
    assert library.arxiv_id in result["candidates"], "the user's paper joins the pool (D30)"
    assert result["candidates"].count(named.arxiv_id) == 1, "a paper found twice counts once"


async def test_prerank_puts_papers_naming_the_subject_first(pool: Pool) -> None:
    """D41: a paper that names "Kimi K3" in its abstract is screened before closer-sounding ones,
    up to MENTIONS; the shortlist grows past SHORTLIST only for such papers."""
    near = [
        card(n, 0.0).model_copy(update={"title": f"Kimi-VL {n}", "abstract": "b"})
        for n in range(1, 31)
    ]
    naming = [
        card(100 + n, 0.0).model_copy(update={"title": f"M{n}", "abstract": f"uses Kimi-K3 {n}"})
        for n in range(1, 46)
    ]
    found = {c.arxiv_id: c for c in [*near, *naming]}
    vectors = {"need": 1, **{f"Kimi-VL {n}. b": 1 for n in range(1, 31)}}
    vectors |= {f"M{n}. uses Kimi-K3 {n}": 3 for n in range(1, 46)}
    await cache(pool, vectors)
    state: Any = {"request": request(need="need", names=["Kimi K3"]), "found": found, "library": []}
    async with ArxivClient() as arxiv:
        ctx = context(pool, arxiv)
        result: Any = await discover.prerank(state, Runtime(context=ctx))
        fewer: Any = {**state, "found": {c.arxiv_id: c for c in [*near, *naming[:2]]}}
        few: Any = await discover.prerank(fewer, Runtime(context=ctx))
        await ctx.gateway.aclose()
    assert len(result["shortlisted"]) == discover.MENTIONS
    assert set(result["shortlisted"]) <= {c.arxiv_id for c in naming}
    assert few["shortlisted"][:2] == [naming[0].arxiv_id, naming[1].arxiv_id]
    assert len(few["shortlisted"]) == discover.SHORTLIST


def test_a_name_matches_as_a_whole_word_whatever_its_spacing() -> None:
    pattern = discover.name_pattern(["Kimi K3", "SWE-bench"])
    assert pattern is not None
    hits = ["the 2.8T Kimi-K3 model", "(kimi k3)", "KimiK3.", "on SWE bench"]
    misses = ["Kimi K30", "Kimi K2.5", "akimi k3"]
    assert all(pattern.search(t) for t in hits) and not any(pattern.search(t) for t in misses)
    assert discover.name_pattern([]) is None and discover.name_pattern([" - "]) is None
