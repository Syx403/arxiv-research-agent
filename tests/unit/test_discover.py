"""Discovery's deterministic parts: the tool budget, what the researcher sees, ranking, and how a
tool turn is sent back to DeepSeek. The researcher and screen calls are checked live."""

from datetime import date
from typing import Any

from ara.arxiv.client import Metadata
from ara.graph import discover
from ara.graph.state import Constraint, PaperCard, ResearchRequest
from ara.llm.prompt import Block, Instructions, Prompt, ToolCall, deepseek_messages


def meta(n: int, published: date = date(2024, 1, 1)) -> Metadata:
    return Metadata(f"2401.0000{n}", 2, f"Paper {n}", "A long abstract. " * 40, published)


def request(**fields: Any) -> ResearchRequest:
    base: dict[str, Any] = {
        "intent": "discover",
        "clarification": None,
        "need": "need",
        "question": None,
        "paper_ids": [],
        "listed": [],
        "count": None,
        "constraints": [],
        "published_after": None,
        "published_before": None,
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


def card(n: int, similarity: float) -> PaperCard:
    m = meta(n)
    return discover.card(m).model_copy(update={"similarity": similarity})


def test_rank_keeps_relevant_papers_that_break_no_quoted_constraint() -> None:
    constraint = Constraint(quote="no fine-tuning", meaning="no training")
    shortlist = [card(1, 0.9), card(2, 0.8), card(3, 0.7), card(4, 0.95)]
    judged = [
        discover.Judgement(id="2401.00001", relevance=2, reason="", violated=[]),
        discover.Judgement(id="2401.00002", relevance=3, reason="", violated=[]),
        discover.Judgement(id="2401.00003", relevance=3, reason="", violated=["no fine-tuning"]),
        discover.Judgement(id="2401.00004", relevance=1, reason="", violated=["made up"]),
    ]
    state: Any = {
        "request": request(constraints=[constraint], count=5),
        "shortlist": shortlist,
        "judged": judged,
    }
    papers = discover.rank(state)["papers"]
    assert [p.arxiv_id for p in papers] == ["2401.00002", "2401.00001"]


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
