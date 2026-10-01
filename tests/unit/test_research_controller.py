"""Bounded behavioral scenarios for the one arXiv workflow. All adapters are offline."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.core.research_policy import ResearchPolicy
from src.core.trace import LocalTrace
from src.core.types import ChatResponse, Decomposition, SubQuestion, SubQResult, RouteDecision
from src.corpus.http import ResponseCache
from src.corpus.ingestion import IngestResult
from src.graph import builder
from src.graph.controller import control_research_node, fingerprint, recovery_actions
from src.graph.nodes import papers
from src.retrieval import paper_search as search
from tests.unit.test_paper_search import metadata, plan, assessment, wire, invoke


def state(**updates):
    return {
        "research_policy": ResearchPolicy().model_dump(),
        "paper_plan": plan().model_dump(),
        "paper_search": {
            "as_of": datetime.now(UTC).isoformat(),
            "queries": [],
            "assessments": 1,
            "coverage": "needs_more",
        },
        "completed_node": "assess_papers",
        "paper_candidates": [],
        "paper_results": [],
        **updates,
    }


def row(query, count):
    return {**query.model_dump(), "key": query.key(), "status": "ok", "results": count}


@pytest.mark.parametrize("question,limit", [
    ("找两三篇2026年MoE论文", 3), ("找2-3篇论文", 3), ("给我两到三篇论文", 3),
    ("Find two or three papers on routing", 3), ("Find 2 papers", 2),
    ("比较2024年的MoE方法和3B模型", None),
])
def test_explicit_result_quantity_is_not_confused_with_dates_or_model_sizes(question, limit):
    assert search.requested_result_limit(question) == limit


async def test_requested_quantity_caps_model_overproduction_after_freshness_sort(monkeypatch):
    from datetime import date

    records = [metadata(f"2601.0000{i}", published=date(2026, i, 1)) for i in range(1, 6)]
    proposal = plan("找两三篇2026年最新MoE论文")
    wire(monkeypatch, planner=AsyncMock(return_value=proposal), finder=AsyncMock(return_value=records))
    result = await invoke(builder.build_graph(), question="找两三篇2026年最新MoE论文")
    assert result["paper_plan"]["max_results"] == 3
    assert [p["paper_id"] for p in result["paper_results"]] == [
        "arxiv:2601.00005", "arxiv:2601.00004", "arxiv:2601.00003",
    ]
    assert len(result["paper_candidates"]) == 5  # Quantity limits delivery, not recall.


async def test_empty_title_is_relaxed_and_successful_source_is_selected(monkeypatch):
    q = search.Query(terms=["New-Architecture"], scope="title")
    proposal = plan("Find New Architecture").model_copy(update={"queries": [q]})
    finder = AsyncMock(side_effect=[[], [metadata()]])
    wire(monkeypatch, planner=AsyncMock(return_value=proposal), finder=finder)
    result = await invoke(builder.build_graph(), "Find New Architecture")
    assert result["status"] == "complete"
    assert [q["scope"] for q in result["paper_search"]["queries"]] == ["title", "title_abstract"]
    assert all(call.kwargs["fresh"] is False for call in finder.call_args_list)
    assert papers.assess_candidates.await_count == 1
    assert any(d["reason"] == "empty_title_relaxed" for d in result["paper_search"]["decisions"])


async def test_unchanged_candidates_do_not_spend_another_semantic_assessment(monkeypatch):
    selector = AsyncMock(
        side_effect=lambda p, c, q, **kw: assessment(c, [search.Query(terms=["new query"])])
    )
    wire(monkeypatch, selector=selector)
    result = await invoke(builder.build_graph())
    assert selector.await_count == 1
    assert result["status"] == "partial"
    assert result["paper_search"]["stop_reason"] == "search_exhausted"
    assert result["paper_results"]


def test_a_crowded_page_has_alternate_ranking_and_pagination_actions():
    q = search.Query(terms=["minimax"], scope="title", sort="recent")
    s = state()
    s["paper_search"]["queries"] = [row(q, 10)]
    actions = recovery_actions(s)
    assert [(a.sort, a.offset) for a, _ in actions] == [("relevance", 0), ("recent", 10)]
    s["paper_search"]["queries"] += [row(a, 10) for a, _ in actions]
    assert all(a.key() != q.key() for a, _ in recovery_actions(s))


async def test_results_survive_action_limit_without_being_called_complete():
    s = state(
        paper_results=[{"paper_id": "arxiv:2608.12345"}],
        pending_paper_queries=[search.Query(terms=["more"]).model_dump()],
    )
    s["research_policy"]["max_queries"] = 1
    s["paper_search"]["queries"] = [row(search.Query(terms=["prior"]), 1)]
    update = await control_research_node(s)
    assert update["next_node"] == "present_papers"
    assert update["paper_search"]["stop_reason"] == "search_limit"
    assert update["paper_search"]["coverage"] == "incomplete"


@pytest.mark.parametrize("too_many", [False, True])
async def test_clarification_stops_before_any_search_or_reading(monkeypatch, too_many):
    p = plan(deep=too_many, ids=[f"2608.0000{i}" for i in range(4)] if too_many else [])
    if not too_many:
        p.clarification = "你指的是模型架构还是 Agent 框架？"
    forbidden = wire(monkeypatch, planner=AsyncMock(return_value=p))
    result = await invoke(builder.build_graph())
    assert result["status"] == "needs_input"
    assert result["stop_reason"] == "clarification_required"
    papers.GLOBAL_ARXIV_CLIENT.search.assert_not_called()
    papers.GLOBAL_ARXIV_CLIENT.fetch_metadata.assert_not_called()
    forbidden.assert_not_called()


async def test_schema_repair_keeps_human_context_and_rejects_invented_constraints(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(search, "CACHE", ResponseCache(tmp_path / "cache"))
    proposal = plan("Find efficient agent memory").model_dump()
    proposal["hard_constraints"] = ["must be open source"]
    model = AsyncMock()
    model.chat.return_value = ChatResponse(
        content=json.dumps(proposal), model="fixture", finish_reason="stop"
    )
    monkeypatch.setattr(search, "get_chat_client", lambda _: model)
    with pytest.raises(search.PaperOutputError, match="invented_user_constraint"):
        await search.plan_search(
            "Need low latency memory",
            ["human: Budget is limited", "ai: We might prefer open source"],
        )
    assert model.chat.await_count == 2
    assert "Budget is limited" in str(model.chat.call_args_list[1])
    assert not list((tmp_path / "cache").glob("*.json"))


async def test_reason_is_preserved_in_visible_output_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "CACHE", ResponseCache(tmp_path / "cache"))
    records = [search.metadata_record(metadata())]
    output = assessment(records).model_copy(
        update={
            "reason": "The supplied pool lacks a recent model report; titles alone do not establish official authorship."
        }
    )
    model = AsyncMock()
    model.chat.return_value = ChatResponse(
        content=output.model_dump_json(), model="fixture", finish_reason="stop"
    )
    monkeypatch.setattr(search, "get_chat_client", lambda _: model)
    with LocalTrace(tmp_path / "trace.jsonl", run_id="decision", mode="offline").activate():
        selected = await search.assess_candidates(plan(), records, [])
    assert selected.reason == output.reason
    events = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert any(
        e["event"] == "paper_model_output" and "official authorship" in json.dumps(e)
        for e in events
    )
    assert model.chat.await_count == 1


async def test_service_judgment_failure_does_not_trigger_related_expansion():
    s = state(
        completed_node="retrieve",
        retrieval_paper_ids=["arxiv:2608.12345v1"],
        subq_results={
            0: SubQResult(
                subq_index=0,
                sufficient=False,
                route_decision=RouteDecision(),
                stop_reason="relevance_judgment_failed",
            )
        },
    )
    s["paper_plan"]["allow_related"] = True
    assert (await control_research_node(s))["next_node"] == "synthesize"
    s["subq_results"][0] = s["subq_results"][0].model_copy(
        update={"stop_reason": "no_new_evidence"}
    )
    assert (await control_research_node(s))["next_node"] == "expand_reading"
    s["related_expansion_attempted"] = True
    assert (await control_research_node(s))["next_node"] == "synthesize"


async def test_related_reading_keeps_original_task_and_completed_subquestions(monkeypatch):
    s = state(
        question="Compare mechanisms",
        related_search=True,
        retrieval_paper_ids=["arxiv:2608.00001v1"],
        reading_attempted_ids=["arxiv:2608.00001"],
        paper_results=[search.metadata_record(metadata())],
        decomposition=Decomposition(
            sub_questions=[SubQuestion(text="Mechanism"), SubQuestion(text="Evidence")]
        ),
        subq_results={
            0: SubQResult(subq_index=0, sufficient=True, route_decision=RouteDecision()),
            1: SubQResult(subq_index=1, sufficient=False, route_decision=RouteDecision()),
        },
    )
    s["paper_search"]["errors"] = []
    monkeypatch.setattr(
        papers,
        "ingest_paper_with_result",
        AsyncMock(return_value=IngestResult("arxiv:2608.12345v1", "skipped")),
    )
    update = await papers.read_papers_node(s)
    assert update["decomposition"] == s["decomposition"]
    assert list(update["subq_results"]) == [0]
    assert update["retrieval_paper_ids"] == ["arxiv:2608.00001v1", "arxiv:2608.12345v1"]


def test_fingerprint_changes_when_source_version_changes():
    a = search.metadata_record(metadata())
    assert fingerprint([a]) != fingerprint([{**a, "version": 2, "version_id": "2608.12345v2"}])


def test_low_recall_compound_relaxes_without_inventing_a_domain_keyword():
    q = search.Query(terms=["ProjectName", "special attention"], sort="relevance")
    s = state()
    s["paper_search"]["queries"] = [row(q, 2)]
    relaxed, reason = recovery_actions(s)[0]
    assert relaxed.terms == q.terms
    assert relaxed.match == "words"
    assert relaxed.api_query() != q.api_query()
    assert reason == "compound_phrase_relaxed"


def test_compound_query_retains_all_words_and_does_not_require_phrase_adjacency():
    q = search.Query(terms=["parallel tool calling", "agent"], match="words")
    assert q.api_query() == ' AND '.join(f'(ti:"{w}" OR abs:"{w}")' for w in ['parallel', 'tool', 'calling', 'agent'])
    assert q.key() != q.model_copy(update={"match": "phrase"}).key()


def test_initial_probe_preserves_precise_queries_and_hard_constraints():
    task = plan().model_copy(update={
        "queries": [search.Query(terms=["parallel tool calling", "LLM agent"])],
        "hard_constraints": ["no training"],
    })
    queries = search.acquisition_queries(task)
    assert queries[0] == task.queries[0]
    assert queries[1].terms == ["parallel tool calling"] and queries[1].match == "words"
    assert task.hard_constraints == ["no training"]
    exact = task.model_copy(update={"paper_ids": ["arxiv:2312.04511"]})
    assert search.acquisition_queries(exact) == exact.queries
    named = task.model_copy(update={"queries": [search.Query(terms=["Specific Paper Title"], scope="title")]})
    assert search.acquisition_queries(named) == named.queries


async def test_model_proposals_cannot_starve_program_recovery():
    q = search.Query(terms=["subject"], sort="recent")
    s = state(pending_paper_queries=[search.Query(terms=["other"]).model_dump()])
    s["paper_search"]["queries"] = [row(q, 10)]
    update = await control_research_node(s)
    actions = [search.Query.model_validate(q) for q in update["pending_paper_queries"]]
    assert actions[0].terms == ["subject"] and actions[0].sort == "relevance"
    assert actions[1].terms == ["other"]


async def test_rejected_sources_are_not_rejudged_until_their_content_changes(monkeypatch):
    rejected = search.metadata_record(metadata("2608.00001", title="A mathematical namesake"))
    selected = search.metadata_record(metadata())
    s = state(question="Find memory papers", paper_candidates=[rejected, selected])
    s["paper_search"].update(
        errors=[], prefer_recent=False, date_mode="none", date_from=None, date_to="2026-09-16"
    )

    async def assess(p, candidates, queries, **kw):
        result = assessment([p for p in candidates if p["paper_id"] == selected["paper_id"]])
        result.rejected = {
            p["paper_id"]: "namesake" for p in candidates if p["paper_id"] != selected["paper_id"]
        }
        return result

    selector = AsyncMock(side_effect=assess)
    monkeypatch.setattr(papers, "assess_candidates", selector)
    s.update(await papers.assess_papers_node(s))
    s["paper_candidates"].append(search.metadata_record(metadata("2608.00002")))
    s.update(await papers.assess_papers_node(s))
    assert rejected["paper_id"] not in [p["paper_id"] for p in selector.call_args.args[1]]
    assert s["paper_search"]["reused_exclusions"] == 1
    s["paper_candidates"][0] = {
        **rejected,
        "abstract": "Changed source evidence in a newer revision.",
    }
    s.update(await papers.assess_papers_node(s))
    assert rejected["paper_id"] in [p["paper_id"] for p in selector.call_args.args[1]]


@pytest.mark.parametrize(
    ("question", "should_read"),
    [
        ("不要只看摘要，请阅读原文解释机制。", True),
        ("只看这篇的摘要，不要读取全文。", False),
        ("Do not read only the abstract; explain the original experiments.", True),
    ],
)
async def test_full_text_opt_out_cannot_misread_a_request_for_original_evidence(
    monkeypatch, question, should_read
):
    monkeypatch.setattr(
        papers, "plan_search", AsyncMock(return_value=plan(question, deep=True, ids=["2608.12345"]))
    )
    result = await papers.plan_papers_node(state(question=question, messages=[]))
    assert result["paper_plan"]["read_full_text"] == should_read


async def test_related_search_without_new_originals_delivers_existing_evidence(monkeypatch):
    record = search.metadata_record(metadata())
    prior = SubQResult(
        subq_index=0,
        sufficient=False,
        route_decision=RouteDecision(),
        missing_aspects=["Unanswered detail"],
    )
    s = state(
        question="Compare mechanisms",
        related_search=True,
        retrieval_paper_ids=["arxiv:2608.12345v1"],
        reading_attempted_ids=["arxiv:2608.12345"],
        paper_results=[record],
        decomposition=Decomposition(sub_questions=[SubQuestion(text="Compare mechanisms")]),
        subq_results={0: prior},
    )
    s["paper_search"]["errors"] = []
    ingest = AsyncMock(
        side_effect=AssertionError("Already acquired source must not be acquired again")
    )
    monkeypatch.setattr(papers, "ingest_paper_with_result", ingest)
    s.update(await papers.read_papers_node(s))
    s["completed_node"] = "read_papers"
    assert s["reading_added_count"] == 0 and s["subq_results"] == {0: prior}
    decision = await control_research_node(s)
    assert decision["next_node"] == "synthesize"
    assert decision["paper_search"]["decisions"][-1]["reason"] == "no_new_original_evidence"
