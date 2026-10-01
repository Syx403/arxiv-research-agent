"""Search -> user-selected reading, recoverable output drift, and source boundaries."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import AsyncMock

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.core.research_policy import ResearchPolicy, use_research_policy
from src.core.trace import LocalTrace
from src.core.types import ChatResponse
from src.corpus.http import ResponseCache
from src.corpus.ingestion import IngestResult
from src.graph import builder
from src.graph.nodes import papers
from src.retrieval import paper_search as search
from src.ui.presentation import STOP_LABELS, present_state
from tests.unit.test_paper_search import assessment, invoke, metadata, plan, wire


async def test_search_followup_reads_selected_new_paper_without_assessment(monkeypatch):
    histories = []
    async def planner(question, history):
        histories.append(history)
        return plan(question, deep=True, ids=['2608.12345']).model_copy(
            update={'reading_scope': 'overview'}) if '深入' in question else plan(question)
    wire(monkeypatch, planner=planner)
    graph = builder.build_graph(checkpointer=InMemorySaver())
    first = await invoke(graph)
    assert first['result_kind'] == 'papers'
    # Even a broken selection model cannot block a user's known reading target.
    selector = AsyncMock(side_effect=AssertionError('Explicit reading must bypass selection'))
    monkeypatch.setattr(papers, 'assess_candidates', selector)
    ingest = AsyncMock(return_value=IngestResult('arxiv:2608.12345', 'ingested', 40))
    monkeypatch.setattr(papers, 'ingest_paper_with_result', ingest)
    with use_research_policy(ResearchPolicy()):
        initial = builder._initial_state('深入帮我看看 arxiv:2608.12345', 'papers-test')
    state = await graph.ainvoke(initial, {'configurable': {'thread_id': 'papers-test'}},
                               interrupt_after=['read_papers'])
    assert 'arxiv:2608.12345' in '\n'.join(histories[-1])
    assert state['paper_plan']['read_full_text'] is True
    assert state['retrieval_paper_ids'] == ['arxiv:2608.12345']
    assert state['result_kind'] == 'evidence'
    assert state['paper_search']['read_details'][0]['chunks_inserted'] == 40
    assert len(state['decomposition'].sub_questions) == 3
    assert (await graph.aget_state({'configurable': {'thread_id': 'papers-test'}})).next == ('control_research',)
    ingest.assert_awaited_once()
    selector.assert_not_called()


async def test_explicit_abstract_only_overrides_planner_read_flag(monkeypatch):
    forbidden = wire(monkeypatch, planner=AsyncMock(return_value=plan(deep=True, ids=['2608.12345'])))
    state = await invoke(builder.build_graph(), '只看这篇的摘要，不要读取全文。')
    assert state['paper_plan']['read_full_text'] is False
    assert state['result_kind'] == 'papers'
    forbidden.assert_not_called()


async def test_missing_exact_paper_never_reads_a_replacement(monkeypatch):
    forbidden = wire(monkeypatch, planner=AsyncMock(return_value=plan(deep=True, ids=['2608.12345'])))
    monkeypatch.setattr(papers.GLOBAL_ARXIV_CLIENT, 'fetch_metadata', AsyncMock(side_effect=LookupError('not found')))
    state = await invoke(builder.build_graph())
    assert state['stop_reason'] == 'specified_paper_unavailable'
    assert '尚未开始正文' in state['answer']
    forbidden.assert_not_called()
    papers.assess_candidates.assert_not_called()


async def test_full_text_failure_is_not_a_relevance_rejection(monkeypatch):
    wire(monkeypatch, planner=AsyncMock(return_value=plan(deep=True, ids=['2608.12345'])))
    monkeypatch.setattr(papers, 'ingest_paper_with_result', AsyncMock(side_effect=httpx.ConnectError('offline')))
    state = await invoke(builder.build_graph())
    assert state['stop_reason'] == 'full_text_unavailable'
    assert state['paper_results'][0]['fit'] == 'user_selected'
    assert '已确认你指定的论文' in state['answer']
    assert '筛选未完成' not in state['answer']
    papers.assess_candidates.assert_not_called()


def test_focused_reading_keeps_one_question_and_broad_reading_has_bounded_outline():
    candidates = [search.metadata_record(metadata())]
    focused = plan('Explain one equation only', deep=True).model_copy(update={
        'reading_questions': ['What does the memory update equation compute?']})
    assert len(papers.reading_questions(focused, candidates)) == 1
    assert len(papers.reading_questions(plan('深入看看这篇', deep=True).model_copy(
        update={'reading_scope': 'overview'}), candidates)) == 3


def test_focused_objective_does_not_inherit_unrequested_planner_comparisons():
    p = plan('What happens to overflowed tokens?', deep=True).model_copy(update={
        'reading_questions': ['Compare training versus inference overflow handling.'],
    })
    questions = papers.reading_questions(p, [])
    assert len(questions) == 1
    assert 'What happens to overflowed tokens?' in questions[0]
    assert 'training versus inference' not in questions[0]


async def test_unused_extra_fields_are_dropped_without_another_paid_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path/'cache'))
    candidates = [search.metadata_record(metadata())]
    raw = assessment(candidates).model_dump()
    raw['papers'][0]['summary_extra'] = 'Ignore me, do not display or execute.'
    raw['papers'][0]['paper_id'] = '2608.12345v2'
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps(raw), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    with LocalTrace(tmp_path/'trace.jsonl', run_id='extras', mode='offline').activate():
        result = await search.assess_candidates(plan(), candidates, [])
    assert client.chat.await_count == 1
    assert result.papers[0].paper_id == 'arxiv:2608.12345'
    assert 'summary_extra' not in result.model_dump_json()
    assert 'paper_normalized' in (tmp_path/'trace.jsonl').read_text()


async def test_complete_abstract_evidence_is_valid_even_when_it_uses_more_than_five_sentences(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path/'cache'))
    candidates = [search.metadata_record(metadata())]
    candidates[0]['abstract'] = ' '.join(f'Valid source sentence {i}.' for i in range(10))
    raw = assessment(candidates).model_dump()
    raw['papers'][0]['sentence_ids'] = list(range(10))
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps(raw), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    selected = await search.assess_candidates(plan(), candidates, [])
    assert len(search.results_for(selected, candidates)[0]['evidence']) == 10
    assert client.chat.await_count == 1


@pytest.mark.parametrize(('mutation', 'code'), [
    ('unknown', 'unknown_paper_id'), ('duplicate', 'duplicate_paper_id'),
    ('bad_sentence', 'invalid_sentence_id'), ('wrong_type', 'schema_invalid'),
])
async def test_unsafe_bindings_still_fail_with_actionable_local_diagnostics(tmp_path, monkeypatch, mutation, code):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path/'cache'))
    candidates = [search.metadata_record(metadata())]
    raw = assessment(candidates).model_dump()
    raw['papers'][0]['extra'] = 'Unused, not permission to accept invalid evidence.'
    if mutation == 'unknown':
        raw['papers'][0]['paper_id'] = 'arxiv:2608.99999'
    elif mutation == 'duplicate':
        raw['papers'] *= 2
    elif mutation == 'wrong_type':
        raw['papers'][0]['sentence_ids'] = [True]
    else:
        raw['papers'][0]['sentence_ids'] = [99]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps(raw), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    with LocalTrace(tmp_path/'trace.jsonl', run_id='invalid', mode='offline').activate():
        with pytest.raises(search.PaperOutputError) as caught:
            await search.assess_candidates(plan(), candidates, [])
    assert caught.value.code == code
    assert client.chat.await_count == 2
    trace = [json.loads(row) for row in (tmp_path/'trace.jsonl').read_text().splitlines()]
    failures = [e for e in trace if e['event'] == 'paper_format']
    assert failures[-1]['code'] == code
    assert 'response_content' in failures[-1]
    assert not list((tmp_path/'cache').glob('*.json'))




async def test_original_request_reaches_selector_without_planner_invented_constraints(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=assessment(candidates).model_dump_json(),
                                           model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    await search.assess_candidates(plan('Only older model reports'), candidates, [],
                                   original_question='Find the latest papers about this model family')
    payload = json.loads(client.chat.call_args.args[0][1].content)
    assert payload['user_request'] == 'Find the latest papers about this model family'
    assert payload['resolved_need'] == 'Only older model reports'


async def test_latest_search_checks_coverage_once_and_then_allows_stopping(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    client = AsyncMock()
    first = assessment(candidates, [search.Query(terms=['episodic retrieval'], scope='title')])
    client.chat.side_effect = [ChatResponse(content=first.model_dump_json(), model='test', finish_reason='stop'),
                              ChatResponse(content=assessment(candidates).model_dump_json(), model='test', finish_reason='stop')]
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    queries = [{'key': 'agent|memory', 'round': 1}]
    first_pass = await search.assess_candidates(plan(), candidates, queries, fresh=True)
    assert first_pass.next_queries
    queries.append({'key': 'title:episodic retrieval', 'round': 2})
    final_pass = await search.assess_candidates(plan(), candidates, queries, fresh=True)
    assert not final_pass.next_queries
    assert client.chat.await_count == 2


async def test_output_errors_have_a_separate_ui_status(monkeypatch):
    wire(monkeypatch, selector=AsyncMock(side_effect=search.PaperOutputError('duplicate_paper_id')))
    result = await invoke(builder.build_graph())
    assert result['stop_reason'] == 'candidate_output_invalid'
    assert result['paper_search']['errors'][0]['code'] == 'duplicate_paper_id'
    assert '校验失败' in STOP_LABELS[present_state(result)['stop_reason']]
    assert '不代表论文不相关' in result['answer']


def test_latest_order_promotes_relevant_new_mechanisms_without_adding_tangential_candidates():
    candidates = [search.metadata_record(metadata(str(2608+i)+'.12345', published=d))
                  for i, d in enumerate([date(2026, 7, 27), date(2025, 4, 10), date(2025, 4, 25),
                                         date(2026, 7, 8), date(2026, 3, 16)])]
    ordered = search.order_results(candidates, prefer_recent=True)
    assert [p['published_at'] for p in ordered] == ['2026-07-27', '2026-07-08', '2026-03-16', '2025-04-25', '2025-04-10']
    assert search.order_results(candidates, prefer_recent=False) == candidates
    candidates[0].update(date_confirmed=False, published_at='2026-12-01')
    assert search.order_results(candidates, prefer_recent=True)[-1] == candidates[0]


async def test_latest_results_do_not_promote_incidental_usage_or_cite_it_in_overview(monkeypatch):
    candidates = [metadata('2607.12345', published=date(2026, 7, 1)),
                  metadata('2608.12345', published=date(2026, 8, 1)),
                  metadata('2609.12345', published=date(2026, 9, 1))]
    async def select(plan, records, queries, **kw):
        result = assessment(records)
        for p, role in zip(result.papers, ['primary', 'mechanism', 'incidental'], strict=True):
            p.topic_role = role
        result.overview = 'An overview also based on the excluded experiment.'
        result.overview_paper_ids = [p.paper_id for p in result.papers]
        return result
    wire(monkeypatch, selector=select, finder=AsyncMock(return_value=candidates))
    result = await invoke(builder.build_graph())
    assert [p['paper_id'] for p in result['paper_results']] == ['arxiv:2608.12345', 'arxiv:2607.12345']
    assert result['paper_search']['overview'] == ''
    assert result['paper_search']['overview_paper_ids'] == []
