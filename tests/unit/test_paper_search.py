from __future__ import annotations

import asyncio
import json
from datetime import date
from unittest.mock import AsyncMock

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from src.core.research_policy import ResearchPolicy, use_research_policy
from src.core.types import ChatResponse
from src.corpus.arxiv_client import ArxivClient, PaperMetadata
from src.corpus.http import Pacer, ResponseCache
from src.graph import builder
from src.graph.nodes import papers
from src.retrieval import paper_search as search
from src.ui.presentation import present_state


def metadata(identifier='2608.12345', *, title='Agent memory', published=date(2026, 8, 20), unconfirmed=False):
    return PaperMetadata(identifier, 'arxiv:' + identifier, title, ['Author'],
                         'We introduce a retrieval memory for agents. It stores episodes without training.',
                         published, f'https://arxiv.org/abs/{identifier}', None,
                         {'publication_date_unconfirmed': unconfirmed, 'version': 1, 'version_id': identifier + 'v1'})


def plan(question='Find recent agent memory papers', *, deep=False, ids=()):
    return search.Plan(question=question, criteria=['Agent memory; avoid training if possible'],
                       queries=[search.Query(terms=['agent', 'memory']), search.Query(terms=['episodic memory'])],
                       paper_ids=list(ids), read_full_text=deep)


def assessment(records, next_queries=()):
    return search.Assessment(papers=[search.Choice(paper_id=p['paper_id'], topic_role='primary', fit='direct',
                             summary='Retrieval memory for agents.', why_relevant='No training is needed.',
                             sentence_ids=[0, 1]) for p in records[:5]],
                             next_queries=list(next_queries), reason='Useful abstract evidence.',
                             coverage='needs_more' if next_queries or not records else 'sufficient')


def wire(monkeypatch, *, planner=None, selector=None, finder=None):
    monkeypatch.setattr(papers, 'plan_search', planner or AsyncMock(return_value=plan()))
    monkeypatch.setattr(papers, 'assess_candidates', selector or AsyncMock(side_effect=lambda p, c, q, **kw: assessment(c)))
    monkeypatch.setattr(papers.GLOBAL_ARXIV_CLIENT, 'search', finder or AsyncMock(return_value=[metadata()]))
    monkeypatch.setattr(papers.GLOBAL_ARXIV_CLIENT, 'fetch_metadata', AsyncMock(side_effect=lambda pid, **kw: metadata(pid.removeprefix('arxiv:').split('v')[0])))
    # The paper-list path must never quietly enter paid PDF/embedding/retrieval work.
    forbidden = AsyncMock(side_effect=AssertionError('Unexpected full-text/embedding path'))
    monkeypatch.setattr(papers, 'ingest_paper_with_result', forbidden)
    monkeypatch.setattr(builder, 'retrieve_node', forbidden)
    return forbidden


async def invoke(graph, question='Find recent agent memory papers', thread='papers-test', timeout=90):
    with use_research_policy(ResearchPolicy(timeout_s=timeout)):
        state = builder._initial_state(question, thread, skip_reflection=True)
    return await graph.ainvoke(state, {'configurable': {'thread_id': thread}})


async def test_graph_deduplicates_returns_linked_abstracts_without_embeddings(monkeypatch):
    forbidden = wire(monkeypatch)
    result = await invoke(builder.build_graph())
    assert result['status'] == 'complete'
    assert result['paper_search']['candidate_count'] == 1
    assert len(result['paper_results']) == 1
    paper = result['paper_results'][0]
    assert paper['url'] == 'https://arxiv.org/abs/2608.12345'
    assert paper['evidence'] == search.sentences(metadata().abstract)
    assert result['verification'] is None
    assert present_state(result)['sources'] == []
    assert present_state(result)['paper_results'][0]['evidence_level'] == 'abstract'
    forbidden.assert_not_called()


async def test_one_or_two_papers_does_not_force_a_second_weaker_result(monkeypatch):
    wire(monkeypatch)
    result = await invoke(builder.build_graph(), '找一两篇工具编排的论文，摘要介绍即可')
    assert result['paper_plan']['min_results'] == 1
    assert result['paper_plan']['max_results'] == 2
    assert result['status'] == 'complete'
    assert len(result['paper_results']) == 1
    assert result['paper_search']['assessments'] == 1


async def test_adaptive_search_runs_once_merges_old_candidates_and_does_not_repeat(monkeypatch):
    seen = []
    async def select(p, candidates, queries, **kw):
        seen.append([c['paper_id'] for c in candidates])
        return assessment(candidates, [search.Query(terms=['memory', 'agent']),
                                      search.Query(terms=['long term memory'])])
    finder = AsyncMock(side_effect=lambda query, **kw: [metadata('2608.22222')] if 'long term' in query else [metadata()])
    wire(monkeypatch, selector=select, finder=finder)
    result = await invoke(builder.build_graph())
    assert 3 <= finder.await_count <= 8
    assert result['paper_search']['rounds'] >= 2
    assert len(seen[1]) == 2
    assert len(result['paper_results']) == 2


async def test_empty_feed_gets_one_semantic_requery_not_a_false_provider_error(monkeypatch):
    async def select(p, candidates, queries, **kw):
        return assessment(candidates, [] if candidates else [search.Query(terms=['episodic agent'])])
    finder = AsyncMock(side_effect=[[], []] + [[metadata()]] * 8)
    wire(monkeypatch, selector=select, finder=finder)
    result = await invoke(builder.build_graph())
    assert result['status'] == 'complete'
    assert result['paper_search']['rounds'] == 2
    assert not result['paper_search']['errors']


async def test_requery_cap_returns_selected_papers_and_explains_remaining_gap(monkeypatch):
    async def select(p, candidates, queries, **kw):
        return assessment(candidates, [search.Query(terms=[f'memory approach {len(queries)}'])])
    wire(monkeypatch, selector=select)
    result = await invoke(builder.build_graph())
    assert len(result['paper_search']['queries']) <= 8
    assert result['paper_search']['assessments'] == 1
    assert result['status'] == 'partial'
    assert '尚未充分覆盖需求' in result['answer']
    assert result['paper_results']


async def test_completed_query_survives_another_query_timeout(monkeypatch):
    async def find(query, **kw):
        if 'episodic' in query:
            await asyncio.sleep(1)
        return [metadata()]
    wire(monkeypatch, finder=find)
    result = await invoke(builder.build_graph(), timeout=.02)
    assert len(result['paper_results']) == 1
    assert result['status'] == 'partial'
    assert result['paper_search']['errors'][0]['stage'] == 'search'


async def test_model_failure_preserves_candidates_without_claiming_relevance(monkeypatch):
    wire(monkeypatch, selector=AsyncMock(side_effect=ValueError('invalid selection')))
    result = await invoke(builder.build_graph())
    assert result['stop_reason'] == 'candidate_assessment_failed'
    assert result['paper_results'][0]['fit'] == 'unassessed'
    assert result['paper_results'][0]['summary'] == ''


async def test_total_transport_failure_does_not_pay_for_semantic_refinement(monkeypatch):
    selector = AsyncMock()
    wire(monkeypatch, selector=selector, finder=AsyncMock(side_effect=httpx.ConnectError('offline')))
    result = await invoke(builder.build_graph())
    assert result['stop_reason'] == 'external_search_unavailable'
    assert result['paper_search']['rounds'] == 1
    selector.assert_not_called()


@pytest.mark.parametrize('failure', ['connect', '429', 'empty'])
async def test_paper_flow_network_api_and_empty_retries_share_single_attempt(tmp_path, failure):
    from src.core.types import Message
    from src.llm.budget import RequestBudget
    from src.llm.client import _RoutedChatClient
    from src.llm.errors import EmptyProviderResponseError, LLMError
    from src.llm.providers.deepseek import DeepSeekChatClient
    from src.llm.registry import get_route
    from src.llm.retry import single_provider_attempt
    calls = []
    async def handler(request):
        calls.append(request)
        if failure == 'connect':
            raise httpx.ConnectError('offline')
        if failure == '429':
            return httpx.Response(429, headers={'Retry-After': '0'})
        return httpx.Response(200, json={'choices': [{'message': {'content': ''}, 'finish_reason': 'stop'}]})
    budget = RequestBudget(1, path=tmp_path/'budget.json')
    async with httpx.AsyncClient(base_url='https://mock.invalid/v1', transport=httpx.MockTransport(handler)) as http:
        client = _RoutedChatClient('main', get_route('main'), DeepSeekChatClient(api_key='test-only', http_client=http))
        with budget.activate(), single_provider_attempt(), pytest.raises((httpx.ConnectError, LLMError, EmptyProviderResponseError)):
            await client.chat([Message(role='user', content='Paper selection')], max_tokens=100)
    assert len(calls) == len(budget.snapshot()['requests']) == 1


async def test_explicit_full_text_failure_keeps_papers(monkeypatch):
    forbidden = wire(monkeypatch, planner=AsyncMock(return_value=plan(deep=True, ids=['2608.12345'])))
    monkeypatch.setattr(papers, 'ingest_paper_with_result', AsyncMock(side_effect=httpx.ConnectError('offline')))
    result = await invoke(builder.build_graph())
    assert result['stop_reason'] == 'full_text_unavailable'
    assert result['paper_results'][0]['read_status'] == 'read_failed'
    papers.GLOBAL_ARXIV_CLIENT.search.assert_not_called()
    forbidden.assert_not_called()


async def test_full_text_acquisition_passes_completion_contract_to_finalizer(monkeypatch):
    from src.corpus.ingestion import IngestResult
    from src.graph.nodes.finalize import finalize_node
    from src.core.types import Citation, CitationVerdict, EvidenceChunk, VerificationReport
    rows = [search.metadata_record(metadata())]
    selected = search.results_for(assessment(rows), rows)
    monkeypatch.setattr(papers, 'ingest_paper_with_result', AsyncMock(return_value=IngestResult('arxiv:2608.12345', 'skipped')))
    state = {'question': 'Read the original', 'research_question': 'Agent memory', 'paper_results': selected,
             'research_policy': ResearchPolicy().model_dump(), 'paper_plan': plan(deep=True, ids=['2608.12345']).model_dump(), 'paper_search': {'errors': []}}
    update = await papers.read_papers_node(state)
    assert update['external_discovery']['status'] == 'complete'
    draft = 'It uses retrieval memory [arxiv:2608.12345#1].'
    citation = Citation(paper_id='arxiv:2608.12345', chunk_id=1, claim_text=draft, claim_span=(0, len(draft)))
    report = VerificationReport(passed=True, verdicts=[CitationVerdict(citation=citation, supports=True, rationale='supported')])
    final = await finalize_node({**state, **update, 'answer': draft, 'verification': report,
                                 'evidence_lookup': {1: EvidenceChunk(paper_id=citation.paper_id, chunk_id=1, text=draft)}})
    assert final['status'] == 'complete' and final['stop_reason'] == 'answer_verified'


async def test_synthesis_preserves_original_answer_constraints(monkeypatch):
    from src.graph.nodes import synthesize
    from tests.integration.test_resume import _hit
    seen = []
    async def generate(question, evidence, **kwargs):
        seen.append(question)
        return 'Mechanism [arxiv:2210.03629#1].', 'stop'
    monkeypatch.setattr(synthesize, '_generate_answer', generate)
    await synthesize.synthesize_node({'question': '只用两三句话，解释 ReAct', 'research_question': 'Explain ReAct',
                                     'paper_plan': {'question': 'Explain ReAct'}, 'evidence': [_hit()]})
    assert '只用两三句话' in seen[0]


async def test_followup_preserves_paper_ids_but_resets_search_state(monkeypatch):
    histories = []
    async def planner(question, history):
        histories.append(history)
        return plan(question)
    wire(monkeypatch, planner=planner)
    graph = builder.build_graph(checkpointer=InMemorySaver())
    await invoke(graph)
    follow = await invoke(graph, 'Which of those avoids training?')
    assert 'arxiv:2608.12345' in '\n'.join(histories[1])
    assert follow['paper_search']['rounds'] == 1
    assert len(follow['paper_search']['queries']) == 2
    await invoke(graph, thread='another-session')
    assert histories[-1] == []


async def test_explicit_dates_reach_search_and_revision_does_not_pass_first_publication_filter(monkeypatch):
    wire(monkeypatch, planner=AsyncMock(return_value=plan('2026 年的 Agent memory 论文')),
         finder=AsyncMock(return_value=[metadata(unconfirmed=True)]))
    monkeypatch.setattr(papers.GLOBAL_ARXIV_CLIENT, 'fetch_metadata', AsyncMock(return_value=metadata(published=date(2024, 1, 1))))
    result = await invoke(builder.build_graph())
    assert papers.GLOBAL_ARXIV_CLIENT.search.call_args.kwargs['date_from'] == '2026-01-01'
    assert papers.GLOBAL_ARXIV_CLIENT.search.call_args.kwargs['latest'] is False
    assert result['paper_results'] == []
    assert result['paper_search']['outside_date_count'] == 1


async def test_unconfirmed_date_never_claims_to_satisfy_date_constraint(monkeypatch):
    wire(monkeypatch, planner=AsyncMock(return_value=plan('2026 年的 Agent memory 论文')),
         finder=AsyncMock(return_value=[metadata(unconfirmed=True)]))
    monkeypatch.setattr(papers.GLOBAL_ARXIV_CLIENT, 'fetch_metadata', AsyncMock(side_effect=httpx.ConnectError('offline')))
    result = await invoke(builder.build_graph())
    assert result['paper_results'][0]['fit'] == 'partial'
    assert '尚未确认' in result['paper_results'][0]['uncertainty']


async def test_structured_cache_reuses_success_but_rebinds_to_question_and_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=assessment(candidates).model_dump_json(), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    first = await search.assess_candidates(plan(), candidates, [])
    assert await search.assess_candidates(plan(), candidates, []) == first
    assert client.chat.await_count == 1
    await search.assess_candidates(plan('Prioritize training-free methods'), candidates, [])
    assert client.chat.await_count == 2
    candidates[0]['abstract'] += ' Additional source context.'
    await search.assess_candidates(plan(), candidates, [])
    assert client.chat.await_count == 3


async def test_truncated_selection_restarts_once_with_bounded_shortlist(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    selected = assessment(candidates)
    client = AsyncMock()
    client.chat.side_effect = [
        ChatResponse(content='', model='test', finish_reason='length'),
        ChatResponse(content=selected.model_dump_json(), model='test', finish_reason='stop'),
    ]
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    task = plan().model_copy(update={'max_results': 2})
    out = await search.assess_candidates(task, candidates, [])
    assert out == selected
    initial, repair = client.chat.call_args_list
    assert json.loads(initial.args[0][1].content)['output_limits']['total_paper_records'] == 2
    assert 'Do not continue a truncated response' in repair.args[0][-1].content
    assert 'output_truncated' in repair.args[0][-1].content
    assert repair.kwargs['thinking'] is False
    assert client.chat.await_count == 2


@pytest.mark.parametrize('status,fit,coverage', [
    ('supported', 'direct', 'sufficient'),
    ('unknown', 'partial', 'needs_more'),
    ('contradicted', None, 'needs_more'),
])
async def test_declared_hard_constraint_controls_fit_and_completion(tmp_path, monkeypatch, status, fit, coverage):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    candidates[0]['abstract'] = ('The method runs on an unchanged black-box API. '
                                 'Optional fine-tuning is evaluated as a separate branch.')
    selected = assessment(candidates)
    selected.overview, selected.overview_paper_ids = 'All conditions satisfied.', [candidates[0]['paper_id']]
    selected.papers[0].why_relevant = 'Guaranteed to work without training.'
    selected.papers[0].constraint_checks = [search.ConstraintCheck(
        constraint_id=0, status=status, sentence_ids=[] if status == 'unknown' else [0])]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=selected.model_dump_json(), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    task = plan().model_copy(update={'hard_constraints': ['no training']})
    out = await search.assess_candidates(task, candidates, [])
    assert out.coverage == coverage
    if fit:
        assert out.papers[0].fit == fit
        if status == 'unknown':
            assert 'Guaranteed' not in out.papers[0].why_relevant
            assert not out.overview
    else:
        assert not out.papers and out.rejected[candidates[0]['paper_id']] == 'constraint_violation'
    # Reused cached interpretation must still go through the same programmatic gate.
    assert await search.assess_candidates(task, candidates, []) == out
    assert client.chat.await_count == 1


@pytest.mark.parametrize('checks', [[],
    [{'constraint_id': 1, 'status': 'supported', 'sentence_ids': [0]}],
    [{'constraint_id': 0, 'status': 'supported', 'sentence_ids': []}],
    [{'constraint_id': 0, 'status': 'supported', 'sentence_ids': [999]}],
    [{'constraint_id': 0, 'status': 'unknown', 'sentence_ids': [True]}],
])
async def test_constraint_checks_cannot_be_missing_or_fabricated(tmp_path, monkeypatch, checks):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    raw = assessment(candidates).model_dump()
    raw['papers'][0]['constraint_checks'] = checks
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps(raw), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    task = plan().model_copy(update={'hard_constraints': ['no training']})
    with pytest.raises(search.PaperOutputError):
        await search.assess_candidates(task, candidates, [])
    assert not list(tmp_path.glob('*.json'))


@pytest.mark.parametrize('fit,coverage', [('partial', 'needs_more'), ('direct', 'sufficient')])
async def test_direct_match_caveat_is_checked_against_required_mechanism(tmp_path, monkeypatch, fit, coverage):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    selected = assessment(candidates)
    selected.papers[0].uncertainty = 'Required dependency handling is not established.'
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({'papers': [{
        'paper_id': candidates[0]['paper_id'], 'fit': fit, 'reason': 'Fixture required-versus-optional judgment.',
        'contribution_sentence_ids': [0, 1],
        'replacement': {'summary': 'Retrieval memory for agents.', 'sentence_ids': [0, 1]},
    }], 'next_queries': [{'terms': ['dependency scheduling'], 'match': 'words'}] if fit == 'partial' else []}),
        model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    out = await search.review_qualified_matches(selected, plan(), candidates)
    assert out.papers[0].fit == fit and out.coverage == coverage
    assert bool(out.next_queries) is (fit == 'partial')
    assert selected.papers[0].fit == 'direct'  # Preserve the prior decision for diagnosis/cache.


async def test_unqualified_or_selected_full_text_papers_do_not_pay_for_caveat_review(monkeypatch):
    candidates = [search.metadata_record(metadata())]
    selected = assessment(candidates)
    client = AsyncMock()
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    assert await search.review_qualified_matches(selected, plan(), candidates) == selected
    selected.papers[0].uncertainty = 'Details require the full text.'
    task = plan().model_copy(update={'read_full_text': True})
    assert await search.review_qualified_matches(selected, task, candidates) == selected
    assert client.chat.await_count == 0


async def test_fit_review_repairs_background_attribution_without_losing_valid_match(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    candidates[0]['abstract'] = (
        'Removing capacity constraints compromises load balancing. '
        'We propose routing as a minimum-cost maximum-flow problem with a SoftTopk operator.'
    )
    selected = assessment(candidates)
    selected.papers[0].summary = 'The method removes capacity constraints.'
    selected.papers[0].uncertainty = 'Training configuration is not described.'
    selected.overview = 'The selected approach removes capacity constraints.'
    selected.overview_paper_ids = [candidates[0]['paper_id']]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({'papers': [{
        'paper_id': candidates[0]['paper_id'], 'fit': 'direct',
        'contribution_sentence_ids': [1],
        'reason': 'The flow mechanism is relevant; capacity removal is background.',
        'replacement': {'summary': 'The method models routing as a minimum-cost maximum-flow problem.',
                        'sentence_ids': [1]},
    }], 'next_queries': [{'terms': ['unnecessary extra search']}]}), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    out = await search.review_qualified_matches(selected, plan(), candidates)
    assert out.papers[0].fit == 'direct' and out.coverage == 'sufficient'
    assert out.papers[0].sentence_ids == [1]
    assert 'minimum-cost' in out.papers[0].summary
    assert out.papers[0].why_relevant == 'The flow mechanism is relevant; capacity removal is background.'
    assert out.overview == '' and out.overview_paper_ids == []
    assert not out.next_queries  # A text repair does not create a mechanism gap.
    assert selected.papers[0].summary == 'The method removes capacity constraints.'
    supplied = json.loads(client.chat.call_args.args[0][1].content)['papers'][0]
    assert 'summary' not in supplied and 'why_relevant' not in supplied
    assert supplied['abstract_sentences'][1]['id'] == 1
    for key in ('published_at', 'authors', 'version_id', 'date_confirmed'):
        assert supplied[key] == candidates[0][key]
    assert client.chat.await_count == 1


@pytest.mark.parametrize('sentence_ids', [[999], [True], []])
async def test_fit_review_rejects_unbound_description_repair(tmp_path, monkeypatch, sentence_ids):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    selected = assessment(candidates)
    selected.papers[0].uncertainty = 'Some implementation details are absent.'
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({'papers': [{
        'paper_id': candidates[0]['paper_id'], 'fit': 'direct', 'reason': 'Relevant.',
        'contribution_sentence_ids': [0],
        'replacement': {'summary': 'A replacement.', 'sentence_ids': sentence_ids},
    }]}), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    with pytest.raises(search.PaperOutputError):
        await search.review_qualified_matches(selected, plan(), candidates)
    assert not list(tmp_path.glob('*.json'))


async def test_first_publication_uses_metadata_not_an_abstract_date(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    selected = assessment(candidates)
    selected.papers[0].constraint_checks = [search.ConstraintCheck(
        constraint_id=0, status='supported', metadata_fields=['published_at'])]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=selected.model_dump_json(), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    task = plan().model_copy(update={'hard_constraints': ['Use the original publication date']})
    out = await search.assess_candidates(task, candidates, [])
    assert out.papers[0].fit == 'direct' and out.coverage == 'sufficient'
    supplied = json.loads(client.chat.call_args.args[0][1].content)['candidates'][0]
    assert supplied['published_at'] == candidates[0]['published_at']
    assert client.chat.await_count == 1


async def test_absent_metadata_cannot_support_a_constraint(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    candidates[0]['published_at'] = None
    selected = assessment(candidates)
    selected.papers[0].constraint_checks = [search.ConstraintCheck(
        constraint_id=0, status='supported', metadata_fields=['published_at'])]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=selected.model_dump_json(), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    with pytest.raises(search.PaperOutputError, match='invalid_constraint_evidence'):
        await search.assess_candidates(plan().model_copy(update={'hard_constraints': ['published in 2026']}), candidates, [])
    assert not list(tmp_path.glob('*.json'))


async def test_two_english_sentences_and_eight_valid_references_are_not_format_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    candidates[0]['abstract'] = ' '.join(f'Method property {i} is reported.' for i in range(8))
    selected = assessment(candidates)
    selected.papers[0].summary = (
        'The paper introduces a system for executing sparsely activated models with uneven expert loads, '
        'using a computation layout that avoids discarding tokens to fit a fixed expert capacity. '
        'Its method reorganizes irregular work into efficient blocks and evaluates the resulting '
        'implementation under the hardware and workload conditions reported in the supplied abstract.')
    selected.papers[0].constraint_checks = [search.ConstraintCheck(
        constraint_id=0, status='supported', sentence_ids=list(range(8)))]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=selected.model_dump_json(), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    out = await search.assess_candidates(plan().model_copy(update={'hard_constraints': ['no token dropping']}), candidates, [])
    assert out == selected and client.chat.await_count == 1


async def test_unconfirmed_metadata_date_cannot_become_a_completed_match(monkeypatch):
    async def selector(p, candidates, queries, **kwargs):
        selected = assessment(candidates)
        selected.papers[0].constraint_checks = [search.ConstraintCheck(
            constraint_id=0, status='supported', metadata_fields=['published_at'])]
        selected.overview, selected.overview_paper_ids = 'Confirmed first publication.', [candidates[0]['paper_id']]
        return selected
    wire(monkeypatch, finder=AsyncMock(return_value=[metadata(unconfirmed=True)]), selector=selector)
    monkeypatch.setattr(papers.GLOBAL_ARXIV_CLIENT, 'fetch_metadata', AsyncMock(side_effect=httpx.ConnectError('offline')))
    result = await invoke(builder.build_graph(), question='Find the named paper and confirm its first publication')
    assert result['paper_results'][0]['fit'] == 'partial'
    assert result['status'] == 'partial'
    assert not result['paper_search']['overview']


async def test_repeated_truncation_is_not_accepted_or_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    client = AsyncMock()
    # Even parseable JSON is not complete when the provider signals truncation.
    client.chat.return_value = ChatResponse(
        content=assessment(candidates).model_dump_json(), model='test', finish_reason='length',
    )
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    with pytest.raises(search.PaperOutputError, match='output_truncated') as exc:
        await search.assess_candidates(plan(), candidates, [])
    assert exc.value.details['finish_reason'] == 'length'
    assert client.chat.await_count == 2
    assert not list(tmp_path.glob('*.json'))


@pytest.mark.parametrize('invalid', ['unknown_paper', 'duplicate', 'bad_sentence'])
async def test_selection_cannot_fabricate_source_bindings(tmp_path, monkeypatch, invalid):
    monkeypatch.setattr(search, 'CACHE', ResponseCache(tmp_path))
    candidates = [search.metadata_record(metadata())]
    raw = assessment(candidates).model_dump()
    if invalid == 'unknown_paper':
        raw['papers'][0]['paper_id'] = 'arxiv:0000.12345'
    elif invalid == 'duplicate':
        raw['papers'] *= 2
    else:
        raw['papers'][0]['sentence_ids'] = [999]
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps(raw), model='test', finish_reason='stop')
    monkeypatch.setattr(search, 'get_chat_client', lambda role: client)
    with pytest.raises(search.PaperOutputError):
        await search.assess_candidates(plan(), candidates, [])
    assert client.chat.await_count == 2  # one shape repair, then stop
    assert not list(tmp_path.glob('*.json'))


def test_sentence_ids_are_strict_and_cache_has_a_shorter_freshness_ttl(tmp_path, monkeypatch):
    with pytest.raises(ValidationError):
        search.Choice(paper_id='arxiv:2608.12345', topic_role='primary', fit='direct',
                      summary='s', why_relevant='r', sentence_ids=[True])
    from src.corpus import http
    monkeypatch.setattr(http.time, 'time', lambda: 100000)
    cache = ResponseCache(tmp_path)
    cache.put('query', ['paper'])
    monkeypatch.setattr(http.time, 'time', lambda: 104000)
    assert cache.get('query') == ['paper']
    assert cache.get('query', max_age_s=3600) is None


def test_reasoning_profile_changes_invalidate_selection_cache():
    from src.llm.registry import ReasoningSettings, snapshot_chat_routes, use_chat_routes
    with use_chat_routes(snapshot_chat_routes(ReasoningSettings(main='high'))):
        high = search.cache_key('assessment', 'main', {'question': 'agent memory'})
    with use_chat_routes(snapshot_chat_routes(ReasoningSettings(main='low'))):
        low = search.cache_key('assessment', 'main', {'question': 'agent memory'})
    assert high != low


async def test_api_cooldown_does_not_disable_existing_successful_cache(tmp_path):
    from tests.unit.test_scholarly_boundaries import FEED
    handler = AsyncMock(return_value=httpx.Response(200, text=FEED))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ArxivClient(http_client=http, cache=ResponseCache(tmp_path))
        first = await client.search('ti:"memory"', fresh=False)
        client._api_retry_at = float('inf')
        assert await client.search('ti:"memory"', fresh=False) == first
    assert handler.await_count == 1


async def test_date_range_is_preserved_by_api_and_html_fallback(tmp_path):
    requests = []
    html = '<p>Sorry, your query returned no results.</p>'
    async def handler(request):
        requests.append(request)
        return httpx.Response(503) if '/api/' in request.url.path else httpx.Response(200, text=html)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ArxivClient(http_client=http, cache=ResponseCache(tmp_path))
        client._pacer = Pacer(0)
        result = await client.search(search.Query(terms=['KV cache', 'quantization']).api_query(),
                                    date_from='2024-01-01', date_to='2024-12-31', latest=True)
    assert result == []
    assert 'submittedDate:[202401010000 TO 202412312359]' in requests[0].url.params['search_query']
    params = requests[-1].url.params
    assert params['date-date_type'] == 'submitted_date_first'
    assert params['terms-0-term'] == '"KV cache"'
    assert params['terms-1-operator'] == 'AND'


@pytest.mark.parametrize('bounded_dates', [False, True])
async def test_title_query_and_fallback_cannot_expand_to_incidental_abstract_mentions(tmp_path, bounded_dates):
    requests = []
    async def handler(request):
        requests.append(request)
        return (httpx.Response(503) if '/api/' in request.url.path else
                httpx.Response(200, text='<p>Sorry, your query returned no results.</p>'))
    query = search.Query(terms=['Atlas'], scope='title')
    assert query.key() != search.Query(terms=['Atlas']).key()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ArxivClient(http_client=http, cache=ResponseCache(tmp_path))
        client._pacer = Pacer(0)
        await client.search(query.api_query(), latest=True,
                            **({'date_from': '2026-01-01', 'date_to': '2026-09-16'} if bounded_dates else {}))
    assert requests[0].url.params['search_query'].startswith('ti:"Atlas"')
    params = requests[-1].url.params
    assert params['terms-0-field' if bounded_dates else 'searchtype'] == 'title'
