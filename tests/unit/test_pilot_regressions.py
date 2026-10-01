"""Behavioral regressions exposed by the frozen 16-run pilot; no paid calls."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from src.core.types import ChatResponse, CitationVerdict, VerificationReport, Message
from src.core.answer_structure import table_context
from src.graph.delivery import supported_text
from src.graph.nodes.synthesize import _parse_citations
from src.llm.budget import (RequestBudget, BudgetExceeded, BudgetWaitTimeout,
                            admit_request, budget_stop_reason, use_turn_budget)
from src.llm.client import _RoutedChatClient
from src.llm.registry import get_route
from src.llm.stages import StageSettings, use_stage_settings
from src.retrieval.self_rag import verifier, claim_audit
from src.retrieval.index.types import Hit


def verified(answer, *, reject_chunk=None):
    return VerificationReport(passed=reject_chunk is None, verdicts=[
        CitationVerdict(citation=c, supports=c.chunk_id != reject_chunk, rationale="fixture")
        for c in _parse_citations(answer)])


def test_ellipsis_in_quoted_execution_example_preserves_whole_claim():
    answer = 'A 与 B 可并行；C 的参数使用 "... $1 ... $2 ..."，须等两者的输出 [p#1]。'
    report = verified(answer)
    assert report.verdicts[0].citation.claim_text == answer
    assert supported_text(answer, report) == answer


def test_table_row_joint_sources_and_safe_structure_survive_delivery():
    table = '| 阶段 | 说明 |\n| --- | --- |\n| 定位 | 查找位置 [p#1]，保留条件 [p#2]。 |\n| 修复 | 生成补丁 [p#3]。 |'
    citations = _parse_citations(table)
    assert citations[0].claim_span == citations[1].claim_span
    assert supported_text(table, verified(table)) == table
    partial = supported_text(table, verified(table, reject_chunk=2))
    assert '| 阶段 | 说明 |\n| --- | --- |' in partial
    assert '定位' not in partial and '修复' in partial
    assert not verifier.uncited_claims(table, citations)


async def test_factual_header_is_verified_as_part_of_each_row(monkeypatch):
    table = '| Agentless has three phases | 来源 |\n| --- | --- |\n| It localizes edits [p#1]. | Paper |'
    citations = _parse_citations(table)
    assert 'three phases' in table_context(table, citations[0].claim_span)
    from src.core.types import EvidenceChunk
    async def chat(messages, **kwargs):
        assert 'Table header context' in messages[-1].content
        assert '| Agentless has three phases | 来源 |' in messages[-1].content
        return ChatResponse(model='fixture', finish_reason='stop', content=json.dumps({
            'scope_status': 'requested', 'claim_kind': 'paper_fact', 'rationale': 'Header contradicts the supplied two-phase source.',
            'all_assertions_supported': False, 'no_unstated_assumptions': True,
            'all_citations_contribute': True, 'supports': False}))
    model = AsyncMock()
    model.chat.side_effect = chat
    monkeypatch.setattr(verifier, 'get_chat_client', lambda _: model)
    report = await verifier.verify_citations(table, citations, {1: EvidenceChunk(paper_id='p', chunk_id=1, text='It has two phases and localizes edits.')})
    assert not report.passed and not supported_text(table, report)


def test_versioned_free_form_table_labels_do_not_trigger_prose_repair():
    table = '| 层级 | 阶段/步骤 | v1 中的做法 |\n|---|---|---|\n| 顶层 | 定位 | 查找位置 [p#1]。 |'
    assert not verifier.uncited_claims(table, _parse_citations(table))
    assert supported_text(table, verified(table)) == table


def test_same_cells_under_different_headers_are_not_duplicate_facts():
    first = '| 方法 | 延迟（秒） |\n|---|---|\n| Method | 1 [p#1] |'
    second = '| 方法 | 费用（美元） |\n|---|---|\n| Method | 1 [p#1] |'
    answer = first + '\n\n' + second
    assert supported_text(answer, verified(answer)) == answer


@pytest.mark.parametrize('invalid', ['new_requirement', 'wrong_requirement_id', 'unknown_statement'])
async def test_scope_cannot_invent_version_facts_or_unbound_coverage(monkeypatch, invalid):
    raw = {'items': [{'id': 0, 'scope_status': 'requested', 'reason': 'Lists phases'}],
           'coverage': [{'requirement_id': 0, 'statement_ids': [0]}]}
    if invalid == 'new_requirement':
        raw['missing_aspects'] = ['Add a third top-level patch validation phase.']
    elif invalid == 'wrong_requirement_id':
        raw['coverage'][0]['requirement_id'] = 1
    else:
        raw['coverage'][0]['statement_ids'] = [99]
    model = AsyncMock()
    model.chat.return_value = ChatResponse(content=json.dumps(raw), model='fixture', finish_reason='stop')
    monkeypatch.setattr(verifier, 'get_chat_client', lambda _: model)
    assert await verifier._answer_scope('List phases in Agentless v1.', ['Localization and repair.']) is None
    assert model.chat.await_count == 2


async def test_coverage_gap_is_bound_to_literal_user_requirement(monkeypatch):
    model = AsyncMock()
    model.chat.return_value = ChatResponse(content=json.dumps({
        'items': [{'id': 0, 'scope_status': 'requested', 'reason': 'General rule only'}],
        'coverage': [{'requirement_id': 0, 'statement_ids': []}]}), model='fixture', finish_reason='stop')
    monkeypatch.setattr(verifier, 'get_chat_client', lambda _: model)
    report = await verifier._answer_scope('Explain GQA and calculate heads.', ['One key head per group.'],
                                         requirements=['calculate heads'])
    assert report.missing_aspects == ['calculate heads']


@pytest.mark.parametrize('table_ids,required_structure,expected_gap', [
    ({0}, 'table', True), ({0, 1}, 'table', False), ({0}, 'any', False)])
async def test_scope_checks_requested_table_content_against_program_positions(
    monkeypatch, table_ids, required_structure, expected_gap
):
    model = AsyncMock()
    model.chat.return_value = ChatResponse(content=json.dumps({
        'items': [{'id': i, 'scope_status': 'requested', 'reason': 'Requested content'} for i in range(2)],
        'coverage': [{'requirement_id': 0, 'statement_ids': [0, 1],
                      'required_structure': required_structure}]}), model='fixture', finish_reason='stop')
    monkeypatch.setattr(verifier, 'get_chat_client', lambda _: model)
    report = await verifier._answer_scope('Put the stages and localization steps in a table.',
                                         ['Stages.', 'Localization steps.'],
                                         requirements=['stages and localization steps'],
                                         table_statement_ids=table_ids)
    assert report.missing_aspects == (['stages and localization steps'] if expected_gap else [])
    sent = json.loads(model.chat.call_args.args[0][1].content)
    assert sent['statements'][1]['structure'] == ('table' if 1 in table_ids else 'prose')


async def test_table_structure_for_scope_comes_from_actual_answer_rows(monkeypatch):
    answer = '| Kind | Meaning |\n| --- | --- |\n| Phase | Localize [p#1] |\n\nEdit locations [p#2].'
    captured = {}
    async def scope(question, claims, **kwargs):
        captured.update(kwargs)
        return verifier._AnswerScope(items=[verifier._ScopeItem(
            id=i, scope_status='requested', reason='Fixture') for i in range(len(claims))],
            coverage=[verifier._Coverage(requirement_id=0, statement_ids=[0, 1])])
    async def source(*args, **kwargs):
        return CitationVerdict(citation=args[3], supports=True, rationale='Fixture')
    monkeypatch.setattr(verifier, '_answer_scope', scope)
    monkeypatch.setattr(verifier, '_verify_one_citation', source)
    await verifier.verify_citations(answer, _parse_citations(answer), {},
                                   discovery_context={'research_question':'List phases and edit locations.'})
    assert captured['table_statement_ids'] == {0}


@pytest.mark.parametrize('turn_limit', [False, True])
async def test_pending_reservation_waits_and_releases_before_admission(tmp_path, turn_limit):
    budget = RequestBudget(1 if turn_limit else .016, path=tmp_path/'ledger.json')
    with budget.activate(), use_turn_budget(.016 if turn_limit else 1):
        first = await admit_request('deepseek_native', 'deepseek-flash', 'first', 10000)
        task = asyncio.create_task(admit_request('deepseek_native', 'deepseek-flash', 'second', 10000, max_wait_s=1))
        await asyncio.sleep(.01)
        assert not task.done() and len(budget.snapshot()['requests']) == 1
        first.settle({'prompt_tokens': 10, 'completion_tokens': 10})
        second = await task
        second.settle({'prompt_tokens': 10, 'completion_tokens': 10})
    assert len(budget.snapshot()['requests']) == 2
    assert budget.snapshot()['committed_usd'] < .016


async def test_uncertain_charge_is_never_waited_away_or_released():
    budget = RequestBudget(.016)
    with budget.activate():
        first = await admit_request('deepseek_native', 'deepseek-flash', 'first', 10000)
        first.uncertain()
        with pytest.raises(BudgetExceeded) as exc:
            await admit_request('deepseek_native', 'deepseek-flash', 'second', 10000, max_wait_s=.001)
        assert not isinstance(exc.value, BudgetWaitTimeout)
    assert len(budget.snapshot()['requests']) == 1
    assert budget.snapshot()['requests'][0]['status'] == 'uncertain'


async def test_budget_wait_has_distinct_timeout_and_cancellation_is_uncharged():
    budget = RequestBudget(.016)
    with budget.activate():
        await admit_request('deepseek_native', 'deepseek-flash', 'first', 10000)
        with pytest.raises(BudgetWaitTimeout) as exc:
            await admit_request('deepseek_native', 'deepseek-flash', 'second', 10000, max_wait_s=.01)
        assert budget_stop_reason(exc.value) == 'budget_wait_timeout'
        pending = asyncio.create_task(admit_request('deepseek_native', 'deepseek-flash', 'third', 10000))
        await asyncio.sleep(.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert len(budget.snapshot()['requests']) == 1


@pytest.mark.parametrize('profile', ['semantic_low', 'semantic_high'])
async def test_equal_cap_ablation_gives_both_efforts_same_total_output(profile):
    provider = AsyncMock()
    provider.chat.return_value = ChatResponse(content='truncated', model='fixture', finish_reason='length')
    client = _RoutedChatClient('main', get_route('main'), provider)
    with use_stage_settings(StageSettings(profile=profile, total_output_cap=8192)):
        response = await client.chat([Message(role='user', content='Compare')], stage='intent', max_tokens=1200)
    assert response.finish_reason == 'length' and provider.chat.await_count == 1
    assert provider.chat.call_args.kwargs['max_tokens'] == 8192
    assert provider.chat.call_args.kwargs['reasoning_effort'] == ('high' if profile.endswith('high') else 'low')


async def test_capacity_recovery_is_bounded_and_keeps_prompt_and_effort():
    provider = AsyncMock()
    provider.chat.return_value = ChatResponse(content='unfinished', model='fixture', finish_reason='length')
    client = _RoutedChatClient('main', get_route('main'), provider)
    with use_stage_settings(StageSettings(profile='semantic_low')):
        response = await client.chat([Message(role='user', content='Compare')], stage='intent', max_tokens=1200)
    calls = provider.chat.call_args_list
    assert response.finish_reason == 'length' and len(calls) == 2
    assert [c.kwargs['max_tokens'] for c in calls] == [4096, 8192]
    assert calls[0].args == calls[1].args
    assert all(c.kwargs['reasoning_effort'] == 'low' for c in calls)


async def test_named_comparison_keeps_unfit_paper_and_scopes_each_retrieval(monkeypatch):
    from src.graph import builder
    from src.graph.nodes import papers, retrieve
    from src.retrieval.paper_search import Plan
    from src.corpus.ingestion import IngestResult
    from src.core.types import SubQResult, GraphExpansion, RouteDecision
    from tests.unit.test_paper_search import metadata, wire
    names = ['Compiler', 'Planner', 'Attention']
    rows = [metadata(f'2601.0000{i}', title=name + ': A mechanism') for i, name in enumerate(names, 1)]
    proposal = Plan(question='Compare Compiler, Planner and Attention for external tool scheduling.',
                    criteria=['Dependency scheduling'], named_papers=names, read_full_text=True)
    wire(monkeypatch, planner=AsyncMock(return_value=proposal), finder=AsyncMock(return_value=rows))
    selector = AsyncMock(side_effect=AssertionError('Selected comparison objects must not be filtered'))
    monkeypatch.setattr(papers, 'assess_candidates', selector)
    monkeypatch.setattr(papers, 'ingest_paper_with_result', AsyncMock(side_effect=lambda pid, **kw:
        IngestResult('arxiv:' + pid.removeprefix('arxiv:'), 'skipped')))
    graph = builder.build_graph(checkpointer=InMemorySaver())
    state = await graph.ainvoke(builder._initial_state(proposal.question, 'comparison'),
                               {'configurable': {'thread_id': 'comparison'}}, interrupt_after=['read_papers'])
    assert state['paper_search']['full_text_status'] == 'complete'
    assert len(state['retrieval_paper_ids']) == 3 and not selector.called
    scopes = []
    async def pipeline(question, **kwargs):
        scopes.append(kwargs['filter_paper_ids'])
        selected = kwargs['subq_index']
        assert names[selected] in question
        assert all(name not in question for i, name in enumerate(names) if i != selected)
        assert 'Dependency scheduling' in question
        assert 'Other papers and their rankings' in question
        assert names[selected] in kwargs['sufficiency_question']
        assert 'Dependency scheduling' not in kwargs['sufficiency_question']
        assert 'own core mechanism' in kwargs['sufficiency_question']
        return SubQResult(subq_index=kwargs['subq_index'], sufficient=True, route_decision=RouteDecision()), GraphExpansion()
    monkeypatch.setattr(retrieve, '_run_subquestion_pipeline', pipeline)
    await retrieve._retrieve_independent(state)
    assert scopes == [[pid] for pid in state['retrieval_paper_ids']]


@pytest.mark.parametrize('name,pid,valid', [
    ('PaperAlpha 2601.00001v1', '2601.00001v1', True),
    ('PaperAlpha (arxiv:2601.00001v1)', '2601.00001v1', True),
    ('InventedTitle 2601.00001v1', '2601.00001v1', False),
    ('PaperAlpha 2601.00001v2', '2601.00001v2', False),
    ('PaperAlpha 2601.00001v2', '2601.00001v1', False)])
async def test_exact_id_allows_redundant_literal_name_only(monkeypatch, name, pid, valid):
    from src.retrieval import paper_search
    from src.core.run_context import RunContext, use_run_context
    question = 'Read PaperAlpha 的 2601.00001v1 and explain its method.'
    plan = paper_search.Plan(question=question, criteria=['method'], paper_ids=[pid],
                             named_papers=[name], read_full_text=True)
    model = AsyncMock()
    model.chat.return_value = ChatResponse(content=plan.model_dump_json(), model='fixture', finish_reason='stop')
    monkeypatch.setattr(paper_search, 'get_chat_client', lambda _: model)
    with use_run_context(RunContext(model_result_cache=False)):
        if valid:
            result = await paper_search.plan_search(question, [])
            assert result.paper_ids == ['arxiv:2601.00001v1']
            assert result.named_papers == ['PaperAlpha']
            assert model.chat.await_count == 1
        else:
            with pytest.raises(paper_search.PaperOutputError):
                await paper_search.plan_search(question, [])


@pytest.mark.parametrize('question,name,pid,valid', [
    ('Read Attention（2601.00001v1，grouped queries）.', 'Attention（grouped queries）', '2601.00001v1', True),
    ('Read Attention (grouped queries, 2601.00001v1).', 'Attention (grouped queries)', '2601.00001v1', True),
    ('阅读2601.00001v1 Attention（分组查询）原文。', 'Attention（分组查询）', '2601.00001v1', True),
    ('Read Attention（2601.00001v1，grouped queries）.', 'Attention（invented meaning）', '2601.00001v1', False),
    ('Read Attention（2601.00001v1，grouped queries）.', 'Attention（2601.00001v2，grouped queries）', '2601.00001v1', False),
    ('Read Attention（2601.00001v12，grouped queries）.', 'Attention', '2601.00001v1', False),
    ('Read Attention（2601.00001v1） and Compiler（dependency scheduling）.', 'Attention（dependency scheduling）', '2601.00001v1', False),
])
async def test_supplied_ids_do_not_break_literal_name_grounding(monkeypatch, question, name, pid, valid):
    from src.retrieval import paper_search
    from src.core.run_context import RunContext, use_run_context
    plan = paper_search.Plan(question=question, criteria=['method'], paper_ids=[pid],
                             named_papers=[name], read_full_text=True)
    model = AsyncMock()
    model.chat.return_value = ChatResponse(content=plan.model_dump_json(), model='fixture', finish_reason='stop')
    monkeypatch.setattr(paper_search, 'get_chat_client', lambda _: model)
    with use_run_context(RunContext(model_result_cache=False)):
        if valid:
            result = await paper_search.plan_search(question, [])
            assert result.paper_ids == ['arxiv:' + pid]
            assert result.named_papers == [paper_search._name_without_supplied_ids(name, [pid]) if name not in question else name]
            assert model.chat.await_count == 1
        else:
            with pytest.raises(paper_search.PaperOutputError):
                await paper_search.plan_search(question, [])


@pytest.mark.parametrize('local_goal', [None, 'Establish the own-paper mechanism.'])
async def test_sufficiency_uses_local_goal_but_keeps_global_retrieval_context(monkeypatch, local_goal):
    from src.core.types import SufficiencyVerdict
    from src.graph.nodes import retrieve
    check = AsyncMock(return_value=SufficiencyVerdict(sufficient=True))
    monkeypatch.setattr(retrieve.sufficiency_check, 'is_sufficient', check)
    query = 'Compare suitability for dependency-aware tool APIs and latency.'
    hit = Hit(1, 'paper', 'Method', 'The planner dispatches work.', 1.0)
    ctx = retrieve.RetrievalContext(subq=query, subq_index=0, question_type='comparison',
                                   question_id='turn', multi_hop_used=False,
                                   sufficiency_question=local_goal, accumulated_hits={1: hit})
    await retrieve._stage_check_sufficiency(ctx)
    assert check.call_args.args[0] == (local_goal or query)
    assert ctx.subq == query and ctx.hits


async def test_retrieval_refills_free_slot_without_waiting_for_slow_sibling(monkeypatch):
    from src.core.types import Decomposition, SubQuestion, SubQResult, GraphExpansion, RouteDecision
    from src.graph.nodes import retrieve
    active, peak, started, finished = set(), 0, [], set()
    third_started = asyncio.Event()
    async def pipeline(question, **kwargs):
        nonlocal peak
        index = kwargs['subq_index']
        active.add(index)
        peak = max(peak, len(active))
        started.append(index)
        try:
            if index == 0:
                await asyncio.sleep(0)
            elif index == 1:
                await third_started.wait()
            elif index == 2:
                assert 0 in finished and 1 in active
                third_started.set()
            elif index == 3:
                assert 1 in finished
            finished.add(index)
            return SubQResult(subq_index=index, sufficient=True, route_decision=RouteDecision()), GraphExpansion()
        finally:
            active.remove(index)
    monkeypatch.setattr(retrieve, '_run_subquestion_pipeline', pipeline)
    state = {'retrieval_paper_ids': ['p'], 'decomposition': Decomposition(sub_questions=[
        SubQuestion(text='first'), SubQuestion(text='slow'),
        SubQuestion(text='ready', depends_on=[0]), SubQuestion(text='dependent', depends_on=[1])])}
    result = await asyncio.wait_for(retrieve._retrieve_independent(state), 1)
    assert started == [0, 1, 2, 3] and peak == 2
    assert sorted(result['subq_results']) == [0, 1, 2, 3]
    assert not active


async def test_retrieval_failure_cancels_running_sibling(monkeypatch):
    from src.core.types import Decomposition, SubQuestion
    from src.graph.nodes import retrieve
    started, cancelled = asyncio.Event(), asyncio.Event()
    async def pipeline(question, **kwargs):
        if kwargs['subq_index'] == 0:
            await started.wait()
            raise ValueError('source failure fixture')
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    monkeypatch.setattr(retrieve, '_run_subquestion_pipeline', pipeline)
    state = {'retrieval_paper_ids': ['p'], 'decomposition': Decomposition(sub_questions=[
        SubQuestion(text='fails'), SubQuestion(text='running')])}
    with pytest.raises(ValueError, match='source failure fixture'):
        await asyncio.wait_for(retrieve._retrieve_independent(state), 1)
    assert cancelled.is_set()


@pytest.mark.parametrize('content', ['first query\nsecond query', '```query```', 'one useful query'])
async def test_rewriter_requires_one_query_without_paid_format_retry(monkeypatch, content):
    from src.retrieval.query import rewriter
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=content, model='fixture', finish_reason='stop')
    monkeypatch.setattr(rewriter, 'get_chat_client', lambda _: client)
    actual = await rewriter.rewrite_for_retrieval('Original scoped question')
    assert actual == ('one useful query' if content == 'one useful query' else 'Original scoped question')
    assert client.chat.await_count == 1


def test_named_resolution_does_not_guess_ambiguous_source():
    from src.retrieval.paper_search import resolve_named_papers
    candidates = [{'paper_id': 'p1', 'title': 'Memory: Fast method'}, {'paper_id': 'p2', 'title': 'Memory: New method'}]
    result = resolve_named_papers(['Memory', 'Absent'], candidates)
    assert result == {'ids': [], 'ambiguous': ['Memory'], 'missing': ['Absent']}


async def test_final_synthesis_does_not_inherit_single_paper_only_instruction(monkeypatch):
    from src.graph.nodes import synthesize
    from src.core.types import Decomposition, SubQuestion
    async def generate(question, evidence, **kwargs):
        assert 'Rank the three papers' in question
        assert 'Only establish this one paper' not in question
        return 'A source-supported conclusion [p#1].', 'stop'
    monkeypatch.setattr(synthesize, '_generate_answer', generate)
    await synthesize.synthesize_node({
        'question': 'Rank the three papers', 'paper_plan': {'read_full_text': True},
        'decomposition': Decomposition(sub_questions=[SubQuestion(text='Only establish this one paper', paper_ids=['p'])]),
        'paper_search': {'reading_questions': ['Only establish this one paper']},
        'evidence': [Hit(1, 'p', 'Methods', 'Source-supported mechanism.', 1)],
    })


async def test_unresolved_comparison_object_keeps_partial_reading_status(monkeypatch):
    from src.graph.nodes import papers
    from src.core.research_policy import ResearchPolicy
    from src.corpus.ingestion import IngestResult
    from src.retrieval.paper_search import Plan, metadata_record
    from tests.unit.test_paper_search import metadata
    record = metadata_record(metadata(title='One: A mechanism'))
    monkeypatch.setattr(papers, 'ingest_paper_with_result', AsyncMock(return_value=IngestResult('arxiv:2608.12345v1', 'skipped')))
    result = await papers.read_papers_node({
        'paper_plan': Plan(question='Compare One and Missing.', criteria=['Compare'],
                           named_papers=['One', 'Missing'], read_full_text=True).model_dump(),
        'research_policy': ResearchPolicy().model_dump(),
        'paper_candidates': [record], 'paper_search': {'errors': [], 'named_resolution': {
            'ids': ['2608.12345v1'], 'missing': ['Missing'], 'ambiguous': []}},
    })
    assert result['paper_search']['full_text_status'] == 'partial'
    assert result['retrieval_paper_ids'] == ['arxiv:2608.12345v1']


async def test_subquestion_cannot_retrieve_unacquired_paper(monkeypatch):
    from src.graph.nodes import retrieve
    from src.core.types import Decomposition, SubQuestion
    forbidden = AsyncMock(side_effect=AssertionError('Unacquired source must be rejected before retrieval'))
    monkeypatch.setattr(retrieve, '_run_subquestion_pipeline', forbidden)
    with pytest.raises(ValueError, match='exceeds acquired papers'):
        await retrieve._retrieve_independent({'decomposition': Decomposition(sub_questions=[
            SubQuestion(text='Look elsewhere', paper_ids=['other'])]), 'retrieval_paper_ids': ['selected']})
    forbidden.assert_not_called()


@pytest.mark.parametrize('compact_retry', [True, False])
async def test_audit_reselects_oversized_evidence_without_bypassing_capacity(monkeypatch, compact_retry):
    text = ('condition ' * 550) + '. Short relevant counterevidence.'
    hit = Hit(1, 'p', 'Methods', text, 1)
    monkeypatch.setattr(claim_audit, 'load_scope', AsyncMock(return_value=[hit]))
    calls = []
    from src.retrieval.evidence import sentence_spans
    source_units = sentence_spans(text)
    async def chat(messages, **kwargs):
        calls.append(kwargs['stage'])
        raw = {'outcome': 'contradicted', 'rationale': 'The source contradicts the proposition.',
               'evidence': [{'chunk_id': 1, 'sentence_ids': [len(source_units)-1] if compact_retry and len(calls) == 2 else list(range(len(source_units)))}]}
        return ChatResponse(content=json.dumps(raw), model='fixture', finish_reason='stop')
    monkeypatch.setattr(claim_audit, 'get_chat_client', lambda _: type('Model', (), {'chat': staticmethod(chat)})())
    review, hits = await claim_audit.audit_claim('Check proposition', ['p'])
    assert calls == ['claim_audit', 'semantic_repair']
    assert review['status'] == ('complete' if compact_retry else 'incomplete')
    if not compact_retry:
        assert review['failure_class'] == 'evidence_capacity'


def test_comparison_score_requires_all_originals_and_does_not_grade_source_card_order():
    from src.eval.agent_metrics import score_turn
    spec = {'expected_route': 'read', 'expected_status': 'complete', 'programmatic_checks': [],
            'gold': {'required_comparison_ids': ['p1', 'p2', 'p3'], 'required_primary_id': 'p1',
                     'relevance_grades': {'p1': 3, 'p2': 1, 'p3': 0}}}
    result = {'route': 'read', 'output': {'status': 'complete', 'paper_results': [
        {'paper_id': 'p1', 'fit': 'unassessed'}, {'paper_id': 'p2', 'fit': 'unassessed'}],
        'sources': [{'paper_id': 'p1'}, {'paper_id': 'p2'}]}}
    score = score_turn(spec, result)
    assert score['comparison_original_coverage'] == 2 / 3
    assert not score['program_pass'] and score['ndcg_at_3'] is None
    assert score['pending_checks'] == ['primary_selection', 'ndcg_at_3']


def test_nested_timeouts_are_reported_and_recovered_errors_are_separate():
    from src.eval.agent_metrics import summarize
    rows = []
    for status in ['complete', 'partial']:
        rows.append({'profile': 'semantic_low', 'result': {'elapsed_s': 10,
            'cost': {'estimated_usd': .01, 'uncertain_usd': 0},
            'output': {'status': status, 'stop_reason': 'candidate_assessment_failed',
                       'paper_search': {'errors': [{'stage': 'assessment', 'error': 'TimeoutError'}]}}}})
    summary = summarize(rows, {'semantic_low': 2})['profiles']['semantic_low']
    assert summary['turns_with_stage_errors'] == 2
    assert summary['runtime_failures'] == 1
    assert summary['stage_error_counts'] == {'assessment:TimeoutError': 2}


async def test_short_remaining_time_skips_optional_work_with_explicit_reason():
    from datetime import UTC, datetime, timedelta
    from src.graph.controller import control_research_node
    from src.core.research_policy import ResearchPolicy
    report = VerificationReport(passed=False, verdicts=[])
    update = await control_research_node({'completed_node': 'self_rag', 'paper_plan': {},
        'paper_search': {'as_of': (datetime.now(UTC)-timedelta(seconds=165)).isoformat()},
        'research_policy': ResearchPolicy(reading_timeout_s=180).model_dump(), 'verification': report})
    assert update['next_node'] == 'finalize'
    assert update['forced_stop_reason'] == 'verification_time_guard'
    assert update['paper_search']['optional_recovery_skipped']['remaining_s'] < 30
