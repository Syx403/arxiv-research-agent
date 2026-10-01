"""Resume a partially completed paper search from real Postgres, without paid APIs."""
import os
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.core.research_policy import ResearchPolicy, use_research_policy
from src.graph import builder
from src.graph.nodes import papers
from src.corpus.ingestion import IngestResult
from tests.integration.test_resume import _cleanup_thread
from tests.unit.test_paper_search import plan, wire

pytestmark = pytest.mark.skipif(os.getenv('INTEGRATION_TESTS') != '1', reason='requires local Postgres')


async def test_new_connection_resumes_candidates_without_repeating_search(monkeypatch):
    wire(monkeypatch)
    thread = 't-paper-resume-' + str(uuid4())
    config = {'configurable': {'thread_id': thread}}
    try:
        graph = await builder._get_graph()
        with use_research_policy(ResearchPolicy()):
            initial = builder._initial_state('Find recent memory papers', thread)
        partial = await graph.ainvoke(initial, config, interrupt_after=['search_papers'])
        assert partial['paper_candidates'] and not partial['paper_results']
        calls = papers.GLOBAL_ARXIV_CLIENT.search.await_count
        await builder.shutdown()
        result = await builder.run('', thread_id=thread, skip_reflection=True)
        assert result['status'] == 'complete'
        assert result['paper_results'][0]['paper_id'] == 'arxiv:2608.12345'
        assert papers.GLOBAL_ARXIV_CLIENT.search.await_count == calls
        snapshot = await (await builder._get_graph()).aget_state(config)
        assert snapshot.next == ()
    finally:
        await _cleanup_thread(thread)
        await builder.shutdown()


async def test_selected_reading_resumes_with_same_identity_and_no_selector(monkeypatch):
    wire(monkeypatch, planner=AsyncMock(return_value=plan('深入阅读这篇', deep=True, ids=['2608.12345']).model_copy(
        update={'reading_scope': 'overview'})))
    selector = AsyncMock(side_effect=AssertionError('A chosen source is not a recommendation'))
    monkeypatch.setattr(papers, 'assess_candidates', selector)
    ingest = AsyncMock(return_value=IngestResult('arxiv:2608.12345', 'ingested', 40))
    monkeypatch.setattr(papers, 'ingest_paper_with_result', ingest)
    thread = 't-reading-resume-' + str(uuid4())
    config = {'configurable': {'thread_id': thread}}
    try:
        graph = await builder._get_graph()
        with use_research_policy(ResearchPolicy()):
            initial = builder._initial_state('深入阅读 arxiv:2608.12345', thread)
        await graph.ainvoke(initial, config, interrupt_after=['search_papers'])
        assert (await graph.aget_state(config)).next == ('control_research',)
        await builder.shutdown()
        graph = await builder._get_graph()
        state = await graph.ainvoke(None, config, interrupt_after=['read_papers'])
        assert state['retrieval_paper_ids'] == ['arxiv:2608.12345']
        assert state['paper_search']['full_text_status'] == 'complete'
        assert len(state['decomposition'].sub_questions) == 3
        assert (await graph.aget_state(config)).next == ('control_research',)
        assert papers.GLOBAL_ARXIV_CLIENT.fetch_metadata.await_count == 1
        ingest.assert_awaited_once()
        selector.assert_not_called()
    finally:
        await _cleanup_thread(thread)
        await builder.shutdown()
