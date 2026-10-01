"""Slow checks cannot discard settled siblings or authorize unresolved context."""
import asyncio
from unittest.mock import AsyncMock

from src.core.types import ChatResponse, CitationVerdict
from src.graph.delivery import supported_text
from src.graph.nodes.finalize import finalize_node
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval.self_rag import verifier


async def test_slow_source_retains_completed_sibling_and_cancels_pending(monkeypatch):
    answer = 'A stores evidence [a#1]. B revises it [b#2].'
    stopped = asyncio.Event()

    async def check(client, sem, draft, citation, lookup, **kwargs):
        if citation.chunk_id == 1:
            return CitationVerdict(citation=citation, supports=True, rationale='source checked')
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(verifier, '_verify_one_citation', check)
    monkeypatch.setattr(verifier, 'remaining_turn_s', lambda: 3.03)
    report = await verifier.verify_citations(answer, _parse_citations(answer), {})
    assert stopped.is_set() and not report.passed
    assert report.verdicts[0].supports and report.verdicts[1].status == 'error'
    assert supported_text(answer, report) == 'A stores evidence [a#1].'
    state = await finalize_node({'answer': answer, 'verification': report})
    assert state['status'] == 'partial' and state['stop_reason'] == 'verification_unavailable'
    assert state['verification'].verdicts[0].rationale == 'source checked'


async def test_unavailable_scope_does_not_start_source_fanout(monkeypatch):
    monkeypatch.setattr(verifier, '_answer_scope', AsyncMock(return_value=None))
    source = AsyncMock(side_effect=AssertionError('No scope, no authorized context'))
    monkeypatch.setattr(verifier, '_verify_one_citation', source)
    answer = 'A [a#1].'
    report = await verifier.verify_citations(answer, _parse_citations(answer), {},
        discovery_context={'research_question': 'Explain A'})
    assert not report.passed and report.coverage_status == 'error'
    assert report.verdicts[0].status == 'error' and source.await_count == 0


async def test_scope_retry_requires_time_for_source_checks(monkeypatch):
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content='', model='fixture', finish_reason='length')
    monkeypatch.setattr(verifier, 'get_chat_client', lambda _: client)
    monkeypatch.setattr(verifier, 'remaining_turn_s', lambda: 25)
    assert await verifier._answer_scope('Explain A', ['A']) is None
    assert client.chat.await_count == 1


async def test_late_source_repair_does_not_charge_a_new_request(monkeypatch):
    client = AsyncMock()
    monkeypatch.setattr(verifier, 'remaining_turn_s', lambda: 20)
    verdict = await verifier._repair_verdict(client, '', _parse_citations('A [a#1].')[0], [])
    assert verdict.status == 'error' and not verdict.supports
    assert client.chat.await_count == 0


async def test_delivery_reserve_admits_no_new_request(monkeypatch):
    operation = AsyncMock()
    monkeypatch.setattr(verifier, 'remaining_turn_s', lambda: 2)
    try:
        await verifier._before_delivery_deadline(operation)
    except TimeoutError:
        pass
    else:
        raise AssertionError('Must preserve delivery time')
    assert operation.await_count == 0
