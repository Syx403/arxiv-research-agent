import asyncio
import json
from uuid import uuid4

import pytest

from src.agent.runtime import AgentRequest, AgentRuntime
from src.core.run_context import current_run
from src.core.trace import record_event
from src.llm import usage
from src.llm.budget import RequestBudget, reserve_request
from src.llm.registry import ReasoningSettings, get_route
from src.llm.stages import StageSettings, current_settings


async def test_concurrent_runs_isolate_cost_profiles_and_usage(tmp_path):
    both_started = asyncio.Event()
    starts = []

    async def runner(question, **kwargs):
        starts.append(question)
        if len(starts) == 2:
            both_started.set()
        reservation = reserve_request("deepseek_native", "deepseek-flash", {"q": question}, 100)
        await both_started.wait()
        amount = 10 if question == "first" else 50
        reservation.settle(
            {"prompt_tokens": amount, "completion_tokens": amount, "prompt_cache_hit_tokens": 0}
        )
        usage.record("main", prompt=amount, completion=amount)
        assert current_run().turn_id
        assert get_route("main").reasoning_effort == ("low" if question == "first" else "high")
        assert current_settings().profile == (
            "semantic_low" if question == "first" else "critical_high"
        )
        return {
            "answer": question,
            "status": "complete",
            "stop_reason": "papers_found",
            "result_kind": "papers",
        }

    budget = RequestBudget(1, path=tmp_path / "ledger.json", cohere_trial=True)
    runtime = AgentRuntime(tmp_path, budget, runner=runner)
    a = AgentRequest(
        question="first",
        reasoning=ReasoningSettings(main="low", fast="low"),
        stages=StageSettings(profile="semantic_low"),
    )
    b = AgentRequest(
        question="second",
        reasoning=ReasoningSettings(main="high", fast="low"),
        stages=StageSettings(profile="critical_high"),
    )
    first, second = await asyncio.gather(runtime.query(a), runtime.query(b))
    assert first.cost["requests"] == second.cost["requests"] == 1
    assert first.cost["prompt_tokens"] == first.role_usage["main"]["prompt_tokens"] == 10
    assert second.cost["prompt_tokens"] == second.role_usage["main"]["prompt_tokens"] == 50
    assert first.cost["estimated_usd"] + second.cost["estimated_usd"] == pytest.approx(
        budget.snapshot()["estimated_settled_usd"]
    )
    assert first.session_id != second.session_id
    assert current_run().turn_id == ""


async def test_repeated_attempt_cannot_reissue_paid_request(tmp_path):
    calls = []

    async def runner(question, **kwargs):
        calls.append(question)
        return {"answer": "ok", "status": "complete", "result_kind": "papers"}

    runtime = AgentRuntime(tmp_path, RequestBudget(1), runner=runner)
    request = AgentRequest(question="query")
    await runtime.query(request)
    with pytest.raises(FileExistsError):
        await runtime.query(request)
    assert calls == ["query"]


async def test_cancel_keeps_uncertain_cost_and_releases_session(tmp_path):
    started = asyncio.Event()

    async def runner(question, **kwargs):
        reservation = reserve_request("deepseek_native", "deepseek-flash", {}, 100)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            reservation.uncertain()

    runtime = AgentRuntime(tmp_path, RequestBudget(1), runner=runner)
    request = AgentRequest(question="query")
    task = asyncio.create_task(runtime.query(request))
    await started.wait()
    task.cancel()
    result = await task
    assert result.output["stop_reason"] == "cancelled"
    assert result.cost["uncertain_usd"] > 0
    assert not runtime._busy


async def test_failure_preserves_progress_and_sanitizes_error(tmp_path):
    async def runner(question, **kwargs):
        record_event(
            "paper_search", report={"queries": []}, papers=[{"paper_id": "arxiv:2305.18323"}]
        )
        raise RuntimeError("secret provider body must not escape")

    runtime = AgentRuntime(tmp_path, RequestBudget(1), runner=runner)
    result = await runtime.query(AgentRequest(question="query"))
    assert result.output["paper_results"][0]["paper_id"] == "arxiv:2305.18323"
    assert "secret provider body" not in json.dumps(result.model_dump(mode="json"))
    assert result.output["error_type"] == "RuntimeError"


@pytest.mark.parametrize('matching_turn', [True, False])
async def test_timeout_projects_only_current_turn_verified_checkpoint(tmp_path, monkeypatch, matching_turn):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from src.graph import builder
    from src.graph.nodes.synthesize import _parse_citations
    from src.core.research_policy import ResearchPolicy
    from src.core.types import CitationVerdict, EvidenceChunk, VerificationReport

    answer = 'Checked mechanism [arxiv:2407.01489v1#1].'
    citations = _parse_citations(answer)
    checked = VerificationReport(passed=True, verdicts=[CitationVerdict(
        citation=citations[0], supports=True, rationale='Previously source-checked')])
    graph = AsyncMock()
    monkeypatch.setattr(builder, '_graph', graph)
    async def interrupted(question, **kwargs):
        graph.aget_state.return_value = SimpleNamespace(values={
            'turn_id': current_run().turn_id if matching_turn else 'previous-turn',
            'question': question, 'research_question': question, 'answer': answer,
            'result_kind': 'evidence', 'paper_plan': {'read_full_text': True},
            'verification': checked, 'citations': citations,
            'evidence_lookup': {1: EvidenceChunk(paper_id='arxiv:2407.01489v1', chunk_id=1, text='Checked mechanism.')},
        })
        await asyncio.Event().wait()
    monkeypatch.setattr(builder, 'run', interrupted)
    runtime = AgentRuntime(tmp_path, RequestBudget(1))
    result = await runtime.query(AgentRequest(question='Read the mechanism', policy=ResearchPolicy(reading_timeout_s=.01)))
    assert result.output['stop_reason'] == 'turn_timeout'
    assert result.cost['requests'] == 0
    if matching_turn:
        assert result.route == 'read' and result.output['status'] == 'partial'
        assert len(result.output['sources']) == 1 and answer in result.output['answer']
        assert graph.aupdate_state.call_args.kwargs['as_node'] == 'finalize'
    else:
        assert result.route == 'unknown' and not result.output['sources']
        assert answer not in result.output['answer']
        graph.aupdate_state.assert_not_called()


async def test_optional_recovery_timeout_keeps_already_checkpointed_verification(tmp_path, monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    from unittest.mock import AsyncMock
    from src.graph import builder
    from src.memory.episodic import session_store
    from src.core.research_policy import ResearchPolicy
    from tests.unit.test_agent_delivery import _wire

    model, _ = _wire(monkeypatch, bad=True, partial=True)
    entered = asyncio.Event()
    async def recovery(state):
        assert state['best_verified']['text']
        assert state['verification'] is not None
        entered.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(builder, 'recover_evidence_node', recovery)
    # Exercise the runtime's interrupt recovery even though production admission
    # now correctly refuses optional work under this artificially tiny deadline.
    monkeypatch.setattr('src.graph.controller.recovery_windows', lambda state: {
        'verification_reserve_s': 0, 'repair_and_verify_s': 0, 'recovery_limit_s': 10})
    # This no-I/O fixture deliberately compresses the whole turn to 0.2s.
    # Source deadline admission has its own realistic sibling-cancellation tests.
    monkeypatch.setattr('src.retrieval.self_rag.verifier.remaining_turn_s', lambda: None)
    graph = builder.build_graph(checkpointer=InMemorySaver(serde=builder.checkpoint_serializer()))
    monkeypatch.setattr(builder, '_graph', graph)
    monkeypatch.setattr(session_store, 'get_or_create_session', AsyncMock())
    request = AgentRequest(question='What is ReAct?', policy=ResearchPolicy(reading_timeout_s=.2, min_optional_recovery_s=0))
    result = await AgentRuntime(tmp_path, RequestBudget(1)).query(request)
    assert entered.is_set() and model.drafts == 1
    assert result.output['stop_reason'] == 'turn_timeout' and result.route == 'read'
    assert result.output['status'] == 'partial' and len(result.output['sources']) == 1
    assert 'interleaves reasoning' in result.output['answer']
    assert 'Monte Carlo' not in result.output['answer']
    checkpoint = await graph.aget_state({'configurable': {'thread_id': f'ara-ui-{request.session_id}'}})
    assert checkpoint.next == ()
    resumed = await builder.run('', thread_id=f'ara-ui-{request.session_id}')
    assert resumed['stop_reason'] == 'turn_timeout' and model.drafts == 1


async def test_same_session_overlap_is_rejected(tmp_path):
    started, finish = asyncio.Event(), asyncio.Event()

    async def runner(question, **kwargs):
        started.set()
        await finish.wait()
        return {"answer": "ok", "result_kind": "papers"}

    runtime = AgentRuntime(tmp_path, RequestBudget(1), runner=runner)
    sid = uuid4()
    task = asyncio.create_task(runtime.query(AgentRequest(question="first", session_id=sid)))
    await started.wait()
    with pytest.raises(RuntimeError, match="already running"):
        await runtime.query(AgentRequest(question="second", session_id=sid))
    finish.set()
    await task
