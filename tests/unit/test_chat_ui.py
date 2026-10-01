from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
import pytest

from src.core.trace import record_event
from src.core.types import Citation, EvidenceChunk
from src.llm.budget import BudgetExceeded, RequestBudget, reserve_request
from src.llm.registry import ReasoningSettings, get_route
from src.ui.app import create_app
from src.ui.presentation import paper_url, present_state
from src.ui.runtime import BusyError, ChatRuntime

HEADERS = {"X-ARA-Client": "local-ui"}


def result(question="What is ReAct?"):
    citation = Citation(paper_id="arxiv:2210.03629", chunk_id=7, claim_text="ReAct interleaves reasoning and actions.")
    return {"question": question, "research_question": question,
            "answer": "ReAct interleaves reasoning and actions [arxiv:2210.03629#7].",
            "status": "complete", "stop_reason": "answer_verified", "citations": [citation],
            "evidence_lookup": {7: EvidenceChunk(paper_id=citation.paper_id, chunk_id=7, text="ReAct interleaves reasoning and acting.")}}


def runtime(tmp_path, runner):
    return ChatRuntime(tmp_path / "sessions", RequestBudget(1, path=tmp_path / "budget.json", cohere_trial=True), runner=runner)


@pytest.mark.asyncio
async def test_ui_retries_and_refresh_never_repeat_a_paid_run(tmp_path):
    calls = []

    async def runner(question, **kwargs):
        calls.append((question, kwargs))
        reservation = reserve_request("deepseek_native", "deepseek-flash", {"prompt": question}, 40)
        reservation.settle({"prompt_tokens": 20, "completion_tokens": 10})
        record_event("node_end", node="decompose", research_question=question, context_turns=2, subquestions=[question])
        return result(question)

    chat = runtime(tmp_path, runner)
    session = chat.create_session()
    request_id = str(uuid4())
    chat.start(session["id"], "What is ReAct?", request_id)
    chat.start(session["id"], "What is ReAct?", request_id)
    await chat.task
    chat.start(session["id"], "What is ReAct?", request_id)
    assert len(calls) == 1
    restored = runtime(tmp_path, runner)
    turn = restored.get_session(session["id"])["turns"][0]
    assert turn["status"] == "complete"
    assert turn["sources"][0]["text"] == "ReAct interleaves reasoning and acting."
    assert turn["sources"][0]["url"] == "https://arxiv.org/abs/2210.03629"
    assert turn["cost"]["estimated_usd"] > 0
    assert restored.budget_summary()["committed_usd"] == chat.budget_summary()["committed_usd"]
    with pytest.raises(ValueError):
        chat.start(session["id"], "Different question", request_id)


@pytest.mark.asyncio
async def test_global_run_guard_stop_and_shared_budget(tmp_path):
    started = asyncio.Event()

    async def runner(question, **kwargs):
        reservation = reserve_request("deepseek_native", "deepseek-flash", {}, 20)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            reservation.uncertain()

    chat = runtime(tmp_path, runner)
    first, second = chat.create_session(), chat.create_session()
    chat.start(first["id"], "A question", str(uuid4()))
    await started.wait()
    with pytest.raises(BusyError):
        chat.start(second["id"], "Another question", str(uuid4()))
    await chat.stop(first["id"])
    assert first["turns"][0]["stop_reason"] == "cancelled"
    assert first["turns"][0]["cost"]["uncertain_usd"] > 0
    assert chat.budget_summary()["uncertain_or_reserved_usd"] > 0
    assert second["turns"] == []
    assert chat.active_session is None


@pytest.mark.asyncio
async def test_immediate_stop_and_failure_do_not_leave_chat_busy(tmp_path):
    async def runner(question, **kwargs):
        raise RuntimeError("a-provider-response-with-a-secret")

    chat = runtime(tmp_path, runner)
    session = chat.create_session()
    chat.start(session["id"], "First", str(uuid4()))
    await chat.stop(session["id"])
    assert session["turns"][0]["status"] == "incomplete"
    chat.start(session["id"], "Second", str(uuid4()))
    await chat.task
    assert session["turns"][1]["stop_reason"] == "runtime_error"
    assert "a-provider-response-with-a-secret" not in json.dumps(session)
    assert chat.active_session is None


@pytest.mark.asyncio
async def test_budget_rejection_during_trace_does_not_deadlock(tmp_path):
    async def runner(question, **kwargs):
        try:
            reserve_request("deepseek_native", "deepseek-flash", {}, 1000000)
        except BudgetExceeded:
            return {"status": "incomplete", "stop_reason": "budget_exhausted", "answer": "Budget stopped."}
        raise AssertionError("The expensive request must be blocked.")

    chat = runtime(tmp_path, runner)
    session = chat.create_session()
    chat.start(session["id"], "A question", str(uuid4()))
    await asyncio.wait_for(chat.task, 2)
    assert session["turns"][0]["stop_reason"] == "budget_exhausted"
    assert chat.budget_summary()["requests"] == 0


def test_only_bound_delivered_sources_are_exposed():
    state = result()
    state["evidence_lookup"][7] = EvidenceChunk(paper_id="arxiv:2303.11366", chunk_id=7, text="Wrong identity.")
    state["evidence_lookup"][8] = EvidenceChunk(paper_id="arxiv:private-draft", chunk_id=8, text="Unused evidence.")
    assert present_state(state)["sources"] == []
    assert paper_url("arxiv:javascript:alert(1)") is None
    assert paper_url("arxiv:../../etc/passwd") is None


def test_interrupted_report_is_recovered_without_claiming_zero_cost(tmp_path):
    async def runner(question, **kwargs):
        raise AssertionError("Loading history must not execute the graph.")

    chat = runtime(tmp_path, runner)
    session = chat.create_session()
    session["turns"] = [{"status": "running", "stage": "检索论文", "answer": ""}]
    chat._save(session)
    restored = runtime(tmp_path, runner)
    turn = restored.get_session(session["id"])["turns"][0]
    assert turn["status"] == "incomplete"
    assert turn["stop_reason"] == "interrupted"
    assert turn["cost_incomplete"] is True
    assert restored.active_session is None


def test_http_validation_history_and_static_assets(tmp_path):
    async def runner(question, **kwargs):
        return result(question)

    chat = runtime(tmp_path, runner)
    with TestClient(create_app(chat)) as client:
        assert client.get("/").status_code == 200
        assert "让每个结论" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/status").json()["clarification_enabled"] is True
        assert client.post("/api/sessions").status_code == 403
        assert client.post("/api/sessions", headers={**HEADERS, "Origin": "https://unrelated.example"}).status_code == 403
        session = client.post("/api/sessions", headers=HEADERS).json()
        url = f"/api/sessions/{session['id']}/messages"
        assert client.post(url, headers=HEADERS, json={"question": " ", "request_id": str(uuid4())}).status_code == 422
        assert client.post(url, headers=HEADERS, json={"question": "x" * 8001, "request_id": str(uuid4())}).status_code == 422
        payload = {"question": "What is ReAct?", "request_id": str(uuid4())}
        assert client.post(url, headers=HEADERS, json=payload).status_code == 202
        assert client.post(url, headers=HEADERS, json=payload).status_code == 202
        assert len(client.get(f"/api/sessions/{session['id']}").json()["turns"]) == 1
        assert len(client.get("/api/sessions").json()) == 1
        assert client.get("/api/sessions/not-a-uuid").status_code == 422


@pytest.mark.asyncio
async def test_chat_followup_uses_graph_history_but_new_session_does_not(tmp_path, monkeypatch):
    # Exercise the real graph, decompose node, finalizer and checkpoint reducer.
    from src.graph import builder
    from src.retrieval.query import decomposer
    from tests.unit.test_agent_delivery import _wire

    _wire(monkeypatch)
    histories = []

    async def resolve(question, history):
        histories.append(history)
        return "What is ReAct?" if "its" in question else question

    monkeypatch.setattr(decomposer, "resolve_followup", resolve)
    graph = builder.build_graph(checkpointer=InMemorySaver())

    async def runner(question, *, thread_id, skip_reflection):
        return await graph.ainvoke(builder._initial_state(question, thread_id, skip_reflection=skip_reflection),
                                   {"configurable": {"thread_id": thread_id}})

    chat = runtime(tmp_path, runner)
    first = chat.create_session()
    chat.start(first["id"], "What is ReAct?", str(uuid4()))
    await chat.task
    chat.start(first["id"], "What is its mechanism?", str(uuid4()))
    await chat.task
    assert first["turns"][1]["research_question"] == "What is ReAct?"
    assert first["turns"][1]["context_messages"] == 2
    assert len(histories) == 1
    second = chat.create_session()
    chat.start(second["id"], "What is MemGPT?", str(uuid4()))
    await chat.task
    assert second["turns"][0]["context_messages"] == 0
    assert len(histories) == 1
    assert "ReAct" not in second["turns"][0]["answer"]


@pytest.mark.asyncio
async def test_each_turn_freezes_reasoning_and_conflicting_retries_are_rejected(tmp_path):
    actual = []
    async def runner(question, **kwargs):
        before = (get_route("main"), get_route("fast"))
        await asyncio.sleep(0)
        assert before == (get_route("main"), get_route("fast"))
        actual.append(before)
        return result(question)
    chat = runtime(tmp_path, runner)
    session = chat.create_session()
    first_id = str(uuid4())
    chat.start(session["id"], "First", first_id, reasoning=ReasoningSettings(main="max", fast="high"))
    await chat.task
    with pytest.raises(ValueError, match="思考档位"):
        chat.start(session["id"], "First", first_id, reasoning=ReasoningSettings(main="low", fast="low"))
    chat.start(session["id"], "Second", str(uuid4()), reasoning=ReasoningSettings(main="low", fast="low"))
    await chat.task
    assert [(a.reasoning_effort, b.reasoning_effort) for a, b in actual] == [("max", "high"), ("low", "low")]
    assert all(a.vendor_model == b.vendor_model == "deepseek-flash" for a, b in actual)
    restored = runtime(tmp_path, runner).get_session(session["id"])
    assert restored["turns"][0]["profiles"]["main"]["effort"] == "max"
    assert restored["turns"][1]["profiles"]["main"]["effort"] == "low"
    assert get_route("main").reasoning_effort == "high"


def test_invalid_ui_effort_cannot_start_a_run(tmp_path):
    async def runner(question, **kwargs):
        raise AssertionError("Invalid input must not launch a paid run")
    chat = runtime(tmp_path, runner)
    with TestClient(create_app(chat)) as client:
        session = client.post("/api/sessions", headers=HEADERS).json()
        for settings in [{"main": "extreme", "fast": "low"}, {"main": "low", "secret": "unexpected"}]:
            response = client.post(f"/api/sessions/{session['id']}/messages", headers=HEADERS, json={
                "question": "What is ReAct?", "request_id": str(uuid4()), "reasoning": settings,
            })
            assert response.status_code == 422
        assert chat.get_session(session["id"])["turns"] == []


@pytest.mark.asyncio
async def test_research_policy_is_frozen_per_turn_and_idempotent(tmp_path):
    from src.core.research_policy import current_policy, ResearchPolicy
    seen = []
    async def runner(question, **kwargs):
        seen.append(current_policy().max_queries)
        return result(question)
    chat = ChatRuntime(tmp_path / "sessions", RequestBudget(1), runner=runner,
                       research_policy=ResearchPolicy(max_queries=4))
    session = chat.create_session()
    request_id = str(uuid4())
    chat.start(session["id"], "latest", request_id)
    await chat.task
    chat.start(session["id"], "latest", request_id)
    assert seen == [4]
    with pytest.raises(ValueError):
        chat.start(session["id"], "different", request_id)
    assert current_policy().max_queries == 8


def test_api_has_one_acquisition_path_and_rejects_legacy_source_mode(tmp_path):
    async def runner(*args, **kwargs):
        raise AssertionError("Invalid legacy options must not start a run")
    chat = ChatRuntime(tmp_path / "sessions", RequestBudget(1), runner=runner)
    with TestClient(create_app(chat)) as client:
        status = client.get("/api/status").json()
        assert "corpus_mode" not in status
        session = client.post("/api/sessions", headers=HEADERS).json()
        response = client.post(f"/api/sessions/{session['id']}/messages", headers=HEADERS, json={
            "question": "latest", "request_id": str(uuid4()), "research_mode": "local",
        })
        assert response.status_code == 422
        assert not chat.get_session(session["id"])["turns"]
