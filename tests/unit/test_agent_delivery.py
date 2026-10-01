from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver
import pytest

from src.core.trace import LocalTrace
from src.core.types import ChatResponse, Decomposition, EvidenceChunk, GraphExpansion, RouteDecision, SubQResult, SubQuestion
from src.graph import builder
from src.graph.nodes import synthesize
from src.graph.nodes.finalize import finalize_node
from src.retrieval.index.types import Hit, SourceSpan
from src.retrieval.query import decomposer
from src.retrieval.self_rag import verifier
import src.graph.nodes.retrieve as retrieve


class _Model:
    def __init__(self, *, bad=False, partial=False):
        self.bad, self.partial, self.drafts = bad, partial, 0

    async def chat(self, messages, **kwargs):
        system, prompt = messages[0].content, messages[-1].content
        if "Review the SCOPE" in system:
            payload = json.loads(messages[1].content)
            statements = payload["statements"]
            text = json.dumps({"items": [{"id": item["id"], "scope_status": "requested", "reason": "Fixture requested claim"}
                                         for item in statements],
                               "coverage": [{"requirement_id": r["id"], "statement_ids": [s["id"] for s in statements]}
                                            for r in payload["requirements"]]})
        elif "You judge whether" in system:
            claim = prompt.split("Specific claim being verified:\n")[1].split("\n\nEvidence")[0]
            text = json.dumps({"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported": True, "no_unstated_assumptions": True, "all_citations_contribute": True, "supports": "Monte Carlo" not in claim, "rationale": "fixture entailment check"})
        else:
            self.drafts += 1
            if "MemGPT" in prompt.split("Evidence:\n")[-1]:
                text = "MemGPT manages memory [arxiv:memgpt#2]."
            elif self.bad:
                text = "ReAct uses Monte Carlo Tree Search [arxiv:react#1]."
                if self.partial:
                    text = "ReAct interleaves reasoning and actions [arxiv:react#1]. " + text
            else:
                text = "ReAct interleaves reasoning and actions [arxiv:react#1]."
        return ChatResponse(content=text, model="offline-fixture", finish_reason="stop")


def _wire(monkeypatch, *, bad=False, partial=False, no_evidence=False):
    model = _Model(bad=bad, partial=partial)
    calls = []

    async def decompose(question):
        return Decomposition(sub_questions=[SubQuestion(text=question)])

    async def resolve(question, history):
        assert history
        return question

    async def pipeline(question, **kwargs):
        question = question.split(". ", 1)[-1]
        calls.append(question)
        memgpt = "MemGPT" in question
        hit = Hit(chunk_id=2 if memgpt else 1, paper_id="arxiv:memgpt" if memgpt else "arxiv:react",
                  section="abstract", text="MemGPT manages memory." if memgpt else "ReAct interleaves reasoning and actions.", score=1.0)
        return SubQResult(subq_index=0, hits=[] if no_evidence else [hit], sufficient=not no_evidence,
                          route_decision=RouteDecision(may_use_semantic_scholar_live=False),
                          missing_aspects=["No relevant source found."] if no_evidence else []), GraphExpansion()

    from src.graph.nodes import papers
    from src.retrieval.paper_search import Plan
    from src.corpus.ingestion import IngestResult
    from tests.unit.test_paper_search import metadata
    from unittest.mock import AsyncMock
    async def plan_search(question, history):
        resolved = await decomposer.resolve_followup(question, history) if history else question
        return Plan(question=resolved, criteria=['Explain the requested mechanism'],
                    paper_ids=['2608.12345'], read_full_text=True, reading_questions=[resolved])
    monkeypatch.setattr(papers, 'plan_search', plan_search)
    monkeypatch.setattr(papers.GLOBAL_ARXIV_CLIENT, 'fetch_metadata', AsyncMock(return_value=metadata()))
    monkeypatch.setattr(papers, 'ingest_paper_with_result', AsyncMock(return_value=IngestResult('arxiv:2608.12345v1', 'skipped')))
    monkeypatch.setattr(decomposer, "decompose", decompose)
    monkeypatch.setattr(decomposer, "resolve_followup", resolve)
    monkeypatch.setattr(retrieve, "_run_subquestion_pipeline", pipeline)
    monkeypatch.setattr(synthesize, "get_chat_client", lambda _: model)
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: model)
    return model, calls


@pytest.mark.asyncio
async def test_new_question_retrieves_fresh_evidence_and_resume_does_no_work(monkeypatch):
    model, calls = _wire(monkeypatch)
    graph = builder.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "same-thread"}}
    first = await graph.ainvoke(builder._initial_state("What is ReAct?", "same-thread", skip_reflection=True), config)
    second = await graph.ainvoke(builder._initial_state("What is MemGPT?", "same-thread", skip_reflection=True), config)
    resumed = await graph.ainvoke(None, config)
    assert first["status"] == second["status"] == "complete"
    assert calls == ["What is ReAct?", "What is MemGPT?"]
    assert [hit.paper_id for hit in second["evidence"]] == ["arxiv:memgpt"]
    assert "ReAct" not in second["answer"]
    assert second["conversation_context"]  # History retained as conversation, not evidence.
    assert second["verification_notes"] == []
    assert resumed["answer"] == second["answer"]
    assert model.drafts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_unchanged_rejected_prose_stops_after_one_repair(monkeypatch, tmp_path, partial):
    model, _ = _wire(monkeypatch, bad=True, partial=partial)
    trace = LocalTrace(tmp_path / "failure.jsonl", run_id="failure", mode="offline_fixture")
    with trace.activate():
        result = await builder.build_graph().ainvoke(builder._initial_state("What is ReAct?", "failure", skip_reflection=True))
    assert model.drafts == 2  # initial + unchanged repair, no useful work remains
    assert result["tool_iters"] == 1
    assert result["verification_stalled"]
    assert "Monte Carlo" not in result["answer"]
    assert result["status"] == ("partial" if partial else "incomplete")
    assert len(result["citations"]) == (1 if partial else 0)
    assert "Research incomplete" in result["answer"]
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert sum(e["event"] == "node_end" and e.get("node") == "synthesize" for e in events) == 2
    assert events[-1]["status"] == result["status"]


@pytest.mark.asyncio
async def test_empty_retrieval_stops_without_drafting_or_judging(monkeypatch):
    model, _ = _wire(monkeypatch, no_evidence=True)
    result = await builder.build_graph().ainvoke(builder._initial_state("Unknown question?", "empty", skip_reflection=True))
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "no_evidence"
    assert model.drafts == 0
    assert not result["citations"]


@pytest.mark.asyncio
async def test_paper_chunk_mismatch_is_rejected_before_judge(monkeypatch):
    def no_model(_):
        raise AssertionError("Identity mismatch must not require an LLM")
    monkeypatch.setattr(verifier, "get_chat_client", no_model)
    answer = "A fabricated attribution [arxiv:wrong#1]."
    report = await verifier.verify_citations(answer, synthesize._parse_citations(answer), {
        1: EvidenceChunk(paper_id="arxiv:react", chunk_id=1, text="ReAct evidence"),
    })
    assert not report.passed
    assert report.verdicts[0].rationale == "source_identity_mismatch"


@pytest.mark.asyncio
async def test_uncited_assertion_cannot_bypass_final_delivery(monkeypatch):
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: _Model())
    answer = "ReAct interleaves reasoning and actions [arxiv:react#1]. It guarantees perfect accuracy."
    report = await verifier.verify_citations(answer, synthesize._parse_citations(answer), {
        1: EvidenceChunk(paper_id="arxiv:react", chunk_id=1, text="ReAct interleaves reasoning and actions."),
    })
    assert report.uncited_claims == ["It guarantees perfect accuracy."]
    final = await finalize_node({"answer": answer, "verification": report})
    assert final["status"] == "partial"
    assert "perfect accuracy" not in final["answer"]
    assert len(final["citations"]) == 1


@pytest.mark.asyncio
async def test_generation_and_verification_share_exact_excerpt(monkeypatch):
    monkeypatch.setattr(synthesize, "get_chat_client", lambda _: _Model())
    hit = Hit(chunk_id=1, paper_id="arxiv:react", section="intro", text="A" * 800 + "UNSEEN CLAIM", score=1, evidence_spans=(SourceSpan(0, 800),))
    draft = await synthesize.synthesize_node({"evidence": [hit], "question": "What is ReAct?"})
    excerpt = draft["evidence_lookup"][1]
    assert excerpt.paper_id == hit.paper_id
    assert excerpt.text == "A" * 800
    assert "UNSEEN" not in excerpt.text


@pytest.mark.asyncio
async def test_budget_stop_is_persisted_as_incomplete_and_resume_does_not_retry(monkeypatch):
    from src.llm.budget import BudgetExceeded
    from src.memory.episodic import session_store
    _wire(monkeypatch)
    calls = []
    async def stop(*args, **kwargs):
        calls.append(1)
        raise BudgetExceeded("fixture exhausted")
    monkeypatch.setattr(retrieve, "_run_subquestion_pipeline", stop)
    graph = builder.build_graph(checkpointer=InMemorySaver(serde=builder.checkpoint_serializer()))
    async def get_graph():
        return graph
    async def session(_):
        return None
    monkeypatch.setattr(builder, "_get_graph", get_graph)
    monkeypatch.setattr(session_store, "get_or_create_session", session)
    result = await builder.run("What is ReAct?", thread_id="budget-stop", skip_reflection=True)
    assert result["status"] == "incomplete"
    assert result["stop_reason"] == "budget_exhausted"
    assert not result["citations"]
    resumed = await builder.run("", thread_id="budget-stop")
    assert resumed["stop_reason"] == "budget_exhausted"
    assert calls == [1]


@pytest.mark.asyncio
async def test_verifier_json_repair_keeps_original_source_context(monkeypatch):
    prompts = []
    class Model:
        async def chat(self, messages, **kwargs):
            prompts.append(messages)
            content = "broken JSON" if len(prompts) == 1 else '{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports": false, "rationale": "contradicted"}'
            return ChatResponse(content=content, model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: Model())
    answer = "ReAct only uses Monte Carlo Tree Search [arxiv:react#1]."
    report = await verifier.verify_citations(answer, synthesize._parse_citations(answer), {
        1: EvidenceChunk(paper_id="arxiv:react", chunk_id=1, text="ReAct interleaves reasoning and actions."),
    })
    assert not report.passed
    assert len(prompts) == 2
    assert any("ReAct interleaves reasoning and actions." in (m.content or "") for m in prompts[1])


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["[arxiv:react#1].", "Some claim [arxiv:react#1]."])
async def test_empty_or_detached_claim_cannot_be_verified(monkeypatch, answer):
    def forbidden(_):
        raise AssertionError("Invalid claim must not call the judge")
    monkeypatch.setattr(verifier, "get_chat_client", forbidden)
    citations = synthesize._parse_citations(answer)
    if answer.startswith("Some"):
        citations = [citations[0].model_copy(update={"claim_text": "A different sentence."})]
    report = await verifier.verify_citations(answer, citations, {
        1: EvidenceChunk(paper_id="arxiv:react", chunk_id=1, text="ReAct evidence."),
    })
    assert not report.passed
