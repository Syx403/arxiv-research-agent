from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from src.core.types import CitationVerdict, RelevanceVerdict, VerificationReport
from src.graph import builder
from src.graph.edges import route_after_verify
from src.graph.nodes import self_rag, synthesize
from src.graph.nodes.finalize import finalize_node
from src.llm.budget import BudgetExceeded
from src.llm.errors import LLMServerError
from src.retrieval.evidence import evidence_text, sentence_spans
from src.retrieval.index.types import Hit, SourceSpan
from src.retrieval.self_rag import evidence_recovery


def fixture_state():
    text = "Memory search is self-directed. Reading external memory uses function calls."
    hit = Hit(1, "p", "Method", text, 1, evidence_spans=(sentence_spans(text)[0],), evidence_role="direct")
    answer = "Reading external memory uses function calls [p#1]."
    citations = synthesize._parse_citations(answer)
    report = VerificationReport(verdicts=[CitationVerdict(citation=citations[0], supports=False,
                                                         rationale="Selected excerpt omits retrieval mechanism.")], passed=False)
    # These fixtures exercise expansion after existing-source rebinding was exhausted.
    state = {"question":"How is external memory accessed?", "answer":answer,
             "verification_citation_repair_attempted": True,
             "citations":citations, "evidence":[hit], "evidence_lookup":synthesize._evidence_lookup([hit])}
    return state, report


@pytest.mark.asyncio
async def test_missing_mechanism_is_recovered_from_same_original_once(monkeypatch):
    state, report = fixture_state()
    judge = AsyncMock(return_value=[RelevanceVerdict(relevant=True, evidence_role="direct", sentence_ids=[1], rationale="Mechanism")])
    monkeypatch.setattr(evidence_recovery, "judge_batch", judge)
    update = await evidence_recovery.recover_evidence(state, report)
    assert update["verification_recovery_attempted"]
    assert "uses function calls" in update["evidence_lookup"][1].text
    assert update["evidence"][0].text == state["evidence"][0].text
    assert all(span in update["evidence"][0].evidence_spans for span in state["evidence"][0].evidence_spans)
    assert await evidence_recovery.recover_evidence({**state, **update}, report) == {}
    assert judge.await_count == 1
    assert not builder._initial_state("Next question", "thread")["verification_recovery_attempted"]


@pytest.mark.asyncio
async def test_recovery_keeps_official_url_and_does_not_force_a_full_rewrite(monkeypatch):
    state, report = fixture_state()
    hit = replace(state["evidence"][0], url="https://huggingface.co/deepseek-ai/Model/resolve/main/Report.pdf")
    state.update(evidence=[hit], evidence_lookup=synthesize._evidence_lookup([hit]))
    monkeypatch.setattr(evidence_recovery, "judge_batch", AsyncMock(return_value=[
        RelevanceVerdict(relevant=True, evidence_role="direct", sentence_ids=[1], rationale="More context")]))
    update = await evidence_recovery.recover_evidence(state, report)
    assert update["evidence_lookup"] == synthesize._evidence_lookup(update["evidence"])
    assert update["evidence_lookup"][1].url == hit.url
    repair = AsyncMock(return_value=(state["answer"], "stop"))
    monkeypatch.setattr(synthesize, "_repair_failed_claims", repair)
    monkeypatch.setattr(synthesize, "_generate_answer", AsyncMock(side_effect=AssertionError("Unexpected full rewrite")))
    await synthesize.synthesize_node({**state, **update, "verification": report})
    repair.assert_awaited_once()


@pytest.mark.asyncio
async def test_added_context_rechecks_affected_claim_and_binds_final_evidence(monkeypatch):
    state, report = fixture_state()
    monkeypatch.setattr(evidence_recovery, "judge_batch", AsyncMock(return_value=[
        RelevanceVerdict(relevant=True, evidence_role="direct", sentence_ids=[1], rationale="Mechanism")]))
    calls = []
    async def verify(answer, citations, lookup, **kwargs):
        calls.append((lookup, kwargs))
        if "uses function calls" not in lookup[1].text:
            return report
        return VerificationReport(verdicts=[CitationVerdict(citation=citations[0], supports=True, rationale="Original mechanism found")], passed=True)
    monkeypatch.setattr(self_rag, "verify_citations", verify)
    first = await self_rag.self_rag_node(state)
    assert not first['verification'].passed and len(calls) == 1
    state.update(first)
    expanded = await self_rag.recover_evidence_node(state)
    assert not expanded['verification'].passed and expanded['best_verified'] == {}
    assert all(not v.supports for v in expanded['verification'].verdicts)
    state.update(expanded)
    update = {**expanded, **await self_rag.self_rag_node(state)}
    assert update["verification"].passed
    assert len(calls) == 2 and calls[1][1]["cached_verdicts"] == report.verdicts
    # Candidates retain the rejection; the verifier's source digest decides
    # whether it is reusable after the bound excerpt changes.
    assert update["best_verified"]["evidence_lookup"] == update["evidence_lookup"]
    assert update["last_verification_evidence"] == update["evidence_lookup"]


@pytest.mark.asyncio
async def test_recovery_can_find_another_bound_fragment_of_the_same_paper(monkeypatch):
    state, report = fixture_state()
    mechanism = replace(state["evidence"][0], chunk_id=2)
    general_text = "Memory is described. Its history persists."
    general = replace(state["evidence"][0], text=general_text, evidence_spans=(sentence_spans(general_text)[0],))
    state.update(evidence=[general, mechanism], evidence_lookup=synthesize._evidence_lookup([general, mechanism]))
    async def select(question, hits):
        assert hits[0].chunk_id == 2
        return [RelevanceVerdict(relevant=h.chunk_id == 2, evidence_role="direct" if h.chunk_id == 2 else "irrelevant",
                                 sentence_ids=[1] if h.chunk_id == 2 else [], rationale="Original mechanism") for h in hits]
    monkeypatch.setattr(evidence_recovery, "judge_batch", select)
    update = await evidence_recovery.recover_evidence(state, report)
    assert "uses function calls" in update["evidence_lookup"][2].text
    assert update["evidence_lookup"][2].paper_id == "p"
    assert update["evidence_lookup"][1] == state["evidence_lookup"][1]


@pytest.mark.asyncio
async def test_recovery_does_not_drop_existing_evidence_to_exceed_chunk_budget(monkeypatch):
    state, report = fixture_state()
    old = "old " * 390 + "."
    text = old + " " + "new " * 150 + "."
    hit = replace(state["evidence"][0], text=text, evidence_spans=(SourceSpan(0, len(old)),))
    state.update(evidence=[hit], evidence_lookup=synthesize._evidence_lookup([hit]))
    judge = AsyncMock(return_value=[RelevanceVerdict(relevant=True, evidence_role="direct",
                                                    sentence_ids=[len(sentence_spans(text))-1], rationale="More detail")])
    monkeypatch.setattr(evidence_recovery, "judge_batch", judge)
    update = await evidence_recovery.recover_evidence(state, report)
    assert update == {"verification_recovery_attempted": True}
    assert evidence_text(state["evidence"][0]) == old


@pytest.mark.asyncio
async def test_wrong_source_identity_does_not_trigger_recovery(monkeypatch):
    state, report = fixture_state()
    verdict = report.verdicts[0]
    report = report.model_copy(update={"verdicts": [verdict.model_copy(update={
        "citation": verdict.citation.model_copy(update={"paper_id": "another-paper"})})]})
    judge = AsyncMock(side_effect=AssertionError("No recovery for a wrong source"))
    monkeypatch.setattr(evidence_recovery, "judge_batch", judge)
    assert await evidence_recovery.recover_evidence(state, report) == {}
    judge.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, reason", [(BudgetExceeded("budget"), "budget_exhausted"),
                                            (LLMServerError("unavailable", provider="fixture"), "provider_unavailable")])
async def test_recovery_failure_preserves_previously_checked_content(monkeypatch, failure, reason):
    state, _ = fixture_state()
    state["answer"] = "Memory search is self-directed [p#1]. " + state["answer"]
    state["citations"] = synthesize._parse_citations(state["answer"])
    report = VerificationReport(verdicts=[CitationVerdict(citation=c, supports=c.claim_text.startswith("Memory"),
                                                          rationale="Original selected evidence check") for c in state["citations"]], passed=False)
    monkeypatch.setattr(self_rag, "verify_citations", AsyncMock(return_value=report))
    monkeypatch.setattr(evidence_recovery, "judge_batch", AsyncMock(side_effect=failure))
    state.update(await self_rag.self_rag_node(state))
    state.update(await self_rag.recover_evidence_node(state))
    assert state["forced_stop_reason"] == reason
    assert route_after_verify(state) == "finalize"
    final = await finalize_node(state)
    assert final["status"] == "partial"
    assert final["verified_answer"] == "Memory search is self-directed [p#1]."
