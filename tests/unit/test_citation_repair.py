import json
from unittest.mock import AsyncMock

import pytest

from src.core.types import ChatResponse, CitationVerdict, EvidenceChunk, VerificationReport
from src.graph import builder, controller
from src.graph.nodes import self_rag, synthesize
from src.graph.nodes.finalize import finalize_node
from src.retrieval.self_rag import citation_repair


def fixture_state():
    answer = "定位分成三个步骤 [p-v1#1]。修复时向模型提供代码片段和 issue 描述 [p-v1#2]。"
    citations = synthesize._parse_citations(answer)
    lookup = {
        1: EvidenceChunk(paper_id="p-v1", chunk_id=1, text="Localization has three steps."),
        2: EvidenceChunk(paper_id="p-v1", chunk_id=2, text="Repair uses code snippets."),
        3: EvidenceChunk(paper_id="p-v1", chunk_id=3,
                         text="Repair provides code snippets together with the issue description."),
        4: EvidenceChunk(paper_id="p-v2", chunk_id=4, text="A newer version uses a different repair."),
    }
    report = VerificationReport(passed=False, verdicts=[CitationVerdict(
        citation=c, supports=c.chunk_id == 1, rationale="supported" if c.chunk_id == 1 else "issue description omitted",
        claim_kind="paper_fact") for c in citations])
    return {"question": "怎样定位并修复代码？", "answer": answer, "citations": citations,
            "verification": report, "evidence_lookup": lookup, "last_verification_evidence": lookup,
            "paper_plan": {}, "paper_search": {}, "research_policy": {}, "completed_node": "self_rag"}


def proposal(monkeypatch, ids, **edits):
    content = {"bindings": [{"id": 0, "chunk_ids": ids}]}
    content.update(edits)
    model = AsyncMock()
    model.chat.return_value = ChatResponse(content=json.dumps(content), model="fixture", finish_reason="stop")
    monkeypatch.setattr(citation_repair, "get_chat_client", lambda _: model)
    return model


async def test_existing_support_rebinds_without_retrieval_or_editing_prose(monkeypatch):
    state = fixture_state()
    model = proposal(monkeypatch, [3])
    forbidden = AsyncMock(side_effect=AssertionError("Existing evidence suffices"))
    monkeypatch.setattr(self_rag, "recover_evidence", forbidden)
    update = await self_rag.recover_evidence_node(state)
    assert update["answer"] == state["answer"].replace("[p-v1#2]", "[p-v1#3]")
    assert "verification" not in update and "evidence_lookup" not in update
    assert update["verification_citation_repair_attempted"]
    assert not update.get("verification_recovery_attempted")
    state.update(update, completed_node="recover_evidence")
    action = await controller.control_research_node(state)
    assert action["next_node"] == "self_rag" and action["paper_search"]["decisions"][-1]["reason"] == "reverify_citation_bindings"
    assert not state["verification"].passed  # A proposal never authorizes delivery.
    body = json.loads(model.chat.call_args.args[0][1].content)
    # DeepSeek JSON mode requires an explicit JSON instruction in the prompt.
    assert "json" in model.chat.call_args.args[0][0].content.lower()
    assert {c["chunk_id"] for c in body["evidence"]} == {1, 2, 3}
    forbidden.assert_not_called()
    assert not builder._initial_state("Next", "thread")["verification_citation_repair_attempted"]


@pytest.mark.parametrize("ids", [[999], [4], [True], [3, 3], [3.0]])
async def test_unknown_wrong_version_and_non_integer_bindings_are_rejected(monkeypatch, ids):
    state = fixture_state()
    proposal(monkeypatch, ids)
    assert await citation_repair.rebind_citations(state) == {}


async def test_model_cannot_change_prose_or_omit_a_target(monkeypatch):
    model = proposal(monkeypatch, [3], rewritten_answer="Fabricated answer")
    assert await citation_repair.rebind_citations(fixture_state()) == {}
    model.chat.return_value.content = '{"bindings":[]}'
    assert await citation_repair.rebind_citations(fixture_state()) == {}


async def test_missing_or_contradictory_evidence_falls_back_to_bounded_expansion(monkeypatch):
    state = fixture_state()
    state["evidence_lookup"][3] = state["evidence_lookup"][3].model_copy(
        update={"text": "Repair explicitly excludes the issue description."})
    model = proposal(monkeypatch, [])
    expansion = AsyncMock(return_value={"verification_recovery_attempted": True})
    monkeypatch.setattr(self_rag, "recover_evidence", expansion)
    update = await self_rag.recover_evidence_node(state)
    assert update["verification_citation_repair_attempted"] and not update["verification_recovery_changed"]
    assert "answer" not in update
    expansion.assert_awaited_once()
    assert model.chat.await_count == 1


async def test_false_binding_cannot_bypass_verifier_and_is_never_reproposed(monkeypatch):
    state = fixture_state()
    proposal(monkeypatch, [1])  # Valid identity, but it only discusses localization.
    update = await self_rag.recover_evidence_node(state)
    state.update(update)
    async def reject(answer, citations, lookup, **kwargs):
        return VerificationReport(passed=False, verdicts=[CitationVerdict(
            citation=c, supports=c.claim_text.startswith("定位"), rationale="No repair support") for c in citations])
    monkeypatch.setattr(self_rag, "verify_citations", reject)
    state.update(await self_rag.self_rag_node(state))
    assert not state["verification"].passed
    assert await citation_repair.rebind_citations(state) == {}
    state["forced_stop_reason"] = "verification_time_guard"
    final = await finalize_node(state)
    assert "issue" not in final["verified_answer"] and final["status"] == "partial"


async def test_structured_bindings_and_missing_source_identity_are_not_rebound(monkeypatch):
    model = proposal(monkeypatch, [3])
    state = fixture_state()
    assert await citation_repair.rebind_citations({**state, "answer_fields": [{"name": "mechanism"}]}) == {}
    state["evidence_lookup"][2] = state["evidence_lookup"][2].model_copy(update={"paper_id": "wrong"})
    assert await citation_repair.rebind_citations(state) == {}
    model.chat.assert_not_called()


async def test_failed_rebinding_keeps_evidence_recovery_available_once(monkeypatch):
    state = fixture_state()
    proposal(monkeypatch, [3])
    state.update(await self_rag.recover_evidence_node(state))
    rebound = AsyncMock(side_effect=AssertionError("No repeated citation proposals"))
    monkeypatch.setattr(self_rag, "rebind_citations", rebound)
    expand = AsyncMock(return_value={"verification_recovery_attempted": True})
    monkeypatch.setattr(self_rag, "recover_evidence", expand)
    state.update(await self_rag.recover_evidence_node(state))
    assert state["verification_recovery_attempted"]
    expand.assert_awaited_once()
    rebound.assert_not_called()
