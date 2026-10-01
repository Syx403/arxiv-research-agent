"""Optional recovery must leave verification time and preserve only unaffected facts."""
import asyncio
from datetime import UTC, datetime
from dataclasses import replace
from unittest.mock import AsyncMock
import pytest

from src.core.research_policy import ResearchPolicy
from src.core.run_context import RunContext, remaining_turn_s, use_run_context
from src.core.types import CitationVerdict, VerificationReport
from src.graph import controller
from src.graph.nodes import self_rag, synthesize
from src.graph.nodes.finalize import finalize_node
from src.retrieval.index.types import Hit
from tests.unit.test_verification_evidence_recovery import fixture_state


def control_state(**edits):
    return {"paper_search": {"as_of": datetime.now(UTC).isoformat()}, "paper_plan": {},
            "research_policy": ResearchPolicy().model_dump(), "completed_node": "self_rag",
            "verification": VerificationReport(passed=False, verdicts=[]),
            "node_timings_s": {"synthesize": 21, "self_rag": 33}, **edits}


async def test_recovery_is_not_admitted_just_because_thirty_seconds_remain(monkeypatch):
    monkeypatch.setattr(controller, "remaining_turn_s", lambda: 42)
    result = await controller.control_research_node(control_state())
    assert result["next_node"] == "finalize"
    assert result["forced_stop_reason"] == "verification_time_guard"
    window = result["paper_search"]["optional_recovery_skipped"]
    assert window["verification_reserve_s"] > 39 and window["recovery_limit_s"] < 5


async def test_admission_leaves_time_for_followup_verification(monkeypatch):
    monkeypatch.setattr(controller, "remaining_turn_s", lambda: 80)
    state = control_state()
    window = controller.recovery_windows(state)
    assert window["recovery_limit_s"] <= state["research_policy"]["max_evidence_recovery_s"]
    assert window["recovery_limit_s"] + window["verification_reserve_s"] < 80
    assert (await controller.control_research_node(state))["next_node"] == "recover_evidence"
    # If timing varies, check again after acquisition rather than blindly starting verification.
    monkeypatch.setattr(controller, "remaining_turn_s", lambda: 15)
    state.update(completed_node="recover_evidence", verification_recovery_changed=True)
    result = await controller.control_research_node(state)
    assert result["next_node"] == "finalize" and result["forced_stop_reason"] == "verification_time_guard"


async def test_repair_reserves_generation_plus_verification(monkeypatch):
    state = control_state(verification_recovery_attempted=True)
    monkeypatch.setattr(controller, "remaining_turn_s", lambda: 50)
    assert (await controller.control_research_node(state))["next_node"] == "finalize"
    monkeypatch.setattr(controller, "remaining_turn_s", lambda: 90)
    assert (await controller.control_research_node(state))["next_node"] == "synthesize"


def test_deadline_isolated_and_independent_of_paper_search_timestamp(monkeypatch):
    monkeypatch.setattr("src.core.run_context.perf_counter", lambda: 100)
    with use_run_context(RunContext(deadline_monotonic=110)):
        assert remaining_turn_s() == 10
        assert controller.reading_remaining(control_state()) == 10
    assert remaining_turn_s() is None


async def test_optional_timeout_cancels_work_and_preserves_original_checkpoint(monkeypatch):
    state, report = fixture_state()
    state["verification"] = report
    cancelled = []
    async def slow(*args):
        try:
            await asyncio.sleep(.1)
        finally:
            cancelled.append(True)
    monkeypatch.setattr(self_rag, "recover_evidence", slow)
    original_timeout = asyncio.timeout
    monkeypatch.setattr(self_rag.asyncio, "timeout", lambda seconds: original_timeout(.001))
    update = await self_rag.recover_evidence_node(state)
    assert cancelled and update == {"verification_recovery_attempted": True, "verification_recovery_changed": False}
    assert state["verification"] is report and "evidence_lookup" not in update


async def test_changed_paper_invalidates_all_its_claims_and_inferences_but_keeps_other_facts():
    hits = [Hit(1, "p", "Method", "P initially appears sufficient.", 1),
            Hit(2, "q", "Method", "Q has a queue.", 1),
            Hit(3, "p", "Discussion", "P has another claim.", 1)]
    answer = ("P initially appears sufficient [p#1]. Q has a queue [q#2]. "
              "P has another claim [p#3]. Engineering inference: Q is preferred [q#2]. "
              "P and Q have combined support [p#1][q#2].")
    citations = synthesize._parse_citations(answer)
    verdicts = [CitationVerdict(citation=c, supports=True, rationale="original check",
                claim_kind="engineering_inference" if "inference" in c.claim_text else "paper_fact") for c in citations]
    state = {"question": "Explain mechanisms", "answer": answer, "citations": citations, "evidence": hits,
             "evidence_lookup": synthesize._evidence_lookup(hits),
             "verification": VerificationReport(verdicts=verdicts, passed=False)}
    lookup = synthesize._evidence_lookup([replace(hits[0], text="P actually has additional conditions."), *hits[1:]])
    state.update(evidence_lookup=lookup, **self_rag._preserve_unaffected_facts(state, lookup))
    kept = state["verification"].verdicts
    assert len(kept) == 1 and kept[0].citation.claim_text == "Q has a queue [q#2]."
    assert state["last_verification_evidence"] == lookup
    state["forced_stop_reason"] = "verification_time_guard"
    final = await finalize_node(state)
    assert final["verified_answer"] == "Q has a queue [q#2]."
    assert final["status"] == "partial"


async def test_no_optional_calls_when_window_cannot_leave_verification_time(monkeypatch):
    state, report = fixture_state()
    state.update(control_state(), verification=report)
    monkeypatch.setattr(controller, "remaining_turn_s", lambda: 20)
    recover = AsyncMock(side_effect=AssertionError("No late model call"))
    monkeypatch.setattr(self_rag, "recover_evidence", recover)
    result = await self_rag.recover_evidence_node(state)
    assert not result["verification_recovery_changed"]
    recover.assert_not_called()


@pytest.mark.parametrize('changed_id,expected_action', [(1, 'self_rag'), (2, 'synthesize')])
async def test_only_changed_bound_evidence_can_trigger_direct_reverification(monkeypatch, changed_id, expected_action):
    answer = 'P needs a queue [p#1].'
    hits = [Hit(1, 'p', 'Method', 'P has no proven queue.', 1),
            Hit(2, 'p', 'Appendix', 'Additional background.', 1)]
    lookup = synthesize._evidence_lookup(hits)
    report = VerificationReport(passed=False, verdicts=[CitationVerdict(
        citation=synthesize._parse_citations(answer)[0], supports=False, rationale='No source support',
        input_digest='old-source-check')])
    state = {**control_state(), 'answer': answer, 'citations': synthesize._parse_citations(answer),
             'evidence_lookup': lookup, 'verification': report,
             'verification_citation_repair_attempted': True}
    changed = {**lookup, changed_id: lookup[changed_id].model_copy(update={'text': 'Expanded original material'})}
    monkeypatch.setattr(self_rag, 'recover_evidence', AsyncMock(return_value={'evidence_lookup': changed}))
    update = await self_rag.recover_evidence_node(state)
    assert update['verification'].verdicts[0].input_digest == 'old-source-check'
    state.update(update, completed_node='recover_evidence')
    assert (await controller.control_research_node(state))['next_node'] == expected_action


async def test_rebinding_counts_as_a_repair_even_when_old_report_was_invalidated(monkeypatch):
    hit = Hit(1, 'p', 'Method', 'Correct source.', 1)
    monkeypatch.setattr(synthesize, '_generate_answer', AsyncMock(return_value=('Supported fact [p#1].', 'stop')))
    result = await synthesize.synthesize_node({'question': 'Explain P', 'evidence': [hit],
        'verification': None, 'verification_recovery_requires_rewrite': True, 'tool_iters': 0})
    assert result['tool_iters'] == 1 and not result['verification_recovery_requires_rewrite']
