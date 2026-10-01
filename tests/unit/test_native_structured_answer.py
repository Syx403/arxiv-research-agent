"""Exercise generation -> source checks -> delivery -> offline grading without APIs."""
import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scripts import eval_agent
from src.agent.results import present_state
from src.agent.runtime import AgentRequest, AgentRuntime
from src.core.answer_contract import RequestedField
from src.core.run_context import RunContext, current_run, use_run_context
from src.core.structured_answer import bind_generated_fields, render_entry
from src.core.types import ChatResponse, CitationVerdict, EvidenceChunk, VerificationReport
from src.eval.agent_metrics import score_turn
from src.eval.answer_metrics import delivered_observations
from src.graph.builder import _initial_state
from src.graph.nodes import synthesize
from src.graph.nodes.finalize import finalize_node
from src.graph.nodes.self_rag import self_rag_node
from src.llm.budget import RequestBudget
from src.retrieval.index.types import Hit


PID = "arxiv:2101.12345v1"
FIELD = RequestedField(name="sensor_count", kind="number", description="How many sensors are used?", paper_ids=(PID,))
OTHER = RequestedField(name="active", kind="boolean", description="Are the sensors active?", paper_ids=(PID,))
HIT = Hit(chunk_id=1, paper_id=PID, section="Methods", text="Three sensors are active.", score=1)


def generated(name="sensor_count", value=3, **edits):
    return {"name": name, "value": value, "basis": "paper_fact", "explanation": "The method specifies this count.",
            "qualifiers": "", "evidence_indices": [0], **edits}


def fixture_state(fields=(FIELD,), rows=None):
    entries, errors = bind_generated_fields({"fields": rows or [generated()]}, fields, [HIT])
    assert not errors
    from src.core.structured_answer import TABLE_HEADER
    answer = TABLE_HEADER + "\n" + "\n".join(e["claim"] for e in entries)
    citations = synthesize._parse_citations(answer)
    report = VerificationReport(passed=True, verdicts=[CitationVerdict(
        citation=c, supports=True, status="ok", scope_status="requested", rationale="Fixture source check") for c in citations])
    return {"question": "Count the sensors.", "answer": answer, "answer_fields": entries,
            "response_fields": [f.model_dump(mode="json") for f in fields],
            "citations": citations, "verification": report, "result_kind": "evidence",
            "status": "complete", "evidence": [HIT],
            "evidence_lookup": {1: EvidenceChunk(paper_id=PID, chunk_id=1, text=HIT.text)}}


def spec():
    return {"question": "Count the sensors.", "response_fields": [FIELD.model_dump(mode="json")],
            "expected_route": "read", "expected_status": "complete", "programmatic_checks": [],
            "review_only": ["Source review of explanation"],
            "gold": {"required_version_ids": [PID], "required_facts": {"count": 3},
                     "fact_rules": {"count": {"all_of": [{"paper_id": PID, "names": ["sensor_count"],
                                                         "kind": "number", "expected": 3}]}}}}


def test_old_checkpoint_without_optional_domain_still_projects():
    state = fixture_state()
    state["answer_fields"][0].pop("allowed_values", None)
    output = present_state(state, response_fields=(FIELD,))
    assert output["structured_facts"][0]["value"] == 3
    assert delivered_observations(output)[2] == []


@pytest.mark.parametrize("row", [generated(value=True), generated(evidence_indices=[True]),
    generated(evidence_indices=[1]), generated(evidence_indices=[0, 0]),
    generated(name="unknown"), generated(qualifiers=[]), generated(value="3")])
def test_invalid_types_and_source_indices_do_not_become_answer_fields(row):
    entries, errors = bind_generated_fields({"fields": [row]}, (FIELD,), [HIT])
    assert not entries and errors


def test_duplicate_fields_wrong_versions_and_gold_keys_rejected():
    assert not bind_generated_fields({"fields": [generated(), generated()]}, (FIELD,), [HIT])[0]
    wrong = replace(HIT, paper_id=PID.replace("v1", "v2"))
    assert not bind_generated_fields({"fields": [generated()]}, (FIELD,), [wrong])[0]
    with pytest.raises(ValueError):
        RequestedField.model_validate({**FIELD.model_dump(), "expected": 3})
    with pytest.raises(ValueError):
        AgentRequest(question="Count?", response_fields=[FIELD, FIELD])
    with pytest.raises(ValueError):
        AgentRequest(question="Count?", response_fields=[FIELD], include_typed_observations=True)


async def test_synthesis_render_verify_finalize_grade_no_extraction_call(monkeypatch):
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({"fields": [generated()]}), model="fixture", finish_reason="stop")
    monkeypatch.setattr(synthesize, "get_chat_client", lambda _: client)
    with use_run_context(RunContext(response_fields=(FIELD,))):
        update = await synthesize.synthesize_node({"question": "Count?", "evidence": [HIT]})
    assert client.chat.await_count == 1
    assert client.chat.call_args.kwargs["stage"] == "synthesis"
    assert "thinking" not in client.chat.call_args.kwargs  # configured profile controls reasoning
    assert update["citations"][0].claim_text == update["answer_fields"][0]["claim"]
    state = {**fixture_state(), **update}
    state["verification"] = VerificationReport(passed=True, verdicts=[CitationVerdict(
        citation=c, supports=True, rationale="checked", scope_status="requested") for c in update["citations"]])
    state.update(await finalize_node(state))
    output = present_state(state, include_typed_observations=False, response_fields=(FIELD,))
    facts, order, errors = delivered_observations(output)
    assert len(facts) == 1 and facts[0]["value"] == 3 and order == [] and errors == []
    assert output["answer_contract"]["version"] == "ara-answer-3"
    assert output["checks"][0]["fields_status"] == "unavailable"  # no verifier extraction
    score = score_turn(spec(), {"route": "read", "output": output})
    assert score["program_pass"] and score["fact_correct_count"] == 1
    assert score["task_pass"] is None and score["content_review"] == "pending"
    # Machine values must be identical to the visible answer; a correct hidden
    # field cannot compensate for an incorrect visible field.
    output["structured_facts"][0]["value"] = 4
    assert delivered_observations(output)[0] == []


async def test_rejected_rows_are_withheld_and_missing_fields_count_as_misses():
    state = fixture_state()
    state["verification"].verdicts[0].supports = False
    state["verification"].passed = False
    state.update(await finalize_node(state))
    output = present_state(state, response_fields=(FIELD,))
    assert output["structured_facts"] == [] and output["native_fields"] == []
    assert output["answer_contract"]["missing_fields"] == ["sensor_count"]
    score = score_turn(spec(), {"route": "read", "output": output})
    assert score["fact_graded_count"] == 1 and score["fact_correct_count"] == 0
    # A total runtime failure cannot remove this required fact from n.
    failed = score_turn(spec(), {"route": "unknown", "output": {"status": "incomplete"}})
    assert failed["fact_graded_count"] == 1 and failed["fact_correct_count"] == 0


async def test_omitted_requested_field_makes_delivery_partial():
    state = fixture_state(fields=(FIELD, OTHER))
    state.update(await finalize_node(state))
    assert state["status"] == "partial"
    assert any("active" in x for x in state["missing_aspects"])
    output = present_state(state, response_fields=(FIELD, OTHER))
    assert output["answer_contract"]["missing_fields"] == ["active"]
    assert len(output["structured_facts"]) == 1


async def test_checkpoint_best_draft_carries_matching_fields_and_new_turn_clears(monkeypatch):
    state = fixture_state()
    verifier = AsyncMock(return_value=state["verification"])
    monkeypatch.setattr("src.graph.nodes.self_rag.verify_citations", verifier)
    state.update(await self_rag_node(state))
    assert state["best_verified"]["answer_fields"] == state["answer_fields"]
    state.update(answer="Bad later draft.", answer_fields=[], citations=[],
                 verification=VerificationReport(passed=False, verdicts=[]))
    state.update(await finalize_node(state))
    output = present_state(state, response_fields=(FIELD,))
    assert state["delivery_from_prior_draft"] and output["structured_facts"][0]["value"] == 3
    with use_run_context(RunContext()):
        fresh = _initial_state("New topic", "a")
    assert fresh["answer_fields"] == fresh["response_fields"] == []


async def test_request_context_isolated_and_only_public_contract_sent(tmp_path):
    seen = []
    async def runner(question, **kwargs):
        await asyncio.sleep(0)
        seen.append((question, current_run().response_fields))
        return {"answer": "No evidence.", "status": "incomplete"}
    runtime = AgentRuntime(tmp_path, RequestBudget(0.01, path=tmp_path / "budget.json"), runner=runner)
    results = await asyncio.gather(runtime.query(AgentRequest(question="Structured", response_fields=(FIELD,))),
                                   runtime.query(AgentRequest(question="Natural")))
    assert ("Natural", ()) in seen
    request = next(q for q, fields in seen if fields)
    assert "How many sensors" in request and "sensor_count" in request and PID in request
    assert '"expected"' not in request and '"gold"' not in request and '"value": 3' not in request
    assert results[0].question == "Structured"
    assert results[1].output["answer_contract"]["status"] == "not_requested"


def test_public_ranking_uses_rendered_value_not_source_order():
    other = "arxiv:2201.12345v1"
    field = RequestedField(name="order", kind="paper_order", description="Order by stated fit.", paper_ids=(PID, other))
    evidence = [HIT, replace(HIT, paper_id=other, chunk_id=2)]
    entries, errors = bind_generated_fields({"fields": [generated("order", [other, PID], evidence_indices=[0, 1],
                                                               basis="engineering_inference")]}, (field,), evidence)
    assert not errors and entries[0]["value"] == [other, PID]
    assert render_entry(entries[0]).index(other) < render_entry(entries[0]).index(PID)
    assert not bind_generated_fields({"fields": [generated("order", [PID, PID], evidence_indices=[0, 1])]}, (field,), evidence)[0]


@pytest.mark.parametrize("content,finish", [("{broken", "stop"), ('{"fields":[]}', "length"), ('{"fields":[]}', "stop")])
async def test_unusable_structured_generation_is_bounded_and_clears_stale_fields(monkeypatch, content, finish):
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=content, model="fixture", finish_reason=finish)
    monkeypatch.setattr(synthesize, "get_chat_client", lambda _: client)
    with use_run_context(RunContext(response_fields=(FIELD,))):
        update = await synthesize.synthesize_node(fixture_state())
    assert client.chat.await_count == 1
    assert update["synthesis_format_degraded"] and not update["citations"] and not update["answer_fields"]


def test_split_dataset_contracts_and_unmodified_natural_questions():
    root = Path("docs/eval_design")
    old = eval_agent.load_dataset(root / "review_cases_v4.json")
    natural = eval_agent.load_dataset(root / "natural_cases_v5.json")
    structured = eval_agent.load_dataset(root / "structured_cases_v5.json")
    assert [t["question"] for c in natural["cases"] for t in c["turns"]] == [t["question"] for c in old["cases"] for t in c["turns"]]
    assert all(not t.get("response_fields") and not t["gold"].get("fact_rules") for c in natural["cases"] for t in c["turns"])
    assert sum(len(t["response_fields"]) for c in structured["cases"] for t in c["turns"]) == 23
    assert sum(len(t["gold"]["required_facts"]) for c in structured["cases"] for t in c["turns"]) == 21
    selection = structured["cases"][3]["turns"][0]["gold"]["fact_rules"]["selection"]
    assert selection["all_of"][0]["kind"] == "boolean"  # no AST implementation overconstraint
    public = json.dumps([t["response_fields"] for c in structured["cases"] for t in c["turns"]])
    assert "expected" not in public and "aliases" not in public
    altered = copy.deepcopy(spec())
    altered["response_fields"][0]["kind"] = "text"
    # API shape validation does not consume gold; dataset validation separately
    # rejects mismatched private rules before a paid execution begins.
    assert AgentRequest(question=altered["question"], response_fields=altered["response_fields"])
