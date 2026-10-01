"""Contract, provenance and grading tests use no provider or database calls."""
import copy
import json

import pytest

from src.agent.results import present_state
from src.core.answer_contract import bind_fields, bind_model_fields, claim_papers
from src.core.types import EvidenceChunk, VerificationReport
from src.eval.agent_metrics import score_turn, summarize
from src.eval.answer_metrics import _equivalent, delivered_observations, validate_fact_rules
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval.self_rag.verifier import _parse_verdict


PID = "arxiv:2101.12345v1"


def fact(value=3, **changes):
    return {"name": "sensor_count", "kind": "number", "value": value,
            "paper_id": PID, "quote": "There are 3 sensors", "basis": "paper_fact", **changes}


def verdict_payload(fields):
    return {"claim_kind": "paper_fact", "scope_status": "requested", "rationale": "Evidence states the count.",
            "supports": True, "all_assertions_supported": True, "no_unstated_assumptions": True,
            "all_citations_contribute": True, "result_fields": fields}


def projected(*, facts=None, rankings=None, answer=None):
    answer = answer or f"There are 3 sensors [{PID}#1]."
    citations = _parse_citations(answer)
    papers = claim_papers(answer)
    fields = {"facts": [{**{k: v for k, v in f.items() if k not in {"paper_id", "quote"}},
                         "source_index": papers.index(f["paper_id"])} for f in ([fact()] if facts is None else facts)],
              "rankings": [{"source_indices": [papers.index(p) for p in r["paper_ids"]]} for r in (rankings or [])]}
    checks = [_parse_verdict(json.dumps(verdict_payload(fields)), c) for c in citations]
    state = {"result_kind": "evidence", "answer": answer, "status": "complete",
             "citations": citations, "draft_verification": VerificationReport(verdicts=checks, passed=True),
             "evidence_lookup": {c.chunk_id: EvidenceChunk(paper_id=c.paper_id, chunk_id=c.chunk_id,
                 text="Original sensor specification.", source_spans=[(0, 30)]) for c in citations}}
    return present_state(state), state


def spec():
    return {"expected_route": "read", "expected_status": "complete", "programmatic_checks": [],
            "review_only": [], "gold": {"required_facts": {"sensors": 3}, "required_version_ids": [PID],
             "fact_rules": {"sensors": {"all_of": [{"paper_id": PID, "names": ["sensor_count"],
                                                     "kind": "number", "expected": 3}]}}}}


@pytest.mark.parametrize("change", [
    {"kind": "number", "value": True}, {"kind": "boolean", "value": 0},
    {"kind": "sequence", "value": ["a", "a"]}, {"kind": "relations", "value": [["a"]]},
    {"kind": "number", "value": float("inf")},
    {"paper_id": "arxiv:2101.12345v2"}, {"quote": "Not actually in the delivered claim"},
])
def test_bad_values_or_unbound_fields_are_rejected(change):
    with pytest.raises(ValueError):
        bind_fields({"facts": [fact(**change)], "rankings": []}, f"There are 3 sensors [{PID}#1].")


def test_malformed_annotation_is_not_a_false_factual_verdict():
    c = _parse_citations(f"There are 3 sensors [{PID}#1].")[0]
    fields = {"facts": [{"name": "sensor_count", "kind": "number", "value": 3,
                         "source_index": 0, "basis": "paper_fact", "qualifiers": []}], "rankings": []}
    v = _parse_verdict(json.dumps(verdict_payload(fields)), c)
    assert v.supports and v.fields_status == "invalid" and v.result_fields is None
    assert "qualifiers:string_type" in v.fields_error


def test_program_binds_model_observations_and_rejects_guessed_source_or_quote():
    claim = f"There are 3 sensors [{PID}#1]."
    raw = {"name": "sensor_count", "kind": "number", "value": 3, "source_index": 0,
           "basis": "paper_fact", "qualifiers": ""}
    parsed = bind_model_fields({"facts": [raw], "rankings": []}, claim)
    assert parsed["facts"][0]["quote"] == claim
    assert parsed["facts"][0]["paper_id"] == PID
    for edit in ({"source_index": 1}, {"source_index": False}, {"quote": "source text"},
                 {"paper_id": PID + "#1"}):
        with pytest.raises(ValueError):
            bind_model_fields({"facts": [{**raw, **edit}], "rankings": []}, claim)


def test_projection_and_grader_bind_exact_delivered_answer_and_sources():
    output, _ = projected()
    observed, order, errors = delivered_observations(output)
    assert len(observed) == 1 and order == [] and errors == []
    score = score_turn(spec(), {"route": "read", "output": output})
    assert score["fact_checks"] == {"sensors": True}
    assert score["fact_graded_count"] == score["fact_observable_count"] == 1
    assert score["program_pass"]
    # Field equality still does not constitute an independent source review.
    assert "semantic extraction" in output["answer_contract"]["origin"]


def test_rejected_prior_draft_observations_cannot_leak():
    output, state = projected()
    old = state["draft_verification"].verdicts[0]
    new_answer = f"Only 2 sensors remain [{PID}#1]."
    state.update(answer=new_answer, citations=_parse_citations(new_answer))
    output = present_state(state)
    assert output["structured_facts"] == []
    assert output["answer_contract"]["status"] == "unavailable"
    # Even a matching draft text is withheld when one required check failed.
    state.update(answer=old.citation.claim_text, citations=[old.citation],
                 draft_verification=VerificationReport(verdicts=[old.model_copy(update={"supports": False})], passed=False))
    assert present_state(state)["structured_facts"] == []


@pytest.mark.parametrize("tamper", ["answer", "span", "source", "check"])
def test_grader_rechecks_bindings_instead_of_trusting_fields(tamper):
    output, _ = projected()
    if tamper == "answer":
        output["answer"] = "Unrelated reply"
    elif tamper == "span":
        output["structured_facts"][0]["answer_span"] = [0, 4]
    elif tamper == "source":
        output["sources"][0]["paper_id"] = "arxiv:2101.12345v2"
    else:
        output["checks"][0]["supports"] = False
    score = score_turn(spec(), {"route": "read", "output": output})
    assert score["fact_graded_count"] == 1 and score["fact_correct_count"] == 0
    assert not score["checks"]["result_field_bindings"]


def test_missing_and_conflicting_observations_count_as_misses():
    output, _ = projected(facts=[])
    absent = score_turn(spec(), {"route": "read", "output": output})
    assert absent["fact_graded_count"] == 1 and absent["fact_correct_count"] == 0
    assert absent["fact_details"]["sensors"]["clauses"][0]["reason"] == "missing_observation"
    output, _ = projected(facts=[fact(), fact(2)])
    conflict = score_turn(spec(), {"route": "read", "output": output})
    assert conflict["fact_observable_count"] == 1 and not conflict["fact_checks"]["sensors"]


def test_reading_rank_comes_from_explicit_order_not_source_card_order():
    other = "arxiv:2201.12345v1"
    answer = f"For this workload, method B is preferred to A [{PID}#1][{other}#2]."
    output, _ = projected(facts=[], answer=answer, rankings=[{"paper_ids": [other, PID],
                         "quote": "method B is preferred to A"}])
    output["paper_results"] = [{"paper_id": PID, "fit": "user_selected"}, {"paper_id": other, "fit": "user_selected"}]
    case = spec()
    case["gold"] = {"required_primary_id": other, "relevance_grades": {"2201.12345": 3, "2101.12345": 1}}
    score = score_turn(case, {"route": "read", "output": output})
    assert score["checks"]["primary_selection"] and score["ndcg_at_3"] == 1
    assert score["ranked_ids"] == [other, PID]
    assert score["ranking_basis"] == "delivered_explicit_recommendation"


def test_missing_rank_is_zero_for_new_contract_but_legacy_is_ungraded():
    output, _ = projected()
    case = spec()
    case["gold"] = {"required_primary_id": PID, "relevance_grades": {"2101.12345": 3}}
    score = score_turn(case, {"route": "read", "output": output})
    assert not score["checks"]["primary_selection"] and score["ndcg_at_3"] == 0
    del output["answer_contract"]
    legacy = score_turn(case, {"route": "read", "output": output})
    assert legacy["ndcg_at_3"] is None and "primary_selection" in legacy["pending_checks"]


def test_typed_comparison_preserves_order_direction_symbols_and_boolean_types():
    assert not _equivalent(1, True, {"kind": "boolean"})
    assert not _equivalent(["b", "a"], ["a", "b"], {"kind": "sequence"})
    assert _equivalent(["b", "a"], ["a", "b"], {"kind": "set"})
    assert not _equivalent([["B", "A"]], [["A", "B"]], {"kind": "relations"})
    assert not _equivalent([["a", "B"]], [["A", "B"]], {"kind": "relations"})
    assert _equivalent([["B", "A"]], [["A", "B"]], {"kind": "relations", "unordered_members": True})
    assert not _equivalent("x > 1", "x < 1", {"kind": "text"})
    assert _equivalent("File-level", "file_level", {"kind": "text"})


def test_absent_rule_remains_ungraded_and_rule_validation_rejects_ambiguous_aliases():
    output, _ = projected()
    case = spec()
    case["gold"].pop("fact_rules")
    score = score_turn(case, {"route": "read", "output": output})
    assert score["fact_graded_count"] == 0 and score["task_pass"] is None
    gold = spec()["gold"]
    gold["fact_rules"]["sensors"]["all_of"][0]["aliases"] = {"a": ["same"], "b": ["same"]}
    with pytest.raises(ValueError, match="Ambiguous"):
        validate_fact_rules(gold)


def test_summary_reports_honest_denominators():
    records = []
    for fields in ([fact()], []):
        output, _ = projected(facts=fields)
        result = {"route": "read", "output": output, "elapsed_s": 1,
                  "cost": {"estimated_usd": 0, "uncertain_usd": 0}}
        records.append({"profile": "semantic_low", "result": result, "score": score_turn(spec(), result)})
    legacy = copy.deepcopy(records[0])
    del legacy["result"]["output"]["answer_contract"]
    legacy["score"] = score_turn(spec(), legacy["result"])
    row = summarize([*records, legacy], {"semantic_low": 4})["profiles"]["semantic_low"]
    assert row["fact_accuracy"] == {"value": 0.5, "n": 2}
    assert row["fact_grading_coverage"] == {"value": 2/3, "n": 3}
    assert row["fact_observable"] == 1 and row["not_run_or_interrupted"] == 1


def test_frozen_v4_does_not_change_queries_or_gold_facts():
    from pathlib import Path
    from scripts.eval_agent import load_dataset
    old = load_dataset(Path("docs/eval_design/review_cases_v3.json"))
    new = load_dataset(Path("docs/eval_design/review_cases_v4.json"))
    for a, b in zip(old["cases"], new["cases"], strict=True):
        assert a["id"] == b["id"]
        for x, y in zip(a["turns"], b["turns"], strict=True):
            assert x["question"] == y["question"]
            assert x["gold"].get("required_facts") == y["gold"].get("required_facts")
            for name in y["gold"].get("required_facts", {}):
                assert name in y["gold"]["fact_rules"]


@pytest.mark.asyncio
async def test_experimental_work_is_opt_in_and_verdict_cache_does_not_cross_modes(monkeypatch):
    from src.core.run_context import RunContext, use_run_context
    from src.core.types import ChatResponse
    from src.retrieval.self_rag import verifier
    calls = []

    class Client:
        async def chat(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return ChatResponse(model="fake", finish_reason="stop", content=json.dumps(verdict_payload(None)))

    monkeypatch.setattr(verifier, "get_chat_client", lambda role: Client())
    answer = f"There are 3 sensors [{PID}#1]."
    citations = _parse_citations(answer)
    lookup = {1: EvidenceChunk(paper_id=PID, chunk_id=1, text="There are 3 sensors.")}
    with use_run_context(RunContext()):
        initial = await verifier.verify_citations(answer, citations, lookup)
    with use_run_context(RunContext(include_typed_observations=True)):
        await verifier.verify_citations(answer, citations, lookup, cached_verdicts=initial.verdicts)
    assert len(calls) == 2
    assert "result_fields" not in calls[0][0][0].content and calls[0][1]["max_tokens"] == 900
    assert "result_fields" in calls[1][0][0].content and calls[1][1]["max_tokens"] == 1600
    assert "Numbered paper identities" in calls[1][0][1].content


@pytest.mark.asyncio
async def test_runtime_opt_in_is_isolated_and_disabled_fields_are_ungraded(tmp_path):
    import asyncio
    from src.agent.runtime import AgentRequest, AgentRuntime
    from src.core.run_context import current_run
    from src.llm.budget import RequestBudget

    async def runner(question, **kwargs):
        enabled = current_run().include_typed_observations
        await asyncio.sleep(0)
        assert current_run().include_typed_observations == enabled
        _, state = projected()
        return state

    runtime = AgentRuntime(tmp_path, RequestBudget(.1, path=tmp_path/'budget.json'), runner=runner)
    disabled, enabled = await asyncio.gather(*(
        runtime.query(AgentRequest(question="How many sensors?", include_typed_observations=value))
        for value in (False, True)
    ))
    assert disabled.output["answer_contract"]["status"] == "not_requested"
    assert enabled.output["answer_contract"]["status"] == "complete"
    assert score_turn(spec(), disabled.model_dump())["fact_graded_count"] == 0
    assert score_turn(spec(), enabled.model_dump())["fact_correct_count"] == 1
    assert not current_run().include_typed_observations
