"""Behavioral checks for direct matches, bounded audits and semantic verification."""
import json
from unittest.mock import AsyncMock

import pytest

from src.core.types import ChatResponse, CitationVerdict, EvidenceChunk, VerificationReport
from src.graph.delivery import supported_text
from src.graph.controller import control_research_node
from src.graph.edges import route_after_verify
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval import paper_search as search
from src.retrieval.index.types import Hit
from src.retrieval.self_rag import claim_audit, verifier
from tests.unit.test_research_controller import state, row


@pytest.mark.parametrize("query,bounds", [
    ("找两三篇长期记忆论文", (2, 3)), ("找2-3篇论文", (2, 3)),
    ("找一两篇工具编排论文", (1, 2)), ("只要一二篇论文", (1, 2)),
    ("先看两篇即可", (2, 2)),
    ("Find two or three papers", (2, 3)), ("找三篇论文", (3, 3)),
    ("2026年的3B模型有哪些论文", None),
])
def test_requested_range_preserves_minimum_and_upper_bound(query, bounds):
    assert search.requested_result_bounds(query) == bounds


async def test_related_papers_do_not_satisfy_requested_quantity():
    q = search.Query(terms=["agent", "memory"])
    s = state(paper_results=[{"paper_id": "one", "fit": "direct"},
                             {"paper_id": "two", "fit": "partial"},
                             {"paper_id": "three", "fit": "partial"}],
              pending_paper_queries=[search.Query(terms=["persistent", "memory"]).model_dump()])
    s["paper_plan"].update(min_results=2, max_results=3)
    s["paper_search"].update(coverage="sufficient", queries=[row(q, 10)])
    assert (await control_research_node(s))["next_node"] == "search_papers"
    s["paper_results"][1]["fit"] = "direct"
    assert (await control_research_node(s))["next_node"] == "present_papers"


def test_conceptual_direct_match_precedes_newer_related_reference():
    proposal = {"paper_id": "concept", "fit": "direct", "evidence_basis": "conceptual",
                "date_confirmed": True, "published_at": "2024-01-01"}
    related = {"paper_id": "related", "fit": "partial", "evidence_basis": "empirical",
               "date_confirmed": True, "published_at": "2026-01-01"}
    assert search.select_results([related, proposal], max_results=1, prefer_recent=True) == [proposal, related]


def hit(cid, section="Method", text="This method controls loads based on previous batches."):
    return Hit(cid, "arxiv:2408.15664v1", section, text, 1, title="A method paper")


def wire_audit(monkeypatch, originals, outcome="not_established", evidence=None):
    monkeypatch.setattr(claim_audit, "load_scope", AsyncMock(return_value=originals))
    payload = {"outcome": outcome, "rationale": "Only bounded empirical results are established.",
               "evidence": evidence if evidence is not None else [{"chunk_id": originals[0].chunk_id, "sentence_ids": [0]}]}
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps(payload), model="fixture", finish_reason="stop")
    monkeypatch.setattr(claim_audit, "get_chat_client", lambda role: client)
    return client


async def test_a_scoped_negative_can_complete_after_actual_material_scan(monkeypatch):
    wire_audit(monkeypatch, [hit(1), hit(2, "Conclusion")])
    review, evidence = await claim_audit.audit_claim("Does this establish a universal guarantee?", ["arxiv:2408.15664v1"])
    assert review["status"] == "complete"
    assert review["outcome"] == "not_established"
    assert review["all_available_material_checked"]
    assert len(review["checked_sources"]) == 2 and evidence


async def test_a_truncated_scan_cannot_complete_a_negative_claim(monkeypatch):
    wire_audit(monkeypatch, [hit(1), hit(2, "Appendix", "A formal guarantee follows under these assumptions.")])
    monkeypatch.setattr(claim_audit, "MAX_AUDIT_CHUNKS", 1)
    review, _ = await claim_audit.audit_claim("Is a guarantee established?", ["arxiv:2408.15664v1"])
    assert review["status"] == "incomplete"
    assert not review["all_available_material_checked"]


async def test_targeted_followup_includes_later_proof_and_exact_source(monkeypatch):
    proof = hit(2, "Appendix: Proof", "The theorem holds for any input distribution under assumption A.")
    client = wire_audit(monkeypatch, [hit(1), proof], outcome="supported",
                        evidence=[{"chunk_id": 2, "sentence_ids": [0]}])
    review, evidence = await claim_audit.audit_claim("Is the guarantee proved?", ["arxiv:2408.15664v1"])
    assert "The theorem holds" in client.chat.call_args.args[0][1].content
    assert review["status"] == "complete" and review["outcome"] == "supported"
    assert evidence[0].chunk_id == 2 and evidence[0].text == proof.text


@pytest.mark.parametrize("evidence", [[{"chunk_id": 999, "sentence_ids": [0]}],
                                      [{"chunk_id": 1, "sentence_ids": [999]}]])
async def test_audit_rejects_fabricated_source_bindings(monkeypatch, evidence):
    client = wire_audit(monkeypatch, [hit(1)], evidence=evidence)
    review, hits = await claim_audit.audit_claim("Check", ["arxiv:2408.15664v1"])
    assert review["status"] == "incomplete" and not hits
    assert client.chat.await_count == 2


async def test_unavailable_paper_never_looks_like_negative_evidence(monkeypatch):
    loader = AsyncMock(return_value=[])
    client = AsyncMock()
    monkeypatch.setattr(claim_audit, "load_scope", loader)
    monkeypatch.setattr(claim_audit, "get_chat_client", lambda role: client)
    review, _ = await claim_audit.audit_claim("Check", ["arxiv:2408.15664v1"])
    assert review["status"] == "incomplete" and review["outcome"] == "insufficient"
    client.chat.assert_not_called()


@pytest.mark.parametrize("prefix", ["证据边界：", "当前证据显示：", ""])
async def test_evidence_limit_type_comes_from_semantic_verdict_not_prefix(monkeypatch, prefix):
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({
        "claim_kind": "evidence_limit", "scope_status": "requested", "supports": True, "rationale": "Scoped to the cited excerpts.",
        "all_assertions_supported": True, "no_unstated_assumptions": True, "all_citations_contribute": True,
    }), model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda role: client)
    text = prefix + "这些摘录不能支持普适最优保证 [arxiv:2408.15664v1#1]。"
    report = await verifier.verify_citations(text, _parse_citations(text), {
        1: EvidenceChunk(paper_id="arxiv:2408.15664v1", chunk_id=1, text="We evaluated two finite datasets."),
    })
    assert report.passed and report.verdicts[0].claim_kind == "evidence_limit"
    assert "Claim kind: paper_fact" not in client.chat.call_args.args[0][1].content


def test_no_new_draft_or_evidence_stops_an_unproductive_repair():
    assert route_after_verify({"verification_stalled": True, "tool_iters": 1,
                               "verification": VerificationReport(verdicts=[], passed=False)}) == "finalize"


def test_supported_but_unrequested_or_duplicate_sentences_are_not_delivered():
    answer = ("It routes to one expert [p#1].\n\nIt uses a single expert [p#1].\n\n"
              "An optional rerouting variant is explored [p#2].")
    citations = _parse_citations(answer)
    report = VerificationReport(passed=True, verdicts=[
        CitationVerdict(citation=c, supports=True, rationale="Checked", scope_status=scope)
        for c, scope in zip(citations, ["requested", "duplicate", "unrequested"], strict=True)
    ])
    assert supported_text(answer, report) == "It routes to one expert [p#1]."


async def test_whole_answer_scope_omits_repetition_before_fact_checks(monkeypatch):
    answer = "It stores code [p#1].\n\nStored skills are code [p#1].\n\nIt uses a vector query [p#2]."
    client = AsyncMock()
    scope = {"items": [
        {"id": 0, "scope_status": "requested", "reason": "Storage"},
        {"id": 1, "scope_status": "duplicate", "reason": "Same storage fact", "duplicate_of": 0},
        {"id": 2, "scope_status": "requested", "reason": "Retrieval"},
    ], "coverage": [{"requirement_id": 0, "statement_ids": [0, 2]}]}
    client.chat.return_value = ChatResponse(content=json.dumps(scope), model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda role: client)
    factual = AsyncMock(side_effect=lambda *args, **kwargs: CitationVerdict(
        citation=args[3], supports=True, rationale="Verified source"))
    monkeypatch.setattr(verifier, "_verify_one_citation", factual)
    report = await verifier.verify_citations(answer, _parse_citations(answer), {},
                                            discovery_context={"research_question": "How are skills stored and retrieved?"})
    assert report.passed and factual.await_count == 2 and client.chat.await_count == 1
    delivered = supported_text(answer, report)
    assert "Stored skills" not in delivered and "vector query" in delivered


@pytest.mark.parametrize("duplicate_of", [1, 2, 99])
async def test_scope_cannot_drop_duplicate_without_earlier_retained_explanation(monkeypatch, duplicate_of):
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({"items": [
        {"id": 0, "scope_status": "requested", "reason": "Core"},
        {"id": 1, "scope_status": "duplicate", "reason": "Invalid target", "duplicate_of": duplicate_of},
    ], "coverage": [{"requirement_id": 0, "statement_ids": [0]}]}), model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda role: client)
    assert await verifier._answer_scope("Question", ["A", "B"]) is None


async def test_irrelevant_supplement_is_not_a_missing_user_requirement(monkeypatch):
    from src.graph.nodes.finalize import finalize_node
    from src.core.types import RouteDecision, SubQResult

    answer = "It routes to one expert [p#1].\n\nAn optional unrelated cost claim [p#2]."
    citations = _parse_citations(answer)
    report = VerificationReport(passed=True, verdicts=[
        CitationVerdict(citation=citations[0], supports=True, rationale="Core mechanism"),
        CitationVerdict(citation=citations[1], supports=False, rationale="Not needed", scope_status="unrequested"),
    ])
    result = await finalize_node({"answer": answer, "verification": report,
        "subq_results": {0: SubQResult(subq_index=0, sufficient=True, route_decision=RouteDecision())}})
    assert result["status"] == "complete"
    assert "unrelated" not in result["answer"]


@pytest.mark.parametrize("scope_status,expect_pass", [("unrequested", True), ("requested", False)])
async def test_uncited_scope_does_not_confuse_extra_caveat_with_missing_answer(monkeypatch, scope_status, expect_pass):
    answer = "Engineering inference: use the fixed pipeline for traceability [p#1]. Uncited prose."
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({"items": [
        {"id": 0, "scope_status": "requested", "reason": "Conditional recommendation"},
        {"id": 1, "scope_status": scope_status, "reason": "Fixture scope decision"},
    ], "coverage": [{"requirement_id": 0, "statement_ids": [0]}]}), model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda role: client)
    monkeypatch.setattr(verifier, "_verify_one_citation", AsyncMock(side_effect=lambda *args, **kw:
        CitationVerdict(citation=args[3], supports=True, rationale="Bound to source")))
    report = await verifier.verify_citations(answer, _parse_citations(answer), {},
                                             discovery_context={"research_question": "Recommend a controllable workflow"})
    assert report.passed is expect_pass
    assert bool(report.uncited_claims) is not expect_pass
    assert bool(report.omitted_uncited_claims) is expect_pass
    from src.graph.nodes.finalize import finalize_node
    result = await finalize_node({"answer": answer, "verification": report})
    assert result["status"] == ("complete" if expect_pass else "partial")
    assert "Uncited prose." not in result["verified_answer"]


async def test_uncited_duplicate_requires_a_supported_retained_source(monkeypatch):
    answer = "A requested claim [p#1]. A duplicate claim."
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({"items": [
        {"id": 0, "scope_status": "requested", "reason": "Core claim"},
        {"id": 1, "scope_status": "duplicate", "reason": "Same claim", "duplicate_of": 0},
    ], "coverage": [{"requirement_id": 0, "statement_ids": [0]}]}), model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda role: client)
    monkeypatch.setattr(verifier, "_verify_one_citation", AsyncMock(side_effect=lambda *args, **kw:
        CitationVerdict(citation=args[3], supports=False, rationale="Unsupported")))
    report = await verifier.verify_citations(answer, _parse_citations(answer), {},
                                             discovery_context={"research_question": "Explain"})
    assert not report.passed and report.uncited_claims == ["A duplicate claim."]
    assert not report.omitted_uncited_claims


async def test_passed_repair_is_not_replaced_by_longer_prior_draft():
    from src.graph.nodes.finalize import finalize_node
    answer = "Concise checked conclusion [p#1]."
    report = VerificationReport(passed=True, verdicts=[
        CitationVerdict(citation=c, supports=True, rationale="Checked") for c in _parse_citations(answer)])
    result = await finalize_node({"question": "Explain", "answer": answer, "verification": report,
        "best_verified": {"question": "Explain", "text": answer + " Extra detail [p#2]."}})
    assert result["status"] == "complete" and not result["delivery_from_prior_draft"]
    assert result["answer"] == answer


def test_identical_verified_recommendation_is_delivered_once():
    sentence = "Engineering inference: prefer the fixed pipeline [p#1]."
    answer = sentence + "\n\n" + sentence
    report = VerificationReport(passed=True, verdicts=[
        CitationVerdict(citation=c, supports=True, rationale="Checked") for c in _parse_citations(answer)])
    assert supported_text(answer, report) == sentence


async def test_supported_rule_without_requested_result_is_not_complete(monkeypatch):
    answer = "Each group shares one key head [p#1]. Each group shares one value head [p#1]."
    client = AsyncMock()
    client.chat.return_value = ChatResponse(content=json.dumps({
        "items": [{"id": i, "scope_status": "requested", "reason": "Supported general rule"} for i in range(2)],
        "coverage": [{"requirement_id": 0, "statement_ids": []}],
    }), model="fixture", finish_reason="stop")
    monkeypatch.setattr(verifier, "get_chat_client", lambda role: client)
    monkeypatch.setattr(verifier, "_verify_one_citation", AsyncMock(side_effect=lambda *args, **kw:
        CitationVerdict(citation=args[3], supports=True, rationale="Source supports the rule")))
    report = await verifier.verify_citations(answer, _parse_citations(answer), {},
                                             discovery_context={"research_question": "How many heads in two groups?"})
    assert not report.passed and report.missing_aspects
    assert all(v.supports for v in report.verdicts)
    from src.graph.nodes.finalize import finalize_node
    result = await finalize_node({"answer": answer, "verification": report})
    assert result["status"] == "partial"


async def test_missing_requested_result_regenerates_instead_of_repairing_only_existing_claims(monkeypatch):
    from src.graph.nodes import synthesize
    answer = "Each group shares one K/V pair [arxiv:2408.15664v1#1]."
    evidence = [hit(1, text="Each group has one key and one value head.")]
    report = VerificationReport(passed=False, verdicts=[
        CitationVerdict(citation=c, supports=True, rationale="Checked") for c in _parse_citations(answer)],
        missing_aspects=["Total K and V heads for two groups."])
    generate = AsyncMock(return_value=("Two groups use two key heads [arxiv:2408.15664v1#1].", "stop"))
    repair = AsyncMock()
    monkeypatch.setattr(synthesize, "_generate_answer", generate)
    monkeypatch.setattr(synthesize, "_repair_failed_claims", repair)
    update = await synthesize.synthesize_node({"question": "How many heads?", "answer": answer,
        "evidence": evidence, "evidence_lookup": synthesize._evidence_lookup(evidence),
        "verification": report, "verification_notes": ["Missing total K and V heads."]})
    assert generate.await_count == 1 and repair.await_count == 0
    assert update["tool_iters"] == 1 and "Two groups" in update["answer"]
