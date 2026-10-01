from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.core.types import ChatResponse, Citation, EvidenceChunk
from src.graph.nodes.synthesize import _parse_citations
from src.retrieval.self_rag import verifier


class _FakeChatClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(
            content='{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports": true, "rationale": "The chunk states the claim."}',
            model="fake",
            finish_reason="stop",
        )


class _BrokenVerifierClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(content="", model="fake", finish_reason="length")


def test_verifier_parser_builds_citation_verdict() -> None:
    citation = _citation(7)

    verdict = verifier._parse_verdict('{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports": true, "rationale": "ok"}', citation)

    assert verdict.citation == citation
    assert verdict.supports is True
    assert verdict.rationale == "ok"


@pytest.mark.parametrize('value', ['"true"', '"yes"', '1', 'null'])
def test_verifier_requires_a_real_boolean(value):
    with pytest.raises(ValidationError):
        verifier._parse_verdict('{"claim_kind":"paper_fact","scope_status":"requested","all_assertions_supported":true,"no_unstated_assumptions":true,"all_citations_contribute":true,"supports": ' + value + ', "rationale": "claim"}', _citation(7))


@pytest.mark.parametrize("criterion", ["all_assertions_supported", "no_unstated_assumptions", "all_citations_contribute"])
def test_overall_true_cannot_override_a_failed_support_check(criterion):
    payload = {"claim_kind": "paper_fact", "scope_status": "requested", "supports": True, "rationale": "The first clause is supported.",
               "all_assertions_supported": True, "no_unstated_assumptions": True,
               "all_citations_contribute": True}
    payload[criterion] = False
    verdict = verifier._parse_verdict(json.dumps(payload), _citation(7))
    assert not verdict.supports
    assert criterion in verdict.rationale


def test_missing_support_checks_are_not_assumed_successful():
    with pytest.raises(ValidationError):
        verifier._parse_verdict('{"supports":true,"rationale":"looks correct"}', _citation(7))


@pytest.mark.parametrize("missing", ["scope_status", "claim_kind"])
def test_missing_semantic_judgment_is_not_silently_accepted(missing):
    payload = {"claim_kind": "paper_fact", "scope_status": "requested", "supports": True,
               "rationale": "Supported", "all_assertions_supported": True,
               "no_unstated_assumptions": True, "all_citations_contribute": True}
    del payload[missing]
    with pytest.raises(ValidationError):
        verifier._parse_verdict(json.dumps(payload), _citation(7))


@pytest.mark.asyncio
async def test_verify_citations_rejects_missing_evidence_without_a_model_call() -> None:
    report = await verifier.verify_citations("claim [p#7].", [_citation(7)], {})
    assert report.passed is False
    assert report.verdicts[0].rationale == "source_missing_or_unbound"


@pytest.mark.asyncio
async def test_verify_citations_uses_evidence_lookup(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: _FakeChatClient())

    answer = "claim text [p#7]."
    report = await verifier.verify_citations(answer, _parse_citations(answer), {7: _evidence(7)})

    assert report.passed is True
    assert report.verdicts[0].citation.chunk_id == 7


@pytest.mark.asyncio
async def test_verify_citations_keeps_citation_when_verifier_json_invalid(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: _BrokenVerifierClient())

    answer = "claim text [p#7]."
    report = await verifier.verify_citations(answer, _parse_citations(answer), {7: _evidence(7)})

    assert report.passed is False
    assert report.verdicts[0].supports is False
    assert report.verdicts[0].rationale.startswith("PARSE_FAILURE")


@pytest.mark.asyncio
async def test_verify_citations_parse_failure_makes_report_fail(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: _BrokenVerifierClient())

    answer = "claim text [p#7] [p#8]."
    report = await verifier.verify_citations(answer, _parse_citations(answer), {7: _evidence(7), 8: _evidence(8)})

    assert report.passed is False
    assert any(verdict.rationale.startswith("PARSE_FAILURE") for verdict in report.verdicts)


def _citation(chunk_id: int) -> Citation:
    return Citation(paper_id="p", chunk_id=chunk_id, claim_text=f"claim for {chunk_id}", claim_span=None)


def _evidence(chunk_id: int) -> EvidenceChunk:
    return EvidenceChunk(paper_id="p", chunk_id=chunk_id, text="claim evidence")
