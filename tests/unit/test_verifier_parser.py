from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Citation
from src.retrieval.self_rag import verifier


class _FakeChatClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(
            content='{"supports": true, "rationale": "The chunk states the claim."}',
            model="fake",
            finish_reason="stop",
        )


class _BrokenVerifierClient:
    async def chat(self, *args, **kwargs) -> ChatResponse:
        return ChatResponse(content="", model="fake", finish_reason="length")


def test_verifier_parser_builds_citation_verdict() -> None:
    citation = _citation(7)

    verdict = verifier._parse_verdict('{"supports": true, "rationale": "ok"}', citation)

    assert verdict.citation == citation
    assert verdict.supports is True
    assert verdict.rationale == "ok"


@pytest.mark.asyncio
async def test_verify_citations_raises_when_evidence_missing() -> None:
    with pytest.raises(KeyError, match="chunk_id=7"):
        await verifier.verify_citations("claim", [_citation(7)], {})


@pytest.mark.asyncio
async def test_verify_citations_uses_evidence_lookup(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: _FakeChatClient())

    report = await verifier.verify_citations("claim text", [_citation(7)], {7: "claim evidence"})

    assert report.passed is True
    assert report.verdicts[0].citation.chunk_id == 7


@pytest.mark.asyncio
async def test_verify_citations_keeps_citation_when_verifier_json_invalid(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: _BrokenVerifierClient())

    report = await verifier.verify_citations("claim text", [_citation(7)], {7: "claim evidence"})

    assert report.passed is False
    assert report.verdicts[0].supports is False
    assert report.verdicts[0].rationale.startswith("PARSE_FAILURE")


@pytest.mark.asyncio
async def test_verify_citations_parse_failure_makes_report_fail(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "get_chat_client", lambda name: _BrokenVerifierClient())

    report = await verifier.verify_citations(
        "claim text",
        [_citation(7), _citation(8)],
        {7: "claim evidence", 8: "claim evidence"},
    )

    assert report.passed is False
    assert any(verdict.rationale.startswith("PARSE_FAILURE") for verdict in report.verdicts)


def _citation(chunk_id: int) -> Citation:
    return Citation(paper_id="p", chunk_id=chunk_id, claim_text=f"claim for {chunk_id}", claim_span=None)
