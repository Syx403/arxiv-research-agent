from __future__ import annotations

import pytest

from src.core.types import ChatResponse, Citation
from src.retrieval.self_rag import verifier


@pytest.mark.asyncio
async def test_verifier_sees_support_beyond_old_character_limit(monkeypatch) -> None:
    evidence = "Unrelated context. " * 150 + "The observed effect is positive."

    class Client:
        async def chat(self, messages, **kwargs):
            assert messages[1].content.endswith(evidence)
            return ChatResponse(
                content='{"supports":true,"rationale":"Support at the end."}',
                model="fake",
                finish_reason="stop",
            )

    monkeypatch.setattr(verifier, "get_chat_client", lambda _: Client())
    citation = Citation(paper_id="p", chunk_id=7, claim_span=(0, 31))
    report = await verifier.verify_citations(
        "The observed effect is positive.", [citation], {"p#7": evidence}
    )
    assert report.passed is True


@pytest.mark.asyncio
async def test_json_repair_retains_claim_and_evidence(monkeypatch) -> None:
    class Client:
        calls = 0

        async def chat(self, messages, **kwargs):
            self.calls += 1
            assert "Claim:\nclaim" in messages[1].content
            assert "Evidence [p#7]:\nNo support here" in messages[1].content
            output = (
                "not json" if self.calls == 1 else '{"supports":false,"rationale":"Unsupported."}'
            )
            return ChatResponse(content=output, model="fake", finish_reason="stop")

    client = Client()
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: client)
    report = await verifier.verify_citations(
        "claim", [Citation(paper_id="p", chunk_id=7, claim_span=(0, 5))], {"p#7": "No support here"}
    )
    assert client.calls == 2
    assert report.passed is False
