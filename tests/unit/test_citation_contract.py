from __future__ import annotations

import pytest

from src.core.citations import claim_text, parse_citations
from src.core.types import ChatResponse
from src.graph.nodes import self_rag, synthesize
from src.retrieval.index.types import Hit
from src.retrieval.self_rag import verifier


@pytest.mark.parametrize(
    ("answer", "claims"),
    [
        (
            "ReAct interleaves reasoning and acting [arxiv:2210.03629#7].",
            ["ReAct interleaves reasoning and acting"],
        ),
        ("A first claim [p#1]. A second claim [p#2].", ["A first claim", "A second claim"]),
        ("A claim [p#1], [p#2].", ["A claim", "A claim"]),
        ("A claim. [p#1] [p#2]", ["A claim.", "A claim."]),
        ("# Heading\n\nA claim [p#1].\nAnother claim [p#2].", ["A claim", "Another claim"]),
        ("The score is 3.14 [p#1].", ["The score is 3.14"]),
        ("[p#1]", [""]),
    ],
)
def test_citations_resolve_the_prose_not_the_marker(answer, claims):
    citations = parse_citations(answer)
    assert [claim_text(answer, citation) for citation in citations] == claims


@pytest.mark.asyncio
async def test_no_citations_or_unknown_paper_do_not_call_a_provider(monkeypatch):
    def no_provider(_):
        raise AssertionError("No provider should be initialized for invalid evidence.")

    monkeypatch.setattr(verifier, "get_chat_client", no_provider)
    report = await verifier.verify_citations("Uncited prose", [], {})
    assert report.passed is False
    assert report.rationale == "NO_CITATIONS"
    answer = "A claim [wrong-paper#7]."
    report = await verifier.verify_citations(
        answer, parse_citations(answer), {"right-paper#7": "A claim"}
    )
    assert report.passed is False
    assert report.verdicts[0].rationale.startswith("UNKNOWN_EVIDENCE")
    report = await verifier.verify_citations("[p#7]", parse_citations("[p#7]"), {"p#7": "Text"})
    assert report.verdicts[0].rationale.startswith("EMPTY_CLAIM")


@pytest.mark.asyncio
async def test_synthesis_and_verification_share_the_same_claim_and_full_evidence(monkeypatch):
    answer = "ReAct interleaves reasoning and acting [arxiv:2210.03629#7]."
    text = "Context. " * 200 + "ReAct interleaves reasoning and acting."
    hit = Hit(chunk_id=7, paper_id="arxiv:2210.03629", text=text, section="Abstract", score=1)

    class Generator:
        async def chat(self, *args, **kwargs):
            return ChatResponse(content=answer, model="fixture", finish_reason="stop")

    class Verifier:
        async def chat(self, messages, **kwargs):
            prompt = messages[1].content
            assert prompt.startswith("Claim:\nReAct interleaves reasoning and acting\n\n")
            assert prompt.endswith(text)
            return ChatResponse(
                content='{"supports":true,"rationale":"Supported by the final sentence."}',
                model="fixture",
                finish_reason="stop",
            )

    monkeypatch.setattr(synthesize, "get_chat_client", lambda _: Generator())
    monkeypatch.setattr(verifier, "get_chat_client", lambda _: Verifier())
    state = await synthesize.synthesize_node({"question": "What is ReAct?", "evidence": [hit]})
    result = await self_rag.self_rag_node(state)
    assert result["verification"].passed is True
    assert state["citations"][0].paper_id == hit.paper_id


@pytest.mark.asyncio
async def test_empty_evidence_returns_an_explicit_gap_without_generation(monkeypatch):
    monkeypatch.setattr(synthesize, "get_chat_client", lambda _: pytest.fail("No evidence"))
    result = await synthesize.synthesize_node({"question": "What is ReAct?", "evidence": []})
    assert result["synthesis_format_degraded"] is True
    assert "not have enough retrieved evidence" in result["answer"]
