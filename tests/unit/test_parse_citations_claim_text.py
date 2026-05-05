from __future__ import annotations

from src.graph.nodes.synthesize import _parse_citations


def test_parse_citations_populates_sentence_claim_text() -> None:
    answer = "ReAct interleaves reasoning and acting [arxiv:2210.03629#123]."

    citations = _parse_citations(answer)

    assert len(citations) == 1
    assert citations[0].paper_id == "arxiv:2210.03629"
    assert citations[0].chunk_id == 123
    assert citations[0].claim_text == answer
    assert citations[0].claim_span == (0, len(answer))


def test_parse_citations_same_sentence_get_same_claim_text() -> None:
    sentence = (
        "Tree of Thoughts extends chain-of-thought search [arxiv:2305.10601#10] "
        "and relates to deliberate reasoning [arxiv:2201.11903#22]."
    )

    citations = _parse_citations(sentence)

    assert len(citations) == 2
    assert citations[0].claim_text == sentence
    assert citations[1].claim_text == sentence


def test_parse_citations_empty_answer() -> None:
    assert _parse_citations("") == []
