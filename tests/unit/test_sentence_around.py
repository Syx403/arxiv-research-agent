from __future__ import annotations

from src.graph.nodes.synthesize import _sentence_around


def test_sentence_around_middle_sentence() -> None:
    text = "First sentence. ReAct interleaves reasoning [arxiv:2210.03629#123]. Final sentence."
    start = text.index("[arxiv")
    assert text[slice(*_sentence_around(text, start, start + 22))] == (
        "ReAct interleaves reasoning [arxiv:2210.03629#123]."
    )


def test_sentence_around_start_of_answer() -> None:
    text = "[arxiv:2210.03629#123] supports the opening claim. Another sentence."
    start = text.index("[arxiv")
    assert _sentence_around(text, start, start + 22)[0] == 0


def test_sentence_around_end_of_answer() -> None:
    text = "The final claim is supported [arxiv:2210.03629#123]"
    start = text.index("[arxiv")
    sent_start, sent_end = _sentence_around(text, start, start + 22)
    assert text[sent_start:sent_end] == text


def test_sentence_around_skips_eg_abbreviation() -> None:
    text = "Prior. Methods, e.g. ReAct, interleave actions [arxiv:2210.03629#123]. Next."
    start = text.index("[arxiv")
    sent_start, sent_end = _sentence_around(text, start, start + 22)
    assert text[sent_start:sent_end] == "Methods, e.g. ReAct, interleave actions [arxiv:2210.03629#123]."


def test_sentence_around_skips_fig_and_et_al_abbreviations() -> None:
    text = "Prior. Fig. 2 in Yao et al. shows a search tree [arxiv:2305.10601#44]. Next."
    start = text.index("[arxiv")
    sent_start, sent_end = _sentence_around(text, start, start + 20)
    assert text[sent_start:sent_end] == "Fig. 2 in Yao et al. shows a search tree [arxiv:2305.10601#44]."


def test_sentence_around_one_sentence_answer() -> None:
    text = "ReAct interleaves reasoning and acting [arxiv:2210.03629#123]."
    start = text.index("[arxiv")
    assert text[slice(*_sentence_around(text, start, start + 22))] == text
