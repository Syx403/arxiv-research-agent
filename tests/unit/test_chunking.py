from ara.rag.chunking import MAX_TOKENS, chunk, sentence_starts
from ara.rag.sources import Paragraph
from ara.tokens import count_tokens


def test_a_short_paragraph_is_one_chunk_with_sentence_offsets() -> None:
    [only] = chunk([Paragraph("T › Intro", "First claim. Second claim, with 3.5 points.")])
    assert (only.ord, only.paragraph, only.heading_path) == (0, 0, "T › Intro")
    assert [only.text[s:] for s in only.sentences] == [
        "First claim. Second claim, with 3.5 points.",
        "Second claim, with 3.5 points.",
    ]
    assert only.search_text == "T › Intro\nFirst claim. Second claim, with 3.5 points."


def test_a_long_paragraph_splits_at_sentences_and_keeps_its_paragraph_index() -> None:
    sentence = "Attention layers store keys and values for every generated token. "
    chunks = chunk([Paragraph("T", "Short."), Paragraph("T › Method", sentence * 60)])
    assert len(chunks) > 2
    assert {c.paragraph for c in chunks[1:]} == {1}
    assert all(count_tokens(c.text) <= MAX_TOKENS for c in chunks)
    assert all(c.text.endswith("token.") for c in chunks[1:])


def test_an_overlong_sentence_is_packed_word_by_word() -> None:
    debris = "x1 " * 1_500  # no sentence end, as in a table dumped from a PDF
    chunks = chunk([Paragraph("T", debris)])
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= MAX_TOKENS for c in chunks)


def test_sentence_starts_skip_decimals_and_lower_case_continuations() -> None:
    text = "We use v2.1 here. e.g. not a break. Done."
    assert sentence_starts(text) == (0, text.index("Done."))
