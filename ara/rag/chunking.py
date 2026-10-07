"""Paragraph-aligned chunks (DESIGN §5): one chunk per paragraph, a long paragraph split at sentence
boundaries into pieces of at most MAX_TOKENS. A chunk never crosses a paragraph or a section."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from ara.rag.sources import Paragraph
from ara.tokens import count_tokens

MAX_TOKENS = 400
# A sentence ends at . ! or ? then space and an upper-case letter, digit, quote, bracket or $.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[$])")


@dataclass(frozen=True)
class Chunk:
    ord: int
    paragraph: int  # index of the source paragraph
    heading_path: str
    text: str
    sentences: tuple[int, ...]  # start offset of each sentence in `text`

    @property
    def search_text(self) -> str:
        """What BM25 and the embedding see: the heading path is a cheap contextual header."""
        return f"{self.heading_path}\n{self.text}"


def chunk(paragraphs: Sequence[Paragraph]) -> list[Chunk]:
    pieces = [
        (index, paragraph.heading_path, piece)
        for index, paragraph in enumerate(paragraphs)
        for piece in _split(paragraph.text)
    ]
    return [
        Chunk(order, index, path, text, sentence_starts(text))
        for order, (index, path, text) in enumerate(pieces)
    ]


def sentence_starts(text: str) -> tuple[int, ...]:
    return (0, *(match.end() for match in SENTENCE_END.finditer(text)))


def _split(text: str) -> list[str]:
    """Pack whole sentences into pieces of at most MAX_TOKENS. A sentence that is itself too long
    (usually PDF debris) is packed word by word, so no piece exceeds the embedding input limit."""
    if count_tokens(text) <= MAX_TOKENS:
        return [text]
    starts = sentence_starts(text)
    sentences = [text[a:b].strip() for a, b in zip(starts, (*starts[1:], len(text)), strict=True)]
    units = [
        unit
        for sentence in sentences
        for unit in (_pack(sentence.split()) if count_tokens(sentence) > MAX_TOKENS else [sentence])
    ]
    return _pack(units)


def _pack(parts: list[str]) -> list[str]:
    """Join consecutive parts while their summed token counts stay within MAX_TOKENS."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for part in parts:
        tokens = count_tokens(part) + 1  # + the joining space
        if current and size + tokens > MAX_TOKENS:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(part)
        size += tokens
    pieces.append(" ".join(current))
    return pieces
