from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

from src.core.types import CHUNK_TOKEN_OVERLAP, CHUNK_TOKEN_SIZE
from src.corpus.pdf_loader import Section


@dataclass(frozen=True)
class Chunk:
    section: str
    text: str
    token_count: int


def chunk_sections(
    sections: list[Section],
    target_tokens: int = CHUNK_TOKEN_SIZE,
    overlap: int = CHUNK_TOKEN_OVERLAP,
) -> list[Chunk]:
    encoding = tiktoken.get_encoding("cl100k_base")
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(_chunk_one_section(section, encoding, target_tokens, overlap))
    return chunks


def _chunk_one_section(
    section: Section,
    encoding,
    target_tokens: int,
    overlap: int,
) -> list[Chunk]:
    sentences = _split_sentences(section.text)
    chunks: list[Chunk] = []
    current = ""

    for sentence in sentences:
        sentence_tokens = encoding.encode(sentence)
        if len(sentence_tokens) > target_tokens:
            if current.strip():
                chunks.append(_make_chunk(section.name, current, encoding, target_tokens))
                current = _overlap_text(current, encoding, overlap)
            chunks.extend(_chunk_long_text(section.name, sentence, encoding, target_tokens, overlap))
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(encoding.encode(candidate)) > target_tokens:
            chunks.append(_make_chunk(section.name, current, encoding, target_tokens))
            current = f"{_overlap_text(current, encoding, overlap)} {sentence}".strip()
        else:
            current = candidate

    if current.strip():
        chunks.append(_make_chunk(section.name, current, encoding, target_tokens))
    return chunks


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def _chunk_long_text(section: str, text: str, encoding, target_tokens: int, overlap: int) -> list[Chunk]:
    tokens = encoding.encode(text)
    chunks: list[Chunk] = []
    step = max(1, target_tokens - overlap)
    for start in range(0, len(tokens), step):
        window = tokens[start : start + target_tokens]
        if not window:
            continue
        chunk_text = encoding.decode(window)
        chunks.append(Chunk(section=section, text=chunk_text, token_count=len(window)))
        if start + target_tokens >= len(tokens):
            break
    return chunks


def _make_chunk(section: str, text: str, encoding, target_tokens: int) -> Chunk:
    tokens = encoding.encode(text)
    if len(tokens) <= target_tokens:
        return Chunk(section=section, text=text.strip(), token_count=len(tokens))
    window = tokens[:target_tokens]
    return Chunk(section=section, text=encoding.decode(window).strip(), token_count=len(window))


def _overlap_text(text: str, encoding, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tokens = encoding.encode(text)
    return encoding.decode(tokens[-overlap:]).strip()
