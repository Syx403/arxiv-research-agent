from __future__ import annotations

from src.corpus.chunking import chunk_sections
from src.corpus.pdf_loader import Section


def test_chunking_respects_section_boundaries_and_token_limits() -> None:
    sections = [
        Section(name="Introduction", text=("Intro sentence. " * 250).strip()),
        Section(name="Method", text=("Method sentence. " * 250).strip()),
    ]

    chunks = chunk_sections(sections, target_tokens=80, overlap=10)

    assert chunks
    assert {chunk.section for chunk in chunks} == {"Introduction", "Method"}
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert all("Intro" in chunk.text for chunk in chunks if chunk.section == "Introduction")
    assert all("Method" in chunk.text for chunk in chunks if chunk.section == "Method")


def test_chunking_splits_long_sentence_with_overlap() -> None:
    section = Section(name="Long", text=" ".join(f"token{i}" for i in range(300)))

    chunks = chunk_sections([section], target_tokens=50, overlap=10)

    assert len(chunks) > 1
    assert all(chunk.section == "Long" for chunk in chunks)
    assert all(chunk.token_count <= 50 for chunk in chunks)


def test_chunking_filters_tiny_or_control_only_chunks() -> None:
    chunks = chunk_sections(
        [
            Section(name="Tiny", text="x"),
            Section(name="Control", text="\x03\x04\x05"),
        ]
    )

    assert chunks == []


def test_chunking_keeps_long_valid_text() -> None:
    section = Section(
        name="Valid",
        text="This valid paragraph has enough readable content to survive the quality filter.",
    )

    chunks = chunk_sections([section])

    assert len(chunks) == 1
    assert chunks[0].section == "Valid"
    assert "readable content" in chunks[0].text
