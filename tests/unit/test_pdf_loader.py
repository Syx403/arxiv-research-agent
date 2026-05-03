from __future__ import annotations

from pathlib import Path

from src.corpus import pdf_loader


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, path: str) -> None:
        self.pages = [
            _FakePage(
                """
                Abstract
                This paper introduces a test.
                1 Introduction
                The introduction \x00text is here.
                METHOD
                The method text is here.
                """
            )
        ]


def test_load_pdf_extracts_sections(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf_loader, "PdfReader", _FakeReader)
    path = tmp_path / "tiny.pdf"
    path.write_bytes(b"%PDF-1.4 fake fixture")

    sections = pdf_loader.load_pdf(path)

    assert [section.name for section in sections] == ["Abstract", "Introduction", "METHOD"]
    assert "introduces a test" in sections[0].text
    assert "introduction text" in sections[1].text
    assert "\x00" not in sections[1].text
    assert "method text" in sections[2].text
