"""Paper sources (DESIGN §5): arXiv HTML first with PDF as fallback, and QASPER's own full text.
Each source yields a ParsedPaper: metadata plus paragraphs in reading order with heading paths."""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader

SEPARATOR = " › "
PARAGRAPH_END = 0.9  # a sentence-final line shorter than this share of a full line ends a paragraph
UNDECODED = re.compile(r"[\x00-\x08\x0b-\x1f]")


@dataclass(frozen=True)
class Paragraph:
    heading_path: str
    text: str


@dataclass(frozen=True)
class ParsedPaper:
    id: str  # "arxiv:2210.03629v3" or "qasper:1912.01214"
    source: Literal["arxiv", "qasper"]
    arxiv_id: str
    version: int | None
    title: str
    abstract: str
    format: Literal["html", "pdf", "qasper"]
    paragraphs: tuple[Paragraph, ...]


def qasper_paper(record: Mapping[str, Any]) -> ParsedPaper:
    """A QASPER record. Its paragraphs are kept verbatim, so gold evidence matches them exactly;
    the abstract comes first. Subsections are written "Section ::: Subsection" in the dataset."""
    title = _clean(record["title"])
    abstract = _clean(record["abstract"])
    full_text = record["full_text"]
    body = [
        Paragraph(SEPARATOR.join([title, *_qasper_sections(section)]), text.strip())
        for section, texts in zip(full_text["section_name"], full_text["paragraphs"], strict=True)
        for text in texts
        if text.strip()
    ]
    return ParsedPaper(
        id=f"qasper:{record['id']}",
        source="qasper",
        arxiv_id=record["id"],
        version=None,
        title=title,
        abstract=abstract,
        format="qasper",
        paragraphs=(Paragraph(f"{title}{SEPARATOR}Abstract", abstract), *body),
    )


def _qasper_sections(name: str | None) -> list[str]:
    return [part.strip() for part in (name or "").split(":::") if part.strip()]


def arxiv_html_paper(html: str, arxiv_id: str, version: int) -> ParsedPaper:
    """arXiv's LaTeXML HTML. Math keeps its LaTeX source; footnotes and references are dropped."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_="ltx_document")
    if not isinstance(article, Tag):
        raise ValueError(f"{arxiv_id}v{version}: no LaTeXML article in the HTML")
    for math in article.find_all("math"):
        math.replace_with(f" ${math.get('alttext') or '[math]'}$ ")
    for node in article.select(".ltx_note, .ltx_bibliography, .ltx_tag"):
        node.decompose()
    title = _clean(_text(article.find(class_="ltx_title_document")))
    abstract_node = article.find(class_="ltx_abstract")
    abstract = ""
    if isinstance(abstract_node, Tag):
        abstract = _clean(" ".join(_text(p) for p in abstract_node.find_all("p")))
    paragraphs = [Paragraph(f"{title}{SEPARATOR}Abstract", abstract)] if abstract else []
    for node in article.select("section .ltx_para, section figcaption"):
        if node.find_parent(class_="ltx_para") is None and (text := _clean(_text(node))):
            paragraphs.append(Paragraph(_heading_path(title, node), text))
    if not paragraphs:
        raise ValueError(f"{arxiv_id}v{version}: the HTML has no readable paragraphs")
    return ParsedPaper(
        id=f"arxiv:{arxiv_id}v{version}",
        source="arxiv",
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        format="html",
        paragraphs=tuple(paragraphs),
    )


def _heading_path(title: str, node: Tag) -> str:
    headings = []
    for section in reversed(node.find_parents("section")):
        heading = section.find(re.compile("^h[1-6]$"), recursive=False)
        if heading is not None:
            headings.append(_clean(heading.get_text()))
    return SEPARATOR.join([title, *headings])


def arxiv_pdf_paper(
    pdf: bytes, arxiv_id: str, version: int, *, title: str, abstract: str
) -> ParsedPaper:
    """The fallback when a version has no HTML. PDFs carry no section tree, so every paragraph sits
    under the title; paragraphs are recovered from the line layout (`_pdf_paragraphs`)."""
    pages = [(page.extract_text() or "").splitlines() for page in PdfReader(BytesIO(pdf)).pages]
    paragraphs = [Paragraph(f"{title}{SEPARATOR}Abstract", abstract)] if abstract else []
    paragraphs += [Paragraph(title, text) for text in _pdf_paragraphs(pages)]
    if len(paragraphs) <= 1:
        raise ValueError(f"{arxiv_id}v{version}: no text could be extracted from the PDF")
    return ParsedPaper(
        id=f"arxiv:{arxiv_id}v{version}",
        source="arxiv",
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        abstract=abstract,
        format="pdf",
        paragraphs=tuple(paragraphs),
    )


def _pdf_paragraphs(pages: list[list[str]]) -> list[str]:
    """pypdf yields lines with no blank line between paragraphs. A paragraph ends at a line that
    ends a sentence well short of the full line width. Running headers (a line on most pages), page
    numbers and undecodable text (control characters, from fonts without a Unicode map) are dropped;
    a line ending in a hyphen joins the next without a space."""
    seen = Counter(line.strip() for lines in pages for line in set(lines))
    lines = [
        line
        for page in pages
        for line in map(str.strip, page)
        if line
        and not line.isdigit()
        and seen[line] <= max(2, len(pages) // 2)
        and not UNDECODED.search(line)
    ]
    if not lines:
        return []
    full = sorted(map(len, lines))[int(0.9 * (len(lines) - 1))]
    paragraphs, current = [], ""
    for line in lines:
        current = current + line if current.endswith("-") else f"{current} {line}".lstrip()
        if line.endswith((".", "!", "?")) and len(line) < PARAGRAPH_END * full:
            paragraphs.append(current)
            current = ""
    return [_clean(text) for text in [*paragraphs, current] if text]


def _text(node: object) -> str:
    return node.get_text() if isinstance(node, Tag) else ""


def _clean(text: str) -> str:
    return " ".join(text.split())
