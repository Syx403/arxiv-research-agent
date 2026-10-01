"""Read official arXiv full text while preserving author-supplied math alttext."""
from __future__ import annotations

import re

from src.corpus.arxiv_html import Document
from src.corpus.pdf_loader import Section


def load_html(content: str, *, expected_id: str | None = None) -> list[Section]:
    document = Document(content)
    if expected_id:
        identity = re.search(r"arXiv:\s*([\w./-]+v\d+)", document.root.text(), re.I)
        if identity is None or identity[1] != expected_id:
            raise ValueError("arXiv full-text version identity mismatch")
    article = next(document.root.find("article", "ltx_document"), None)
    if article is None:
        raise ValueError("No arXiv full-text article found")
    sections, lines = [], []
    heading = "Document"

    def flush():
        value = re.sub(r"[ \t]+", " ", "".join(lines)).strip()
        if value:
            sections.append(Section(heading, value))
        lines.clear()

    def text(node):
        if isinstance(node, str):
            return node
        if node.tag == "math":
            if node.attrs.get("alttext"):
                return " $" + node.attrs["alttext"] + "$ "
            annotation = next((a for a in node.find("annotation")
                               if "tex" in a.attrs.get("encoding", "").lower()), None)
            if annotation:
                return " $" + annotation.text() + "$ "
            # Never silently flatten MathML: this can reverse an equation's meaning.
            return " [Mathematical notation unavailable in this extraction] "
        if node.tag in {"script", "style", "nav", "button"}:
            return ""
        return "".join(text(child) for child in node.children)

    def visit(node):
        nonlocal heading
        if isinstance(node, str):
            lines.append(node)
        elif node.tag in {"script", "style", "nav", "button"}:
            return
        elif node.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            flush()
            heading = " ".join(text(node).split())
        elif node.tag == "math":
            lines.append(text(node))
        else:
            for child in node.children:
                visit(child)
            if node.tag in {"p", "li", "tr", "figure", "div"}:
                lines.append("\n")

    visit(article)
    flush()
    if sum(len(section.text) for section in sections) < 300:
        raise ValueError("arXiv HTML has no usable original text")
    return sections
