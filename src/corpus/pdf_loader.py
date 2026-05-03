from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


COMMON_SECTION_NAMES = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "approach",
    "experiments",
    "evaluation",
    "results",
    "discussion",
    "conclusion",
    "appendix",
    "acknowledgements",
    "acknowledgments",
    "references",
}
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True)
class Section:
    name: str
    text: str


def load_pdf(path: Path) -> list[Section]:
    reader = PdfReader(str(path))
    sections: list[Section] = []
    current_name = "Document"
    current_lines: list[str] = []

    for page in reader.pages:
        page_text = _sanitize_text(page.extract_text() or "")
        for raw_line in page_text.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line:
                continue
            if _is_section_header(line):
                _append_section(sections, current_name, current_lines)
                current_name = _normalize_header(line)
                current_lines = []
            else:
                current_lines.append(line)

    _append_section(sections, current_name, current_lines)
    if sections:
        return sections
    text = _sanitize_text("\n".join((page.extract_text() or "") for page in reader.pages)).strip()
    return [Section(name="Document", text=text)] if text else []


def _append_section(sections: list[Section], name: str, lines: list[str]) -> None:
    text = "\n".join(lines).strip()
    if text:
        sections.append(Section(name=name, text=text))


def _sanitize_text(text: str) -> str:
    return _CTRL_CHAR_RE.sub("", text)


def _is_section_header(line: str) -> bool:
    if sum(1 for char in line if char.isalpha() and char.isascii()) < 3:
        return False
    if any(ord(char) < 32 for char in line):
        return False
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9 ,:/()-]*$", line):
        return False
    if len(line) > 90 or line.endswith("."):
        return False
    normalized = _normalize_header(line).lower()
    if normalized in COMMON_SECTION_NAMES:
        return True
    if line.isupper() and 3 <= len(line) <= 60:
        if " " not in line:
            return False
        return True
    return bool(re.match(r"^\d+(\.\d+)*\s+[A-Z][A-Za-z0-9 ,:/()-]{2,}$", line))


def _normalize_header(line: str) -> str:
    return re.sub(r"^\d+(\.\d+)*\s+", "", line).strip()
