"""The shared contract between answer text, evidence, and citation verification."""

from __future__ import annotations

import re

from src.core.types import Citation


CITATION_RE = re.compile(r"\[(?P<pid>[^\]#\s]+)#(?P<cid>\d+)\]")
_TOKEN = r"\[[^\]#\s]+#\d+\]"
_GROUP_RE = re.compile(rf"{_TOKEN}(?:[ \t]*[,;]?[ \t]*{_TOKEN})*")
_BOUNDARY_RE = re.compile(r"[.!?。！？][\"'”’)]*\s+|\n+")


def evidence_key(paper_id: str, chunk_id: int) -> str:
    return f"{paper_id}#{chunk_id}"


def parse_citations(answer: str) -> list[Citation]:
    """Attach each citation group to its preceding sentence, not its marker.

    This is a sentence-level heuristic for the compact prose requested by the
    synthesizer, not a general natural-language claim segmenter. Marker masking
    preserves offsets and prevents dots in arXiv IDs from splitting sentences.
    """
    masked = CITATION_RE.sub(lambda match: " " * len(match.group()), answer)
    citations: list[Citation] = []
    for group in _GROUP_RE.finditer(answer):
        end = len(masked[: group.start()].rstrip())
        start = 0
        for boundary in _BOUNDARY_RE.finditer(masked[:end]):
            start = boundary.end()
        while start < end and masked[start].isspace():
            start += 1
        for marker in CITATION_RE.finditer(answer, group.start(), group.end()):
            citations.append(
                Citation(
                    paper_id=marker.group("pid"),
                    chunk_id=int(marker.group("cid")),
                    claim_span=(start, end),
                )
            )
    return citations


def claim_text(answer: str, citation: Citation) -> str:
    if citation.claim_span is None:
        return (citation.quote or "").strip()
    start, end = citation.claim_span
    if not 0 <= start < end <= len(answer):
        return ""
    return CITATION_RE.sub("", answer[start:end]).strip()
