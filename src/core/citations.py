"""The shared contract between answer text, evidence, and citation verification."""

from __future__ import annotations

import re

from src.core.types import Citation


CITATION_RE = re.compile(r"\[(?P<pid>[^\]#\s]+)#(?P<cid>\d+)\]")


def evidence_key(paper_id: str, chunk_id: int) -> str:
    return f"{paper_id}#{chunk_id}"


def parse_citations(answer: str) -> list[Citation]:
    """Compatibility entry point; use the runtime's sentence/table parser."""
    from src.graph.nodes.synthesize import _parse_citations
    return _parse_citations(answer)


def claim_text(answer: str, citation: Citation) -> str:
    if citation.claim_span is None:
        return (citation.quote or "").strip()
    start, end = citation.claim_span
    if not 0 <= start < end <= len(answer):
        return ""
    prose = CITATION_RE.sub("", answer[start:end]).strip()
    return re.sub(r"[ \t]+([.!?。！？])", r"\1", prose)
