from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class SourceSpan:
    """Character offsets into an immutable chunk, never model-authored prose."""

    start: int
    end: int


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    paper_id: str
    section: str | None
    text: str
    score: float
    title: str = ""
    evidence_spans: tuple[SourceSpan, ...] = ()
    evidence_role: str = "unreviewed"
    published_at: str | None = None
    url: str = ""

    def __post_init__(self):
        # Msgpack restores arrays as lists; preserve immutable state on resume.
        object.__setattr__(self, "evidence_spans", tuple(self.evidence_spans))
