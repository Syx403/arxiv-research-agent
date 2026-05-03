from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    paper_id: str
    section: str | None
    text: str
    score: float
