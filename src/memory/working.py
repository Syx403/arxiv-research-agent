from __future__ import annotations

from typing import TypedDict

from src.core.types import Citation, Decomposition, ToolCall, VerificationReport
from src.retrieval.index.types import Hit


class WorkingMemory(TypedDict, total=False):
    current_question: str
    decomposition: Decomposition | None
    subq_results: dict[int, list[Hit]]
    evidence: list[Hit]
    tool_calls: list[ToolCall]
    answer: str | None
    citations: list[Citation]
    verification: VerificationReport | None
