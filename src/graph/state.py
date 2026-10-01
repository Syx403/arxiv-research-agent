from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.core.types import (
    Citation,
    Decomposition,
    GraphExpansion,
    ReflectionReport,
    SubQResult,
    VerificationReport,
)
from src.retrieval.index.types import Hit


def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    question: str
    decomposition: Decomposition | None
    subq_results: Annotated[dict[int, SubQResult], merge_dicts]
    evidence: list[Hit]
    graph_expansion: GraphExpansion | None
    multi_hop_visited: list[str]
    multi_hop_calls: int
    answer: str | None
    citations: list[Citation]
    verification: VerificationReport | None
    reflection: ReflectionReport | None
    tool_iters: int
    active_subq_index: int | None
    evidence_lookup: dict[str, str]
    skip_reflection: bool
    synthesis_format_degraded: bool
