from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.core.types import (
    Citation,
    Decomposition,
    EvidenceChunk,
    GraphExpansion,
    ReflectionReport,
    SubQResult,
    VerificationReport,
)
from src.retrieval.index.types import Hit


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    turn_id: str
    session_context: dict
    question: str
    decomposition: Decomposition | None
    # Retrieval merges parallel results and emits the full mapping. An accumulating
    # reducer would retain results from an earlier question on the same thread.
    subq_results: dict[int, SubQResult]
    evidence: list[Hit]
    graph_expansion: GraphExpansion | None
    multi_hop_visited: list[str]
    multi_hop_attempts: int
    multi_hop_expansions: int
    multi_hop_preemptive_requested: bool
    multi_hop_preemptive_subqs: list[int]
    next_subqueries: list[str]
    question_type: str
    answer: str | None
    answer_fields: list[dict]
    response_fields: list[dict]
    citations: list[Citation]
    verification: VerificationReport | None
    draft_verification: VerificationReport | None
    reflection: ReflectionReport | None
    tool_iters: int
    active_subq_index: int | None
    evidence_lookup: dict[int, EvidenceChunk]
    skip_reflection: bool
    synthesis_format_degraded: bool
    synthesis_finish_reason: str
    verification_notes: list[str]
    research_question: str
    conversation_context: list[str]
    context_message_count: int
    status: str
    stop_reason: str
    missing_aspects: list[str]
    verified_answer: str
    forced_stop_reason: str
    research_policy: dict
    external_discovery: dict
    retrieval_paper_ids: list[str] | None
    best_verified: dict
    last_verification_evidence: dict[int, EvidenceChunk]
    delivery_from_prior_draft: bool
    verification_recovery_attempted: bool
    verification_citation_repair_attempted: bool
    verification_recovery_changed: bool
    verification_recovery_requires_rewrite: bool
    result_kind: str
    paper_plan: dict
    paper_search: dict
    paper_candidates: list[dict]
    paper_results: list[dict]
    paper_intro: str
    pending_paper_queries: list[dict]

    completed_node: str
    node_timings_s: dict[str, float]
    next_node: str
    rejected_candidate_fingerprints: dict[str, str]
    assessed_fingerprint: str
    related_expansion_attempted: bool
    related_search: bool
    reading_attempted_ids: list[str]
    reading_added_count: int
    claim_review: dict
    verification_stalled: bool
