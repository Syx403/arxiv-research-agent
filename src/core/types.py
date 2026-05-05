from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.retrieval.index.types import Hit


EMBEDDING_DIM = 1536
CHUNK_TOKEN_SIZE = 800
CHUNK_TOKEN_OVERLAP = 100
RETRIEVAL_TOP_K_VECTOR = 30
RETRIEVAL_TOP_K_LEXICAL = 30
RRF_K = 60
RETRIEVAL_TOP_K_AFTER_FUSION = 30
RERANK_TOP_K = 10
SYNTHESIS_EVIDENCE_MAX_CHUNKS = 8
SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK = 800
SUFFICIENCY_EVIDENCE_MAX_CHUNKS = 5
SUFFICIENCY_EVIDENCE_CHARS_PER_CHUNK = 600
MULTI_HOP_MAX_DEPTH = 2
MULTI_HOP_FRONTIER_LIMIT = 5
MULTI_HOP_MAX_CALLS = 1
MULTI_HOP_WALL_CLOCK_BUDGET_S = 60
PAPER_HINT_BOOST_FACTOR = 1.1
MAX_SUB_QUESTIONS = 5
SELF_RAG_MAX_RETRIES = 2
EPISODIC_VERBATIM_TURNS = 6
LLM_REQUEST_TIMEOUT_SECONDS = 60
RETRY_MAX_ATTEMPTS = 4
RETRY_WAIT_INITIAL_SECONDS = 1.0
RETRY_WAIT_MAX_SECONDS = 30.0


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = None
    model: str
    finish_reason: str


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    paper_id: str
    chunk_id: int
    claim_text: str
    claim_span: tuple[int, int] | None = None
    quote: str | None = None


class CitationVerdict(BaseModel):
    citation: Citation
    supports: bool
    rationale: str


class VerificationReport(BaseModel):
    verdicts: list[CitationVerdict]
    passed: bool
    rationale: str = ""


class SubQuestion(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    depends_on: list[int] = Field(default_factory=list)


class Decomposition(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub_questions: list[SubQuestion]

    @model_validator(mode="after")
    def validate_dependency_dag(self) -> "Decomposition":
        for idx, sub_question in enumerate(self.sub_questions):
            for dependency in sub_question.depends_on:
                if dependency < 0 or dependency >= idx:
                    raise ValueError("Sub-question dependencies must point to earlier sub-questions")
        return self


class RouteDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    sources: list[str] = Field(default_factory=lambda: ["local_pgvector"])
    use_concept_graph: bool = False
    may_use_semantic_scholar_live: bool = False


class SubQResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    subq_index: int
    hits: list[Hit] = Field(default_factory=list)
    sufficient: bool
    route_decision: RouteDecision
    retries_used: int = 0
    multi_hop_used: bool = False


class RelevanceVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    relevant: bool
    rationale: str


class SufficiencyVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    sufficient: bool
    missing_aspects: list[str] = Field(default_factory=list)


class FrontierItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    paper_id: str
    score: float
    depth: int


class WalkResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    visited_paper_ids: list[str]
    newly_ingested_paper_ids: list[str]


class Session(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    title: str | None = None
    created_at: datetime
    last_active: datetime


class StoredMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    message_id: int
    session_id: str
    role: str
    content: str
    metadata: dict = Field(default_factory=dict)
    created_at: datetime


class ConceptRelationExtraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_display: str
    type: str
    evidence_chunk_ids: list[int] = Field(default_factory=list)


class ConceptExtraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    display_name: str
    definition: str
    aliases: list[str] = Field(default_factory=list)
    evidence_chunk_ids: list[int] = Field(default_factory=list)
    relations: list[ConceptRelationExtraction] = Field(default_factory=list)


class StoredConcept(BaseModel):
    model_config = ConfigDict(frozen=True)

    concept_id: int
    canonical: str
    display: str
    definition: str | None = None
    evidence_chunk_ids: list[int] = Field(default_factory=list)


class ConceptLinkResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    concept: StoredConcept
    created: bool
    link_path: Literal["alias", "canonical", "embedding", "new"]


class Neighbor(BaseModel):
    model_config = ConfigDict(frozen=True)

    concept: StoredConcept
    rel_type: str
    weight: float
    evidence_chunk_ids: list[int] = Field(default_factory=list)


class GraphExpansion(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed_concept_ids: list[int] = Field(default_factory=list)
    neighbor_concept_ids: list[int] = Field(default_factory=list)
    extra_paper_hints: list[str] = Field(default_factory=list)


class ReflectionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    concepts_created: int = 0
    concepts_reused: int = 0
    relations_created: int = 0
    relations_reused: int = 0


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    correctness: int = Field(ge=1, le=5)
    groundedness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    rationale: str
