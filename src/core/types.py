from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EMBEDDING_DIM = 1536
CHUNK_TOKEN_SIZE = 800
CHUNK_TOKEN_OVERLAP = 100
RETRIEVAL_TOP_K_VECTOR = 30
RETRIEVAL_TOP_K_LEXICAL = 30
RRF_K = 60
RETRIEVAL_TOP_K_AFTER_FUSION = 30
RERANK_TOP_K = 5
MULTI_HOP_MAX_DEPTH = 2
MULTI_HOP_FRONTIER_LIMIT = 8
MAX_SUB_QUESTIONS = 5
SELF_RAG_MAX_RETRIES = 2
EPISODIC_VERBATIM_TURNS = 6
LLM_REQUEST_TIMEOUT_SECONDS = 60
RETRY_MAX_ATTEMPTS = 4


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
    paper_id: str
    chunk_id: int
    quote: str | None = None
    claim_span: tuple[int, int] | None = None


class CitationVerdict(BaseModel):
    citation: Citation
    supports: bool
    rationale: str


class VerificationReport(BaseModel):
    verdicts: list[CitationVerdict]
    passed: bool
