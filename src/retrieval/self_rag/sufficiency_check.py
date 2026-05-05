from __future__ import annotations

from pydantic import ValidationError

from src.core.diagnostics import log_raw_response
from src.core.types import (
    SUFFICIENCY_EVIDENCE_CHARS_PER_CHUNK,
    SUFFICIENCY_EVIDENCE_MAX_CHUNKS,
    Message,
    SufficiencyVerdict,
)
from src.llm.client import get_chat_client
from src.retrieval.index.types import Hit


async def is_sufficient(question: str, evidence: list[Hit]) -> SufficiencyVerdict:
    client = get_chat_client("main")
    capped_evidence = _select_evidence(evidence)
    evidence_text = _format_evidence(capped_evidence)
    response = await client.chat(
        [
            Message(
                role="system",
                content=(
                    "Decide whether the evidence is sufficient to answer the question. "
                    'Return only JSON: {"sufficient": true, "missing_aspects": []}'
                ),
            ),
            Message(role="user", content=f"Question:\n{question}\n\nEvidence:\n{evidence_text}"),
        ],
        temperature=0.0,
        max_tokens=420,
        response_format={"type": "json_object"},
    )
    log_raw_response(
        "sufficiency",
        response.content,
        evidence_count=len(capped_evidence),
        original_evidence_count=len(evidence),
        event="initial",
        finish_reason=response.finish_reason,
    )
    if _is_non_parseable_finish(response.finish_reason):
        return _parse_failure_verdict(
            f"PARSE_FAILURE: sufficiency check finish_reason={response.finish_reason}; treating as insufficient."
        )
    broken_output = response.content or ""
    try:
        return SufficiencyVerdict.model_validate_json(broken_output)
    except ValidationError:
        repair = await client.chat(
            [
                Message(
                    role="system",
                    content='Fix invalid JSON. Return compact valid JSON only: {"sufficient": true, "missing_aspects": []}',
                ),
                Message(role="user", content=f"Invalid output:\n{broken_output}"),
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        try:
            return SufficiencyVerdict.model_validate_json(repair.content or "")
        except ValidationError:
            log_raw_response(
                "sufficiency",
                repair.content,
                evidence_count=len(capped_evidence),
                original_evidence_count=len(evidence),
                event="parse_failure_twice",
                finish_reason=repair.finish_reason,
            )
            return _parse_failure_verdict(
                "PARSE_FAILURE: sufficiency check returned invalid JSON; treating as insufficient."
            )


def _select_evidence(evidence: list[Hit]) -> list[Hit]:
    return sorted(evidence, key=lambda hit: hit.score, reverse=True)[:SUFFICIENCY_EVIDENCE_MAX_CHUNKS]


def _format_evidence(evidence: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.paper_id}#{hit.chunk_id}] {hit.section or 'Unknown'}\n{_truncate(hit.text)}"
        for hit in evidence
    )


def _truncate(text: str) -> str:
    if len(text) <= SUFFICIENCY_EVIDENCE_CHARS_PER_CHUNK:
        return text
    return f"{text[:SUFFICIENCY_EVIDENCE_CHARS_PER_CHUNK]}...[truncated from {len(text)} chars]"


def _is_non_parseable_finish(finish_reason: str) -> bool:
    return finish_reason in {"length", "content_filter", "function_call"}


def _parse_failure_verdict(message: str) -> SufficiencyVerdict:
    return SufficiencyVerdict(sufficient=False, missing_aspects=[message])
