from __future__ import annotations

import asyncio

from pydantic import BaseModel, ValidationError

from src.core.diagnostics import log_raw_response
from src.core.types import Citation, CitationVerdict, Message, VerificationReport
from src.llm.client import get_chat_client


VERIFIER_EVIDENCE_CHAR_LIMIT = 1200


async def verify_citations(
    answer: str,
    citations: list[Citation],
    evidence_lookup: dict[int, str],
) -> VerificationReport:
    client = get_chat_client("main")
    semaphore = asyncio.Semaphore(8)
    verdicts = await asyncio.gather(
        *(_verify_one_citation(client, semaphore, answer, citation, evidence_lookup) for citation in citations)
    )
    return VerificationReport(verdicts=list(verdicts), passed=all(verdict.supports for verdict in verdicts))


async def _verify_one_citation(
    client,
    semaphore: asyncio.Semaphore,
    answer: str,
    citation: Citation,
    evidence_lookup: dict[int, str],
) -> CitationVerdict:
    async with semaphore:
        if citation.chunk_id not in evidence_lookup:
            raise KeyError(f"Missing evidence for chunk_id={citation.chunk_id}")
        raw_evidence = evidence_lookup[citation.chunk_id]
        evidence_text = raw_evidence[:VERIFIER_EVIDENCE_CHAR_LIMIT]
        truncated = len(raw_evidence) > VERIFIER_EVIDENCE_CHAR_LIMIT
        truncation_note = (
            ""
            if not truncated
            else f" (truncated to 1200 chars from original {len(raw_evidence)} chars)"
        )
        truncation_marker = " [truncated]" if truncated else ""
        response = await client.chat(
            [
                Message(
                    role="system",
                    content='Verify whether evidence supports the cited claim. Return JSON: {"supports": true, "rationale": "..."}',
                ),
                Message(
                    role="user",
                    content=(
                        f"Answer:\n{answer}\n\n"
                        f"Claim span:\n{_claim_span_text(answer, citation)}\n\n"
                        f"Evidence chunk {citation.chunk_id}{truncation_marker}{truncation_note}:\n"
                        f"{evidence_text}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=320,
            response_format={"type": "json_object"},
        )
        log_raw_response(
            "verifier",
            response.content,
            citation_chunk_id=citation.chunk_id,
            event="initial",
        )
        return await _parse_or_repair_verdict(client, response.content or "", citation)


async def _parse_or_repair_verdict(client, content: str, citation: Citation) -> CitationVerdict:
    try:
        return _parse_verdict(content, citation)
    except ValidationError:
        repair = await client.chat(
            [
                Message(
                    role="system",
                    content='Fix invalid verification JSON. Return compact valid JSON only: {"supports": true, "rationale": "..."}',
                ),
                Message(role="user", content=f"Invalid output:\n{content}"),
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        try:
            return _parse_verdict(repair.content or "", citation)
        except ValidationError:
            log_raw_response(
                "verifier",
                repair.content,
                citation_chunk_id=citation.chunk_id,
                event="parse_failure_twice",
            )
            return CitationVerdict(
                citation=citation,
                supports=False,
                rationale="PARSE_FAILURE: verifier returned invalid JSON twice; treating as unverified.",
            )


def _parse_verdict(content: str, citation: Citation) -> CitationVerdict:
    parsed = _CitationVerdictPayload.model_validate_json(content)
    return CitationVerdict(citation=citation, supports=parsed.supports, rationale=parsed.rationale)


def _claim_span_text(answer: str, citation: Citation) -> str:
    if citation.claim_span is None:
        return citation.quote or ""
    start, end = citation.claim_span
    return answer[start:end]


class _CitationVerdictPayload(BaseModel):
    supports: bool
    rationale: str
