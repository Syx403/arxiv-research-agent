from __future__ import annotations

import asyncio

from pydantic import BaseModel, ValidationError

from src.core.diagnostics import log_raw_response
from src.core.types import Citation, CitationVerdict, Message, VerificationReport
from src.llm.client import get_chat_client, log_parse_status


VERIFIER_EVIDENCE_CHAR_LIMIT = 1200


async def verify_citations(
    answer: str,
    citations: list[Citation],
    evidence_lookup: dict[int, str],
) -> VerificationReport:
    if not citations:
        return VerificationReport(verdicts=[], passed=True, rationale="no_claims_to_verify")

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
        evidence_text = evidence_lookup[citation.chunk_id][:VERIFIER_EVIDENCE_CHAR_LIMIT]
        response = await client.chat(
            [
                Message(
                    role="system",
                    content=(
                        'You judge whether a research claim is SUBSTANTIALLY SUPPORTED by an evidence chunk.\n'
                        'Substantially supported = the chunk states or directly implies the core assertion of the claim. '
                        'Paraphrased wording, hedges, and reasonable inference all count as supported. '
                        'A claim is NOT supported only when the chunk is unrelated to the claim, contradicts it, or omits the asserted fact entirely.\n'
                        'Examples:\n'
                        'Claim: "ReAct interleaves reasoning steps with actions." Chunk: "ReAct alternates thought and action tokens." \u2192 supports=true (paraphrase).\n'
                        'Claim: "Reflexion uses verbal feedback in episodic memory." Chunk: "Reflexion stores reflective texts in long-term memory." \u2192 supports=true (substantial).\n'
                        'Claim: "ReAct uses Monte Carlo Tree Search." Chunk: "ReAct alternates thought and action." \u2192 supports=false (omitted).\n'
                        'Reply ONLY with JSON: {"supports": true|false, "rationale": "<one sentence>"}.'
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        f"Answer:\n{answer}\n\n"
                        f"Specific claim being verified:\n{citation.claim_text}\n\n"
                        f"Evidence chunk {citation.chunk_id} (truncated to 1200 chars):\n"
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
        verdict = _parse_verdict(content, citation)
        log_parse_status("main", "ok")
        return verdict
    except ValidationError:
        log_parse_status("main", "parse_failure_first")
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
            verdict = _parse_verdict(repair.content or "", citation)
            log_parse_status("main", "ok")
            return verdict
        except ValidationError:
            log_parse_status("main", "parse_failure_repair")
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


class _CitationVerdictPayload(BaseModel):
    supports: bool
    rationale: str
