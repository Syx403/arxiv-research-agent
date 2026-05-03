from __future__ import annotations

from pydantic import BaseModel

from src.core.types import Citation, CitationVerdict, Message, VerificationReport
from src.llm.client import get_chat_client


async def verify_citations(
    answer: str,
    citations: list[Citation],
    evidence_lookup: dict[int, str],
) -> VerificationReport:
    client = get_chat_client("main")
    verdicts: list[CitationVerdict] = []
    for citation in citations:
        if citation.chunk_id not in evidence_lookup:
            raise KeyError(f"Missing evidence for chunk_id={citation.chunk_id}")
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
                        f"Evidence chunk {citation.chunk_id}:\n{evidence_lookup[citation.chunk_id]}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=240,
            response_format={"type": "json_object"},
        )
        verdicts.append(_parse_verdict(response.content or "", citation))
    return VerificationReport(verdicts=verdicts, passed=all(verdict.supports for verdict in verdicts))


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
