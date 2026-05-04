from __future__ import annotations

import logging
import re

from src.core.diagnostics import log_raw_response
from src.core.types import Citation, Message
from src.graph.state import AgentState
from src.llm.client import EmptyProviderResponseError, get_chat_client


CITATION_RE = re.compile(r"\[(?P<pid>[^\]#\s]+)#(?P<cid>\d+)\]")
logger = logging.getLogger(__name__)


class SynthesizerError(RuntimeError):
    pass


async def synthesize_node(state: AgentState) -> dict:
    evidence = state.get("evidence", [])
    failure_notes = _system_notes(state)
    question = state.get("question", "")
    try:
        answer = await _generate_answer(question, evidence, failure_notes=failure_notes)
    except EmptyProviderResponseError as exc:
        logger.warning("synthesis_empty_provider_response", extra={"question": question[:160], "error": str(exc)})
        return _degraded_update("", evidence)
    citations = _parse_citations(answer)
    if not citations:
        try:
            answer = await _fix_citation_format(answer, question, evidence)
        except EmptyProviderResponseError as exc:
            logger.warning("synthesis_repair_empty_provider_response", extra={"question": question[:160], "error": str(exc)})
            return _degraded_update(answer, evidence)
        citations = _parse_citations(answer)
        if not citations:
            return _degraded_update(answer, evidence)

    return {
        "answer": answer,
        "citations": citations,
        "evidence_lookup": {hit.chunk_id: hit.text for hit in evidence},
        "synthesis_format_degraded": False,
    }


async def _generate_answer(question: str, evidence, *, failure_notes: list[str]) -> str:
    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(
                role="system",
                content=(
                    "Answer research questions using only the supplied evidence. "
                    "Keep the answer concise: one paragraph, 4-6 sentences. "
                    "Use the bracketed [paper_id#chunk_id] identifier shown at the top of each evidence chunk as the inline citation. "
                    "For example, after a claim derived from the chunk headed [arxiv:2210.03629#123], end the sentence with [arxiv:2210.03629#123]. "
                    "Never use [paper_id]-only or [chunk_id]-only citations."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Question:\n{question}\n\n"
                    f"Prior verification notes:\n{chr(10).join(failure_notes) or 'None'}\n\n"
                    f"Evidence:\n{_format_evidence(evidence)}"
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    log_raw_response(
        "synthesize_initial",
        response.content,
        evidence_count=len(evidence),
        available_chunk_ids_count=len(evidence),
    )
    return (response.content or "").strip()


async def _fix_citation_format(answer: str, question: str, evidence) -> str:
    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(
                role="system",
                content=(
                    "Fix citations to use the bracketed [paper_id#chunk_id] identifier shown at the top of each evidence chunk. "
                    "Replace any [chunk_id]-only or [paper_id]-only citations with the full [paper_id#chunk_id] form using the matching evidence header. "
                    "If the previous answer was empty, write a concise new answer using the evidence."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Question:\n{question}\n\n"
                    "If the invalid answer below is empty, write a concise answer using the evidence below.\n\n"
                    f"Invalid answer:\n{answer}\n\n"
                    f"Evidence:\n{_format_evidence(evidence)}"
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    log_raw_response(
        "synthesize_repair",
        response.content,
        evidence_count=len(evidence),
        available_chunk_ids_count=len(evidence),
    )
    return (response.content or "").strip()


def _format_evidence(evidence) -> str:
    return "\n".join(
        f"[{hit.paper_id}#{hit.chunk_id}] (section={hit.section or 'Unknown'})\n{hit.text}\n---"
        for hit in evidence
    )


def _parse_citations(answer: str) -> list[Citation]:
    return [
        Citation(
            paper_id=match.group("pid"),
            chunk_id=int(match.group("cid")),
            claim_span=(match.start(), match.end()),
            quote=None,
        )
        for match in CITATION_RE.finditer(answer)
    ]


def _degraded_update(answer: str, evidence) -> dict:
    return {
        "answer": answer,
        "citations": [],
        "evidence_lookup": {hit.chunk_id: hit.text for hit in evidence},
        "synthesis_format_degraded": True,
    }


def _system_notes(state: AgentState) -> list[str]:
    notes: list[str] = []
    for message in state.get("messages", []):
        if getattr(message, "type", None) == "system":
            content = getattr(message, "content", "")
            if isinstance(content, str):
                notes.append(content)
    return notes
