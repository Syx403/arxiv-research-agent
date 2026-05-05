from __future__ import annotations

import logging
import re

from src.core.diagnostics import log_raw_response
from src.core.types import (
    SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK,
    SYNTHESIS_EVIDENCE_MAX_CHUNKS,
    Citation,
    Message,
)
from src.graph.state import AgentState
from src.llm.client import EmptyProviderResponseError, get_chat_client


CITATION_RE = re.compile(r"\[(?P<pid>[^\]#\s]+)#(?P<cid>\d+)\]")
_BOUNDARY_PATTERNS = (". ", "! ", "? ")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "Fig.", "Eq.", "Sec.", "No.", "vs.")
logger = logging.getLogger(__name__)


class SynthesizerError(RuntimeError):
    pass


async def synthesize_node(state: AgentState) -> dict:
    evidence = _select_synthesis_evidence(state.get("evidence", []))
    failure_notes = _system_notes(state)
    question = state.get("question", "")
    try:
        answer, finish_reason = await _generate_answer(question, evidence, failure_notes=failure_notes)
    except EmptyProviderResponseError as exc:
        logger.warning("synthesis_empty_provider_response", extra={"question": question[:160], "error": str(exc)})
        return _degraded_update("", evidence, finish_reason="empty_provider_response")
    if finish_reason == "length":
        return _degraded_update(answer, evidence, finish_reason=finish_reason)

    citations = _parse_citations(answer)
    if not citations:
        try:
            answer, finish_reason = await _fix_citation_format(answer, question, evidence)
        except EmptyProviderResponseError as exc:
            logger.warning("synthesis_repair_empty_provider_response", extra={"question": question[:160], "error": str(exc)})
            return _degraded_update(answer, evidence, finish_reason="empty_provider_response")
        citations = _parse_citations(answer)
        if not citations:
            return _degraded_update(answer, evidence, finish_reason=finish_reason)

    return {
        "answer": answer,
        "citations": citations,
        "evidence_lookup": {hit.chunk_id: hit.text for hit in evidence},
        "synthesis_format_degraded": False,
        "synthesis_finish_reason": finish_reason,
    }


async def _generate_answer(question: str, evidence, *, failure_notes: list[str]) -> tuple[str, str]:
    client = get_chat_client("main")
    messages = _generate_messages(question, evidence, failure_notes=failure_notes)
    response = await client.chat(
        messages,
        temperature=0.0,
        max_tokens=2000,
    )
    log_raw_response(
        "synthesize_initial",
        response.content,
        evidence_count=len(evidence),
        available_chunk_ids_count=len(evidence),
        finish_reason=response.finish_reason,
    )
    _log_non_stop_finish("synthesize_initial", response.finish_reason, question)
    if response.finish_reason == "length":
        retry_messages = [
            *messages,
            Message(
                role="system",
                content=(
                    "Your previous response was truncated. Produce the same answer in fewer sentences "
                    "while preserving all citations."
                ),
            ),
        ]
        retry = await client.chat(
            retry_messages,
            temperature=0.0,
            max_tokens=2000,
        )
        log_raw_response(
            "synthesize_initial_compact_retry",
            retry.content,
            evidence_count=len(evidence),
            available_chunk_ids_count=len(evidence),
            finish_reason=retry.finish_reason,
        )
        _log_non_stop_finish("synthesize_initial_compact_retry", retry.finish_reason, question)
        return (retry.content or "").strip(), retry.finish_reason
    return (response.content or "").strip(), response.finish_reason


async def _fix_citation_format(answer: str, question: str, evidence) -> tuple[str, str]:
    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(
                role="system",
                content=(
                    "The previous answer is missing required [paper_id#chunk_id] citations. "
                    "Rewrite it as 4-6 sentences in a single paragraph, where every substantive claim ends with the canonical [paper_id#chunk_id] identifier shown at the top of an evidence chunk. "
                    "If the previous answer is empty, write a fresh concise answer using the supplied evidence. "
                    "Use ONLY chunk identifiers from the evidence section. Do not use [paper_id]-only or [chunk_id]-only forms. "
                    "If the previous answer contains substantive prose without citations, preserve the meaning while attaching citations to each claim."
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
        max_tokens=2000,
    )
    log_raw_response(
        "synthesize_repair",
        response.content,
        evidence_count=len(evidence),
        available_chunk_ids_count=len(evidence),
        finish_reason=response.finish_reason,
    )
    _log_non_stop_finish("synthesize_repair", response.finish_reason, question)
    return (response.content or "").strip(), response.finish_reason


def _generate_messages(question: str, evidence, *, failure_notes: list[str]) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "Answer research questions using ONLY the supplied evidence. "
                "Format requirement: 4-6 sentences in a single paragraph (no bullet lists, no markdown headers, no bold/italic emphasis). "
                "MANDATORY CITATION RULE: every substantive claim MUST end with an inline [paper_id#chunk_id] citation referencing the evidence chunk that supports it. "
                "Use the bracketed identifier shown at the top of each evidence chunk verbatim — "
                "for example, after a claim derived from the chunk headed [arxiv:2210.03629#123], end the sentence with [arxiv:2210.03629#123]. "
                "If you cannot cite a claim with the supplied evidence, do NOT make that claim. "
                "An answer containing zero [paper_id#chunk_id] citations is invalid output. "
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
    ]


def _format_evidence(evidence) -> str:
    return "\n".join(
        f"[{hit.paper_id}#{hit.chunk_id}] (section={hit.section or 'Unknown'})\n{_truncate_evidence_text(hit.text)}\n---"
        for hit in evidence
    )


def _select_synthesis_evidence(evidence):
    return sorted(evidence, key=lambda hit: hit.score, reverse=True)[:SYNTHESIS_EVIDENCE_MAX_CHUNKS]


def _truncate_evidence_text(text: str) -> str:
    if len(text) <= SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK:
        return text
    return f"{text[:SYNTHESIS_EVIDENCE_CHARS_PER_CHUNK]}...[truncated from {len(text)} chars]"


def _log_non_stop_finish(event: str, finish_reason: str, question: str) -> None:
    if finish_reason and finish_reason != "stop":
        logger.warning(
            "synthesis_non_stop_finish",
            extra={"event": event, "finish_reason": finish_reason, "question": question[:160]},
        )


def _parse_citations(answer: str) -> list[Citation]:
    result: list[Citation] = []
    for match in CITATION_RE.finditer(answer):
        sent_start, sent_end = _sentence_around(answer, match.start(), match.end())
        sentence = answer[sent_start:sent_end].strip()
        result.append(
            Citation(
                paper_id=match.group("pid"),
                chunk_id=int(match.group("cid")),
                claim_text=sentence,
                claim_span=(sent_start, sent_end),
                quote=None,
            )
        )
    return result


def _sentence_around(text: str, start: int, end: int) -> tuple[int, int]:
    if not text:
        return (0, 0)
    sent_start = _find_sentence_start(text, start)
    sent_end = _find_sentence_end(text, end)
    if sent_start is None and sent_end is None:
        return (max(0, start - 200), min(len(text), end + 200))
    return (0 if sent_start is None else sent_start, len(text) if sent_end is None else sent_end)


def _find_sentence_start(text: str, start: int) -> int | None:
    best: int | None = None
    for pattern in _BOUNDARY_PATTERNS:
        search_from = start
        while True:
            index = text.rfind(pattern, 0, search_from)
            if index < 0:
                break
            if _is_abbreviation_boundary(text, index):
                search_from = index
                continue
            candidate = index + len(pattern)
            if best is None or candidate > best:
                best = candidate
            break
    return best


def _find_sentence_end(text: str, end: int) -> int | None:
    best: int | None = None
    for pattern in _BOUNDARY_PATTERNS:
        search_from = end
        while True:
            index = text.find(pattern, search_from)
            if index < 0:
                break
            if _is_abbreviation_boundary(text, index):
                search_from = index + len(pattern)
                continue
            candidate = index + 1
            if best is None or candidate < best:
                best = candidate
            break
    return best


def _is_abbreviation_boundary(text: str, period_index: int) -> bool:
    prefix = text[: period_index + 1]
    return any(prefix.endswith(abbreviation) for abbreviation in _ABBREVIATIONS)


def _degraded_update(answer: str, evidence, *, finish_reason: str = "") -> dict:
    return {
        "answer": answer,
        "citations": [],
        "evidence_lookup": {hit.chunk_id: hit.text for hit in evidence},
        "synthesis_format_degraded": True,
        "synthesis_finish_reason": finish_reason,
    }


def _system_notes(state: AgentState) -> list[str]:
    notes: list[str] = []
    for message in state.get("messages", []):
        if getattr(message, "type", None) == "system":
            content = getattr(message, "content", "")
            if isinstance(content, str):
                notes.append(content)
    return notes
