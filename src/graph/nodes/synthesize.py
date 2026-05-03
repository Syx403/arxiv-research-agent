from __future__ import annotations

import re

from src.core.types import Citation, Message
from src.graph.state import AgentState
from src.llm.client import get_chat_client


CITATION_RE = re.compile(r"\[(?P<pid>[^\]#\s]+)#(?P<cid>\d+)\]")
MALFORMED_CITATION_RE = re.compile(r"\[[^\]#\s]+\]")


class SynthesizerError(RuntimeError):
    pass


async def synthesize_node(state: AgentState) -> dict:
    evidence = state.get("evidence", [])
    failure_notes = _system_notes(state)
    answer = await _generate_answer(state.get("question", ""), evidence, failure_notes=failure_notes)
    citations = _parse_citations(answer)
    if not citations or MALFORMED_CITATION_RE.search(answer):
        answer = await _fix_citation_format(answer, state.get("question", ""), evidence)
        citations = _parse_citations(answer)
        if not citations or MALFORMED_CITATION_RE.search(answer):
            raise SynthesizerError("Synthesizer failed to produce valid [paper_id#chunk_id] citations")

    return {
        "answer": answer,
        "citations": citations,
        "evidence_lookup": {hit.chunk_id: hit.text for hit in evidence},
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
                    "Every cited claim must end with [paper_id#chunk_id], for example [arxiv:2210.03629#123]. "
                    "Never use [paper_id]-only citations."
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
    return (response.content or "").strip()


async def _fix_citation_format(answer: str, question: str, evidence) -> str:
    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(
                role="system",
                content=(
                    "Fix citation formatting only. Every citation must be [paper_id#chunk_id]. "
                    "Do not invent chunk IDs; use the evidence headers."
                ),
            ),
            Message(
                role="user",
                content=(
                    f"Question:\n{question}\n\n"
                    f"Invalid answer:\n{answer}\n\n"
                    f"Evidence:\n{_format_evidence(evidence)}"
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    return (response.content or "").strip()


def _format_evidence(evidence) -> str:
    return "\n".join(
        f"[chunk_id={hit.chunk_id} paper_id={hit.paper_id} section={hit.section or 'Unknown'}]\n{hit.text}\n---"
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


def _system_notes(state: AgentState) -> list[str]:
    notes: list[str] = []
    for message in state.get("messages", []):
        if getattr(message, "type", None) == "system":
            content = getattr(message, "content", "")
            if isinstance(content, str):
                notes.append(content)
    return notes
