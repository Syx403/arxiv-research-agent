from __future__ import annotations

import logging

from src.core import db
from src.core.types import ConceptExtraction, ReflectionReport, StoredMessage
from src.memory.episodic import session_store
from src.memory.semantic import extractor, graph_store, linker
from src.memory.semantic.graph_store import _canonical


logger = logging.getLogger(__name__)


async def reflect_session(session_id: str) -> ReflectionReport:
    messages = await _fetch_assistant_messages(session_id)
    concepts_created = 0
    concepts_reused = 0
    relations_created = 0
    relations_reused = 0
    linked_by_canonical: dict[str, int] = {}

    for message in messages:
        chunk_ids = _citation_chunk_ids(message.metadata)
        if not chunk_ids:
            continue
        chunks = await _fetch_chunks(chunk_ids)
        if not chunks:
            continue
        combined_text = _combined_text(message.content, chunks)
        extractions = await extractor.extract_concepts_from_text(
            combined_text,
            source_chunk_ids=sorted(chunks),
        )
        for extraction in extractions:
            result = await linker.link_concept(extraction)
            linked_by_canonical[_canonical(extraction.display_name)] = result.concept.concept_id
            if result.created:
                concepts_created += 1
            else:
                concepts_reused += 1

        for extraction in extractions:
            src_id = linked_by_canonical[_canonical(extraction.display_name)]
            for relation in extraction.relations:
                target_canonical = _canonical(relation.target_display)
                target_id = linked_by_canonical.get(target_canonical)
                if target_id is None:
                    target = ConceptExtraction(
                        display_name=relation.target_display,
                        definition="",
                        aliases=[],
                        evidence_chunk_ids=relation.evidence_chunk_ids or extraction.evidence_chunk_ids,
                        relations=[],
                    )
                    target_result = await linker.link_concept(target)
                    target_id = target_result.concept.concept_id
                    linked_by_canonical[target_canonical] = target_id
                    if target_result.created:
                        concepts_created += 1
                    else:
                        concepts_reused += 1
                relation_created = await graph_store.upsert_relation(
                    src_id,
                    target_id,
                    relation.type,
                    relation.evidence_chunk_ids or extraction.evidence_chunk_ids,
                )
                if relation_created:
                    relations_created += 1
                else:
                    relations_reused += 1

    return ReflectionReport(
        concepts_created=concepts_created,
        concepts_reused=concepts_reused,
        relations_created=relations_created,
        relations_reused=relations_reused,
    )


async def _fetch_assistant_messages(session_id: str) -> list[StoredMessage]:
    return await session_store.get_messages(session_id, role="assistant")


async def _fetch_chunks(chunk_ids: list[int]) -> dict[int, str]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            SELECT chunk_id, text
            FROM chunks
            WHERE chunk_id = ANY($1::bigint[])
            ORDER BY chunk_id
            """,
            sorted(set(chunk_ids)),
        )
    return {int(row["chunk_id"]): row["text"] for row in rows}


def _citation_chunk_ids(metadata: dict) -> list[int]:
    citations = metadata.get("citations")
    if not isinstance(citations, list):
        return []
    chunk_ids: list[int] = []
    for citation in citations:
        if not isinstance(citation, dict) or "chunk_id" not in citation:
            continue
        try:
            chunk_ids.append(int(citation["chunk_id"]))
        except (TypeError, ValueError):
            logger.warning("reflection_invalid_citation_chunk_id", extra={"chunk_id": citation.get("chunk_id")})
    return sorted(set(chunk_ids))


def _combined_text(answer: str, chunks: dict[int, str]) -> str:
    excerpts = "\n\n".join(f"Chunk {chunk_id}:\n{text}" for chunk_id, text in chunks.items())
    return f"Assistant answer:\n{answer}\n\nCited evidence:\n{excerpts}"
