from __future__ import annotations

import json
import logging
import re
from typing import NamedTuple

from pydantic import BaseModel

from src.core import db
from src.core.types import ConceptExtraction, ConceptLinkResult, Message, StoredConcept
from src.llm.client import get_chat_client, get_embedding_client
from src.memory.semantic import graph_store
from src.memory.semantic.graph_store import _canonical


logger = logging.getLogger(__name__)
EMBEDDING_MATCH_THRESHOLD = 0.85


async def link_concept(extraction: ConceptExtraction) -> ConceptLinkResult:
    canonical = _canonical(extraction.display_name)

    alias_hit = await graph_store.get_concept(by_alias=canonical)
    if alias_hit is not None:
        concept = await _merge_existing(alias_hit, extraction)
        return ConceptLinkResult(concept=concept, created=False, link_path="alias")

    canonical_hit = await graph_store.get_concept(by_canonical=canonical)
    if canonical_hit is not None:
        concept = await _merge_existing(canonical_hit, extraction)
        return ConceptLinkResult(concept=concept, created=False, link_path="canonical")

    embedding = await _embed_extraction(extraction)
    nearest = await _nearest_by_embedding(embedding)
    if nearest is not None and nearest.similarity >= EMBEDDING_MATCH_THRESHOLD:
        if await _judge_same_concept(extraction, nearest.concept):
            concept = await _merge_existing(nearest.concept, extraction)
            await _insert_aliases(concept.concept_id, [canonical, *extraction.aliases])
            return ConceptLinkResult(concept=concept, created=False, link_path="embedding")

    concept = await graph_store.upsert_concept(
        extraction.display_name,
        definition=extraction.definition,
        evidence_chunk_ids=extraction.evidence_chunk_ids,
        embedding=embedding,
    )
    await _insert_aliases(concept.concept_id, [canonical, *extraction.aliases])
    return ConceptLinkResult(concept=concept, created=True, link_path="new")


async def _merge_existing(concept: StoredConcept, extraction: ConceptExtraction) -> StoredConcept:
    merged_evidence = sorted({*concept.evidence_chunk_ids, *extraction.evidence_chunk_ids})
    async with db.acquire_app() as conn:
        await conn.execute(
            """
            UPDATE concepts
            SET evidence_chunks = $2::jsonb
            WHERE concept_id = $1
            """,
            concept.concept_id,
            json.dumps(merged_evidence),
        )
    await _insert_aliases(concept.concept_id, [_canonical(extraction.display_name), *extraction.aliases])
    return concept.model_copy(update={"evidence_chunk_ids": merged_evidence})


async def _insert_aliases(concept_id: int, aliases: list[str]) -> None:
    seen: set[str] = set()
    async with db.acquire_app() as conn:
        for raw_alias in aliases:
            alias = _canonical(raw_alias)
            if not alias or alias in seen:
                continue
            seen.add(alias)
            existing = await conn.fetchrow(
                "SELECT concept_id FROM concept_aliases WHERE alias = $1",
                alias,
            )
            if existing is not None:
                existing_id = int(existing["concept_id"])
                if existing_id != concept_id:
                    logger.warning(
                        "concept_alias_collision",
                        extra={"alias": alias, "existing_concept_id": existing_id, "new_concept_id": concept_id},
                    )
                continue
            await conn.execute(
                """
                INSERT INTO concept_aliases (alias, concept_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                alias,
                concept_id,
            )


async def _embed_extraction(extraction: ConceptExtraction) -> list[float]:
    text = f"{extraction.display_name}. {extraction.definition}"
    return (await get_embedding_client("embed-small").embed([text]))[0]


async def _nearest_by_embedding(embedding: list[float]) -> "_NearestConcept | None":
    async with db.acquire_app() as conn:
        row = await conn.fetchrow(
            """
            SELECT concept_id,
                   canonical,
                   display,
                   definition,
                   evidence_chunks,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM concepts
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> $1::vector
            LIMIT 1
            """,
            embedding,
        )
    if row is None:
        return None
    return _NearestConcept(
        concept=graph_store._row_to_concept(row),
        similarity=float(row["similarity"]),
    )


async def _judge_same_concept(extraction: ConceptExtraction, concept: StoredConcept) -> bool:
    client = get_chat_client("main")
    response = await client.chat(
        [
            Message(
                role="system",
                content='Decide whether two concept descriptions refer to the same concept. Reply JSON {"same": true}.',
            ),
            Message(
                role="user",
                content=(
                    "Candidate extraction:\n"
                    f"{json.dumps(extraction.model_dump(), sort_keys=True)}\n\n"
                    "Existing concept:\n"
                    f"{json.dumps(concept.model_dump(), sort_keys=True)}"
                ),
            ),
        ],
        temperature=0.0,
        max_tokens=120,
        response_format={"type": "json_object"},
    )
    parsed = _SameConceptPayload.model_validate_json(_extract_json_object(response.content or ""))
    return parsed.same


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).removesuffix("```").strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return stripped
    return stripped[start : end + 1]


class _NearestConcept(NamedTuple):
    concept: StoredConcept
    similarity: float


class _SameConceptPayload(BaseModel):
    same: bool
