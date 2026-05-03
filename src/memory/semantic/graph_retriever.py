from __future__ import annotations

import re

from pydantic import BaseModel, Field

from src.core import db
from src.core.types import GraphExpansion, Message, StoredConcept
from src.llm.client import get_chat_client, get_embedding_client
from src.memory.semantic import graph_store
from src.memory.semantic.graph_store import _canonical


MENTION_PROMPT = """List noun phrases in the question that could be technical concepts.
Reply JSON {"phrases": ["..."]}.
"""
NEIGHBOR_REL_TYPES = ["extends", "compares", "is-a", "uses"]
EMBEDDING_MATCH_THRESHOLD = 0.85


async def expand_query_via_graph(question: str) -> GraphExpansion:
    phrases = await _detect_concept_phrases(question)
    seeds: list[StoredConcept] = []
    seen_seed_ids: set[int] = set()
    for phrase in phrases:
        concept = await _resolve_phrase(phrase)
        if concept is None or concept.concept_id in seen_seed_ids:
            continue
        seen_seed_ids.add(concept.concept_id)
        seeds.append(concept)

    if not seeds:
        return GraphExpansion()

    neighbors = []
    seen_neighbor_ids: set[int] = set()
    for seed in seeds:
        for neighbor in await graph_store.get_neighbors(seed.concept_id, rel_types=NEIGHBOR_REL_TYPES, limit=20):
            if neighbor.concept.concept_id in seen_neighbor_ids:
                continue
            seen_neighbor_ids.add(neighbor.concept.concept_id)
            neighbors.append(neighbor)

    concept_ids = [seed.concept_id for seed in seeds]
    concept_ids.extend(neighbor.concept.concept_id for neighbor in neighbors)
    paper_hints = await _paper_hints_for_concepts(concept_ids)
    return GraphExpansion(
        seed_concept_ids=[seed.concept_id for seed in seeds],
        neighbor_concept_ids=[neighbor.concept.concept_id for neighbor in neighbors],
        extra_paper_hints=paper_hints,
    )


async def _detect_concept_phrases(question: str) -> list[str]:
    client = get_chat_client("fast")
    response = await client.chat(
        [
            Message(role="system", content=MENTION_PROMPT),
            Message(role="user", content=question),
        ],
        temperature=0.0,
        max_tokens=240,
        response_format={"type": "json_object"},
    )
    payload = _MentionPayload.model_validate_json(_extract_json_object(response.content or ""))
    return [_canonical(phrase) for phrase in payload.phrases if _canonical(phrase)]


async def _resolve_phrase(phrase: str) -> StoredConcept | None:
    canonical = _canonical(phrase)
    alias_hit = await graph_store.get_concept(by_alias=canonical)
    if alias_hit is not None:
        return alias_hit
    canonical_hit = await graph_store.get_concept(by_canonical=canonical)
    if canonical_hit is not None:
        return canonical_hit
    embedding = (await get_embedding_client("embed-small").embed([phrase]))[0]
    return await _resolve_by_embedding(embedding)


async def _resolve_by_embedding(embedding: list[float]) -> StoredConcept | None:
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
    if row is None or float(row["similarity"]) < EMBEDDING_MATCH_THRESHOLD:
        return None
    return graph_store._row_to_concept(row)


async def _paper_hints_for_concepts(concept_ids: list[int]) -> list[str]:
    if not concept_ids:
        return []
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            WITH ec AS (
              SELECT (jsonb_array_elements_text(evidence_chunks))::bigint AS chunk_id
              FROM concept_relations
              WHERE src_id = ANY($1) OR dst_id = ANY($1)
            )
            SELECT DISTINCT c.paper_id
            FROM ec
            JOIN chunks c ON c.chunk_id = ec.chunk_id
            ORDER BY c.paper_id
            """,
            concept_ids,
        )
    return [row["paper_id"] for row in rows]


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


class _MentionPayload(BaseModel):
    phrases: list[str] = Field(default_factory=list)
