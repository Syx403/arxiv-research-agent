from __future__ import annotations

import json
from typing import Any

from src.core import db
from src.core.types import Neighbor, StoredConcept


def _canonical(s: str) -> str:
    return " ".join(s.lower().strip().split())


async def upsert_concept(
    display: str,
    *,
    definition: str | None = None,
    evidence_chunk_ids: list[int] | None = None,
    embedding: list[float] | None = None,
    aliases: list[str] | None = None,
) -> StoredConcept:
    canonical = _canonical(display)
    evidence = _dedupe_ints(evidence_chunk_ids or [])
    async with db.acquire_app() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO concepts (canonical, display, definition, embedding, evidence_chunks)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            ON CONFLICT (canonical) DO UPDATE SET
              display = EXCLUDED.display,
              definition = COALESCE(EXCLUDED.definition, concepts.definition),
              evidence_chunks = (
                SELECT COALESCE(jsonb_agg(DISTINCT v.value), '[]'::jsonb)
                FROM jsonb_array_elements(concepts.evidence_chunks || EXCLUDED.evidence_chunks) AS v(value)
              )
            RETURNING concept_id, canonical, display, definition, evidence_chunks
            """,
            canonical,
            display,
            definition,
            embedding,
            json.dumps(evidence),
        )
        for alias in [canonical, *(aliases or [])]:
            await conn.execute(
                """
                INSERT INTO concept_aliases (alias, concept_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                _canonical(alias),
                row["concept_id"],
            )
    return _row_to_concept(row)


async def upsert_relation(
    src_concept_id: int,
    dst_concept_id: int,
    rel_type: str,
    evidence_chunk_ids: list[int],
) -> bool:
    evidence = json.dumps(_dedupe_ints(evidence_chunk_ids))
    async with db.acquire_app() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO concept_relations (src_id, dst_id, rel_type, evidence_chunks)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (src_id, dst_id, rel_type) DO UPDATE SET
              evidence_chunks = (
                SELECT COALESCE(jsonb_agg(DISTINCT v.value), '[]'::jsonb)
                FROM jsonb_array_elements(concept_relations.evidence_chunks || EXCLUDED.evidence_chunks) AS v(value)
              )
            RETURNING (xmax = 0) AS created
            """,
            src_concept_id,
            dst_concept_id,
            rel_type,
            evidence,
        )
    return bool(row["created"])


async def get_concept(
    *,
    by_id: int | None = None,
    by_canonical: str | None = None,
    by_alias: str | None = None,
) -> StoredConcept | None:
    provided = [by_id is not None, by_canonical is not None, by_alias is not None]
    if sum(provided) != 1:
        raise ValueError("Exactly one concept lookup key is required")

    async with db.acquire_app() as conn:
        if by_id is not None:
            row = await conn.fetchrow(
                """
                SELECT concept_id, canonical, display, definition, evidence_chunks
                FROM concepts
                WHERE concept_id = $1
                """,
                by_id,
            )
        elif by_canonical is not None:
            row = await conn.fetchrow(
                """
                SELECT concept_id, canonical, display, definition, evidence_chunks
                FROM concepts
                WHERE canonical = $1
                """,
                _canonical(by_canonical),
            )
        else:
            row = await conn.fetchrow(
                """
                SELECT c.concept_id, c.canonical, c.display, c.definition, c.evidence_chunks
                FROM concept_aliases a
                JOIN concepts c ON c.concept_id = a.concept_id
                WHERE a.alias = $1
                """,
                _canonical(by_alias or ""),
            )
    return _row_to_concept(row) if row is not None else None


async def get_neighbors(
    concept_id: int,
    rel_types: list[str] | None = None,
    limit: int = 20,
) -> list[Neighbor]:
    async with db.acquire_app() as conn:
        rows = await conn.fetch(
            """
            WITH edges AS (
              SELECT dst_id AS neighbor_id, rel_type, weight, evidence_chunks, 'out' AS direction
              FROM concept_relations
              WHERE src_id = $1
                AND ($2::text[] IS NULL OR rel_type = ANY($2))
              UNION ALL
              SELECT src_id AS neighbor_id, rel_type, weight, evidence_chunks, 'in' AS direction
              FROM concept_relations
              WHERE dst_id = $1
                AND ($2::text[] IS NULL OR rel_type = ANY($2))
            )
            SELECT c.concept_id, c.canonical, c.display, c.definition, c.evidence_chunks AS concept_evidence,
                   e.rel_type, e.weight, e.evidence_chunks AS relation_evidence, e.direction
            FROM edges e
            JOIN concepts c ON c.concept_id = e.neighbor_id
            ORDER BY e.weight DESC, c.concept_id
            LIMIT $3
            """,
            concept_id,
            rel_types,
            limit,
        )
    return [
        Neighbor(
            concept=StoredConcept(
                concept_id=row["concept_id"],
                canonical=row["canonical"],
                display=row["display"],
                definition=row["definition"],
                evidence_chunk_ids=_jsonb_ints(row["concept_evidence"]),
            ),
            rel_type=row["rel_type"],
            weight=float(row["weight"]),
            evidence_chunk_ids=_jsonb_ints(row["relation_evidence"]),
        )
        for row in rows
    ]


def _row_to_concept(row: Any) -> StoredConcept:
    return StoredConcept(
        concept_id=row["concept_id"],
        canonical=row["canonical"],
        display=row["display"],
        definition=row["definition"],
        evidence_chunk_ids=_jsonb_ints(row["evidence_chunks"]),
    )


def _jsonb_ints(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return _dedupe_ints(int(item) for item in parsed)


def _dedupe_ints(values) -> list[int]:
    return sorted({int(value) for value in values})
