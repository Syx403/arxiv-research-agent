from __future__ import annotations

from src.core import db
from src.core.types import RouteDecision


async def route(subq: str) -> RouteDecision:
    use_concept_graph = await _has_matching_concept_alias(subq)
    return RouteDecision(
        sources=["local_pgvector"],
        use_concept_graph=use_concept_graph,
        may_use_semantic_scholar_live=True,
    )


async def _has_matching_concept_alias(subq: str) -> bool:
    async with db.acquire_app() as conn:
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
              SELECT 1
              FROM concept_aliases
              WHERE POSITION(LOWER(alias) IN LOWER($1)) > 0
              LIMIT 1
            )
            """,
            subq,
        )
    return bool(exists)
