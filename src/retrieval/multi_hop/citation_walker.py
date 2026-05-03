from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.core import db
from src.core.types import (
    FrontierItem,
    MULTI_HOP_FRONTIER_LIMIT,
    MULTI_HOP_MAX_DEPTH,
    MULTI_HOP_WALL_CLOCK_BUDGET_S,
    Message,
    WalkResult,
)
from src.corpus import ingestion
from src.llm.client import get_chat_client
from src.retrieval.multi_hop.frontier import Frontier


logger = logging.getLogger(__name__)
SCORE_BATCH_SIZE = 16


@dataclass(frozen=True)
class CitationCandidate:
    paper_id: str
    title: str | None
    abstract: str | None
    depth: int
    edge_type: str
    is_complete: bool


async def walk_citations(
    seed_paper_ids: list[str],
    question: str,
    *,
    max_depth: int = MULTI_HOP_MAX_DEPTH,
    frontier_limit: int = MULTI_HOP_FRONTIER_LIMIT,
    wall_clock_budget_s: float = MULTI_HOP_WALL_CLOCK_BUDGET_S,
) -> WalkResult:
    started_at = time.monotonic()
    frontier = Frontier()
    visited_order: list[str] = []
    newly_ingested: list[str] = []

    for paper_id in seed_paper_ids:
        normalized = _normalize_paper_id(paper_id)
        frontier.mark_visited(normalized)
        visited_order.append(normalized)

    initial_candidates: list[CitationCandidate] = []
    for paper_id in visited_order:
        if _budget_exceeded(started_at, wall_clock_budget_s):
            return _partial_result(visited_order, newly_ingested, "budget exceeded during seed expansion")
        initial_candidates.extend(await _fetch_candidates(paper_id, depth=1))
    await _score_and_enqueue(frontier, question, initial_candidates, frontier_limit, started_at, wall_clock_budget_s)

    while len(frontier) > 0:
        if _budget_exceeded(started_at, wall_clock_budget_s):
            return _partial_result(visited_order, newly_ingested, "budget exceeded before next frontier pop")
        item = frontier.pop()
        if item is None:
            break
        frontier.mark_visited(item.paper_id)
        visited_order.append(item.paper_id)

        is_complete = await _paper_is_complete(item.paper_id)
        if not is_complete:
            try:
                await ingestion.ingest_paper(_strip_arxiv_prefix(item.paper_id))
                newly_ingested.append(item.paper_id)
            except Exception as exc:
                logger.warning("multi_hop_lazy_ingest_failed", extra={"paper_id": item.paper_id, "error": str(exc)})
                continue

        if item.depth < max_depth:
            candidates = await _fetch_candidates(item.paper_id, depth=item.depth + 1)
            await _score_and_enqueue(frontier, question, candidates, frontier_limit, started_at, wall_clock_budget_s)

    return WalkResult(visited_paper_ids=visited_order, newly_ingested_paper_ids=newly_ingested)


async def _fetch_candidates(paper_id: str, *, depth: int) -> list[CitationCandidate]:
    async with db.acquire_app() as conn:
        outbound_rows = await conn.fetch(
            """
            SELECT c.cited_paper_id AS paper_id,
                   c.cited_title AS title,
                   p.abstract AS abstract,
                   p.ingestion_status = 'complete' AS is_complete
            FROM citations c
            LEFT JOIN papers p ON p.paper_id = c.cited_paper_id
            WHERE c.citing_paper_id = $1
            """,
            paper_id,
        )
        incoming_rows = await conn.fetch(
            """
            SELECT c.citing_paper_id AS paper_id,
                   p.title AS title,
                   p.abstract AS abstract,
                   p.ingestion_status = 'complete' AS is_complete
            FROM citations c
            JOIN papers p ON p.paper_id = c.citing_paper_id
            WHERE c.cited_paper_id = $1
              AND c.citing_paper_id LIKE 'arxiv:%'
            """,
            paper_id,
        )

    skipped_s2 = sum(1 for row in outbound_rows if str(row["paper_id"]).startswith("s2:"))
    skipped_title = sum(1 for row in outbound_rows if str(row["paper_id"]).startswith("title:"))
    logger.info(
        "multi_hop_skipped_non_arxiv_candidates",
        extra={"hop_depth": depth, "skipped_s2": skipped_s2, "skipped_title": skipped_title},
    )

    candidates: list[CitationCandidate] = []
    candidates.extend(
        _candidate_from_row(row, depth=depth, edge_type="outbound")
        for row in outbound_rows
        if str(row["paper_id"]).startswith("arxiv:")
    )
    candidates.extend(
        _candidate_from_row(row, depth=depth, edge_type="incoming")
        for row in incoming_rows
        if str(row["paper_id"]).startswith("arxiv:")
    )
    return candidates


async def _score_and_enqueue(
    frontier: Frontier,
    question: str,
    candidates: list[CitationCandidate],
    frontier_limit: int,
    started_at: float,
    wall_clock_budget_s: float,
) -> None:
    filtered = [
        candidate
        for candidate in candidates
        if candidate.paper_id not in frontier.visited and candidate.paper_id not in frontier.enqueued
    ]
    if not filtered or _budget_exceeded(started_at, wall_clock_budget_s):
        return

    scored = await _score_candidates(question, filtered)
    for candidate, score in sorted(scored, key=lambda item: item[1], reverse=True)[:frontier_limit]:
        frontier.add(FrontierItem(paper_id=candidate.paper_id, score=score, depth=candidate.depth))


async def _score_candidates(question: str, candidates: list[CitationCandidate]) -> list[tuple[CitationCandidate, float]]:
    semaphore = asyncio.Semaphore(8)

    async def score_batch(batch: list[CitationCandidate]) -> list[tuple[CitationCandidate, float]]:
        async with semaphore:
            return await _score_candidate_batch(question, batch)

    batches = [candidates[idx : idx + SCORE_BATCH_SIZE] for idx in range(0, len(candidates), SCORE_BATCH_SIZE)]
    scored_batches = await asyncio.gather(*(score_batch(batch) for batch in batches))
    return [item for batch in scored_batches for item in batch]


async def _score_candidate_batch(
    question: str,
    candidates: list[CitationCandidate],
) -> list[tuple[CitationCandidate, float]]:
    client = get_chat_client("fast")
    candidate_payload = [
        {
            "paper_id": candidate.paper_id,
            "edge_type": candidate.edge_type,
            "title": candidate.title or "",
            "abstract": candidate.abstract or "",
        }
        for candidate in candidates
    ]
    response = await client.chat(
        [
            Message(
                role="system",
                content=(
                    "Score each paper for relevance to the research question. "
                    'Reply only with JSON: {"scores":[{"paper_id":"arxiv:...", "score":0}]} '
                    "Use scores from 0 to 10."
                ),
            ),
            Message(
                role="user",
                content=f"Question:\n{question}\n\nPapers:\n{json.dumps(candidate_payload)}",
            ),
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )
    fallback = [(candidate, _heuristic_score(question, candidate)) for candidate in candidates]
    try:
        parsed = _ScoreBatchPayload.model_validate_json(_extract_json_object(response.content or ""))
    except Exception as exc:
        logger.warning("multi_hop_score_parse_failed", extra={"error": str(exc)})
        return fallback

    scores_by_id = {score.paper_id: max(0.0, min(10.0, score.score)) for score in parsed.scores}
    return [
        (candidate, scores_by_id.get(candidate.paper_id, fallback_score))
        for candidate, fallback_score in fallback
    ]


async def _paper_is_complete(paper_id: str) -> bool:
    async with db.acquire_app() as conn:
        return bool(
            await conn.fetchval(
                "SELECT ingestion_status = 'complete' FROM papers WHERE paper_id = $1",
                paper_id,
            )
        )


def _candidate_from_row(row, *, depth: int, edge_type: str) -> CitationCandidate:
    return CitationCandidate(
        paper_id=row["paper_id"],
        title=row["title"],
        abstract=row["abstract"],
        depth=depth,
        edge_type=edge_type,
        is_complete=bool(row["is_complete"]),
    )


def _normalize_paper_id(paper_id: str) -> str:
    cleaned = paper_id.strip()
    if cleaned.lower().startswith("arxiv:"):
        return f"arxiv:{cleaned.split(':', 1)[1]}"
    return f"arxiv:{cleaned}"


def _strip_arxiv_prefix(paper_id: str) -> str:
    return paper_id.split(":", 1)[1] if paper_id.lower().startswith("arxiv:") else paper_id


def _budget_exceeded(started_at: float, wall_clock_budget_s: float) -> bool:
    return time.monotonic() - started_at > wall_clock_budget_s


def _partial_result(visited: list[str], newly_ingested: list[str], reason: str) -> WalkResult:
    logger.warning("multi_hop_partial_return", extra={"reason": reason})
    return WalkResult(visited_paper_ids=visited, newly_ingested_paper_ids=newly_ingested)


class _ScorePayload(BaseModel):
    paper_id: str
    score: float = Field(ge=0, le=10)


class _ScoreBatchPayload(BaseModel):
    scores: list[_ScorePayload]


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


def _heuristic_score(question: str, candidate: CitationCandidate) -> float:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 3}
    candidate_text = f"{candidate.title or ''} {candidate.abstract or ''}".lower()
    matched = sum(1 for term in query_terms if term in candidate_text)
    score = min(10.0, float(matched * 2))
    if "reflexion" in candidate_text:
        score += 4.0
    if "react" in candidate_text:
        score += 2.0
    if candidate.edge_type == "outbound" and not candidate.is_complete:
        score += 0.5
    return min(10.0, score)
