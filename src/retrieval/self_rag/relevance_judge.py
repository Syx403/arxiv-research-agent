from __future__ import annotations

import asyncio

from pydantic import ValidationError

from src.core.diagnostics import log_raw_response
from src.core.types import Message, RelevanceVerdict
from src.llm.client import get_chat_client
from src.retrieval.index.types import Hit


async def judge_relevance(subq: str, hit: Hit) -> RelevanceVerdict:
    client = get_chat_client("fast")
    response = await client.chat(
        [
            Message(
                role="system",
                content='Judge retrieval relevance. Return only JSON: {"relevant": true, "rationale": "..."}',
            ),
            Message(
                role="user",
                content=f"Sub-question:\n{subq}\n\nChunk:\n{hit.text}",
            ),
        ],
        temperature=0.0,
        max_tokens=260,
        response_format={"type": "json_object"},
    )
    log_raw_response(
        "relevance_judge",
        response.content,
        chunk_id=hit.chunk_id,
        event="initial",
    )
    broken_output = response.content or ""
    try:
        return RelevanceVerdict.model_validate_json(broken_output)
    except ValidationError:
        repair = await client.chat(
            [
                Message(
                    role="system",
                    content='Fix invalid JSON. Return compact valid JSON only: {"relevant": true, "rationale": "..."}',
                ),
                Message(role="user", content=f"Invalid output:\n{broken_output}"),
            ],
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        try:
            return RelevanceVerdict.model_validate_json(repair.content or "")
        except ValidationError:
            log_raw_response(
                "relevance_judge",
                repair.content,
                chunk_id=hit.chunk_id,
                event="parse_failure_twice",
            )
            return RelevanceVerdict(
                relevant=False,
                rationale="PARSE_FAILURE: relevance judge returned invalid JSON; dropping hit.",
            )


async def judge_batch(subq: str, hits: list[Hit]) -> list[RelevanceVerdict]:
    semaphore = asyncio.Semaphore(8)

    async def run_one(hit: Hit) -> RelevanceVerdict:
        async with semaphore:
            return await judge_relevance(subq, hit)

    return await asyncio.gather(*(run_one(hit) for hit in hits))
