from __future__ import annotations

import asyncio

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
        max_tokens=180,
        response_format={"type": "json_object"},
    )
    return RelevanceVerdict.model_validate_json(response.content or "")


async def judge_batch(subq: str, hits: list[Hit]) -> list[RelevanceVerdict]:
    semaphore = asyncio.Semaphore(8)

    async def run_one(hit: Hit) -> RelevanceVerdict:
        async with semaphore:
            return await judge_relevance(subq, hit)

    return await asyncio.gather(*(run_one(hit) for hit in hits))
