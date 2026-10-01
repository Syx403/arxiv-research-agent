from __future__ import annotations

from dataclasses import replace
import json

from src.core.types import RERANK_TOP_K
from src.llm.client import get_rerank_client
from src.retrieval.index.types import Hit


async def rerank(query: str, hits: list[Hit], top_n: int = RERANK_TOP_K) -> list[Hit]:
    if not hits:
        return []
    client = get_rerank_client("rerank")
    results = await client.rerank(
        query,
        [json.dumps({"title": hit.title, "section": hit.section, "content": hit.text}, ensure_ascii=False)
         for hit in hits],
        top_n=min(top_n, len(hits)),
    )
    reranked: list[Hit] = []
    for result in results:
        # Cohere relevance_score replaces RRF/vector scores; score units change here.
        reranked.append(replace(hits[result.index], score=result.score))
    return reranked
