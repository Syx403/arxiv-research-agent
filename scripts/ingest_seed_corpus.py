from __future__ import annotations

import asyncio

from src.core import db
from src.eval.datasets.seed_papers import SEED_PAPER_IDS
from src.corpus.ingestion import IngestResult, ingest_paper_with_result
from src.llm.client import close_llm_clients


async def _ingest_one(arxiv_id: str, semaphore: asyncio.Semaphore) -> IngestResult:
    async with semaphore:
        try:
            result = await ingest_paper_with_result(arxiv_id)
            print(f"{arxiv_id}: {result.status}")
            return result
        except Exception as exc:
            print(f"{arxiv_id}: failed ({exc.__class__.__name__})")
            return IngestResult(paper_id=f"arxiv:{arxiv_id}", status="failed")


async def main() -> None:
    semaphore = asyncio.Semaphore(4)
    try:
        results = await asyncio.gather(
            *[_ingest_one(arxiv_id, semaphore) for arxiv_id in SEED_PAPER_IDS]
        )
        ingested = sum(1 for result in results if result.status == "ingested")
        skipped = sum(1 for result in results if result.status == "skipped")
        failed = sum(1 for result in results if result.status == "failed")
        chunks = sum(result.chunks_inserted for result in results)
        citations = sum(result.citations_inserted for result in results)
        print(
            "Summary: "
            f"papers ingested={ingested}, papers skipped={skipped}, papers failed={failed}, "
            f"chunks inserted={chunks}, citations inserted={citations}"
        )
    finally:
        await close_llm_clients()
        await db.close_pools()


if __name__ == "__main__":
    asyncio.run(main())
