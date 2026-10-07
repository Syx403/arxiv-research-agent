"""The dry-run plan reads only the local database: no API request is made."""

from decimal import Decimal
from typing import Literal

import numpy as np
from pydantic import SecretStr

from ara.db.pool import Pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.rag.chunking import chunk
from ara.rag.embed import cache_key
from ara.rag.ingest import ingest
from ara.rag.sources import Paragraph, ParsedPaper
from ara.settings import Settings
from evals.runner import plan

PARAGRAPHS = (Paragraph("T › Abstract", "We study caches."),)


def paper(source: Literal["arxiv", "qasper"]) -> ParsedPaper:
    if source == "arxiv":
        return ParsedPaper(
            "arxiv:1912.01214v1", source, "1912.01214", 1, "T", "", "html", PARAGRAPHS
        )
    return ParsedPaper(
        "qasper:1912.01214", source, "1912.01214", None, "T", "", "qasper", PARAGRAPHS
    )


async def test_plan_tells_an_arxiv_edition_from_the_qasper_one(pool: Pool) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO embedding_cache (key, model, embedding) VALUES (%s, %s, %s)",
            (cache_key(chunk(PARAGRAPHS)[0].search_text), "test", np.ones(1536, np.float32)),
        )
    keys = SecretStr("unused")
    settings = Settings(openai_api_key=keys, deepseek_api_key=keys, cohere_api_key=keys)
    gateway = Gateway(settings, Ledger(pool, global_cap_usd=Decimal(1), turn_cap_usd=Decimal(1)))
    await ingest(paper("arxiv"), pool=pool, gateway=gateway, scope=Scope())
    assert (await plan(pool, [], {"1912.01214": paper("qasper")})).papers_to_ingest == 1
    await ingest(paper("qasper"), pool=pool, gateway=gateway, scope=Scope())
    assert (await plan(pool, [], {"1912.01214": paper("qasper")})).papers_to_ingest == 0
    await gateway.aclose()
