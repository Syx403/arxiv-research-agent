"""Ingestion and search against the real test database. Embeddings are placed in the cache first, so
the gateway is never called: no API request is made."""

from dataclasses import replace
from datetime import date

import numpy as np
from langgraph.runtime import Runtime
from pydantic import SecretStr

from ara.arxiv.client import ArxivClient
from ara.db.pool import Pool
from ara.graph import app
from ara.graph.read import ingest_paper
from ara.graph.state import Context
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.rag import retrieve, search
from ara.rag.chunking import chunk
from ara.rag.embed import cache_key, embed
from ara.rag.ingest import ingest, set_published, undated
from ara.rag.sources import Paragraph, ParsedPaper
from ara.settings import Settings

PARAGRAPHS = (
    Paragraph("Cache › Abstract", "We study cache eviction."),
    Paragraph("Cache › Method", "Evicting keys by attention score reduces memory."),
    Paragraph("Cache › Results", "Tool calls are scheduled in parallel."),
)


def paper(arxiv_id: str) -> ParsedPaper:
    return ParsedPaper(f"arxiv:{arxiv_id}v1", "arxiv", arxiv_id, 1, "Cache", "", "html", PARAGRAPHS)


def unit(i: int) -> np.ndarray:
    vector = np.zeros(1536, dtype=np.float32)
    vector[i] = 1.0
    return vector


async def setup(pool: Pool) -> tuple[Gateway, int, int]:
    """Two identical papers; paragraph i embeds as the i-th unit vector."""
    async with pool.connection() as conn:
        await conn.cursor().executemany(
            "INSERT INTO embedding_cache (key, model, embedding) VALUES (%s, %s, %s)",
            [(cache_key(c.search_text), "test", unit(c.paragraph)) for c in chunk(PARAGRAPHS)],
        )
    settings = Settings(
        openai_api_key=SecretStr("unused"),
        deepseek_api_key=SecretStr("unused"),
        cohere_api_key=SecretStr("unused"),
    )
    ledger = Ledger(
        pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
    )
    gateway = Gateway(settings, ledger)
    first = await ingest(paper("2401.00001"), pool=pool, gateway=gateway, scope=Scope())
    second = await ingest(paper("2401.00002"), pool=pool, gateway=gateway, scope=Scope())
    return gateway, first, second


async def paragraphs(pool: Pool, chunk_ids: list[int]) -> list[int]:
    async with pool.connection() as conn:
        cursor = await conn.execute("SELECT id, paragraph FROM chunks")
        where = {row["id"]: row["paragraph"] for row in await cursor.fetchall()}
    return [where[c] for c in chunk_ids]


async def test_ingest_is_idempotent(pool: Pool) -> None:
    gateway, first, _ = await setup(pool)
    again = await ingest(paper("2401.00001"), pool=pool, gateway=gateway, scope=Scope())
    async with pool.connection() as conn:
        count = await (await conn.execute("SELECT count(*) AS n FROM chunks")).fetchone()
    assert again == first
    assert count is not None and count["n"] == 6
    await gateway.aclose()


async def test_the_read_graph_does_not_fetch_a_stored_paper_again(pool: Pool) -> None:
    gateway, first, _ = await setup(pool)

    async def fetch(reference: str) -> ParsedPaper:
        raise AssertionError(f"fetched {reference}")

    async with ArxivClient() as arxiv:  # sends nothing
        runtime = Runtime(context=Context(pool, gateway, Scope(), fetch, arxiv))
        stored = await ingest_paper({"reference": "arxiv:2401.00001v1"}, runtime)
    assert stored == {"documents": [first]}
    await gateway.aclose()


async def test_stemming_widens_bm25_and_search_stays_in_scope(pool: Pool) -> None:
    gateway, first, _ = await setup(pool)
    async with pool.connection() as conn:
        plain = await search.bm25(conn, "eviction", [first], stemmed=False)
        stemmed = await search.bm25(conn, "eviction", [first], stemmed=True)
        fts = await search.fts(conn, "evicting keys", [first])
        dense = await search.dense(conn, unit(2) + np.float32(0.1) * unit(1), [first])
    assert await paragraphs(pool, plain) == [0]
    assert sorted(await paragraphs(pool, stemmed)) == [0, 1]
    assert sorted(await paragraphs(pool, fts)) == [0, 1]
    assert await paragraphs(pool, dense) == [2, 1, 0]
    await gateway.aclose()


def test_rrf_rewards_items_ranked_high_in_several_lists() -> None:
    assert search.rrf([[1, 2, 3], [3, 1, 4]]) == [1, 3, 2, 4]


async def test_cached_embeddings_come_back_as_numpy_vectors(pool: Pool) -> None:
    gateway, _, _ = await setup(pool)
    [vector] = await embed(
        [chunk(PARAGRAPHS)[1].search_text], pool=pool, gateway=gateway, scope=Scope()
    )
    assert isinstance(vector, np.ndarray) and vector.shape == (1536,) and vector @ unit(1) == 1.0
    await gateway.aclose()


async def test_the_papers_just_read_become_cards_for_follow_ups(pool: Pool) -> None:
    gateway, first, second = await setup(pool)
    async with pool.connection() as conn:
        cards = await app.read_cards(conn, [second, first])
    assert sorted((c.arxiv_id, c.version, c.title) for c in cards) == [
        ("2401.00001", 1, "Cache"),
        ("2401.00002", 1, "Cache"),
    ]
    await gateway.aclose()


async def test_each_paper_brings_its_own_candidates_and_keeps_its_own_best(pool: Pool) -> None:
    """D30: candidates per paper, one reranked list, then each paper's first TOP_K."""
    gateway, first, second = await setup(pool)
    async with pool.connection() as conn:
        found = await retrieve.candidates(conn, "eviction", unit(1), [first, second])
        cursor = await conn.execute("SELECT id, document_id FROM chunks")
        owner = {row["id"]: row["document_id"] for row in await cursor.fetchall()}
    assert [owner[c] for c in found] == [first] * 3 + [second] * 3

    def passage(n: int, document: int) -> retrieve.Passage:
        return retrieve.Passage(n, document, "p", 0, "h", "t", "t", (0,))

    ranked = [passage(n, first) for n in range(10)] + [passage(10, second)]
    kept = retrieve.best_of_each(ranked, [first, second, 99])
    assert [len(kept[d]) for d in (first, second, 99)] == [retrieve.TOP_K, 1, 0]
    assert kept[first][0].chunk_id == 0, "the reranked order is kept within a paper"
    await gateway.aclose()


async def test_ingestion_stores_the_date_and_read_cards_show_it(pool: Pool) -> None:
    gateway, _, _ = await setup(pool)
    dated = replace(paper("2401.00003"), published=date(2023, 5, 23))
    document = await ingest(dated, pool=pool, gateway=gateway, scope=Scope())
    async with pool.connection() as conn:
        [card] = await app.read_cards(conn, [document])
        assert card.published == "2023-05-23"
        assert await undated(conn) == ["2401.00001", "2401.00002"]
        assert await set_published(conn, {"2401.00001": date(2024, 1, 1)}) == 1
        assert await undated(conn) == ["2401.00002"]
    await gateway.aclose()
