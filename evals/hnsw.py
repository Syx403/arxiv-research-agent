"""HNSW against the exact scan for the library search (D27, D29): free and read-only. The S1
questions, whose vectors are cached, are run over the whole corpus and over a probe library (every
second stored paper, added in a transaction that is rolled back). Reported: overlap of the top 10
chunks, how often the library's top three papers agree, median latency, and the plan the planner
picks when nothing is forced. `uv run ara eval hnsw`."""

import json
import statistics
import time
from typing import Any, LiteralString

from ara.db.pool import Connection, make_pool
from ara.memory.library import CANDIDATES
from ara.rag.embed import cache_key
from ara.rag.ingest import PIPELINE_VERSION
from ara.settings import get_settings
from evals.suites import s1

TOP, PAPERS, PROBE = 10, 3, "hnsw-probe"
WHOLE: LiteralString = """
SELECT c.id, d.paper_id FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY c.embedding <=> %(vector)s LIMIT %(limit)s
"""
# the product's library query (ara/memory/library.py NEAREST), returning chunk ids as well
LIBRARY: LiteralString = """
SELECT c.id, d.paper_id FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.pipeline_version = %(pipeline)s
  AND d.paper_id IN (SELECT paper_id FROM library_items WHERE user_id = %(user)s)
ORDER BY c.embedding <=> %(vector)s LIMIT %(limit)s
"""
# Each mode sets every switch, since SET LOCAL lasts to the end of the transaction. Forcing HNSW
# needs sorting off too: with sequential scans off alone the planner sorts an index scan on
# chunks(document_id) instead.
MODES: dict[str, tuple[LiteralString, ...]] = {
    "exact": (
        "SET LOCAL enable_indexscan = off",
        "SET LOCAL enable_seqscan = on",
        "SET LOCAL enable_sort = on",
    ),
    "hnsw": (
        "SET LOCAL enable_indexscan = on",
        "SET LOCAL enable_seqscan = off",
        "SET LOCAL enable_sort = off",
    ),
    "planner": (
        "SET LOCAL enable_indexscan = on",
        "SET LOCAL enable_seqscan = on",
        "SET LOCAL enable_sort = on",
    ),
}


async def measure() -> dict[str, Any]:
    items, _ = s1.load()
    settings = get_settings()
    async with make_pool(settings.database_url) as pool, pool.connection() as conn:
        vectors = [await _vector(conn, item.question) for item in items]
        async with conn.transaction(force_rollback=True):
            await conn.execute(
                "INSERT INTO library_items (user_id, paper_id) SELECT %s, id FROM"
                " (SELECT id, row_number() OVER (ORDER BY id) AS n FROM papers) p WHERE n %% 2 = 0",
                (PROBE,),
            )
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            size = await conn.execute(
                "SELECT count(*) AS chunks, count(DISTINCT d.paper_id) AS papers FROM chunks c"
                " JOIN documents d ON d.id = c.document_id"
                " WHERE d.paper_id IN (SELECT paper_id FROM library_items WHERE user_id = %s)",
                (PROBE,),
            )
            library_size = await size.fetchone()
            report = {
                "questions": len(vectors),
                "library": library_size,
                "whole": await _compare(conn, WHOLE, vectors),
                "library_filtered": await _compare(conn, LIBRARY, vectors),
            }
    return report


async def _vector(conn: Connection, text: str) -> Any:
    cursor = await conn.execute(
        "SELECT embedding FROM embedding_cache WHERE key = %s", (cache_key(text),)
    )
    row = await cursor.fetchone()
    if row is None:
        raise SystemExit(f"no cached vector for {text!r}: run S1 first")
    return row["embedding"]


async def _compare(conn: Connection, sql: LiteralString, vectors: list[Any]) -> dict[str, Any]:
    runs: dict[str, list[list[tuple[int, str]]]] = {}
    latency: dict[str, float] = {}
    plans: dict[str, bool] = {}
    for mode, settings in MODES.items():
        for setting in settings:
            await conn.execute(setting)
        params = [_params(v) for v in vectors]
        plans[mode] = await _uses_hnsw(conn, sql, params[0])
        times, results = [], []
        for p in params:
            start = time.perf_counter()
            rows = await (await conn.execute(sql, p)).fetchall()
            times.append((time.perf_counter() - start) * 1000)
            results.append([(row["id"], row["paper_id"]) for row in rows])
        runs[mode], latency[mode] = results, statistics.median(times)
    exact, hnsw = runs["exact"], runs["hnsw"]
    overlap = [
        len({c for c, _ in a[:TOP]} & {c for c, _ in b[:TOP]}) / TOP
        for a, b in zip(exact, hnsw, strict=True)
    ]
    same = [_papers(a) == _papers(b) for a, b in zip(exact, hnsw, strict=True)]
    return {
        "top10_overlap": round(statistics.mean(overlap), 3),
        "same_top3_papers": round(sum(same) / len(same), 3),
        "median_ms": {m: round(v, 2) for m, v in latency.items()},
        "uses_hnsw": plans,
    }


def _params(vector: Any) -> dict[str, Any]:
    return {"vector": vector, "limit": CANDIDATES, "pipeline": PIPELINE_VERSION, "user": PROBE}


def _papers(rows: list[tuple[int, str]]) -> list[str]:
    return list(dict.fromkeys(p for _, p in rows))[:PAPERS]


async def _uses_hnsw(conn: Connection, sql: LiteralString, params: dict[str, Any]) -> bool:
    cursor = await conn.execute("EXPLAIN (FORMAT JSON) " + sql, params)
    row = await cursor.fetchone()
    return row is not None and "chunks_embedding_hnsw" in json.dumps(row["QUERY PLAN"])
