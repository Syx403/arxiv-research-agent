"""Evaluation runner (DESIGN §11.5): dry run by default; with --execute it runs a suite through
LangSmith `aevaluate`, under a run cap in the ledger, and stores every result in Postgres."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.schemas import Example

from ara.db.migrate import migrate
from ara.db.pool import Pool, make_pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.llm.pricing import EMBEDDING, MILLION
from ara.rag.chunking import chunk
from ara.rag.embed import cache_key
from ara.rag.ingest import PIPELINE_VERSION
from ara.rag.sources import ParsedPaper
from ara.settings import Settings, configure_tracing, get_settings
from ara.tokens import count_tokens
from evals.graders.retrieval import score
from evals.suites import s1
from evals.suites.s1 import S1, Item


@dataclass(frozen=True)
class Plan:
    items: int
    papers_to_ingest: int
    embed_requests: int
    embed_tokens: int
    rerank_calls: int

    @property
    def usd(self) -> Decimal:
        return self.embed_tokens * EMBEDDING.input / MILLION

    def __str__(self) -> str:
        return (
            f"{self.items} items; {self.papers_to_ingest} papers to ingest;"
            f" ~{self.embed_requests} embedding requests ({self.embed_tokens:,} tokens),"
            f" {self.rerank_calls} rerank calls (Cohere trial, $0); estimated ${self.usd:.5f}"
        )


async def plan(pool: Pool, items: list[Item], papers: dict[str, ParsedPaper]) -> Plan:
    """What a run would send, from the local database alone: no API calls."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT p.arxiv_id FROM documents d JOIN papers p ON p.id = d.paper_id"
            " WHERE d.pipeline_version = %s",
            (PIPELINE_VERSION,),
        )
        ingested = {row["arxiv_id"] for row in await cursor.fetchall()}
        new = [paper for arxiv_id, paper in papers.items() if arxiv_id not in ingested]
        texts = {c.search_text for p in new for c in chunk(p.paragraphs)}
        texts |= {item.question for item in items}
        cursor = await conn.execute(
            "SELECT key FROM embedding_cache WHERE key = ANY(%s)",
            ([cache_key(text) for text in texts],),
        )
        cached = {bytes(row["key"]) for row in await cursor.fetchall()}
    missing = [text for text in texts if cache_key(text) not in cached]
    questions_missing = any(cache_key(item.question) not in cached for item in items)
    return Plan(
        items=len(items),
        papers_to_ingest=len(new),
        embed_requests=len(new) + int(questions_missing),
        embed_tokens=sum(count_tokens(text) for text in missing),
        rerank_calls=len(items),
    )


async def run_s1(*, papers: int | None, execute: bool, max_usd: Decimal) -> str | None:
    """Print the plan; with `execute`, run S1 and return the run id."""
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    items, parsed = s1.load(papers)
    async with make_pool(settings.database_url) as pool:
        estimate = await plan(pool, items, parsed)
        print(f"S1 plan: {estimate}")
        if not execute:
            print("Dry run: nothing was sent. Add --execute to run.")
            return None
        if estimate.usd > max_usd:
            raise SystemExit(f"estimated ${estimate.usd:.5f} exceeds --max-usd {max_usd}")
        run_id = f"s1-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        config = {
            "suite": "s1",
            "manifest": s1.manifest_hash(),
            "pipeline_version": PIPELINE_VERSION,
            "arms": list(s1.ARMS),
            "papers": list(parsed),
            "max_usd": str(max_usd),
        }
        await _start(pool, run_id, config)
        status = "failed"
        try:
            status = await _evaluate(pool, settings, run_id, config, items, parsed, max_usd)
        finally:
            await _finish(pool, run_id, status)
        return run_id


async def _evaluate(
    pool: Pool,
    settings: Settings,
    run_id: str,
    config: dict[str, Any],
    items: list[Item],
    parsed: dict[str, ParsedPaper],
    max_usd: Decimal,
) -> str:
    ledger = Ledger(
        pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
    )
    gateway = Gateway(settings, ledger)
    suite = S1(pool, gateway, Scope(run_id=run_id, run_cap_usd=max_usd))
    try:
        await suite.prepare(items, parsed)
        client = Client()
        by_id = {item.id: item for item in items}

        async def target(inputs: dict[str, Any]) -> dict[str, Any]:
            return {"rankings": await suite.rankings(by_id[inputs["item_id"]])}

        def grade(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
            references = [set(gold) for gold in reference_outputs["references"]]
            return {
                "results": [
                    {"key": f"{arm}.{name}", "score": value}
                    for arm, ranking in outputs["rankings"].items()
                    for name, value in score(ranking, references).items()
                ]
            }

        results = await aevaluate(
            target,
            data=_examples(client, set(by_id)),
            evaluators=[grade],
            experiment_prefix=run_id,
            metadata=config,
            max_concurrency=1,
            client=client,
        )
        failed = 0
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE eval_runs SET langsmith_experiment = %s WHERE id = %s",
                (results.experiment_name, run_id),
            )
            async for row in results:
                item = by_id[(row["example"].inputs or {})["item_id"]]
                outputs, error = row["run"].outputs or {}, row["run"].error
                failed += error is not None
                for arm in s1.ARMS:
                    ranking = outputs.get("rankings", {}).get(arm, [])
                    metrics = score(ranking, item.references) if error is None else {}
                    await conn.execute(
                        "INSERT INTO eval_results (run_id, item_id, arm, metrics, output, error)"
                        " VALUES (%s, %s, %s, %s, %s, %s)",
                        (run_id, item.id, arm, json.dumps(metrics), json.dumps(ranking), error),
                    )
        return "partial" if failed else "complete"
    finally:
        await gateway.aclose()


def _examples(client: Client, wanted: set[str]) -> list[Example]:
    """The suite's LangSmith dataset, created once per manifest version (QASPER is CC BY 4.0)."""
    name = f"ara-s1-{s1.manifest_hash()}"
    if not client.has_dataset(dataset_name=name):
        items, _ = s1.load()
        dataset = client.create_dataset(name, description="ARA S1 retrieval: QASPER validation")
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"item_id": i.id, "paper": i.paper, "question": i.question},
                    "outputs": {"references": [sorted(gold) for gold in i.references]},
                    "metadata": {"split": i.split},
                }
                for i in items
            ],
        )
    return [
        e
        for e in client.list_examples(dataset_name=name)
        if (e.inputs or {}).get("item_id") in wanted
    ]


async def _start(pool: Pool, run_id: str, config: dict[str, Any]) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO eval_runs (id, suite, config) VALUES (%s, %s, %s)",
            (run_id, config["suite"], json.dumps(config)),
        )


async def _finish(pool: Pool, run_id: str, status: str) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE eval_runs SET status = %s, finished_at = now() WHERE id = %s",
            (status, run_id),
        )
