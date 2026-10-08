"""Evaluation runner (DESIGN §11.5): dry run by default; with --execute it runs a suite through
LangSmith `aevaluate`, under a run cap in the ledger, and stores every result in Postgres."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any

from langsmith import Client
from langsmith.evaluation import aevaluate
from langsmith.schemas import Example

from ara.arxiv.client import ArxivClient
from ara.db.migrate import migrate
from ara.db.pool import Connection, Pool, make_pool
from ara.graph import app
from ara.graph.discover import build as build_discover
from ara.graph.qa import answer_question
from ara.graph.state import Context, PaperCard, ResearchRequest, Verdict
from ara.llm.gateway import Gateway, InvalidOutput
from ara.llm.ledger import Ledger, Scope
from ara.llm.pricing import EMBEDDING, MILLION
from ara.memory import store
from ara.rag.chunking import chunk
from ara.rag.embed import cache_key
from ara.rag.ingest import PIPELINE_VERSION
from ara.rag.sources import ParsedPaper
from ara.settings import Settings, configure_tracing, get_settings
from ara.tokens import count_tokens
from evals.graders import discovery
from evals.graders.retrieval import score
from evals.suites import s1, s2, s3, s4, s5, s5_data, s6
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


async def plan(pool: Pool, questions: list[str], papers: dict[str, ParsedPaper]) -> Plan:
    """What a run would send, from the local database alone: no API calls."""
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT paper_id FROM documents WHERE pipeline_version = %s", (PIPELINE_VERSION,)
        )
        ingested = {row["paper_id"] for row in await cursor.fetchall()}
        new = [paper for paper in papers.values() if paper.id not in ingested]
        texts = {c.search_text for p in new for c in chunk(p.paragraphs)}
        texts |= set(questions)
        cursor = await conn.execute(
            "SELECT key FROM embedding_cache WHERE key = ANY(%s)",
            ([cache_key(text) for text in texts],),
        )
        cached = {bytes(row["key"]) for row in await cursor.fetchall()}
    missing = [text for text in texts if cache_key(text) not in cached]
    questions_missing = any(cache_key(q) not in cached for q in questions)
    return Plan(
        items=len(questions),
        papers_to_ingest=len(new),
        embed_requests=len(new) + int(questions_missing),
        embed_tokens=sum(count_tokens(text) for text in missing),
        rerank_calls=len(questions),
    )


async def run_s1(*, papers: int | None, execute: bool, max_usd: Decimal) -> str | None:
    """Print the plan; with `execute`, run S1 and return the run id."""
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    items, parsed = s1.load(papers)
    async with make_pool(settings.database_url) as pool:
        estimate = await plan(pool, [item.question for item in items], parsed)
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
            """Rank, then store this item's results at once, so an interrupted run keeps them."""
            item = by_id[inputs["item_id"]]
            rankings = await suite.rankings(item)
            async with pool.connection() as conn:
                for arm, ranking in rankings.items():
                    await _store(
                        conn, run_id, item.id, arm, score(ranking, item.references), ranking
                    )
            return {"rankings": rankings}

        def grade(outputs: dict[str, Any], reference_outputs: dict[str, Any]) -> dict[str, Any]:
            references = [set(gold) for gold in reference_outputs["references"]]
            return {
                "results": [
                    {"key": f"{arm}.{name}", "score": value}
                    for arm, ranking in outputs["rankings"].items()
                    for name, value in score(ranking, references).items()
                ]
            }

        dataset = _dataset(client, "s1", [_s1_example(i) for i in s1.load()[0]])
        results = await aevaluate(
            target,
            data=[e for e in client.list_examples(dataset_name=dataset) if _item(e) in by_id],
            evaluators=[grade],
            experiment_prefix=run_id,
            metadata={**config, "dataset": dataset},
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
                if (error := row["run"].error) is not None:
                    failed += 1
                    for arm in s1.ARMS:
                        await _store(conn, run_id, _item(row["example"]), arm, {}, [], error)
        return "partial" if failed else "complete"
    finally:
        await gateway.aclose()


def _dataset(client: Client, suite: str, examples: list[dict[str, Any]]) -> str:
    """The suite's LangSmith dataset (QASPER is CC BY 4.0). Its name hashes the examples
    themselves, gold references included, so a change to how gold is derived makes a new dataset
    instead of grading against stale references."""
    digest = sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()[:12]
    name = f"ara-{suite}-{digest}"
    if not client.has_dataset(dataset_name=name):
        dataset = client.create_dataset(name, description=f"ARA {suite.upper()}: QASPER validation")
        client.create_examples(dataset_id=dataset.id, examples=examples)
    return name


def _s1_example(item: Item) -> dict[str, Any]:
    return {
        "inputs": {"item_id": item.id, "paper": item.paper, "question": item.question},
        "outputs": {"references": [sorted(gold) for gold in item.references]},
        "metadata": {"split": item.split},
    }


def _item(example: Example) -> str:
    return str((example.inputs or {})["item_id"])


async def _store(
    conn: Connection,
    run_id: str,
    item_id: str,
    arm: str,
    metrics: dict[str, float],
    output: Any,
    error: str | None = None,
) -> None:
    await conn.execute(
        "INSERT INTO eval_results (run_id, item_id, arm, metrics, output, error)"
        " VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
        (run_id, item_id, arm, json.dumps(metrics), json.dumps(output), error),
    )


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


async def run_s2(*, limit: int | None, execute: bool, max_usd: Decimal) -> str | None:
    """Print the plan; with `execute`, answer the first `limit` S2 items and return the run id."""
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    items, papers = s2.load(limit)
    async with make_pool(settings.database_url) as pool:
        ingestion = await plan(pool, [item.question for item in items], papers)
        usd = Decimal(str(s2.UNIT_COST_USD)) * len(items) + ingestion.usd
        print(
            f"S2 plan: {len(items)} items; {ingestion.papers_to_ingest} papers to ingest"
            f" (${ingestion.usd:.5f}); per item 1 select_evidence and 1 rerank per paper (more"
            f" on a requery), 1 synthesize, ~4 verify, 0-1 prewarm, 0-1 repair;"
            f" estimated ${usd:.4f}"
        )
        if not execute:
            print("Dry run: nothing was sent. Add --execute to run.")
            return None
        if usd > max_usd:
            raise SystemExit(f"estimated ${usd:.4f} exceeds --max-usd {max_usd}")
        run_id = f"s2-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        config = {
            "suite": "s2",
            "manifest": s2.manifest_hash(),
            "pipeline_version": PIPELINE_VERSION,
            "items": [item.id for item in items],
            "max_usd": str(max_usd),
        }
        await _start(pool, run_id, config)
        status = "failed"
        try:
            status = await _answer_items(pool, settings, run_id, config, items, papers, max_usd)
        finally:
            await _finish(pool, run_id, status)
        return run_id


async def _answer_items(
    pool: Pool,
    settings: Settings,
    run_id: str,
    config: dict[str, Any],
    items: list[s2.Item],
    papers: dict[str, ParsedPaper],
    max_usd: Decimal,
) -> str:
    ledger = Ledger(
        pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
    )
    gateway = Gateway(settings, ledger)

    async def fetch(reference: str) -> ParsedPaper:
        return papers[reference]

    arxiv = ArxivClient()  # QASPER papers come from the dataset: it sends nothing
    context = Context(pool, gateway, Scope(run_id=run_id, run_cap_usd=max_usd), fetch, arxiv)
    try:
        client = Client()
        by_id = {item.id: item for item in items}

        async def target(inputs: dict[str, Any]) -> dict[str, Any]:
            item = by_id[inputs["item_id"]]
            answer = await answer_question(item.question, [f"qasper:{item.paper}"], context)
            result = s2.output(answer)
            async with pool.connection() as conn:
                await _store(conn, run_id, item.id, "product", s2.score(result, item), result)
            return result

        def grade(outputs: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
            metrics = s2.score(outputs, by_id[inputs["item_id"]])
            return {"results": [{"key": k, "score": v} for k, v in metrics.items()]}

        examples = [
            {
                "inputs": {"item_id": i.id, "paper": i.paper, "question": i.question},
                "outputs": {"golds": [[g.kind, g.text, sorted(g.evidence)] for g in i.golds]},
                "metadata": {"split": i.split},
            }
            for i in s2.load()[0]
        ]
        dataset = _dataset(client, "s2", examples)
        results = await aevaluate(
            target,
            data=[e for e in client.list_examples(dataset_name=dataset) if _item(e) in by_id],
            evaluators=[grade],
            experiment_prefix=run_id,
            metadata={**config, "dataset": dataset},
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
                if (error := row["run"].error) is not None:
                    failed += 1
                    await _store(conn, run_id, _item(row["example"]), "product", {}, {}, error)
        return "partial" if failed else "complete"
    finally:
        await gateway.aclose()
        await arxiv.aclose()


async def perturb_s5(*, execute: bool, max_usd: Decimal) -> None:
    """Show the S5 sources and kinds; with `execute`, generate the claims for Ewan's review."""
    picks = s5_data.sources()
    kinds = {kind: sum(p["kind"] == kind for p in picks) for kind in s5_data.QUOTA}
    requests = -(-len(picks) // s5_data.BATCH)
    print(f"S5 data: {len(picks)} sentences, kinds {kinds}; {requests} DeepSeek requests")
    if not execute:
        print("Dry run: nothing was sent. Add --execute to generate.")
        return
    if s5_data.OUTPUT.exists():
        raise SystemExit(f"{s5_data.OUTPUT} exists and holds Ewan's review; delete it to redo")
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    async with make_pool(settings.database_url) as pool:
        ledger = Ledger(
            pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
        )
        gateway = Gateway(settings, ledger)
        run_id = f"s5-data-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        try:
            claims = await s5_data.generate(gateway, Scope(run_id=run_id, run_cap_usd=max_usd))
        finally:
            await gateway.aclose()
    s5_data.write(claims)
    print(f"wrote {len(claims)} claims to {s5_data.OUTPUT} (ledger run {run_id}); review them")


async def run_s5(*, limit: int | None, execute: bool, max_usd: Decimal) -> str | None:
    """Print the plan; with `execute`, verify the first `limit` claim pairs with both models."""
    items = s5.load(limit)
    usd = sum(Decimal(str(c)) for c in s5.UNIT_COST_USD.values()) * len(items)
    print(
        f"S5 plan: {len(items)} claims x {len(s5.ARMS)} arms = {len(items) * len(s5.ARMS)}"
        f" verify requests; estimated ${usd:.4f}"
    )
    if not execute:
        print("Dry run: nothing was sent. Add --execute to run.")
        return None
    if usd > max_usd:
        raise SystemExit(f"estimated ${usd:.4f} exceeds --max-usd {max_usd}")
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    async with make_pool(settings.database_url) as pool:
        run_id = f"s5-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        config = {
            "suite": "s5",
            "data": sha256(s5.DATA.read_bytes()).hexdigest()[:12],
            "arms": {arm: f"{st.model}/{st.effort}" for arm, st in s5.ARMS.items()},
            "items": len(items),
            "max_usd": str(max_usd),
        }
        await _start(pool, run_id, config)
        status = "failed"
        try:
            status = await _verify_items(pool, settings, run_id, config, items, max_usd)
        finally:
            await _finish(pool, run_id, status)
        return run_id


async def _verify_items(
    pool: Pool,
    settings: Settings,
    run_id: str,
    config: dict[str, Any],
    items: list[s5.Item],
    max_usd: Decimal,
) -> str:
    ledger = Ledger(
        pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
    )
    gateway = Gateway(settings, ledger)
    scope = Scope(run_id=run_id, run_cap_usd=max_usd)
    invalid = 0
    try:
        client = Client()
        by_id = {item.id: item for item in items}

        async def target(inputs: dict[str, Any]) -> dict[str, Any]:
            """Both arms on one claim. A reply DeepSeek's JSON mode cannot validate is stored as
            that arm's error and counted, never repaired (D10)."""
            nonlocal invalid
            item = by_id[inputs["item_id"]]
            outputs: dict[str, Any] = {}
            async with pool.connection() as conn:
                for arm in s5.ARMS:
                    try:
                        verdict = await s5.check(gateway, arm, item, scope)
                    except InvalidOutput as error:
                        invalid += 1
                        await _store(conn, run_id, item.id, arm, {}, {}, f"invalid: {error}")
                        continue
                    outputs[arm] = verdict.model_dump()
                    await _store(conn, run_id, item.id, arm, s5.score(verdict, item), outputs[arm])
            return outputs

        def grade(outputs: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
            item = by_id[inputs["item_id"]]
            return {
                "results": [
                    {"key": f"{arm}.correct", "score": s5.score(Verdict(**v), item)["correct"]}
                    for arm, v in outputs.items()
                ]
            }

        examples = [
            {
                "inputs": {"item_id": i.id, "claim": i.claim, "evidence": i.evidence.text},
                "outputs": {"label": i.label, "kind": i.kind},
                "metadata": {"split": i.split},
            }
            for i in s5.load()
        ]
        dataset = _dataset(client, "s5", examples)
        results = await aevaluate(
            target,
            data=[e for e in client.list_examples(dataset_name=dataset) if _item(e) in by_id],
            evaluators=[grade],
            experiment_prefix=run_id,
            metadata={**config, "dataset": dataset},
            max_concurrency=4,
            client=client,
        )
        failed = invalid
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE eval_runs SET langsmith_experiment = %s WHERE id = %s",
                (results.experiment_name, run_id),
            )
            async for row in results:
                if (error := row["run"].error) is not None:
                    failed += 1
                    for arm in s5.ARMS:
                        await _store(conn, run_id, _item(row["example"]), arm, {}, {}, error)
        return "partial" if failed else "complete"
    finally:
        await gateway.aclose()


async def run_s3(*, limit: int | None, execute: bool, max_usd: Decimal) -> str | None:
    """Print the plan; with `execute`, run discovery on the first `limit` queries of each split.
    PaSa is gated: no LangSmith dataset or experiment is made (D22), the results stay in Postgres
    and the report names queries by id only. Traces of the calls still reach the private project."""
    items = s3.load(limit)
    usd = Decimal(str(s3.UNIT_COST_USD)) * len(items)
    print(
        f"S3 plan: {len(items)} queries; each ≤ 8 researcher steps, 1 embedding request, ≤ 3"
        f" screen batches, ≤ 8 arXiv searches; estimated ${usd:.4f}"
    )
    if not execute:
        print("Dry run: nothing was sent. Add --execute to run.")
        return None
    if usd > max_usd:
        raise SystemExit(f"estimated ${usd:.4f} exceeds --max-usd {max_usd}")
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    discover = build_discover()
    async with make_pool(settings.database_url) as pool, ArxivClient() as arxiv:
        run_id = f"s3-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        config = {"suite": "s3", "manifest": s3.manifest_hash(), "items": len(items)}
        await _start(pool, run_id, {**config, "max_usd": str(max_usd)})
        ledger = Ledger(
            pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
        )
        gateway = Gateway(settings, ledger)

        async def fetch(reference: str) -> ParsedPaper:
            raise RuntimeError(f"discovery does not read papers: {reference}")

        failed, status = 0, "failed"
        try:
            for _, query in items:
                scope = Scope(run_id=run_id, turn_id=query.id, run_cap_usd=max_usd)
                # the query was asked on its cutoff date: that is "today" for the agent (D24)
                context = Context(pool, gateway, scope, fetch, arxiv, today=query.cutoff)
                async with pool.connection() as conn:
                    try:
                        found = await discover.ainvoke(
                            {"request": s3.request(query)},
                            {"recursion_limit": 60, "metadata": {"run_id": run_id}},
                            context=context,
                        )
                    except Exception as error:  # one failed query must not end the round
                        failed += 1
                        await _store(conn, run_id, query.id, "product", {}, {}, repr(error))
                        continue
                    listed = [p.arxiv_id for p in found["papers"]]
                    metrics = discovery.score(
                        found["candidates"], found["shortlisted"], listed, query.gold
                    )
                    # papers the screen silently skipped would otherwise look like rejections
                    metrics["unjudged"] = float(len(found["unjudged"]))
                    output = {
                        "listed": listed,
                        "candidates": len(found["candidates"]),
                        "unjudged": found["unjudged"],
                    }
                    await _store(conn, run_id, query.id, "product", metrics, output)
            status = "partial" if failed else "complete"
        finally:
            await gateway.aclose()
            await _finish(pool, run_id, status)
        return run_id


async def run_s4(*, limit: int | None, execute: bool, max_usd: Decimal) -> str | None:
    """Print the plan; with `execute`, run understand on the first `limit` reviewed items."""
    items = s4.load(limit, reviewed=execute)
    usd = Decimal(str(s4.UNIT_COST_USD)) * len(items)
    print(f"S4 plan: {len(items)} understand requests; estimated ${usd:.4f}")
    if not execute:
        print("Dry run: nothing was sent. Add --execute to run.")
        return None
    if usd > max_usd:
        raise SystemExit(f"estimated ${usd:.4f} exceeds --max-usd {max_usd}")
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    async with make_pool(settings.database_url) as pool:
        run_id = f"s4-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        config = {"suite": "s4", "data": s4.data_hash(), "items": len(items)}
        await _start(pool, run_id, {**config, "max_usd": str(max_usd)})
        status = "failed"
        try:
            status = await _understand_items(pool, settings, run_id, config, items, max_usd)
        finally:
            await _finish(pool, run_id, status)
        return run_id


async def _understand_items(
    pool: Pool,
    settings: Settings,
    run_id: str,
    config: dict[str, Any],
    items: list[s4.Item],
    max_usd: Decimal,
) -> str:
    ledger = Ledger(
        pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
    )
    gateway = Gateway(settings, ledger)
    scope = Scope(run_id=run_id, run_cap_usd=max_usd)
    try:
        client = Client()
        by_id = {item.id: item for item in items}

        async def target(inputs: dict[str, Any]) -> dict[str, Any]:
            item = by_id[inputs["item_id"]]
            request = await s4.understand(gateway, item, scope)
            output = request.model_dump()
            async with pool.connection() as conn:
                await _store(conn, run_id, item.id, "product", s4.score(request, item), output)
            return output

        def grade(outputs: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
            item = by_id[inputs["item_id"]]
            metrics = s4.score(ResearchRequest(**outputs), item)
            return {"results": [{"key": k, "score": v} for k, v in metrics.items()]}

        examples = [
            {
                "inputs": {
                    "item_id": i.id,
                    "asked_at": i.asked_at.isoformat(),
                    "messages": [m.text for m in i.messages],
                },
                "outputs": {"expected": i.expected},
                "metadata": {"split": i.split},
            }
            for i in s4.load()
        ]
        dataset = _dataset(client, "s4", examples)
        results = await aevaluate(
            target,
            data=[e for e in client.list_examples(dataset_name=dataset) if _item(e) in by_id],
            evaluators=[grade],
            experiment_prefix=run_id,
            metadata={**config, "dataset": dataset},
            max_concurrency=4,
            client=client,
        )
        failed = 0
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE eval_runs SET langsmith_experiment = %s WHERE id = %s",
                (results.experiment_name, run_id),
            )
            async for row in results:
                if (error := row["run"].error) is not None:
                    failed += 1
                    await _store(conn, run_id, _item(row["example"]), "product", {}, {}, error)
        return "partial" if failed else "complete"
    finally:
        await gateway.aclose()


async def run_s6(
    *, limit: int | None, execute: bool, max_usd: Decimal, only: str | None = None
) -> str | None:
    """Print the plan; with `execute`, run the first `limit` scenarios (or the one named `only`)
    through the conversation graph, each as its own user, each session as its own thread (D27).
    Results stay in Postgres."""
    scenarios = s6.load(limit, only)
    usd = Decimal(str(s6.UNIT_COST_USD)) * len(scenarios)
    turns = sum(len(t) for s in scenarios for t in s.sessions)
    print(f"S6 plan: {len(scenarios)} scenarios, {turns} turns; estimated ${usd:.4f}")
    if not execute:
        print("Dry run: nothing was sent. Add --execute to run.")
        return None
    if usd > max_usd:
        raise SystemExit(f"estimated ${usd:.4f} exceeds --max-usd {max_usd}")
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    async with make_pool(settings.database_url) as pool, ArxivClient() as arxiv:
        run_id = f"s6-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        config = {"suite": "s6", "data": s6.data_hash(), "items": len(scenarios)}
        await _start(pool, run_id, {**config, "max_usd": str(max_usd)})
        ledger = Ledger(
            pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
        )
        gateway = Gateway(settings, ledger)
        memory = await store.open_store(pool, gateway)
        graph = app.build(await app.checkpointer(pool), memory)

        async def fetch(reference: str) -> ParsedPaper:
            return await arxiv.paper(reference.removeprefix("arxiv:"))

        failed, status = 0, "failed"
        try:
            for scenario in scenarios:
                user = f"{run_id}:{scenario.id}"
                failures: list[str] = []
                try:
                    for n, session in enumerate(scenario.sessions):
                        thread, shown = f"{user}:{n}", list[PaperCard]()
                        for k, turn in enumerate(session):
                            scope = Scope(
                                run_id=run_id, turn_id=f"{user}:{n}:{k}", run_cap_usd=max_usd
                            )
                            context = Context(pool, gateway, scope, fetch, arxiv, user_id=user)
                            done = await app.send(graph, thread, context, message=turn["message"])
                            profile = await store.profile(memory, user)
                            failures += [
                                f"session {n} turn {k}: {f}"
                                for f in s6.check(turn["expect"], done.state, profile, shown)
                            ]
                            if done.waiting is not None:
                                failures.append(f"session {n} turn {k}: waiting {done.waiting}")
                            shown = done.state.get("shown", [])
                except Exception as error:  # one failed scenario must not end the round
                    failed += 1
                    async with pool.connection() as conn:
                        await _store(conn, run_id, scenario.id, "product", {}, {}, repr(error))
                    continue
                # each turn also checks that it did not stop to wait for the user
                checks = sum(len(t["expect"]) + 1 for s in scenario.sessions for t in s)
                metrics = {
                    "passed": float(not failures),
                    "checks_passed": (checks - len(failures)) / checks,
                }
                async with pool.connection() as conn:
                    await _store(
                        conn, run_id, scenario.id, "product", metrics, {"failures": failures}
                    )
            status = "partial" if failed else "complete"
        finally:
            await gateway.aclose()
            await _finish(pool, run_id, status)
        return run_id
