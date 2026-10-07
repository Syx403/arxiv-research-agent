"""Markdown report for one evaluation run (DESIGN §11.5): metrics per arm with n and 95% bootstrap
CIs, then efficiency per stage from the ledger (requests, tokens, cache hits, cost, latency)."""

from collections import defaultdict

from ara.db.pool import Pool
from evals.stats import mean_ci

EFFICIENCY = """
SELECT stage, model, count(*) AS requests, sum(input_tokens) AS input_tokens,
       sum(cached_tokens) AS cached_tokens, sum(output_tokens) AS output_tokens,
       sum(charge_usd) AS usd,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
FROM llm_calls WHERE run_id = %s GROUP BY stage, model ORDER BY stage
"""


async def report(pool: Pool, run_id: str) -> str:
    async with pool.connection() as conn:
        run = await (
            await conn.execute("SELECT * FROM eval_runs WHERE id = %s", (run_id,))
        ).fetchone()
        if run is None:
            raise LookupError(f"no evaluation run {run_id}")
        results = await (
            await conn.execute(
                "SELECT item_id, arm, metrics, error FROM eval_results WHERE run_id = %s"
                " ORDER BY arm, item_id",
                (run_id,),
            )
        ).fetchall()
        usage = await (await conn.execute(EFFICIENCY, (run_id,))).fetchall()

    by_arm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    items, errors = set(), set()
    for row in results:
        items.add(row["item_id"])
        if row["error"]:
            errors.add(row["item_id"])
        for name, value in row["metrics"].items():
            by_arm[row["arm"]][name].append(value)
    names = sorted({name for metrics in by_arm.values() for name in metrics})

    lines = [
        f"# {run_id} ({run['suite']}, {run['status']})",
        "",
        f"Items: {len(items)} ({len(errors)} failed). LangSmith experiment:"
        f" {run['langsmith_experiment'] or '—'}. Config: `{run['config']}`",
        "",
        "Mean [95% bootstrap CI] per arm. Small n: read as directional.",
        "",
        "| arm | n | " + " | ".join(names) + " |",
        "|---|---:|" + "---|" * len(names),
    ]
    for arm, metrics in by_arm.items():
        cells = []
        for name in names:
            mean, low, high = mean_ci(metrics[name])
            cells.append(f"{mean:.2f} [{low:.2f}, {high:.2f}]")
        n = len(next(iter(metrics.values())))
        lines.append(f"| {arm} | {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| stage | model | requests | input | cached | output | US$ | p50 ms | p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for u in usage:
        lines.append(
            f"| {u['stage']} | {u['model']} | {u['requests']} | {u['input_tokens'] or 0} |"
            f" {u['cached_tokens'] or 0} | {u['output_tokens'] or 0} | {u['usd']:.6f} |"
            f" {u['p50_ms'] or 0:.0f} | {u['p95_ms'] or 0:.0f} |"
        )
    return "\n".join(lines) + "\n"
