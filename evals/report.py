"""Report for one evaluation run (DESIGN §11.1, §11.5). Held-out (test) items give the reported
numbers and dev items the ones used for choices; each split has metrics per arm with n and 95%
bootstrap CIs, then paired differences between arms. Efficiency per stage comes from the ledger.
The per-item results are also exported as JSONL, a portable copy outside Postgres (§10)."""

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence

from psycopg.rows import DictRow

from ara.db.pool import Pool
from evals.stats import mean_ci, paired
from evals.suites import s1, s2, s5

EFFICIENCY = """
SELECT stage, model, count(*) AS requests, sum(input_tokens) AS input_tokens,
       sum(cached_tokens) AS cached_tokens, sum(output_tokens) AS output_tokens,
       sum(charge_usd) AS usd,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
FROM llm_calls WHERE run_id = %s GROUP BY stage, model ORDER BY stage
"""
SPLITS = (("test", "Held-out (test): reported numbers"), ("dev", "Dev: used for choices"))
SUITES = {"s1": s1, "s2": s2, "s5": s5}

type Scores = dict[str, dict[str, float]]  # item → metric → value, for one arm


async def report(pool: Pool, run_id: str) -> tuple[str, str]:
    """(markdown report, JSONL of per-item results)."""
    async with pool.connection() as conn:
        run = await (
            await conn.execute("SELECT * FROM eval_runs WHERE id = %s", (run_id,))
        ).fetchone()
        if run is None:
            raise LookupError(f"no evaluation run {run_id}")
        results = await (
            await conn.execute(
                "SELECT item_id, arm, metrics, output, error FROM eval_results WHERE run_id = %s"
                " ORDER BY arm, item_id",
                (run_id,),
            )
        ).fetchall()
        usage = await (await conn.execute(EFFICIENCY, (run_id,))).fetchall()

    suite = SUITES[run["suite"]]
    split_of = suite.splits()
    by_arm: dict[str, Scores] = defaultdict(dict)
    items, errors = set(), set()
    for row in results:
        items.add(row["item_id"])
        if row["error"]:
            errors.add(row["item_id"])
        else:
            by_arm[row["arm"]][row["item_id"]] = row["metrics"]

    lines = [
        f"# {run_id} ({run['suite']}, {run['status']})",
        "",
        f"Items: {len(items)} ({len(errors)} failed). LangSmith experiment:"
        f" {run['langsmith_experiment'] or '—'}. Config: `{run['config']}`",
        "",
        "Mean [95% bootstrap CI]. Small n: read as directional.",
    ]
    for split, heading in SPLITS:
        scoped = {
            arm: {i: m for i, m in scores.items() if split_of[i] == split}
            for arm, scores in by_arm.items()
        }
        lines += ["", f"## {heading}", "", *_arms(scoped), *_abstention(scoped)]
        if suite is s5:
            lines += _detection(scoped, s5.kinds())
        if suite.PAIRS:
            lines += ["", *_pairs(scoped, suite.PAIRS, suite.LABELS)]
    lines += ["", "## Efficiency", "", *_efficiency(usage)]
    export = "".join(
        json.dumps({**row, "split": split_of[row["item_id"]]}, sort_keys=True) + "\n"
        for row in results
    )
    return "\n".join(lines) + "\n", export


def _metrics(by_arm: Mapping[str, Scores]) -> list[str]:
    return sorted({name for scores in by_arm.values() for m in scores.values() for name in m})


def _arms(by_arm: Mapping[str, Scores]) -> list[str]:
    """Mean [CI] per metric over the items that have it; "(n=k)" marks a metric defined on fewer
    items than the arm completed (citation precision is undefined when nothing is cited)."""
    names = _metrics(by_arm)
    lines = ["| arm | n | " + " | ".join(names) + " |", "|---|---:|" + "---|" * len(names)]
    for arm, scores in by_arm.items():
        if not scores:
            continue
        cells = []
        for name in names:
            values = [m[name] for m in scores.values() if name in m]
            if not values:
                cells.append("—")
                continue
            mean, low, high = mean_ci(values)
            note = f" (n={len(values)})" if len(values) < len(scores) else ""
            cells.append(f"{mean:.2f} [{low:.2f}, {high:.2f}]{note}")
        lines.append(f"| {arm} | {len(scores)} | " + " | ".join(cells) + " |")
    return lines


def _abstention(by_arm: Mapping[str, Scores]) -> list[str]:
    """Abstention precision and recall (S2): unanswerable is the positive class."""
    lines = []
    for arm, scores in by_arm.items():
        rows = [m for m in scores.values() if "abstained" in m]
        if not rows:
            continue
        hits = sum(m["abstained"] * m["unanswerable"] for m in rows)
        predicted = sum(m["abstained"] for m in rows)
        actual = sum(m["unanswerable"] for m in rows)
        precision = f"{hits / predicted:.2f}" if predicted else "—"
        recall = f"{hits / actual:.2f}" if actual else "—"
        lines.append(
            f"{arm} abstention: precision {precision} ({int(hits)}/{int(predicted)} abstentions"
            f" correct), recall {recall} ({int(hits)}/{int(actual)} unanswerable caught)"
        )
    return ["", *lines] if lines else []


def _detection(by_arm: Mapping[str, Scores], kinds: Mapping[str, str]) -> list[str]:
    """Verifier quality (S5): precision, recall and F1 on "unsupported", then recall by the kind
    of perturbation, so a model blind to one kind shows."""
    lines = []
    for arm, scores in by_arm.items():
        flagged = [i for i, m in scores.items() if m["flagged"]]
        positives = [i for i, m in scores.items() if m["unsupported"]]
        hits = len(set(flagged) & set(positives))
        p = hits / len(flagged) if flagged else 0.0
        r = hits / len(positives) if positives else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        by_kind = defaultdict(list)
        for i in positives:
            by_kind[kinds[i]].append(scores[i]["flagged"])
        recalls = ", ".join(f"{kind} {int(sum(v))}/{len(v)}" for kind, v in sorted(by_kind.items()))
        lines.append(
            f"{arm} on unsupported: precision {p:.2f} ({hits}/{len(flagged)}), recall {r:.2f}"
            f" ({hits}/{len(positives)}), F1 {f1:.2f}; caught by kind: {recalls}"
        )
    return ["", *lines] if lines else []


def _pairs(
    by_arm: Mapping[str, Scores], pairs: Sequence[tuple[str, str]], labels: Sequence[str]
) -> list[str]:
    """a - b on the items both arms completed: mean Δ [CI], then items better / worse. Columns
    that record the item's label rather than the arm's output are left out."""
    names = [name for name in _metrics(by_arm) if name not in labels]
    lines = [
        "Paired a - b: mean Δ [95% CI], items better/worse",
        "",
        "| a - b | n | " + " | ".join(names) + " |",
        "|---|---:|" + "---|" * len(names),
    ]
    for a, b in pairs:
        shared = sorted(by_arm.get(a, {}).keys() & by_arm.get(b, {}).keys())
        if not shared:
            continue
        cells = []
        for name in names:
            d = paired([by_arm[a][i][name] for i in shared], [by_arm[b][i][name] for i in shared])
            cells.append(f"{d.mean:+.3f} [{d.low:+.3f}, {d.high:+.3f}] {d.better}/{d.worse}")
        lines.append(f"| {a} - {b} | {len(shared)} | " + " | ".join(cells) + " |")
    return lines


def _efficiency(usage: Sequence[DictRow]) -> list[str]:
    lines = [
        "| stage | model | requests | input | cached | output | US$ | p50 ms | p95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for u in usage:
        lines.append(
            f"| {u['stage']} | {u['model']} | {u['requests']} | {u['input_tokens'] or 0} |"
            f" {u['cached_tokens'] or 0} | {u['output_tokens'] or 0} | {u['usd']:.6f} |"
            f" {u['p50_ms'] or 0:.0f} | {u['p95_ms'] or 0:.0f} |"
        )
    return lines
