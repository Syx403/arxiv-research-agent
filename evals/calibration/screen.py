"""Screen calibration (D42): the product's screen prompt and a candidate judge the same 40 (request,
paper) pairs that Ewan labelled; each arm's grades are compared with his. The pairs and labels stay
in git-ignored `data/calibration/` (abstracts from arXiv, labels from the labelling page).

    uv run python -m evals.calibration.screen            # plan only
    uv run python -m evals.calibration.screen --execute  # billable: two arms, one call per batch
"""

import asyncio
import json
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from ara.db.pool import make_pool
from ara.graph.discover import BATCH, SCREEN, Screening, in_batch, request_block
from ara.graph.state import Constraint, PaperCard, Priority, ResearchRequest
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger, Scope
from ara.llm.prompt import Instructions, Prompt, data
from ara.llm.stages import STAGES
from ara.settings import get_settings

DATA = Path("data/calibration")
CAP_USD = Decimal("0.04")
ARMS = {
    "product": SCREEN,
    "candidate": Instructions.load("screen_v2", Path(__file__).parent),
}


def load() -> tuple[list[dict[str, Any]], dict[str, int]]:
    items = json.loads((DATA / "screen_items.json").read_text())["items"]
    labels = json.loads((DATA / "screen_labels.json").read_text())
    # Ewan settled on four grades (D42): his 4 and 3 are both "about what was asked" (3)
    grades = {k: min(int(v["grade"]), 3) for k, v in labels.items()}
    return items, grades


def request(raw: dict[str, Any]) -> ResearchRequest:
    return ResearchRequest(
        intent="discover",
        language="English",
        clarification=None,
        need=raw["need"],
        question=None,
        paper_ids=[],
        listed=[],
        count=None,
        constraints=[Constraint(**c) for c in raw["constraints"]],
        priorities=[Priority(**p) for p in raw["priorities"]],
        titles=raw["titles"],
        names=[],
        history=None,
        prefer_recent=raw.get("prefer_recent", False),
        published_after=None,
        published_before=None,
    )


def batches(items: list[dict[str, Any]]) -> list[tuple[ResearchRequest, list[PaperCard]]]:
    """Per request, its papers in batches of BATCH, as discovery sends them."""
    by_need: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_need[item["request"]["need"]].append(item)
    out = []
    for group in by_need.values():
        cards = [PaperCard(version=1, **g["paper"]) for g in group]
        for i in range(0, len(cards), BATCH):
            out.append((request(group[0]["request"]), cards[i : i + BATCH]))
    return out


async def judge(
    gateway: Gateway, scope: Scope, instructions: Instructions, work: Any
) -> dict[str, int]:
    """Each arm's grade per paper, by arXiv id (every paper occurs in one request only)."""
    grades: dict[str, int] = {}
    for req, cards in work:
        papers = [
            {"id": p.arxiv_id, "title": p.title, "published": p.published, "abstract": p.abstract}
            for p in cards
        ]
        prompt = Prompt(instructions, shared=(request_block(req),), item=(data("Papers", papers),))
        result = await gateway.structured(STAGES["screen"], prompt, Screening, scope=scope)
        grades |= {j.id: j.relevance for j in in_batch(result.papers, cards)}
    return grades


def agreement(model: dict[str, int], human: dict[str, int]) -> dict[str, float]:
    """Exact agreement, quadratic-weighted κ, agreement on what is read (≥ 2), Kendall's τ-b, and
    how often the model grades above or below Ewan."""
    keys = sorted(k for k in human if k in model)
    a, b = [human[k] for k in keys], [model[k] for k in keys]
    n, k = len(keys), 4
    observed = [[0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[x][y] += 1
    rows = [sum(r) for r in observed]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    weight = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    seen = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k))
    chance = sum(weight[i][j] * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da * db > 0:
                concordant += 1
            else:
                discordant += 1
    tau = (concordant - discordant) / (
        ((concordant + discordant + ties_a) * (concordant + discordant + ties_b)) ** 0.5 or 1
    )
    return {
        "n": n,
        "exact": sum(x == y for x, y in zip(a, b, strict=True)) / n,
        "qwk": 1 - seen / chance if chance else 0.0,
        "read_agree": sum((x >= 2) == (y >= 2) for x, y in zip(a, b, strict=True)) / n,
        "tau_b": tau,
        "higher": sum(y > x for x, y in zip(a, b, strict=True)),
        "lower": sum(y < x for x, y in zip(a, b, strict=True)),
    }


def paired(arms: dict[str, dict[str, int]], human: dict[str, int], metric: str) -> str:
    """Bootstrap over pairs of the candidate minus product difference in one metric."""
    keys = [k for k in human if all(k in g for g in arms.values())]
    random.seed(20261011)
    diffs = []
    for _ in range(2000):
        sample = random.choices(keys, k=len(keys))
        one = {f"{k}#{i}": human[k] for i, k in enumerate(sample)}

        def on(arm: str, sample: list[str] = sample) -> dict[str, int]:
            return {f"{k}#{i}": arms[arm][k] for i, k in enumerate(sample)}

        diffs.append(
            agreement(on("candidate"), one)[metric] - agreement(on("product"), one)[metric]
        )
    diffs.sort()
    return f"{metric}: Δ {sum(diffs) / len(diffs):+.3f} [{diffs[50]:+.3f}, {diffs[1949]:+.3f}]"


async def main(execute: bool, saved: str | None) -> None:
    items, human = load()
    work = batches(items)
    print(f"{len(items)} pairs, {len(work)} batches per arm, {2 * len(work)} calls, cap ${CAP_USD}")
    if saved:
        report(json.loads(Path(saved).read_text()), items, human)
        return
    if not execute:
        return
    settings = get_settings()
    run_id = f"calib-screen-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    async with make_pool(settings.database_url) as pool:
        ledger = Ledger(pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=CAP_USD)
        gateway = Gateway(settings, ledger)
        try:
            arms = {}
            for name, instructions in ARMS.items():
                scope = Scope(run_id=run_id, turn_id=f"{run_id}:{name}", run_cap_usd=CAP_USD)
                arms[name] = await judge(gateway, scope, instructions, work)
        finally:
            await gateway.aclose()
    (DATA / f"{run_id}.json").write_text(json.dumps(arms, indent=1))
    print("run", run_id, "| prompt versions:", {n: i.version for n, i in ARMS.items()})
    report(arms, items, human)


def report(
    by_paper: dict[str, dict[str, int]], items: list[dict[str, Any]], human: dict[str, int]
) -> None:
    ids = {i["paper"]["arxiv_id"]: i["id"] for i in items}
    assert len(ids) == len(items), "a paper occurs in two requests"
    arms = {name: {ids[p]: g for p, g in grades.items()} for name, grades in by_paper.items()}
    for name, grades in arms.items():
        print(name, {k: round(v, 3) for k, v in agreement(grades, human).items()})
    for metric in ("qwk", "read_agree", "tau_b", "exact"):
        print(paired(arms, human, metric))
    for k in sorted(human):
        p, c = arms["product"].get(k), arms["candidate"].get(k)
        if p != human[k] or c != human[k]:
            print(f"  {k} ewan {human[k]} product {p} candidate {c}")


if __name__ == "__main__":
    saved = sys.argv[sys.argv.index("--from") + 1] if "--from" in sys.argv else None
    asyncio.run(main("--execute" in sys.argv, saved))
