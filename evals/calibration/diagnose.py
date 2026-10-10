"""Screen diagnosis (D44): is disagreement with Ewan's labels the model's capability, the rubric,
or noise? The product prompt judges the 40 labelled pairs with several models, each repeated, and
the grades are compared with Ewan's, with themselves (repeats) and with each other.

    uv run python -m evals.calibration.diagnose            # plan only
    uv run python -m evals.calibration.diagnose --execute  # billable
    uv run python -m evals.calibration.diagnose --from data/calibration/<run>.json
"""

import asyncio
import json
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Any

from ara.db.pool import make_pool
from ara.graph.discover import SCREEN, Screening, in_batch, request_block
from ara.llm.gateway import Gateway, InvalidOutput
from ara.llm.ledger import Ledger, Scope
from ara.llm.prompt import Prompt, data
from ara.llm.stages import FANOUT, FLASH, HAIKU, LUNA, STAGES, Stage
from ara.settings import configure_tracing, get_settings
from evals.calibration import screen as calibration

REPEATS = 3
CAP_USD = Decimal("0.10")  # the approval for the whole diagnosis; --cap gives what is left
DEFAULT_ARMS = ("luna-medium", "luna-high", "haiku-medium")
OUTPUT = STAGES["screen"].max_output_tokens
ARMS = {  # the product's screen stage first; then one change of model or effort each
    "luna-medium": STAGES["screen"],
    "luna-high": Stage("screen", LUNA, "high", OUTPUT, FANOUT),
    # DeepSeek Flash at high effort spent up to all 6,000 output tokens thinking (one empty reply,
    # about US$0.0025 a call) in the first attempt; it runs only when named with --arms
    "flash-high": Stage("screen", FLASH, "high", OUTPUT, FANOUT),
    "haiku-medium": Stage("screen", HAIKU, "medium", OUTPUT, FANOUT),
}


async def run(work: Any, arms: list[str], cap: Decimal) -> dict[str, list[dict[str, int]]]:
    """Each arm's grades by arXiv id, once per repeat; batches go one at a time, so a repeat
    reads the cache the previous one wrote. Results are saved after every batch; a batch whose
    reply is unusable is counted and left ungraded."""
    settings = get_settings()
    configure_tracing(settings)
    run_id = f"diag-screen-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    path = calibration.DATA / f"{run_id}.json"
    grades: dict[str, list[dict[str, int]]] = {name: [] for name in arms}
    failed: Counter[str] = Counter()
    async with make_pool(settings.database_url) as pool:
        ledger = Ledger(pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=cap)
        gateway = Gateway(settings, ledger)
        try:
            for repeat in range(REPEATS):
                for name in arms:
                    scope = Scope(
                        run_id=run_id, turn_id=f"{run_id}:{name}:{repeat}", run_cap_usd=cap
                    )
                    grades[name].append({})
                    for batch in work:
                        try:
                            got = await judge(gateway, scope, ARMS[name], [batch])
                        except InvalidOutput:
                            failed[name] += 1
                            got = {}
                        grades[name][-1] |= got
                        saved = {"run_id": run_id, "grades": grades, "failed": failed}
                        path.write_text(json.dumps(saved, indent=1))
        finally:
            await gateway.aclose()
    print("run", run_id, "saved to", path, "| failed batches:", dict(failed))
    return grades


async def judge(gateway: Gateway, scope: Scope, stage: Stage, work: Any) -> dict[str, int]:
    grades: dict[str, int] = {}
    for req, cards in work:
        papers = [
            {"id": p.arxiv_id, "title": p.title, "published": p.published, "abstract": p.abstract}
            for p in cards
        ]
        prompt = Prompt(SCREEN, shared=(request_block(req),), item=(data("Papers", papers),))
        result = await gateway.structured(stage, prompt, Screening, scope=scope)
        grades |= {j.id: j.relevance for j in in_batch(result.papers, cards)}
    return grades


def spread(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f}+-{statistics.pstdev(values):.3f}"


def report(
    grades: dict[str, list[dict[str, int]]], items: list[dict[str, Any]], human: dict[str, int]
) -> None:
    ids = {i["paper"]["arxiv_id"]: i["id"] for i in items}
    by_item = {
        name: [{ids[p]: g for p, g in rep.items()} for rep in reps] for name, reps in grades.items()
    }
    keys = sorted(human)

    def read(g: int | None) -> bool:
        return g is not None and g >= 2

    print("\nper arm, against Ewan (mean ± sd over repeats), and repeat stability:")
    majority: dict[str, dict[str, int]] = {}
    for name, reps in by_item.items():
        scores = [calibration.agreement(rep, human) for rep in reps]
        line = {m: spread([s[m] for s in scores]) for m in ("exact", "qwk", "read_agree", "tau_b")}
        flips = sum(len({read(rep.get(k)) for rep in reps}) > 1 for k in keys)
        same = sum(len({rep.get(k) for rep in reps}) == 1 for k in keys)
        majority[name] = {
            k: Counter(rep[k] for rep in reps if k in rep).most_common(1)[0][0] for k in keys
        }
        voted = calibration.agreement(majority[name], human)
        print(
            f"  {name:13} {line} | read decision changes across repeats: {flips}/40,"
            f" grade identical in all repeats: {same}/40 | majority vote read_agree"
            f" {voted['read_agree']:.3f}, exact {voted['exact']:.3f}"
        )
    print("\nread agreement between arms (majority votes):")
    for a, b in combinations(majority, 2):
        agree = sum(read(majority[a][k]) == read(majority[b][k]) for k in keys) / len(keys)
        print(f"  {a} vs {b}: {agree:.3f}")
    shared = [k for k in keys if len({read(majority[n][k]) for n in majority}) == 1]
    wrong = [k for k in shared if read(majority["luna-medium"][k]) != read(human[k])]
    print(
        f"\nall arms make the same read decision on {len(shared)}/40 pairs;"
        f" on {len(wrong)} of them it differs from Ewan: {wrong}"
    )
    split = [k for k in keys if k not in shared]
    right = {n: sum(read(majority[n][k]) == read(human[k]) for k in split) for n in majority}
    print(f"on the {len(split)} pairs where arms differ, read decisions that match Ewan: {right}")


def option(name: str, default: str) -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


async def main() -> None:
    items, human = calibration.load()
    work = calibration.batches(items)
    arms = option("--arms", ",".join(DEFAULT_ARMS)).split(",")
    cap = Decimal(option("--cap", str(CAP_USD)))
    calls = len(arms) * REPEATS * len(work)
    print(f"{arms}, {REPEATS} repeats, {len(work)} batches: {calls} calls, cap ${cap}")
    if "--from" in sys.argv:
        saved = json.loads(Path(option("--from", "")).read_text())
        report(saved["grades"], items, human)
    elif "--execute" in sys.argv:
        report(await run(work, arms, cap), items, human)


if __name__ == "__main__":
    asyncio.run(main())
