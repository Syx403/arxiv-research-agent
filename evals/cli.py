"""`ara eval prepare|plan|run|report` (DESIGN §11.5); `run` is a dry run without --execute."""

import argparse
import asyncio
import json
from decimal import Decimal

from ara.db.migrate import migrate
from ara.db.pool import make_pool
from ara.settings import ROOT, get_settings
from evals import runner
from evals.report import report
from evals.suites import s1, s2

REPORTS = ROOT / "data/eval_reports"


def main() -> None:
    parser = argparse.ArgumentParser(prog="ara")
    commands = parser.add_subparsers(dest="area", required=True)
    evals = commands.add_parser("eval", help="evaluation suites").add_subparsers(
        dest="command", required=True
    )
    evals.add_parser("prepare", help="download QASPER and draw any missing S1/S2 manifest")
    for name in ("plan", "run"):
        sub = evals.add_parser(name)
        sub.add_argument("suite", choices=["s1", "s2"])
        sub.add_argument("--papers", type=int, help="S1: only the first N papers of the manifest")
        sub.add_argument("--limit", type=int, help="S2: only the first N items of the manifest")
        if name == "run":
            sub.add_argument("--execute", action="store_true", help="send billable requests")
            sub.add_argument("--max-usd", type=Decimal, default=Decimal("1.0"))
    perturb = evals.add_parser("perturb", help="generate the S5 claims for review (billable)")
    perturb.add_argument("--execute", action="store_true", help="send billable requests")
    perturb.add_argument("--max-usd", type=Decimal, default=Decimal("0.01"))
    evals.add_parser("report").add_argument("run_id")
    args = parser.parse_args()

    match args.command:
        case "prepare":
            for suite in (s1, s2):  # S2 extends S1's items, so S1 comes first
                if suite.MANIFEST.exists():
                    print(f"{suite.MANIFEST} exists; delete it to redraw")
                else:
                    suite.MANIFEST.write_text(json.dumps(suite.build_manifest(), indent=2) + "\n")
                    print(f"wrote {suite.MANIFEST}")
        case "plan" | "run":
            execute = args.command == "run" and args.execute
            max_usd = getattr(args, "max_usd", Decimal("1.0"))
            if args.suite == "s1":
                run = runner.run_s1(papers=args.papers, execute=execute, max_usd=max_usd)
            else:
                run = runner.run_s2(limit=args.limit, execute=execute, max_usd=max_usd)
            run_id = asyncio.run(run)
            if run_id:
                print(asyncio.run(_report(run_id)))
        case "perturb":
            asyncio.run(runner.perturb_s5(execute=args.execute, max_usd=args.max_usd))
        case "report":
            print(asyncio.run(_report(args.run_id)))


async def _report(run_id: str) -> str:
    settings = get_settings()
    migrate(settings.database_url)
    async with make_pool(settings.database_url) as pool:
        text, export = await report(pool, run_id)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{run_id}.md").write_text(text)
    (REPORTS / f"{run_id}.jsonl").write_text(export)
    return text


if __name__ == "__main__":
    main()
