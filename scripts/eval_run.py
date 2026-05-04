from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from src.eval.runner import EvalThresholdError, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the arxiv-research-agent gold eval suite.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    parser.add_argument("--ids", nargs="*", default=None, help="Run specific question IDs.")
    parser.add_argument("--question-ids", default=None, help="Run comma-separated question IDs.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    question_ids = args.ids
    if args.question_ids:
        question_ids = [question_id.strip() for question_id in args.question_ids.split(",") if question_id.strip()]
    try:
        await run_eval(limit=args.limit, question_ids=question_ids)
    except EvalThresholdError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    asyncio.run(main())
