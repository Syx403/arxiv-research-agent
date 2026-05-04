from __future__ import annotations

import argparse
import asyncio
import logging

from src.eval.runner import run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the arxiv-research-agent gold eval suite.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    parser.add_argument("--ids", nargs="*", default=None, help="Run specific question IDs.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    await run_eval(limit=args.limit, question_ids=args.ids)


if __name__ == "__main__":
    asyncio.run(main())
