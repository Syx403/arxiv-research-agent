from __future__ import annotations

import asyncio
from pathlib import Path

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.core import db


SQL_DIR = Path(__file__).resolve().parents[1] / "src" / "core" / "sql"


def sql_files() -> list[Path]:
    return sorted(SQL_DIR.glob("*.sql"))


async def bootstrap() -> None:
    async with db.acquire_app() as conn:
        for path in sql_files():
            await conn.execute(path.read_text())

    pool = await db.get_psycopg_pool()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()


async def main() -> None:
    try:
        await bootstrap()
    finally:
        await db.close_pools()


if __name__ == "__main__":
    asyncio.run(main())
