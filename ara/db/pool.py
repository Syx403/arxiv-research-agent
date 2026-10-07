"""The async connection pool, shared by the ledger and later by LangGraph's checkpointer."""

from collections.abc import Mapping
from typing import Any, LiteralString

from psycopg import AsyncConnection
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

type Connection = AsyncConnection[DictRow]
type Pool = AsyncConnectionPool[Connection]


def make_pool(url: str, max_size: int = 10) -> Pool:
    """A closed pool; open it with `async with make_pool(url) as pool:`."""
    return AsyncConnectionPool(
        url,
        max_size=max_size,
        open=False,
        connection_class=AsyncConnection[DictRow],
        kwargs={"autocommit": True, "row_factory": dict_row},
    )


async def fetch_one(conn: Connection, sql: LiteralString, params: Mapping[str, Any]) -> DictRow:
    """Run a query that always returns exactly one row (an aggregate, or INSERT ... RETURNING)."""
    row = await (await conn.execute(sql, params)).fetchone()
    if row is None:
        raise LookupError(f"no row returned by: {sql}")
    return row
