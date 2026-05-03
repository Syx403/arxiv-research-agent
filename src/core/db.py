from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from pgvector.asyncpg import register_vector
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.core.config import get_settings


_asyncpg_pool: asyncpg.Pool | None = None
_psycopg_pool: AsyncConnectionPool | None = None


def _asyncpg_dsn() -> str:
    settings = get_settings()
    password = settings.postgres_password.get_secret_value()
    return (
        f"postgresql://{settings.postgres_user}:{password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def _psycopg_conninfo() -> str:
    settings = get_settings()
    password = settings.postgres_password.get_secret_value()
    return (
        f"host={settings.postgres_host} "
        f"port={settings.postgres_port} "
        f"dbname={settings.postgres_db} "
        f"user={settings.postgres_user} "
        f"password={password}"
    )


async def _init_asyncpg_connection(conn: asyncpg.Connection) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    await register_vector(conn)


async def get_asyncpg_pool() -> asyncpg.Pool:
    global _asyncpg_pool
    if _asyncpg_pool is None:
        _asyncpg_pool = await asyncpg.create_pool(
            dsn=_asyncpg_dsn(),
            min_size=1,
            max_size=10,
            init=_init_asyncpg_connection,
        )
    return _asyncpg_pool


async def get_psycopg_pool() -> AsyncConnectionPool:
    global _psycopg_pool
    if _psycopg_pool is None:
        pool = AsyncConnectionPool(
            _psycopg_conninfo(),
            min_size=1,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        await pool.open()
        _psycopg_pool = pool
    return _psycopg_pool


@asynccontextmanager
async def acquire_app() -> AsyncIterator[asyncpg.Connection]:
    pool = await get_asyncpg_pool()
    async with pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def acquire_checkpoint() -> AsyncIterator[AsyncConnection]:
    pool = await get_psycopg_pool()
    async with pool.connection() as conn:
        yield conn


async def close_pools() -> None:
    global _asyncpg_pool, _psycopg_pool
    if _asyncpg_pool is not None:
        await _asyncpg_pool.close()
        _asyncpg_pool = None
    if _psycopg_pool is not None:
        await _psycopg_pool.close()
        _psycopg_pool = None
