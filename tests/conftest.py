"""Unit tests run against a real Postgres: the `ara_test` database (DESIGN §12, no mocks)."""

import os
from collections.abc import AsyncIterator

import pytest

from ara.db.migrate import migrate
from ara.db.pool import Pool, make_pool

TEST_DATABASE_URL = os.environ.get(
    "ARA_TEST_DATABASE_URL", "postgresql://ara:ara@127.0.0.1:5434/ara_test"
)


@pytest.fixture(scope="session")
def test_database() -> str:
    migrate(TEST_DATABASE_URL)
    return TEST_DATABASE_URL


@pytest.fixture
async def pool(test_database: str) -> AsyncIterator[Pool]:
    async with make_pool(test_database) as pool:
        async with pool.connection() as conn:
            await conn.execute("TRUNCATE llm_calls RESTART IDENTITY")
        yield pool
