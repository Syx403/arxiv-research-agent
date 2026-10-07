"""Live tests call real, billable models and run only with `pytest -m live` (CLAUDE.md rule 1).
Their spend goes to the main ledger (database `ara`), never to the unit-test database."""

from collections.abc import AsyncIterator, Iterator

import pytest
from langsmith.run_trees import get_cached_client

from ara.db.migrate import migrate
from ara.db.pool import make_pool
from ara.llm.gateway import Gateway
from ara.llm.ledger import Ledger
from ara.settings import Settings, configure_tracing, get_settings


@pytest.fixture(scope="session")
def settings() -> Iterator[Settings]:
    settings = get_settings()
    configure_tracing(settings)
    migrate(settings.database_url)
    yield settings
    get_cached_client().flush()  # send buffered LangSmith runs before the process exits


@pytest.fixture
async def gateway(settings: Settings) -> AsyncIterator[Gateway]:
    async with make_pool(settings.database_url) as pool:
        ledger = Ledger(
            pool, global_cap_usd=settings.global_cap_usd, turn_cap_usd=settings.turn_cap_usd
        )
        gateway = Gateway(settings, ledger)
        yield gateway
        await gateway.aclose()
