from __future__ import annotations

import os

import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from scripts.bootstrap_db import bootstrap
from src.core import db


pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION_TESTS") != "1",
    reason="checkpoint integration tests require INTEGRATION_TESTS=1 and running Postgres",
)


@pytest.mark.asyncio
async def test_checkpoint_roundtrip_with_psycopg_pool() -> None:
    try:
        await bootstrap()
        pool = await db.get_psycopg_pool()
        saver = AsyncPostgresSaver(pool)

        config = {"configurable": {"thread_id": "phase2-roundtrip", "checkpoint_ns": ""}}
        checkpoint = {
            "v": 1,
            "id": "phase2-checkpoint-1",
            "ts": "2026-05-03T00:00:00+00:00",
            "channel_values": {"question": "What is ReAct?"},
            "channel_versions": {"question": 1},
            "versions_seen": {},
        }
        metadata = {"source": "phase2-test"}
        new_versions = {"question": 1}

        saved_config = await saver.aput(config, checkpoint, metadata, new_versions)
        loaded = await saver.aget_tuple(saved_config)

        assert loaded is not None
        assert loaded.checkpoint["id"] == checkpoint["id"]
        assert loaded.checkpoint["channel_values"] == checkpoint["channel_values"]
    finally:
        await db.close_pools()
