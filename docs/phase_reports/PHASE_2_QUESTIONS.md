# Phase 2 Questions

## Question 1: Postgres is reachable but missing expected role

**Context**: After H-2 confirmation, Codex resumed Phase 2 validation and ran `uv run python scripts/bootstrap_db.py` as instructed.
**Problem**: The bootstrap script failed before executing schema SQL because asyncpg connected to localhost:5432 but PostgreSQL reported `role "arxiv_agent" does not exist`. This suggests the running Postgres instance/volume was not initialized with the credentials from `docker/docker-compose.yml`, or another local Postgres service is occupying port 5432.

**Command run**:

```bash
uv run python scripts/bootstrap_db.py
```

**Traceback**:

```text
Traceback (most recent call last):
  File "/Users/richsion/Desktop/arxiv-research-agent/scripts/bootstrap_db.py", line 36, in <module>
    asyncio.run(main())
    ~~~~~~~~~~~^^^^^^^^
  File "/opt/homebrew/anaconda3/lib/python3.13/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "/opt/homebrew/anaconda3/lib/python3.13/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "/opt/homebrew/anaconda3/lib/python3.13/asyncio/base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "/Users/richsion/Desktop/arxiv-research-agent/scripts/bootstrap_db.py", line 30, in main
    await bootstrap()
  File "/Users/richsion/Desktop/arxiv-research-agent/scripts/bootstrap_db.py", line 19, in bootstrap
    async with db.acquire_app() as conn:
               ~~~~~~~~~~~~~~^^
  File "/opt/homebrew/anaconda3/lib/python3.13/contextlib.py", line 214, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/richsion/Desktop/arxiv-research-agent/src/core/db.py", line 75, in acquire_app
    pool = await get_asyncpg_pool()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/richsion/Desktop/arxiv-research-agent/src/core/db.py", line 49, in get_asyncpg_pool
    _asyncpg_pool = await asyncpg.create_pool(
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
    )
    ^
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/pool.py", line 439, in _async__init__
    await self._initialize()
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/pool.py", line 466, in _initialize
    await first_ch.connect()
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/pool.py", line 153, in connect
    self._con = await self._pool._get_new_connection()
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/pool.py", line 538, in _get_new_connection
    con = await self._connect(
          ^^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
    )
    ^
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/connection.py", line 2443, in connect
    return await connect_utils._connect(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    ...<22 lines>...
    )
    ^
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/connect_utils.py", line 1218, in _connect
    conn = await _connect_addr(
           ^^^^^^^^^^^^^^^^^^^^
    ...<6 lines>...
    )
    ^
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/connect_utils.py", line 1054, in _connect_addr
    return await __connect_addr(params, True, *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/richsion/Desktop/arxiv-research-agent/.venv/lib/python3.13/site-packages/asyncpg/connect_utils.py", line 1102, in __connect_addr
    await connected
asyncpg.exceptions.InvalidAuthorizationSpecificationError: role "arxiv_agent" does not exist
```

**Options considered**:
1. Owner verifies the running service is the planned Docker container (`arxiv_agent_pg`) and not another local Postgres on port 5432; if needed, stop the conflicting service and rerun `make db-up`.
2. Owner resets the Phase 2 Docker volume with `make db-reset` so Postgres initializes from `docker/docker-compose.yml` with `POSTGRES_USER=arxiv_agent`, `POSTGRES_DB=arxiv_agent`, and `POSTGRES_PASSWORD=arxiv_agent_dev`. Pros: aligns with plan defaults. Cons: destroys the local Postgres volume, but Phase 2 has no project data yet.
3. Owner manually creates the `arxiv_agent` role and `arxiv_agent` database in the currently running Postgres. Pros: preserves the existing service. Cons: more manual drift from the plan.

**Recommendation**: Option 1 first, then Option 2 if the container is correct but the volume was initialized with different credentials.

## Question 2: pgvector roundtrip test uses exact float equality

**Context**: After the owner resolved the port conflict by moving project Postgres to host port 5433, Codex reran Phase 2 validation. `uv run python scripts/bootstrap_db.py` completed successfully twice, confirming bootstrap and idempotency. Then Codex ran the requested integration tests.
**Problem**: `tests/integration/test_checkpoint_roundtrip.py` passed, but `tests/integration/test_db_bootstrap.py` failed only on exact float equality for a pgvector roundtrip. The selected vector prints with the same apparent value, but exact binary equality is not stable for PostgreSQL/pgvector float storage. The test should likely use an approximate comparison while still asserting length and values.

**Command run**:

```bash
INTEGRATION_TESTS=1 uv run pytest tests/integration/test_db_bootstrap.py tests/integration/test_checkpoint_roundtrip.py -q
```

**Traceback**:

```text
=================================== FAILURES ===================================
_ test_bootstrap_creates_schema_extensions_indexes_trigger_and_vector_roundtrip _

    @pytest.mark.asyncio
    async def test_bootstrap_creates_schema_extensions_indexes_trigger_and_vector_roundtrip() -> None:
        try:
            await bootstrap()

            async with db.acquire_app() as conn:
                table_rows = await conn.fetch(
                    """
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    """
                )
                assert EXPECTED_TABLES.issubset({row["tablename"] for row in table_rows})

                extension_rows = await conn.fetch("SELECT extname FROM pg_extension")
                assert {"vector", "pg_trgm"}.issubset({row["extname"] for row in extension_rows})

                index_rows = await conn.fetch(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    """
                )
                index_names = {row["indexname"] for row in index_rows}
                assert "chunks_embedding_hnsw" in index_names
                assert "chunks_tsv_idx" in index_names
                assert "concept_relations_dst_idx" in index_names

                await conn.execute(
                    """
                    INSERT INTO concepts (canonical, display, definition)
                    VALUES ('phase2-test', 'Phase 2 Test', 'trigger check')
                    ON CONFLICT (canonical) DO UPDATE SET definition = EXCLUDED.definition
                    """
                )
                before = await conn.fetchval(
                    "SELECT updated_at FROM concepts WHERE canonical = 'phase2-test'"
                )
                await conn.execute(
                    "UPDATE concepts SET definition = 'trigger check updated' WHERE canonical = 'phase2-test'"
                )
                after = await conn.fetchval(
                    "SELECT updated_at FROM concepts WHERE canonical = 'phase2-test'"
                )
                assert after >= before

                vector = [0.001] * EMBEDDING_DIM
                await conn.execute(
                    """
                    INSERT INTO papers (paper_id, source, title)
                    VALUES ('test:vector-roundtrip', 'arxiv', 'Vector Roundtrip')
                    ON CONFLICT (paper_id) DO UPDATE SET title = EXCLUDED.title
                    """
                )
                chunk_id = await conn.fetchval(
                    """
                    INSERT INTO chunks (paper_id, section, ord, text, token_count, embedding)
                    VALUES ('test:vector-roundtrip', 'Test', 1, 'Vector roundtrip text', 3, $1)
                    ON CONFLICT (paper_id, ord) DO UPDATE SET embedding = EXCLUDED.embedding
                    RETURNING chunk_id
                    """,
                    vector,
                )
                selected = await conn.fetchval(
                    "SELECT embedding FROM chunks WHERE chunk_id = $1",
                    chunk_id,
                )
>               assert list(selected) == vector
E               assert [0.001, 0.001...1, 0.001, ...] == [0.001, 0.001...1, 0.001, ...]
E
E                 At index 0 diff: 0.001 != 0.001
E                 Use -v to get more diff

tests/integration/test_db_bootstrap.py:104: AssertionError
=========================== short test summary info ============================
FAILED tests/integration/test_db_bootstrap.py::test_bootstrap_creates_schema_extensions_indexes_trigger_and_vector_roundtrip
1 failed, 1 passed in 0.41s
```

**Options considered**:
1. Change the vector assertion to `assert list(selected) == pytest.approx(vector)`. Pros: matches floating-point roundtrip reality while preserving value validation. Cons: small test-only deviation from exact list equality.
2. Use a vector of `0.0` values to make exact equality likely. Pros: minimal assertion change. Cons: weaker coverage because all-zero vectors can hide adaptation/conversion issues.
3. Cast both lists to rounded decimal strings before comparing. Pros: deterministic. Cons: less idiomatic than `pytest.approx`.

**Recommendation**: Option 1. It keeps the test focused on roundtrip correctness without requiring exact binary float identity.

**Resolution**: Option 1 applied. pytest.approx with default relative tolerance (1e-6) is correct for float4 round-trip and keeps the test focused on encoding/decoding correctness rather than impossible bit-level identity.
