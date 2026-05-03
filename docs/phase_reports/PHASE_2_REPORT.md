# Phase 2 Report

**Phase title**: Database Schema and Connection
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 2 implemented the PostgreSQL connection layer, schema, bootstrap script, and integration coverage for app data plus LangGraph checkpointing. The asyncpg app pool initializes `vector` and `pg_trgm` extensions and registers the pgvector codec; the psycopg pool uses the `autocommit=True` and `dict_row` contract required by `AsyncPostgresSaver`. Bootstrap completed cleanly twice against local Postgres on host port 5433, proving idempotency. Integration tests verify schema tables, extensions, indexes, trigger behavior, vector roundtrip, and a real LangGraph checkpoint write/read.

## Files created
- `src/core/db.py` - asyncpg app pool, psycopg checkpoint pool, acquire context managers, and shutdown.
- `src/core/sql/001_papers.sql` - papers table and indexes.
- `src/core/sql/002_chunks.sql` - chunks table with `vector(1536)`, generated `tsv`, HNSW, GIN, and paper indexes.
- `src/core/sql/003_citations.sql` - citation edge table and cited-paper index.
- `src/core/sql/004_concepts.sql` - concept, alias, and relation tables.
- `src/core/sql/005_sessions.sql` - sessions and session_messages tables.
- `src/core/sql/006_session_summaries.sql` - cached episodic summary table.
- `src/core/sql/007_triggers.sql` - `concepts.updated_at` trigger function and trigger.
- `scripts/bootstrap_db.py` - lexical SQL runner plus `AsyncPostgresSaver.setup()`.
- `tests/unit/test_sql_files.py` - non-live checks for schema file presence/order and key SQL features.
- `tests/integration/test_db_bootstrap.py` - skipped-by-default DB bootstrap integration test.
- `tests/integration/test_checkpoint_roundtrip.py` - skipped-by-default LangGraph checkpoint roundtrip test.

## Files modified
- `.env.example` - Postgres default port changed to `5433`.
- `docker/docker-compose.yml` - host port changed to `5433` mapped to container `5432`.
- `docs/STATUS.md` - H-2 confirmed and Phase 2 marked complete.
- `docs/interview_talking_points.md` - Phase 2 talking points appended.
- `docs/phase_reports/PHASE_2_QUESTIONS.md` - retained blocker audit trail and recorded the float4 assertion resolution.

## Tests run
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `39 passed in 0.52s`.
- Command: `uv run python -c "from scripts.bootstrap_db import sql_files; print([p.name for p in sql_files()])"`
- Outcome: pass; returned all seven SQL files in lexical order.
- Command: `uv run ruff check src scripts tests`
- Outcome: pass; `All checks passed!`.
- Command: `uv run python scripts/bootstrap_db.py`
- Outcome: pass; no output, exit code 0.
- Command: `uv run python scripts/bootstrap_db.py`
- Outcome: pass; idempotency rerun had no output, exit code 0.
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_db_bootstrap.py tests/integration/test_checkpoint_roundtrip.py -q`
- Outcome: pass; `2 passed in 0.52s`.
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c '\dt'`
- Outcome: pass; listed 13 tables.

## Validation commands
- Command: `uv run python scripts/bootstrap_db.py`
- Output (truncated if long): ```
<no output; exit code 0>
```
- Command: `uv run python scripts/bootstrap_db.py`
- Output (truncated if long): ```
<no output; exit code 0>
```
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_db_bootstrap.py tests/integration/test_checkpoint_roundtrip.py -q`
- Output (truncated if long): ```
..                                                                       [100%]
2 passed in 0.52s
```
- Command: `uv run pytest tests/unit -q`
- Output (truncated if long): ```
39 passed in 0.52s
```
- Command: `uv run python -c "from scripts.bootstrap_db import sql_files; print([p.name for p in sql_files()])"`
- Output (truncated if long): ```
['001_papers.sql', '002_chunks.sql', '003_citations.sql', '004_concepts.sql', '005_sessions.sql', '006_session_summaries.sql', '007_triggers.sql']
```
- Command: `uv run ruff check src scripts tests`
- Output (truncated if long): ```
All checks passed!
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c '\dt'`
- Output (truncated if long): ```
                  List of relations
 Schema |         Name          | Type  |    Owner
--------+-----------------------+-------+-------------
 public | checkpoint_blobs      | table | arxiv_agent
 public | checkpoint_migrations | table | arxiv_agent
 public | checkpoint_writes     | table | arxiv_agent
 public | checkpoints           | table | arxiv_agent
 public | chunks                | table | arxiv_agent
 public | citations             | table | arxiv_agent
 public | concept_aliases       | table | arxiv_agent
 public | concept_relations     | table | arxiv_agent
 public | concepts              | table | arxiv_agent
 public | papers                | table | arxiv_agent
 public | session_messages      | table | arxiv_agent
 public | session_summaries     | table | arxiv_agent
 public | sessions              | table | arxiv_agent
(13 rows)
```

## Acceptance checklist
- [x] `src/core/db.py` implemented with asyncpg and psycopg singleton pools.
- [x] psycopg pool uses `kwargs={"autocommit": True, "row_factory": dict_row}`.
- [x] `acquire_app()`, `acquire_checkpoint()`, and `close_pools()` are implemented.
- [x] Seven SQL files exist in lexical order with the planned schema bodies.
- [x] `scripts/bootstrap_db.py` runs SQL files in lexical order and calls `AsyncPostgresSaver.setup()`.
- [x] Integration tests are present and skipped unless `INTEGRATION_TESTS=1`.
- [x] Unit tests pass.
- [x] Live bootstrap passes.
- [x] Bootstrap idempotency rerun passes.
- [x] DB integration tests pass.
- [x] `psql \dt` shows 13 tables.
- [x] Ready for Phase 2 commit.

## Deviations from the plan
- The local Postgres host port was adapted from `5432` to `5433` in `docker/docker-compose.yml` and `.env.example` because a host-level Postgres process already occupied `5432`. Container port remains `5432`; this is a dev-environment workaround, not an architecture change.
- The vector round-trip assertion was tightened to `pytest.approx(vector)` because pgvector stores as float4 and exact double equality after round-trip is impossible. Question 2 in PHASE_2_QUESTIONS.md documents the failure and resolution.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and run those commands to verify.
