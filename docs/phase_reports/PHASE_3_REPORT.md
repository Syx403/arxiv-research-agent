# Phase 3 Report

**Phase title**: Corpus Clients and Ingestion
**Status**: complete
**Commit**: `HEAD (see git log)`
**Owner handoff pending**: none

## Summary
Phase 3 implemented the corpus acquisition and transactional ingestion path for arXiv seed papers. The ingestion flow collects external artifacts first, embeds chunks outside the database transaction, then performs a short atomic write for `papers`, `chunks`, and `citations`. Validation exposed two real-world ingestion edges: transient/incomplete arXiv PDF downloads and PDF text containing NUL bytes; both were hardened before final acceptance. The full seed corpus now ingests 30/30 papers, produces 2049 chunks, and has citation rows for 100% of completed papers.

## Files created
- `src/corpus/__init__.py` - corpus package marker.
- `src/corpus/arxiv_client.py` - async wrapper around the sync arXiv client with PDF download retry handling.
- `src/corpus/semantic_scholar.py` - async Semantic Scholar Graph API client with rate limiting and 429 retry.
- `src/corpus/github_client.py` - minimal async README fetcher for later phases.
- `src/corpus/pdf_loader.py` - PDF text extraction with section-header heuristics and NUL-byte sanitization.
- `src/corpus/chunking.py` - section-aware token chunking using `cl100k_base`.
- `src/corpus/ingestion.py` - collect-then-short-transaction paper ingestion with idempotent retry.
- `src/eval/__init__.py` - eval package marker.
- `src/eval/datasets/__init__.py` - dataset package marker.
- `src/eval/datasets/seed_papers.py` - 30-paper seed arXiv ID list from Appendix B.
- `scripts/ingest_seed_corpus.py` - concurrency-capped seed ingestion script.
- `tests/unit/test_chunking.py` - chunk size and section-boundary tests.
- `tests/unit/test_pdf_loader.py` - PDF section extraction and NUL sanitization test.
- `tests/integration/test_ingest_one.py` - live single-paper ingestion integration test.
- `tests/integration/test_ingest_partial_failure.py` - rollback test for embedding failure after the first batch.

## Files modified
- `docs/STATUS.md` - H-3 confirmed and Phase 3 marked complete.
- `docs/interview_talking_points.md` - Phase 3 talking points appended.

## Tests run
- Command: `uv run pytest tests/unit -q`
- Outcome: pass; `42 passed in 0.69s`.
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_ingest_one.py tests/integration/test_ingest_partial_failure.py -q`
- Outcome: pass; `2 passed in 9.61s`.
- Command: `make db-reset`
- Outcome: pass; Docker volume recreated and `scripts/bootstrap_db.py` exited 0.
- Command: `make ingest`
- Outcome: pass after hardening; final idempotency rerun skipped all 30 complete papers.
- Command: `uv run ruff check src scripts tests`
- Outcome: pass; `All checks passed!`.

## Validation commands
- Command: `make db-reset`
- Output (truncated if long): ```
docker compose -f docker/docker-compose.yml down -v
docker compose -f docker/docker-compose.yml up -d
sleep 3
uv run python scripts/bootstrap_db.py
```
- Command: `make ingest`
- Output (truncated if long): ```
Summary: papers ingested=14, papers skipped=0, papers failed=16, chunks inserted=804, citations inserted=670
Summary: papers ingested=15, papers skipped=14, papers failed=1, chunks inserted=1099, citations inserted=736
Summary: papers ingested=1, papers skipped=29, papers failed=0, chunks inserted=146, citations inserted=50
```
- Command: `make ingest`
- Output (truncated if long): ```
Summary: papers ingested=0, papers skipped=30, papers failed=0, chunks inserted=0, citations inserted=0
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM papers WHERE ingestion_status='complete';"`
- Output (truncated if long): ```
 count
-------
    30
(1 row)
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM chunks;"`
- Output (truncated if long): ```
 count
-------
  2049
(1 row)
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT count(*) FROM citations;"`
- Output (truncated if long): ```
 count
-------
  1456
(1 row)
```
- Command: `docker exec arxiv_agent_pg psql -U arxiv_agent -d arxiv_agent -c "SELECT 100.0 * count(DISTINCT citing_paper_id) / NULLIF((SELECT count(*) FROM papers WHERE ingestion_status='complete'),0) AS pct FROM citations;"`
- Output (truncated if long): ```
         pct
----------------------
 100.0000000000000000
(1 row)
```
- Command: `uv run pytest tests/unit -q`
- Output (truncated if long): ```
..........................................                               [100%]
42 passed in 0.69s
```
- Command: `INTEGRATION_TESTS=1 uv run pytest tests/integration/test_ingest_one.py tests/integration/test_ingest_partial_failure.py -q`
- Output (truncated if long): ```
..                                                                       [100%]
2 passed in 9.61s
```
- Command: `uv run ruff check src scripts tests`
- Output (truncated if long): ```
All checks passed!
```

## Acceptance checklist
- [x] `make ingest` ingests the full seed list without crashing.
- [x] Database contains at least 25 complete papers; observed 30.
- [x] Database contains at least 800 chunks; observed 2049.
- [x] At least 80% of ingested papers have at least one citation row; observed 100%.
- [x] `tests/unit/test_chunking.py` passes.
- [x] `tests/unit/test_pdf_loader.py` passes.
- [x] Re-running `make ingest` is a no-op for already-complete papers.
- [x] Partial-failure integration test passes.

## Deviations from the plan
- The arXiv PDF download path was hardened with manual streaming, partial-file cleanup, and bounded retry because the package `download_pdf()` left incomplete 1 MiB files on transient failures.
- Failed-paper retries now redownload PDFs so a prior partial file cannot poison the retry path.
- PDF loader output removes NUL bytes because PostgreSQL text columns cannot store `\x00`; validation found one seed PDF with this condition.

## Open questions for the planner
None.

## Ready for review
The planner can read these files and run those commands to verify.
