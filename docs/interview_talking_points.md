# Interview Talking Points

## Entries

- Provider abstraction: business code calls role aliases like `main` and `fast` through `src/llm/client.py`, never vendor SDKs directly.
- Retry policy: `src/llm/retry.py` retries transient provider failures with jitter while refusing deterministic auth, invalid request, and context-length errors.
- Secret masking: `src/core/logging.py` redacts bearer tokens, API keys, and key-like values at the logging-filter layer.
- Embedding dimension validation: `src/llm/providers/openai_embed.py` rejects vectors that do not match the 1536-dimensional `text-embedding-3-small` contract.
- Client lifecycle: `src/llm/client.py` caches provider clients and closes all underlying async clients through `close_llm_clients()`.
- Split-driver architecture: `src/core/db.py` uses asyncpg for application SQL because it is fast and supports `pgvector.asyncpg.register_vector`, while psycopg is reserved for LangGraph PostgresSaver with `autocommit=True` and `dict_row`.
- Hybrid retrieval-ready schema: `src/core/sql/002_chunks.sql` stores `vector(1536)` with HNSW plus a generated stored `tsvector` with GIN, ready for Phase 4 RRF fusion.
- Concept graph schema: `src/core/sql/004_concepts.sql` models concepts, aliases, and relations with JSONB `evidence_chunks` for chunk-level grounding.
- Updated-at trigger: `src/core/sql/007_triggers.sql` keeps concept mutation timestamps observable for tests and later reasoning.
- LangGraph checkpointing: `scripts/bootstrap_db.py` calls `AsyncPostgresSaver.setup()` over the psycopg pool, proving LangGraph 1.x checkpoint integration against local Postgres.
- pgvector precision: I learned the hard way that pgvector stores as float4. The integration test uses `pytest.approx` to validate the round-trip semantically rather than bit-equality.
- Lazy ingestion architecture: Phase 3 collects arXiv metadata, PDFs, PDF text, embeddings, and Semantic Scholar references before opening the short database write transaction.
- Section-aware chunking: `src/corpus/chunking.py` uses `tiktoken` `cl100k_base` with an 800-token target and 100-token overlap while preserving section boundaries.
- Crash-resilient ingestion lifecycle: papers move through `in_progress`, `complete`, and `failed`, so complete rows are skipped and failed rows can be cleaned and retried.
- Semantic Scholar pacing: the async client uses 1 request/sec without a key and 10 requests/sec with a key, with tenacity retry on HTTP 429.
- arXiv pacing: the sync `arxiv` package runs through `asyncio.to_thread`, and a single global client plus lock preserves 3-second pacing across the process.
- Decomposition produces a dependency DAG of sub-questions, not just a flat list.
- HyDE is heuristic-gated, so short or keyword-style queries do not spend LLM calls on hypothetical answers.
- Self-RAG uses cost tiers: batched relevance judging runs on `fast`, while sufficiency and citation verification use `main`.
- Router stays pure: it queries concept aliases and emits policy flags, but never performs retrieval or live expansion.
- JSON repair retry gives decomposition one "fix your JSON" attempt before surfacing validation failure.
- Bounded multi-hop with LLM-judged frontier prevents combinatorial explosion of citation expansion.
- Lazy ingestion means the corpus grows during a session; the agent extends its own memory.
- Wall-clock budget is separate from depth and frontier limits, giving three independent safety knobs.
- Phase 6's walker traverses citation edges bidirectionally inside the local corpus and outbound-only for lazy ingest: outbound finds what this paper depended on, incoming finds what built on this paper.
- Compression policy keeps recent messages verbatim and prepends a recursive summary of older history; the cache is keyed by upper message id so cost stays bounded.
- The compressor is a pure consumer of `session_messages` and `session_summaries`, with no global state, making cache behavior fully testable.
- Three-tier entity linking in `src/memory/semantic/linker.py` uses alias, canonical, then embedding plus an LLM judge, which is a practical fallback chain for noisy LLM concept extractions.
- The concept graph is an Input Mode signal, not decoration: `graph_retriever.py` expands a query through graph neighbors and derives paper hints from chunk-level relation evidence.
- Reflection idempotency is enforced through canonical aliases and upserted relation evidence, so replaying a session promotion is safe.
