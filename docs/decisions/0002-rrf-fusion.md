# ADR 0002: Reciprocal Rank Fusion for Hybrid Retrieval

**Status**: accepted
**Date**: 2026-05-03

## Context

Phase 4 combines pgvector cosine retrieval with PostgreSQL full-text retrieval over the `chunks.tsv` column. These two legs produce scores on different scales, and those score distributions can drift as the corpus, tokenizer, embedding model, or text extraction quality changes.

## Decision

Use Reciprocal Rank Fusion (RRF) to combine the vector and lexical result lists. RRF is rank-based, score-unit-free, and robust to score-distribution drift because each leg contributes `1 / (k + rank)` rather than raw similarity or lexical scores.

Use `k=60`, following the common TREC-OAR convention. This keeps high-ranked results influential while still allowing agreement between both legs to lift shared chunks.

## Naming

Call the lexical leg "lexical full-text rank", not "BM25". PostgreSQL `ts_rank_cd` is a tf-based cover-density ranking function, not BM25.
