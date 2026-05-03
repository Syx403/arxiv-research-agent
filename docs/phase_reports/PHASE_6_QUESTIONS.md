# Phase 6 Questions

**Phase title**: Bounded Multi-hop Citation Walker
**Status**: resolved; implementation proceeded with approved options
**Date**: 2026-05-03

## Q1: Citation identifier shape is mixed, but lazy ingestion only accepts arXiv IDs

### Context

Phase 6 requires lazy ingestion through `ingestion.ingest_paper(arxiv_id)`, which only accepts a normalized arXiv ID. Phase 3 stores references in `citations.cited_paper_id` from `src/corpus/semantic_scholar.py::_paper_identifier()`, which prioritizes `externalIds.ArXiv`, then falls back to `CorpusId`, then Semantic Scholar `paperId`, then a title slug.

Live corpus evidence:

```sql
SELECT CASE
  WHEN cited_paper_id LIKE 'arxiv:%' THEN 'arxiv'
  WHEN cited_paper_id LIKE 's2:%' THEN 's2'
  WHEN cited_paper_id LIKE 'title:%' THEN 'title'
  ELSE 'other'
END AS id_kind, count(*)
FROM citations
GROUP BY 1
ORDER BY 1;
```

Observed:

```text
arxiv | 1224
s2    |  163
title |   69
```

Sample non-arXiv rows include `s2:226305980`, `s2:263890629`, and `title:*` slugs. These cannot be passed directly to `ingestion.ingest_paper()`.

### Options

1. **ArXiv-only Phase 6 traversal**: enqueue only `cited_paper_id LIKE 'arxiv:%'`; explicitly log skipped `s2:*` and `title:*` references with counts. No schema change, no extra Semantic Scholar calls, and lazy ingestion remains well-defined. This misses about 16% of current citation rows.
2. **Resolve non-arXiv IDs at walk time**: add Semantic Scholar resolution for `s2:*` / `title:*` candidates and enqueue only those with `externalIds.ArXiv`. This improves recall but adds live API calls, rate-limit risk, and extra failure modes inside the walker.
3. **Extend the citations schema**: add raw Semantic Scholar external IDs or a `cited_arxiv_id` column, then backfill/re-ingest. This is the most explicit long-term model but expands Phase 6 into a schema/data migration and corpus refresh.

### Recommendation

Choose option 1 for Phase 6: implement arXiv-only traversal, but make skips observable via structured logging so non-arXiv references are not silently ignored. Defer schema enrichment or S2 resolution until there is a measured need.

## Q2: The required ReAct → Reflexion integration test needs reverse citations, not outbound references

### Context

§11.2 says the walker marks seeds visited and enqueues each seed paper's references. The requested integration test seeds ReAct (`arxiv:2210.03629`) and expects Reflexion (`arxiv:2303.11366`) to be visited at depth >= 1.

Live corpus evidence:

```sql
SELECT citing_paper_id, cited_paper_id, cited_title
FROM citations
WHERE citing_paper_id='arxiv:2210.03629'
  AND (cited_paper_id='arxiv:2303.11366' OR cited_title ILIKE '%Reflexion%');
```

Observed: zero rows. ReAct's outbound references do not include Reflexion.

```sql
SELECT citing_paper_id, cited_paper_id, cited_title
FROM citations
WHERE citing_paper_id='arxiv:2303.11366'
  AND (cited_paper_id='arxiv:2210.03629' OR cited_title ILIKE '%ReAct%');
```

Observed:

```text
arxiv:2303.11366 | arxiv:2210.03629 | ReAct: Synergizing Reasoning and Acting in Language Models
```

Reflexion cites ReAct. It is therefore reachable from ReAct only if the walker also considers incoming citation edges, at least from papers already present in the local corpus.

### Options

1. **Follow §11 literally, outbound references only**: implement only `citing_paper_id = seed` expansion. This will not satisfy the requested ReAct→Reflexion hard assertion unless the test question/seed is changed.
2. **Use bidirectional local citation expansion**: enqueue outbound references and local incoming citations (`citing_paper_id` where `cited_paper_id = current`). This makes Reflexion reachable from ReAct while still allowing outbound non-seed arXiv references to exercise lazy ingestion.
3. **Fetch live Semantic Scholar citations for incoming edges**: call Semantic Scholar `fetch_citations()` for each current paper in addition to references. This may find Reflexion and other citing papers even if not in the local corpus, but it adds live API cost/rate-limit exposure and is not in §11.

### Recommendation

Choose option 2: implement bidirectional local citation expansion for Phase 6, limited to arXiv IDs per Q1. It satisfies the specified ReAct→Reflexion test without introducing new schema or new live S2 citation calls, and it still permits lazy ingestion through outbound arXiv references that are not already complete.

## Resolution

The owner approved Q1 option 1 and Q2 option 2. Phase 6 therefore implements arXiv-only traversal with logged non-arXiv skip counts, plus bidirectional local citation expansion where outbound references can lazy-ingest and incoming local citers make follow-up papers reachable without extra external API calls.
