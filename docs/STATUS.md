# Project Status

## Current entry point and validation

The single-question CLI is available via `uv run python -m src.cli`, with a
credential-free `--example` preview. The browser chat and graph interfaces remain
unimplemented; the Phase 11 row below still refers to those interfaces.

The October 1, 2026 documentation/citation pass corrects claim spans, evidence
identity, verification context, evidence retention across citation expansion,
and failed-answer memory handling. See [the architecture note](architecture.md)
for the exact changes. Offline regression checks pass; no new live-model score
or 30-question benchmark is claimed. The historical Phase 10.5 citation-quality
limitation remains pending a fresh evaluation.

## Historical phases

| Phase | Title | Status | Commit | Date | Owner handoff |
|-------|-------|--------|--------|------|---------------|
| 0 | Bootstrap | complete | 0d7530e | 2026-05-03 | H-1 confirmed 2026-05-03; H-2 confirmed 2026-05-03; H-3 confirmed 2026-05-03 |
| 1 | Provider Adapter Layer | complete | b69aa26 + HEAD (see git log) | 2026-05-03 | none |
| 2 | Database Schema and Connection | complete | HEAD (see git log) | 2026-05-03 | none |
| 3 | Corpus Clients and Ingestion | complete | 609ab95 + HEAD (see git log) | 2026-05-03 | none |
| 4 | Vector Index and Hybrid Retrieval | complete | HEAD (see git log) | 2026-05-03 | none |
| 5 | Agentic RAG Components | complete | HEAD (see git log) | 2026-05-03 | none |
| 6 | Multi-hop Citation Walker | complete | HEAD (see git log) | 2026-05-03 | none |
| 7 | Episodic Memory and Compression | complete | HEAD (see git log) | 2026-05-03 | none |
| 8 | Semantic Memory and Concept Graph | complete | HEAD (see git log) | 2026-05-03 | none |
| 9 | LangGraph Orchestration | complete | HEAD (see git log) | 2026-05-03 | none |
| 10 | Evaluation Harness | complete | HEAD (see git log) | 2026-05-04 | none |
| 10.5 | Fail-closed Self-RAG and Sample Eval Gate | complete | HEAD (see git log) | 2026-05-04 | self-RAG fail-closed + token cost visible |
| 11 | User Interface | not started | - | - | - |
| 12 | Polish and Documentation | not started | - | - | - |
