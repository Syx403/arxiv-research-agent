# Project Status

## Current runtime — 2026-10-02

The FastAPI browser UI is implemented. `make start` installs locked dependencies, starts and waits for PostgreSQL, applies migrations, and serves http://127.0.0.1:8000. `make ui` starts only the web server. See [INTERVIEW_DEMO.md](INTERVIEW_DEMO.md).

The remote CLI/offline example have been reconciled with the current project. Live CLI questions now use the same AgentRuntime, cost ledger and delivery contract as the browser; `--session` continues a conversation. `make demo` remains an explicitly hand-authored, credential-free output example.

The current architecture performs arXiv discovery and on-demand original-text RAG. It no longer runs the old concept graph, citation walker, HyDE or reflection pipeline. September evaluation records are in [RESEARCH_EVALUATION.md](RESEARCH_EVALUATION.md); historical phase completion below does not describe the active architecture or its current accuracy.

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
| 10.6 | Engineering Rigor + Foundation | complete | HEAD (see git log) | 2026-05-05 | F3=substantial support; F6=typed counters; F8 deferred items in ENGINEERING_BACKLOG.md |
| 11 | User Interface | not started | - | - | - |
| 12 | Polish and Documentation | not started | - | - | - |
