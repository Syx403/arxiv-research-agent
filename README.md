# arxiv-research-agent

A local research-oriented conversational agent for exploring the literature on LLM-based agents. It combines agentic RAG, citation-aware retrieval, multi-hop citation traversal, and tiered memory over a curated arXiv seed corpus.

## Prerequisites

- Python 3.11+
- Docker
- `uv`
- API keys for OpenRouter, OpenAI, Cohere, and DeepSeek

## Quickstart

```bash
cp .env.example .env
# Fill .env with provider keys and local Postgres settings.
make db-up
make install
make ingest
make ui
```

## Directory Layout

```text
arxiv-research-agent/
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── retry.py
│   │   ├── errors.py
│   │   ├── registry.py
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── openrouter.py
│   │       ├── deepseek.py
│   │       ├── openai_embed.py
│   │       └── cohere_rerank.py
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── arxiv_client.py
│   │   ├── semantic_scholar.py
│   │   ├── github_client.py
│   │   ├── pdf_loader.py
│   │   ├── chunking.py
│   │   └── ingestion.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── index/
│   │   │   ├── __init__.py
│   │   │   ├── pgvector_store.py
│   │   │   └── hybrid_search.py
│   │   ├── query/
│   │   │   ├── __init__.py
│   │   │   ├── decomposer.py
│   │   │   ├── rewriter.py
│   │   │   └── router.py
│   │   ├── postprocess/
│   │   │   ├── __init__.py
│   │   │   ├── reranker.py
│   │   │   └── deduplicator.py
│   │   ├── self_rag/
│   │   │   ├── __init__.py
│   │   │   ├── relevance_judge.py
│   │   │   ├── sufficiency_check.py
│   │   │   └── verifier.py
│   │   └── multi_hop/
│   │       ├── __init__.py
│   │       ├── citation_walker.py
│   │       └── frontier.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── working.py
│   │   ├── reflection.py
│   │   ├── episodic/
│   │   │   ├── __init__.py
│   │   │   ├── session_store.py
│   │   │   └── compressor.py
│   │   └── semantic/
│   │       ├── __init__.py
│   │       ├── extractor.py
│   │       ├── linker.py
│   │       ├── graph_store.py
│   │       └── graph_retriever.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── edges.py
│   │   ├── builder.py
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── decompose.py
│   │       ├── retrieve.py
│   │       ├── self_rag.py
│   │       ├── multi_hop.py
│   │       ├── synthesize.py
│   │       └── reflect.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   ├── seed_papers.py
│   │   │   └── gold_questions.py
│   │   └── metrics/
│   │       ├── __init__.py
│   │       ├── retrieval_metrics.py
│   │       ├── citation_metrics.py
│   │       └── llm_judge.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── elements.py
│   │   └── graph_view/
│   │       ├── __init__.py
│   │       ├── server.py
│   │       └── static/
│   │           └── index.html
│   └── core/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       ├── types.py
│       ├── db.py
│       └── sql/
│           ├── 001_papers.sql
│           ├── 002_chunks.sql
│           ├── 003_citations.sql
│           ├── 004_concepts.sql
│           ├── 005_sessions.sql
│           ├── 006_session_summaries.sql
│           └── 007_triggers.sql
├── scripts/
│   ├── bootstrap_db.py
│   ├── ingest_seed_corpus.py
│   └── eval_run.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   └── __init__.py
│   └── integration/
│       └── __init__.py
├── docker/
│   └── docker-compose.yml
├── docs/
│   ├── EXECUTION_PLAN.md
│   ├── STATUS.md
│   ├── architecture.md
│   ├── interview_talking_points.md
│   ├── decisions/
│   ├── phase_reports/
│   └── review_feedback/
│       ├── REVIEW_FEEDBACK.md
│       ├── REVIEW_FEEDBACK_V2.md
│       └── REVIEW_FEEDBACK_V3.md
├── data/
├── .env.example
├── .gitignore
├── pyproject.toml
├── Makefile
├── LICENSE
└── README.md
```

## Status

In development.
