# ARA v2 — arXiv research agent

ARA clarifies a research need, finds papers on arXiv, reads the chosen papers and answers with
sentence-level citations that are checked against the source. It is a personal project that shows,
with measured evidence, how an agent system is built and evaluated: LangGraph orchestration, RAG,
an LLM layer with cost and cache control, memory and PostgreSQL. It is not a production service.

**Status:** v2 is being rebuilt; milestone M0 (foundation) is done. The full README arrives with
the UI in M5.

- Design: [docs/v2/DESIGN.md](docs/v2/DESIGN.md)
- Decisions and their reasons: [docs/v2/DECISIONS.md](docs/v2/DECISIONS.md)
- Progress and spend: [docs/v2/PROGRESS.md](docs/v2/PROGRESS.md)
- The first version (v1) is kept at tag `baseline-2026-10-07`.

## Develop

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Python 3.13.

```sh
test -f .env || cp .env.example .env   # then fill in the API keys
make install
make db-up      # PostgreSQL 18 + pgvector + pg_search on 127.0.0.1:5434
make migrate
make check      # ruff, mypy, and unit tests against the real database
```

`make test-live` calls the real models and is billed; every call is recorded in the `llm_calls`
ledger and traced in LangSmith.

## License

MIT
