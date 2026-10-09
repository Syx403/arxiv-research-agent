# ARA v2 — arXiv research agent

ARA clarifies a research need, finds papers on arXiv, reads the chosen papers and answers with
sentence-level citations that are checked against the source. It is a personal project that shows,
with measured evidence, how an agent system is built and evaluated: LangGraph orchestration, RAG,
an LLM layer with cost and cache control, memory and PostgreSQL. It is not a production service.

**Status:** v2 milestones M0–M5 are built (M5b: the local web app); the architecture review and
the evaluation rounds come next. The full README follows the review.

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

## Run the app

Requirements as above, plus Node (only to build the UI).

```sh
make start      # database, migrations, UI build, server
```

Then open http://127.0.0.1:8000 (bound to this machine only, one user, no login). It is a chat
with stored conversations: under each reply a fold shows how the turn ran, and a side panel shows
that turn's path through the LangGraph graphs (drawn from the compiled code), its model calls from
the cost ledger, the cited sentences marked in their papers, and the papers. Evaluation rounds and
memory are pages of their own. During UI
work, `uv run ara serve` with `make ui-dev` gives hot reload on http://127.0.0.1:5173.

`make test-live` calls the real models and is billed; every call is recorded in the `llm_calls`
ledger and traced in LangSmith.

## License

MIT
