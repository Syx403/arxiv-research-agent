.PHONY: install db-up db-down migrate fmt lint typecheck test check test-live

COMPOSE = docker compose -f docker/compose.yml

install:
	uv sync --locked

# PostgreSQL 18 + pgvector + pg_search on 127.0.0.1:5434 (databases `ara` and `ara_test`).
db-up:
	$(COMPOSE) up -d --wait

db-down:
	$(COMPOSE) down

migrate:
	uv run python -m ara.db.migrate

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

# Unit tests: deterministic code against the real `ara_test` database (needs `make db-up`).
test:
	uv run pytest

check: lint typecheck test

# Billable: real model calls, recorded in the ledger. Needs approval (CLAUDE.md rule 1).
test-live:
	uv run pytest -m live -v
