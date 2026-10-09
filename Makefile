.PHONY: install db-up db-down migrate fmt lint typecheck test check test-live ui ui-check ui-dev start

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

# The UI (React + Vite + TypeScript, M5b). `ui` builds ui/dist, which the server serves at /.
ui/node_modules: ui/package-lock.json
	cd ui && npm ci
	touch ui/node_modules

ui: ui/node_modules
	cd ui && npm run build

ui-check: ui/node_modules
	cd ui && npm run typecheck

# Development: Vite on :5173 with hot reload, API calls proxied to `uv run ara serve` on :8000.
ui-dev: ui/node_modules
	cd ui && npm run dev

# One command: database, migrations, UI build, server. Then open http://127.0.0.1:8000.
start: db-up ui
	uv run ara serve
