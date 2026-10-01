.PHONY: install fmt lint test test-unit start demo ask ingest ui db-up db-init db-down db-reset

PORT ?= 8000
COMPOSE = docker compose $(if $(wildcard .env),--env-file .env) -f docker/docker-compose.yml

install:
	uv sync --locked

fmt:
	uv run ruff format src tests scripts

lint:
	uv run ruff check src tests scripts

test:
	uv run pytest -q

test-unit:
	uv run pytest tests/unit -q

# Local browser demo: preserve the database volume and run migrations before serving.
start:
	$(MAKE) install
	$(MAKE) db-up
	$(MAKE) db-init
	$(MAKE) ui

# The remote repository's credential-free example remains explicitly offline.
demo:
	uv run python -m src.cli --example

ask:
	uv run python -m src.cli $(ARGS)

db-up:
	$(COMPOSE) up -d --wait --wait-timeout 60

db-init:
	uv run python -m scripts.bootstrap_db

db-down:
	$(COMPOSE) down

db-reset:
	$(COMPOSE) down -v
	$(MAKE) db-up
	$(MAKE) db-init

ingest:
	uv run python scripts/ingest_seed_corpus.py

ui:
	uv run uvicorn src.ui.app:app --host 127.0.0.1 --port $(PORT)
