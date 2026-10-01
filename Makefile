.PHONY: install fmt lint test ingest ui db-up db-down db-reset

install:
	uv sync

fmt:
	uv run ruff format src tests scripts

lint:
	uv run ruff check src tests scripts

test:
	uv run pytest -q

db-up:
	docker compose -f docker/docker-compose.yml up -d

db-down:
	docker compose -f docker/docker-compose.yml down

db-reset:
	docker compose -f docker/docker-compose.yml down -v
	docker compose -f docker/docker-compose.yml up -d
	sleep 3
	uv run python scripts/bootstrap_db.py

ingest:
	uv run python scripts/ingest_seed_corpus.py

ui:
	uv run uvicorn src.ui.app:app --host 127.0.0.1 --port 8000
