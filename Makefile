.PHONY: install fmt lint test test-unit demo ask ingest eval eval-sample eval-fresh ui ui-graph db-up db-init db-down db-reset

install:
	uv sync

fmt:
	uv run ruff format src tests scripts

lint:
	uv run ruff check src tests scripts

test:
	uv run pytest -q

test-unit:
	uv run pytest tests/unit -q

demo:
	uv run python -m src.cli --example

ask:
	uv run python -m src.cli $(ARGS)

db-up:
	docker compose --env-file .env -f docker/docker-compose.yml up -d --wait

db-init:
	uv run python scripts/bootstrap_db.py

db-down:
	docker compose -f docker/docker-compose.yml down

db-reset:
	docker compose -f docker/docker-compose.yml down -v
	docker compose -f docker/docker-compose.yml up -d
	sleep 3
	uv run python scripts/bootstrap_db.py

ingest:
	uv run python scripts/ingest_seed_corpus.py

eval: ## Run or resume the gold evaluation harness.
	uv run python scripts/eval_run.py

eval-sample: ## Run a 5-question pre-flight eval before the full eval.
	uv run python scripts/eval_run.py --question-ids q-005,q-013,q-019,q-025,q-029

eval-fresh: ## Delete the latest eval checkpoint and run from a clean CSV.
	rm -rf data/eval_outputs/latest
	$(MAKE) eval

ui:
	@echo 'The browser UI is not implemented. Use make demo or make ask ARGS="..."; see README.md.'
	@exit 2

ui-graph:
	@echo 'The graph UI is not implemented. See docs/architecture.md for the implemented flow.'
	@exit 2
