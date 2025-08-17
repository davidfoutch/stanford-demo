.PHONY: bootstrap dev api test lint

bootstrap:
	bash scripts/bootstrap.sh

dev:
	ENV_FILE=.env uvicorn src.api.server:app --reload --port 8000

api:
	ENV_FILE=.env uvicorn src.api.server:app --port 8000

test:
	pytest -q

lint:
	ruff check src
