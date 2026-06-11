.PHONY: run probe generate sandbox-build sandbox check judge aggregate publish test lint \
	pairwise-estimate pairwise-submit pairwise-deepseek pairwise-status pairwise-collect pairwise-csv

DOCKER_GUARD = @docker info >/dev/null 2>&1 || { echo "ERROR: Docker daemon is not running. Start Docker Desktop, then retry."; exit 1; }

run:
	uv run python -m footbench run

probe:
	uv run python -m footbench probe

generate:
	uv run python -m footbench generate

sandbox-build:
	$(DOCKER_GUARD)
	docker build -t footbench-sandbox sandbox/

sandbox:
	$(DOCKER_GUARD)
	uv run python -m footbench sandbox

check:
	uv run python -m footbench check

judge:
	uv run python -m footbench judge

aggregate:
	uv run python -m footbench aggregate

publish:
	uv run python -m footbench publish

pairwise-estimate:
	uv run python -m footbench pairwise estimate

pairwise-submit:
	uv run python -m footbench pairwise submit

pairwise-deepseek:
	uv run python -m footbench pairwise deepseek

pairwise-status:
	uv run python -m footbench pairwise status

pairwise-collect:
	uv run python -m footbench pairwise collect

pairwise-csv:
	uv run python -m footbench pairwise csv

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .
