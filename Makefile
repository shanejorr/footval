.PHONY: run probe generate sandbox-build sandbox check judge aggregate publish test lint

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

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .
