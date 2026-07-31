.PHONY: run probe probe-full generate check tables test lint \
	pairwise-estimate pairwise-submit pairwise-sync pairwise-status pairwise-collect pairwise-csv \
	bradley-terry

run:
	uv run python -m footval run

probe:
	uv run python -m footval probe

probe-full:
	uv run python -m footval probe --full

generate:
	uv run python -m footval generate

check:
	uv run python -m footval check

tables:
	uv run python -m footval tables

pairwise-estimate:
	uv run python -m footval pairwise estimate

pairwise-submit:
	uv run python -m footval pairwise submit

pairwise-sync:
	uv run python -m footval pairwise sync

pairwise-status:
	uv run python -m footval pairwise status

pairwise-collect:
	uv run python -m footval pairwise collect

pairwise-csv:
	uv run python -m footval pairwise csv

bradley-terry:
	uv run python -m footval bradley-terry

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check .
