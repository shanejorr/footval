"""Stage 3 — consolidate per-instance artifacts into objective tables.

Reads each instance's response and check artifacts and writes the ``responses``
and ``checks`` tables under ``artifacts/tables/``. These are purely objective (no
judge opinion); the subjective judging lives in the pairwise stage. The published
priors visualization reads ``responses.json``.
"""

from __future__ import annotations

from typing import Any

from . import artifacts
from .config import Config

Row = dict[str, Any]

_RESPONSE_FIELDS = (
    "instance_id",
    "model",
    "sample_idx",
    "snapshot_id",
    "ts",
    "temperature",
    "seed",
    "stop_reason",
    "usage",
    "raw_text",
    "parsed_json",
    "parse_mode",
)
_CHECK_FIELDS = (
    "instance_id",
    "json_valid",
    "grid_complete",
    "intervals_ok",
    "family_ok",
    "params_reproduce_ok",
    "detail",
)


def run(cfg: Config) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    responses, checks = collect_rows(store)
    for name, rows in {"responses": responses, "checks": checks}.items():
        store.write_json(store.table_path(name), rows)
    print(f"tables: {len(responses)} responses, {len(checks)} checks")


def collect_rows(store: artifacts.Store) -> tuple[list[Row], list[Row]]:
    responses, checks = [], []
    for iid in store.instance_ids():
        resp = store.load_response(iid)
        if resp is not None:
            responses.append({k: resp.get(k) for k in _RESPONSE_FIELDS})
        ck = store.load_checks(iid)
        if ck is not None:
            checks.append({k: ck.get(k) for k in _CHECK_FIELDS})
    return responses, checks
