"""Stage 3 — consolidate per-instance artifacts into objective tables.

Reads each instance's response and check artifacts and writes the ``responses``
and ``checks`` tables under ``artifacts/tables/``, plus the per-candidate
generation-cost summary ``outputs/data/candidate_costs.csv``. These are purely
objective (no judge opinion); the subjective judging lives in the pairwise
stage. The published priors visualization reads ``responses.json``.
"""

from __future__ import annotations

import csv
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

COSTS_FILENAME = "candidate_costs.csv"


def billable_output_tokens(usage: dict[str, Any] | None) -> int | None:
    """Provider-billed output tokens for one response.

    Visible output plus thinking where the provider reports it separately:
    Gemini bills ``thoughts_tokens`` at the output rate; Anthropic, OpenAI,
    and GLM already fold thinking into ``output_tokens``.
    """
    if not usage or usage.get("output_tokens") is None:
        return None
    return int(usage["output_tokens"]) + int(usage.get("thoughts_tokens") or 0)


def output_cost_usd(usage: dict[str, Any] | None, rate_usd_per_mtok: float | None) -> float | None:
    """Cost of one response's billable output at the model's list price."""
    tokens = billable_output_tokens(usage)
    if tokens is None or rate_usd_per_mtok is None:
        return None
    return tokens * rate_usd_per_mtok / 1e6


def run(cfg: Config) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    responses, checks = collect_rows(store)
    for row in responses:
        rate = _rate(cfg, row.get("model"))
        row["billable_output_tokens"] = billable_output_tokens(row.get("usage"))
        row["output_cost_usd"] = output_cost_usd(row.get("usage"), rate)
    for name, rows in {"responses": responses, "checks": checks}.items():
        store.write_json(store.table_path(name), rows)
    costs = candidate_costs(cfg, responses)
    _write_costs_csv(cfg, costs)
    print(f"tables: {len(responses)} responses, {len(checks)} checks")
    print("candidate output costs:")
    for row in costs:
        cost = f"${row['output_cost_usd']:.4f}" if row["output_cost_usd"] is not None else "n/a"
        print(f"  {row['model']:<26} {row['billable_output_tokens'] or 0:>7} tok  {cost}")


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


def candidate_costs(cfg: Config, responses: list[Row]) -> list[Row]:
    """Per-candidate totals: billable output tokens x the model's output price.

    A model with no configured ``output_usd_per_mtok`` (e.g. one retired from
    the roster whose artifacts are still on disk) gets a blank cost, never a
    silent zero.
    """
    by_model: dict[str, dict[str, Any]] = {}
    for row in responses:
        model = row.get("model")
        agg = by_model.setdefault(model, {"samples": 0, "tokens": 0, "tokens_known": True})
        agg["samples"] += 1
        tokens = billable_output_tokens(row.get("usage"))
        if tokens is None:
            agg["tokens_known"] = False
        else:
            agg["tokens"] += tokens
    out: list[Row] = []
    for model in sorted(by_model):
        agg = by_model[model]
        rate = _rate(cfg, model)
        tokens = agg["tokens"] if agg["tokens_known"] else None
        cost = tokens * rate / 1e6 if (tokens is not None and rate is not None) else None
        out.append(
            {
                "model": model,
                "samples": agg["samples"],
                "billable_output_tokens": tokens,
                "output_usd_per_mtok": rate,
                "output_cost_usd": round(cost, 6) if cost is not None else None,
            }
        )
    return out


def _rate(cfg: Config, model: str | None) -> float | None:
    mcfg = cfg.models.get(model) if model else None
    return mcfg.output_usd_per_mtok if mcfg else None


def _write_costs_csv(cfg: Config, costs: list[Row]) -> None:
    cfg.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.outputs_data_dir / COSTS_FILENAME
    fields = (
        "model",
        "samples",
        "billable_output_tokens",
        "output_usd_per_mtok",
        "output_cost_usd",
    )
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(fields)
        for row in costs:
            writer.writerow(["" if row[f] is None else row[f] for f in fields])
    print(f"wrote {path}")
