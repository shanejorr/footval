"""Stage 5 — scores, rankings, interrater reliability, and bias diagnostic.

Reads every per-instance artifact and writes the spec section-8 tables under
``artifacts/tables/``. Scoring semantics are locked: integer 1-5 judge scores,
arithmetic means, weighted composite, min-method tie ranks.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import artifacts, stats
from .config import CRITERIA, Config

COMPOSITE = "composite"

_RESPONSE_FIELDS = (
    "instance_id",
    "model",
    "sample_idx",
    "snapshot_id",
    "ts",
    "temperature",
    "seed",
    "stop_reason",
    "raw_text",
    "parsed_json",
    "parse_mode",
)
_EXEC_FIELDS = (
    "instance_id",
    "ran_ok",
    "exit_code",
    "timed_out",
    "n_figures",
    "stderr",
    "figure_paths",
    "duration_s",
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

Row = dict[str, Any]


def run(cfg: Config) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    responses, execs, checks, judgments = collect_rows(store)
    tables = build_tables(cfg, judgments)
    tables["responses"] = responses
    tables["exec_results"] = execs
    tables["checks"] = checks
    tables["judgments"] = judgments
    for name, rows in tables.items():
        store.write_json(store.table_path(name), rows)
    _print_summary(tables)


def collect_rows(store: artifacts.Store) -> tuple[list[Row], list[Row], list[Row], list[Row]]:
    responses, execs, checks, judgments = [], [], [], []
    for iid in store.instance_ids():
        resp = store.load_response(iid)
        if resp is not None:
            responses.append({k: resp.get(k) for k in _RESPONSE_FIELDS})
        ex = store.load_exec(iid)
        if ex is not None:
            execs.append({k: ex.get(k) for k in _EXEC_FIELDS})
        ck = store.load_checks(iid)
        if ck is not None:
            checks.append({k: ck.get(k) for k in _CHECK_FIELDS})
        model = (resp or {}).get("model") or artifacts.model_from_instance_id(iid)
        for path in store.judgment_files(iid):
            j = store.read_json(path) or {}
            scores = j.get("scores") or {}
            for crit in CRITERIA:
                entry = scores.get(crit) if isinstance(scores.get(crit), dict) else None
                judgments.append(
                    {
                        "instance_id": iid,
                        "model": model,
                        "judge": j.get("judge"),
                        "judge_family": j.get("judge_family"),
                        "is_self": bool(j.get("is_self")),
                        "criterion": crit,
                        "score": entry.get("score") if entry else None,
                        "justification": entry.get("justification") if entry else None,
                    }
                )
    return responses, execs, checks, judgments


def build_tables(cfg: Config, judgment_rows: list[Row]) -> dict[str, list[Row]]:
    """Pure aggregation core: judgments -> model_scores / reliability / bias."""
    weights = dict(zip(CRITERIA, cfg.criteria_weights, strict=True))
    models = sorted({r["model"] for r in judgment_rows})
    judges = [j for j in cfg.judge_models]

    scored = [r for r in judgment_rows if r["score"] is not None]
    means_all = _with_composite(_means(scored), models, weights)
    means_self_ex = _with_composite(
        _means([r for r in scored if not r["is_self"]]), models, weights
    )
    n_by_mc: dict[tuple[str, str], int] = {}
    for r in scored:
        key = (r["model"], r["criterion"])
        n_by_mc[key] = n_by_mc.get(key, 0) + 1

    model_scores: list[Row] = []
    for crit in [*CRITERIA, COMPOSITE]:
        ranks_all = _ranks(means_all, models, crit)
        ranks_self = _ranks(means_self_ex, models, crit)
        for model in models:
            mean = means_all.get((model, crit))
            mean_se = means_self_ex.get((model, crit))
            model_scores.append(
                {
                    "model": model,
                    "criterion": crit,
                    "mean_score": _round(mean),
                    "rank": ranks_all.get(model),
                    "mean_score_self_excluded": _round(mean_se),
                    "rank_self_excluded": ranks_self.get(model),
                    "n_judgments": n_by_mc.get((model, crit), 0),
                }
            )
    order = {c: i for i, c in enumerate([*CRITERIA, COMPOSITE])}
    model_scores.sort(
        key=lambda r: (order[r["criterion"]], r["rank"] is None, r["rank"], r["model"])
    )

    reliability = _reliability_rows(judgment_rows, judges, models)
    bias = _bias_rows(cfg, scored, judges)
    return {"model_scores": model_scores, "reliability": reliability, "bias": bias}


def _means(rows: list[Row]) -> dict[tuple[str, str], float]:
    acc: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        acc.setdefault((r["model"], r["criterion"]), []).append(float(r["score"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _with_composite(
    means: dict[tuple[str, str], float], models: list[str], weights: dict[str, float]
) -> dict[tuple[str, str], float]:
    out = dict(means)
    total_w = sum(weights.values())
    for model in models:
        vals = [means.get((model, c)) for c in CRITERIA]
        if all(v is not None for v in vals):
            out[(model, COMPOSITE)] = (
                sum(weights[c] * v for c, v in zip(CRITERIA, vals, strict=True)) / total_w
            )
    return out


def _ranks(
    means: dict[tuple[str, str], float], models: list[str], criterion: str
) -> dict[str, int]:
    vals = {m: means[(m, criterion)] for m in models if (m, criterion) in means}
    return {m: 1 + sum(1 for v in vals.values() if v > vals[m]) for m in vals}


def _reliability_rows(judgment_rows: list[Row], judges: list[str], models: list[str]) -> list[Row]:
    iids = sorted({r["instance_id"] for r in judgment_rows})
    rows: list[Row] = []
    for crit in CRITERIA:
        matrix = np.full((len(judges), len(iids)), np.nan)
        per_judge_model: dict[tuple[str, str], list[float]] = {}
        for r in judgment_rows:
            if r["criterion"] != crit or r["score"] is None or r["judge"] not in judges:
                continue
            matrix[judges.index(r["judge"]), iids.index(r["instance_id"])] = r["score"]
            per_judge_model.setdefault((r["judge"], r["model"]), []).append(float(r["score"]))

        units = matrix.T
        complete = units[~np.isnan(units).any(axis=1)] if units.size else units

        jm = np.full((len(judges), len(models)), np.nan)
        for (judge, model), vals in per_judge_model.items():
            if model in models:
                jm[judges.index(judge), models.index(model)] = sum(vals) / len(vals)

        alpha = stats.krippendorff_alpha_ordinal(matrix)
        icc = stats.icc2k(complete)
        rows.append(
            {
                "criterion": crit,
                "krippendorff_alpha": _round(alpha),
                "alpha_band": stats.alpha_band(alpha),
                "icc2k": _round(icc),
                "icc_band": stats.icc_band(icc),
                "icc_n_complete": int(complete.shape[0]) if complete.size else 0,
                "kendall_w": _round(stats.kendalls_w(jm)),
                "pairwise_spearman": stats.pairwise_spearman(jm, judges),
            }
        )
    return rows


def _bias_rows(cfg: Config, scored: list[Row], judges: list[str]) -> list[Row]:
    family_of = {name: m.family for name, m in cfg.models.items()}
    rows: list[Row] = []
    for judge in judges:
        judge_family = family_of.get(judge)
        for crit in CRITERIA:
            own = [
                float(r["score"])
                for r in scored
                if r["judge"] == judge
                and r["criterion"] == crit
                and family_of.get(r["model"]) == judge_family
            ]
            other = [
                float(r["score"])
                for r in scored
                if r["judge"] == judge
                and r["criterion"] == crit
                and family_of.get(r["model"]) != judge_family
            ]
            own_mean = sum(own) / len(own) if own else None
            other_mean = sum(other) / len(other) if other else None
            gap = own_mean - other_mean if own_mean is not None and other_mean is not None else None
            rows.append(
                {
                    "judge": judge,
                    "criterion": crit,
                    "own_family_mean": _round(own_mean),
                    "other_mean": _round(other_mean),
                    "gap": _round(gap),
                    "n_own": len(own),
                    "n_other": len(other),
                }
            )
    return rows


def _round(v: float | None) -> float | None:
    return None if v is None else round(float(v), 4)


def _print_summary(tables: dict[str, list[Row]]) -> None:
    comp = [r for r in tables["model_scores"] if r["criterion"] == COMPOSITE]
    print("composite leaderboard:")
    for r in comp:
        if r["mean_score"] is not None:
            print(f"  {r['rank']:>2}. {r['model']:<28} {r['mean_score']:.3f}")
    for r in tables["reliability"]:
        print(
            f"reliability[{r['criterion']}]: alpha={r['krippendorff_alpha']} ({r['alpha_band']}),"
            f" ICC(2,k)={r['icc2k']} ({r['icc_band']}), W={r['kendall_w']}"
        )
