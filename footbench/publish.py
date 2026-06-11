"""Stage 6 — final deliverables, read from artifacts only.

Writes to ``outputs/``:
  - scores.csv            long form: judge, candidate, soundness, priors, code
"""

from __future__ import annotations

import csv
from pathlib import Path

from . import artifacts
from .config import CRITERIA, Config


def run(cfg: Config) -> None:
    store = artifacts.Store(cfg.artifacts_dir)
    judgments = store.read_json(store.table_path("judgments")) or []
    model_scores = store.read_json(store.table_path("model_scores")) or []
    if not judgments:
        raise SystemExit("no judgments table found — run `make aggregate` first")
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)

    model_order = composite_model_order(model_scores, cfg.candidate_models)
    judge_order = list(cfg.judge_models)

    csv_path = cfg.outputs_dir / "scores.csv"
    write_scores_csv(judgments, model_order, judge_order, csv_path)

    print(f"outputs written to {cfg.outputs_dir}: scores.csv")


def composite_model_order(model_scores: list[dict], candidates: tuple[str, ...]) -> list[str]:
    """Models sorted by composite rank; anything unranked appended in config order."""
    ranked = [
        r["model"]
        for r in sorted(
            (r for r in model_scores if r["criterion"] == "composite" and r["rank"] is not None),
            key=lambda r: (r["rank"], r["model"]),
        )
    ]
    return ranked + [m for m in candidates if m not in ranked]


def _score_lookup(judgment_rows: list[dict]) -> dict[tuple[str, str, str], float]:
    """(judge, model, criterion) -> mean score across samples."""
    acc: dict[tuple[str, str, str], list[float]] = {}
    for r in judgment_rows:
        if r.get("score") is None:
            continue
        acc.setdefault((r["judge"], r["model"], r["criterion"]), []).append(float(r["score"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _fmt_score(v: float | None) -> str:
    if v is None:
        return ""
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.3f}"


def write_scores_csv(
    judgments: list[dict], model_order: list[str], judge_order: list[str], path: Path
) -> None:
    lookup = _score_lookup(judgments)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["judge", "candidate", *CRITERIA])
        for judge in judge_order:
            for model in model_order:
                writer.writerow(
                    [judge, model]
                    + [_fmt_score(lookup.get((judge, model, crit))) for crit in CRITERIA]
                )
