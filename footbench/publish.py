"""Stage 6 — final deliverables, read from artifacts only.

Writes to ``outputs/``:
  - scores.csv            long form: judge, candidate, soundness, priors, code
  - scores_by_judge.png   annotated heatmaps, one panel per criterion
  - priors_comparison.png 2x3 grid of prior pdfs re-plotted from the REPORTED
                          parameters via distributions.build_dist (never from
                          the candidates' own scripts)
"""

from __future__ import annotations

# ruff: noqa: E402  (matplotlib backend must be forced before pyplot import)
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from . import artifacts
from .checks import BUCKETS, TYPES
from .config import CRITERIA, Config
from .distributions import REQUIRED_PARAMS, build_dist, is_number, validate_params

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#393b79",
    "#637939",
]


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

    heat_path = cfg.outputs_dir / "scores_by_judge.png"
    plot_scores_by_judge(judgments, model_order, judge_order, heat_path)

    priors_path = cfg.outputs_dir / "priors_comparison.png"
    plot_priors_comparison(cfg, store, priors_path)
    print(
        f"outputs written to {cfg.outputs_dir}: scores.csv, scores_by_judge.png,"
        " priors_comparison.png"
    )


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


def plot_scores_by_judge(
    judgments: list[dict], model_order: list[str], judge_order: list[str], path: Path
) -> None:
    lookup = _score_lookup(judgments)
    n_m, n_j = len(model_order), len(judge_order)
    fig, axes = plt.subplots(
        1,
        len(CRITERIA),
        figsize=(2.1 * len(CRITERIA) * max(n_j, 3) / 4 + 4, 0.42 * n_m + 2.4),
        sharey=True,
        layout="constrained",
    )
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d9d9d9")
    im = None
    for ax, crit in zip(axes, CRITERIA, strict=True):
        mat = np.full((n_m, n_j), np.nan)
        for i, model in enumerate(model_order):
            for j, judge in enumerate(judge_order):
                v = lookup.get((judge, model, crit))
                if v is not None:
                    mat[i, j] = v
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=1, vmax=5, aspect="auto")
        ax.set_title(crit.capitalize(), fontsize=11)
        ax.set_xticks(range(n_j))
        ax.set_xticklabels(judge_order, rotation=35, ha="right", fontsize=8)
        for i in range(n_m):
            for j in range(n_j):
                v = mat[i, j]
                ax.text(
                    j,
                    i,
                    "–" if np.isnan(v) else f"{v:g}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        ax.set_xticks(np.arange(-0.5, n_j), minor=True)
        ax.set_yticks(np.arange(-0.5, n_m), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
    axes[0].set_yticks(range(n_m))
    axes[0].set_yticklabels(model_order, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.55, label="score (1–5)")
    fig.suptitle("Footbench — judge scores by candidate", fontsize=13)
    fig.text(
        0.01,
        -0.02,
        "Candidates ordered by composite score (best at top). – = judge returned no valid score.",
        fontsize=7,
        color="gray",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- prior comparison ---------------------------------------------------------


def compute_cell_xrange(ranges: list[tuple[float, float]]) -> tuple[float, float]:
    """Shared x-range for one grid cell, clamped so one heavy-tailed prior
    cannot flatten everyone else (median-width rule), then 5% padded."""
    los = [r[0] for r in ranges]
    his = [r[1] for r in ranges]
    med_lo = float(np.median(los))
    med_hi = float(np.median(his))
    med_w = med_hi - med_lo
    if med_w <= 0:
        med_w = (max(his) - min(los)) or 1.0
    lo = max(min(los), med_lo - 1.5 * med_w)
    hi = min(max(his), med_hi + 1.5 * med_w)
    if lo >= hi:
        lo, hi = min(los), max(his)
    if lo >= hi:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def compute_ymax(peaks: list[float]) -> float:
    """Shared y-limit: cap at 3x the median peak so one hyper-confident spike
    doesn't flatten the rest (it clips at the top instead)."""
    finite = [p for p in peaks if np.isfinite(p) and p > 0]
    if not finite:
        return 1.0
    return float(min(max(finite), 3 * float(np.median(finite)))) * 1.05


def collect_priors(
    cfg: Config, store: artifacts.Store
) -> tuple[dict[tuple[str, str], list[dict]], list[tuple[str, str]], list[str]]:
    """Per-cell plottable priors from each model's first sample, plus exclusions."""
    by_model: dict[str, list[str]] = {}
    for iid in store.instance_ids():
        by_model.setdefault(artifacts.model_from_instance_id(iid), []).append(iid)
    model_list = [m for m in cfg.candidate_models if m in by_model] + sorted(
        set(by_model) - set(cfg.candidate_models)
    )

    cells: dict[tuple[str, str], list[dict]] = {}
    exclusions: list[tuple[str, str]] = []
    for model in model_list:
        iid = sorted(by_model[model])[0]
        resp = store.load_response(iid) or {}
        parsed = resp.get("parsed_json")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("strategies"), list):
            exclusions.append((model, "response was not valid JSON"))
            continue
        skipped = 0
        for s in parsed["strategies"]:
            entry = _prior_entry(model, s)
            if entry is None:
                skipped += 1
                continue
            cells.setdefault(entry.pop("cell"), []).append(entry)
        if skipped:
            exclusions.append(
                (model, f"{skipped} strategie(s) skipped (invalid cell/family/parameters)")
            )
    return cells, exclusions, model_list


def _prior_entry(model: str, s: Any) -> dict | None:
    if not isinstance(s, dict):
        return None
    bucket, stype = s.get("bucket"), s.get("type")
    if bucket not in BUCKETS or stype not in TYPES:
        return None
    prior = s.get("prior") if isinstance(s.get("prior"), dict) else {}
    fam_obj = prior.get("distribution_family")
    family = fam_obj.get("name") if isinstance(fam_obj, dict) else None
    params = prior.get("parameters")
    if family is None or validate_params(family, params) is not None:
        return None
    numeric = {k: float(params[k]) for k in REQUIRED_PARAMS[family]}
    bs = prior.get("belief_summaries")
    bs = bs if isinstance(bs, dict) else {}
    low, high, mpv = (
        bs.get("interval_95_low"),
        bs.get("interval_95_high"),
        bs.get("most_plausible_value"),
    )
    return {
        "cell": (bucket, stype),
        "model": model,
        "dist": build_dist(family, numeric),
        "family": family,
        "low": float(low) if is_number(low) else None,
        "high": float(high) if is_number(high) else None,
        "mpv": float(mpv) if is_number(mpv) else None,
    }


def plot_priors_comparison(cfg: Config, store: artifacts.Store, path: Path) -> None:
    cells, exclusions, model_list = collect_priors(cfg, store)
    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(model_list)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for r, bucket in enumerate(BUCKETS):
        for c, stype in enumerate(TYPES):
            ax = axes[r][c]
            _draw_cell(ax, cells.get((bucket, stype), []), colors)
            ax.set_title(f"{bucket} · {stype.replace('_', ' ')}", fontsize=11)
            if r == 1:
                ax.set_xlabel("Change in own scoring margin (points)")

    handles = [Line2D([0], [0], color=colors[m], lw=2, label=m) for m in model_list]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        "Footbench — prior comparison, re-plotted from each model's reported parameters",
        fontsize=13,
    )
    if exclusions:
        note = "Not shown — " + "; ".join(f"{m}: {reason}" for m, reason in exclusions)
        fig.text(0.01, -0.01, note, fontsize=7, color="gray", va="top")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_cell(ax, entries: list[dict], colors: dict[str, str]) -> None:
    if not entries:
        ax.text(0.5, 0.5, "no valid priors", transform=ax.transAxes, ha="center", color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ranges = []
    for e in entries:
        lo = float(e["dist"].ppf(0.025))
        hi = float(e["dist"].ppf(0.975))
        if e["low"] is not None:
            lo = min(lo, e["low"])
        if e["high"] is not None:
            hi = max(hi, e["high"])
        ranges.append((lo, hi))
    x_lo, x_hi = compute_cell_xrange(ranges)
    x = np.linspace(x_lo, x_hi, 400)
    peaks = []
    for e in entries:
        y = e["dist"].pdf(x)
        peaks.append(float(np.nanmax(y)) if y.size else 0.0)
        ax.plot(x, y, color=colors[e["model"]], lw=1.6, alpha=0.9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, compute_ymax(peaks))
    if x_lo < 0 < x_hi:
        ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.7)
    for e in entries:
        if e["mpv"] is not None and x_lo <= e["mpv"] <= x_hi:
            ax.plot([e["mpv"]], [0], marker="|", ms=10, color=colors[e["model"]], clip_on=False)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
