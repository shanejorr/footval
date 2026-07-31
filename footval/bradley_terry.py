"""Hierarchical Bayesian Bradley-Terry aggregation of the pairwise verdicts.

Reads the long-form `outputs/data/pairwise_results.csv` produced by `pairwise csv`
and fits, per track (`soundness`, `priors`, `overall`), a rater-aware Bradley-Terry
model in PyMC:

    eta = kappa_j * (beta_a - beta_b) + gamma_j + delta_j * self_sign
    won_a ~ Bernoulli(logit_p = eta)

where `beta` is each model's latent strength and the judges are modeled as raters
with their own discrimination (`kappa`), position/slot-A bias (`gamma`), and
self-family preference (`delta`), all partially pooled. The published payoff over
raw win% is principled uncertainty: every strength and the panel-level biases
(`mu_gamma`, `mu_delta`) come with HDIs.

This stage is purely additive — it consumes the same CSV the notebook charts and
touches neither the judge prompt nor any prior-run artifact. PyMC/ArviZ are
imported lazily so the pure data-prep helpers stay importable (and testable)
without the sampling stack.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config

# Tracks fit independently: each per-criterion plus a pooled "overall".
TRACKS = ("soundness", "priors", "overall")

# Sampler settings. Hierarchical pieces are non-centered, but target_accept is
# kept high to keep divergences at zero on this small panel.
DRAWS = 2000
TUNE = 2000
CHAINS = 4
TARGET_ACCEPT = 0.95
HDI_PROB = 0.94

RESULTS_FILENAME = "pairwise_results.csv"
RANKINGS_FILENAME = "bradley_terry_rankings.csv"
JUDGE_EFFECTS_FILENAME = "bradley_terry_judge_effects.csv"
SUMMARY_FILENAME = "bradley_terry.json"

# Full posterior traces are saved here (under artifacts/, gitignored) so the blog and
# diagnostics notebooks can recompute any HDI / posterior-predictive quantity and run
# ArviZ convergence checks without re-fitting.
TRACE_DIRNAME = "bradley_terry"


# --- data prep (pure; no PyMC/ArviZ) ------------------------------------------------


@dataclass(frozen=True)
class Design:
    """Integer-indexed comparison arrays for one track, ready for PyMC."""

    track: str
    models: list[str]
    judges: list[str]
    a_idx: np.ndarray  # slot-A model index per observation
    b_idx: np.ndarray  # slot-B model index per observation
    j_idx: np.ndarray  # judge index per observation
    self_sign: np.ndarray  # 1[fam(a)==fam(j)] - 1[fam(b)==fam(j)] in {-1, 0, +1}
    won_a: np.ndarray  # 1 if the slot-A response won, else 0

    @property
    def n_obs(self) -> int:
        return int(self.won_a.shape[0])


def family_map(cfg: Config) -> dict[str, str]:
    """Model/judge name -> family, for the self-preference indicator."""
    return {name: m.family for name, m in cfg.models.items()}


def read_results(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found — run `make pairwise-csv` first")
    return pl.read_csv(path)


def trace_path(cfg: Config, track: str) -> Path:
    """Where the persisted InferenceData for a track lives (under artifacts/)."""
    return cfg.artifacts_dir / TRACE_DIRNAME / f"{track}.nc"


def self_sign(family_of: dict[str, str], model_a: str, model_b: str, judge: str) -> float:
    """Signed same-family indicator: +1 if only A shares the judge's family,
    -1 if only B does, 0 if both or neither do."""
    jf = family_of.get(judge)
    a = 1.0 if family_of.get(model_a) == jf else 0.0
    b = 1.0 if family_of.get(model_b) == jf else 0.0
    return a - b


def _track_frame(df: pl.DataFrame, track: str) -> pl.DataFrame:
    rows = df if track == "overall" else df.filter(pl.col("criterion") == track)
    # Drop comparisons where the judge returned no valid verdict (blank winner).
    return rows.filter(pl.col("winner").is_not_null() & (pl.col("winner").str.strip_chars() != ""))


def build_design(df: pl.DataFrame, track: str, family_of: dict[str, str]) -> Design:
    rows = _track_frame(df, track)
    model_a = rows["model_a"].to_list()
    model_b = rows["model_b"].to_list()
    judge = rows["judge"].to_list()
    winner = rows["winner"].to_list()

    for w, ma, mb in zip(winner, model_a, model_b, strict=True):
        if w not in (ma, mb):
            raise ValueError(f"winner {w!r} is neither model_a {ma!r} nor model_b {mb!r}")

    models = sorted(set(model_a) | set(model_b))
    judges = sorted(set(judge))
    m_index = {m: i for i, m in enumerate(models)}
    j_index = {j: i for i, j in enumerate(judges)}

    return Design(
        track=track,
        models=models,
        judges=judges,
        a_idx=np.array([m_index[m] for m in model_a], dtype="int64"),
        b_idx=np.array([m_index[m] for m in model_b], dtype="int64"),
        j_idx=np.array([j_index[j] for j in judge], dtype="int64"),
        self_sign=np.array(
            [
                self_sign(family_of, ma, mb, jg)
                for ma, mb, jg in zip(model_a, model_b, judge, strict=True)
            ],
            dtype="float64",
        ),
        won_a=np.array(
            [1 if w == ma else 0 for w, ma in zip(winner, model_a, strict=True)], dtype="int64"
        ),
    )


# --- model fit & posterior summary (lazy PyMC/ArviZ) --------------------------------


def _fit(design: Design, seed: int):
    import pymc as pm

    coords = {"model": design.models, "judge": design.judges, "obs": np.arange(design.n_obs)}
    with pm.Model(coords=coords):
        # Strength: sum-to-zero fixes the location; sigma=1 fixes the scale (resolving
        # the kappa*beta redundancy) and weakly regularizes against separation.
        beta = pm.ZeroSumNormal("beta", sigma=1.0, dims="model")

        # Judge discrimination kappa>0, centered at 1 (non-centered log scale).
        sigma_kappa = pm.HalfNormal("sigma_kappa", 0.5)
        z_kappa = pm.Normal("z_kappa", 0.0, 1.0, dims="judge")
        kappa = pm.Deterministic("kappa", pm.math.exp(sigma_kappa * z_kappa), dims="judge")

        # Per-judge position (slot-A) bias, partially pooled; mu_gamma is panel-wide.
        mu_gamma = pm.Normal("mu_gamma", 0.0, 0.5)
        sigma_gamma = pm.HalfNormal("sigma_gamma", 0.5)
        z_gamma = pm.Normal("z_gamma", 0.0, 1.0, dims="judge")
        gamma = pm.Deterministic("gamma", mu_gamma + sigma_gamma * z_gamma, dims="judge")

        # Per-judge self-family preference, partially pooled; mu_delta is panel-wide.
        mu_delta = pm.Normal("mu_delta", 0.0, 0.5)
        sigma_delta = pm.HalfNormal("sigma_delta", 0.5)
        z_delta = pm.Normal("z_delta", 0.0, 1.0, dims="judge")
        delta = pm.Deterministic("delta", mu_delta + sigma_delta * z_delta, dims="judge")

        eta = (
            kappa[design.j_idx] * (beta[design.a_idx] - beta[design.b_idx])
            + gamma[design.j_idx]
            + delta[design.j_idx] * design.self_sign
        )
        pm.Bernoulli("won_a", logit_p=eta, observed=design.won_a, dims="obs")

        idata = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            target_accept=TARGET_ACCEPT,
            random_seed=seed,
            progressbar=False,
        )
    return idata


def _save_trace(path: Path, idata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and atomically swap into place. Avoids HDF5 file-lock
    # contention (errno 35) when a notebook kernel still holds the trace open for
    # reading, and never leaves a truncated 0-byte file if the write fails.
    tmp = path.with_name(path.name + ".tmp")
    idata.to_netcdf(str(tmp), engine="h5netcdf")
    os.replace(tmp, path)


def _summarize(design: Design, idata) -> dict:
    import arviz as az

    post = idata.posterior
    beta_mean = post["beta"].mean(dim=("chain", "draw")).to_numpy()
    beta_sd = post["beta"].std(dim=("chain", "draw")).to_numpy()
    beta_hdi = az.hdi(idata, var_names=["beta"], prob=HDI_PROB)["beta"].to_numpy()
    wpf_mean, wpf_hdi = _win_prob_vs_field(post["beta"].stack(s=("chain", "draw")).to_numpy())

    # rank 1 = highest posterior-mean strength
    ranks = (-beta_mean).argsort().argsort() + 1
    rankings = [
        {
            "model": m,
            "strength_mean": float(beta_mean[i]),
            "strength_sd": float(beta_sd[i]),
            "hdi_low": float(beta_hdi[i, 0]),
            "hdi_high": float(beta_hdi[i, 1]),
            "rank": int(ranks[i]),
            "win_prob_vs_field": float(wpf_mean[i]),
            "win_prob_hdi_low": float(wpf_hdi[i, 0]),
            "win_prob_hdi_high": float(wpf_hdi[i, 1]),
        }
        for i, m in enumerate(design.models)
    ]
    rankings.sort(key=lambda r: r["rank"])

    kappa_m, kappa_h = _effect_summary(idata, "kappa")
    gamma_m, gamma_h = _effect_summary(idata, "gamma")
    delta_m, delta_h = _effect_summary(idata, "delta")
    judge_effects = [
        {
            "judge": j,
            "kappa_mean": float(kappa_m[i]),
            "kappa_hdi_low": float(kappa_h[i, 0]),
            "kappa_hdi_high": float(kappa_h[i, 1]),
            "gamma_mean": float(gamma_m[i]),
            "gamma_hdi_low": float(gamma_h[i, 0]),
            "gamma_hdi_high": float(gamma_h[i, 1]),
            "delta_mean": float(delta_m[i]),
            "delta_hdi_low": float(delta_h[i, 0]),
            "delta_hdi_high": float(delta_h[i, 1]),
        }
        for i, j in enumerate(design.judges)
    ]

    summ = az.summary(idata, ci_prob=HDI_PROB)
    diagnostics = {
        "max_r_hat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "n_divergences": int(idata.sample_stats["diverging"].to_numpy().sum()),
    }
    return {
        "n_obs": design.n_obs,
        "models": design.models,
        "judges": design.judges,
        "rankings": rankings,
        "judge_effects": judge_effects,
        "population": {
            "mu_gamma": _scalar_summary(idata, "mu_gamma"),
            "mu_delta": _scalar_summary(idata, "mu_delta"),
        },
        "diagnostics": diagnostics,
    }


def _win_prob_vs_field(beta_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected win prob of each model against a uniformly random other model under
    a neutral judge (kappa=1, gamma=delta=0): mean_{n!=m} sigmoid(beta_m - beta_n).

    `beta_samples` has shape (model, sample); returns posterior mean and HDI per model.
    """
    import arviz as az

    samples = beta_samples.T  # (sample, model)
    n_models = samples.shape[1]
    diff = samples[:, :, None] - samples[:, None, :]
    probs = 1.0 / (1.0 + np.exp(-diff))
    off_diag = ~np.eye(n_models, dtype=bool)
    wpf = (probs * off_diag).sum(axis=2) / (n_models - 1)  # (sample, model)
    hdi = np.stack([az.hdi(wpf[:, m], prob=HDI_PROB) for m in range(n_models)], axis=0)
    return wpf.mean(axis=0), hdi


def _effect_summary(idata, name: str) -> tuple[np.ndarray, np.ndarray]:
    import arviz as az

    mean = idata.posterior[name].mean(dim=("chain", "draw")).to_numpy()
    hdi = az.hdi(idata, var_names=[name], prob=HDI_PROB)[name].to_numpy()
    return mean, hdi


def _scalar_summary(idata, name: str) -> dict:
    import arviz as az

    mean = float(idata.posterior[name].mean(dim=("chain", "draw")).to_numpy())
    hdi = az.hdi(idata, var_names=[name], prob=HDI_PROB)[name].to_numpy()
    return {"mean": mean, "hdi_low": float(hdi[0]), "hdi_high": float(hdi[1])}


# --- output ------------------------------------------------------------------------


def _print_track(track: str, res: dict) -> None:
    pct = int(HDI_PROB * 100)
    print(f"  leaderboard [{track}] (strength β, {pct}% HDI | win% vs field):")
    for r in res["rankings"]:
        print(
            f"    {r['rank']:>2}. {r['model']:<28} "
            f"β={r['strength_mean']:+.3f} [{r['hdi_low']:+.3f}, {r['hdi_high']:+.3f}]  "
            f"wpf={r['win_prob_vs_field'] * 100:5.1f}%"
        )
    pop = res["population"]
    mg, md = pop["mu_gamma"], pop["mu_delta"]
    print(f"  position bias  μ_γ = {mg['mean']:+.3f} [{mg['hdi_low']:+.3f}, {mg['hdi_high']:+.3f}]")
    print(f"  self-pref      μ_δ = {md['mean']:+.3f} [{md['hdi_low']:+.3f}, {md['hdi_high']:+.3f}]")
    d = res["diagnostics"]
    flag = "  <-- CHECK CONVERGENCE" if d["max_r_hat"] > 1.01 or d["n_divergences"] > 0 else ""
    print(
        f"  diagnostics: max_r_hat={d['max_r_hat']:.3f} "
        f"min_ess={d['min_ess_bulk']:.0f} divergences={d['n_divergences']}{flag}"
    )


def _write_outputs(cfg: Config, tracks_out: dict[str, dict]) -> None:
    cfg.outputs_data_dir.mkdir(parents=True, exist_ok=True)
    rank_rows: list[dict] = []
    eff_rows: list[dict] = []
    for track, res in tracks_out.items():
        for r in res["rankings"]:
            rank_rows.append({"track": track, **r})
        for e in res["judge_effects"]:
            eff_rows.append({"track": track, **e})
        pop = res["population"]
        eff_rows.append(
            {
                "track": track,
                "judge": "panel (mean)",
                "kappa_mean": None,
                "kappa_hdi_low": None,
                "kappa_hdi_high": None,
                "gamma_mean": pop["mu_gamma"]["mean"],
                "gamma_hdi_low": pop["mu_gamma"]["hdi_low"],
                "gamma_hdi_high": pop["mu_gamma"]["hdi_high"],
                "delta_mean": pop["mu_delta"]["mean"],
                "delta_hdi_low": pop["mu_delta"]["hdi_low"],
                "delta_hdi_high": pop["mu_delta"]["hdi_high"],
            }
        )

    rankings_path = cfg.outputs_data_dir / RANKINGS_FILENAME
    effects_path = cfg.outputs_data_dir / JUDGE_EFFECTS_FILENAME
    summary_path = cfg.outputs_data_dir / SUMMARY_FILENAME
    pl.DataFrame(rank_rows).write_csv(rankings_path)
    pl.DataFrame(eff_rows).write_csv(effects_path)
    payload = {
        "hdi_prob": HDI_PROB,
        "sampler": {
            "draws": DRAWS,
            "tune": TUNE,
            "chains": CHAINS,
            "target_accept": TARGET_ACCEPT,
        },
        "tracks": tracks_out,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    for path in (rankings_path, effects_path, summary_path):
        print(f"wrote {path}")


def run(cfg: Config) -> None:
    df = read_results(cfg.outputs_data_dir / RESULTS_FILENAME)
    family_of = family_map(cfg)
    tracks_out: dict[str, dict] = {}
    for track in TRACKS:
        design = build_design(df, track, family_of)
        if design.n_obs == 0:
            print(f"=== bradley-terry: {track} — no verdicts, skipping ===")
            continue
        print(
            f"=== bradley-terry: {track} "
            f"(n_obs={design.n_obs}, models={len(design.models)}, judges={len(design.judges)}) ==="
        )
        idata = _fit(design, cfg.seed)
        tpath = trace_path(cfg, track)
        _save_trace(tpath, idata)
        tracks_out[track] = _summarize(design, idata)
        _print_track(track, tracks_out[track])
        print(f"  trace -> {tpath}")
    if tracks_out:
        _write_outputs(cfg, tracks_out)
