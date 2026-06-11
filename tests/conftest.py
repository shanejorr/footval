"""Shared fixture builders. Numeric fixtures are derived *with scipy* so the
expected pass/fail outcomes are correct by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scipy import stats as st

from footbench.checks import BUCKETS, TYPES
from footbench.config import Config, ModelCfg, SandboxCfg
from footbench.distributions import skewnorm_mode

CELLS = [(b, t) for b in BUCKETS for t in TYPES]

_FAMILY_PROVIDER = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
    "deepseek": "deepseek",
}


def make_cfg(
    root: Path,
    candidates: list[str],
    judges: list[str],
    families: dict[str, str],
    weights: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Config:
    models = {
        name: ModelCfg(name=name, provider=_FAMILY_PROVIDER.get(fam, "openai"), family=fam)
        for name, fam in families.items()
    }
    return Config(
        root=root,
        candidate_models=tuple(candidates),
        judge_models=tuple(judges),
        n_samples=1,
        gen_temperature=1.0,
        judge_temperature=0.0,
        seed=42,
        criteria_weights=tuple(weights),
        interval_tol=0.10,
        sandbox=SandboxCfg(),
        generation_max_output_tokens=1000,
        judging_max_output_tokens=1000,
        judge_max_format_retries=2,
        models=models,
    )


def jrow(
    model: str,
    judge: str,
    criterion: str,
    score: int | None,
    judge_family: str = "openai",
    is_self: bool = False,
    iid: str | None = None,
) -> dict[str, Any]:
    return {
        "instance_id": iid or f"{model}__s0",
        "model": model,
        "judge": judge,
        "judge_family": judge_family,
        "is_self": is_self,
        "criterion": criterion,
        "score": score,
        "justification": "because",
    }


def jrows(
    model: str, judge: str, s: int | None, p: int | None, c: int | None, **kw: Any
) -> list[dict[str, Any]]:
    return [
        jrow(model, judge, "soundness", s, **kw),
        jrow(model, judge, "priors", p, **kw),
        jrow(model, judge, "code", c, **kw),
    ]


def strategy(
    bucket: str,
    stype: str,
    title: str,
    family: str,
    params: dict[str, float],
    mpv: float,
    low: float,
    high: float,
    shape: str = "symmetric",
    tails: str = "moderate",
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "type": stype,
        "title": title,
        "recommendation": "Do the thing more often.",
        "rationale": "It raises win probability.",
        "prior": {
            "belief_summaries": {
                "most_plausible_value": mpv,
                "interval_95_low": low,
                "interval_95_high": high,
                "shape": shape,
                "tails": tails,
            },
            "distribution_family": {"name": family, "justification": "matches stated shape"},
            "parameters": {**params, "conversion_note": "derived in fixture"},
            "consistency_check": "mpv inside interval; support signed",
        },
    }


def normal_strategy(
    bucket: str, stype: str, title: str, mpv: float = 0.5, half_width: float = 1.0
) -> dict[str, Any]:
    low, high = mpv - half_width, mpv + half_width
    sigma = (high - low) / 3.92
    return strategy(bucket, stype, title, "normal", {"mu": mpv, "sigma": sigma}, mpv, low, high)


def t_strategy(
    bucket: str, stype: str, title: str, mpv: float = 0.5, half_width: float = 1.0, nu: float = 4
) -> dict[str, Any]:
    low, high = mpv - half_width, mpv + half_width
    crit = float(st.t.ppf(0.975, nu))
    sigma = (high - low) / (2 * crit)
    return strategy(
        bucket,
        stype,
        title,
        "student_t",
        {"nu": nu, "mu": mpv, "sigma": sigma},
        mpv,
        low,
        high,
        tails="heavy",
    )


def skew_strategy(
    bucket: str,
    stype: str,
    title: str,
    xi: float = 0.3,
    omega: float = 1.2,
    alpha: float = 4.0,
) -> dict[str, Any]:
    dist = st.skewnorm(alpha, loc=xi, scale=omega)
    low, high = float(dist.ppf(0.025)), float(dist.ppf(0.975))
    mpv = skewnorm_mode(xi, omega, alpha)
    return strategy(
        bucket,
        stype,
        title,
        "skew_normal",
        {"xi": xi, "omega": omega, "alpha": alpha},
        mpv,
        low,
        high,
        shape="right_skewed" if alpha > 0 else "left_skewed",
    )


def valid_response(strategies: list[dict] | None = None) -> dict[str, Any]:
    if strategies is None:
        strategies = [
            normal_strategy(b, t, f"Strategy {i}", mpv=0.2 * i, half_width=1.0 + 0.1 * i)
            for i, (b, t) in enumerate(CELLS)
        ]
    return {
        "quantity": {
            "description": "Change in own-team expected single-game scoring margin",
            "units": "points",
            "support": "unbounded real; positive favors own team",
        },
        "strategies": strategies,
        "plot_script": "import matplotlib.pyplot as plt\nplt.plot([0, 1])\n",
    }
