"""Shared fixture builders. Numeric fixtures are derived *with scipy* so the
expected pass/fail outcomes are correct by construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scipy import stats as st

from footval.checks import BUCKETS, TYPES
from footval.config import Config, ModelCfg
from footval.distributions import skewnorm_mode

CELLS = [(b, t) for b in BUCKETS for t in TYPES]

_FAMILY_PROVIDER = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
    "zai": "zai",
}

# The real judge system prompt, copied byte-for-byte into each test root so
# judge_system(cfg) resolves against the same text the pipeline uses.
JUDGE_PROMPT_TEXT = (Path(__file__).resolve().parent.parent / "footval.judge.prompt.md").read_text()


def make_cfg(
    root: Path,
    candidates: list[str],
    families: dict[str, str],
    pairwise_judges: tuple[str, ...] = (),
    pairwise_both_orders: bool = False,
    judge_params: dict[str, dict] | None = None,
) -> Config:
    judge_params = judge_params or {}
    models = {
        name: ModelCfg(
            name=name,
            provider=_FAMILY_PROVIDER.get(fam, "openai"),
            family=fam,
            judge_params=judge_params.get(name, {}),
        )
        for name, fam in families.items()
    }
    (root / "footval.judge.prompt.md").write_text(JUDGE_PROMPT_TEXT)
    return Config(
        root=root,
        candidate_models=tuple(candidates),
        n_samples=1,
        gen_temperature=1.0,
        judge_temperature=0.0,
        seed=42,
        interval_tol=0.10,
        generation_max_output_tokens=1000,
        pairwise_max_output_tokens=1000,
        pairwise_max_format_retries=2,
        models=models,
        pairwise_judges=pairwise_judges,
        pairwise_both_orders=pairwise_both_orders,
    )


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
    }
