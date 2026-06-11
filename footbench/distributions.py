"""Canonical prior-distribution routines.

Every recomputation from a candidate's *reported* parameters — the stage-3
parameter-reproduction check and the stage-6 prior-comparison plot — goes
through this module and nowhere else, so checks and plots can never disagree.
"""

from __future__ import annotations

import math
from typing import Any

from scipy import optimize, stats

FAMILIES = ("normal", "student_t", "skew_normal")

REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "normal": ("mu", "sigma"),
    "student_t": ("nu", "mu", "sigma"),
    "skew_normal": ("xi", "omega", "alpha"),
}

_POSITIVE_PARAMS = {"sigma", "omega", "nu"}


def is_number(v: Any) -> bool:
    """True for finite ints/floats; bools are rejected."""
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def validate_params(family: str, params: Any) -> str | None:
    """Return an error string, or None when the family/params pair is usable."""
    if family not in REQUIRED_PARAMS:
        return f"invalid family {family!r}"
    if not isinstance(params, dict):
        return "parameters is not a JSON object"
    problems = []
    for key in REQUIRED_PARAMS[family]:
        v = params.get(key)
        if not is_number(v):
            problems.append(f"{key} missing or non-numeric")
        elif key in _POSITIVE_PARAMS and v <= 0:
            problems.append(f"{key} must be > 0 (got {v})")
    return "; ".join(problems) or None


def build_dist(family: str, params: dict[str, float]):
    """The canonical routine: a frozen scipy distribution from reported parameters."""
    if family == "normal":
        return stats.norm(loc=params["mu"], scale=params["sigma"])
    if family == "student_t":
        return stats.t(df=params["nu"], loc=params["mu"], scale=params["sigma"])
    if family == "skew_normal":
        return stats.skewnorm(params["alpha"], loc=params["xi"], scale=params["omega"])
    raise ValueError(f"unknown family {family!r}")


def interval_95(family: str, params: dict[str, float]) -> tuple[float, float]:
    """Recompute the central 95% interval using the spec's formulas."""
    if family == "normal":
        mu, sigma = params["mu"], params["sigma"]
        return mu - 1.96 * sigma, mu + 1.96 * sigma
    if family == "student_t":
        crit = float(stats.t.ppf(0.975, params["nu"]))
        return params["mu"] - crit * params["sigma"], params["mu"] + crit * params["sigma"]
    if family == "skew_normal":
        dist = build_dist(family, params)
        return float(dist.ppf(0.025)), float(dist.ppf(0.975))
    raise ValueError(f"unknown family {family!r}")


def family_center(family: str, params: dict[str, float]) -> float:
    """The family's center: mu, or the numerically computed mode for skew_normal."""
    if family in ("normal", "student_t"):
        return float(params["mu"])
    if family == "skew_normal":
        return skewnorm_mode(params["xi"], params["omega"], params["alpha"])
    raise ValueError(f"unknown family {family!r}")


def skewnorm_mode(xi: float, omega: float, alpha: float) -> float:
    """Mode of a skew-normal, by bounded maximization of the pdf.

    No closed form exists; the bounded search over the 0.1–99.9% quantile
    range is robust for any alpha (alpha=0 converges to xi).
    """
    dist = stats.skewnorm(alpha, loc=xi, scale=omega)
    lo, hi = float(dist.ppf(0.001)), float(dist.ppf(0.999))
    res = optimize.minimize_scalar(lambda x: -dist.pdf(x), bounds=(lo, hi), method="bounded")
    return float(res.x)
