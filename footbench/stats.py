"""Interrater-reliability statistics for stage 5.

Krippendorff's alpha comes from the `krippendorff` package; ICC(2,k) and
Kendall's W are implemented here (two-way ANOVA mean squares / tie-corrected
rank concordance) to avoid dragging pandas+statsmodels in for two formulas.
"""

from __future__ import annotations

import math
import warnings

import krippendorff as _krippendorff
import numpy as np
from scipy.stats import ConstantInputWarning, rankdata, spearmanr


def krippendorff_alpha_ordinal(matrix: np.ndarray) -> float | None:
    """Krippendorff's alpha, ordinal metric.

    ``matrix`` is raters x units with ``np.nan`` for missing judgments.
    """
    try:
        a = _krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
    except (ValueError, ZeroDivisionError):
        return None
    return None if math.isnan(a) else float(a)


def icc2k(matrix: np.ndarray) -> float | None:
    """ICC(2,k): two-way random effects, average measures, absolute agreement.

    ``matrix`` is units x raters and must be complete (caller drops rows with
    missing cells). Formula per Shrout & Fleiss (1979):
    ICC(2,k) = (MSR - MSE) / (MSR + (MSC - MSE) / n).
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        return None
    n, k = matrix.shape
    if n < 2 or k < 2 or np.isnan(matrix).any():
        return None
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    grand = matrix.mean()
    msr = k * ((row_means - grand) ** 2).sum() / (n - 1)
    msc = n * ((col_means - grand) ** 2).sum() / (k - 1)
    mse = ((matrix - row_means[:, None] - col_means[None, :] + grand) ** 2).sum() / (
        (n - 1) * (k - 1)
    )
    denom = msr + (msc - mse) / n
    if denom == 0:
        return None
    return float((msr - mse) / denom)


def kendalls_w(score_matrix: np.ndarray) -> float | None:
    """Kendall's W (tie-corrected) over judges' derived rankings.

    ``score_matrix`` is judges x items (e.g. mean score per model); items with
    any missing value are dropped; higher scores rank better. Each judge's
    scores are converted to average ranks before computing W.
    """
    m_all = np.asarray(score_matrix, dtype=float)
    if m_all.ndim != 2:
        return None
    complete = ~np.isnan(m_all).any(axis=0)
    m_scores = m_all[:, complete]
    m, n = m_scores.shape
    if m < 2 or n < 2:
        return None
    ranks = np.vstack([rankdata(-row, method="average") for row in m_scores])
    totals = ranks.sum(axis=0)
    s = ((totals - totals.mean()) ** 2).sum()
    tie_term = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_term += float((counts**3 - counts).sum())
    denom = m**2 * (n**3 - n) - m * tie_term
    if denom <= 0:
        return None
    return float(12 * s / denom)


def pairwise_spearman(score_matrix: np.ndarray, rater_names: list[str]) -> dict[str, float | None]:
    """Spearman rho for every rater pair, on pairwise-complete items."""
    m = np.asarray(score_matrix, dtype=float)
    out: dict[str, float | None] = {}
    for i in range(len(rater_names)):
        for j in range(i + 1, len(rater_names)):
            mask = ~np.isnan(m[i]) & ~np.isnan(m[j])
            key = f"{rater_names[i]} vs {rater_names[j]}"
            if mask.sum() < 2:
                out[key] = None
                continue
            with warnings.catch_warnings():
                # constant input (e.g. a judge giving everyone the same score)
                # is a legitimate "no correlation defined" case, not a bug
                warnings.simplefilter("ignore", ConstantInputWarning)
                rho = spearmanr(m[i][mask], m[j][mask]).statistic
            out[key] = None if math.isnan(rho) else round(float(rho), 4)
    return out


def alpha_band(alpha: float | None) -> str:
    """Interpretation band for Krippendorff's alpha."""
    if alpha is None:
        return "not computable"
    if alpha >= 0.80:
        return "strong agreement"
    if alpha >= 0.667:
        return "tentative agreement"
    return "weak agreement"


def icc_band(icc: float | None) -> str:
    """Interpretation band for ICC(2,k), per Koo & Li."""
    if icc is None:
        return "not computable"
    if icc < 0.50:
        return "poor"
    if icc <= 0.75:
        return "moderate"
    if icc <= 0.90:
        return "good"
    return "excellent"
