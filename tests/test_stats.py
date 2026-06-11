import numpy as np
import pytest
from scipy.stats import spearmanr

from footbench.stats import (
    alpha_band,
    icc2k,
    icc_band,
    kendalls_w,
    krippendorff_alpha_ordinal,
    pairwise_spearman,
)

# Shrout & Fleiss (1979), Table 2: 6 targets x 4 judges. Published ICC(2,k)=0.62.
SF_MATRIX = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def test_icc2k_matches_shrout_fleiss():
    assert icc2k(SF_MATRIX) == pytest.approx(0.62, abs=0.01)


def test_icc2k_degenerate_inputs():
    assert icc2k(np.ones((4, 3))) is None  # zero variance
    assert icc2k(np.array([1.0, 2.0])) is None  # not 2-D
    m = SF_MATRIX.copy()
    m[0, 0] = np.nan
    assert icc2k(m) is None  # caller must pass complete cases


def test_krippendorff_perfect_agreement():
    matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=float)
    assert krippendorff_alpha_ordinal(matrix) == pytest.approx(1.0)


def test_krippendorff_disagreement_lowers_alpha():
    matrix = np.array([[1, 2, 3, 4], [1, 2, 3, 1]], dtype=float)
    a = krippendorff_alpha_ordinal(matrix)
    assert a is not None and a < 1.0


def test_krippendorff_tolerates_missing_cells():
    matrix = np.array(
        [
            [1, 2, 3, np.nan, 4],
            [1, 2, np.nan, 4, 4],
            [np.nan, 2, 3, 4, 5],
        ],
        dtype=float,
    )
    a = krippendorff_alpha_ordinal(matrix)
    assert a is not None and -1.0 <= a <= 1.0


def test_krippendorff_no_variation_does_not_crash():
    matrix = np.array([[2, 2], [2, 2]], dtype=float)
    a = krippendorff_alpha_ordinal(matrix)
    assert a is None or a == pytest.approx(1.0)


def test_kendalls_w_identical_rankings():
    scores = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1]], dtype=float)
    assert kendalls_w(scores) == pytest.approx(1.0)


def test_kendalls_w_opposed_rankings():
    scores = np.array([[3, 2, 1], [1, 2, 3]], dtype=float)
    assert kendalls_w(scores) == pytest.approx(0.0)


def test_kendalls_w_tie_correction_hand_computed():
    # j1, j2 rank A>B>C; j3 ties A and B. Hand computation: W = 186/198.
    scores = np.array([[3, 2, 1], [3, 2, 1], [3, 3, 1]], dtype=float)
    assert kendalls_w(scores) == pytest.approx(186 / 198, abs=1e-6)


def test_kendalls_w_drops_incomplete_items():
    scores = np.array([[3, 2, np.nan, 1], [3, 2, 5, 1], [3, 2, 4, 1]], dtype=float)
    assert kendalls_w(scores) == pytest.approx(1.0)  # computed on the 3 complete items


def test_pairwise_spearman_matches_scipy():
    rng = np.random.default_rng(7)
    m = rng.normal(size=(3, 8))
    out = pairwise_spearman(m, ["a", "b", "c"])
    expected = spearmanr(m[0], m[1]).statistic
    assert out["a vs b"] == pytest.approx(expected, abs=1e-4)
    assert set(out) == {"a vs b", "a vs c", "b vs c"}


def test_alpha_bands():
    assert alpha_band(0.85) == "strong agreement"
    assert alpha_band(0.80) == "strong agreement"
    assert alpha_band(0.70) == "tentative agreement"
    assert alpha_band(0.667) == "tentative agreement"
    assert alpha_band(0.5) == "weak agreement"
    assert alpha_band(None) == "not computable"


def test_icc_bands():
    assert icc_band(0.49) == "poor"
    assert icc_band(0.50) == "moderate"
    assert icc_band(0.75) == "moderate"
    assert icc_band(0.76) == "good"
    assert icc_band(0.90) == "good"
    assert icc_band(0.91) == "excellent"
    assert icc_band(None) == "not computable"
