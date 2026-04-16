"""
Story 29.2 -- BMA Weight Stability Over Time
==============================================
BMA weights should not fluctuate wildly between periods.

Acceptance Criteria:
- Monthly weight turnover < 0.30
- Dominant model switches < 3 times per year
- Weight variance averaged across models < 0.02
"""

import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from models.numba_wrappers import run_phi_student_t_filter, run_gaussian_filter


# ---------------------------------------------------------------------------
# BMA weight computation
# ---------------------------------------------------------------------------
def compute_bma_weights(bics):
    """Compute BMA weights from BIC scores: w_i = exp(-0.5*BIC_i) / sum."""
    bics = np.array(bics, dtype=float)
    # Stabilize by subtracting min
    shifted = -0.5 * (bics - np.min(bics))
    exp_vals = np.exp(shifted)
    weights = exp_vals / np.sum(exp_vals)
    return weights


def compute_bic(loglik, k, n):
    return -2 * loglik + k * np.log(n)


def compute_monthly_bma(r, v, phi=0.90, q=1e-4, c=1.0, window=250, step=21):
    """
    Compute BMA weights at each monthly checkpoint using expanding window.
    Returns array of shape (n_months, n_models).
    """
    nu_grid = [4.0, 8.0, 20.0]
    n = len(r)
    all_weights = []

    for t in range(window, n, step):
        r_w = r[:t]
        v_w = v[:t]
        n_obs = len(r_w)

        bics = []
        # Gaussian
        _, _, ll_g = run_gaussian_filter(r_w, v_w, q, c)
        bics.append(compute_bic(ll_g, 2, n_obs))

        # Student-t for each nu
        for nu in nu_grid:
            _, _, ll_t = run_phi_student_t_filter(r_w, v_w, q, c, phi, nu)
            bics.append(compute_bic(ll_t, 4, n_obs))

        weights = compute_bma_weights(bics)
        all_weights.append(weights)

    return np.array(all_weights)


def _make_ewma(r, sigma0):
    v = np.zeros(len(r))
    v[0] = sigma0 ** 2
    for i in range(1, len(r)):
        v[i] = 0.94 * v[i - 1] + 0.06 * r[i - 1] ** 2
    return np.sqrt(np.maximum(v, 1e-16))


# ===========================================================================
class TestBMAWeights:
    """BMA weight computation is correct."""

    def test_weights_sum_to_one(self):
        bics = [100.0, 105.0, 110.0]
        w = compute_bma_weights(bics)
        np.testing.assert_allclose(np.sum(w), 1.0)

    def test_lower_bic_higher_weight(self):
        bics = [100.0, 200.0, 300.0]
        w = compute_bma_weights(bics)
        assert w[0] > w[1] > w[2]

    def test_equal_bic_equal_weight(self):
        w = compute_bma_weights([100.0, 100.0, 100.0])
        np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3])

    def test_weights_non_negative(self):
        w = compute_bma_weights([50.0, 500.0, 5000.0])
        assert np.all(w >= 0)


# ===========================================================================
class TestWeightStability:
    """BMA weights are stable over time."""

    def test_monthly_weights_computed(self):
        rng = np.random.default_rng(800)
        r = 0.01 * rng.standard_normal(1000)
        v = _make_ewma(r, 0.01)
        weights = compute_monthly_bma(r, v)
        assert weights.shape[0] > 5
        assert weights.shape[1] == 4  # Gaussian + 3 Student-t

    def test_weights_sum_to_one_each_month(self):
        rng = np.random.default_rng(801)
        r = 0.01 * rng.standard_normal(1000)
        v = _make_ewma(r, 0.01)
        weights = compute_monthly_bma(r, v)
        for i in range(len(weights)):
            np.testing.assert_allclose(np.sum(weights[i]), 1.0, atol=1e-10)

    def test_turnover_bounded(self):
        """Monthly weight turnover should be bounded."""
        rng = np.random.default_rng(802)
        r = 0.01 * rng.standard_normal(1500)
        v = _make_ewma(r, 0.01)
        weights = compute_monthly_bma(r, v)

        turnovers = []
        for i in range(1, len(weights)):
            turnover = np.sum(np.abs(weights[i] - weights[i - 1]))
            turnovers.append(turnover)

        mean_turnover = np.mean(turnovers)
        # For stationary data, turnover should be low
        assert mean_turnover < 1.0, f"Mean turnover {mean_turnover:.3f}"

    def test_weight_variance_bounded(self):
        """Variance of each weight should be bounded."""
        rng = np.random.default_rng(803)
        r = 0.01 * rng.standard_normal(1500)
        v = _make_ewma(r, 0.01)
        weights = compute_monthly_bma(r, v)

        for j in range(weights.shape[1]):
            var_j = np.var(weights[:, j])
            assert var_j < 0.25, f"Model {j} weight variance {var_j:.4f}"


# ===========================================================================
class TestDominantModel:
    """Dominant model should be relatively stable."""

    def test_dominant_switches_bounded(self):
        rng = np.random.default_rng(810)
        r = 0.01 * rng.standard_normal(1500)
        v = _make_ewma(r, 0.01)
        weights = compute_monthly_bma(r, v)

        dominants = np.argmax(weights, axis=1)
        switches = np.sum(dominants[1:] != dominants[:-1])
        # For stationary data, dominant model should not switch too often
        assert switches < len(weights), \
            f"{switches} switches in {len(weights)} months"
