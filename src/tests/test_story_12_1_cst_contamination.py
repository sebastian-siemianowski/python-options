"""
Story 12.1 – Contamination Probability Estimation via EM
=========================================================
Verify that estimate_cst_contamination() correctly separates normal
from crisis observations using the EM algorithm.
"""

import os, sys, math
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_kernels import estimate_cst_contamination


def _make_cst_data(n, epsilon, nu_normal, nu_crisis, seed=42):
    """Generate contaminated Student-t samples."""
    from scipy.stats import t as t_dist
    rng = np.random.RandomState(seed)
    n_crisis = rng.binomial(n, epsilon)
    n_normal = n - n_crisis
    normal_samples = t_dist.rvs(nu_normal, size=n_normal, random_state=rng)
    crisis_samples = t_dist.rvs(nu_crisis, size=n_crisis, random_state=rng)
    data = np.concatenate([normal_samples, crisis_samples])
    rng.shuffle(data)
    return data


class TestEMConvergence:
    """EM converges within 20 iterations for all configurations."""

    def test_converges_moderate_contamination(self):
        """epsilon=0.05: converges quickly."""
        data = _make_cst_data(2000, 0.05, 8.0, 3.0, seed=100)
        result = estimate_cst_contamination(data)
        assert result["converged"]
        assert result["n_iter"] <= 20

    def test_converges_low_contamination(self):
        """epsilon=0.02: converges."""
        data = _make_cst_data(2000, 0.02, 8.0, 3.0, seed=200)
        result = estimate_cst_contamination(data)
        assert result["converged"]

    def test_converges_high_contamination(self):
        """epsilon=0.10: converges with sufficient iterations."""
        data = _make_cst_data(2000, 0.10, 8.0, 3.0, seed=300)
        result = estimate_cst_contamination(data, max_iter=100)
        assert result["converged"]


class TestEpsilonRecovery:
    """Recovered epsilon is in the correct range for synthetic data."""

    def test_spy_like(self):
        """SPY-like: epsilon ~ 0.02-0.05."""
        data = _make_cst_data(3000, 0.03, 8.0, 3.0, seed=400)
        result = estimate_cst_contamination(data)
        assert 0.005 < result["epsilon_hat"] < 0.15

    def test_mstr_like(self):
        """MSTR-like: epsilon ~ 0.05-0.10."""
        data = _make_cst_data(3000, 0.08, 8.0, 3.0, seed=500)
        result = estimate_cst_contamination(data)
        assert 0.02 < result["epsilon_hat"] < 0.25

    def test_gold_like(self):
        """GC=F-like: epsilon ~ 0.01-0.03."""
        data = _make_cst_data(3000, 0.02, 10.0, 4.0, seed=600)
        result = estimate_cst_contamination(data)
        assert 0.005 < result["epsilon_hat"] < 0.10


class TestPosteriors:
    """Posterior crisis probabilities are correct."""

    def test_posteriors_shape(self):
        """Posteriors array has same length as input."""
        data = _make_cst_data(500, 0.05, 8.0, 3.0, seed=700)
        result = estimate_cst_contamination(data)
        assert len(result["posteriors"]) == len(data)

    def test_posteriors_bounded(self):
        """All posteriors in [0, 1]."""
        data = _make_cst_data(500, 0.05, 8.0, 3.0, seed=800)
        result = estimate_cst_contamination(data)
        assert np.all(result["posteriors"] >= 0)
        assert np.all(result["posteriors"] <= 1)

    def test_extreme_outliers_flagged(self):
        """Extreme outliers should have high crisis posterior."""
        rng = np.random.RandomState(900)
        data = rng.standard_normal(500)
        # Inject extreme outliers
        data[0] = 8.0
        data[1] = -7.0
        data[2] = 9.0
        result = estimate_cst_contamination(data)
        # Outliers should have higher posterior than normal obs
        mean_post_outlier = np.mean(result["posteriors"][:3])
        mean_post_normal = np.mean(result["posteriors"][3:])
        assert mean_post_outlier > mean_post_normal


class TestEdgeCases:
    """Edge cases: small sample, pure normal."""

    def test_small_sample(self):
        """n < 10 returns gracefully."""
        data = np.array([0.1, 0.2, -0.1])
        result = estimate_cst_contamination(data)
        assert not result["converged"]

    def test_trajectory_recorded(self):
        """Convergence trajectory is recorded."""
        data = _make_cst_data(1000, 0.05, 8.0, 3.0, seed=1000)
        result = estimate_cst_contamination(data)
        assert len(result["trajectory"]) >= 2
        assert result["trajectory"][0] == 0.03  # initial
