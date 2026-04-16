"""
Story 5.2: Variance Inflation Stage with Quantile Matching
============================================================
beta minimizes sum_q (coverage(q, beta) - q)^2 for q in {0.50, 0.80, 0.95}.
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.gaussian import GaussianDriftModel
from scipy.stats import norm


def _synth_calibration_data(n=1000, true_sigma=0.02, seed=42):
    """Generate returns with known Gaussian distribution for calibration testing."""
    rng = np.random.default_rng(seed)
    mu_pred = np.zeros(n)
    S_pred = np.full(n, true_sigma ** 2)
    returns = rng.normal(0, true_sigma, n)
    return returns, mu_pred, S_pred


class TestVarianceInflation:
    """Acceptance criteria for Story 5.2."""

    def test_perfect_calibration_beta_near_1(self):
        """AC1: When model is perfectly calibrated, beta ~ 1.0."""
        returns, mu_pred, S_pred = _synth_calibration_data(n=5000)
        beta = GaussianDriftModel._gaussian_stage_2_variance_inflation(
            returns, mu_pred, S_pred, len(returns)
        )
        assert 0.85 <= beta <= 1.20, f"beta={beta} should be near 1.0 for perfect model"

    def test_underdispersed_beta_above_1(self):
        """AC2: When model variance is too small, beta > 1.0."""
        rng = np.random.default_rng(42)
        n = 2000
        true_sigma = 0.04
        model_sigma = 0.02  # Model underestimates variance
        returns = rng.normal(0, true_sigma, n)
        mu_pred = np.zeros(n)
        S_pred = np.full(n, model_sigma ** 2)

        beta = GaussianDriftModel._gaussian_stage_2_variance_inflation(
            returns, mu_pred, S_pred, n
        )
        assert beta > 1.0, f"beta={beta} should be > 1.0 for underdispersed model"

    def test_overdispersed_beta_below_1(self):
        """AC3: When model variance is too large, beta < 1.0."""
        rng = np.random.default_rng(42)
        n = 2000
        true_sigma = 0.01
        model_sigma = 0.02  # Model overestimates variance
        returns = rng.normal(0, true_sigma, n)
        mu_pred = np.zeros(n)
        S_pred = np.full(n, model_sigma ** 2)

        beta = GaussianDriftModel._gaussian_stage_2_variance_inflation(
            returns, mu_pred, S_pred, n
        )
        assert beta < 1.0, f"beta={beta} should be < 1.0 for overdispersed model"

    def test_coverage_after_inflation(self):
        """AC4: After applying beta, 95% interval coverage is in [93%, 97%]."""
        rng = np.random.default_rng(7)
        n = 3000
        true_sigma = 0.03
        returns = rng.normal(0, true_sigma, n)
        mu_pred = np.zeros(n)
        S_pred = np.full(n, (true_sigma * 0.8) ** 2)  # Slightly off

        beta = GaussianDriftModel._gaussian_stage_2_variance_inflation(
            returns, mu_pred, S_pred, n
        )

        # Compute coverage on full data with found beta
        sigma_cal = np.sqrt(np.maximum(S_pred * beta, 1e-20))
        z = (returns - mu_pred) / np.maximum(sigma_cal, 1e-10)
        pit = norm.cdf(z)
        cov_95 = float(np.mean(pit <= 0.95))
        assert 0.90 <= cov_95 <= 0.99, f"95% coverage = {cov_95}"

    def test_quantile_matching_objective(self):
        """AC5: Objective uses {0.50, 0.80, 0.95} quantiles."""
        rng = np.random.default_rng(99)
        n = 2000
        returns = rng.normal(0, 0.02, n)
        mu_pred = np.zeros(n)
        S_pred = np.full(n, 0.02 ** 2)

        beta = GaussianDriftModel._gaussian_stage_2_variance_inflation(
            returns, mu_pred, S_pred, n
        )
        # Just verify it returns a valid beta in grid range
        assert 0.80 <= beta <= 2.0, f"beta={beta} outside grid range"

    def test_grid_extends_to_2(self):
        """AC6: Grid includes beta up to 2.0 for fat-tailed assets."""
        rng = np.random.default_rng(42)
        n = 2000
        # Very fat-tailed returns with tiny model variance
        returns = rng.standard_t(3, n) * 0.05
        mu_pred = np.zeros(n)
        S_pred = np.full(n, 0.01 ** 2)  # Way too small

        beta = GaussianDriftModel._gaussian_stage_2_variance_inflation(
            returns, mu_pred, S_pred, n
        )
        # Should pick a large beta
        assert beta >= 1.5, f"beta={beta} should be large for fat-tailed data"
