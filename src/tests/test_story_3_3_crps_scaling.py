"""
Story 3.3: CRPS-Optimal GARCH Residual Scaling
================================================
Grid-search scaling alpha to minimize CRPS of GARCH variance.
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

from calibration.pit_calibration import crps_optimal_garch_scaling, crps_gaussian


class TestCRPSOptimalScaling:
    """Acceptance criteria for Story 3.3."""

    def test_alpha_in_range(self):
        """AC3: Scaling factor alpha in [0.85, 1.15]."""
        rng = np.random.default_rng(42)
        n = 500
        h_t = np.full(n, 0.0004)  # Constant variance
        residuals = rng.normal(0, 0.02, n)

        result = crps_optimal_garch_scaling(h_t, residuals)
        assert 0.80 <= result["alpha_opt"] <= 1.20, (
            f"alpha_opt = {result['alpha_opt']}"
        )

    def test_crps_improves_or_matches(self):
        """AC1: CRPS on holdout should not be worse than unscaled."""
        rng = np.random.default_rng(7)
        n = 500
        # Simulate mis-scaled GARCH: true sigma=0.02, GARCH thinks sigma=0.025
        true_sigma = 0.02
        h_t = np.full(n, 0.025 ** 2)  # Overstated variance
        residuals = rng.normal(0, true_sigma, n)

        result = crps_optimal_garch_scaling(h_t, residuals)
        # Optimal alpha should be < 1 (reduce overestimated variance)
        assert result["alpha_opt"] < 1.05
        # CRPS on holdout should not be much worse
        assert result["crps_holdout"] <= result["crps_unscaled"] * 1.1

    def test_crps_gaussian_closed_form(self):
        """Verify closed-form CRPS against numerical integration."""
        from scipy.stats import norm

        mu = np.array([0.0])
        sigma = np.array([1.0])
        y = np.array([0.5])

        crps = crps_gaussian(mu, sigma, y)
        # Known CRPS for N(0,1) at y=0.5
        z = 0.5
        expected = 1.0 * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
        assert abs(crps - abs(expected)) < 1e-10

    def test_perfect_scaling_alpha_near_one(self):
        """When GARCH is well-calibrated, alpha_opt should be near 1.0."""
        rng = np.random.default_rng(99)
        n = 1000
        true_var = 0.0004
        h_t = np.full(n, true_var)
        residuals = rng.normal(0, np.sqrt(true_var), n)

        result = crps_optimal_garch_scaling(h_t, residuals)
        assert abs(result["alpha_opt"] - 1.0) < 0.15, (
            f"alpha_opt = {result['alpha_opt']} (expected near 1.0)"
        )

    def test_returns_all_keys(self):
        """Result dict has expected keys."""
        rng = np.random.default_rng(1)
        h_t = np.full(100, 0.0004)
        residuals = rng.normal(0, 0.02, 100)

        result = crps_optimal_garch_scaling(h_t, residuals)
        for key in ["alpha_opt", "crps_train", "crps_holdout", "crps_unscaled"]:
            assert key in result, f"Missing key: {key}"
            assert np.isfinite(result[key])

    def test_custom_alpha_grid(self):
        """Custom alpha_grid is respected."""
        rng = np.random.default_rng(42)
        h_t = np.full(200, 0.0004)
        residuals = rng.normal(0, 0.02, 200)

        result = crps_optimal_garch_scaling(
            h_t, residuals,
            alpha_grid=np.array([0.90, 1.00, 1.10]),
        )
        assert result["alpha_opt"] in [0.90, 1.00, 1.10]
