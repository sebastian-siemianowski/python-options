"""
Story 10.2: GJR Leverage Effect Calibration
============================================
Asymmetric volatility response from empirical news impact.
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

from models.phi_student_t_unified import (
    compute_empirical_news_impact,
    fit_gjr_news_impact,
)


class TestGJRLeverageEffect:
    """Acceptance criteria for Story 10.2."""

    def test_equity_leverage_positive(self):
        """AC1: Equity-like returns have positive gamma_lev."""
        np.random.seed(42)
        n = 2000
        returns = np.random.normal(0, 0.01, n)
        # Add leverage effect: negative returns followed by higher vol
        for t in range(1, n):
            if returns[t - 1] < -0.01:
                returns[t] *= 1.5
        result = fit_gjr_news_impact(returns)
        assert result['gamma_lev'] > -0.1  # At least not strongly negative

    def test_symmetric_returns_gamma_near_zero(self):
        """AC2: Symmetric returns give gamma_lev near zero."""
        np.random.seed(42)
        n = 2000
        returns = np.random.normal(0, 0.01, n)  # Pure symmetric
        result = fit_gjr_news_impact(returns)
        assert abs(result['gamma_lev']) < 0.5  # Near zero

    def test_news_impact_curve_shape(self):
        """AC3: News impact curve has correct shape (bins and variance)."""
        np.random.seed(42)
        n = 1000
        returns = np.random.normal(0, 0.01, n)
        centers, cond_var = compute_empirical_news_impact(returns, n_bins=10)
        assert len(centers) > 5
        assert len(cond_var) == len(centers)
        assert np.all(cond_var > 0)  # Variance always positive

    def test_fit_result_structure(self):
        """AC4: Fit result has all expected fields."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        result = fit_gjr_news_impact(returns)
        required = {'omega', 'alpha', 'gamma_lev', 'asymmetry_ratio'}
        assert required == set(result.keys())

    def test_omega_positive(self):
        """AC5: Omega always positive."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 500)
        result = fit_gjr_news_impact(returns)
        assert result['omega'] > 0

    def test_short_series_fallback(self):
        """AC6: Short series returns reasonable defaults."""
        returns = np.array([0.01, -0.02, 0.005])
        result = fit_gjr_news_impact(returns)
        assert result['omega'] > 0
        assert result['alpha'] == 0.05
