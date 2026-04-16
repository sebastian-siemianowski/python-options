"""
Story 12.3 – CST vs Hansen Model Selection
============================================
Verify that select_tail_model() correctly identifies CST for
contaminated data and Hansen for systematically skewed data.
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

from models.numba_kernels import select_tail_model


class TestHeuristicSelection:
    """Heuristic selects CST for contamination, Hansen for skewness."""

    def test_cst_for_outlier_contamination(self):
        """Symmetric data with outliers -> CST."""
        rng = np.random.RandomState(42)
        data = rng.standard_normal(2000)
        # Inject 5% outliers (symmetric)
        n_out = 100
        data[:n_out] = rng.choice([-1, 1], n_out) * rng.uniform(4, 8, n_out)
        result = select_tail_model(data)
        assert result["heuristic_model"] == "cst"
        assert result["outlier_fraction"] > 0.03
        assert abs(result["skewness"]) < 0.3

    def test_hansen_for_skewed_data(self):
        """Skewed data without contamination -> Hansen."""
        rng = np.random.RandomState(43)
        # Generate skewed data via exponential transform
        data = rng.exponential(1.0, 2000) - 1.0  # positive skew
        data = data / np.std(data)
        result = select_tail_model(data)
        assert result["heuristic_model"] == "hansen"
        assert abs(result["skewness"]) >= 0.3


class TestBICSelection:
    """BIC correctly selects the better model."""

    def test_bic_computed(self):
        """BIC values are finite for both models."""
        rng = np.random.RandomState(44)
        data = rng.standard_normal(1000)
        result = select_tail_model(data)
        assert math.isfinite(result["bic_cst"])
        assert math.isfinite(result["bic_hansen"])

    def test_selected_model_valid(self):
        """Selected model is either 'cst' or 'hansen'."""
        rng = np.random.RandomState(45)
        data = rng.standard_normal(1000)
        result = select_tail_model(data)
        assert result["selected_model"] in ("cst", "hansen")


class TestDiagnostics:
    """Data diagnostics are correctly computed."""

    def test_skewness_sign(self):
        """Positive skew data has positive skewness."""
        rng = np.random.RandomState(46)
        data = np.abs(rng.standard_normal(2000))
        result = select_tail_model(data)
        assert result["skewness"] > 0

    def test_excess_kurtosis_heavy_tails(self):
        """Student-t data has positive excess kurtosis."""
        from scipy.stats import t as t_dist
        data = t_dist.rvs(4, size=2000, random_state=47)
        result = select_tail_model(data)
        assert result["excess_kurtosis"] > 0

    def test_outlier_fraction(self):
        """Outlier fraction computed correctly."""
        rng = np.random.RandomState(48)
        data = rng.standard_normal(1000)
        result = select_tail_model(data)
        # For standard normal, ~0.27% exceed 3 sigma
        assert result["outlier_fraction"] < 0.05


class TestEdgeCases:
    """Edge cases: small sample, pure normal."""

    def test_small_sample(self):
        """n < 20 returns default."""
        data = np.array([0.1, 0.2, -0.1, 0.3])
        result = select_tail_model(data)
        assert result["selected_model"] == "hansen"

    def test_heuristic_agrees_field(self):
        """heuristic_agrees is a boolean."""
        rng = np.random.RandomState(49)
        data = rng.standard_normal(1000)
        result = select_tail_model(data)
        assert isinstance(result["heuristic_agrees"], bool)
