"""
Story 2.2: Log-Sum-Exp Stability in BMA Weight Computation
===========================================================
Verify BMA weights sum to 1, never underflow to zero, even with
extreme BIC spreads.
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

from models.vectorized_ops import vectorized_bma_weights


class TestBMAWeightStability:
    """Acceptance criteria for Story 2.2."""

    def test_weights_sum_to_one(self):
        """AC1: weights sum to 1.0 +/- 1e-14."""
        bic = np.array([100.0, 102.0, 105.0, 110.0, 120.0])
        w = vectorized_bma_weights(bic)
        assert abs(np.sum(w) - 1.0) < 1e-14, f"Sum = {np.sum(w)}"

    def test_large_bic_spread_no_zero(self):
        """AC2: BIC spread > 500 nats -- worst model gets w > 0."""
        best_bic = 100.0
        bic = np.array([best_bic + i * 100 for i in range(20)])
        # Spread = 1900 nats
        w = vectorized_bma_weights(bic)
        assert np.all(w > 0), f"Zero weight detected: {w}"
        assert abs(np.sum(w) - 1.0) < 1e-14

    def test_1000_nat_spread_20_models(self):
        """Task 3: 20 models with BIC spread of 1000 nats."""
        bic = np.linspace(100.0, 1100.0, 20)
        w = vectorized_bma_weights(bic)
        assert np.all(w > 0), f"Zero weights: {w}"
        assert abs(np.sum(w) - 1.0) < 1e-14

    def test_no_model_exactly_zero(self):
        """AC3: No model receives exactly w = 0.0."""
        # Extreme spread: 5000 nats
        bic = np.array([0.0, 5000.0, 10000.0])
        w = vectorized_bma_weights(bic)
        assert np.all(w > 0), f"Zero weight: {w}"

    def test_identical_bic_equal_weights(self):
        """Sanity: identical BIC -> equal weights."""
        bic = np.full(10, 200.0)
        w = vectorized_bma_weights(bic)
        np.testing.assert_allclose(w, 0.1, atol=1e-15)

    def test_ordering_preserved(self):
        """Better BIC (lower) should get higher weight."""
        bic = np.array([100.0, 110.0, 120.0, 130.0])
        w = vectorized_bma_weights(bic)
        for i in range(len(w) - 1):
            assert w[i] >= w[i + 1], f"w[{i}]={w[i]} < w[{i+1}]={w[i+1]}"

    def test_empty_input(self):
        """Empty BIC array should return empty."""
        w = vectorized_bma_weights(np.array([]))
        assert len(w) == 0

    def test_single_model(self):
        """Single model should get weight 1.0."""
        w = vectorized_bma_weights(np.array([42.0]))
        assert abs(w[0] - 1.0) < 1e-15
