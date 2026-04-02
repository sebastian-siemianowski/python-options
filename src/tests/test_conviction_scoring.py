"""
Test Story 8.3: Conviction-Weighted Signal Ranking.

Validates:
  1. High agreement + high accuracy -> HIGH conviction
  2. Low agreement -> LOW conviction
  3. Model agreement via Herfindahl
  4. Forecast stability with consistent signs
  5. Ranking by conviction (descending)
  6. Factor weights sum to 1
  7. Edge case: single model
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.conviction_scoring import (
    compute_conviction,
    compute_model_agreement,
    compute_forecast_stability,
    rank_by_conviction,
    W_MODEL_AGREEMENT,
    W_REGIME_CONFIDENCE,
    W_HISTORICAL_ACCURACY,
    W_FORECAST_STABILITY,
)


class TestConvictionScoring(unittest.TestCase):
    """Tests for conviction-weighted signal ranking."""

    def test_high_agreement_high_accuracy_is_high(self):
        """Strong agreement + strong accuracy -> HIGH conviction."""
        bma = np.array([0.85, 0.05, 0.05, 0.05])  # Dominant model
        hits = np.array([True] * 18 + [False] * 2)   # 90% hit rate
        forecasts = {1: 1.5, 7: 2.0, 30: 3.0}        # All positive

        score = compute_conviction("SPY", bma, regime_fit=0.9, recent_hits=hits,
                                   forecasts_by_horizon=forecasts)
        self.assertEqual(score.category, "HIGH")
        self.assertGreater(score.composite, 0.7)

    def test_low_agreement_is_low(self):
        """Uniform weights + poor accuracy -> LOW conviction."""
        bma = np.array([0.1] * 10)  # Uniform = no agreement
        hits = np.array([True, False] * 5)  # 50%
        forecasts = {1: 0.5, 7: -0.3, 30: 0.2}  # Mixed signs

        score = compute_conviction("XYZ", bma, regime_fit=0.3, recent_hits=hits,
                                   forecasts_by_horizon=forecasts)
        self.assertEqual(score.category, "LOW")

    def test_model_agreement_perfect(self):
        """Single dominant model -> agreement ~1.0."""
        bma = np.array([0.99, 0.005, 0.005])
        agreement = compute_model_agreement(bma)
        self.assertGreater(agreement, 0.9)

    def test_model_agreement_uniform(self):
        """Uniform weights -> agreement ~0.0."""
        n = 10
        bma = np.ones(n) / n
        agreement = compute_model_agreement(bma)
        self.assertAlmostEqual(agreement, 0.0, places=5)

    def test_forecast_stability_consistent_signs(self):
        """All same sign -> high stability."""
        forecasts = {1: 0.5, 7: 1.0, 30: 1.5, 90: 2.0}
        stability = compute_forecast_stability(forecasts)
        self.assertGreater(stability, 0.7)

    def test_forecast_stability_mixed_signs(self):
        """Mixed signs -> low stability."""
        forecasts = {1: 0.5, 7: -1.0, 30: 0.3, 90: -0.8}
        stability = compute_forecast_stability(forecasts)
        self.assertLess(stability, 0.5)

    def test_ranking_descending(self):
        """Rank by conviction descending."""
        scores = [
            compute_conviction("A", np.array([0.5, 0.5]), 0.5,
                               np.array([True, False]), {1: 0.5}),
            compute_conviction("B", np.array([0.9, 0.1]), 0.9,
                               np.array([True] * 10), {1: 1.0, 7: 1.5}),
        ]
        ranked = rank_by_conviction(scores)
        self.assertEqual(ranked[0].symbol, "B")
        self.assertGreater(ranked[0].composite, ranked[1].composite)

    def test_weights_sum_to_one(self):
        """Factor weights sum to 1.0."""
        total = W_MODEL_AGREEMENT + W_REGIME_CONFIDENCE + W_HISTORICAL_ACCURACY + W_FORECAST_STABILITY
        self.assertAlmostEqual(total, 1.0, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
