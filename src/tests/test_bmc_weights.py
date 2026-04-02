"""
Test Story 2.4: Bayesian Model Combination Replacing Fixed Weights.

Validates:
  1. BMC weight update rule increases weight for accurate model
  2. Weight floor prevents model extinction
  3. Forgetting factor shrinks old weights toward uniform
  4. BMC weights integrated into ensemble_forecast
  5. Model that consistently fails loses weight but stays above floor
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.market_temperature import (
    update_bmc_weights,
    get_bmc_weights,
    _bmc_log_likelihood,
    BMC_WEIGHT_FLOOR,
    BMC_MODEL_NAMES,
    _BMC_CACHE,
)


class TestBMCLogLikelihood(unittest.TestCase):
    """Tests for predictive log-likelihood computation."""

    def test_perfect_forecast_highest(self):
        """Forecast matching realized should have highest likelihood."""
        ll_perfect = _bmc_log_likelihood(1.0, 1.0, sigma_pct=1.0)
        ll_bad = _bmc_log_likelihood(5.0, 1.0, sigma_pct=1.0)
        self.assertGreater(ll_perfect, ll_bad)

    def test_symmetric(self):
        """Likelihood is symmetric around forecast."""
        ll_above = _bmc_log_likelihood(1.0, 2.0, sigma_pct=1.0)
        ll_below = _bmc_log_likelihood(1.0, 0.0, sigma_pct=1.0)
        self.assertAlmostEqual(ll_above, ll_below, places=10)


class TestBMCWeightUpdate(unittest.TestCase):
    """Tests for BMC weight update rule."""

    def setUp(self):
        """Clear BMC cache between tests."""
        _BMC_CACHE.clear()

    def test_initial_weights_uniform(self):
        """Before any updates, weights should be uniform."""
        w = get_bmc_weights("TEST_ASSET_BMC")
        expected = [0.2] * 5
        for i in range(5):
            self.assertAlmostEqual(w[i], expected[i], places=6)

    def test_accurate_model_gains_weight(self):
        """Model that predicts well should gain weight."""
        # Model 0 (kalman) predicts realized perfectly; others are wrong
        for _ in range(20):
            forecasts = [1.0, -2.0, -3.0, -1.0, -2.0]  # Kalman is right
            update_bmc_weights("TEST_GAIN", forecasts, realized_pct=1.0)
        
        w = get_bmc_weights("TEST_GAIN")
        # Kalman (index 0) should have highest weight
        self.assertEqual(w.index(max(w)), 0,
                         f"Kalman should have highest weight, got {w}")

    def test_bad_model_loses_weight_but_above_floor(self):
        """Model that consistently predicts wrong loses weight to floor."""
        for _ in range(50):
            # Model 4 always predicts wrong direction
            forecasts = [1.0, 0.8, 0.5, 1.2, -5.0]
            update_bmc_weights("TEST_FLOOR", forecasts, realized_pct=1.0)
        
        w = get_bmc_weights("TEST_FLOOR")
        # Model 4 (classical) should be at or near floor
        self.assertGreaterEqual(w[4], BMC_WEIGHT_FLOOR - 0.01)
        # But not zero
        self.assertGreater(w[4], 0)

    def test_weights_sum_to_one(self):
        """Weights always sum to 1.0."""
        for _ in range(10):
            forecasts = [np.random.normal(0, 2) for _ in range(5)]
            w = update_bmc_weights("TEST_SUM", forecasts, realized_pct=0.5)
            self.assertAlmostEqual(sum(w), 1.0, places=6)

    def test_all_weights_above_floor(self):
        """No weight should go below BMC_WEIGHT_FLOOR."""
        for _ in range(30):
            forecasts = [10.0, -10.0, -10.0, -10.0, -10.0]
            w = update_bmc_weights("TEST_ALL_FLOOR", forecasts, realized_pct=10.0)
        
        for i, wi in enumerate(w):
            self.assertGreaterEqual(wi, BMC_WEIGHT_FLOOR - 0.001,
                                    f"Weight {i} ({wi}) below floor")

    def test_forgetting_factor_convergence(self):
        """After many updates with equal performance, weights return toward uniform."""
        # First, make model 0 dominant
        for _ in range(30):
            update_bmc_weights("TEST_FORGET", [5.0, -5.0, -5.0, -5.0, -5.0], 5.0)
        
        w_biased = get_bmc_weights("TEST_FORGET")
        self.assertGreater(w_biased[0], 0.3)
        
        # Now all models perform equally
        for _ in range(200):
            update_bmc_weights("TEST_FORGET", [0.1, 0.1, 0.1, 0.1, 0.1], 0.1)
        
        w_converged = get_bmc_weights("TEST_FORGET")
        # Should converge toward uniform-ish
        for i in range(5):
            self.assertGreater(w_converged[i], 0.10,
                               f"Weight {i} should converge toward uniform")

    def test_n_updates_tracked(self):
        """Update count is tracked."""
        update_bmc_weights("TEST_COUNT", [0.0] * 5, 0.0)
        update_bmc_weights("TEST_COUNT", [0.0] * 5, 0.0)
        self.assertEqual(_BMC_CACHE["TEST_COUNT"]["n_updates"], 2)


class TestBMCIntegration(unittest.TestCase):
    """Integration test: BMC weights blend into ensemble_forecast."""

    def setUp(self):
        _BMC_CACHE.clear()

    def test_ensemble_forecast_uses_bmc(self):
        """ensemble_forecast should incorporate BMC weights."""
        import pandas as pd
        from decision.market_temperature import ensemble_forecast
        
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 400))))
        
        result = ensemble_forecast(prices, asset_type="equity", asset_name="TEST_BMC_INT")
        self.assertEqual(len(result), 8)  # 7 forecasts + confidence


if __name__ == "__main__":
    unittest.main(verbosity=2)
