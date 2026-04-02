"""
Test Story 1.1: Adaptive Process Noise Floor with Regime Conditioning.

Validates:
  1. Q_FLOOR_BY_REGIME config has correct values for all 5 regimes
  2. apply_regime_q_floor() correctly clips q and marks q_floor_applied
  3. BIC adjustment is applied when floor binds
  4. Floor does NOT modify models already above the floor
  5. Synthetic trending series produces q >= q_floor
  6. Synthetic ranging series produces q >= q_floor (smaller floor)
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from tuning.tune import (
    Q_FLOOR_BY_REGIME,
    Q_FLOOR_BIC_LAMBDA,
    apply_regime_q_floor,
    MarketRegime,
    REGIME_LABELS,
)


class TestQFloorConfig(unittest.TestCase):
    """Test Q_FLOOR_BY_REGIME configuration."""

    def test_all_five_regimes_defined(self):
        """All 5 regimes must have a floor value."""
        for regime in MarketRegime:
            self.assertIn(
                regime.value, Q_FLOOR_BY_REGIME,
                f"Missing floor for regime {regime.name} ({regime.value})"
            )

    def test_floor_values_are_positive(self):
        """All floor values must be positive floats."""
        for regime, floor in Q_FLOOR_BY_REGIME.items():
            self.assertGreater(floor, 0, f"Floor for regime {regime} must be > 0")
            self.assertIsInstance(floor, float)

    def test_crisis_has_highest_floor(self):
        """CRISIS_JUMP should have the highest floor (maximum adaptivity)."""
        crisis_floor = Q_FLOOR_BY_REGIME[MarketRegime.CRISIS_JUMP]
        for regime, floor in Q_FLOOR_BY_REGIME.items():
            if regime != MarketRegime.CRISIS_JUMP:
                self.assertGreaterEqual(
                    crisis_floor, floor,
                    f"CRISIS_JUMP floor ({crisis_floor}) should be >= {REGIME_LABELS.get(regime)} floor ({floor})"
                )

    def test_low_vol_range_has_lowest_floor(self):
        """LOW_VOL_RANGE should have the lowest floor (minimal intervention)."""
        lvr_floor = Q_FLOOR_BY_REGIME[MarketRegime.LOW_VOL_RANGE]
        for regime, floor in Q_FLOOR_BY_REGIME.items():
            if regime != MarketRegime.LOW_VOL_RANGE:
                self.assertLessEqual(
                    lvr_floor, floor,
                    f"LOW_VOL_RANGE floor ({lvr_floor}) should be <= {REGIME_LABELS.get(regime)} floor ({floor})"
                )

    def test_exact_floor_values(self):
        """Verify exact floor values match specification."""
        expected = {
            MarketRegime.LOW_VOL_TREND: 5e-5,
            MarketRegime.HIGH_VOL_TREND: 1e-4,
            MarketRegime.LOW_VOL_RANGE: 2e-5,
            MarketRegime.HIGH_VOL_RANGE: 5e-5,
            MarketRegime.CRISIS_JUMP: 5e-4,
        }
        for regime, expected_floor in expected.items():
            self.assertAlmostEqual(
                Q_FLOOR_BY_REGIME[regime.value], expected_floor,
                places=10,
                msg=f"Floor for {regime.name} should be {expected_floor}"
            )

    def test_bic_lambda_positive(self):
        """BIC penalty lambda must be positive."""
        self.assertGreater(Q_FLOOR_BIC_LAMBDA, 0)


class TestApplyRegimeQFloor(unittest.TestCase):
    """Test apply_regime_q_floor() function."""

    def _make_models(self, q_values):
        """Create a minimal models dict with given q values."""
        models = {}
        for i, q in enumerate(q_values):
            models[f"model_{i}"] = {
                "fit_success": True,
                "q": q,
                "c": 1.0,
                "phi": 0.95,
                "nu": None,
                "n_params": 2,
                "bic": -1000.0,
                "log_likelihood": -500.0,
                "mean_log_likelihood": -0.5,
            }
        return models

    def _make_returns_vol(self, n=200):
        """Create synthetic returns and vol arrays."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0, 0.01, n)
        vol = np.full(n, 0.01)
        return returns, vol

    def test_floor_binds_when_q_below(self):
        """Models with q < q_floor should be clipped to q_floor."""
        models = self._make_models([1e-8, 1e-7, 1e-6])
        returns, vol = self._make_returns_vol()

        regime = MarketRegime.LOW_VOL_TREND  # floor = 5e-5
        n_floored, n_total = apply_regime_q_floor(models, regime, returns, vol)

        self.assertEqual(n_floored, 3)  # all below 5e-5
        self.assertEqual(n_total, 3)

        for name, info in models.items():
            self.assertEqual(info["q"], 5e-5)
            self.assertTrue(info["q_floor_applied"])
            self.assertIn("q_mle_original", info)

    def test_floor_does_not_bind_when_q_above(self):
        """Models with q >= q_floor should not be modified."""
        models = self._make_models([1e-3, 5e-3, 1e-2])
        returns, vol = self._make_returns_vol()

        regime = MarketRegime.CRISIS_JUMP  # floor = 5e-4
        n_floored, n_total = apply_regime_q_floor(models, regime, returns, vol)

        self.assertEqual(n_floored, 0)
        self.assertEqual(n_total, 3)

        for name, info in models.items():
            self.assertFalse(info["q_floor_applied"])
            self.assertNotIn("q_mle_original", info)

    def test_mixed_floor_application(self):
        """Some models floored, some not."""
        models = self._make_models([1e-8, 1e-3])
        returns, vol = self._make_returns_vol()

        regime = MarketRegime.LOW_VOL_RANGE  # floor = 2e-5
        n_floored, n_total = apply_regime_q_floor(models, regime, returns, vol)

        self.assertEqual(n_floored, 1)
        self.assertEqual(n_total, 2)

        self.assertTrue(models["model_0"]["q_floor_applied"])
        self.assertEqual(models["model_0"]["q"], 2e-5)

        self.assertFalse(models["model_1"]["q_floor_applied"])
        self.assertEqual(models["model_1"]["q"], 1e-3)

    def test_bic_penalty_applied_when_floor_binds(self):
        """BIC should increase (worsen) when floor overrides MLE."""
        # Use phi=None, nu=None for simple Gaussian filter (most robust path)
        models = {
            "model_0": {
                "fit_success": True,
                "q": 1e-8,
                "c": 1.0,
                "phi": None,
                "nu": None,
                "n_params": 2,
                "bic": -1000.0,
                "log_likelihood": -500.0,
                "mean_log_likelihood": -0.5,
            }
        }
        returns, vol = self._make_returns_vol()

        regime = MarketRegime.HIGH_VOL_TREND  # floor = 1e-4
        apply_regime_q_floor(models, regime, returns, vol)

        # BIC penalty = lambda * log(q_floor / q_mle)
        expected_penalty = Q_FLOOR_BIC_LAMBDA * math.log(1e-4 / 1e-8)
        self.assertIn("bic_floor_penalty", models["model_0"],
                       "bic_floor_penalty should be set when floor binds")
        self.assertAlmostEqual(
            models["model_0"]["bic_floor_penalty"], expected_penalty, places=2
        )
        # BIC penalty should be positive (worsening)
        self.assertGreater(models["model_0"]["bic_floor_penalty"], 0)

    def test_q_mle_original_preserved(self):
        """Original q_mle should be preserved for diagnostics."""
        q_original = 3e-7
        models = self._make_models([q_original])
        returns, vol = self._make_returns_vol()

        apply_regime_q_floor(models, MarketRegime.CRISIS_JUMP, returns, vol)

        self.assertAlmostEqual(
            models["model_0"]["q_mle_original"], q_original, places=15
        )

    def test_failed_models_skipped(self):
        """Models with fit_success=False should be skipped."""
        models = {
            "good": {"fit_success": True, "q": 1e-8, "c": 1.0, "phi": None, "nu": None, "n_params": 2,
                     "bic": -1000, "log_likelihood": -500, "mean_log_likelihood": -0.5},
            "bad": {"fit_success": False, "q": 1e-8},
        }
        returns, vol = self._make_returns_vol()

        n_floored, n_total = apply_regime_q_floor(models, MarketRegime.LOW_VOL_TREND, returns, vol)

        self.assertEqual(n_total, 1)  # only "good" counted
        self.assertEqual(n_floored, 1)
        self.assertNotIn("q_floor_applied", models["bad"])

    def test_each_regime_applies_correct_floor(self):
        """Each regime should apply its own specific floor."""
        returns, vol = self._make_returns_vol()

        for regime in MarketRegime:
            models = self._make_models([1e-10])
            apply_regime_q_floor(models, regime, returns, vol)

            expected_floor = Q_FLOOR_BY_REGIME[regime]
            actual_q = models["model_0"]["q"]
            self.assertAlmostEqual(
                actual_q, expected_floor, places=12,
                msg=f"Regime {regime.name}: expected q={expected_floor}, got q={actual_q}"
            )


class TestQFloorSyntheticData(unittest.TestCase):
    """Test q floor with synthetic trending and ranging series."""

    def test_trending_series_q_above_floor(self):
        """Synthetic trending series should have q >= q_floor after optimization."""
        rng = np.random.RandomState(123)
        n = 500
        # Create trending returns: drift + noise
        drift = 0.001  # 0.1% daily drift
        noise = rng.normal(0, 0.01, n)
        returns = drift + noise
        vol = np.full(n, 0.015)

        # Construct models with tiny q (simulating MLE collapse)
        models = {
            "kalman_gaussian": {
                "fit_success": True,
                "q": 1e-8,  # MLE collapsed
                "c": 1.0,
                "phi": None,
                "nu": None,
                "n_params": 2,
                "bic": -2000.0,
                "log_likelihood": -1000.0,
                "mean_log_likelihood": -2.0,
            }
        }

        # Apply floor for LOW_VOL_TREND (drift matters)
        n_floored, _ = apply_regime_q_floor(
            models, MarketRegime.LOW_VOL_TREND, returns, vol
        )

        self.assertEqual(n_floored, 1)
        self.assertGreaterEqual(models["kalman_gaussian"]["q"], Q_FLOOR_BY_REGIME[MarketRegime.LOW_VOL_TREND])

    def test_ranging_series_smaller_floor(self):
        """Ranging series should use the LOW_VOL_RANGE floor (smallest)."""
        rng = np.random.RandomState(456)
        n = 500
        # Mean-reverting returns
        returns = rng.normal(0, 0.005, n)
        vol = np.full(n, 0.005)

        models = {
            "kalman_gaussian": {
                "fit_success": True,
                "q": 1e-9,
                "c": 1.0,
                "phi": None,
                "nu": None,
                "n_params": 2,
                "bic": -2500.0,
                "log_likelihood": -1200.0,
                "mean_log_likelihood": -2.4,
            }
        }

        # Apply floor for LOW_VOL_RANGE (small but nonzero)
        n_floored, _ = apply_regime_q_floor(
            models, MarketRegime.LOW_VOL_RANGE, returns, vol
        )

        self.assertEqual(n_floored, 1)
        self.assertGreaterEqual(models["kalman_gaussian"]["q"], Q_FLOOR_BY_REGIME[MarketRegime.LOW_VOL_RANGE])
        # LOW_VOL_RANGE floor should be smaller than LOW_VOL_TREND floor
        self.assertLess(
            Q_FLOOR_BY_REGIME[MarketRegime.LOW_VOL_RANGE],
            Q_FLOOR_BY_REGIME[MarketRegime.LOW_VOL_TREND],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
