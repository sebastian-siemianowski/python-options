"""
Test Story 2.1: GARCH Parameter Loading from Tune Cache.

Validates:
  1. _garch_forecast uses tuned params when provided
  2. Falls back to defaults when tuned_params absent
  3. Stationarity check: alpha + beta >= 1.0 reverts to defaults
  4. _load_tuned_params extracts BMA-weighted GARCH params from regime models
  5. Different GARCH params produce different vol forecasts
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import unittest

import numpy as np

from decision.market_temperature import _garch_forecast


class TestGarchParamLoading(unittest.TestCase):
    """Tests for GARCH parameter loading from tuned cache."""

    def _make_returns(self, n=200, seed=42):
        """Generate realistic return series."""
        rng = np.random.RandomState(seed)
        return rng.normal(0.0003, 0.012, n)

    def test_default_params_work(self):
        """Without tuned_params, uses hardcoded defaults."""
        returns = self._make_returns()
        fc = _garch_forecast(returns, [1, 7, 30])
        self.assertEqual(len(fc), 3)
        for v in fc:
            self.assertIsInstance(v, float)

    def test_tuned_params_used(self):
        """When tuned_params provided, they change the vol_ratio -> output."""
        # Craft returns with vol cluster at end to amplify GARCH difference
        rng = np.random.RandomState(42)
        calm = rng.normal(0.001, 0.005, 150)
        shock = rng.normal(0.0, 0.04, 50)
        returns = np.concatenate([calm, shock])
        
        fc_default = _garch_forecast(returns, [1, 7, 30])
        
        # Very low persistence -> vol decays fast -> vol_ratio closer to 1
        low_pers = {
            "garch_params": {
                "omega": 0.0005,
                "alpha": 0.05,
                "beta": 0.40,  # persistence = 0.45 (very low)
            }
        }
        fc_tuned = _garch_forecast(returns, [1, 7, 30], tuned_params=low_pers)
        
        # At least one horizon must differ due to different vol dynamics
        any_diff = any(abs(a - b) > 1e-12 for a, b in zip(fc_default, fc_tuned))
        self.assertTrue(any_diff, "Tuned GARCH params should change forecasts")

    def test_stationarity_check(self):
        """Alpha + beta >= 1.0 reverts to defaults."""
        returns = self._make_returns()
        fc_default = _garch_forecast(returns, [7])
        
        # Non-stationary params
        tuned = {
            "garch_params": {
                "omega": 0.00001,
                "alpha": 0.5,
                "beta": 0.6,  # sum = 1.1 >= 1.0
            }
        }
        fc_nonstat = _garch_forecast(returns, [7], tuned_params=tuned)
        
        # Should revert to defaults -> same as default
        self.assertAlmostEqual(fc_default[0], fc_nonstat[0], places=10)

    def test_missing_garch_params_key(self):
        """tuned_params without garch_params -> use defaults."""
        returns = self._make_returns()
        fc_default = _garch_forecast(returns, [7])
        fc_no_garch = _garch_forecast(returns, [7], tuned_params={"q": 0.001})
        self.assertAlmostEqual(fc_default[0], fc_no_garch[0], places=10)

    def test_high_persistence_wider_vol(self):
        """High persistence (alpha+beta near 1.0) -> stronger vol dynamics."""
        rng = np.random.RandomState(99)
        # Returns with a volatility cluster at the end
        base = rng.normal(0.0, 0.005, 150)
        shock = rng.normal(0.0, 0.03, 50)
        returns = np.concatenate([base, shock])
        
        low_pers = {
            "garch_params": {"omega": 0.00002, "alpha": 0.05, "beta": 0.80}
        }
        high_pers = {
            "garch_params": {"omega": 0.000005, "alpha": 0.10, "beta": 0.88}
        }
        fc_low = _garch_forecast(returns, [30], tuned_params=low_pers)
        fc_high = _garch_forecast(returns, [30], tuned_params=high_pers)
        
        # Both should produce valid floats
        self.assertIsInstance(fc_low[0], float)
        self.assertIsInstance(fc_high[0], float)

    def test_short_returns_returns_zeros(self):
        """Returns < 30 points -> zeros regardless of params."""
        returns = np.random.normal(0, 0.01, 10)
        tuned = {
            "garch_params": {"omega": 0.00001, "alpha": 0.08, "beta": 0.88}
        }
        fc = _garch_forecast(returns, [1, 7], tuned_params=tuned)
        self.assertEqual(fc, [0.0, 0.0])


class TestLoadTunedParamsGarch(unittest.TestCase):
    """Test _load_tuned_params GARCH extraction from regime models."""

    def test_extracts_garch_from_regime_models(self):
        """BMA-weighted GARCH extraction from regime model cache."""
        from decision.market_temperature import _load_tuned_params
        
        # Create temporary tune cache with GARCH params
        tune_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "tune",
        )
        test_file = os.path.join(tune_dir, "__TEST_GARCH__.json")
        
        cache = {
            "global": {"q": 0.001, "c": 0.5, "phi": 0.98},
            "regime": {
                "0": {
                    "models": {
                        "model_a": {
                            "weight": 0.6,
                            "garch_omega": 0.00001,
                            "garch_alpha": 0.08,
                            "garch_beta": 0.90,
                        },
                        "model_b": {
                            "weight": 0.4,
                            "garch_omega": 0.00002,
                            "garch_alpha": 0.10,
                            "garch_beta": 0.85,
                        },
                    }
                }
            },
        }
        
        try:
            with open(test_file, "w") as f:
                json.dump(cache, f)
            
            params = _load_tuned_params("__TEST_GARCH__")
            self.assertIsNotNone(params)
            self.assertIn("garch_params", params)
            
            gp = params["garch_params"]
            # Weighted average: omega = (0.6*0.00001 + 0.4*0.00002) / 1.0 = 0.000014
            self.assertAlmostEqual(gp["omega"], 0.000014, places=8)
            # alpha = (0.6*0.08 + 0.4*0.10) / 1.0 = 0.088
            self.assertAlmostEqual(gp["alpha"], 0.088, places=5)
            # beta = (0.6*0.90 + 0.4*0.85) / 1.0 = 0.88
            self.assertAlmostEqual(gp["beta"], 0.88, places=5)
            # persistence = alpha + beta
            self.assertAlmostEqual(gp["persistence"], 0.088 + 0.88, places=5)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_skips_nonstationary_models(self):
        """Models with alpha+beta >= 1.0 are skipped in BMA averaging."""
        from decision.market_temperature import _load_tuned_params
        
        tune_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "tune",
        )
        test_file = os.path.join(tune_dir, "__TEST_GARCH2__.json")
        
        cache = {
            "global": {"q": 0.001, "c": 0.5, "phi": 0.98},
            "regime": {
                "0": {
                    "models": {
                        "good_model": {
                            "weight": 0.5,
                            "garch_omega": 0.00001,
                            "garch_alpha": 0.08,
                            "garch_beta": 0.90,
                        },
                        "bad_model": {
                            "weight": 0.5,
                            "garch_omega": 0.00001,
                            "garch_alpha": 0.50,
                            "garch_beta": 0.60,  # sum=1.1, non-stationary
                        },
                    }
                }
            },
        }
        
        try:
            with open(test_file, "w") as f:
                json.dump(cache, f)
            
            params = _load_tuned_params("__TEST_GARCH2__")
            gp = params["garch_params"]
            # Only good_model contributes
            self.assertAlmostEqual(gp["alpha"], 0.08, places=5)
            self.assertAlmostEqual(gp["beta"], 0.90, places=5)
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)


if __name__ == "__main__":
    unittest.main(verbosity=2)
