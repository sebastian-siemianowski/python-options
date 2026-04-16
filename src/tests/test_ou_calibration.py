"""
Test Story 2.2: Ornstein-Uhlenbeck Calibration with Asset-Specific Mean Reversion.

Validates:
  1. AR(1) kappa estimation: fast mean reversion -> high kappa
  2. EWMA theta tracks price level
  3. Half-life bounds: 5 < half_life < 252
  4. Calibrated _ou_forecast uses proper OU projection
  5. Fallback to MA-based estimation when no tuned params
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest

from decision.market_temperature import (
    estimate_ou_kappa,
    estimate_ou_theta,
    estimate_ou_params,
    _ou_forecast,
    OU_HALF_LIFE_MIN,
    OU_HALF_LIFE_MAX,
)


class TestOUKappaEstimation(unittest.TestCase):
    """Tests for AR(1) kappa estimation."""

    def test_mean_reverting_series_positive_kappa(self):
        """Strongly mean-reverting series -> higher kappa."""
        rng = np.random.RandomState(42)
        n = 500
        x = np.zeros(n)
        kappa_true = 0.05
        for i in range(1, n):
            x[i] = x[i - 1] - kappa_true * x[i - 1] + rng.normal(0, 0.01)
        # Convert to log prices (centered at 0 = log(100))
        log_prices = 4.6 + x  # log(100) ~ 4.605
        
        kappa_est = estimate_ou_kappa(log_prices)
        # Should be in ballpark of true kappa (0.05)
        self.assertGreater(kappa_est, 0.02)
        self.assertLess(kappa_est, 0.12)

    def test_random_walk_low_kappa(self):
        """Random walk (no mean reversion) -> kappa near minimum."""
        rng = np.random.RandomState(99)
        random_walk = np.cumsum(rng.normal(0, 0.01, 500))
        log_prices = 4.6 + random_walk
        
        kappa_est = estimate_ou_kappa(log_prices)
        # Should be at or near minimum
        kappa_min = np.log(2) / OU_HALF_LIFE_MAX
        self.assertLessEqual(kappa_est, 0.03)

    def test_short_series_returns_default(self):
        """< 30 points returns default kappa."""
        from decision.market_temperature import OU_KAPPA_DEFAULT
        kappa = estimate_ou_kappa(np.array([4.6] * 10))
        self.assertAlmostEqual(kappa, OU_KAPPA_DEFAULT)

    def test_kappa_bounded(self):
        """Kappa always in [ln2/252, ln2/5]."""
        rng = np.random.RandomState(7)
        for _ in range(10):
            prices = 4.6 + np.cumsum(rng.normal(0, 0.02, 200))
            kappa = estimate_ou_kappa(prices)
            half_life = np.log(2) / kappa
            self.assertGreaterEqual(half_life, OU_HALF_LIFE_MIN - 0.1)
            self.assertLessEqual(half_life, OU_HALF_LIFE_MAX + 0.1)


class TestOUThetaEstimation(unittest.TestCase):
    """Tests for EWMA theta estimation."""

    def test_theta_near_recent_level(self):
        """Theta should track recent price level."""
        prices = np.array([100.0] * 200)
        theta = estimate_ou_theta(prices, kappa=0.05)
        self.assertAlmostEqual(theta, 100.0, places=1)

    def test_theta_responds_to_trend(self):
        """After uptrend, theta should be above starting price."""
        prices = np.linspace(100, 120, 200)
        theta = estimate_ou_theta(prices, kappa=0.05)
        self.assertGreater(theta, 110)

    def test_theta_with_fast_kappa_tracks_recent(self):
        """Fast kappa (short half-life) -> theta closer to recent price."""
        prices = np.concatenate([np.ones(100) * 100, np.ones(100) * 120])
        theta_fast = estimate_ou_theta(prices, kappa=0.10)
        theta_slow = estimate_ou_theta(prices, kappa=0.01)
        # Fast kappa -> shorter EWMA span -> more weight on recent 120
        self.assertGreater(theta_fast, theta_slow)


class TestOUParamsEstimation(unittest.TestCase):
    """Tests for full OU parameter estimation."""

    def test_returns_all_keys(self):
        """estimate_ou_params returns kappa, theta, sigma_ou, half_life_days."""
        prices = np.exp(np.cumsum(np.random.RandomState(42).normal(0, 0.01, 300)) + 4.6)
        params = estimate_ou_params(prices)
        
        self.assertIn("kappa", params)
        self.assertIn("theta", params)
        self.assertIn("sigma_ou", params)
        self.assertIn("half_life_days", params)

    def test_half_life_consistent_with_kappa(self):
        """half_life = ln(2) / kappa."""
        prices = np.exp(np.cumsum(np.random.RandomState(42).normal(0, 0.01, 300)) + 4.6)
        params = estimate_ou_params(prices)
        
        expected_hl = np.log(2) / params["kappa"]
        self.assertAlmostEqual(params["half_life_days"], expected_hl, places=2)


class TestOUForecastCalibrated(unittest.TestCase):
    """Test _ou_forecast with calibrated OU params."""

    def _make_prices(self, n=300, seed=42):
        rng = np.random.RandomState(seed)
        returns = rng.normal(0.0003, 0.01, n)
        prices = pd.Series(100.0 * np.exp(np.cumsum(returns)))
        return prices

    def test_calibrated_reversion(self):
        """With OU params, forecast reverts toward theta."""
        prices = self._make_prices()
        current = float(prices.iloc[-1])
        theta = current * 0.95  # 5% below current -> expect negative forecast
        
        tuned = {
            "ou_params": {
                "kappa": 0.05,
                "theta": theta,
                "sigma_ou": 0.01,
                "half_life_days": np.log(2) / 0.05,
            }
        }
        fc = _ou_forecast(prices, [30], tuned_params=tuned)
        # Price above theta -> expect negative (reversion down)
        self.assertLess(fc[0], 0)

    def test_fast_kappa_stronger_reversion(self):
        """Higher kappa -> faster/stronger mean reversion forecast."""
        prices = self._make_prices()
        current = float(prices.iloc[-1])
        theta = current * 0.90  # 10% below
        
        slow = {"ou_params": {"kappa": 0.01, "theta": theta, "sigma_ou": 0.01,
                               "half_life_days": np.log(2) / 0.01}}
        fast = {"ou_params": {"kappa": 0.10, "theta": theta, "sigma_ou": 0.01,
                               "half_life_days": np.log(2) / 0.10}}
        
        fc_slow = _ou_forecast(prices, [30], tuned_params=slow)
        fc_fast = _ou_forecast(prices, [30], tuned_params=fast)
        
        # Both negative (above theta), fast should be more negative
        self.assertLess(fc_fast[0], fc_slow[0])

    def test_invalid_half_life_falls_back(self):
        """Half-life outside [5, 252] -> reverts to MA-based fallback."""
        prices = self._make_prices()
        
        # Half-life = 2 (too short)
        tuned = {"ou_params": {"kappa": 0.35, "theta": 100.0, "sigma_ou": 0.01,
                                "half_life_days": 2.0}}
        fc = _ou_forecast(prices, [7], tuned_params=tuned)
        self.assertIsInstance(fc[0], float)

    def test_no_tuned_params_fallback(self):
        """Without tuned params, uses MA-based estimation."""
        prices = self._make_prices()
        fc = _ou_forecast(prices, [7, 30])
        self.assertEqual(len(fc), 2)
        for v in fc:
            self.assertIsInstance(v, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
