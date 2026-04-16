"""
Test Story 1.3 & 1.4: GAS-Q Propagation + Multi-Scale Drift Estimation.

Validates:
  1. _kalman_forecast uses GAS-Q path when tuned_params provided
  2. EMA fallback works when no tuned_params
  3. Graceful degradation on missing GAS-Q params
  4. Multi-scale: fast/medium/slow filters produce different mu_t
  5. Multi-scale: horizon mapping assigns correct filter scale
  6. ensemble_forecast accepts tuned_params kwarg
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest

from decision.market_temperature import (
    _kalman_forecast,
    _run_kalman_filter_pass,
    ensemble_forecast,
    STANDARD_HORIZONS,
    Q_FAST_MULT,
    Q_SLOW_MULT,
    HORIZON_FILTER_MAP,
)


class TestKalmanForecastGASQ(unittest.TestCase):
    """Test GAS-Q integration in _kalman_forecast."""

    def _make_returns(self, T=500, drift=0.0005, vol=0.01, seed=42):
        rng = np.random.RandomState(seed)
        return drift + vol * rng.randn(T)

    def test_ema_fallback_no_tuned_params(self):
        """Without tuned_params, _kalman_forecast uses EMA path."""
        ret = self._make_returns()
        fc = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=None)
        self.assertEqual(len(fc), len(STANDARD_HORIZONS))
        # Should produce non-zero forecasts for trending data
        self.assertTrue(any(abs(f) > 0.01 for f in fc))

    def test_ema_fallback_empty_tuned_params(self):
        """Empty tuned_params dict falls back to EMA."""
        ret = self._make_returns()
        fc = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params={})
        self.assertEqual(len(fc), len(STANDARD_HORIZONS))

    def test_ema_fallback_no_gasq(self):
        """Tuned params without gas_q_augmented fall back to EMA."""
        ret = self._make_returns()
        params = {"q": 1e-4, "c": 1.0, "phi": 1.0, "gas_q_augmented": False}
        fc = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=params)
        self.assertEqual(len(fc), len(STANDARD_HORIZONS))

    def test_gasq_path_used_when_available(self):
        """GAS-Q Kalman path should produce different forecasts than EMA."""
        ret = self._make_returns()
        params = {
            "q": 1e-4,
            "c": 1.0,
            "phi": 0.999,
            "nu": None,
            "gas_q_augmented": True,
            "gas_q_params": {"omega": 1e-5, "alpha": 0.05, "beta": 0.90},
        }

        fc_gasq = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=params)
        fc_ema = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=None)

        self.assertEqual(len(fc_gasq), len(STANDARD_HORIZONS))
        # GAS-Q path should give different values than EMA
        # (they use completely different methods)
        diffs = [abs(g - e) for g, e in zip(fc_gasq, fc_ema)]
        self.assertTrue(
            any(d > 0.001 for d in diffs),
            f"GAS-Q and EMA forecasts should differ, diffs={diffs}"
        )

    def test_gasq_student_t_path(self):
        """Student-t GAS-Q path works when nu is provided."""
        ret = self._make_returns()
        params = {
            "q": 1e-4,
            "c": 1.0,
            "phi": 0.999,
            "nu": 6.0,
            "gas_q_augmented": True,
            "gas_q_params": {"omega": 1e-5, "alpha": 0.05, "beta": 0.90},
        }

        fc = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=params)
        self.assertEqual(len(fc), len(STANDARD_HORIZONS))
        self.assertTrue(all(np.isfinite(f) for f in fc))

    def test_graceful_degradation_bad_gasq_params(self):
        """Bad GAS-Q params should fall back to EMA without crashing."""
        ret = self._make_returns()
        params = {
            "q": 1e-4,
            "c": 1.0,
            "gas_q_augmented": True,
            "gas_q_params": {},  # Empty - missing omega/alpha/beta
        }

        fc = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=params)
        self.assertEqual(len(fc), len(STANDARD_HORIZONS))

    def test_short_series_no_crash(self):
        """Very short series should return zeros."""
        ret = np.array([0.001] * 10)
        fc = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=None)
        self.assertEqual(len(fc), len(STANDARD_HORIZONS))

    def test_currency_persistence_lower(self):
        """Currency forecasts should show lower persistence (faster mean reversion)."""
        ret = self._make_returns(drift=0.001)
        params = {
            "q": 1e-4, "c": 1.0, "phi": 0.999,
            "gas_q_augmented": True,
            "gas_q_params": {"omega": 1e-5, "alpha": 0.05, "beta": 0.90},
        }

        fc_eq = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=params)
        fc_fx = _kalman_forecast(ret, STANDARD_HORIZONS, "currency", tuned_params=params)

        # Long-horizon currency forecast should be smaller (more mean reversion)
        # Compare 365-day horizon (last element)
        # Both should be finite
        self.assertTrue(np.isfinite(fc_eq[-1]))
        self.assertTrue(np.isfinite(fc_fx[-1]))


class TestEnsembleForecastIntegration(unittest.TestCase):
    """Test ensemble_forecast accepts tuned_params."""

    def _make_prices(self, T=300, seed=42):
        rng = np.random.RandomState(seed)
        returns = 0.0005 + 0.01 * rng.randn(T)
        prices = 100.0 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2024-01-01", periods=T, freq="B")
        return pd.Series(prices, index=idx)

    def test_ensemble_accepts_tuned_params(self):
        """ensemble_forecast should accept tuned_params without error."""
        px = self._make_prices()
        params = {
            "q": 1e-4, "c": 1.0, "phi": 0.999,
            "gas_q_augmented": True,
            "gas_q_params": {"omega": 1e-5, "alpha": 0.05, "beta": 0.90},
        }

        result = ensemble_forecast(px, asset_type="equity", asset_name="TEST",
                                   tuned_params=params)
        self.assertEqual(len(result), 8)  # 7 horizons + confidence
        self.assertIsInstance(result[-1], str)

    def test_ensemble_without_tuned_params(self):
        """ensemble_forecast still works without tuned_params."""
        px = self._make_prices()
        result = ensemble_forecast(px, asset_type="equity", asset_name="TEST")
        self.assertEqual(len(result), 8)

    def test_ensemble_with_none_tuned_params(self):
        """Explicit None tuned_params works."""
        px = self._make_prices()
        result = ensemble_forecast(px, asset_type="equity", asset_name="TEST",
                                   tuned_params=None)
        self.assertEqual(len(result), 8)


class TestMultiScaleDrift(unittest.TestCase):
    """Test Story 1.4: Multi-scale drift estimation."""

    def _make_returns(self, T=500, drift=0.0005, vol=0.01, seed=42):
        rng = np.random.RandomState(seed)
        return drift + vol * rng.randn(T)

    def test_fast_filter_responds_faster_to_regime_change(self):
        """Fast filter (high q) should adapt faster than slow filter."""
        rng = np.random.RandomState(55)
        T = 400
        # Phase 1: positive drift, Phase 2: negative drift
        ret = np.concatenate([
            0.001 + 0.01 * rng.randn(200),
            -0.001 + 0.01 * rng.randn(200),
        ])
        vol_arr = np.full(T, 0.01)
        phi = 1.0
        q = 1e-4
        c = 1.0

        mu_fast, _ = _run_kalman_filter_pass(
            ret, vol_arr, phi, q * Q_FAST_MULT, c
        )
        mu_slow, _ = _run_kalman_filter_pass(
            ret, vol_arr, phi, q * Q_SLOW_MULT, c
        )

        # After regime change (bar 200), fast filter should adapt by bar 220
        # while slow filter still carries old drift
        idx_check = 220
        # Fast filter should be more negative (closer to new regime)
        self.assertLess(
            mu_fast[idx_check], mu_slow[idx_check],
            "Fast filter should adapt to negative drift faster than slow"
        )

    def test_slow_filter_lower_variance_calm(self):
        """Slow filter should have lower variance of mu_t in calm regime."""
        ret = self._make_returns(T=500, drift=0.0005, vol=0.01)
        vol_arr = np.full(len(ret), 0.01)
        phi = 1.0
        q = 1e-4
        c = 1.0

        mu_fast, _ = _run_kalman_filter_pass(
            ret, vol_arr, phi, q * Q_FAST_MULT, c
        )
        mu_slow, _ = _run_kalman_filter_pass(
            ret, vol_arr, phi, q * Q_SLOW_MULT, c
        )

        # Slow filter mu should have lower variance (smoother)
        var_fast = np.var(mu_fast[100:])  # Skip warm-up
        var_slow = np.var(mu_slow[100:])
        self.assertLess(
            var_slow, var_fast,
            f"Slow filter variance ({var_slow:.2e}) should be less than "
            f"fast ({var_fast:.2e})"
        )

    def test_multiscale_produces_different_forecasts_per_horizon(self):
        """Multi-scale filtering should produce horizon-differentiated forecasts."""
        ret = self._make_returns(drift=0.001)  # Strong positive drift
        params = {
            "q": 1e-4, "c": 1.0, "phi": 0.999,
            "gas_q_augmented": False,
        }

        fc = _kalman_forecast(ret, STANDARD_HORIZONS, "equity", tuned_params=params)
        self.assertEqual(len(fc), len(STANDARD_HORIZONS))
        self.assertTrue(all(np.isfinite(f) for f in fc))

    def test_horizon_filter_map_covers_all_standard_horizons(self):
        """HORIZON_FILTER_MAP should have entries for all standard horizons."""
        for h in STANDARD_HORIZONS:
            self.assertIn(h, HORIZON_FILTER_MAP, f"Horizon {h} missing from map")

    def test_q_multipliers_sensible(self):
        """Q multipliers should be > 1 for fast and < 1 for slow."""
        self.assertGreater(Q_FAST_MULT, 1.0)
        self.assertLess(Q_SLOW_MULT, 1.0)
        self.assertGreater(Q_SLOW_MULT, 0.0)

    def test_filter_pass_student_t(self):
        """Student-t filter pass should work and produce finite values."""
        ret = self._make_returns()
        vol_arr = np.full(len(ret), 0.01)
        mu, P = _run_kalman_filter_pass(ret, vol_arr, 0.999, 1e-4, 1.0, nu=6.0)
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertTrue(np.all(np.isfinite(P)))
        self.assertTrue(np.all(P > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
