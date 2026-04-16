"""
Test Story 1.7: Market Temperature Kalman to Use Tuned Parameters.

Validates:
  1. _load_tuned_params() loads from tune cache files
  2. ensemble_forecast() auto-loads params when asset_name is provided
  3. Fallback to EMA when tune cache is missing
  4. Different forecasts with vs without tuned params
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch

from decision.market_temperature import (
    _load_tuned_params,
    ensemble_forecast,
    STANDARD_HORIZONS,
)


def _make_price_series(n=400, base=100.0, daily_return=0.0005, seed=42):
    """Create a synthetic price series."""
    rng = np.random.RandomState(seed)
    log_returns = daily_return + rng.normal(0, 0.015, n)
    log_prices = np.log(base) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


class TestLoadTunedParams(unittest.TestCase):
    """Test _load_tuned_params() from tune cache."""

    def test_loads_spy_params(self):
        """SPY tune cache should exist and load correctly."""
        params = _load_tuned_params("SPY")
        if params is None:
            self.skipTest("SPY tune cache not available")
        self.assertIn("q", params)
        self.assertIn("c", params)
        self.assertIn("phi", params)
        self.assertIsInstance(params["q"], float)
        self.assertIsInstance(params["c"], float)
        self.assertIsInstance(params["phi"], float)
        self.assertGreater(params["q"], 0)
        self.assertGreater(params["c"], 0)

    def test_missing_asset_returns_none(self):
        """Non-existent asset should return None gracefully."""
        params = _load_tuned_params("ZZZZZ_FAKE_ASSET")
        self.assertIsNone(params)

    def test_nu_loaded_when_available(self):
        """If tune cache has nu, it should be in params."""
        params = _load_tuned_params("SPY")
        if params is None:
            self.skipTest("SPY tune cache not available")
        # SPY should have nu (Student-t models are commonly best)
        if "nu" in params:
            self.assertIsInstance(params["nu"], float)
            self.assertGreater(params["nu"], 2.0)


class TestAutoLoadIntegration(unittest.TestCase):
    """Test that ensemble_forecast auto-loads tuned params."""

    def test_asset_name_triggers_loading(self):
        """Passing asset_name should trigger tune cache loading."""
        prices = _make_price_series(400)
        # With asset_name="SPY" -- will auto-load if cache exists
        result_named = ensemble_forecast(prices, horizons=STANDARD_HORIZONS,
                                         asset_name="SPY")
        # Without asset_name
        result_unknown = ensemble_forecast(prices, horizons=STANDARD_HORIZONS,
                                           asset_name="unknown")
        
        # Both should produce valid 8-element tuples
        self.assertEqual(len(result_named), 8)
        self.assertEqual(len(result_unknown), 8)

    def test_explicit_params_override_cache(self):
        """Explicit tuned_params should be used even if asset_name is given."""
        prices = _make_price_series(400)
        custom_params = {"q": 1e-3, "c": 1.0, "phi": 0.99}
        result = ensemble_forecast(prices, horizons=STANDARD_HORIZONS,
                                   asset_name="SPY", tuned_params=custom_params)
        self.assertEqual(len(result), 8)
        # Should not crash and should produce valid output
        for i in range(7):
            self.assertTrue(np.isfinite(result[i]))

    def test_tuned_vs_ema_different_forecasts(self):
        """Forecasts with tuned params should differ from EMA fallback."""
        prices = _make_price_series(400, seed=77)
        tuned = {"q": 5e-5, "c": 1.2, "phi": 0.998, "nu": 8.0}
        result_tuned = ensemble_forecast(prices, horizons=STANDARD_HORIZONS,
                                         asset_name="unknown", tuned_params=tuned)
        result_ema = ensemble_forecast(prices, horizons=STANDARD_HORIZONS,
                                       asset_name="unknown", tuned_params=None)
        
        # At least one horizon should differ
        diffs = [abs(result_tuned[i] - result_ema[i]) for i in range(7)]
        max_diff = max(diffs)
        self.assertGreater(max_diff, 0.0,
                           "Tuned params should produce different forecasts than EMA")


class TestFallbackEMA(unittest.TestCase):
    """Test EMA fallback when tuned params unavailable."""

    def test_no_params_uses_ema(self):
        """Without tuned_params, should fall back to EMA path."""
        prices = _make_price_series(400)
        result = ensemble_forecast(prices, horizons=STANDARD_HORIZONS,
                                   asset_name="unknown", tuned_params=None)
        self.assertEqual(len(result), 8)
        # Should still produce a valid confidence string (may include tag)
        base_label = result[7].split(' [')[0] if ' [' in result[7] else result[7]
        self.assertIn(base_label, ["High", "Medium", "Low", "Contested"])

    def test_empty_params_uses_ema(self):
        """Empty dict should fall back to EMA path."""
        prices = _make_price_series(400)
        result = ensemble_forecast(prices, horizons=STANDARD_HORIZONS,
                                   asset_name="unknown", tuned_params={})
        self.assertEqual(len(result), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
