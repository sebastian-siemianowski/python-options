"""
Test Story 2.5: Forecast Confidence Intervals and Fan Charts.

Validates:
  1. Quantile ordering: p10 < p25 < p50 < p75 < p90
  2. Higher volatility -> wider intervals
  3. Quantiles stored and retrievable via get_forecast_quantiles
  4. All 7 horizons have quantile entries
  5. Fan chart width increases with horizon (sqrt scaling)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest

from decision.market_temperature import (
    ensemble_forecast,
    get_forecast_quantiles,
    _FORECAST_QUANTILES_CACHE,
)


class TestForecastQuantiles(unittest.TestCase):
    """Tests for forecast confidence interval computation."""

    @classmethod
    def setUpClass(cls):
        """Run ensemble_forecast once to populate quantiles cache."""
        _FORECAST_QUANTILES_CACHE.clear()
        np.random.seed(42)
        prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, 400))))
        cls.result = ensemble_forecast(prices, asset_type="equity", asset_name="TEST_QUANT")
        cls.quantiles = get_forecast_quantiles("TEST_QUANT")

    def test_quantiles_exist(self):
        """Quantiles should be computed and stored."""
        self.assertIsNotNone(self.quantiles)
        self.assertIn("horizons", self.quantiles)
        self.assertIn("quantiles", self.quantiles)

    def test_seven_horizons(self):
        """Should have quantiles for all 7 horizons."""
        self.assertEqual(len(self.quantiles["quantiles"]), 7)

    def test_quantile_ordering(self):
        """p10 < p25 < p50 < p75 < p90 for each horizon."""
        for i, q in enumerate(self.quantiles["quantiles"]):
            self.assertLess(q["p10"], q["p25"],
                            f"Horizon {i}: p10 >= p25")
            self.assertLess(q["p25"], q["p50"],
                            f"Horizon {i}: p25 >= p50")
            self.assertLess(q["p50"], q["p75"],
                            f"Horizon {i}: p50 >= p75")
            self.assertLess(q["p75"], q["p90"],
                            f"Horizon {i}: p75 >= p90")

    def test_all_keys_present(self):
        """Each quantile entry has p10, p25, p50, p75, p90."""
        for q in self.quantiles["quantiles"]:
            for key in ["p10", "p25", "p50", "p75", "p90"]:
                self.assertIn(key, q)
                self.assertIsInstance(q[key], float)

    def test_fan_width_increases_with_horizon(self):
        """Wider confidence intervals at longer horizons (sqrt scaling)."""
        q = self.quantiles["quantiles"]
        widths = [q[i]["p90"] - q[i]["p10"] for i in range(7)]
        # First horizon width should be less than last
        self.assertLess(widths[0], widths[-1],
                        "Short horizon should have narrower fan than long")

    def test_missing_asset_returns_none(self):
        """Unknown asset returns None."""
        result = get_forecast_quantiles("NONEXISTENT_ASSET_XYZ")
        self.assertIsNone(result)


class TestHighVolWiderIntervals(unittest.TestCase):
    """High-vol assets should have wider confidence intervals."""

    def test_vol_affects_width(self):
        """Higher vol returns -> wider quantile spread."""
        _FORECAST_QUANTILES_CACHE.clear()
        
        np.random.seed(42)
        # Low vol asset
        prices_low = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, 400))))
        ensemble_forecast(prices_low, asset_type="equity", asset_name="LOW_VOL_Q")
        q_low = get_forecast_quantiles("LOW_VOL_Q")
        
        # High vol asset
        prices_high = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.03, 400))))
        ensemble_forecast(prices_high, asset_type="equity", asset_name="HIGH_VOL_Q")
        q_high = get_forecast_quantiles("HIGH_VOL_Q")
        
        # Compare 30-day horizon (index 3) widths
        width_low = q_low["quantiles"][3]["p90"] - q_low["quantiles"][3]["p10"]
        width_high = q_high["quantiles"][3]["p90"] - q_high["quantiles"][3]["p10"]
        
        self.assertGreater(width_high, width_low,
                           "High-vol asset should have wider intervals")


if __name__ == "__main__":
    unittest.main(verbosity=2)
