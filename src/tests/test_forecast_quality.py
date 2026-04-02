"""
Test Story 6.3: Per-Asset Forecast Quality Report.

Validates:
  1. Perfect forecast -> hit rate 1.0
  2. Random forecast -> hit rate ~0.5
  3. High IC for correlated forecasts
  4. ECE well-calibrated
  5. MAE is correct
  6. Multiple assets
  7. Empty input
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from calibration.forecast_quality import (
    compute_forecast_quality,
    QualityReport,
    AssetQuality,
)


class TestForecastQuality(unittest.TestCase):
    """Tests for forecast quality reporting."""

    def test_perfect_forecast_hitrate(self):
        """Perfect sign forecast -> hit rate near 1.0."""
        np.random.seed(42)
        n = 200
        a = np.random.normal(0, 1, n)
        f = np.sign(a) * abs(np.random.normal(1, 0.1, n))  # Correct sign
        
        report = compute_forecast_quality(
            {"AAPL": a}, {"AAPL": f}, horizons=[1]
        )
        self.assertGreater(report.avg_hit_rate, 0.95)

    def test_random_forecast_hitrate(self):
        """Random forecast -> hit rate ~0.5."""
        np.random.seed(42)
        a = np.random.normal(0, 1, 500)
        f = np.random.normal(0, 1, 500)
        
        report = compute_forecast_quality(
            {"SPY": a}, {"SPY": f}, horizons=[1]
        )
        self.assertAlmostEqual(report.avg_hit_rate, 0.5, delta=0.1)

    def test_ic_positive_correlated(self):
        """Correlated forecast -> positive IC."""
        np.random.seed(42)
        n = 300
        a = np.random.normal(0, 1, n)
        f = a + np.random.normal(0, 0.3, n)  # Noisy copy
        
        report = compute_forecast_quality(
            {"TSLA": a}, {"TSLA": f}, horizons=[1]
        )
        self.assertGreater(report.avg_ic, 0.5)

    def test_mae_correct(self):
        """MAE is correct for known difference."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([1.5, 2.5, 3.5, 4.5, 5.5])  # All off by 0.5
        
        report = compute_forecast_quality(
            {"TEST": a}, {"TEST": f}, horizons=[1]
        )
        self.assertAlmostEqual(report.avg_mae, 0.5, places=3)

    def test_multiple_assets(self):
        """Multiple assets produce per-asset entries."""
        np.random.seed(42)
        n = 100
        actuals = {
            "AAPL": np.random.normal(0, 1, n),
            "MSFT": np.random.normal(0, 1, n),
        }
        forecasts = {
            "AAPL": np.random.normal(0, 1, n),
            "MSFT": np.random.normal(0, 1, n),
        }
        report = compute_forecast_quality(actuals, forecasts, horizons=[1])
        self.assertEqual(len(report.entries), 2)

    def test_empty_input(self):
        """Empty input -> empty report."""
        report = compute_forecast_quality({}, {}, horizons=[1])
        self.assertEqual(len(report.entries), 0)
        self.assertEqual(report.avg_hit_rate, 0.0)

    def test_ece_bounds(self):
        """ECE should be between 0 and 1."""
        np.random.seed(42)
        n = 200
        a = np.random.normal(0, 1, n)
        f = np.random.normal(0, 1, n)
        
        report = compute_forecast_quality(
            {"SPY": a}, {"SPY": f}, horizons=[1]
        )
        for entry in report.entries:
            self.assertGreaterEqual(entry.ece, 0.0)
            self.assertLessEqual(entry.ece, 1.0)

    def test_report_structure(self):
        """Report has correct type and fields."""
        a = np.random.normal(0, 1, 50)
        f = np.random.normal(0, 1, 50)
        report = compute_forecast_quality({"X": a}, {"X": f}, horizons=[7])
        
        self.assertIsInstance(report, QualityReport)
        self.assertGreater(len(report.entries), 0)
        self.assertIsInstance(report.entries[0], AssetQuality)
        self.assertEqual(report.entries[0].horizon, 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
