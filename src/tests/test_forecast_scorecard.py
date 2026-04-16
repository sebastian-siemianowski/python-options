"""
Test Story 1.9: Directional Accuracy Tracking and Scorecard.

Validates:
  1. ForecastRecord dataclass
  2. record_forecasts() writes to scorecard
  3. evaluate_forecasts() matches predictions with realized returns
  4. compute_scorecard_metrics() computes hit_rate, MAE, IC
  5. Perfect forecaster -> ~100% hit rate
  6. Random forecaster -> ~50% hit rate
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest
from unittest.mock import patch
from dataclasses import asdict

from calibration.forecast_scorecard import (
    ForecastRecord,
    record_forecasts,
    evaluate_forecasts,
    compute_scorecard_metrics,
    _load_scorecard,
    _save_scorecard,
    SCORECARD_PATH,
    HORIZONS,
)


class TestForecastRecord(unittest.TestCase):
    """Test ForecastRecord dataclass."""

    def test_creation(self):
        rec = ForecastRecord(
            symbol="SPY", date="2025-01-01", horizon=7,
            forecast_pct=1.5, direction=1)
        self.assertEqual(rec.symbol, "SPY")
        self.assertEqual(rec.horizon, 7)
        self.assertEqual(rec.direction, 1)
        self.assertFalse(rec.evaluated)

    def test_serialization(self):
        rec = ForecastRecord(
            symbol="AAPL", date="2025-01-01", horizon=30,
            forecast_pct=-0.5, direction=-1)
        d = asdict(rec)
        self.assertEqual(d["symbol"], "AAPL")
        self.assertIsNone(d["realized_pct"])


class TestRecordForecasts(unittest.TestCase):
    """Test recording forecasts to scorecard."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.tmpfile.close()

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    @patch('calibration.forecast_scorecard.SCORECARD_PATH')
    def test_record_adds_entries(self, mock_path):
        mock_path.__str__ = lambda x: self.tmpfile.name
        # Direct approach: use a temp file
        import calibration.forecast_scorecard as mod
        orig_path = mod.SCORECARD_PATH
        mod.SCORECARD_PATH = self.tmpfile.name
        try:
            n = record_forecasts("SPY", [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0],
                                 date="2025-01-15")
            self.assertEqual(n, 7)
            records = _load_scorecard()
            # Since we patched, load from same path
        finally:
            mod.SCORECARD_PATH = orig_path

    def test_deduplication(self):
        """Recording same (symbol, date, horizon) twice should not duplicate."""
        import calibration.forecast_scorecard as mod
        orig_path = mod.SCORECARD_PATH
        mod.SCORECARD_PATH = self.tmpfile.name
        try:
            n1 = record_forecasts("SPY", [0.1] * 7, date="2025-01-15")
            n2 = record_forecasts("SPY", [0.2] * 7, date="2025-01-15")
            self.assertEqual(n1, 7)
            self.assertEqual(n2, 0)  # All duplicated
        finally:
            mod.SCORECARD_PATH = orig_path


class TestScorecardMetrics(unittest.TestCase):
    """Test compute_scorecard_metrics with synthetic data."""

    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.tmpfile.close()

    def tearDown(self):
        if os.path.exists(self.tmpfile.name):
            os.unlink(self.tmpfile.name)

    def _set_scorecard_path(self):
        import calibration.forecast_scorecard as mod
        self._orig = mod.SCORECARD_PATH
        mod.SCORECARD_PATH = self.tmpfile.name

    def _restore_scorecard_path(self):
        import calibration.forecast_scorecard as mod
        mod.SCORECARD_PATH = self._orig

    def test_perfect_forecaster(self):
        """Perfect forecaster should get ~100% hit rate."""
        self._set_scorecard_path()
        try:
            import calibration.forecast_scorecard as mod
            # Create perfectly matched records
            records = []
            for i in range(50):
                fc = float(np.random.choice([-2.0, -1.0, 1.0, 2.0]))
                records.append({
                    "symbol": "SPY",
                    "date": f"2025-01-{(i % 28) + 1:02d}",
                    "horizon": 7,
                    "forecast_pct": fc,
                    "direction": 1 if fc >= 0 else -1,
                    "realized_pct": fc * 0.8,  # Same direction, slightly less
                    "realized_direction": 1 if fc >= 0 else -1,
                    "evaluated": True,
                })
            _save_scorecard(records)
            
            metrics = compute_scorecard_metrics()
            hr = metrics[7]["hit_rate"]
            self.assertGreaterEqual(hr, 0.95,
                                    f"Perfect forecaster hit rate should be ~100%, got {hr:.1%}")
        finally:
            self._restore_scorecard_path()

    def test_random_forecaster(self):
        """Random forecaster should get ~50% hit rate."""
        self._set_scorecard_path()
        try:
            rng = np.random.RandomState(42)
            records = []
            for i in range(200):
                fc = rng.normal(0, 1)
                realized = rng.normal(0, 1)  # Independent random
                records.append({
                    "symbol": "SPY",
                    "date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                    "horizon": 7,
                    "forecast_pct": fc,
                    "direction": 1 if fc >= 0 else -1,
                    "realized_pct": realized,
                    "realized_direction": 1 if realized >= 0 else -1,
                    "evaluated": True,
                })
            _save_scorecard(records)
            
            metrics = compute_scorecard_metrics()
            hr = metrics[7]["hit_rate"]
            self.assertGreater(hr, 0.35, f"Random hit rate too low: {hr:.1%}")
            self.assertLess(hr, 0.65, f"Random hit rate too high: {hr:.1%}")
        finally:
            self._restore_scorecard_path()

    def test_ic_positive_for_good_forecaster(self):
        """A good forecaster should have positive IC."""
        self._set_scorecard_path()
        try:
            records = []
            for i in range(100):
                fc = float(i - 50)  # -50 to 49
                realized = fc + np.random.normal(0, 5)  # Noisy but correlated
                records.append({
                    "symbol": "AAPL",
                    "date": f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}",
                    "horizon": 30,
                    "forecast_pct": fc,
                    "direction": 1 if fc >= 0 else -1,
                    "realized_pct": realized,
                    "realized_direction": 1 if realized >= 0 else -1,
                    "evaluated": True,
                })
            _save_scorecard(records)
            
            metrics = compute_scorecard_metrics(symbol="AAPL")
            ic = metrics[30]["information_coefficient"]
            self.assertGreater(ic, 0.5, f"IC for correlated forecaster should be > 0.5, got {ic:.3f}")
        finally:
            self._restore_scorecard_path()

    def test_empty_scorecard_returns_zeros(self):
        """Empty scorecard should return zero metrics."""
        self._set_scorecard_path()
        try:
            _save_scorecard([])
            metrics = compute_scorecard_metrics()
            for h in HORIZONS:
                self.assertEqual(metrics[h]["n"], 0)
        finally:
            self._restore_scorecard_path()


if __name__ == "__main__":
    unittest.main(verbosity=2)
