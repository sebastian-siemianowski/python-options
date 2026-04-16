"""
Test Story 2.8: Forecast Caching and Staleness Detection.

Validates:
  1. Timestamp recorded after ensemble_forecast call
  2. Staleness computation returns correct structure
  3. Old data flagged as stale
  4. Fresh data not flagged
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import unittest
from datetime import datetime, timedelta, timezone

from decision.market_temperature import (
    ensemble_forecast,
    get_forecast_staleness,
    record_forecast_timestamp,
    _FORECAST_TIMESTAMP_CACHE,
    STALENESS_THRESHOLD_HOURS,
)


class TestForecastStaleness(unittest.TestCase):
    """Tests for forecast caching and staleness detection."""

    def test_timestamp_recorded(self):
        """ensemble_forecast records timestamp."""
        np.random.seed(42)
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 400))))
        ensemble_forecast(prices, asset_type="equity", asset_name="STALE_TEST")
        
        info = get_forecast_staleness("STALE_TEST")
        self.assertIsNotNone(info, "Staleness info should exist after forecast")
        self.assertIn("generated_at", info)
        self.assertIn("staleness_hours", info)
        self.assertIn("is_stale", info)

    def test_fresh_forecast_not_stale(self):
        """Forecast generated just now should not be stale."""
        record_forecast_timestamp("FRESH_TEST")
        info = get_forecast_staleness("FRESH_TEST")
        self.assertFalse(info["is_stale"])
        self.assertLess(info["staleness_hours"], 1.0)

    def test_old_data_flagged_stale(self):
        """Data from 5 hours ago should be flagged as stale."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=5)
        _FORECAST_TIMESTAMP_CACHE["OLD_TEST"] = {
            "generated_at": datetime.now(timezone.utc),
            "data_through": old_time,
        }
        info = get_forecast_staleness("OLD_TEST")
        self.assertTrue(info["is_stale"])
        self.assertGreater(info["staleness_hours"], 4.0)

    def test_within_threshold_not_stale(self):
        """Data from 2 hours ago should NOT be stale."""
        recent_time = datetime.now(timezone.utc) - timedelta(hours=2)
        _FORECAST_TIMESTAMP_CACHE["RECENT_TEST"] = {
            "generated_at": datetime.now(timezone.utc),
            "data_through": recent_time,
        }
        info = get_forecast_staleness("RECENT_TEST")
        self.assertFalse(info["is_stale"])

    def test_nonexistent_asset_returns_none(self):
        """Unknown asset returns None."""
        info = get_forecast_staleness("NONEXISTENT_ASSET_XYZ")
        self.assertIsNone(info)

    def test_staleness_structure(self):
        """Staleness info has correct keys and types."""
        record_forecast_timestamp("STRUCT_TEST")
        info = get_forecast_staleness("STRUCT_TEST")
        self.assertIsInstance(info["generated_at"], str)
        self.assertIsInstance(info["data_through"], str)
        self.assertIsInstance(info["staleness_hours"], float)
        self.assertIsInstance(info["is_stale"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
