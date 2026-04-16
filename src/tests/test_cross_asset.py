"""
Test Story 2.7: Cross-Asset Signal Propagation.

Validates:
  1. Rising VIX produces negative equity adjustment
  2. Safety cap prevents over-adjustment
  3. Cross-asset signals handle missing data gracefully
  4. DXY affects metals, not equities directly
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.market_temperature import (
    compute_cross_asset_adjustment,
    compute_cross_asset_signals,
    CROSS_ASSET_MAX_ADJ,
)


class TestCrossAssetAdjustment(unittest.TestCase):
    """Tests for cross-asset drift adjustment."""

    def test_rising_vix_negative_equity(self):
        """Rising VIX (positive change) -> negative adjustment for equities."""
        signals = {"VIX_change_5d": 5.0, "DXY_change_5d": 0.0, "SPY_return_20d": 0.0}
        returns = np.random.normal(0, 0.01, 100)
        adj = compute_cross_asset_adjustment(returns, signals, asset_type="equity")
        self.assertLess(adj, 0, "Rising VIX should produce negative equity adj")

    def test_falling_vix_positive_equity(self):
        """Falling VIX (negative change) -> positive adjustment for equities."""
        signals = {"VIX_change_5d": -5.0, "DXY_change_5d": 0.0, "SPY_return_20d": 0.0}
        returns = np.random.normal(0, 0.01, 100)
        adj = compute_cross_asset_adjustment(returns, signals, asset_type="equity")
        self.assertGreater(adj, 0, "Falling VIX should produce positive equity adj")

    def test_rising_dxy_negative_metals(self):
        """Rising DXY (stronger USD) -> negative adjustment for metals."""
        signals = {"VIX_change_5d": 0.0, "DXY_change_5d": 3.0, "SPY_return_20d": 0.0}
        returns = np.random.normal(0, 0.01, 100)
        adj = compute_cross_asset_adjustment(returns, signals, asset_type="metals")
        self.assertLess(adj, 0, "Rising DXY should produce negative metals adj")

    def test_safety_cap(self):
        """Extreme signals should be capped at +-5%."""
        signals = {"VIX_change_5d": 1000.0, "DXY_change_5d": 0.0, "SPY_return_20d": 0.0}
        returns = np.random.normal(0, 0.01, 100)
        adj = compute_cross_asset_adjustment(returns, signals, asset_type="equity")
        self.assertGreaterEqual(adj, -5.0)
        self.assertLessEqual(adj, 5.0)

    def test_no_signals_zero_adj(self):
        """Empty signals -> zero adjustment."""
        returns = np.random.normal(0, 0.01, 100)
        adj = compute_cross_asset_adjustment(returns, {}, asset_type="equity")
        self.assertAlmostEqual(adj, 0.0, places=6)

    def test_spy_momentum_positive(self):
        """Positive SPY momentum -> positive adjustment for equities."""
        signals = {"VIX_change_5d": 0.0, "DXY_change_5d": 0.0, "SPY_return_20d": 0.05}
        returns = np.random.normal(0, 0.01, 100)
        adj = compute_cross_asset_adjustment(returns, signals, asset_type="equity")
        self.assertGreater(adj, 0, "Positive SPY momentum -> positive equity adj")


class TestCrossAssetSignals(unittest.TestCase):
    """Tests for cross-asset signal extraction."""

    def test_missing_directory_returns_empty(self):
        """Non-existent price directory returns empty dict."""
        signals = compute_cross_asset_signals(prices_dir="/tmp/nonexistent_dir_xyz")
        self.assertIsInstance(signals, dict)
        self.assertEqual(len(signals), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
