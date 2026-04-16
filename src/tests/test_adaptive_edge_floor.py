"""test_adaptive_edge_floor.py -- Story 5.1

Validates volatility-scaled adaptive edge floor:
1. USDJPY (low vol) gets small floor (~0.02)
2. BTC (high vol) gets large floor (~0.15)
3. Formula: edge_floor = EDGE_FLOOR_Z * vol_annual / sqrt(H)
4. Clamping to [floor_min, floor_max]
5. Fallback to EDGE_FLOOR for invalid vol
"""
import os
import sys
import math
import unittest

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from decision.signals import (
    compute_adaptive_edge_floor, EDGE_FLOOR, EDGE_FLOOR_Z,
)


class TestAdaptiveEdgeFloorFormula(unittest.TestCase):

    def test_basic_formula(self):
        """edge_floor = z * vol_annual / sqrt(H)."""
        vol_daily = 0.01  # ~15.9% annualized
        H = 7
        expected = EDGE_FLOOR_Z * (vol_daily * math.sqrt(252)) / math.sqrt(H)
        result = compute_adaptive_edge_floor(vol_daily, H)
        self.assertAlmostEqual(result, expected, places=8)

    def test_horizon_scaling(self):
        """Longer horizons get smaller floor (edge_floor ~ 1/sqrt(H))."""
        vol = 0.015  # Mid-range vol for clear separation
        f7 = compute_adaptive_edge_floor(vol, 7)
        f21 = compute_adaptive_edge_floor(vol, 21)
        f252 = compute_adaptive_edge_floor(vol, 252)
        self.assertGreater(f7, f21)
        self.assertGreater(f21, f252)

    def test_vol_scaling(self):
        """Higher vol gets higher floor."""
        low_vol = 0.003   # Currency-like
        high_vol = 0.04   # Crypto-like
        H = 7
        f_low = compute_adaptive_edge_floor(low_vol, H)
        f_high = compute_adaptive_edge_floor(high_vol, H)
        self.assertGreater(f_high, f_low)


class TestUsdjpyEdgeFloor(unittest.TestCase):

    def test_usdjpy_floor_small(self):
        """USDJPY daily vol ~0.005 -> annual ~7.9% -> floor ~0.015 at H=7."""
        vol_daily = 0.005
        H = 7
        result = compute_adaptive_edge_floor(vol_daily, H)
        # 0.05 * 0.005 * sqrt(252) / sqrt(7) ~ 0.05 * 0.0794 / 2.646 ~ 0.0015
        # Actually: 0.05 * (0.005 * 15.87) / 2.646 = 0.05 * 0.03 = 0.015? No.
        # = 0.05 * 0.0794 / 2.646 = 0.0015 -- well below 0.10
        self.assertLess(result, 0.10, "USDJPY floor should be much less than 0.10")
        self.assertLess(result, 0.03, "USDJPY floor should be < 0.03")


class TestBtcEdgeFloor(unittest.TestCase):

    def test_btc_floor_large(self):
        """BTC daily vol ~0.03 -> annual ~47.6% -> floor ~0.12 at H=7."""
        vol_daily = 0.03
        H = 7
        result = compute_adaptive_edge_floor(vol_daily, H)
        self.assertGreater(result, EDGE_FLOOR, "BTC floor should exceed static 0.10")

    def test_btc_floor_higher_than_spy(self):
        """BTC floor > SPY floor at same horizon."""
        vol_btc = 0.03
        vol_spy = 0.01
        H = 21
        f_btc = compute_adaptive_edge_floor(vol_btc, H)
        f_spy = compute_adaptive_edge_floor(vol_spy, H)
        self.assertGreater(f_btc, f_spy)


class TestEdgeFloorClamping(unittest.TestCase):

    def test_minimum_floor(self):
        """Very low vol should be clamped to floor_min."""
        vol_daily = 0.0001  # Essentially zero vol
        result = compute_adaptive_edge_floor(vol_daily, 7)
        self.assertGreaterEqual(result, 0.005)

    def test_maximum_floor(self):
        """Extreme vol should be clamped to floor_max."""
        vol_daily = 1.0  # Absurd vol
        result = compute_adaptive_edge_floor(vol_daily, 1)
        self.assertLessEqual(result, 0.50)

    def test_custom_bounds(self):
        result = compute_adaptive_edge_floor(0.01, 7, floor_min=0.02, floor_max=0.03)
        self.assertGreaterEqual(result, 0.02)
        self.assertLessEqual(result, 0.03)


class TestEdgeFloorFallback(unittest.TestCase):

    def test_nan_vol_fallback(self):
        result = compute_adaptive_edge_floor(float('nan'), 7)
        self.assertEqual(result, EDGE_FLOOR)

    def test_zero_vol_fallback(self):
        result = compute_adaptive_edge_floor(0.0, 7)
        self.assertEqual(result, EDGE_FLOOR)

    def test_negative_vol_fallback(self):
        result = compute_adaptive_edge_floor(-0.01, 7)
        self.assertEqual(result, EDGE_FLOOR)

    def test_inf_vol_fallback(self):
        result = compute_adaptive_edge_floor(float('inf'), 7)
        self.assertEqual(result, EDGE_FLOOR)


class TestEdgeFloorZConstant(unittest.TestCase):

    def test_value(self):
        self.assertEqual(EDGE_FLOOR_Z, 0.65)


class TestEdgeFloorHorizonEdgeCases(unittest.TestCase):

    def test_horizon_zero(self):
        """H=0 should not crash (treated as H=1)."""
        result = compute_adaptive_edge_floor(0.01, 0)
        self.assertTrue(np.isfinite(result))

    def test_horizon_one(self):
        result = compute_adaptive_edge_floor(0.01, 1)
        # z * vol_annual / 1.0
        expected = EDGE_FLOOR_Z * 0.01 * math.sqrt(252)
        self.assertAlmostEqual(result, expected, places=6)

    def test_large_horizon(self):
        result = compute_adaptive_edge_floor(0.01, 10000)
        self.assertTrue(np.isfinite(result))
        self.assertGreater(result, 0)


class TestSignalRateDirection(unittest.TestCase):
    """Demonstrate that adaptive floor changes signal rate direction."""

    def test_currency_lower_floor(self):
        """Currency (low vol) gets floor < static EDGE_FLOOR."""
        vol_daily = 0.004  # ~6.3% annual
        H = 21
        adaptive = compute_adaptive_edge_floor(vol_daily, H)
        self.assertLess(adaptive, EDGE_FLOOR,
                        "Low-vol asset should get lower floor than static 0.10")

    def test_crypto_may_get_higher_floor(self):
        """Crypto (high vol) gets floor > static EDGE_FLOOR at short horizons."""
        vol_daily = 0.04  # ~63.5% annual
        H = 7
        adaptive = compute_adaptive_edge_floor(vol_daily, H)
        self.assertGreater(adaptive, EDGE_FLOOR,
                           "High-vol asset should exceed static floor")


if __name__ == "__main__":
    unittest.main()
