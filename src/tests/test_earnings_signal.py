"""
Test Story 8.1: Earnings Event Signal Augmentation.

Validates:
  1. Pre-earnings detection
  2. Post-earnings detection
  3. Normal (no earnings) detection
  4. Confidence reduction in pre-earnings
  5. Interval widening in pre-earnings
  6. Post-earnings amplification
  7. Historical earnings vol multiple
  8. Nearest earnings finder
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.earnings_signal import (
    detect_earnings_window,
    adjust_for_earnings,
    compute_historical_earnings_vol,
    find_nearest_earnings,
    EarningsContext,
    CONFIDENCE_REDUCTION,
    INTERVAL_WIDEN_FACTOR,
    POST_AMPLIFICATION,
)


class TestEarningsSignal(unittest.TestCase):
    """Tests for earnings event signal augmentation."""

    def test_pre_earnings_detection(self):
        """T-2 before earnings -> PRE_EARNINGS."""
        ctx = detect_earnings_window(current_date_idx=98, earnings_date_idx=100, total_days=200)
        self.assertTrue(ctx.in_pre_earnings)
        self.assertFalse(ctx.in_post_earnings)
        self.assertEqual(ctx.context_label, "PRE_EARNINGS")
        self.assertEqual(ctx.days_to_earnings, 2)

    def test_post_earnings_detection(self):
        """T+1 after earnings -> POST_EARNINGS."""
        ctx = detect_earnings_window(current_date_idx=101, earnings_date_idx=100, total_days=200)
        self.assertTrue(ctx.in_post_earnings)
        self.assertFalse(ctx.in_pre_earnings)
        self.assertEqual(ctx.context_label, "POST_EARNINGS")

    def test_normal_no_earnings(self):
        """No earnings date -> NORMAL."""
        ctx = detect_earnings_window(current_date_idx=50, earnings_date_idx=None, total_days=200)
        self.assertEqual(ctx.context_label, "NORMAL")
        self.assertFalse(ctx.in_pre_earnings)
        self.assertFalse(ctx.in_post_earnings)

    def test_normal_far_from_earnings(self):
        """Far from earnings -> NORMAL."""
        ctx = detect_earnings_window(current_date_idx=50, earnings_date_idx=100, total_days=200)
        self.assertEqual(ctx.context_label, "NORMAL")

    def test_confidence_reduction_pre_earnings(self):
        """Pre-earnings reduces confidence by configured amount."""
        ctx = EarningsContext(
            in_pre_earnings=True, in_post_earnings=False,
            days_to_earnings=2, earnings_date=None,
            context_label="PRE_EARNINGS", historical_earnings_vol=1.5,
        )
        result = adjust_for_earnings(forecast_pct=1.0, confidence=0.8, sigma=0.01, context=ctx)
        
        expected_conf = 0.8 * (1.0 - CONFIDENCE_REDUCTION)
        self.assertAlmostEqual(result["confidence"], expected_conf, places=5)

    def test_interval_widening_pre_earnings(self):
        """Pre-earnings widens sigma."""
        ctx = EarningsContext(
            in_pre_earnings=True, in_post_earnings=False,
            days_to_earnings=1, earnings_date=None,
            context_label="PRE_EARNINGS", historical_earnings_vol=2.0,
        )
        result = adjust_for_earnings(forecast_pct=1.0, confidence=0.5, sigma=0.01, context=ctx)
        
        expected_sigma = 0.01 * INTERVAL_WIDEN_FACTOR * 2.0
        self.assertAlmostEqual(result["sigma"], expected_sigma, places=8)

    def test_post_earnings_amplification(self):
        """Post-earnings amplifies forecast."""
        ctx = EarningsContext(
            in_pre_earnings=False, in_post_earnings=True,
            days_to_earnings=-1, earnings_date=None,
            context_label="POST_EARNINGS", historical_earnings_vol=1.5,
        )
        result = adjust_for_earnings(forecast_pct=2.0, confidence=0.7, sigma=0.01, context=ctx)
        
        self.assertAlmostEqual(result["forecast_pct"], 2.0 * POST_AMPLIFICATION)

    def test_historical_earnings_vol(self):
        """Earnings days are more volatile than normal days."""
        rng = np.random.default_rng(42)
        # Normal days: small vol
        returns = rng.normal(0, 0.01, 500)
        # Earnings days: big vol
        earnings_idx = [100, 200, 300, 400]
        for idx in earnings_idx:
            returns[idx] = rng.normal(0, 0.05)
        
        vol_mult = compute_historical_earnings_vol(returns, earnings_idx, window=0)
        self.assertGreater(vol_mult, 1.0)

    def test_find_nearest_earnings(self):
        """Finds nearest upcoming earnings."""
        earnings = [50, 100, 200]
        nearest = find_nearest_earnings(current_idx=95, earnings_indices=earnings)
        self.assertEqual(nearest, 100)

    def test_find_nearest_no_match(self):
        """No earnings in window -> None."""
        earnings = [200, 300]
        nearest = find_nearest_earnings(current_idx=50, earnings_indices=earnings, look_ahead=10)
        self.assertIsNone(nearest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
