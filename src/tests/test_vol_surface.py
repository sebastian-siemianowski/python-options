"""
Test Story 8.6: Volatility Surface Integration.

Validates:
  1. Skew ratio >1 when put IV > call IV
  2. Term structure slope positive in contango
  3. IV rank percentile
  4. HIGH_FEAR classification
  5. COMPLACENT classification
  6. Forecast adjustment widens sigma in HIGH_FEAR
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.vol_surface import (
    compute_skew_ratio,
    compute_term_structure_slope,
    compute_iv_rank,
    compute_iv_context,
    adjust_forecast_with_iv,
)


class TestVolSurface(unittest.TestCase):
    """Tests for volatility surface integration."""

    def test_skew_ratio_fear(self):
        """Put IV > call IV -> skew > 1."""
        skew = compute_skew_ratio(put_iv=0.30, call_iv=0.22)
        self.assertGreater(skew, 1.0)

    def test_skew_ratio_symmetric(self):
        """Equal IVs -> skew = 1."""
        skew = compute_skew_ratio(put_iv=0.25, call_iv=0.25)
        self.assertAlmostEqual(skew, 1.0)

    def test_term_structure_contango(self):
        """Long IV > short IV -> positive slope."""
        slope = compute_term_structure_slope(short_iv=0.20, long_iv=0.25)
        self.assertGreater(slope, 0)

    def test_term_structure_backwardation(self):
        """Short IV > long IV -> negative slope (fear)."""
        slope = compute_term_structure_slope(short_iv=0.30, long_iv=0.22)
        self.assertLess(slope, 0)

    def test_iv_rank_high(self):
        """Current IV at top -> rank near 1.0."""
        hist = np.array([0.15, 0.18, 0.20, 0.22, 0.25])
        rank = compute_iv_rank(0.30, hist)
        self.assertAlmostEqual(rank, 1.0)

    def test_iv_rank_low(self):
        """Current IV at bottom -> rank near 0.0."""
        hist = np.array([0.20, 0.25, 0.30, 0.35])
        rank = compute_iv_rank(0.15, hist)
        self.assertAlmostEqual(rank, 0.0)

    def test_high_fear_classification(self):
        """High rank + high skew -> HIGH_FEAR."""
        hist = np.linspace(0.15, 0.25, 100)
        ctx = compute_iv_context("SPY", put_iv=0.35, call_iv=0.25,
                                  short_iv=0.30, long_iv=0.28,
                                  current_iv=0.32, historical_iv=hist)
        self.assertEqual(ctx.iv_signal, "HIGH_FEAR")

    def test_complacent_classification(self):
        """Low rank + low skew -> COMPLACENT."""
        hist = np.linspace(0.15, 0.35, 100)
        ctx = compute_iv_context("SPY", put_iv=0.14, call_iv=0.16,
                                  short_iv=0.15, long_iv=0.18,
                                  current_iv=0.14, historical_iv=hist)
        self.assertEqual(ctx.iv_signal, "COMPLACENT")

    def test_high_fear_widens_sigma(self):
        """HIGH_FEAR adjustment widens sigma."""
        hist = np.linspace(0.15, 0.25, 100)
        ctx = compute_iv_context("SPY", put_iv=0.35, call_iv=0.25,
                                  short_iv=0.30, long_iv=0.28,
                                  current_iv=0.32, historical_iv=hist)
        
        result = adjust_forecast_with_iv(1.0, 0.01, 0.8, ctx)
        self.assertGreater(result["sigma"], 0.01)
        self.assertLess(result["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
