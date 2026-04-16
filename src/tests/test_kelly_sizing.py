"""
Test Story 8.7: Kelly Criterion Position Sizing.

Validates:
  1. f* = (p*b - q) / b formula
  2. Half-Kelly scaling
  3. Per-asset 10% cap
  4. Portfolio normalization
  5. No position for negative edge
  6. Known analytic case: coin flip with 2:1 payout
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.kelly_sizing import (
    compute_kelly_fraction,
    recommend_position_size,
    normalize_portfolio,
    KellyRecommendation,
    HALF_KELLY_SCALE,
    MAX_POSITION_PCT,
)


class TestKellySizing(unittest.TestCase):
    """Tests for Kelly criterion position sizing."""

    def test_kelly_formula_known_case(self):
        """Coin flip 60/40 with 1:1 payout -> f* = 0.20."""
        # f = (p*b - q)/b = (0.6*1 - 0.4)/1 = 0.20
        f = compute_kelly_fraction(p_win=0.6, avg_win=1.0, avg_loss=1.0)
        self.assertAlmostEqual(f, 0.20, places=5)

    def test_kelly_2to1_payout(self):
        """50/50 with 2:1 payout -> f* = 0.25."""
        # f = (0.5*2 - 0.5)/2 = (1.0 - 0.5)/2 = 0.25
        f = compute_kelly_fraction(p_win=0.5, avg_win=2.0, avg_loss=1.0)
        self.assertAlmostEqual(f, 0.25, places=5)

    def test_negative_edge_zero(self):
        """Losing bet -> negative Kelly."""
        f = compute_kelly_fraction(p_win=0.4, avg_win=1.0, avg_loss=1.0)
        self.assertLess(f, 0)  # Negative edge

    def test_half_kelly_scaling(self):
        """Half-Kelly is 50% of full Kelly."""
        rec = recommend_position_size("SPY", p_win=0.6, avg_win=1.0, avg_loss=1.0)
        self.assertAlmostEqual(rec.half_kelly, rec.full_kelly * HALF_KELLY_SCALE, places=5)

    def test_per_asset_cap(self):
        """Position capped at 10%."""
        rec = recommend_position_size("SPY", p_win=0.8, avg_win=5.0, avg_loss=1.0)
        self.assertLessEqual(rec.capped_size, MAX_POSITION_PCT)

    def test_no_position_negative_edge(self):
        """Negative edge -> zero position."""
        rec = recommend_position_size("XYZ", p_win=0.3, avg_win=1.0, avg_loss=1.0)
        self.assertEqual(rec.capped_size, 0.0)

    def test_portfolio_normalization(self):
        """Portfolio normalized to 100% max exposure."""
        recs = [
            recommend_position_size(f"SYM_{i}", p_win=0.65, avg_win=2.0, avg_loss=1.0)
            for i in range(20)
        ]
        
        normalized = normalize_portfolio(recs, max_exposure=1.0)
        total = sum(abs(r.capped_size) for r in normalized)
        self.assertLessEqual(total, 1.0 + 1e-10)

    def test_edge_computation(self):
        """Edge = p*win - q*loss."""
        rec = recommend_position_size("SPY", p_win=0.6, avg_win=1.0, avg_loss=0.8)
        expected_edge = 0.6 * 1.0 - 0.4 * 0.8
        self.assertAlmostEqual(rec.edge, expected_edge, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
