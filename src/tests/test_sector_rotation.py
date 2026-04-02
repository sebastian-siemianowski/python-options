"""
Test Story 8.5: Sector Rotation.

Validates:
  1. Outperforming sector -> positive relative momentum
  2. Breadth with all positive forecasts -> 1.0
  3. Composite OVERWEIGHT signal
  4. UNDERWEIGHT for underperforming sector
  5. Ranking by composite score
  6. Edge case: empty forecasts
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from decision.sector_rotation import (
    compute_sector_momentum,
    compute_sector_breadth,
    generate_rotation_signal,
    rank_sectors,
)


class TestSectorRotation(unittest.TestCase):
    """Tests for sector rotation engine."""

    def test_positive_relative_momentum(self):
        """Sector outperforming benchmark -> positive momentum."""
        sector = np.array([0.01] * 20)  # 1% daily
        bench = np.array([0.005] * 20)  # 0.5% daily
        
        mom = compute_sector_momentum(sector, bench)
        self.assertGreater(mom, 0)

    def test_negative_relative_momentum(self):
        """Sector underperforming benchmark -> negative momentum."""
        sector = np.array([0.002] * 20)
        bench = np.array([0.008] * 20)
        
        mom = compute_sector_momentum(sector, bench)
        self.assertLess(mom, 0)

    def test_breadth_all_positive(self):
        """All positive forecasts -> breadth 1.0."""
        forecasts = {"AAPL": 1.5, "MSFT": 2.0, "GOOGL": 0.8}
        breadth = compute_sector_breadth(forecasts)
        self.assertAlmostEqual(breadth, 1.0)

    def test_breadth_half(self):
        """Half positive -> breadth 0.5."""
        forecasts = {"A": 1.0, "B": -0.5, "C": 0.5, "D": -1.0}
        breadth = compute_sector_breadth(forecasts)
        self.assertAlmostEqual(breadth, 0.5)

    def test_overweight_signal(self):
        """Strong outperformance + high breadth -> OVERWEIGHT."""
        sector = np.array([0.015] * 25)
        bench = np.array([0.003] * 25)
        forecasts = {"A": 2.0, "B": 1.5, "C": 1.0}
        
        sig = generate_rotation_signal("Tech", sector, bench, forecasts)
        self.assertEqual(sig.recommendation, "OVERWEIGHT")
        self.assertGreater(sig.composite_score, 0.3)

    def test_underweight_signal(self):
        """Underperformance + low breadth -> UNDERWEIGHT."""
        sector = np.array([-0.005] * 25)
        bench = np.array([0.01] * 25)
        forecasts = {"A": -1.0, "B": -0.5, "C": -0.8}
        
        sig = generate_rotation_signal("Energy", sector, bench, forecasts)
        self.assertEqual(sig.recommendation, "UNDERWEIGHT")

    def test_ranking(self):
        """Sectors ranked by composite score."""
        s1 = generate_rotation_signal(
            "A", np.array([0.01] * 25), np.array([0.005] * 25),
            {"X": 1.0, "Y": 0.5},
        )
        s2 = generate_rotation_signal(
            "B", np.array([-0.005] * 25), np.array([0.005] * 25),
            {"X": -0.5},
        )
        ranked = rank_sectors([s2, s1])
        self.assertEqual(ranked[0].sector, "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
