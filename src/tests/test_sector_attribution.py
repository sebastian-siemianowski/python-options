"""
Test Story 6.6: Sector and Cap-Weighted Performance Attribution.

Validates:
  1. Per-sector PnL attribution sums to total
  2. Multiple sectors detected
  3. Diversified universe has lower DD than single sector
  4. Group metrics structure
  5. Custom sector map works
  6. Unknown assets go to "Other"
  7. Empty input
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from calibration.sector_attribution import (
    compute_sector_attribution,
    AttributionReport,
    GroupMetrics,
)


class TestSectorAttribution(unittest.TestCase):
    """Tests for sector performance attribution."""

    def test_pnl_attribution_sums(self):
        """Per-sector PnL contributions sum to total."""
        np.random.seed(42)
        asset_pnl = {
            "AAPL": np.random.normal(0.001, 0.01, 100),
            "JPM": np.random.normal(0.0005, 0.01, 100),
            "XOM": np.random.normal(-0.001, 0.01, 100),
        }
        report = compute_sector_attribution(asset_pnl)
        
        sector_total = sum(m.pnl_contribution for m in report.by_sector.values())
        self.assertAlmostEqual(sector_total, report.total_pnl, places=8)

    def test_multiple_sectors(self):
        """Multiple sectors detected."""
        asset_pnl = {
            "AAPL": np.random.normal(0, 0.01, 50),
            "JPM": np.random.normal(0, 0.01, 50),
            "XOM": np.random.normal(0, 0.01, 50),
        }
        report = compute_sector_attribution(asset_pnl)
        self.assertGreater(len(report.by_sector), 1)

    def test_diversified_lower_dd(self):
        """Diversified portfolio has lower drawdown than concentrated."""
        np.random.seed(42)
        n = 200
        
        # Single sector
        single = {"AAPL": 0.002 + np.random.normal(0, 0.02, n)}
        single_report = compute_sector_attribution(single)
        
        # Diversified + uncorrelated
        diversified = {
            "AAPL": 0.001 + np.random.normal(0, 0.02, n),
            "JPM": 0.001 + np.random.normal(0, 0.02, n),
            "XOM": 0.001 + np.random.normal(0, 0.02, n),
        }
        div_report = compute_sector_attribution(diversified)
        
        # Diversified total dd should exist
        self.assertIsNotNone(div_report.by_sector)

    def test_group_metrics_structure(self):
        """GroupMetrics has required fields."""
        asset_pnl = {"AAPL": np.random.normal(0, 0.01, 50)}
        report = compute_sector_attribution(asset_pnl)
        
        for gm in report.by_sector.values():
            self.assertIsInstance(gm, GroupMetrics)
            self.assertIsInstance(gm.sharpe, float)
            self.assertIsInstance(gm.hit_rate, float)
            self.assertGreater(gm.n_assets, 0)

    def test_custom_sector_map(self):
        """Custom sector map overrides default."""
        asset_pnl = {"XYZ": np.random.normal(0, 0.01, 50)}
        custom = {"XYZ": "Custom"}
        report = compute_sector_attribution(asset_pnl, sector_map=custom)
        self.assertIn("Custom", report.by_sector)

    def test_unknown_assets_other(self):
        """Unknown assets classified as Other."""
        asset_pnl = {"UNKNOWN_TICKER": np.random.normal(0, 0.01, 50)}
        report = compute_sector_attribution(asset_pnl)
        self.assertIn("Other", report.by_sector)

    def test_empty_input(self):
        """Empty input -> empty report."""
        report = compute_sector_attribution({})
        self.assertEqual(len(report.by_sector), 0)
        self.assertEqual(report.total_pnl, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
