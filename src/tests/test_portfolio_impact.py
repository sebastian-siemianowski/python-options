"""
Test Story 5.7: Portfolio Impact Calculation.

Validates:
  1. Equal-weight portfolio expected return
  2. Sector exposure computed
  3. Concentration warning fired
  4. Risk decomposition sorted
  5. Empty portfolio
  6. Custom allocations
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from decision.portfolio_impact import (
    compute_portfolio_impact,
    SECTOR_CONCENTRATION_WARN,
)


class TestPortfolioImpact(unittest.TestCase):
    """Tests for portfolio impact calculation."""

    def _make_signals(self):
        return {
            "AAPL": {
                "sector": "Technology",
                "horizon_forecasts": {"30": {"point_forecast_pct": 5.0}},
            },
            "MSFT": {
                "sector": "Technology",
                "horizon_forecasts": {"30": {"point_forecast_pct": 3.0}},
            },
            "JPM": {
                "sector": "Finance",
                "horizon_forecasts": {"30": {"point_forecast_pct": -2.0}},
            },
        }

    def test_equal_weight_return(self):
        """Equal weight: (5 + 3 - 2)/3 = 2.0."""
        signals = self._make_signals()
        impact = compute_portfolio_impact(signals, horizon=30)
        self.assertAlmostEqual(impact["expected_return_pct"], 2.0, places=2)

    def test_sector_exposure(self):
        """Two tech, one finance."""
        signals = self._make_signals()
        impact = compute_portfolio_impact(signals, horizon=30)
        self.assertAlmostEqual(impact["sector_exposure"]["Technology"], 2/3, places=2)
        self.assertAlmostEqual(impact["sector_exposure"]["Finance"], 1/3, places=2)

    def test_concentration_warning(self):
        """Tech at 66% > 40% threshold -> warning."""
        signals = self._make_signals()
        impact = compute_portfolio_impact(signals, horizon=30)
        self.assertEqual(len(impact["concentration_warnings"]), 1)
        self.assertIn("Technology", impact["concentration_warnings"][0])

    def test_risk_decomposition_sorted(self):
        """Sorted by contribution descending."""
        signals = self._make_signals()
        impact = compute_portfolio_impact(signals, horizon=30)
        contribs = [r["contribution_pct"] for r in impact["risk_decomposition"]]
        self.assertEqual(contribs, sorted(contribs, reverse=True))

    def test_empty_portfolio(self):
        """No signals -> zero everything."""
        impact = compute_portfolio_impact({})
        self.assertEqual(impact["expected_return_pct"], 0.0)
        self.assertEqual(impact["asset_count"], 0)

    def test_custom_allocations(self):
        """80% AAPL, 20% JPM: 0.8*5 + 0.2*(-2) = 3.6."""
        signals = {
            "AAPL": {
                "sector": "Technology",
                "horizon_forecasts": {"30": {"point_forecast_pct": 5.0}},
            },
            "JPM": {
                "sector": "Finance",
                "horizon_forecasts": {"30": {"point_forecast_pct": -2.0}},
            },
        }
        impact = compute_portfolio_impact(
            signals, allocations={"AAPL": 0.8, "JPM": 0.2}, horizon=30
        )
        self.assertAlmostEqual(impact["expected_return_pct"], 3.6, places=2)

    def test_sharpe_positive(self):
        """Positive expected return -> positive Sharpe."""
        signals = self._make_signals()
        impact = compute_portfolio_impact(signals, horizon=30)
        self.assertGreater(impact["sharpe_estimate"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
