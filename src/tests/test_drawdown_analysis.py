"""
Test Story 6.9: Drawdown Analysis and Risk Budgeting.

Validates:
  1. Max drawdown computed correctly
  2. Top-N events identified
  3. Drawdown event structure (start, trough, recovery)
  4. Ongoing drawdown (no recovery yet)
  5. Per-asset risk contribution
  6. Monotonically increasing equity has no drawdowns
  7. Empty input
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from calibration.drawdown_analysis import (
    analyze_drawdowns,
    DrawdownReport,
    DrawdownEvent,
)


class TestDrawdownAnalysis(unittest.TestCase):
    """Tests for drawdown analysis."""

    def test_max_drawdown_correct(self):
        """Max drawdown matches known value."""
        # equity: 0, 10, 5, 8, 3, 7
        equity = np.array([0.0, 10.0, 5.0, 8.0, 3.0, 7.0])
        report = analyze_drawdowns(equity)
        
        # Peak at 10, trough at 3 -> dd = 3 - 10 = -7
        self.assertAlmostEqual(report.max_drawdown, -7.0, places=5)

    def test_top_events_identified(self):
        """Multiple drawdown events identified."""
        np.random.seed(42)
        # Create equity with multiple drawdowns
        equity = np.cumsum(np.random.normal(0.01, 0.1, 300))
        
        report = analyze_drawdowns(equity, top_n=3)
        self.assertLessEqual(len(report.top_events), 3)
        
        if report.top_events:
            # First event is most severe
            self.assertEqual(
                report.top_events[0].magnitude,
                min(e.magnitude for e in report.top_events),
            )

    def test_event_structure(self):
        """Drawdown event has start, trough, recovery."""
        equity = np.array([0, 5, 10, 7, 4, 6, 11, 8, 12])
        report = analyze_drawdowns(equity)
        
        for event in report.top_events:
            self.assertIsInstance(event, DrawdownEvent)
            self.assertLessEqual(event.start_idx, event.trough_idx)
            self.assertLess(event.magnitude, 0)
            self.assertGreater(event.duration, 0)

    def test_ongoing_drawdown(self):
        """Drawdown at end has recovery_idx = None."""
        equity = np.array([0, 5, 10, 7, 4, 3])  # Ends in drawdown
        report = analyze_drawdowns(equity)
        
        # Last event should have no recovery
        ongoing = [e for e in report.top_events if e.recovery_idx is None]
        self.assertGreater(len(ongoing), 0)

    def test_per_asset_risk(self):
        """Per-asset risk contribution computed."""
        np.random.seed(42)
        equity = np.cumsum(np.random.normal(-0.01, 0.05, 100))
        
        asset_pnl = {
            "AAPL": np.random.normal(-0.005, 0.02, 100),
            "MSFT": np.random.normal(-0.01, 0.02, 100),
        }
        
        report = analyze_drawdowns(equity, asset_pnl=asset_pnl)
        self.assertGreater(len(report.per_asset_risk), 0)

    def test_no_drawdowns_monotonic(self):
        """Monotonically increasing equity has no drawdowns."""
        equity = np.arange(100, dtype=float)
        report = analyze_drawdowns(equity)
        
        self.assertEqual(report.max_drawdown, 0.0)
        self.assertEqual(len(report.top_events), 0)

    def test_empty_input(self):
        """Empty equity curve -> empty report."""
        report = analyze_drawdowns(np.array([]))
        self.assertEqual(len(report.top_events), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
