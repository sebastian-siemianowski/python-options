"""
Test Story 6.5: Regime-Specific Profitability Analysis.

Validates:
  1. Trending regime has better metrics than range
  2. All regimes present in output
  3. Regime transitions counted
  4. Best/worst regime identified
  5. Per-regime structure
  6. Empty input
  7. Single regime
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from calibration.regime_profitability import (
    compute_regime_profitability,
    RegimeProfitabilityReport,
    RegimeMetrics,
)


class TestRegimeProfitability(unittest.TestCase):
    """Tests for regime-specific profitability."""

    def test_trending_better_than_range(self):
        """Trending regime (0) profits, range regime (2) flat."""
        np.random.seed(42)
        n = 500
        pnl = np.zeros(n)
        regimes = np.zeros(n, dtype=int)
        
        # Regime 0 (LOW_VOL_TREND): profitable
        pnl[:250] = 0.003 + np.random.normal(0, 0.005, 250)
        regimes[:250] = 0
        
        # Regime 2 (LOW_VOL_RANGE): flat
        pnl[250:] = np.random.normal(0, 0.005, 250)
        regimes[250:] = 2
        
        report = compute_regime_profitability(pnl, regimes)
        trend = report.per_regime["LOW_VOL_TREND"]
        rng = report.per_regime["LOW_VOL_RANGE"]
        
        self.assertGreater(trend.sharpe, rng.sharpe)

    def test_all_regimes_present(self):
        """Multiple regimes produce multiple entries."""
        np.random.seed(42)
        n = 300
        pnl = np.random.normal(0, 0.01, n)
        regimes = np.repeat([0, 1, 2], 100)
        
        report = compute_regime_profitability(pnl, regimes)
        self.assertEqual(len(report.per_regime), 3)

    def test_transitions_counted(self):
        """Regime transitions are counted."""
        pnl = np.random.normal(0, 0.01, 10)
        regimes = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        
        report = compute_regime_profitability(pnl, regimes)
        self.assertEqual(report.regime_transitions, 4)

    def test_best_worst_identified(self):
        """Best and worst regime identified."""
        np.random.seed(42)
        n = 200
        pnl = np.zeros(n)
        regimes = np.zeros(n, dtype=int)
        
        pnl[:100] = 0.005 + np.random.normal(0, 0.002, 100)  # Best
        regimes[:100] = 0
        pnl[100:] = -0.005 + np.random.normal(0, 0.002, 100)  # Worst
        regimes[100:] = 4
        
        report = compute_regime_profitability(pnl, regimes)
        self.assertEqual(report.best_regime, "LOW_VOL_TREND")
        self.assertEqual(report.worst_regime, "CRISIS_JUMP")

    def test_per_regime_structure(self):
        """RegimeMetrics has all required fields."""
        pnl = np.random.normal(0, 0.01, 100)
        regimes = np.zeros(100, dtype=int)
        
        report = compute_regime_profitability(pnl, regimes)
        m = list(report.per_regime.values())[0]
        
        self.assertIsInstance(m, RegimeMetrics)
        self.assertIsInstance(m.sharpe, float)
        self.assertIsInstance(m.hit_rate, float)
        self.assertGreater(m.n_days, 0)

    def test_empty_input(self):
        """Empty input -> empty report."""
        report = compute_regime_profitability(
            np.array([]), np.array([], dtype=int)
        )
        self.assertEqual(len(report.per_regime), 0)

    def test_single_regime(self):
        """Single regime -> one entry."""
        pnl = np.random.normal(0, 0.01, 100)
        regimes = np.ones(100, dtype=int)
        
        report = compute_regime_profitability(pnl, regimes)
        self.assertEqual(len(report.per_regime), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
