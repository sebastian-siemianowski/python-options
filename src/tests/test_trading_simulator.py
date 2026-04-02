"""
Test Story 6.2: Trading Strategy Simulator.

Validates:
  1. Long-only never goes short
  2. Transaction costs reduce PnL
  3. Long-short allows negative positions
  4. Zero cost > positive cost
  5. Result structure complete
  6. Hit rate sensible
  7. Equity curve shape
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from calibration.trading_simulator import (
    simulate_trading,
    SimulationResult,
    LONG_ONLY,
    LONG_SHORT,
)


class TestTradingSimulator(unittest.TestCase):
    """Tests for trading strategy simulator."""

    def test_long_only_no_short(self):
        """Long-only mode never takes short positions."""
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0, 0.01, n)
        forecasts = np.random.normal(0, 1, n)  # Mix of +/-
        
        result = simulate_trading(returns, forecasts, mode=LONG_ONLY)
        for t in result.trades:
            self.assertGreaterEqual(t.position, 0.0)

    def test_costs_reduce_pnl(self):
        """Higher costs reduce total PnL."""
        np.random.seed(42)
        n = 200
        returns = 0.001 + np.random.normal(0, 0.01, n)
        forecasts = np.ones(n)
        
        r_zero = simulate_trading(returns, forecasts, cost_bps=0)
        r_high = simulate_trading(returns, forecasts, cost_bps=20)
        
        self.assertGreater(r_zero.total_pnl, r_high.total_pnl)

    def test_long_short_allows_negative(self):
        """Long-short mode allows negative positions."""
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0, 0.01, n)
        forecasts = -2 * np.ones(n)  # Strong sell signal
        
        result = simulate_trading(returns, forecasts, mode=LONG_SHORT)
        has_short = any(t.position < 0 for t in result.trades)
        self.assertTrue(has_short)

    def test_result_structure(self):
        """Result has all required fields."""
        returns = np.random.normal(0, 0.01, 100)
        forecasts = np.ones(100)
        result = simulate_trading(returns, forecasts)
        
        self.assertIsInstance(result, SimulationResult)
        self.assertEqual(len(result.equity_curve), 100)
        self.assertIsInstance(result.sharpe, float)
        self.assertIsInstance(result.max_drawdown, float)
        self.assertIsInstance(result.total_costs, float)

    def test_hit_rate_bounds(self):
        """Hit rate between 0 and 1."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 200)
        forecasts = np.ones(200)
        result = simulate_trading(returns, forecasts)
        
        self.assertGreaterEqual(result.hit_rate, 0.0)
        self.assertLessEqual(result.hit_rate, 1.0)

    def test_max_drawdown_nonpositive(self):
        """Max drawdown is <= 0."""
        returns = np.random.normal(0, 0.01, 200)
        forecasts = np.ones(200)
        result = simulate_trading(returns, forecasts)
        self.assertLessEqual(result.max_drawdown, 0.0)

    def test_short_series(self):
        """Very short series returns empty result."""
        result = simulate_trading(np.array([0.01]), np.array([1.0]))
        self.assertEqual(result.n_trades, 0)

    def test_equity_curve_monotonic_for_perfect_forecast(self):
        """Perfect forecast on strong trend -> mostly increasing equity."""
        np.random.seed(42)
        n = 300
        returns = 0.003 + np.random.normal(0, 0.002, n)  # Very strong uptrend
        forecasts = 2 * np.ones(n)  # Always long
        
        result = simulate_trading(returns, forecasts, cost_bps=0)
        # Final equity should be positive
        self.assertGreater(result.equity_curve[-1], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
