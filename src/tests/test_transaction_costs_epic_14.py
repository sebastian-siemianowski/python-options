"""
Test Suite for Epic 14: Walk-Forward Backtest with Transaction Costs
====================================================================

Story 14.1: Realistic Transaction Cost Model
Story 14.2: Turnover-Penalized Signal Generation
Story 14.3: Optimal Rebalancing Frequency per Asset Class
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.transaction_costs import (
    # Story 14.1
    TransactionCostResult,
    transaction_cost,
    get_spread_bps,
    compute_cost_adjusted_returns,
    SPREAD_BPS, IMPACT_COEFFICIENT,
    # Story 14.2
    turnover_filter,
    turnover_filter_array,
    compute_break_even_threshold,
    compute_turnover,
    compute_turnover_reduction,
    DEAD_ZONE_MULTIPLIER,
    # Story 14.3
    RebalanceResult,
    optimal_rebalance_freq,
    DEFAULT_FREQ_OPTIONS,
)


# ===================================================================
# Story 14.1 Tests: Transaction Cost Model
# ===================================================================

class TestTransactionCost(unittest.TestCase):
    """Test transaction_cost()."""

    def test_returns_result(self):
        result = transaction_cost(100.0, 1000, spread_bps=2.0)
        self.assertIsInstance(result, TransactionCostResult)

    def test_spread_cost_formula(self):
        """Spread cost = 2 * half_spread * trade_value."""
        price, shares, bps = 100.0, 100, 5.0
        result = transaction_cost(price, shares, spread_bps=bps, adv=1e9)
        # trade_value = 10000, round-trip spread = 2*5/10000 = 0.001
        expected_spread = 10000 * 5.0 / 10000 * 2.0
        self.assertAlmostEqual(result.spread_cost, expected_spread, places=5)

    def test_impact_positive(self):
        result = transaction_cost(100.0, 10000, spread_bps=2.0, adv=100000, daily_vol=0.02)
        self.assertGreater(result.impact_cost, 0)

    def test_total_is_sum(self):
        result = transaction_cost(100.0, 1000, spread_bps=5.0, adv=1e6)
        self.assertAlmostEqual(result.total_cost, result.spread_cost + result.impact_cost, places=10)

    def test_zero_shares(self):
        result = transaction_cost(100.0, 0, spread_bps=2.0)
        self.assertAlmostEqual(result.total_cost, 0.0)

    def test_zero_price(self):
        result = transaction_cost(0.0, 1000, spread_bps=2.0)
        self.assertAlmostEqual(result.total_cost, 0.0)

    def test_larger_trade_higher_cost(self):
        r1 = transaction_cost(100.0, 100, spread_bps=2.0, adv=1e6)
        r2 = transaction_cost(100.0, 10000, spread_bps=2.0, adv=1e6)
        self.assertGreater(r2.total_cost, r1.total_cost)

    def test_larger_spread_higher_cost(self):
        r1 = transaction_cost(100.0, 1000, spread_bps=2.0, adv=1e6)
        r2 = transaction_cost(100.0, 1000, spread_bps=15.0, adv=1e6)
        self.assertGreater(r2.spread_cost, r1.spread_cost)

    def test_cost_bps_reasonable(self):
        result = transaction_cost(100.0, 100, spread_bps=5.0, adv=1e6)
        self.assertGreater(result.cost_bps, 0)
        self.assertLess(result.cost_bps, 1000)  # Not absurd

    def test_negative_shares_abs(self):
        r1 = transaction_cost(100.0, 1000, spread_bps=5.0, adv=1e6)
        r2 = transaction_cost(100.0, -1000, spread_bps=5.0, adv=1e6)
        self.assertAlmostEqual(r1.total_cost, r2.total_cost)


class TestGetSpreadBps(unittest.TestCase):
    """Test get_spread_bps()."""

    def test_large_cap(self):
        self.assertAlmostEqual(get_spread_bps("large_cap"), 2.0)

    def test_small_cap(self):
        self.assertAlmostEqual(get_spread_bps("small_cap"), 15.0)

    def test_crypto(self):
        self.assertAlmostEqual(get_spread_bps("crypto"), 10.0)

    def test_metals(self):
        self.assertAlmostEqual(get_spread_bps("metals"), 3.0)

    def test_unknown_default(self):
        self.assertAlmostEqual(get_spread_bps("unknown"), 5.0)


class TestCostAdjustedReturns(unittest.TestCase):
    """Test compute_cost_adjusted_returns()."""

    def test_output_shape(self):
        r = np.random.default_rng(42).normal(0, 0.01, 100)
        p = np.full(100, 0.5)
        net = compute_cost_adjusted_returns(r, p, spread_bps=5.0)
        self.assertEqual(len(net), 100)

    def test_no_turnover_minimal_cost(self):
        """Constant position -> only entry cost at t=0."""
        r = np.full(50, 0.001)
        p = np.full(50, 0.3)
        net = compute_cost_adjusted_returns(r, p, spread_bps=5.0)
        # Only first period has cost (entry)
        self.assertLess(net[0], r[0])
        # Rest should be equal (no turnover)
        np.testing.assert_allclose(net[1:], r[1:], atol=1e-15)

    def test_high_turnover_reduces_returns(self):
        """Flipping positions every day incurs heavy costs."""
        r = np.full(50, 0.001)
        p = np.array([0.5 if i % 2 == 0 else -0.5 for i in range(50)])
        net = compute_cost_adjusted_returns(r, p, spread_bps=5.0)
        self.assertLess(np.mean(net), np.mean(r))

    def test_larger_spread_more_cost(self):
        r = np.random.default_rng(42).normal(0, 0.01, 50)
        p = np.random.default_rng(43).uniform(-0.5, 0.5, 50)
        net2 = compute_cost_adjusted_returns(r, p, spread_bps=2.0)
        net15 = compute_cost_adjusted_returns(r, p, spread_bps=15.0)
        self.assertGreater(np.mean(net2), np.mean(net15))


# ===================================================================
# Story 14.2 Tests: Turnover-Penalized Signal Generation
# ===================================================================

class TestTurnoverFilter(unittest.TestCase):
    """Test turnover_filter()."""

    def test_large_change_passes(self):
        result = turnover_filter(0.5, 0.0, cost_threshold=0.01)
        self.assertAlmostEqual(result, 0.5)

    def test_small_change_suppressed(self):
        result = turnover_filter(0.501, 0.5, cost_threshold=0.01)
        self.assertAlmostEqual(result, 0.5)

    def test_exact_threshold(self):
        result = turnover_filter(0.51, 0.5, cost_threshold=0.01)
        self.assertAlmostEqual(result, 0.51)

    def test_zero_threshold_passes_all(self):
        result = turnover_filter(0.5, 0.49, cost_threshold=0.0)
        self.assertAlmostEqual(result, 0.5)


class TestTurnoverFilterArray(unittest.TestCase):
    """Test turnover_filter_array()."""

    def test_output_shape(self):
        signals = np.random.default_rng(42).uniform(-1, 1, 100)
        filtered = turnover_filter_array(signals, cost_threshold=0.1)
        self.assertEqual(len(filtered), 100)

    def test_reduces_turnover(self):
        rng = np.random.default_rng(42)
        signals = rng.uniform(-0.5, 0.5, 200)
        filtered = turnover_filter_array(signals, cost_threshold=0.2)
        raw_to = compute_turnover(signals)
        filt_to = compute_turnover(filtered)
        self.assertLess(filt_to, raw_to)

    def test_first_element_preserved(self):
        signals = np.array([0.5, 0.51, 0.49, 0.8])
        filtered = turnover_filter_array(signals, cost_threshold=0.1)
        self.assertAlmostEqual(filtered[0], 0.5)

    def test_large_changes_pass(self):
        signals = np.array([0.0, 0.5, -0.5, 0.5])
        filtered = turnover_filter_array(signals, cost_threshold=0.1)
        np.testing.assert_allclose(filtered, signals)

    def test_empty_array(self):
        signals = np.array([])
        filtered = turnover_filter_array(signals, cost_threshold=0.1)
        self.assertEqual(len(filtered), 0)


class TestBreakEvenThreshold(unittest.TestCase):
    """Test compute_break_even_threshold()."""

    def test_spread_only(self):
        """Break-even = 2 * spread (round trip)."""
        threshold = compute_break_even_threshold(5.0)
        self.assertAlmostEqual(threshold, 10.0 / 10000.0)

    def test_with_impact(self):
        threshold = compute_break_even_threshold(5.0, impact_bps=3.0)
        self.assertAlmostEqual(threshold, 13.0 / 10000.0)

    def test_zero_spread(self):
        threshold = compute_break_even_threshold(0.0)
        self.assertAlmostEqual(threshold, 0.0)


class TestComputeTurnover(unittest.TestCase):
    """Test compute_turnover()."""

    def test_constant_position(self):
        positions = np.full(50, 0.5)
        to = compute_turnover(positions)
        self.assertAlmostEqual(to, 0.0)

    def test_flipping_positions(self):
        positions = np.array([0.5, -0.5, 0.5, -0.5])
        to = compute_turnover(positions)
        self.assertAlmostEqual(to, 1.0)  # |delta| = 1.0 every step

    def test_single_element(self):
        to = compute_turnover(np.array([0.5]))
        self.assertAlmostEqual(to, 0.0)


class TestTurnoverReduction(unittest.TestCase):
    """Test compute_turnover_reduction()."""

    def test_half_turnover(self):
        raw = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        filt = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        reduction = compute_turnover_reduction(raw, filt)
        self.assertGreater(reduction, 0.0)
        self.assertLess(reduction, 1.0)

    def test_no_reduction(self):
        raw = np.array([0.0, 1.0, 0.0])
        reduction = compute_turnover_reduction(raw, raw)
        self.assertAlmostEqual(reduction, 0.0)

    def test_full_reduction(self):
        raw = np.array([0.0, 1.0, 0.0])
        filt = np.full(3, 0.0)
        reduction = compute_turnover_reduction(raw, filt)
        # filt turnover = 0, raw > 0, so reduction = 1.0
        self.assertAlmostEqual(reduction, 1.0)


# ===================================================================
# Story 14.3 Tests: Optimal Rebalancing Frequency
# ===================================================================

class TestOptimalRebalanceFreq(unittest.TestCase):
    """Test optimal_rebalance_freq()."""

    def test_returns_result(self):
        n = 400
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        signals = np.sign(rng.normal(0, 1, n)) * 0.3
        result = optimal_rebalance_freq(returns, signals, spread_bps=5.0)
        self.assertIsInstance(result, RebalanceResult)

    def test_optimal_in_options(self):
        n = 400
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        signals = np.sign(rng.normal(0, 1, n)) * 0.3
        result = optimal_rebalance_freq(returns, signals, spread_bps=5.0)
        self.assertIn(result.optimal_freq, DEFAULT_FREQ_OPTIONS)

    def test_n_folds_positive(self):
        n = 400
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        signals = np.sign(rng.normal(0, 1, n)) * 0.3
        result = optimal_rebalance_freq(returns, signals, spread_bps=5.0)
        self.assertGreater(result.n_folds, 0)

    def test_sharpes_length(self):
        n = 400
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        signals = np.sign(rng.normal(0, 1, n)) * 0.3
        result = optimal_rebalance_freq(returns, signals, spread_bps=5.0)
        self.assertEqual(len(result.net_sharpes), len(DEFAULT_FREQ_OPTIONS))
        self.assertEqual(len(result.gross_sharpes), len(DEFAULT_FREQ_OPTIONS))

    def test_high_cost_prefers_less_frequent(self):
        """Very high costs should favor less frequent rebalancing."""
        n = 500
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        # Noisy signals that flip often
        signals = rng.choice([-0.3, 0.3], n)
        result_low = optimal_rebalance_freq(returns, signals, spread_bps=2.0)
        result_high = optimal_rebalance_freq(returns, signals, spread_bps=50.0)
        # High cost should prefer equal or less frequent rebalancing
        self.assertGreaterEqual(result_high.optimal_freq, result_low.optimal_freq)

    def test_custom_freq_options(self):
        n = 400
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0005, 0.01, n)
        signals = np.sign(rng.normal(0, 1, n)) * 0.3
        result = optimal_rebalance_freq(returns, signals, freq_options=[1, 5, 21])
        self.assertIn(result.optimal_freq, [1, 5, 21])

    def test_short_data_fallback(self):
        returns = np.array([0.01, -0.01, 0.005])
        signals = np.array([0.3, -0.3, 0.3])
        result = optimal_rebalance_freq(returns, signals)
        self.assertEqual(result.n_folds, 0)
        self.assertEqual(result.optimal_freq, 1)


# ===================================================================
# Constants Tests
# ===================================================================

class TestEpic14Constants(unittest.TestCase):
    """Test constant values."""

    def test_spread_bps_values(self):
        self.assertAlmostEqual(SPREAD_BPS["large_cap"], 2.0)
        self.assertAlmostEqual(SPREAD_BPS["mid_cap"], 5.0)
        self.assertAlmostEqual(SPREAD_BPS["small_cap"], 15.0)
        self.assertAlmostEqual(SPREAD_BPS["crypto"], 10.0)
        self.assertAlmostEqual(SPREAD_BPS["metals"], 3.0)

    def test_impact_coefficient(self):
        self.assertAlmostEqual(IMPACT_COEFFICIENT, 0.1)

    def test_default_freq_options(self):
        self.assertEqual(DEFAULT_FREQ_OPTIONS, [1, 3, 5, 10, 21])


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic14EdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_zero_adv_no_crash(self):
        result = transaction_cost(100.0, 1000, adv=0.0)
        self.assertGreaterEqual(result.total_cost, 0.0)

    def test_turnover_filter_identical_signals(self):
        result = turnover_filter(0.5, 0.5, cost_threshold=0.01)
        self.assertAlmostEqual(result, 0.5)

    def test_cost_adjusted_single_element(self):
        r = np.array([0.01])
        p = np.array([0.5])
        net = compute_cost_adjusted_returns(r, p, spread_bps=5.0)
        self.assertEqual(len(net), 1)
        self.assertLess(net[0], r[0])

    def test_rebalance_all_zero_signals(self):
        n = 400
        returns = np.random.default_rng(42).normal(0, 0.01, n)
        signals = np.zeros(n)
        result = optimal_rebalance_freq(returns, signals)
        self.assertIsInstance(result, RebalanceResult)


if __name__ == "__main__":
    unittest.main()
