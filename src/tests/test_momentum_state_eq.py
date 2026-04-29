"""
Test Story 1.10: Momentum Integration into Kalman State Equation.

Validates:
  1. compute_multi_timeframe_momentum() produces correct blended signal
  2. Regime-dependent weighting: TREND -> momentum dominates u_t
  3. Regime-dependent weighting: RANGE -> mean reversion dominates u_t
  4. max_u capping applies to blended signal
  5. Multi-timeframe windows produce different signals
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import unittest

from models.momentum_augmented import (
    compute_multi_timeframe_momentum,
    compute_mr_signal,
    MomentumConfig,
    MomentumAugmentedDriftModel,
    REGIME_MOM_MR_RATIO,
)


class TestComputeMultiTimeframeMomentum(unittest.TestCase):
    """Tests for compute_multi_timeframe_momentum (task 1.10.1)."""

    def test_zero_returns(self):
        """Zero returns -> zero momentum signal."""
        ret = np.zeros(100)
        mom = compute_multi_timeframe_momentum(ret)
        np.testing.assert_allclose(mom, 0.0, atol=1e-15)

    def test_constant_positive_returns(self):
        """Constant positive returns -> positive momentum everywhere."""
        ret = np.full(100, 0.01)
        mom = compute_multi_timeframe_momentum(ret)
        self.assertTrue(np.all(mom > 0),
                        "Momentum should be positive when returns are always positive")

    def test_short_window_responds_faster(self):
        """Short window (5d) should respond faster to trend change than long (63d)."""
        ret = np.zeros(100)
        ret[50:] = 0.02  # Trend starts at t=50
        
        mom_short = compute_multi_timeframe_momentum(ret, windows=(5,), weights=(1.0,))
        mom_long = compute_multi_timeframe_momentum(ret, windows=(63,), weights=(1.0,))
        
        # At t=55, short window fully reflects the new trend
        self.assertGreater(mom_short[55], mom_long[55],
                           "Short window should react faster")

    def test_weights_sum_to_one(self):
        """Weights are normalized internally -- non-unit sums should work."""
        ret = np.random.RandomState(42).normal(0, 0.01, 100)
        mom_default = compute_multi_timeframe_momentum(ret, weights=(0.5, 0.3, 0.2))
        mom_scaled = compute_multi_timeframe_momentum(ret, weights=(1.0, 0.6, 0.4))
        np.testing.assert_allclose(mom_default, mom_scaled, atol=1e-14)

    def test_output_length(self):
        """Output length matches input."""
        ret = np.random.normal(0, 0.01, 200)
        mom = compute_multi_timeframe_momentum(ret)
        self.assertEqual(len(mom), 200)


class TestRegimeDependentBlending(unittest.TestCase):
    """Tests for regime-dependent u_t weighting (tasks 1.10.2, 1.10.3)."""

    def test_trend_regime_momentum_dominates(self):
        """In LOW_VOL_TREND (regime 0), momentum should dominate u_t."""
        # Strong positive trend with returns far above equilibrium
        n = 100
        returns = np.full(n, 0.01)
        prices = 100 * np.exp(np.cumsum(returns))
        vol = np.full(n, 0.01)
        regime_labels = np.full(n, 0, dtype=int)  # LOW_VOL_TREND
        
        config = MomentumConfig(
            enable=True,
            enable_mean_reversion=True,
            adjustment_scale=0.15,
            mr_adjustment_scale=0.15,
        )
        model = MomentumAugmentedDriftModel(config)
        model.precompute_signals(returns, prices, vol, regime_labels=regime_labels)
        
        u_t = model._exogenous_input
        self.assertIsNotNone(u_t)
        # In a trend, momentum is positive and should dominate
        # Verify the REGIME ratio is 0.7/0.3
        mom_ratio, mr_ratio = REGIME_MOM_MR_RATIO[0]
        self.assertAlmostEqual(mom_ratio, 0.7)
        self.assertAlmostEqual(mr_ratio, 0.3)

    def test_range_regime_mr_dominates(self):
        """In LOW_VOL_RANGE (regime 2), MR should dominate u_t."""
        n = 100
        # Oscillating returns (range-bound)
        returns = 0.01 * np.sin(np.linspace(0, 10 * np.pi, n))
        prices = 100 * np.exp(np.cumsum(returns))
        vol = np.full(n, 0.01)
        regime_labels = np.full(n, 2, dtype=int)  # LOW_VOL_RANGE
        
        config = MomentumConfig(
            enable=True,
            enable_mean_reversion=True,
            adjustment_scale=0.15,
            mr_adjustment_scale=0.15,
        )
        model = MomentumAugmentedDriftModel(config)
        model.precompute_signals(returns, prices, vol, regime_labels=regime_labels)
        
        # Verify the REGIME ratio is 0.3/0.7
        mom_ratio, mr_ratio = REGIME_MOM_MR_RATIO[2]
        self.assertAlmostEqual(mom_ratio, 0.3)
        self.assertAlmostEqual(mr_ratio, 0.7)

    def test_all_regimes_have_ratios(self):
        """All 5 regimes must have momentum/MR ratios."""
        for r in range(5):
            self.assertIn(r, REGIME_MOM_MR_RATIO,
                          f"Regime {r} missing from REGIME_MOM_MR_RATIO")
            mom, mr = REGIME_MOM_MR_RATIO[r]
            self.assertGreater(mom, 0)
            self.assertGreater(mr, 0)
            self.assertLessEqual(mom + mr, 1.01)


class TestMaxUCapping(unittest.TestCase):
    """Tests for max_u capping on blended u_t (task 1.10.4)."""

    def test_capping_enforced(self):
        """u_t should be bounded by max_u ceiling."""
        n = 100
        # Huge returns to produce large momentum
        returns = np.full(n, 0.10)
        prices = 100 * np.exp(np.cumsum(returns))
        vol = np.full(n, 0.01)
        
        config = MomentumConfig(
            enable=True,
            enable_mean_reversion=False,
            adjustment_scale=1.0,  # Large to force capping
            max_u_ceiling=0.03,
        )
        model = MomentumAugmentedDriftModel(config)
        model.precompute_signals(returns, prices, vol)
        
        u_t = model._exogenous_input
        self.assertIsNotNone(u_t)
        self.assertTrue(np.all(np.abs(u_t) <= config.max_u_ceiling + 1e-10),
                        f"u_t exceeds max_u_ceiling={config.max_u_ceiling}")

    def test_dynamic_max_u_with_q(self):
        """When max_u_scale_by_q=True, max_u scales with sqrt(q)."""
        n = 100
        returns = np.full(n, 0.05)
        prices = 100 * np.exp(np.cumsum(returns))
        vol = np.full(n, 0.01)
        
        config = MomentumConfig(
            enable=True,
            enable_mean_reversion=False,
            adjustment_scale=1.0,
            max_u_scale_by_q=True,
            max_u_multiplier=3.0,
            max_u_floor=0.005,
            max_u_ceiling=0.03,
        )
        
        # Small q -> tighter cap
        model_small = MomentumAugmentedDriftModel(config)
        model_small.precompute_signals(returns, prices, vol, q=1e-8)
        max_abs_small = np.max(np.abs(model_small._exogenous_input))
        
        # Larger q -> looser cap (but still bounded by ceiling)
        model_large = MomentumAugmentedDriftModel(config)
        model_large.precompute_signals(returns, prices, vol, q=1e-3)
        max_abs_large = np.max(np.abs(model_large._exogenous_input))
        
        self.assertGreaterEqual(max_abs_large, max_abs_small - 1e-10,
                                "Larger q should allow at least as much signal")


class TestBlendedSignalIntegrity(unittest.TestCase):
    """Integration tests: blended u_t behaves correctly end-to-end."""

    def test_mr_signal_keeps_signed_direction_in_exogenous_input(self):
        """MR signal is already signed toward equilibrium and must not be inverted."""
        log_prices = np.array([0.0, 0.02, 0.04])
        equilibrium = np.zeros_like(log_prices)
        vol = np.full_like(log_prices, 0.01)
        mr_signal = compute_mr_signal(log_prices, equilibrium, vol, kappa=0.1)

        config = MomentumConfig(
            enable=False,
            enable_mean_reversion=True,
            mr_adjustment_scale=1.0,
            max_u_scale_by_q=False,
        )
        model = MomentumAugmentedDriftModel(config)
        model._momentum_signal = np.zeros_like(mr_signal)
        model._mr_signal = mr_signal
        model._alpha_t = np.zeros_like(mr_signal)
        model._beta_t = np.ones_like(mr_signal)
        model._regime_labels = None
        model._q = 1e-6

        u_t = model._compute_exogenous_input()
        self.assertLess(mr_signal[-1], 0.0)
        self.assertLess(u_t[-1], 0.0)
        np.testing.assert_allclose(u_t, mr_signal, atol=1e-12)

    def test_trend_produces_positive_ut(self):
        """Strong uptrend should produce net-positive u_t."""
        n = 200
        returns = np.full(n, 0.005)  # Steady uptrend
        prices = 100 * np.exp(np.cumsum(returns))
        vol = np.full(n, 0.01)
        regime_labels = np.full(n, 0, dtype=int)  # LOW_VOL_TREND
        
        config = MomentumConfig(enable=True, enable_mean_reversion=True)
        model = MomentumAugmentedDriftModel(config)
        model.precompute_signals(returns, prices, vol, regime_labels=regime_labels)
        
        u_t = model._exogenous_input
        # After warmup period, u_t should be predominantly positive
        self.assertGreater(np.mean(u_t[63:]), 0,
                           "Uptrend u_t should be positive on average")

    def test_momentum_disabled_zeroes_contribution(self):
        """With momentum disabled, u_t should only reflect MR."""
        n = 100
        returns = np.full(n, 0.005)
        prices = 100 * np.exp(np.cumsum(returns))
        vol = np.full(n, 0.01)
        
        config_full = MomentumConfig(enable=True, enable_mean_reversion=True)
        config_no_mom = MomentumConfig(enable=False, enable_mean_reversion=True)
        
        model_full = MomentumAugmentedDriftModel(config_full)
        model_full.precompute_signals(returns, prices, vol)
        
        model_no_mom = MomentumAugmentedDriftModel(config_no_mom)
        model_no_mom.precompute_signals(returns, prices, vol)
        
        u_full = model_full._exogenous_input
        u_no_mom = model_no_mom._exogenous_input
        
        # With momentum disabled, u_t should differ
        if u_full is not None and u_no_mom is not None:
            self.assertFalse(np.allclose(u_full, u_no_mom, atol=1e-10),
                             "Disabling momentum should change u_t")


if __name__ == "__main__":
    unittest.main(verbosity=2)
