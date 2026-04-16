"""
Test Suite for Epic 15: Regime-Aware Position Sizing
=====================================================

Story 15.1: Regime-Specific Position Limits
Story 15.2: Dynamic Leverage via Forecast Confidence
Story 15.3: Volatility-Targeting Overlay
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.regime_position_sizing import (
    # Story 15.1
    RegimePositionResult,
    regime_position_limit,
    regime_position_limit_array,
    REGIME_LIMITS,
    LOW_VOL_TREND, HIGH_VOL_TREND, LOW_VOL_RANGE, HIGH_VOL_RANGE, CRISIS_JUMP,
    # Story 15.2
    LeverageResult,
    dynamic_leverage,
    apply_dynamic_leverage,
    dynamic_leverage_array,
    DEFAULT_CONF_THRESHOLD, DEFAULT_LEVERAGE_K, DEFAULT_LEVERAGE_MAX,
    # Story 15.3
    VolTargetResult,
    vol_target_weight,
    vol_target_weight_array,
    compute_portfolio_vol,
    DEFAULT_VOL_TARGET, MIN_VOL_WEIGHT, MAX_VOL_WEIGHT,
)


# ===================================================================
# Story 15.1 Tests: Regime-Specific Position Limits
# ===================================================================

class TestRegimePositionLimit(unittest.TestCase):
    """Test regime_position_limit()."""

    def test_low_vol_trend_max(self):
        result = regime_position_limit(LOW_VOL_TREND, 0.90)
        self.assertAlmostEqual(result.limited_fraction, 0.80)
        self.assertTrue(result.was_limited)

    def test_low_vol_trend_within(self):
        result = regime_position_limit(LOW_VOL_TREND, 0.50)
        self.assertAlmostEqual(result.limited_fraction, 0.50)
        self.assertFalse(result.was_limited)

    def test_high_vol_trend_max(self):
        result = regime_position_limit(HIGH_VOL_TREND, 0.70)
        self.assertAlmostEqual(result.limited_fraction, 0.50)

    def test_low_vol_range_max(self):
        result = regime_position_limit(LOW_VOL_RANGE, 0.50)
        self.assertAlmostEqual(result.limited_fraction, 0.20)

    def test_high_vol_range_max(self):
        result = regime_position_limit(HIGH_VOL_RANGE, 0.50)
        self.assertAlmostEqual(result.limited_fraction, 0.10)

    def test_crisis_jump_max(self):
        result = regime_position_limit(CRISIS_JUMP, 0.50)
        self.assertAlmostEqual(result.limited_fraction, 0.05)

    def test_negative_fraction_limited(self):
        result = regime_position_limit(CRISIS_JUMP, -0.50)
        self.assertAlmostEqual(result.limited_fraction, -0.05)

    def test_unknown_regime_conservative(self):
        result = regime_position_limit("UNKNOWN", 0.50)
        self.assertAlmostEqual(result.limited_fraction, 0.20)

    def test_zero_fraction(self):
        result = regime_position_limit(LOW_VOL_TREND, 0.0)
        self.assertAlmostEqual(result.limited_fraction, 0.0)
        self.assertFalse(result.was_limited)

    def test_returns_dataclass(self):
        result = regime_position_limit(LOW_VOL_TREND, 0.5)
        self.assertIsInstance(result, RegimePositionResult)
        self.assertEqual(result.regime, LOW_VOL_TREND)

    def test_custom_limits(self):
        custom = {LOW_VOL_TREND: 0.30}
        result = regime_position_limit(LOW_VOL_TREND, 0.50, custom_limits=custom)
        self.assertAlmostEqual(result.limited_fraction, 0.30)


class TestRegimePositionLimitArray(unittest.TestCase):
    """Test regime_position_limit_array()."""

    def test_output_shape(self):
        regimes = [LOW_VOL_TREND, HIGH_VOL_TREND, CRISIS_JUMP]
        fracs = np.array([0.5, 0.5, 0.5])
        limited = regime_position_limit_array(regimes, fracs)
        self.assertEqual(len(limited), 3)

    def test_correct_limits(self):
        regimes = [LOW_VOL_TREND, CRISIS_JUMP]
        fracs = np.array([0.90, 0.50])
        limited = regime_position_limit_array(regimes, fracs)
        self.assertAlmostEqual(limited[0], 0.80)
        self.assertAlmostEqual(limited[1], 0.05)


# ===================================================================
# Story 15.2 Tests: Dynamic Leverage
# ===================================================================

class TestDynamicLeverage(unittest.TestCase):
    """Test dynamic_leverage()."""

    def test_below_threshold_no_leverage(self):
        lev = dynamic_leverage(0.50)
        self.assertAlmostEqual(lev, 1.0)

    def test_at_threshold_no_leverage(self):
        lev = dynamic_leverage(0.55)
        self.assertAlmostEqual(lev, 1.0)

    def test_above_threshold_levered(self):
        lev = dynamic_leverage(0.65)
        # L = 1 + 2*(0.65-0.55) = 1.2
        self.assertAlmostEqual(lev, 1.2)

    def test_high_confidence_capped(self):
        lev = dynamic_leverage(0.95)
        # L = 1 + 2*(0.95-0.55) = 1.8, capped at 1.5
        self.assertAlmostEqual(lev, 1.5)

    def test_nan_confidence(self):
        lev = dynamic_leverage(float('nan'))
        self.assertAlmostEqual(lev, 1.0)

    def test_zero_confidence(self):
        lev = dynamic_leverage(0.0)
        self.assertAlmostEqual(lev, 1.0)

    def test_custom_params(self):
        lev = dynamic_leverage(0.80, conf_threshold=0.50, k=3.0, L_max=2.0)
        # L = 1 + 3*(0.80-0.50) = 1.9
        self.assertAlmostEqual(lev, 1.9)

    def test_custom_params_capped(self):
        lev = dynamic_leverage(0.90, conf_threshold=0.50, k=3.0, L_max=2.0)
        # L = 1 + 3*(0.90-0.50) = 2.2, capped at 2.0
        self.assertAlmostEqual(lev, 2.0)


class TestApplyDynamicLeverage(unittest.TestCase):
    """Test apply_dynamic_leverage()."""

    def test_returns_result(self):
        result = apply_dynamic_leverage(0.3, 0.70)
        self.assertIsInstance(result, LeverageResult)

    def test_leveraged_fraction(self):
        result = apply_dynamic_leverage(0.3, 0.65)
        # lev = 1.2, effective = 0.3 * 1.2 = 0.36
        self.assertAlmostEqual(result.effective_fraction, 0.36)
        self.assertTrue(result.is_leveraged)

    def test_unleveraged(self):
        result = apply_dynamic_leverage(0.3, 0.40)
        self.assertAlmostEqual(result.effective_fraction, 0.3)
        self.assertFalse(result.is_leveraged)


class TestDynamicLeverageArray(unittest.TestCase):
    """Test dynamic_leverage_array()."""

    def test_output_shape(self):
        fracs = np.array([0.3, 0.3, 0.3])
        confs = np.array([0.40, 0.65, 0.90])
        result = dynamic_leverage_array(fracs, confs)
        self.assertEqual(len(result), 3)

    def test_correct_values(self):
        fracs = np.array([0.3, 0.3])
        confs = np.array([0.40, 0.65])
        result = dynamic_leverage_array(fracs, confs)
        self.assertAlmostEqual(result[0], 0.3)    # No leverage
        self.assertAlmostEqual(result[1], 0.36)   # 1.2x leverage


# ===================================================================
# Story 15.3 Tests: Volatility-Targeting Overlay
# ===================================================================

class TestVolTargetWeight(unittest.TestCase):
    """Test vol_target_weight()."""

    def test_equal_vol(self):
        """Asset vol = target vol -> weight = 1.0."""
        result = vol_target_weight(0.15, sigma_target=0.15)
        self.assertAlmostEqual(result.weight, 1.0)

    def test_high_vol_asset(self):
        """MSTR-like (80% vol) -> weight ~0.19."""
        result = vol_target_weight(0.80, sigma_target=0.15)
        self.assertAlmostEqual(result.weight, 0.15 / 0.80, places=3)

    def test_low_vol_asset(self):
        """SPY-like (15% vol) -> weight = 1.0."""
        result = vol_target_weight(0.15, sigma_target=0.15)
        self.assertAlmostEqual(result.weight, 1.0)

    def test_very_low_vol_capped(self):
        """Very low vol -> weight capped at MAX_VOL_WEIGHT."""
        result = vol_target_weight(0.01, sigma_target=0.15)
        self.assertAlmostEqual(result.weight, MAX_VOL_WEIGHT)
        self.assertTrue(result.was_capped)

    def test_zero_vol_floored(self):
        result = vol_target_weight(0.0)
        self.assertAlmostEqual(result.weight, MIN_VOL_WEIGHT)
        self.assertTrue(result.was_floored)

    def test_negative_vol_floored(self):
        result = vol_target_weight(-0.1)
        self.assertAlmostEqual(result.weight, MIN_VOL_WEIGHT)

    def test_nan_vol_floored(self):
        result = vol_target_weight(float('nan'))
        self.assertAlmostEqual(result.weight, MIN_VOL_WEIGHT)

    def test_returns_dataclass(self):
        result = vol_target_weight(0.20)
        self.assertIsInstance(result, VolTargetResult)

    def test_inverse_relationship(self):
        """Higher vol -> lower weight."""
        r1 = vol_target_weight(0.10)
        r2 = vol_target_weight(0.30)
        self.assertGreater(r1.weight, r2.weight)


class TestVolTargetWeightArray(unittest.TestCase):
    """Test vol_target_weight_array()."""

    def test_output_shape(self):
        sigmas = np.array([0.15, 0.30, 0.80])
        weights = vol_target_weight_array(sigmas)
        self.assertEqual(len(weights), 3)

    def test_correct_values(self):
        sigmas = np.array([0.15, 0.30])
        weights = vol_target_weight_array(sigmas, sigma_target=0.15)
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], 0.5)


class TestComputePortfolioVol(unittest.TestCase):
    """Test compute_portfolio_vol()."""

    def test_single_asset(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (500, 1))
        weights = np.array([1.0])
        vol = compute_portfolio_vol(returns, weights)
        self.assertGreater(vol, 0)
        # Should be around 0.01 * sqrt(252) ~ 0.159
        self.assertAlmostEqual(vol, 0.01 * np.sqrt(252), delta=0.02)

    def test_diversification(self):
        """Portfolio vol < sum of individual vols (diversification)."""
        rng = np.random.default_rng(42)
        r1 = rng.normal(0, 0.01, 500)
        r2 = rng.normal(0, 0.01, 500)  # Independent
        returns = np.column_stack([r1, r2])
        weights = np.array([0.5, 0.5])
        port_vol = compute_portfolio_vol(returns, weights)
        # Portfolio vol should be less than 0.01*sqrt(252) (diversification benefit)
        self.assertLess(port_vol, 0.01 * np.sqrt(252))

    def test_short_data(self):
        returns = np.array([[0.01], [-0.01]])
        weights = np.array([1.0])
        vol = compute_portfolio_vol(returns, weights)
        self.assertGreater(vol, 0)


# ===================================================================
# Constants Tests
# ===================================================================

class TestEpic15Constants(unittest.TestCase):
    """Test constant values."""

    def test_regime_limits(self):
        self.assertAlmostEqual(REGIME_LIMITS[LOW_VOL_TREND], 0.80)
        self.assertAlmostEqual(REGIME_LIMITS[HIGH_VOL_TREND], 0.50)
        self.assertAlmostEqual(REGIME_LIMITS[LOW_VOL_RANGE], 0.20)
        self.assertAlmostEqual(REGIME_LIMITS[HIGH_VOL_RANGE], 0.10)
        self.assertAlmostEqual(REGIME_LIMITS[CRISIS_JUMP], 0.05)

    def test_leverage_defaults(self):
        self.assertAlmostEqual(DEFAULT_CONF_THRESHOLD, 0.55)
        self.assertAlmostEqual(DEFAULT_LEVERAGE_K, 2.0)
        self.assertAlmostEqual(DEFAULT_LEVERAGE_MAX, 1.5)

    def test_vol_target_defaults(self):
        self.assertAlmostEqual(DEFAULT_VOL_TARGET, 0.15)
        self.assertAlmostEqual(MIN_VOL_WEIGHT, 0.05)
        self.assertAlmostEqual(MAX_VOL_WEIGHT, 3.0)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic15EdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_regime_limit_exact_boundary(self):
        result = regime_position_limit(LOW_VOL_TREND, 0.80)
        self.assertAlmostEqual(result.limited_fraction, 0.80)
        self.assertFalse(result.was_limited)

    def test_leverage_exactly_one(self):
        lev = dynamic_leverage(DEFAULT_CONF_THRESHOLD)
        self.assertAlmostEqual(lev, 1.0)

    def test_vol_target_extreme_vol(self):
        result = vol_target_weight(10.0)  # 1000% annualized vol
        # 0.15/10.0 = 0.015, floored at MIN_VOL_WEIGHT = 0.05
        self.assertAlmostEqual(result.weight, MIN_VOL_WEIGHT)

    def test_portfolio_vol_all_zeros(self):
        returns = np.zeros((100, 3))
        weights = np.array([0.33, 0.33, 0.34])
        vol = compute_portfolio_vol(returns, weights)
        self.assertAlmostEqual(vol, 0.0)


if __name__ == "__main__":
    unittest.main()
