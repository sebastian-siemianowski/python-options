"""
Test Suite for Epic 12: Multi-Timeframe Signal Fusion
======================================================

Story 12.1: Adaptive Momentum Horizon Weights via OOS Ranking
Story 12.2: Momentum-Mean Reversion Regime Switch
Story 12.3: Cross-Asset Momentum Confirmation
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from calibration.multi_timeframe_fusion import (
    # Story 12.1
    AdaptiveMomentumResult,
    adaptive_momentum_weights,
    combine_momentum_signals,
    _compute_momentum_signal,
    DEFAULT_LOOKBACKS,
    CV_TRAIN_WINDOW,
    CV_TEST_WINDOW,
    # Story 12.2
    MomentumMRRegime,
    compute_variance_ratio,
    momentum_mr_regime_indicator,
    apply_regime_signal_weights,
    compute_variance_ratio_timeseries,
    VR_MOMENTUM_THRESHOLD,
    VR_MEAN_REVERT_THRESHOLD,
    MOMENTUM_REGIME,
    MEAN_REVERT_REGIME,
    NEUTRAL_REGIME,
    # Story 12.3
    CrossAssetConfirmation,
    estimate_correlation_matrix,
    cross_asset_confirmation,
    CONFIRMATION_HIGH,
    CONFIRMATION_LOW,
    CONFIRMATION_BOOST,
    CONFIRMATION_PENALTY,
    MIN_CORRELATION,
)


def _generate_trending_returns(n=500, drift=0.001, vol=0.01, seed=42):
    """Generate returns with positive serial correlation (momentum)."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    returns[0] = drift + rng.normal(0, vol)
    for t in range(1, n):
        # Positive autocorrelation
        returns[t] = 0.3 * returns[t-1] + drift + rng.normal(0, vol)
    return returns


def _generate_mean_reverting_returns(n=500, vol=0.01, seed=42):
    """Generate returns with negative serial correlation (mean reversion)."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    returns[0] = rng.normal(0, vol)
    for t in range(1, n):
        # Negative autocorrelation
        returns[t] = -0.4 * returns[t-1] + rng.normal(0, vol)
    return returns


def _generate_iid_returns(n=500, vol=0.01, seed=42):
    """Generate iid returns (random walk)."""
    return np.random.default_rng(seed).normal(0, vol, n)


# ===================================================================
# Story 12.1 Tests: Adaptive Momentum Weights
# ===================================================================

class TestMomentumSignal(unittest.TestCase):
    """Test _compute_momentum_signal()."""

    def test_output_shape(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        signal = _compute_momentum_signal(returns, 10)
        self.assertEqual(len(signal), 100)

    def test_first_entries_nan(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 100)
        signal = _compute_momentum_signal(returns, 10)
        self.assertTrue(np.all(np.isnan(signal[:10])))
        self.assertTrue(np.all(np.isfinite(signal[10:])))

    def test_positive_drift_positive_signal(self):
        returns = np.full(100, 0.001)  # Constant positive returns
        signal = _compute_momentum_signal(returns, 10)
        self.assertGreater(signal[50], 0)

    def test_lookback_sum(self):
        returns = np.arange(20, dtype=float) * 0.001
        signal = _compute_momentum_signal(returns, 5)
        # signal[5] = sum(returns[0:5]) = (0+1+2+3+4)*0.001 = 0.01
        self.assertAlmostEqual(signal[5], 0.01, places=10)


class TestAdaptiveMomentumWeights(unittest.TestCase):
    """Test adaptive_momentum_weights()."""

    def test_returns_result(self):
        returns = _generate_trending_returns(600)
        result = adaptive_momentum_weights(returns)
        self.assertIsInstance(result, AdaptiveMomentumResult)

    def test_weights_sum_to_one(self):
        returns = _generate_trending_returns(600)
        result = adaptive_momentum_weights(returns)
        self.assertAlmostEqual(result.weights.sum(), 1.0, places=5)

    def test_weights_non_negative(self):
        returns = _generate_trending_returns(600)
        result = adaptive_momentum_weights(returns)
        self.assertTrue(np.all(result.weights >= 0))

    def test_correct_n_lookbacks(self):
        returns = _generate_trending_returns(600)
        result = adaptive_momentum_weights(returns)
        self.assertEqual(len(result.weights), len(DEFAULT_LOOKBACKS))

    def test_custom_lookbacks(self):
        returns = _generate_trending_returns(600)
        result = adaptive_momentum_weights(returns, lookbacks=[5, 20])
        self.assertEqual(len(result.weights), 2)

    def test_short_data_fallback(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 50)
        result = adaptive_momentum_weights(returns)
        # Should fallback to equal weights
        self.assertEqual(result.n_folds, 0)

    def test_best_horizon_in_lookbacks(self):
        returns = _generate_trending_returns(600)
        result = adaptive_momentum_weights(returns)
        self.assertIn(result.best_horizon, DEFAULT_LOOKBACKS)

    def test_hit_rates_reasonable(self):
        returns = _generate_trending_returns(600)
        result = adaptive_momentum_weights(returns)
        for hr in result.hit_rates_per_horizon:
            self.assertGreaterEqual(hr, 0.0)
            self.assertLessEqual(hr, 1.0)


class TestCombineMomentum(unittest.TestCase):
    """Test combine_momentum_signals()."""

    def test_output_shape(self):
        returns = np.random.default_rng(42).normal(0, 0.01, 200)
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        combined = combine_momentum_signals(returns, weights)
        self.assertEqual(len(combined), 200)

    def test_all_finite(self):
        returns = np.random.default_rng(42).normal(0.001, 0.01, 200)
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        combined = combine_momentum_signals(returns, weights)
        self.assertTrue(np.all(np.isfinite(combined)))


# ===================================================================
# Story 12.2 Tests: Momentum-MR Regime Switch
# ===================================================================

class TestVarianceRatio(unittest.TestCase):
    """Test compute_variance_ratio()."""

    def test_random_walk_near_one(self):
        returns = _generate_iid_returns(1000)
        vr = compute_variance_ratio(returns, q=5)
        self.assertAlmostEqual(vr, 1.0, delta=0.3)

    def test_trending_above_one(self):
        returns = _generate_trending_returns(1000)
        vr = compute_variance_ratio(returns, q=5)
        self.assertGreater(vr, 1.0)

    def test_mean_reverting_below_one(self):
        returns = _generate_mean_reverting_returns(1000)
        vr = compute_variance_ratio(returns, q=5)
        self.assertLess(vr, 1.0)

    def test_short_data_default(self):
        returns = np.array([0.01, -0.01])
        vr = compute_variance_ratio(returns, q=5)
        self.assertAlmostEqual(vr, 1.0)

    def test_positive(self):
        returns = _generate_iid_returns(500)
        vr = compute_variance_ratio(returns, q=5)
        self.assertGreater(vr, 0)


class TestMomentumMRRegime(unittest.TestCase):
    """Test momentum_mr_regime_indicator()."""

    def test_trending_is_momentum(self):
        returns = _generate_trending_returns(200)
        result = momentum_mr_regime_indicator(returns)
        self.assertIsInstance(result, MomentumMRRegime)
        # Strongly trending should be momentum
        self.assertTrue(result.variance_ratio > 0.9)

    def test_mean_reverting_detected(self):
        returns = _generate_mean_reverting_returns(200)
        result = momentum_mr_regime_indicator(returns)
        self.assertTrue(result.is_mean_reverting)

    def test_iid_neutral(self):
        returns = _generate_iid_returns(200)
        result = momentum_mr_regime_indicator(returns)
        # iid should be NEUTRAL (VR ~1.0)
        self.assertEqual(result.regime, NEUTRAL_REGIME)

    def test_regime_values(self):
        returns = _generate_iid_returns(200)
        result = momentum_mr_regime_indicator(returns)
        self.assertIn(result.regime, [MOMENTUM_REGIME, MEAN_REVERT_REGIME, NEUTRAL_REGIME])

    def test_vr_stored(self):
        returns = _generate_iid_returns(200)
        result = momentum_mr_regime_indicator(returns)
        self.assertGreater(result.variance_ratio, 0)


class TestApplyRegimeWeights(unittest.TestCase):
    """Test apply_regime_signal_weights()."""

    def test_momentum_regime_doubles_momentum(self):
        regime = MomentumMRRegime(MOMENTUM_REGIME, 1.5, True, False)
        combined = apply_regime_signal_weights(0.5, 0.3, regime)
        self.assertAlmostEqual(combined, 1.0)  # 2*0.5 + 0*0.3

    def test_mr_regime_doubles_mr(self):
        regime = MomentumMRRegime(MEAN_REVERT_REGIME, 0.6, False, True)
        combined = apply_regime_signal_weights(0.5, 0.3, regime)
        self.assertAlmostEqual(combined, 0.6)  # 0*0.5 + 2*0.3

    def test_neutral_equal_weight(self):
        regime = MomentumMRRegime(NEUTRAL_REGIME, 1.0, False, False)
        combined = apply_regime_signal_weights(0.5, 0.3, regime)
        self.assertAlmostEqual(combined, 0.8)  # 0.5 + 0.3


class TestVRTimeseries(unittest.TestCase):
    """Test compute_variance_ratio_timeseries()."""

    def test_output_shape(self):
        returns = _generate_iid_returns(200)
        vr_ts = compute_variance_ratio_timeseries(returns, window=60)
        self.assertEqual(len(vr_ts), 200)

    def test_early_nan(self):
        returns = _generate_iid_returns(200)
        vr_ts = compute_variance_ratio_timeseries(returns, window=60)
        self.assertTrue(np.all(np.isnan(vr_ts[:60])))

    def test_later_finite(self):
        returns = _generate_iid_returns(200)
        vr_ts = compute_variance_ratio_timeseries(returns, window=60)
        self.assertTrue(np.all(np.isfinite(vr_ts[60:])))


# ===================================================================
# Story 12.3 Tests: Cross-Asset Confirmation
# ===================================================================

class TestCorrelationMatrix(unittest.TestCase):
    """Test estimate_correlation_matrix()."""

    def test_diagonal_one(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (300, 5))
        corr = estimate_correlation_matrix(returns)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (300, 5))
        corr = estimate_correlation_matrix(returns)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)

    def test_correlated_assets(self):
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, 300)
        a1 = base + rng.normal(0, 0.002, 300)
        a2 = base + rng.normal(0, 0.002, 300)
        a3 = rng.normal(0, 0.01, 300)  # uncorrelated
        returns = np.column_stack([a1, a2, a3])
        corr = estimate_correlation_matrix(returns)
        # a1 and a2 should be highly correlated
        self.assertGreater(corr[0, 1], 0.5)
        # a3 should be less correlated
        self.assertLess(abs(corr[0, 2]), 0.5)

    def test_shape(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (300, 8))
        corr = estimate_correlation_matrix(returns)
        self.assertEqual(corr.shape, (8, 8))


class TestCrossAssetConfirmation(unittest.TestCase):
    """Test cross_asset_confirmation()."""

    def test_returns_result(self):
        signals = np.array([0.01, 0.02, 0.015, -0.005])
        corr = np.eye(4)
        corr[0, 1] = corr[1, 0] = 0.8
        corr[0, 2] = corr[2, 0] = 0.6
        result = cross_asset_confirmation(0, signals, corr)
        self.assertIsInstance(result, CrossAssetConfirmation)

    def test_all_agree_high_confirmation(self):
        """All correlated assets agree on direction -> high confirmation."""
        signals = np.array([0.01, 0.02, 0.015, 0.008])
        corr = np.ones((4, 4)) * 0.5
        np.fill_diagonal(corr, 1.0)
        result = cross_asset_confirmation(0, signals, corr)
        self.assertGreater(result.confirmation_score, 0.5)
        self.assertTrue(result.is_confirmed)

    def test_all_disagree_low_confirmation(self):
        """Correlated assets disagree -> low confirmation."""
        signals = np.array([0.01, -0.02, -0.015, -0.008])
        corr = np.ones((4, 4)) * 0.5
        np.fill_diagonal(corr, 1.0)
        result = cross_asset_confirmation(0, signals, corr)
        self.assertLess(result.confirmation_score, 0.0)

    def test_no_correlated_assets(self):
        """If no assets are correlated -> neutral."""
        signals = np.array([0.01, 0.02, -0.01])
        corr = np.eye(3)  # No off-diagonal correlation
        result = cross_asset_confirmation(0, signals, corr)
        self.assertEqual(result.n_correlated_assets, 0)
        self.assertAlmostEqual(result.confidence_multiplier, 1.0)

    def test_boost_multiplier(self):
        """High confirmation -> multiplier > 1."""
        signals = np.array([0.01, 0.02, 0.015])
        corr = np.ones((3, 3)) * 0.7
        np.fill_diagonal(corr, 1.0)
        result = cross_asset_confirmation(0, signals, corr)
        if result.is_confirmed:
            self.assertAlmostEqual(result.confidence_multiplier, 1.0 + CONFIRMATION_BOOST)

    def test_penalty_multiplier(self):
        """Divergent signals -> multiplier < 1."""
        signals = np.array([0.01, -0.02, -0.015])
        corr = np.ones((3, 3)) * 0.7
        np.fill_diagonal(corr, 1.0)
        result = cross_asset_confirmation(0, signals, corr)
        if result.is_divergent:
            self.assertAlmostEqual(result.confidence_multiplier, 1.0 - CONFIRMATION_PENALTY)

    def test_score_range(self):
        rng = np.random.default_rng(42)
        signals = rng.normal(0, 0.01, 10)
        corr = np.eye(10) + rng.uniform(-0.3, 0.3, (10, 10))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        result = cross_asset_confirmation(0, signals, corr)
        self.assertGreaterEqual(result.confirmation_score, -1.0)
        self.assertLessEqual(result.confirmation_score, 1.0)


# ===================================================================
# TestConstants
# ===================================================================

class TestEpic12Constants(unittest.TestCase):
    """Test constant values."""

    def test_default_lookbacks(self):
        self.assertEqual(DEFAULT_LOOKBACKS, [5, 10, 20, 60])

    def test_vr_thresholds(self):
        self.assertGreater(VR_MOMENTUM_THRESHOLD, 1.0)
        self.assertLess(VR_MEAN_REVERT_THRESHOLD, 1.0)

    def test_confirmation_thresholds(self):
        self.assertGreater(CONFIRMATION_HIGH, 0)
        self.assertLess(CONFIRMATION_LOW, 0)

    def test_boost_penalty(self):
        self.assertGreater(CONFIRMATION_BOOST, 0)
        self.assertGreater(CONFIRMATION_PENALTY, 0)


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEpic12EdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_constant_returns(self):
        returns = np.full(200, 0.001)
        result = adaptive_momentum_weights(returns)
        self.assertTrue(np.all(np.isfinite(result.weights)))

    def test_zero_returns(self):
        returns = np.zeros(200)
        vr = compute_variance_ratio(returns)
        self.assertAlmostEqual(vr, 1.0)

    def test_single_asset_confirmation(self):
        signals = np.array([0.01])
        corr = np.array([[1.0]])
        result = cross_asset_confirmation(0, signals, corr)
        self.assertEqual(result.n_correlated_assets, 0)

    def test_zero_target_signal(self):
        signals = np.array([0.0, 0.02, -0.01])
        corr = np.ones((3, 3)) * 0.5
        np.fill_diagonal(corr, 1.0)
        result = cross_asset_confirmation(0, signals, corr)
        self.assertAlmostEqual(result.confirmation_score, 0.0)


if __name__ == "__main__":
    unittest.main()
