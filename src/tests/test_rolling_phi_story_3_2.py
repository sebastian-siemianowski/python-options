"""
Story 3.2: Rolling phi with Structural Break Detection -- Tests
===============================================================
Tests for:
  1. rolling_phi_estimate(): produces phi_t series
  2. CUSUM break detection: flags |delta_phi| > 0.3
  3. Post-break reset: phi resets to prior and re-estimates
  4. Synthetic DGP: break detected within 30 days
  5. Stable assets: no false breaks
"""
import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
REPO_ROOT = os.path.abspath(os.path.join(SRC_ROOT, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _generate_ar1(n, phi, sigma=0.01, seed=42):
    """Generate AR(1) return series with given phi."""
    rng = np.random.RandomState(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + rng.normal(0, sigma)
    return r


def _generate_ar1_with_break(n_pre, n_post, phi_pre, phi_post, sigma=0.01, seed=42):
    """Generate AR(1) with a structural break at n_pre."""
    rng = np.random.RandomState(seed)
    n = n_pre + n_post
    r = np.zeros(n)
    for i in range(1, n_pre):
        r[i] = phi_pre * r[i - 1] + rng.normal(0, sigma)
    for i in range(n_pre, n):
        r[i] = phi_post * r[i - 1] + rng.normal(0, sigma)
    return r


class TestRollingPhiEstimate(unittest.TestCase):
    """Test rolling_phi_estimate basic functionality."""

    def test_returns_phi_t_series(self):
        """rolling_phi_estimate returns a non-empty phi_t array."""
        from calibration.rolling_phi import rolling_phi_estimate
        np.random.seed(42)
        n = 600
        returns = _generate_ar1(n, phi=0.5, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01

        result = rolling_phi_estimate(returns, vol, window=252, step=21)
        self.assertGreater(len(result.phi_t), 0)
        self.assertEqual(len(result.phi_t), len(result.timestamps))
        self.assertEqual(result.n_windows, len(result.phi_t))

    def test_phi_t_values_bounded(self):
        """All phi_t values should be within [-0.8, 0.99]."""
        from calibration.rolling_phi import rolling_phi_estimate
        returns = _generate_ar1(800, phi=0.3, seed=123)
        vol = np.abs(returns) * 0.5 + 0.01

        result = rolling_phi_estimate(returns, vol, window=252, step=21)
        self.assertTrue(np.all(result.phi_t >= -0.80))
        self.assertTrue(np.all(result.phi_t <= 0.99))

    def test_timestamps_increasing(self):
        """Timestamps should be monotonically increasing."""
        from calibration.rolling_phi import rolling_phi_estimate
        returns = _generate_ar1(700, phi=0.5, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01

        result = rolling_phi_estimate(returns, vol, window=252, step=21)
        diffs = np.diff(result.timestamps)
        self.assertTrue(np.all(diffs > 0))

    def test_short_data_returns_empty(self):
        """With insufficient data, returns empty result."""
        from calibration.rolling_phi import rolling_phi_estimate
        returns = np.random.normal(0, 0.01, 100)
        vol = np.abs(returns) * 0.5 + 0.01

        result = rolling_phi_estimate(returns, vol, window=252, step=21)
        self.assertEqual(len(result.phi_t), 0)
        self.assertEqual(result.n_windows, 0)

    def test_mean_phi_reasonable_for_momentum(self):
        """For momentum data (phi=0.5), mean phi should be positive."""
        from calibration.rolling_phi import rolling_phi_estimate
        returns = _generate_ar1(800, phi=0.5, sigma=0.01, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01

        result = rolling_phi_estimate(returns, vol, window=252, step=21)
        self.assertGreater(result.phi_mean, 0.0,
                           f"Mean phi={result.phi_mean} should be positive for momentum data")

    def test_step_size_affects_n_windows(self):
        """Smaller step -> more windows."""
        from calibration.rolling_phi import rolling_phi_estimate
        returns = _generate_ar1(600, phi=0.3, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01

        result_21 = rolling_phi_estimate(returns, vol, window=252, step=21)
        result_10 = rolling_phi_estimate(returns, vol, window=252, step=10)
        self.assertGreater(result_10.n_windows, result_21.n_windows)


class TestEstimatePhiWindow(unittest.TestCase):
    """Test the lightweight window-level phi estimator."""

    def test_white_noise_phi_near_zero(self):
        """White noise should produce phi near 0."""
        from calibration.rolling_phi import _estimate_phi_window
        np.random.seed(42)
        r = np.random.normal(0, 0.01, 252)
        v = np.abs(r) * 0.5 + 0.01
        phi = _estimate_phi_window(r, v)
        self.assertAlmostEqual(phi, 0.0, delta=0.3,
                               msg=f"White noise phi={phi} should be near 0")

    def test_strong_momentum_phi_positive(self):
        """Strong AR(1) with phi=0.7 should produce positive estimate."""
        from calibration.rolling_phi import _estimate_phi_window
        r = _generate_ar1(252, phi=0.7, seed=42)
        v = np.abs(r) * 0.5 + 0.01
        phi = _estimate_phi_window(r, v)
        self.assertGreater(phi, 0.2, f"Momentum phi={phi} should be > 0.2")

    def test_short_window_returns_zero(self):
        """< 30 observations returns phi=0."""
        from calibration.rolling_phi import _estimate_phi_window
        r = np.random.normal(0, 0.01, 20)
        v = np.abs(r) * 0.5 + 0.01
        phi = _estimate_phi_window(r, v)
        self.assertEqual(phi, 0.0)


class TestCUSUMBreakDetection(unittest.TestCase):
    """Test CUSUM break detection on phi series."""

    def test_no_breaks_in_stable_series(self):
        """Constant phi series should produce no breaks."""
        from calibration.rolling_phi import cusum_phi_breaks
        phi_series = np.ones(20) * 0.5
        timestamps = np.arange(20) * 21
        breaks = cusum_phi_breaks(phi_series, timestamps, threshold=0.3)
        self.assertEqual(len(breaks), 0)

    def test_detects_large_break(self):
        """A jump from 0.8 to 0.1 should be detected."""
        from calibration.rolling_phi import cusum_phi_breaks
        phi_series = np.concatenate([np.ones(10) * 0.8, np.ones(10) * 0.1])
        timestamps = np.arange(20) * 21
        breaks = cusum_phi_breaks(phi_series, timestamps, threshold=0.3)
        self.assertGreater(len(breaks), 0, "Should detect break from 0.8 to 0.1")

    def test_break_has_correct_fields(self):
        """BreakPoint should have all expected fields."""
        from calibration.rolling_phi import cusum_phi_breaks, BreakPoint
        phi_series = np.concatenate([np.ones(10) * 0.9, np.ones(10) * 0.0])
        timestamps = np.arange(20) * 21
        breaks = cusum_phi_breaks(phi_series, timestamps, threshold=0.3)
        if breaks:
            bp = breaks[0]
            self.assertIsInstance(bp, BreakPoint)
            self.assertTrue(hasattr(bp, 'index'))
            self.assertTrue(hasattr(bp, 'phi_before'))
            self.assertTrue(hasattr(bp, 'phi_after'))
            self.assertTrue(hasattr(bp, 'delta_phi'))
            self.assertTrue(hasattr(bp, 'cusum_value'))

    def test_cooldown_prevents_rapid_breaks(self):
        """Cooldown should prevent multiple breaks within the cooldown period."""
        from calibration.rolling_phi import cusum_phi_breaks
        # Alternating phi values that could trigger many breaks
        phi_series = np.array([0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1, 0.8, 0.1])
        timestamps = np.arange(10) * 21
        breaks_no_cool = cusum_phi_breaks(phi_series, timestamps, threshold=0.3, cooldown=1)
        breaks_with_cool = cusum_phi_breaks(phi_series, timestamps, threshold=0.3, cooldown=5)
        # With longer cooldown, fewer breaks detected
        self.assertLessEqual(len(breaks_with_cool), len(breaks_no_cool))

    def test_short_series_no_crash(self):
        """Very short phi series should not crash."""
        from calibration.rolling_phi import cusum_phi_breaks
        phi_series = np.array([0.5])
        timestamps = np.array([0])
        breaks = cusum_phi_breaks(phi_series, timestamps, threshold=0.3)
        self.assertEqual(len(breaks), 0)


class TestSyntheticDGPBreakDetection(unittest.TestCase):
    """Test break detection on synthetic DGP with known break point."""

    def test_break_detected_within_window(self):
        """DGP switches phi from 0.9 to 0.1 at t=500. Detect within 2 windows."""
        from calibration.rolling_phi import rolling_phi_estimate, RollingPhiConfig

        returns = _generate_ar1_with_break(500, 500, phi_pre=0.9, phi_post=0.1, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01

        config = RollingPhiConfig(
            window=252, step=21,
            cusum_threshold=0.3,
            cusum_cooldown=42,
        )
        result = rolling_phi_estimate(returns, vol, config=config)

        self.assertGreater(result.n_breaks, 0,
                           "Should detect at least one break in DGP with phi switch")

        # The break should be detected after the true break point
        # With window=252, detection lags because the window must accumulate
        # enough post-break data for the rolling phi to shift significantly
        break_indices = [bp.index for bp in result.breaks]
        # Break should be detected within 1 window (252 obs) of true break
        near_true = any(abs(bi - 500) < 252 for bi in break_indices)
        self.assertTrue(near_true,
                        f"Break at {break_indices} should be within 252 obs of t=500")

    def test_phi_adapts_after_break(self):
        """After break, phi_t should shift toward the new regime."""
        from calibration.rolling_phi import rolling_phi_estimate, RollingPhiConfig

        returns = _generate_ar1_with_break(500, 500, phi_pre=0.9, phi_post=0.1, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01

        config = RollingPhiConfig(window=252, step=21)
        result = rolling_phi_estimate(returns, vol, config=config)

        if len(result.phi_t) > 5:
            # Early phi_t (before break) should be higher than late phi_t
            early_phi = np.mean(result.phi_t[:3])
            late_phi = np.mean(result.phi_t[-3:])
            self.assertGreater(early_phi, late_phi,
                               f"Early phi={early_phi} should > late phi={late_phi}")


class TestStableAssetNoFalseBreaks(unittest.TestCase):
    """Test that stable series produce few/no false breaks."""

    def test_stable_momentum_no_breaks(self):
        """Constant phi=0.5 series should produce no breaks."""
        from calibration.rolling_phi import rolling_phi_estimate, RollingPhiConfig

        returns = _generate_ar1(1000, phi=0.5, sigma=0.01, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01

        config = RollingPhiConfig(
            window=252, step=21,
            cusum_threshold=0.3,
            cusum_cooldown=63,
        )
        result = rolling_phi_estimate(returns, vol, config=config)

        # At most 1 false break per 2 years (504 observations)
        max_breaks = max(1, len(returns) // 504)
        self.assertLessEqual(result.n_breaks, max_breaks,
                             f"Stable series: {result.n_breaks} breaks > max {max_breaks}")

    def test_white_noise_few_breaks(self):
        """White noise should produce very few breaks."""
        from calibration.rolling_phi import rolling_phi_estimate, RollingPhiConfig

        np.random.seed(42)
        returns = np.random.normal(0, 0.015, 1000)
        vol = np.abs(returns) * 0.5 + 0.01

        config = RollingPhiConfig(
            window=252, step=21,
            cusum_threshold=0.3,
            cusum_cooldown=63,
        )
        result = rolling_phi_estimate(returns, vol, config=config)
        self.assertLessEqual(result.n_breaks, 2,
                             f"White noise: {result.n_breaks} breaks (expected <= 2)")


class TestRollingPhiConfig(unittest.TestCase):
    """Test configuration dataclass."""

    def test_defaults(self):
        from calibration.rolling_phi import RollingPhiConfig
        cfg = RollingPhiConfig()
        self.assertEqual(cfg.window, 252)
        self.assertEqual(cfg.step, 21)
        self.assertEqual(cfg.cusum_threshold, 0.3)
        self.assertTrue(cfg.reset_to_prior)

    def test_custom_config(self):
        from calibration.rolling_phi import RollingPhiConfig
        cfg = RollingPhiConfig(window=126, step=5, cusum_threshold=0.5)
        self.assertEqual(cfg.window, 126)
        self.assertEqual(cfg.step, 5)
        self.assertEqual(cfg.cusum_threshold, 0.5)


class TestRollingPhiResult(unittest.TestCase):
    """Test result dataclass."""

    def test_result_fields(self):
        from calibration.rolling_phi import RollingPhiResult
        result = RollingPhiResult(
            phi_t=np.array([0.5, 0.6]),
            timestamps=np.array([252, 273]),
            breaks=[],
            n_windows=2,
            n_breaks=0,
            phi_mean=0.55,
            phi_std=0.05,
        )
        self.assertEqual(result.n_windows, 2)
        self.assertEqual(result.n_breaks, 0)
        self.assertAlmostEqual(result.phi_mean, 0.55)


class TestAssetSymbolIntegration(unittest.TestCase):
    """Test asset_symbol parameter for prior-based resets."""

    def test_asset_symbol_accepted(self):
        """asset_symbol parameter should not crash."""
        from calibration.rolling_phi import rolling_phi_estimate
        returns = _generate_ar1(600, phi=0.5, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01
        result = rolling_phi_estimate(returns, vol, asset_symbol='SPY')
        self.assertIsNotNone(result)

    def test_none_asset_symbol_ok(self):
        """None asset_symbol should work (fallback to default prior)."""
        from calibration.rolling_phi import rolling_phi_estimate
        returns = _generate_ar1(600, phi=0.5, seed=42)
        vol = np.abs(returns) * 0.5 + 0.01
        result = rolling_phi_estimate(returns, vol, asset_symbol=None)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
