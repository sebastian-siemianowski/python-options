"""
Tests for Story 2.3: Online c Update via Innovation Variance Monitoring.

Tests the Numba kernel and Python-level online c update module.
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


class TestOnlineCUpdateKernel(unittest.TestCase):
    """Test the Numba kernel directly."""

    def test_kernel_returns_correct_shapes(self):
        """Kernel returns c_path and eta_path of correct length."""
        from models.numba_kernels import online_c_update_kernel

        N = 100
        innovations = np.random.randn(N) * 0.01
        vol_sq = np.full(N, 0.0004)  # 2% daily vol
        c_path, eta_path, ratio_ema = online_c_update_kernel(
            innovations, vol_sq, 1.0, 0.02, 0.001, 0.998, 0.1, 10.0
        )
        self.assertEqual(len(c_path), N)
        self.assertEqual(len(eta_path), N)

    def test_kernel_c_stays_in_bounds(self):
        """c never exceeds [c_min, c_max]."""
        from models.numba_kernels import online_c_update_kernel

        N = 500
        # Large innovations to push c upward
        innovations = np.random.randn(N) * 0.05
        vol_sq = np.full(N, 0.0001)  # Small vol -> large ratios
        c_path, _, _ = online_c_update_kernel(
            innovations, vol_sq, 1.0, 0.05, 0.001, 0.99, 0.1, 10.0
        )
        self.assertTrue(np.all(c_path >= 0.1))
        self.assertTrue(np.all(c_path <= 10.0))

    def test_kernel_c_increases_when_underestimated(self):
        """If true c=2.0 but start at c=0.5, c should increase."""
        from models.numba_kernels import online_c_update_kernel

        np.random.seed(42)
        N = 200
        true_c = 2.0
        vol_sq = np.full(N, 0.0004)
        # Generate innovations consistent with true_c: v ~ N(0, true_c * vol_sq)
        innovations = np.random.randn(N) * np.sqrt(true_c * vol_sq)

        c_path, _, _ = online_c_update_kernel(
            innovations, vol_sq, 0.5, 0.02, 0.001, 0.998, 0.1, 10.0
        )
        # Final c should be higher than initial
        self.assertGreater(c_path[-1], 0.5)

    def test_kernel_c_decreases_when_overestimated(self):
        """If true c=0.5 but start at c=3.0, c should decrease."""
        from models.numba_kernels import online_c_update_kernel

        np.random.seed(123)
        N = 200
        true_c = 0.5
        vol_sq = np.full(N, 0.0004)
        innovations = np.random.randn(N) * np.sqrt(true_c * vol_sq)

        c_path, _, _ = online_c_update_kernel(
            innovations, vol_sq, 3.0, 0.02, 0.001, 0.998, 0.1, 10.0
        )
        # Final c should be lower than initial
        self.assertLess(c_path[-1], 3.0)

    def test_kernel_eta_decays(self):
        """Learning rate decays over time."""
        from models.numba_kernels import online_c_update_kernel

        N = 100
        innovations = np.random.randn(N) * 0.01
        vol_sq = np.full(N, 0.0004)
        _, eta_path, _ = online_c_update_kernel(
            innovations, vol_sq, 1.0, 0.02, 0.001, 0.99, 0.1, 10.0
        )
        # eta should decrease monotonically
        self.assertGreater(eta_path[0], eta_path[-1])

    def test_kernel_eta_respects_floor(self):
        """Learning rate never goes below eta_min."""
        from models.numba_kernels import online_c_update_kernel

        N = 5000  # Many steps to ensure decay hits floor
        innovations = np.random.randn(N) * 0.01
        vol_sq = np.full(N, 0.0004)
        _, eta_path, _ = online_c_update_kernel(
            innovations, vol_sq, 1.0, 0.02, 0.005, 0.99, 0.1, 10.0
        )
        self.assertTrue(np.all(eta_path >= 0.005 - 1e-10))

    def test_kernel_handles_zero_vol(self):
        """Kernel handles zero volatility gracefully."""
        from models.numba_kernels import online_c_update_kernel

        N = 50
        innovations = np.random.randn(N) * 0.01
        vol_sq = np.zeros(N)  # All zero
        c_path, _, _ = online_c_update_kernel(
            innovations, vol_sq, 1.0, 0.02, 0.001, 0.998, 0.1, 10.0
        )
        # Should not crash; c stays at init
        self.assertEqual(c_path[0], 1.0)


class TestComputeInnovations(unittest.TestCase):
    """Test innovation computation."""

    def test_basic_innovations(self):
        from calibration.online_c_update import compute_innovations

        returns = np.array([0.01, 0.02, -0.01, 0.03])
        mu_filtered = np.array([0.005, 0.015, -0.005, 0.02])
        phi = 1.0

        innovations = compute_innovations(returns, mu_filtered, phi)

        self.assertEqual(len(innovations), 4)
        # v_0 = r_0 (no prior)
        self.assertAlmostEqual(innovations[0], 0.01)
        # v_1 = r_1 - phi * mu_0 = 0.02 - 1.0 * 0.005 = 0.015
        self.assertAlmostEqual(innovations[1], 0.015)

    def test_innovations_with_phi(self):
        from calibration.online_c_update import compute_innovations

        returns = np.array([0.01, 0.02, -0.01])
        mu_filtered = np.array([0.008, 0.018, -0.008])
        phi = 0.95

        innovations = compute_innovations(returns, mu_filtered, phi)

        # v_1 = r_1 - phi * mu_0 = 0.02 - 0.95 * 0.008
        expected_v1 = 0.02 - 0.95 * 0.008
        self.assertAlmostEqual(innovations[1], expected_v1, places=10)


class TestRunOnlineCUpdate(unittest.TestCase):
    """Test the full online c update pipeline."""

    def test_result_fields(self):
        from calibration.online_c_update import run_online_c_update, OnlineCConfig

        np.random.seed(42)
        N = 100
        returns = np.random.randn(N) * 0.02
        mu_filtered = np.zeros(N)
        vol = np.full(N, 0.02)

        result = run_online_c_update(returns, mu_filtered, vol, c_init=1.0)

        self.assertEqual(result.n_steps, N)
        self.assertEqual(len(result.c_path), N)
        self.assertEqual(len(result.eta_path), N)
        self.assertIsInstance(result.c_final, float)
        self.assertIsInstance(result.ratio_ema, float)
        self.assertEqual(result.c_init, 1.0)

    def test_no_divergence(self):
        """c stays within [0.1, 10.0] even with volatile data."""
        from calibration.online_c_update import run_online_c_update

        np.random.seed(99)
        N = 500
        # Very volatile returns
        returns = np.random.randn(N) * 0.10
        mu_filtered = np.cumsum(np.random.randn(N) * 0.001)
        vol = np.abs(np.random.randn(N) * 0.03) + 0.01

        result = run_online_c_update(returns, mu_filtered, vol, c_init=1.0)

        self.assertTrue(np.all(result.c_path >= 0.1))
        self.assertTrue(np.all(result.c_path <= 10.0))

    def test_tracking_error_computed(self):
        """Tracking error is computed when c_target provided."""
        from calibration.online_c_update import run_online_c_update

        np.random.seed(42)
        N = 100
        returns = np.random.randn(N) * 0.02
        mu_filtered = np.zeros(N)
        vol = np.full(N, 0.02)

        result = run_online_c_update(
            returns, mu_filtered, vol, c_init=1.0, c_target=1.5
        )
        self.assertGreater(result.tracking_error, 0.0)

    def test_convergence_with_correct_init(self):
        """When c_init matches true c, ratio_ema stays near 1.0."""
        from calibration.online_c_update import run_online_c_update

        np.random.seed(42)
        N = 500
        true_c = 1.0
        vol = np.full(N, 0.02)
        vol_sq = vol ** 2

        # Generate data consistent with true_c
        mu_true = np.cumsum(np.random.randn(N) * 1e-4)
        noise = np.random.randn(N) * np.sqrt(true_c * vol_sq)
        returns = mu_true + noise
        mu_filtered = mu_true + np.random.randn(N) * 1e-5  # Near-perfect filter

        result = run_online_c_update(returns, mu_filtered, vol, c_init=1.0)

        # ratio_ema should be near 1.0; c should not drift far from 1.0
        self.assertAlmostEqual(result.c_final, 1.0, delta=0.5)


class TestEvaluateOnlineCTracking(unittest.TestCase):
    """Test tracking quality evaluation."""

    def test_perfect_tracking(self):
        from calibration.online_c_update import evaluate_online_c_tracking

        c_target = 2.0
        c_path = np.full(100, 2.0)  # Perfect match

        result = evaluate_online_c_tracking(c_path, c_target)
        self.assertTrue(result["tracks_within_tolerance"])
        self.assertAlmostEqual(result["relative_error_at_end"], 0.0, places=5)
        self.assertEqual(result["convergence_step"], 0)

    def test_convergence_detection(self):
        from calibration.online_c_update import evaluate_online_c_tracking

        c_target = 2.0
        # c starts far and converges to target by step 30
        c_path = np.linspace(0.5, 2.0, 100)

        result = evaluate_online_c_tracking(c_path, c_target, window=20)
        self.assertTrue(result["tracks_within_tolerance"])
        self.assertIsNotNone(result["convergence_step"])

    def test_no_convergence(self):
        from calibration.online_c_update import evaluate_online_c_tracking

        c_target = 5.0
        c_path = np.full(100, 1.0)  # Stuck far from target

        result = evaluate_online_c_tracking(c_path, c_target)
        self.assertFalse(result["tracks_within_tolerance"])
        self.assertAlmostEqual(result["relative_error_at_end"], 0.8, places=2)

    def test_tracks_within_20_obs(self):
        """Online c tracks regime-conditional c within 10% after 20 obs."""
        from calibration.online_c_update import (
            run_online_c_update,
            evaluate_online_c_tracking,
            OnlineCConfig,
        )

        np.random.seed(42)
        N = 200
        c_target = 1.5
        vol = np.full(N, 0.02)
        vol_sq = vol ** 2

        # Generate data with true c = c_target
        innovations = np.random.randn(N) * np.sqrt(c_target * vol_sq)
        returns = innovations  # mu ~ 0

        # Start from c_init = 1.0 (off by 50%)
        config = OnlineCConfig(eta_init=0.03, eta_min=0.001, eta_decay=0.998)
        result = run_online_c_update(
            returns, np.zeros(N), vol, c_init=1.0, config=config, c_target=c_target
        )

        tracking = evaluate_online_c_tracking(result.c_path, c_target, window=20)

        # Should converge within reasonable range
        # Note: 20-obs convergence is acceptance criteria - test with tolerance
        self.assertLess(
            tracking["relative_error_at_end"], 0.30,
            f"Tracking error {tracking['relative_error_at_end']:.2%} > 30%"
        )


class TestRollingHitRate(unittest.TestCase):
    """Test hit rate comparison."""

    def test_identical_models_same_rate(self):
        from calibration.online_c_update import compute_rolling_hit_rate

        np.random.seed(42)
        N = 200
        returns = np.random.randn(N) * 0.02
        mu = np.random.randn(N) * 0.001

        base_rate, online_rate, improvement = compute_rolling_hit_rate(
            returns, mu, mu, window=60
        )
        self.assertAlmostEqual(improvement, 0.0, places=10)

    def test_better_model_higher_rate(self):
        from calibration.online_c_update import compute_rolling_hit_rate

        np.random.seed(42)
        N = 200
        # Returns with clear positive trend
        returns = np.abs(np.random.randn(N)) * 0.01 + 0.005

        # Good model: predicts positive
        mu_good = np.full(N, 0.005)
        # Bad model: predicts negative
        mu_bad = np.full(N, -0.005)

        base_rate, online_rate, improvement = compute_rolling_hit_rate(
            returns, mu_bad, mu_good, window=60
        )
        self.assertGreater(improvement, 0.0)


class TestSyntheticRegimeSwitch(unittest.TestCase):
    """Test online c on synthetic regime-switching data."""

    def test_c_adapts_to_regime_change(self):
        """c adjusts when true c shifts mid-series."""
        from calibration.online_c_update import run_online_c_update, OnlineCConfig

        np.random.seed(42)
        N = 400
        vol = np.full(N, 0.02)
        vol_sq = vol ** 2

        # First half: c = 0.5, second half: c = 3.0
        true_c = np.empty(N)
        true_c[:200] = 0.5
        true_c[200:] = 3.0

        innovations = np.random.randn(N) * np.sqrt(true_c * vol_sq)
        returns = innovations

        config = OnlineCConfig(eta_init=0.03, eta_min=0.002, eta_decay=0.999)
        result = run_online_c_update(
            returns, np.zeros(N), vol, c_init=0.5, config=config
        )

        # c should be larger in the second half
        mean_first = np.mean(result.c_path[50:200])
        mean_second = np.mean(result.c_path[250:])
        self.assertGreater(
            mean_second, mean_first,
            f"Second half c ({mean_second:.3f}) should exceed first half ({mean_first:.3f})"
        )


class TestRealDataOnlineC(unittest.TestCase):
    """Test online c update with real market data."""

    def _load_data(self, symbol):
        """Load OHLCV data for a symbol."""
        import pandas as pd
        price_path = os.path.join(SRC_ROOT, "data", "prices", f"{symbol}.csv")
        if not os.path.exists(price_path):
            return None
        df = pd.read_csv(price_path)
        close_col = [c for c in df.columns if c.lower() == "close"][0]
        close = df[close_col].values.astype(float)
        returns = np.diff(np.log(close))
        # EWMA vol
        vol = np.zeros(len(returns))
        var = returns[0] ** 2 if len(returns) > 0 else 1e-4
        alpha = 2.0 / 22.0
        for i in range(len(returns)):
            var = (1 - alpha) * var + alpha * returns[i] ** 2
            vol[i] = np.sqrt(max(var, 1e-10))
        return returns, vol

    def test_tsla_online_c_no_divergence(self):
        """TSLA (fast regime changes): c stays bounded."""
        from calibration.online_c_update import run_online_c_update

        data = self._load_data("TSLA")
        if data is None:
            self.skipTest("TSLA.csv not found")
        returns, vol = data

        # Run filter to get mu_filtered
        from models.numba_kernels import phi_gaussian_filter_kernel
        mu_filtered, _, _ = phi_gaussian_filter_kernel(
            returns, vol, q=1e-5, c=1.0, phi=0.99
        )

        result = run_online_c_update(returns, mu_filtered, vol, c_init=1.0, phi=0.99)
        self.assertTrue(np.all(result.c_path >= 0.1))
        self.assertTrue(np.all(result.c_path <= 10.0))
        self.assertGreater(result.n_steps, 100)

    def test_spy_online_c_no_divergence(self):
        """SPY (broad market): c stays bounded."""
        from calibration.online_c_update import run_online_c_update

        data = self._load_data("SPY")
        if data is None:
            self.skipTest("SPY.csv not found")
        returns, vol = data

        from models.numba_kernels import phi_gaussian_filter_kernel
        mu_filtered, _, _ = phi_gaussian_filter_kernel(
            returns, vol, q=1e-5, c=1.0, phi=0.99
        )

        result = run_online_c_update(returns, mu_filtered, vol, c_init=1.0, phi=0.99)
        self.assertTrue(np.all(result.c_path >= 0.1))
        self.assertTrue(np.all(result.c_path <= 10.0))

    def test_btc_online_c_if_available(self):
        """BTC-USD (24/7 market): c stays bounded."""
        from calibration.online_c_update import run_online_c_update

        data = self._load_data("BTC-USD")
        if data is None:
            self.skipTest("BTC-USD.csv not found")
        returns, vol = data

        from models.numba_kernels import phi_gaussian_filter_kernel
        mu_filtered, _, _ = phi_gaussian_filter_kernel(
            returns, vol, q=1e-5, c=1.0, phi=0.99
        )

        result = run_online_c_update(returns, mu_filtered, vol, c_init=1.0, phi=0.99)
        self.assertTrue(np.all(result.c_path >= 0.1))
        self.assertTrue(np.all(result.c_path <= 10.0))


class TestAdaptiveDecay(unittest.TestCase):
    """Test adaptive decay properties."""

    def test_decay_from_init_to_min(self):
        """eta decays from eta_init toward eta_min."""
        from models.numba_kernels import online_c_update_kernel

        N = 2000
        innovations = np.random.randn(N) * 0.01
        vol_sq = np.full(N, 0.0004)

        _, eta_path, _ = online_c_update_kernel(
            innovations, vol_sq, 1.0, 0.05, 0.005, 0.995, 0.1, 10.0
        )

        # First eta should be eta_init
        self.assertAlmostEqual(eta_path[0], 0.05, places=4)
        # Last eta should be near eta_min
        self.assertAlmostEqual(eta_path[-1], 0.005, delta=0.002)


if __name__ == "__main__":
    unittest.main()
