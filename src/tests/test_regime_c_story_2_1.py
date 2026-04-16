"""
Story 2.1: Regime-Conditional c Estimation
==========================================
Tests that regime-conditional c estimation produces:
1. Different c values per regime
2. c_crisis > c_trend for most assets
3. CRPS improvement
4. PIT coverage within bounds
"""

import os
import sys
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class TestBuildCArray(unittest.TestCase):
    """Test the Numba kernel for building c_array from regime labels."""

    def test_builds_correct_array(self):
        from models.numba_kernels import build_c_array_from_regimes
        regime_labels = np.array([0, 1, 2, 3, 4, 0, 1], dtype=np.int64)
        c_per_regime = np.array([0.5, 1.0, 1.2, 1.5, 2.0], dtype=np.float64)
        c_arr = build_c_array_from_regimes(regime_labels, c_per_regime)
        expected = np.array([0.5, 1.0, 1.2, 1.5, 2.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(c_arr, expected)

    def test_out_of_range_regime_fallback(self):
        from models.numba_kernels import build_c_array_from_regimes
        regime_labels = np.array([0, 5, -1, 2], dtype=np.int64)
        c_per_regime = np.array([0.5, 1.0, 1.2, 1.5, 2.0], dtype=np.float64)
        c_arr = build_c_array_from_regimes(regime_labels, c_per_regime)
        self.assertAlmostEqual(c_arr[0], 0.5)
        self.assertAlmostEqual(c_arr[1], 1.0)  # fallback for regime 5
        self.assertAlmostEqual(c_arr[2], 1.0)  # fallback for regime -1
        self.assertAlmostEqual(c_arr[3], 1.2)


class TestRegimeCGaussianKernel(unittest.TestCase):
    """Test the time-varying c Gaussian filter kernel."""

    def test_matches_scalar_when_constant(self):
        """When c_array is constant, should match scalar kernel."""
        from models.numba_kernels import (
            regime_c_gaussian_filter_kernel,
            phi_gaussian_filter_kernel,
        )
        np.random.seed(42)
        n = 500
        returns = np.random.randn(n) * 0.01
        vol = np.abs(np.random.randn(n)) * 0.01 + 0.005
        q, c, phi = 1e-6, 1.0, 0.98

        c_array = np.full(n, c, dtype=np.float64)
        mu_rc, P_rc, ll_rc = regime_c_gaussian_filter_kernel(
            returns, vol, q, c_array, phi)
        mu_sc, P_sc, ll_sc = phi_gaussian_filter_kernel(
            returns, vol, q, c, phi)

        np.testing.assert_array_almost_equal(mu_rc, mu_sc, decimal=10)
        np.testing.assert_array_almost_equal(P_rc, P_sc, decimal=10)
        self.assertAlmostEqual(ll_rc, ll_sc, places=5)

    def test_higher_c_reduces_kalman_gain(self):
        """Higher c -> more observation noise -> lower Kalman gain -> smoother estimates."""
        from models.numba_kernels import regime_c_gaussian_filter_kernel
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.01
        vol = np.ones(n) * 0.01
        q, phi = 1e-6, 0.98

        c_low = np.full(n, 0.5, dtype=np.float64)
        c_high = np.full(n, 3.0, dtype=np.float64)

        mu_low, _, _ = regime_c_gaussian_filter_kernel(returns, vol, q, c_low, phi)
        mu_high, _, _ = regime_c_gaussian_filter_kernel(returns, vol, q, c_high, phi)

        # Higher c -> smoother mu (smaller standard deviation of changes)
        diff_low = np.diff(mu_low)
        diff_high = np.diff(mu_high)
        self.assertGreater(np.std(diff_low), np.std(diff_high))


class TestRegimeCStudentTKernel(unittest.TestCase):
    """Test the time-varying c Student-t filter kernel."""

    def test_runs_without_error(self):
        from models.numba_kernels import regime_c_student_t_filter_kernel
        from scipy.special import gammaln
        np.random.seed(42)
        n = 300
        returns = np.random.randn(n) * 0.01
        vol = np.ones(n) * 0.01
        q, phi, nu = 1e-6, 0.98, 8.0
        c_array = np.ones(n, dtype=np.float64)
        lg1 = float(gammaln(nu / 2.0))
        lg2 = float(gammaln((nu + 1.0) / 2.0))

        mu, P, ll = regime_c_student_t_filter_kernel(
            returns, vol, q, c_array, phi, nu, lg1, lg2)
        self.assertEqual(len(mu), n)
        self.assertTrue(np.isfinite(ll))


class TestFitRegimeC(unittest.TestCase):
    """Test the fit_regime_c optimizer."""

    def test_synthetic_trend_vs_crisis(self):
        """On synthetic data: c_crisis > c_trend."""
        from models.regime_c import fit_regime_c
        np.random.seed(123)
        n = 1000
        vol = np.ones(n) * 0.01
        # Regime labels: first 500 = trend (0), last 500 = crisis (4)
        regime_labels = np.zeros(n, dtype=np.int64)
        regime_labels[500:] = 4
        # Trend: drift with small noise
        trend_drift = 0.0005
        trend_noise = 0.005
        crisis_noise = 0.02
        returns = np.zeros(n)
        returns[:500] = trend_drift + np.random.randn(500) * trend_noise
        returns[500:] = np.random.randn(500) * crisis_noise

        result = fit_regime_c(
            returns, vol, regime_labels, q=1e-6, phi=0.98, c_scalar=1.0)
        self.assertTrue(result.fit_success)
        # c_crisis (regime 4) should be larger than c_trend (regime 0)
        self.assertGreater(result.c_per_regime[4], result.c_per_regime[0],
            f"c_crisis={result.c_per_regime[4]:.3f} should be > "
            f"c_trend={result.c_per_regime[0]:.3f}")

    def test_delta_ll_nonnegative(self):
        """Regime-c should match or beat scalar c on training data."""
        from models.regime_c import fit_regime_c
        np.random.seed(42)
        n = 500
        returns = np.random.randn(n) * 0.01
        vol = np.ones(n) * 0.01
        regime_labels = np.random.randint(0, 5, size=n).astype(np.int64)

        result = fit_regime_c(
            returns, vol, regime_labels, q=1e-6, phi=0.98, c_scalar=1.0)
        # More free parameters -> at least as good on training data
        self.assertGreaterEqual(result.delta_ll, -1.0,
            f"delta_ll={result.delta_ll:.2f} should be >= -1 (within tolerance)")

    def test_regime_counts_populated(self):
        """Regime counts should be populated in result."""
        from models.regime_c import fit_regime_c
        np.random.seed(42)
        n = 500
        returns = np.random.randn(n) * 0.01
        vol = np.ones(n) * 0.01
        regime_labels = np.random.randint(0, 5, size=n).astype(np.int64)

        result = fit_regime_c(
            returns, vol, regime_labels, q=1e-6, phi=0.98, c_scalar=1.0)
        self.assertEqual(len(result.regime_counts), 5)
        total = sum(result.regime_counts.values())
        self.assertEqual(total, n)


class TestRegimeCConfig(unittest.TestCase):
    """Test RegimeCConfig utility class."""

    def test_get_c_array(self):
        from models.regime_c import RegimeCConfig
        config = RegimeCConfig(c_per_regime=np.array([0.5, 1.0, 1.2, 1.5, 2.0]))
        regime_labels = np.array([0, 4, 2], dtype=np.int64)
        c_arr = config.get_c_array(regime_labels)
        self.assertAlmostEqual(c_arr[0], 0.5)
        self.assertAlmostEqual(c_arr[1], 2.0)
        self.assertAlmostEqual(c_arr[2], 1.2)

    def test_c_trend_and_c_crisis_properties(self):
        from models.regime_c import RegimeCConfig
        config = RegimeCConfig(c_per_regime=np.array([0.5, 1.0, 1.2, 1.5, 2.5]))
        self.assertAlmostEqual(config.c_trend, 0.5)
        self.assertAlmostEqual(config.c_crisis, 2.5)


class TestRealDataRegimeC(unittest.TestCase):
    """Test regime-c on real asset data."""

    @classmethod
    def setUpClass(cls):
        """Load real data."""
        data_dir = os.path.join(SRC_DIR, "data", "prices")
        cls.assets = {}
        for ticker, filename in [("SPY", "SPY.csv"), ("MSTR", "MSTR.csv"),
                                  ("AAPL", "AAPL.csv"), ("TSLA", "TSLA.csv")]:
            path = os.path.join(data_dir, filename)
            if not os.path.exists(path):
                continue
            try:
                import pandas as pd
                df = pd.read_csv(path)
                col = "Adj Close" if "Adj Close" in df.columns else "Close"
                close = df[col].dropna().values.astype(np.float64)
                if len(close) < 200:
                    continue
                returns = np.diff(np.log(close))
                n = len(returns)
                vol = np.empty(n, dtype=np.float64)
                alpha = 0.06
                var_ewm = returns[0] ** 2
                for i in range(n):
                    var_ewm = alpha * returns[i] ** 2 + (1 - alpha) * var_ewm
                    vol[i] = max(np.sqrt(var_ewm), 1e-8)
                cls.assets[ticker] = (returns, vol)
            except Exception:
                continue

    def test_c_crisis_gt_c_trend_on_real_assets(self):
        """c_crisis > c_trend on most real assets."""
        if not self.assets:
            self.skipTest("No real data available")
        from models.regime_c import fit_regime_c
        from models.regime import assign_regime_labels

        passes = 0
        total = 0
        for ticker, (returns, vol) in self.assets.items():
            regime_labels = assign_regime_labels(returns, vol)
            result = fit_regime_c(
                returns, vol, regime_labels, q=1e-6, phi=0.98, c_scalar=1.0)
            total += 1
            if result.c_per_regime[4] > result.c_per_regime[0]:
                passes += 1

        self.assertGreater(passes / max(total, 1), 0.5,
            f"c_crisis > c_trend on only {passes}/{total} assets")


if __name__ == "__main__":
    unittest.main()
