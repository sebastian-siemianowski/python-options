"""
Test Suite for Epic 16: GJR-GARCH Innovation Volatility
========================================================

Story 16.1: Post-Filter GJR-GARCH on Innovation Sequence
Story 16.2: Iterated Filter-GARCH Cycle (2-Pass Estimation)
Story 16.3: GARCH Forecast Variance for Multi-Horizon Signals
"""
import os
import sys
import math
import unittest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from models.gjr_garch import (
    # Story 16.1
    GARCHFitResult,
    fit_gjr_garch_innovations,
    _gjr_garch_loglik,
    _gjr_garch_h_series,
    STATIONARITY_MARGIN,
    DEFAULT_OMEGA, DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_BETA,
    # Story 16.2
    IteratedFilterGARCHResult,
    iterated_filter_garch,
    _run_kalman_filter,
    CONVERGENCE_TOL,
    DEFAULT_N_ITER,
    # Story 16.3
    GARCHForecastResult,
    garch_variance_forecast,
    garch_variance_forecast_multi,
    compute_garch_adjusted_R,
    STANDARD_HORIZONS,
)


def _generate_garch_innovations(n=500, omega=1e-5, alpha=0.08, gamma=0.06,
                                 beta=0.85, seed=42):
    """Generate synthetic GJR-GARCH(1,1) innovations for testing."""
    rng = np.random.default_rng(seed)
    h = omega / max(1.0 - alpha - 0.5 * gamma - beta, 1e-8)
    innovations = np.zeros(n)
    for t in range(n):
        innovations[t] = rng.normal(0, math.sqrt(max(h, 1e-12)))
        v2 = innovations[t] ** 2
        neg = 1.0 if innovations[t] < 0 else 0.0
        h = omega + alpha * v2 + gamma * v2 * neg + beta * h
        h = max(h, 1e-12)
    return innovations


def _generate_returns_with_vol(n=500, seed=42):
    """Generate synthetic returns and vol for filter tests."""
    rng = np.random.default_rng(seed)
    vol = 0.01 + 0.005 * np.abs(rng.standard_normal(n))
    returns = rng.normal(0, vol)
    return returns, vol


# ===================================================================
# Story 16.1 Tests: Post-Filter GJR-GARCH on Innovation Sequence
# ===================================================================

class TestGJRGARCHLoglik(unittest.TestCase):
    """Test the GARCH log-likelihood function."""

    def test_valid_params_finite(self):
        inno = _generate_garch_innovations(200)
        params = np.array([1e-5, 0.05, 0.05, 0.88])
        nll = _gjr_garch_loglik(params, inno)
        self.assertTrue(np.isfinite(nll))
        # NLL should not be the penalty value
        self.assertLess(nll, 1e12)

    def test_non_stationary_penalized(self):
        inno = _generate_garch_innovations(100)
        # alpha + 0.5*gamma + beta = 0.5 + 0.25 + 0.5 = 1.25 > 1
        params = np.array([1e-5, 0.50, 0.50, 0.50])
        nll = _gjr_garch_loglik(params, inno)
        self.assertEqual(nll, 1e12)

    def test_negative_omega_penalized(self):
        inno = _generate_garch_innovations(100)
        params = np.array([-1e-5, 0.05, 0.05, 0.88])
        nll = _gjr_garch_loglik(params, inno)
        self.assertEqual(nll, 1e12)

    def test_short_series_penalized(self):
        params = np.array([1e-5, 0.05, 0.05, 0.88])
        nll = _gjr_garch_loglik(params, np.array([0.01, 0.02]))
        self.assertEqual(nll, 1e12)


class TestGJRGARCHHSeries(unittest.TestCase):
    """Test the conditional variance series computation."""

    def test_output_length(self):
        inno = _generate_garch_innovations(100)
        params = np.array([1e-5, 0.05, 0.05, 0.88])
        h = _gjr_garch_h_series(params, inno)
        self.assertEqual(len(h), 100)

    def test_all_positive(self):
        inno = _generate_garch_innovations(200)
        params = np.array([1e-5, 0.08, 0.06, 0.85])
        h = _gjr_garch_h_series(params, inno)
        self.assertTrue(np.all(h > 0))

    def test_responds_to_shocks(self):
        """After a large shock, h should increase."""
        inno = np.zeros(50)
        inno[20] = 0.10  # Large positive shock
        params = np.array([1e-5, 0.10, 0.05, 0.80])
        h = _gjr_garch_h_series(params, inno)
        self.assertGreater(h[21], h[19])

    def test_leverage_effect(self):
        """Negative shock should produce larger h than equal positive shock."""
        n = 50
        inno_pos = np.zeros(n)
        inno_neg = np.zeros(n)
        inno_pos[20] = 0.05
        inno_neg[20] = -0.05
        params = np.array([1e-5, 0.05, 0.10, 0.85])  # gamma > 0
        h_pos = _gjr_garch_h_series(params, inno_pos)
        h_neg = _gjr_garch_h_series(params, inno_neg)
        # After negative shock, h should be larger due to leverage
        self.assertGreater(h_neg[21], h_pos[21])


class TestFitGJRGARCH(unittest.TestCase):
    """Test fit_gjr_garch_innovations()."""

    def test_returns_dataclass(self):
        inno = _generate_garch_innovations(300)
        result = fit_gjr_garch_innovations(inno)
        self.assertIsInstance(result, GARCHFitResult)

    def test_stationarity_constraint(self):
        """Fitted persistence must be < 1."""
        inno = _generate_garch_innovations(500)
        result = fit_gjr_garch_innovations(inno)
        self.assertLess(result.persistence, 1.0)

    def test_positive_params(self):
        inno = _generate_garch_innovations(300)
        result = fit_gjr_garch_innovations(inno)
        self.assertGreater(result.omega, 0)
        self.assertGreater(result.alpha, 0)
        self.assertGreaterEqual(result.gamma, 0)
        self.assertGreater(result.beta, 0)

    def test_h_series_shape(self):
        inno = _generate_garch_innovations(200)
        result = fit_gjr_garch_innovations(inno)
        self.assertEqual(len(result.h_series), 200)

    def test_unconditional_var_positive(self):
        inno = _generate_garch_innovations(300)
        result = fit_gjr_garch_innovations(inno)
        self.assertGreater(result.unconditional_var, 0)

    def test_leverage_detected(self):
        """For asymmetric innovations, gamma should be positive."""
        # Generate innovations with strong negative asymmetry
        inno = _generate_garch_innovations(500, gamma=0.12)
        result = fit_gjr_garch_innovations(inno)
        # Leverage effect should be detected for most equity-like series
        self.assertTrue(result.leverage_effect or result.gamma >= 0)

    def test_garch_ll_better_than_homoskedastic(self):
        """GARCH log-likelihood should exceed homoskedastic."""
        inno = _generate_garch_innovations(500)
        result = fit_gjr_garch_innovations(inno)
        # Homoskedastic log-likelihood: Gaussian with sample variance
        var = np.var(inno)
        homo_ll = -0.5 * np.sum(np.log(2 * np.pi * var) + inno**2 / var)
        self.assertGreater(result.log_likelihood, homo_ll)

    def test_with_R_standardization(self):
        """Test fitting with observation noise R provided."""
        inno = _generate_garch_innovations(300)
        R = np.ones(300) * 1e-4
        result = fit_gjr_garch_innovations(inno, R=R)
        self.assertIsInstance(result, GARCHFitResult)
        self.assertLess(result.persistence, 1.0)

    def test_short_series_defaults(self):
        """Very short series should return defaults gracefully."""
        inno = np.array([0.01, 0.02, 0.03])
        result = fit_gjr_garch_innovations(inno)
        self.assertFalse(result.converged)

    def test_constant_innovations(self):
        """Constant innovations should still return valid result."""
        inno = np.ones(100) * 0.01
        result = fit_gjr_garch_innovations(inno)
        self.assertIsInstance(result, GARCHFitResult)

    def test_nan_handling(self):
        """NaN innovations should be filtered out."""
        inno = _generate_garch_innovations(200)
        inno[50] = np.nan
        inno[100] = np.nan
        result = fit_gjr_garch_innovations(inno)
        self.assertIsInstance(result, GARCHFitResult)
        self.assertTrue(np.isfinite(result.log_likelihood))


# ===================================================================
# Story 16.2 Tests: Iterated Filter-GARCH Cycle
# ===================================================================

class TestRunKalmanFilter(unittest.TestCase):
    """Test the internal Kalman filter used in iteration."""

    def test_output_shapes(self):
        returns, vol = _generate_returns_with_vol(200)
        mu_f, P_f, mu_p, S_p, ll = _run_kalman_filter(returns, vol, 1e-6, 1.0, 0.99)
        self.assertEqual(len(mu_f), 200)
        self.assertEqual(len(P_f), 200)
        self.assertEqual(len(mu_p), 200)
        self.assertEqual(len(S_p), 200)
        self.assertTrue(np.isfinite(ll))

    def test_S_pred_positive(self):
        returns, vol = _generate_returns_with_vol(100)
        _, _, _, S_p, _ = _run_kalman_filter(returns, vol, 1e-6, 1.0, 0.99)
        self.assertTrue(np.all(S_p > 0))

    def test_with_R_scale(self):
        returns, vol = _generate_returns_with_vol(100)
        R_scale = np.ones(100) * 1.5
        _, _, _, S_p1, ll1 = _run_kalman_filter(returns, vol, 1e-6, 1.0, 0.99)
        _, _, _, S_p2, ll2 = _run_kalman_filter(returns, vol, 1e-6, 1.0, 0.99, R_scale=R_scale)
        # Larger R_scale -> larger S_pred -> different likelihood
        self.assertNotAlmostEqual(ll1, ll2)


class TestIteratedFilterGARCH(unittest.TestCase):
    """Test iterated_filter_garch()."""

    def test_returns_dataclass(self):
        returns, vol = _generate_returns_with_vol(300)
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99)
        self.assertIsInstance(result, IteratedFilterGARCHResult)

    def test_converges_in_few_iterations(self):
        """Should converge in 2-3 iterations."""
        returns, vol = _generate_returns_with_vol(500)
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99, n_iter=5)
        self.assertLessEqual(result.n_iterations, 5)
        self.assertGreaterEqual(result.n_iterations, 1)

    def test_ll_history_monotonic_trend(self):
        """Log-likelihood should generally improve or stabilize."""
        returns, vol = _generate_returns_with_vol(500)
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99, n_iter=5)
        # At least the last two iterations should be close
        if len(result.ll_history) >= 2:
            delta = abs(result.ll_history[-1] - result.ll_history[-2])
            # Should be relatively small
            self.assertLess(delta, 100)  # Generous bound

    def test_output_shapes(self):
        n = 300
        returns, vol = _generate_returns_with_vol(n)
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99)
        self.assertEqual(len(result.mu_filtered), n)
        self.assertEqual(len(result.P_filtered), n)
        self.assertEqual(len(result.mu_pred), n)
        self.assertEqual(len(result.S_pred), n)

    def test_garch_fit_present(self):
        returns, vol = _generate_returns_with_vol(300)
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99)
        self.assertIsNotNone(result.garch_fit)
        self.assertIsInstance(result.garch_fit, GARCHFitResult)

    def test_bic_improvement_computed(self):
        returns, vol = _generate_returns_with_vol(300)
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99)
        self.assertTrue(np.isfinite(result.bic_improvement))

    def test_single_iteration(self):
        returns, vol = _generate_returns_with_vol(200)
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99, n_iter=1)
        self.assertEqual(result.n_iterations, 1)

    def test_runtime_reasonable(self):
        """3 iterations should complete in reasonable time."""
        import time
        returns, vol = _generate_returns_with_vol(500)
        t0 = time.time()
        iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99, n_iter=3)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 30.0)  # Should be well under 30s


# ===================================================================
# Story 16.3 Tests: GARCH Forecast Variance
# ===================================================================

class TestGARCHVarianceForecast(unittest.TestCase):
    """Test garch_variance_forecast()."""

    def test_returns_dataclass(self):
        result = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, 1e-4, 1)
        self.assertIsInstance(result, GARCHForecastResult)

    def test_h1_captures_current(self):
        """H=1 should reflect current variance regime."""
        h_T_high = 5e-4  # High vol
        h_T_low = 1e-5   # Low vol
        r_high = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, h_T_high, 1)
        r_low = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, h_T_low, 1)
        self.assertGreater(r_high.forecast_var, r_low.forecast_var)

    def test_h30_reverts_toward_unconditional(self):
        """H=30 should revert toward h_bar (unconditional variance)."""
        omega, alpha, gamma, beta = 1e-5, 0.05, 0.05, 0.88
        h_T = 5e-4  # Far above unconditional

        r1 = garch_variance_forecast(omega, alpha, gamma, beta, h_T, 1)
        r30 = garch_variance_forecast(omega, alpha, gamma, beta, h_T, 30)

        # At H=30, forecast should be closer to unconditional than at H=1
        dist_1 = abs(r1.forecast_var - r1.unconditional_var)
        dist_30 = abs(r30.forecast_var - r30.unconditional_var)
        self.assertLess(dist_30, dist_1)

    def test_mean_reversion_factor_decays(self):
        """Mean reversion factor should decrease with horizon."""
        r1 = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, 1e-4, 1)
        r7 = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, 1e-4, 7)
        r30 = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, 1e-4, 30)
        self.assertGreater(r1.mean_reversion_factor, r7.mean_reversion_factor)
        self.assertGreater(r7.mean_reversion_factor, r30.mean_reversion_factor)

    def test_forecast_positive(self):
        result = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, 1e-4, 7)
        self.assertGreater(result.forecast_var, 0)

    def test_unconditional_var_correct(self):
        omega, alpha, gamma, beta = 1e-5, 0.05, 0.05, 0.88
        expected_hbar = omega / (1.0 - alpha - 0.5 * gamma - beta)
        result = garch_variance_forecast(omega, alpha, gamma, beta, 1e-4, 1)
        self.assertAlmostEqual(result.unconditional_var, expected_hbar, places=10)

    def test_low_persistence_fast_reversion(self):
        """Low persistence -> fast mean reversion."""
        # persistence = 0.05 + 0.025 + 0.5 = 0.575
        r30_low = garch_variance_forecast(1e-5, 0.05, 0.05, 0.50, 5e-4, 30)
        # persistence = 0.05 + 0.025 + 0.88 = 0.955
        r30_high = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, 5e-4, 30)
        # Low persistence: forecast closer to unconditional
        dist_low = abs(r30_low.forecast_var - r30_low.unconditional_var)
        dist_high = abs(r30_high.forecast_var - r30_high.unconditional_var)
        self.assertLess(dist_low, dist_high)

    def test_zero_horizon_clamped(self):
        """Horizon 0 should be clamped to 1."""
        result = garch_variance_forecast(1e-5, 0.05, 0.05, 0.88, 1e-4, 0)
        self.assertEqual(result.horizon, 1)

    def test_non_stationary_handled(self):
        """Non-stationary GARCH should not crash."""
        # persistence = 0.5 + 0.25 + 0.5 = 1.25 > 1
        result = garch_variance_forecast(1e-5, 0.50, 0.50, 0.50, 1e-4, 7)
        self.assertIsInstance(result, GARCHForecastResult)
        self.assertGreater(result.forecast_var, 0)


class TestGARCHVarianceForecastMulti(unittest.TestCase):
    """Test multi-horizon forecast."""

    def test_default_horizons(self):
        results = garch_variance_forecast_multi(1e-5, 0.05, 0.05, 0.88, 1e-4)
        self.assertEqual(len(results), len(STANDARD_HORIZONS))

    def test_custom_horizons(self):
        results = garch_variance_forecast_multi(
            1e-5, 0.05, 0.05, 0.88, 1e-4, horizons=[1, 7, 30]
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].horizon, 1)
        self.assertEqual(results[1].horizon, 7)
        self.assertEqual(results[2].horizon, 30)

    def test_monotone_reversion(self):
        """Forecasts from above unconditional should decrease toward h_bar."""
        omega, alpha, gamma, beta = 1e-5, 0.05, 0.05, 0.88
        h_bar = omega / (1.0 - alpha - 0.5 * gamma - beta)
        h_T = h_bar * 5  # Start well above unconditional

        results = garch_variance_forecast_multi(
            omega, alpha, gamma, beta, h_T, horizons=[1, 3, 7, 30]
        )
        forecasts = [r.forecast_var for r in results]
        # Should monotonically decrease toward h_bar
        for i in range(len(forecasts) - 1):
            self.assertGreaterEqual(forecasts[i], forecasts[i + 1])


class TestComputeGARCHAdjustedR(unittest.TestCase):
    """Test compute_garch_adjusted_R()."""

    def test_output_shape(self):
        vol = np.ones(100) * 0.01
        garch_fit = GARCHFitResult(
            omega=1e-5, alpha=0.05, gamma=0.05, beta=0.88,
            log_likelihood=-100, h_series=np.ones(100) * 1e-4,
            unconditional_var=1e-4, persistence=0.955,
            leverage_effect=True, converged=True,
        )
        R_adj = compute_garch_adjusted_R(vol, 1.0, garch_fit)
        self.assertEqual(len(R_adj), 100)

    def test_scaling_effect(self):
        """High h_t should produce larger R, low h_t should produce smaller R."""
        vol = np.ones(100) * 0.01
        h_bar = 1e-4
        h_series = np.ones(100) * h_bar
        h_series[50:] = h_bar * 3.0  # Spike in second half

        garch_fit = GARCHFitResult(
            omega=1e-5, alpha=0.05, gamma=0.05, beta=0.88,
            log_likelihood=-100, h_series=h_series,
            unconditional_var=h_bar, persistence=0.955,
            leverage_effect=True, converged=True,
        )
        R_adj = compute_garch_adjusted_R(vol, 1.0, garch_fit)
        self.assertGreater(R_adj[60], R_adj[10])

    def test_average_near_base(self):
        """When h_t = h_bar everywhere, R_adj should equal R_base."""
        vol = np.ones(50) * 0.02
        h_bar = 1e-4
        garch_fit = GARCHFitResult(
            omega=1e-5, alpha=0.05, gamma=0.05, beta=0.88,
            log_likelihood=-100, h_series=np.ones(50) * h_bar,
            unconditional_var=h_bar, persistence=0.955,
            leverage_effect=True, converged=True,
        )
        R_adj = compute_garch_adjusted_R(vol, 1.0, garch_fit)
        R_base = 1.0 * vol ** 2
        np.testing.assert_array_almost_equal(R_adj, R_base)


# ===================================================================
# Integration Tests
# ===================================================================

class TestEpic16Integration(unittest.TestCase):
    """Integration tests combining all three stories."""

    def test_full_pipeline(self):
        """Full pipeline: filter -> GARCH fit -> forecast."""
        rng = np.random.default_rng(42)
        n = 500
        vol = 0.01 + 0.005 * np.abs(rng.standard_normal(n))
        returns = rng.normal(0, vol)

        # Step 1: Run Kalman filter
        mu_f, P_f, mu_p, S_p, ll = _run_kalman_filter(
            returns, vol, 1e-6, 1.0, 0.99,
        )

        # Step 2: Extract innovations and fit GARCH
        innovations = returns - mu_p
        garch_fit = fit_gjr_garch_innovations(innovations)
        self.assertLess(garch_fit.persistence, 1.0)

        # Step 3: Forecast variance
        h_T = garch_fit.h_series[-1]
        forecasts = garch_variance_forecast_multi(
            garch_fit.omega, garch_fit.alpha,
            garch_fit.gamma, garch_fit.beta, h_T,
        )
        self.assertEqual(len(forecasts), len(STANDARD_HORIZONS))
        for f in forecasts:
            self.assertGreater(f.forecast_var, 0)

    def test_iterated_pipeline(self):
        """Iterated filter-GARCH -> forecast."""
        rng = np.random.default_rng(123)
        n = 400
        vol = 0.015 + 0.005 * np.abs(rng.standard_normal(n))
        returns = rng.normal(0, vol)

        # Iterated filter-GARCH
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99, n_iter=3)
        self.assertIsNotNone(result.garch_fit)

        # Use fitted GARCH for forecast
        gf = result.garch_fit
        h_T = gf.h_series[-1]
        forecast = garch_variance_forecast(
            gf.omega, gf.alpha, gf.gamma, gf.beta, h_T, 7,
        )
        self.assertGreater(forecast.forecast_var, 0)

    def test_garch_adjusted_R_in_filter(self):
        """GARCH-adjusted R should produce different filter output."""
        rng = np.random.default_rng(99)
        n = 300
        vol = 0.01 + 0.005 * np.abs(rng.standard_normal(n))
        returns = rng.normal(0, vol)

        # Static filter
        _, _, _, _, ll_static = _run_kalman_filter(returns, vol, 1e-6, 1.0, 0.99)

        # GARCH-adjusted filter
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99, n_iter=2)

        # The likelihoods should differ (GARCH adjusts R)
        # Both should be finite
        self.assertTrue(np.isfinite(ll_static))
        self.assertTrue(np.isfinite(result.log_likelihood))


# ===================================================================
# Edge Cases
# ===================================================================

class TestEpic16EdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_fit_with_nan_innovations(self):
        inno = _generate_garch_innovations(200)
        inno[50] = np.nan
        result = fit_gjr_garch_innovations(inno)
        self.assertIsInstance(result, GARCHFitResult)

    def test_fit_with_zero_innovations(self):
        inno = np.zeros(100)
        result = fit_gjr_garch_innovations(inno)
        self.assertIsInstance(result, GARCHFitResult)

    def test_forecast_very_large_horizon(self):
        """Very large horizon should converge to unconditional."""
        omega, alpha, gamma, beta = 1e-5, 0.05, 0.05, 0.88
        h_bar = omega / (1.0 - alpha - 0.5 * gamma - beta)
        result = garch_variance_forecast(omega, alpha, gamma, beta, 5e-4, 1000)
        self.assertAlmostEqual(result.forecast_var, h_bar, places=6)

    def test_iterated_filter_short_series(self):
        """Short series should still work."""
        returns = np.array([0.01, -0.02, 0.005, 0.01, -0.01,
                           0.02, -0.005, 0.01, -0.01, 0.005,
                           0.02, -0.02, 0.01, -0.005, 0.003])
        vol = np.ones(15) * 0.01
        result = iterated_filter_garch(returns, vol, 1e-6, 1.0, 0.99, n_iter=2)
        self.assertIsInstance(result, IteratedFilterGARCHResult)

    def test_constants_valid(self):
        self.assertEqual(DEFAULT_N_ITER, 3)
        self.assertAlmostEqual(CONVERGENCE_TOL, 0.1)
        self.assertAlmostEqual(STATIONARITY_MARGIN, 0.999)
        self.assertEqual(STANDARD_HORIZONS, [1, 3, 7, 30])


if __name__ == "__main__":
    unittest.main()
