"""
===============================================================================
NUMBA KALMAN FILTER TESTS
===============================================================================

Comprehensive tests for Numba JIT optimization of Kalman filters.

Coverage:
    - Gaussian vs φ-Gaussian correctness vs Python
    - φ-Student-t correctness across ν ∈ {4, 6, 8, 12, 20}
    - Momentum-augmented Gaussian correctness
    - Momentum-augmented φ-Student-t correctness
    - Batch ν execution consistency
    - Extreme volatility mismatch stability
    - Low-ν heavy-tail stability
    - Log-likelihood clamping behavior
    - Hierarchical λ paths (Hλ←, Hλ→)

ARCHITECTURAL INVARIANT:
    There is NO bare Student-t model. All Student-t tests use φ-Student-t.
    Any test implying a bare Student-t model is invalid.

Author: Quantitative Systems Team
Date: 2026-02-04
"""

import pytest
import numpy as np
import time
from typing import List

# Discrete ν grid for Student-t BMA
NU_GRID: List[float] = [4, 6, 8, 12, 20]


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate reproducible test data."""
    np.random.seed(42)
    n = 10_000
    returns = np.random.randn(n) * 0.02
    vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
    momentum = np.random.randn(n) * 0.5  # Normalized momentum signal
    return returns, vol, momentum


@pytest.fixture
def short_sample_data():
    """Generate shorter reproducible test data for quick tests."""
    np.random.seed(42)
    n = 1_000
    returns = np.random.randn(n) * 0.02
    vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
    momentum = np.random.randn(n) * 0.5
    return returns, vol, momentum


@pytest.fixture
def heavy_tail_data():
    """Generate Student-t distributed returns for tail testing."""
    np.random.seed(123)
    n = 5_000
    # True ν=4 (heavy tails)
    returns = np.random.standard_t(df=4, size=n) * 0.02
    vol = np.abs(np.random.randn(n)) * 0.01 + 0.01
    return returns, vol


# =============================================================================
# AVAILABILITY CHECKS
# =============================================================================

def _numba_available() -> bool:
    """Check if Numba is available."""
    try:
        from models.numba_wrappers import is_numba_available
        return is_numba_available()
    except ImportError:
        return False


# =============================================================================
# CORRECTNESS TESTS
# =============================================================================

class TestGaussianCorrectness:
    """Verify Gaussian filter Numba results match pure-Python implementation."""
    
    def test_gaussian_filter_matches(self, sample_data):
        """Gaussian filter: Numba matches Python."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.gaussian import GaussianDriftModel
        from models.numba_wrappers import run_gaussian_filter
        
        returns, vol, _ = sample_data
        q, c = 1e-6, 1.0
        
        # Python implementation
        mu_py, P_py, ll_py = GaussianDriftModel._filter_python(returns, vol, q, c)
        
        # Numba implementation
        mu_nb, P_nb, ll_nb = run_gaussian_filter(returns, vol, q, c)
        
        np.testing.assert_allclose(mu_nb, mu_py, rtol=1e-10)
        np.testing.assert_allclose(P_nb, P_py, rtol=1e-10)
        np.testing.assert_allclose(ll_nb, ll_py, rtol=1e-8)
    
    def test_phi_gaussian_filter_matches(self, sample_data):
        """φ-Gaussian filter: Numba matches Python."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.gaussian import GaussianDriftModel
        from models.numba_wrappers import run_phi_gaussian_filter
        
        returns, vol, _ = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        # Python implementation
        mu_py, P_py, ll_py = GaussianDriftModel._filter_phi_python(returns, vol, q, c, phi)
        
        # Numba implementation
        mu_nb, P_nb, ll_nb = run_phi_gaussian_filter(returns, vol, q, c, phi)
        
        np.testing.assert_allclose(mu_nb, mu_py, rtol=1e-10)
        np.testing.assert_allclose(P_nb, P_py, rtol=1e-10)
        np.testing.assert_allclose(ll_nb, ll_py, rtol=1e-8)


class TestPhiStudentTCorrectness:
    """Verify φ-Student-t filter Numba results match pure-Python implementation.
    
    Note: There is NO bare Student-t test. All Student-t behavior requires φ.
    """
    
    @pytest.mark.parametrize("nu", NU_GRID)
    def test_phi_student_t_filter_matches_all_nu(self, sample_data, nu):
        """φ-Student-t filter matches Python for all ν in discrete grid."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.phi_student_t import PhiStudentTDriftModel
        from models.numba_wrappers import run_phi_student_t_filter
        
        returns, vol, _ = sample_data
        q, c, phi = 1e-6, 1.0, 0.5
        
        # Python implementation
        mu_py, P_py, ll_py = PhiStudentTDriftModel._filter_phi_python(
            returns, vol, q, c, phi, nu
        )
        
        # Numba implementation
        mu_nb, P_nb, ll_nb = run_phi_student_t_filter(
            returns, vol, q, c, phi, nu
        )
        
        np.testing.assert_allclose(mu_nb, mu_py, rtol=1e-8)
        np.testing.assert_allclose(P_nb, P_py, rtol=1e-8)
        # Slightly looser for likelihood due to gamma computation
        np.testing.assert_allclose(ll_nb, ll_py, rtol=1e-6)
    
    def test_phi_student_t_batch_consistency(self, short_sample_data):
        """Batch ν execution produces same results as individual calls."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.phi_student_t import PhiStudentTDriftModel
        
        returns, vol, _ = short_sample_data
        q, c, phi = 1e-6, 1.0, 0.5
        
        # Batch execution
        batch_results = PhiStudentTDriftModel.filter_phi_batch(
            returns, vol, q, c, phi, NU_GRID
        )
        
        # Individual execution
        for nu in NU_GRID:
            mu_single, P_single, ll_single = PhiStudentTDriftModel.filter_phi(
                returns, vol, q, c, phi, nu
            )
            
            mu_batch, P_batch, ll_batch = batch_results[nu]
            
            np.testing.assert_allclose(mu_batch, mu_single, rtol=1e-12)
            np.testing.assert_allclose(P_batch, P_single, rtol=1e-12)
            np.testing.assert_allclose(ll_batch, ll_single, rtol=1e-12)


class TestMomentumCorrectness:
    """Verify momentum-augmented filter Numba results match Python."""
    
    def test_momentum_phi_gaussian_matches(self, sample_data):
        """Momentum-augmented φ-Gaussian filter matches."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.momentum_augmented import MomentumPhiGaussianFilter
        from models.numba_wrappers import run_momentum_phi_gaussian_filter
        
        returns, vol, momentum = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        momentum_weight = 0.1
        momentum_adjustment = momentum_weight * momentum * vol
        
        # Python implementation
        mu_py, P_py, ll_py = MomentumPhiGaussianFilter._filter_python(
            returns, vol, q, c, phi, momentum_adjustment
        )
        
        # Numba implementation
        mu_nb, P_nb, ll_nb = run_momentum_phi_gaussian_filter(
            returns, vol, q, c, phi, momentum_adjustment
        )
        
        np.testing.assert_allclose(mu_nb, mu_py, rtol=1e-10)
        np.testing.assert_allclose(P_nb, P_py, rtol=1e-10)
        np.testing.assert_allclose(ll_nb, ll_py, rtol=1e-8)
    
    @pytest.mark.parametrize("nu", [4, 8, 20])
    def test_momentum_phi_student_t_matches(self, sample_data, nu):
        """Momentum-augmented φ-Student-t filter matches (GLDW/MAGD/BKSY/ASTS)."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.momentum_augmented import MomentumPhiStudentTFilter
        from models.numba_wrappers import run_momentum_phi_student_t_filter
        
        returns, vol, momentum = sample_data
        q, c, phi = 1e-6, 1.0, 0.5
        momentum_weight = 0.1
        momentum_adjustment = momentum_weight * momentum * vol
        
        # Python implementation
        mu_py, P_py, ll_py = MomentumPhiStudentTFilter._filter_python(
            returns, vol, q, c, phi, nu, momentum_adjustment
        )
        
        # Numba implementation
        mu_nb, P_nb, ll_nb = run_momentum_phi_student_t_filter(
            returns, vol, q, c, phi, nu, momentum_adjustment
        )
        
        np.testing.assert_allclose(mu_nb, mu_py, rtol=1e-8)
        np.testing.assert_allclose(P_nb, P_py, rtol=1e-8)
        np.testing.assert_allclose(ll_nb, ll_py, rtol=1e-6)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Verify Numba provides meaningful speedup."""
    
    def test_gaussian_speedup(self, sample_data):
        """Gaussian filter achieves >= 5× speedup."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.gaussian import GaussianDriftModel
        from models.numba_wrappers import run_gaussian_filter
        
        returns, vol, _ = sample_data
        q, c = 1e-6, 1.0
        
        # Warm-up JIT
        run_gaussian_filter(returns, vol, q, c)
        
        n_runs = 50
        
        start = time.perf_counter()
        for _ in range(n_runs):
            run_gaussian_filter(returns, vol, q, c)
        numba_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(n_runs):
            GaussianDriftModel._filter_python(returns, vol, q, c)
        python_time = time.perf_counter() - start
        
        speedup = python_time / numba_time
        print(f"\nGaussian filter speedup: {speedup:.1f}×")
        
        assert speedup >= 5.0, f"Expected >= 5× speedup, got {speedup:.1f}×"
    
    def test_phi_student_t_speedup(self, sample_data):
        """φ-Student-t filter achieves >= 5× speedup."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.phi_student_t import PhiStudentTDriftModel
        from models.numba_wrappers import run_phi_student_t_filter
        
        returns, vol, _ = sample_data
        q, c, phi, nu = 1e-6, 1.0, 0.5, 6.0
        
        # Warm-up
        run_phi_student_t_filter(returns, vol, q, c, phi, nu)
        
        n_runs = 50
        
        start = time.perf_counter()
        for _ in range(n_runs):
            run_phi_student_t_filter(returns, vol, q, c, phi, nu)
        numba_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(n_runs):
            PhiStudentTDriftModel._filter_phi_python(returns, vol, q, c, phi, nu)
        python_time = time.perf_counter() - start
        
        speedup = python_time / numba_time
        print(f"\nφ-Student-t filter speedup: {speedup:.1f}×")
        
        assert speedup >= 5.0, f"Expected >= 5× speedup, got {speedup:.1f}×"


# =============================================================================
# NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Verify numerical stability in edge cases."""
    
    def test_extreme_volatility_mismatch(self):
        """Handle vol/return mismatch without overflow."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.numba_wrappers import run_gaussian_filter
        
        np.random.seed(42)
        n = 1000
        returns = np.random.randn(n) * 0.5  # Large moves
        vol = np.ones(n) * 0.001  # Tiny vol (mismatch)
        
        mu, P, ll = run_gaussian_filter(returns, vol, 1e-6, 1.0)
        
        assert np.all(np.isfinite(mu)), "mu contains NaN/Inf"
        assert np.all(np.isfinite(P)), "P contains NaN/Inf"
        assert np.isfinite(ll), "ll is NaN/Inf"
    
    @pytest.mark.parametrize("nu", [4, 6])  # Low ν = heavy tails
    def test_low_nu_stability(self, heavy_tail_data, nu):
        """φ-Student-t with low ν remains stable."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.numba_wrappers import run_phi_student_t_filter
        
        returns, vol = heavy_tail_data
        
        mu, P, ll = run_phi_student_t_filter(returns, vol, 1e-6, 1.0, 0.5, nu)
        
        assert np.all(np.isfinite(mu)), f"mu contains NaN/Inf at ν={nu}"
        assert np.all(np.isfinite(P)), f"P contains NaN/Inf at ν={nu}"
        assert np.isfinite(ll), f"ll is NaN/Inf at ν={nu}"
        # Likelihood should be negative (log-prob)
        assert ll < 0, f"Expected negative log-likelihood at ν={nu}"
    
    def test_gamma_precomputation_accuracy(self):
        """Verify gamma precomputation matches scipy."""
        from scipy.special import gammaln
        from models.numba_wrappers import precompute_gamma_values
        
        for nu in NU_GRID:
            log_g1, log_g2 = precompute_gamma_values(nu)
            
            expected_g1 = gammaln(nu / 2.0)
            expected_g2 = gammaln((nu + 1.0) / 2.0)
            
            np.testing.assert_allclose(log_g1, expected_g1, rtol=1e-14)
            np.testing.assert_allclose(log_g2, expected_g2, rtol=1e-14)
    
    def test_ll_clamping_prevents_domination(self, sample_data):
        """Verify LL clamping prevents single outlier domination."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.numba_wrappers import run_gaussian_filter
        
        returns, vol, _ = sample_data
        
        # Inject extreme outlier
        returns_with_outlier = returns.copy()
        returns_with_outlier[5000] = 10.0  # 500σ event
        
        mu_normal, P_normal, ll_normal = run_gaussian_filter(returns, vol, 1e-6, 1.0)
        mu_outlier, P_outlier, ll_outlier = run_gaussian_filter(
            returns_with_outlier, vol, 1e-6, 1.0
        )
        
        # LL should decrease but not catastrophically
        ll_diff = ll_normal - ll_outlier
        assert ll_diff > 0, "Outlier should reduce likelihood"
        assert ll_diff < 100, "Clamping should prevent huge drop"


# =============================================================================
# MODEL VARIANT COVERAGE TESTS
# =============================================================================

class TestModelVariantCoverage:
    """Ensure all documented model variants work with Numba."""
    
    @pytest.mark.parametrize("model_config", [
        {"name": "CRSP", "base": "phi_gaussian", "evt": "M", "cvar": 0.20},
        {"name": "CELH", "base": "phi_gaussian", "evt": "H", "cvar": 0.17},
        {"name": "DPRO", "base": "phi_gaussian", "evt": "H", "cvar": 0.19},
    ])
    def test_phi_gaussian_augmented_variants(self, sample_data, model_config):
        """φ-Gaussian+Mom variants run without error."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.momentum_augmented import MomentumPhiGaussianFilter
        
        returns, vol, momentum = sample_data
        q, c, phi = 1e-6, 1.0, 0.7
        
        mu, P, ll = MomentumPhiGaussianFilter.filter(
            returns, vol, q, c, phi, momentum, momentum_weight=0.1
        )
        
        assert np.all(np.isfinite(mu)), f"{model_config['name']} produced NaN in mu"
        assert np.all(np.isfinite(P)), f"{model_config['name']} produced NaN in P"
        assert np.isfinite(ll), f"{model_config['name']} produced NaN in ll"
    
    @pytest.mark.parametrize("model_config", [
        {"name": "GLDW", "base": "phi_student_t", "hlambda": "none", "evt": "H", "cvar": 0.17},
        {"name": "MAGD", "base": "phi_student_t", "hlambda": "backward", "evt": "M", "cvar": 0.17},
        {"name": "BKSY", "base": "phi_student_t", "hlambda": "forward", "evt": "H", "cvar": 0.17},
        {"name": "ASTS", "base": "phi_student_t", "hlambda": "forward", "evt": "H", "cvar": 0.14},
    ])
    def test_phi_student_t_augmented_variants(self, sample_data, model_config):
        """φ-Student-t+Mom+Hλ variants run without error."""
        if not _numba_available():
            pytest.skip("Numba not available")
        
        from models.momentum_augmented import MomentumPhiStudentTFilter
        
        returns, vol, momentum = sample_data
        q, c, phi, nu = 1e-6, 1.0, 0.5, 6.0
        
        # Simulate hierarchical λ effect
        n = len(returns)
        if model_config['hlambda'] == 'backward':
            h_lambda = np.exp(-0.1 * np.arange(n) / n)
        elif model_config['hlambda'] == 'forward':
            h_lambda = np.exp(0.1 * np.arange(n) / n)
        else:
            h_lambda = None
        
        mu, P, ll = MomentumPhiStudentTFilter.filter(
            returns, vol, q, c, phi, nu, momentum,
            momentum_weight=0.1,
            hierarchical_lambda=h_lambda,
            lambda_direction=model_config['hlambda'],
        )
        
        assert np.all(np.isfinite(mu)), f"{model_config['name']} produced NaN in mu"
        assert np.all(np.isfinite(P)), f"{model_config['name']} produced NaN in P"
        assert np.isfinite(ll), f"{model_config['name']} produced NaN in ll"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with tune.py and signals.py."""
    
    def test_gaussian_model_integration(self, short_sample_data):
        """GaussianDriftModel uses Numba transparently."""
        from models.gaussian import GaussianDriftModel
        
        returns, vol, _ = short_sample_data
        
        # This should use Numba if available, Python otherwise
        mu, P, ll = GaussianDriftModel.filter(returns, vol, 1e-6, 1.0)
        
        assert np.all(np.isfinite(mu))
        assert np.isfinite(ll)
    
    def test_phi_student_t_model_integration(self, short_sample_data):
        """PhiStudentTDriftModel uses Numba transparently."""
        from models.phi_student_t import PhiStudentTDriftModel
        
        returns, vol, _ = short_sample_data
        
        # This should use Numba if available, Python otherwise
        mu, P, ll = PhiStudentTDriftModel.filter_phi(
            returns, vol, 1e-6, 1.0, 0.5, 6.0
        )
        
        assert np.all(np.isfinite(mu))
        assert np.isfinite(ll)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
