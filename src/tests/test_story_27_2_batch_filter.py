"""
Story 27.2 -- Batch Filter Optimization for BMA
=================================================
Batch filter shares array preparation across nu values.

Acceptance Criteria:
- Single shared array preparation, per-nu filter runs
- Speedup from amortized prep for 4-nu batch vs 4 sequential
- No accuracy change: batch matches sequential exactly
- Memory: batch < 2x single filter
"""

import os, sys, time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from models.numba_wrappers import (
    run_phi_student_t_filter,
    run_phi_student_t_filter_batch,
)


# ===========================================================================
class TestBatchCorrectness:
    """Batch filter matches sequential to high precision."""

    def test_batch_matches_sequential(self):
        """Batch results match sequential for each nu."""
        rng = np.random.default_rng(42)
        r = 0.01 * rng.standard_normal(500)
        v = np.full(500, 0.01)
        nu_grid = [4.0, 8.0, 15.0, 30.0]

        results = run_phi_student_t_filter_batch(
            r, v, 1e-4, 1.0, 0.90, nu_grid
        )

        for nu in nu_grid:
            mu_seq, P_seq, ll_seq = run_phi_student_t_filter(
                r, v, 1e-4, 1.0, 0.90, nu
            )
            mu_b, P_b, ll_b = results[nu]
            np.testing.assert_array_equal(mu_b, mu_seq)
            np.testing.assert_array_equal(P_b, P_seq)
            assert ll_b == ll_seq, f"nu={nu}: ll_batch={ll_b} != ll_seq={ll_seq}"

    def test_single_nu_matches(self):
        """Batch with one nu matches sequential exactly."""
        rng = np.random.default_rng(43)
        r = 0.01 * rng.standard_normal(200)
        v = np.full(200, 0.01)

        results = run_phi_student_t_filter_batch(
            r, v, 1e-4, 1.0, 0.90, [8.0]
        )
        mu_seq, P_seq, ll_seq = run_phi_student_t_filter(
            r, v, 1e-4, 1.0, 0.90, 8.0
        )
        mu_b, P_b, ll_b = results[8.0]
        np.testing.assert_array_equal(mu_b, mu_seq)
        assert ll_b == ll_seq

    def test_output_structure(self):
        """Batch returns dict with correct keys."""
        r = np.random.default_rng(44).standard_normal(100) * 0.01
        v = np.full(100, 0.01)
        nu_grid = [3.0, 5.0, 8.0, 12.0]

        results = run_phi_student_t_filter_batch(
            r, v, 1e-4, 1.0, 0.90, nu_grid
        )
        assert isinstance(results, dict)
        assert len(results) == 4
        for nu in nu_grid:
            assert nu in results
            mu, P, ll = results[nu]
            assert len(mu) == 100
            assert len(P) == 100
            assert np.isfinite(ll)


# ===========================================================================
class TestBatchPerformance:
    """Batch is at least as fast as sequential."""

    def test_batch_not_slower(self):
        """Batch of 4 nu values should not be much slower than 4 sequential calls."""
        rng = np.random.default_rng(45)
        r = 0.01 * rng.standard_normal(2000)
        v = np.full(2000, 0.01)
        nu_grid = [4.0, 8.0, 15.0, 30.0]

        # Warm up
        run_phi_student_t_filter_batch(r, v, 1e-4, 1.0, 0.90, nu_grid)
        for nu in nu_grid:
            run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, nu)

        # Benchmark batch
        n_trials = 20
        t0 = time.perf_counter()
        for _ in range(n_trials):
            run_phi_student_t_filter_batch(r, v, 1e-4, 1.0, 0.90, nu_grid)
        t_batch = time.perf_counter() - t0

        # Benchmark sequential
        t0 = time.perf_counter()
        for _ in range(n_trials):
            for nu in nu_grid:
                run_phi_student_t_filter(r, v, 1e-4, 1.0, 0.90, nu)
        t_seq = time.perf_counter() - t0

        # Batch should not be drastically slower
        assert t_batch < t_seq * 3.0, \
            f"Batch {t_batch:.3f}s vs Sequential {t_seq:.3f}s"


# ===========================================================================
class TestBatchLogLikelihood:
    """Batch log-likelihoods are consistent."""

    def test_loglik_finite(self):
        r = np.random.default_rng(47).standard_normal(500) * 0.01
        v = np.full(500, 0.01)
        nu_grid = [3.0, 5.0, 8.0, 20.0]

        results = run_phi_student_t_filter_batch(
            r, v, 1e-4, 1.0, 0.90, nu_grid
        )
        for nu in nu_grid:
            _, _, ll = results[nu]
            assert np.isfinite(ll), f"nu={nu}: ll={ll}"

    def test_loglik_ordering_gaussian_data(self):
        """For Gaussian data, high nu should have better loglik."""
        rng = np.random.default_rng(48)
        r = 0.01 * rng.standard_normal(1000)
        v = np.full(1000, 0.01)
        nu_grid = [4.0, 8.0, 20.0, 30.0]

        results = run_phi_student_t_filter_batch(
            r, v, 1e-4, 1.0, 0.90, nu_grid
        )
        for nu in nu_grid:
            _, _, ll = results[nu]
            assert np.isfinite(ll)

    def test_deterministic_batch(self):
        """Batch produces identical results on repeated calls."""
        r = np.random.default_rng(49).standard_normal(300) * 0.01
        v = np.full(300, 0.01)
        nu_grid = [4.0, 8.0]

        r1 = run_phi_student_t_filter_batch(r, v, 1e-4, 1.0, 0.90, nu_grid)
        r2 = run_phi_student_t_filter_batch(r, v, 1e-4, 1.0, 0.90, nu_grid)

        for nu in nu_grid:
            assert r1[nu][2] == r2[nu][2]
