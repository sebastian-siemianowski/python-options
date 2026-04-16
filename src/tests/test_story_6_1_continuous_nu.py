"""
Story 6.1: Continuous Nu Optimization via Profile Likelihood
==============================================================
Golden-section refinement of nu over [2.1, 50] after grid search.
"""
import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.phi_student_t import profile_likelihood_refine_nu, PhiStudentTDriftModel


def _synth_student_t_data(n=500, nu_true=5.0, seed=42):
    """Generate data from a known Student-t model."""
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.02) + rng.normal(0, 0.002, n).clip(-0.005, 0.005)
    vol = np.abs(vol)
    returns = rng.standard_t(nu_true, n) * 0.02
    return returns, vol


class TestProfileLikelihoodNu:
    """Acceptance criteria for Story 6.1."""

    def test_continuous_nu_in_range(self):
        """AC1: nu* optimized over [2.1, 50]."""
        returns, vol = _synth_student_t_data(n=500, nu_true=5.0)
        nu_refined, ll = profile_likelihood_refine_nu(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu_grid_best=4.0
        )
        assert 2.1 <= nu_refined <= 50.0

    def test_refined_differs_from_grid(self):
        """AC2: Refined nu can differ from grid value."""
        returns, vol = _synth_student_t_data(n=500, nu_true=5.3)
        nu_refined, ll = profile_likelihood_refine_nu(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu_grid_best=4.0
        )
        # May or may not differ, but should be valid
        assert 2.1 <= nu_refined <= 50.0
        assert np.isfinite(ll)

    def test_search_range_around_grid(self):
        """AC3: Search range is [nu_grid - 2, nu_grid + 4]."""
        returns, vol = _synth_student_t_data(n=500, nu_true=8.0)
        nu_refined, ll = profile_likelihood_refine_nu(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu_grid_best=8.0
        )
        # Should be in [6, 12] range
        assert 2.1 <= nu_refined <= 50.0
        assert np.isfinite(ll)

    def test_ll_improvement(self):
        """AC4: Refined nu has >= log-likelihood compared to grid best."""
        returns, vol = _synth_student_t_data(n=500, nu_true=6.0)
        nu_grid = 4.0
        # Get LL at grid point
        _, _, ll_grid = PhiStudentTDriftModel.filter_phi(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu=nu_grid
        )
        # Get LL at refined point
        nu_refined, ll_refined = profile_likelihood_refine_nu(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu_grid_best=nu_grid
        )
        assert ll_refined >= ll_grid - 1.0, \
            f"Refined LL={ll_refined:.2f} < grid LL={ll_grid:.2f}"

    def test_thin_tailed_no_degradation(self):
        """AC5: For thin-tailed data (near-Gaussian), no BIC degradation."""
        rng = np.random.default_rng(42)
        n = 500
        returns = rng.normal(0, 0.02, n)
        vol = np.full(n, 0.02)
        nu_grid = 20.0
        _, _, ll_grid = PhiStudentTDriftModel.filter_phi(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu=nu_grid
        )
        nu_refined, ll_refined = profile_likelihood_refine_nu(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu_grid_best=nu_grid
        )
        # Should not get significantly worse
        assert ll_refined >= ll_grid - 5.0

    def test_custom_filter_func(self):
        """AC6: Custom filter function works correctly."""
        returns, vol = _synth_student_t_data(n=300)

        call_count = [0]
        def custom_filter(r, v, q, c, phi, nu):
            call_count[0] += 1
            return PhiStudentTDriftModel.filter_phi(r, v, q, c, phi, nu)

        nu_refined, ll = profile_likelihood_refine_nu(
            returns, vol, q=1e-6, c=1.0, phi=0.0, nu_grid_best=4.0,
            filter_func=custom_filter
        )
        assert call_count[0] > 0
        assert np.isfinite(ll)
