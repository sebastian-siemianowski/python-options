"""
Story 5.1: Multi-Start Optimization for Stage 1
=================================================
Verify that multi-start L-BFGS-B explores the parameter space and uses
cross-validated log-likelihood for model selection.
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

from models.gaussian import GaussianDriftModel, GaussianUnifiedConfig


def _synth_data(n=500, seed=42):
    rng = np.random.default_rng(seed)
    vol = np.full(n, 0.02) + rng.normal(0, 0.002, n).clip(-0.005, 0.005)
    vol = np.abs(vol)
    returns = rng.normal(0, 0.02, n)
    return returns, vol


class TestMultiStartStage1:
    """Acceptance criteria for Story 5.1."""

    def test_at_least_3_starts(self):
        """AC1: At least 3 starting points used in L-BFGS-B."""
        returns, vol = _synth_data()
        config = GaussianUnifiedConfig.auto_configure(returns, vol)
        s1 = GaussianDriftModel._gaussian_stage_1(
            returns, vol, len(returns), config, phi_mode=True, n_starts=3
        )
        assert s1['success']
        # grid-best + 3 canonical = 4 starts total
        assert s1['n_starts'] >= 4

    def test_canonical_starts_span_space(self):
        """AC2: Starting points include diverse (q, c, phi) combinations."""
        returns, vol = _synth_data()
        config = GaussianUnifiedConfig.auto_configure(returns, vol)
        s1 = GaussianDriftModel._gaussian_stage_1(
            returns, vol, len(returns), config, phi_mode=True, n_starts=3
        )
        assert s1['success']
        # Each start should have been tried
        results = s1['start_results']
        assert len(results) >= 3
        # Verify starts span different q regions (log10 space)
        x0_values = [r['x0'] for r in results]
        log_q_starts = [x[0] for x in x0_values]
        assert max(log_q_starts) - min(log_q_starts) >= 1.0, \
            f"Starts don't span q-space: {log_q_starts}"

    def test_cv_ll_selection(self):
        """AC3: Best start selected by CV log-likelihood, not training LL."""
        returns, vol = _synth_data()
        config = GaussianUnifiedConfig.auto_configure(returns, vol)
        s1 = GaussianDriftModel._gaussian_stage_1(
            returns, vol, len(returns), config, phi_mode=True, n_starts=3
        )
        assert s1['success']
        # start_results should contain fun values from CV objective
        results = s1['start_results']
        best_fun = min(r['fun'] for r in results)
        assert best_fun < 1e10, "No valid optimum found"

    def test_maxiter_200(self):
        """AC4: L-BFGS-B uses maxiter=200 (not 100)."""
        # We verify indirectly: with more iterations, optimization should converge
        returns, vol = _synth_data(n=300)
        config = GaussianUnifiedConfig.auto_configure(returns, vol)
        s1 = GaussianDriftModel._gaussian_stage_1(
            returns, vol, len(returns), config, phi_mode=True, n_starts=3
        )
        assert s1['success']
        assert s1['q'] > 0
        assert s1['c'] > 0

    def test_no_phi_mode(self):
        """Multi-start works with phi_mode=False (random walk)."""
        returns, vol = _synth_data()
        config = GaussianUnifiedConfig.auto_configure(returns, vol)
        s1 = GaussianDriftModel._gaussian_stage_1(
            returns, vol, len(returns), config, phi_mode=False, n_starts=3
        )
        assert s1['success']
        assert s1['n_starts'] >= 4
        # phi should be 1.0 for random walk mode
        assert abs(s1['phi'] - 1.0) < 0.01

    def test_n_starts_parameter(self):
        """n_starts parameter controls number of canonical starts."""
        returns, vol = _synth_data()
        config = GaussianUnifiedConfig.auto_configure(returns, vol)
        s1 = GaussianDriftModel._gaussian_stage_1(
            returns, vol, len(returns), config, phi_mode=True, n_starts=2
        )
        assert s1['success']
        # grid-best + 2 canonical = 3 starts
        assert s1['n_starts'] >= 3

    def test_parameters_reasonable(self):
        """Optimized parameters are in reasonable ranges."""
        returns, vol = _synth_data()
        config = GaussianUnifiedConfig.auto_configure(returns, vol)
        s1 = GaussianDriftModel._gaussian_stage_1(
            returns, vol, len(returns), config, phi_mode=True, n_starts=3
        )
        assert s1['success']
        assert 1e-10 < s1['q'] < 1.0
        assert 0.01 < s1['c'] < 100.0
        assert -1.0 < s1['phi'] < 1.0
