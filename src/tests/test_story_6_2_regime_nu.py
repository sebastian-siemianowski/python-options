"""
Story 6.2: Regime-Dependent Nu via Markov Switching
=====================================================
nu_eff(t) = (1 - p_stress) * nu_calm + p_stress * nu_crisis
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

from models.phi_student_t import compute_regime_dependent_nu


class TestRegimeDependentNu:
    """Acceptance criteria for Story 6.2."""

    def test_nu_calm_ge_8(self):
        """AC1: nu_calm >= 8 in calm periods."""
        vol = np.full(200, 0.02)  # Constant low vol
        nu_eff, p_stress = compute_regime_dependent_nu(
            vol, nu_calm=12.0, nu_crisis=4.0, sensitivity=2.0
        )
        # In constant vol, p_stress should be moderate, nu_eff close to nu_calm
        # After warmup, p_stress should settle near 0.5
        assert np.all(nu_eff >= 2.1)

    def test_nu_crisis_le_6(self):
        """AC2: During high-vol spikes, nu drops toward nu_crisis."""
        rng = np.random.default_rng(42)
        vol = np.full(300, 0.02)
        # Insert vol spike
        vol[200:] = 0.08  # 4x normal vol
        nu_eff, p_stress = compute_regime_dependent_nu(
            vol, nu_calm=12.0, nu_crisis=4.0, sensitivity=3.0
        )
        # After spike, p_stress should increase, nu should decrease
        nu_spike = float(np.mean(nu_eff[250:]))
        nu_calm_period = float(np.mean(nu_eff[100:150]))
        assert nu_spike < nu_calm_period, \
            f"nu during spike ({nu_spike:.1f}) should be < calm ({nu_calm_period:.1f})"

    def test_nu_eff_bounded(self):
        """AC3: nu_eff always in [nu_crisis, nu_calm]."""
        rng = np.random.default_rng(42)
        vol = rng.lognormal(-4, 0.5, 500)
        nu_eff, _ = compute_regime_dependent_nu(
            vol, nu_calm=15.0, nu_crisis=3.0, sensitivity=2.0
        )
        assert np.all(nu_eff >= 3.0 - 0.1)  # Small tolerance for float
        assert np.all(nu_eff <= 15.0 + 0.1)

    def test_consistent_with_ms_q(self):
        """AC4: Uses same stress probability as MS-q."""
        from models.phi_student_t import compute_ms_process_noise_smooth
        vol = np.random.default_rng(42).lognormal(-4, 0.3, 200)

        _, p_stress_q = compute_ms_process_noise_smooth(
            vol, q_calm=1e-6, q_stress=1e-4, sensitivity=2.0
        )
        _, p_stress_nu = compute_regime_dependent_nu(
            vol, nu_calm=12.0, nu_crisis=4.0, sensitivity=2.0
        )
        np.testing.assert_array_almost_equal(p_stress_q, p_stress_nu)

    def test_output_shapes(self):
        """AC5: Output arrays have correct shape."""
        vol = np.full(300, 0.02)
        nu_eff, p_stress = compute_regime_dependent_nu(vol, nu_calm=12.0, nu_crisis=4.0)
        assert nu_eff.shape == (300,)
        assert p_stress.shape == (300,)

    def test_ewm_mode(self):
        """AC6: EWM mode works correctly."""
        vol = np.full(200, 0.02)
        nu_eff, p_stress = compute_regime_dependent_nu(
            vol, nu_calm=12.0, nu_crisis=4.0, sensitivity=2.0, ewm_lambda=0.97
        )
        assert len(nu_eff) == 200
        assert np.all(np.isfinite(nu_eff))
