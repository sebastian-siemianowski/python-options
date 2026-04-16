"""
Story 10.1: GARCH-Kalman Variance Coherence Check
===================================================
Detect and correct GARCH/Kalman variance divergence.
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

from models.phi_student_t_unified import compute_variance_coherence


class TestVarianceCoherence:
    """Acceptance criteria for Story 10.1."""

    def test_coherent_series(self):
        """AC1: Coherent series has < 5% flagged."""
        np.random.seed(42)
        n = 500
        h = np.abs(np.random.normal(0.001, 0.0002, n))
        P = np.abs(np.random.normal(0.0005, 0.0001, n))
        S = P + h  # Perfectly coherent: S - P = h, ratio = 1
        R, rho, diag = compute_variance_coherence(h, S, P)
        assert diag['frac_incoherent'] < 0.05

    def test_incoherent_detected(self):
        """AC2: Incoherent series is detected."""
        np.random.seed(42)
        n = 500
        h = np.abs(np.random.normal(0.001, 0.0002, n))
        P = np.full(n, 0.0001)
        S = P + h * 10  # ratio ~ 0.1, very incoherent
        R, rho, diag = compute_variance_coherence(h, S, P)
        assert diag['frac_incoherent'] > 0.5

    def test_blending_applied(self):
        """AC3: When incoherent, R_t is blended."""
        n = 100
        h = np.full(n, 0.001)
        P = np.full(n, 0.0001)
        S = P + h * 5  # S - P = 5*h, ratio = 0.2 (incoherent)
        R, rho, diag = compute_variance_coherence(h, S, P, c_obs=1.0)
        R_kalman = S - P
        # Blended should be between pure GARCH and pure Kalman
        assert np.all(R >= np.minimum(h, R_kalman) * 0.4)

    def test_coherence_ratio_range(self):
        """AC4: Coherence ratio reported correctly."""
        n = 100
        h = np.full(n, 0.001)
        P = np.full(n, 0.0001)
        S = P + h  # Perfect coherence
        _, rho, diag = compute_variance_coherence(h, S, P)
        assert diag['mean_rho'] == pytest.approx(1.0, rel=0.1)

    def test_diagnostics_complete(self):
        """AC5: All diagnostic fields present."""
        n = 50
        h = np.full(n, 0.001)
        P = np.full(n, 0.0001)
        S = P + h
        _, _, diag = compute_variance_coherence(h, S, P)
        required = {'frac_incoherent', 'n_incoherent', 'mean_rho', 'median_rho'}
        assert required.issubset(diag.keys())

    def test_no_nans(self):
        """AC6: No NaN in output."""
        np.random.seed(42)
        n = 200
        h = np.abs(np.random.normal(0.001, 0.0005, n))
        P = np.abs(np.random.normal(0.0003, 0.0001, n))
        S = P + np.abs(np.random.normal(0.001, 0.0003, n))
        R, rho, _ = compute_variance_coherence(h, S, P)
        assert not np.any(np.isnan(R))
        assert not np.any(np.isnan(rho))
