"""
Story 14.2 – Phi Shrinkage Direction for Momentum Models
==========================================================
Verify phi shrinkage centers on 1.0 when momentum active, 0.0 otherwise.
Check collinearity safety and bounds.
"""

import os, sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
for _p in (REPO_ROOT, SRC_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.numba_wrappers import compute_phi_shrinkage_for_momentum


class TestMomentumShrinkageCenter:
    """Prior center is 1.0 when momentum active."""

    def test_momentum_center_1(self):
        result = compute_phi_shrinkage_for_momentum(0.95, momentum_enabled=True)
        assert result["center"] == 1.0
        assert result["tau"] == 0.10

    def test_no_momentum_center_0(self):
        result = compute_phi_shrinkage_for_momentum(0.95, momentum_enabled=False)
        assert result["center"] == 0.0
        assert result["tau"] == 0.20


class TestPhiShrinkageDirection:
    """Phi shrinks toward the correct center."""

    def test_momentum_shrinks_toward_1(self):
        """With momentum, phi_mle=0.8 shrinks toward 1.0."""
        result = compute_phi_shrinkage_for_momentum(0.8, momentum_enabled=True)
        assert result["phi_shrunk"] > 0.8  # pulled toward 1.0

    def test_no_momentum_shrinks_toward_0(self):
        """Without momentum, phi_mle=0.8 shrinks toward 0.0."""
        result = compute_phi_shrinkage_for_momentum(0.8, momentum_enabled=False)
        assert result["phi_shrunk"] < 0.8  # pulled toward 0.0

    def test_momentum_phi_at_center_unchanged(self):
        """phi_mle=1.0 with momentum -> stays near 1.0."""
        result = compute_phi_shrinkage_for_momentum(1.0, momentum_enabled=True)
        assert abs(result["phi_shrunk"] - 1.0) < 0.01


class TestBoundsEnforcement:
    """No explosive or oscillatory phi."""

    def test_no_explosive(self):
        """phi_shrunk <= 1.05."""
        result = compute_phi_shrinkage_for_momentum(1.5, momentum_enabled=True)
        assert result["phi_shrunk"] <= 1.05

    def test_no_oscillatory(self):
        """phi_shrunk >= -0.5."""
        result = compute_phi_shrinkage_for_momentum(-2.0, momentum_enabled=False)
        assert result["phi_shrunk"] >= -0.5


class TestCollinearitySafety:
    """Collinearity check for phi vs kappa."""

    def test_within_3_tau_safe(self):
        """phi_mle near center -> collinearity_safe = True."""
        result = compute_phi_shrinkage_for_momentum(0.95, momentum_enabled=True)
        assert result["collinearity_safe"]

    def test_far_from_center_unsafe(self):
        """phi_mle far from center -> collinearity_safe = False."""
        result = compute_phi_shrinkage_for_momentum(0.0, momentum_enabled=True)
        # deviation=1.0, 3*tau=0.3, 1.0 > 0.3 -> unsafe
        assert not result["collinearity_safe"]


class TestLogPrior:
    """Gaussian log prior computed correctly."""

    def test_at_center_maximal(self):
        """Log prior maximized at center."""
        result_at = compute_phi_shrinkage_for_momentum(1.0, momentum_enabled=True)
        result_away = compute_phi_shrinkage_for_momentum(0.5, momentum_enabled=True)
        assert result_at["log_prior"] > result_away["log_prior"]

    def test_log_prior_negative(self):
        """Log prior is <= 0."""
        result = compute_phi_shrinkage_for_momentum(0.5, momentum_enabled=True)
        assert result["log_prior"] <= 0.0

    def test_log_prior_at_center_zero(self):
        """Log prior at center is exactly 0."""
        result = compute_phi_shrinkage_for_momentum(1.0, momentum_enabled=True)
        assert result["log_prior"] == 0.0


class TestParametrization:
    """Custom center/tau overrides work."""

    def test_custom_center(self):
        result = compute_phi_shrinkage_for_momentum(
            0.5, momentum_enabled=True,
            center_momentum=0.8, tau_momentum=0.15
        )
        assert result["center"] == 0.8
        assert result["tau"] == 0.15

    @pytest.mark.parametrize("phi_mle", [0.0, 0.5, 0.9, 1.0, 1.03])
    def test_all_reasonable_phi(self, phi_mle):
        """All reasonable phi values handled without error."""
        result = compute_phi_shrinkage_for_momentum(phi_mle, momentum_enabled=True)
        assert -0.5 <= result["phi_shrunk"] <= 1.05
        assert np.isfinite(result["log_prior"])
