"""
Story 4.2: Adaptive Log-Likelihood Clipping
=============================================
MAD-based adaptive clip preserves signal, caps crisis outliers.
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

from models.phi_gaussian import adaptive_ll_clip


class TestAdaptiveLLClip:
    """Acceptance criteria for Story 4.2."""

    def test_well_behaved_minimal_clipping(self):
        """AC2: On well-behaved data, < 0.1% clipped."""
        rng = np.random.default_rng(42)
        # Normal LL values around -2
        traj = rng.normal(-2.0, 1.0, 2000)
        clipped, frac = adaptive_ll_clip(traj, mode='adaptive')
        assert frac < 0.001, f"Clip fraction = {frac:.4f} (expect < 0.1%)"

    def test_crisis_moderate_clipping(self):
        """AC3: On crisis data, < 2% clipped."""
        rng = np.random.default_rng(7)
        traj = rng.normal(-2.0, 1.5, 500)
        # Insert crisis spikes
        traj[100] = -80.0
        traj[200] = -120.0
        traj[300] = -90.0
        clipped, frac = adaptive_ll_clip(traj, mode='adaptive')
        assert frac < 0.02, f"Clip fraction = {frac:.4f} (expect < 2%)"

    def test_clipped_values_bounded(self):
        """Clipped LL values are bounded by adaptive threshold."""
        traj = np.array([-2.0] * 50 + [-200.0, -300.0] + [-2.0] * 48)
        clipped, frac = adaptive_ll_clip(traj, mode='adaptive')
        # Extreme values should be clipped
        assert abs(clipped[50]) < 200.0
        assert abs(clipped[51]) < 300.0

    def test_fixed_mode(self):
        """Fixed mode clips at exactly 50."""
        traj = np.array([-10.0, -60.0, 30.0, 70.0, -40.0])
        clipped, frac = adaptive_ll_clip(traj, mode='fixed', min_clip=50.0)
        assert clipped[1] == -50.0
        assert clipped[3] == 50.0
        assert frac == 2.0 / 5.0

    def test_none_mode(self):
        """None mode returns trajectory unchanged."""
        traj = np.array([-200.0, 300.0, -1.0])
        clipped, frac = adaptive_ll_clip(traj, mode='none')
        np.testing.assert_array_equal(clipped, traj)
        assert frac == 0.0

    def test_preserves_moderate_values(self):
        """Moderate LL values are never clipped."""
        rng = np.random.default_rng(99)
        traj = rng.normal(-3.0, 2.0, 1000)
        clipped, frac = adaptive_ll_clip(traj, mode='adaptive')
        # All values within +/-15 should be preserved
        mask = np.abs(traj) < 15.0
        np.testing.assert_array_equal(clipped[mask], traj[mask])

    def test_returns_copy_not_inplace(self):
        """Should return a copy, not modify input."""
        traj = np.array([-2.0, -100.0, -2.0])
        traj_orig = traj.copy()
        clipped, _ = adaptive_ll_clip(traj, mode='fixed', min_clip=50.0)
        np.testing.assert_array_equal(traj, traj_orig)
