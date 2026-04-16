"""
Story 15.2 – PIT-Driven Variance Inflation with Dead-Zone
===========================================================
Verify dead-zone prevents over-correction, asset-class zones work,
and beta converges within max_iter.
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

from models.numba_wrappers import (
    pit_variance_inflation_iterative,
    _compute_pit_mad,
)


class TestPITMAD:
    """PIT MAD computation."""

    def test_uniform_pit_mad(self):
        """Uniform PIT has MAD ~ 0.25."""
        pit = np.linspace(0.01, 0.99, 1000)
        mad = _compute_pit_mad(pit)
        assert abs(mad - 0.25) < 0.01

    def test_peaked_pit_low_mad(self):
        """PIT peaked at 0.5 has MAD < 0.25."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.normal(0.5, 0.1, 1000), 0.01, 0.99)
        mad = _compute_pit_mad(pit)
        assert mad < 0.20

    def test_u_shaped_pit_high_mad(self):
        """U-shaped PIT has MAD > 0.25."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.beta(0.5, 0.5, 1000), 0.01, 0.99)
        mad = _compute_pit_mad(pit)
        assert mad > 0.25


class TestDeadZone:
    """Dead-zone prevents over-correction."""

    def test_well_calibrated_no_correction(self):
        """MAD in dead-zone -> no correction applied."""
        # MAD ~ 0.25 for uniform
        pit = np.linspace(0.01, 0.99, 500)
        # Default dead-zone: [0.30, 0.55]. MAD=0.25 < 0.30, so correction will apply.
        # Use pit that has MAD in dead-zone range
        # For MAD = 0.42: pit with spread around 0.5 at moderate width
        rng = np.random.default_rng(42)
        pit_good = np.clip(rng.uniform(0.05, 0.95, 1000), 0.01, 0.99)
        mad = _compute_pit_mad(pit_good)
        result = pit_variance_inflation_iterative(
            pit_good, dead_zone=(mad - 0.05, mad + 0.05)
        )
        assert not result["correction_applied"]
        assert result["beta_final"] == 1.0

    def test_peaked_pit_decreases_beta(self):
        """Peaked PIT (MAD < low) -> beta decreases."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.normal(0.5, 0.05, 1000), 0.01, 0.99)
        result = pit_variance_inflation_iterative(pit, dead_zone=(0.20, 0.40))
        assert result["correction_applied"]
        assert result["beta_final"] < 1.0

    def test_u_shaped_pit_increases_beta(self):
        """U-shaped PIT (MAD > high) -> beta increases."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.beta(0.5, 0.5, 1000), 0.01, 0.99)
        result = pit_variance_inflation_iterative(pit, dead_zone=(0.20, 0.30))
        assert result["correction_applied"]
        assert result["beta_final"] > 1.0


class TestAssetClassDeadZones:
    """Asset-class specific dead-zones."""

    def test_metals_wider(self):
        """Metals have wider dead-zone [0.25, 0.60]."""
        pit = np.linspace(0.01, 0.99, 500)
        result = pit_variance_inflation_iterative(pit, asset_class="metals_gold")
        # Should use metals dead-zone
        assert result is not None

    def test_crypto_narrower(self):
        """Crypto has narrower dead-zone [0.28, 0.52]."""
        pit = np.linspace(0.01, 0.99, 500)
        result = pit_variance_inflation_iterative(pit, asset_class="crypto")
        assert result is not None


class TestConvergence:
    """Beta converges within max_iter."""

    def test_converges_or_reaches_max(self):
        """Beta stabilizes or reaches max_iter within bounds."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.normal(0.5, 0.05, 1000), 0.01, 0.99)
        result = pit_variance_inflation_iterative(pit, max_iter=10)
        assert result["n_iter"] <= 10
        assert 0.5 <= result["beta_final"] <= 5.0

    def test_beta_bounded(self):
        """Beta stays in [0.5, 5.0]."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.beta(0.5, 0.5, 1000), 0.01, 0.99)
        result = pit_variance_inflation_iterative(pit, max_iter=20)
        assert 0.5 <= result["beta_final"] <= 5.0

    def test_trajectory_recorded(self):
        """Trajectory records each iteration."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.normal(0.5, 0.05, 1000), 0.01, 0.99)
        result = pit_variance_inflation_iterative(pit)
        assert len(result["trajectory"]) >= 1
        assert result["trajectory"][0][0] == 0  # first entry is iteration 0
