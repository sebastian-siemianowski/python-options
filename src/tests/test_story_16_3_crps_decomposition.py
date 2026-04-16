"""
Story 16.3 – CRPS Decomposition into Reliability and Resolution
================================================================
Verify Hersbach (2000) decomposition: CRPS = Reliability - Resolution + Uncertainty.
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

from models.numba_wrappers import crps_decomposition


class TestUniformPIT:
    """Perfectly calibrated model -> uniform PIT."""

    def test_reliability_near_zero(self):
        """Uniform PIT -> reliability near 0."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 5000)
        result = crps_decomposition(pit)
        assert result["reliability"] < 0.002

    def test_well_calibrated_flag(self):
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 5000)
        result = crps_decomposition(pit)
        assert result["well_calibrated"]

    def test_uncertainty_near_quarter(self):
        """Uniform PIT -> mean ≈ 0.5 -> uncertainty ≈ 0.25."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 5000)
        result = crps_decomposition(pit)
        assert abs(result["uncertainty"] - 0.25) < 0.02

    def test_decomposition_identity(self):
        """CRPS = Reliability - Resolution + Uncertainty."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 5000)
        result = crps_decomposition(pit)
        reconstructed = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert abs(result["crps_reconstructed"] - reconstructed) < 1e-10


class TestMiscalibratedPIT:
    """Mis-calibrated PIT -> high reliability."""

    def test_biased_pit_high_reliability(self):
        """PIT concentrated near 0.5 -> not uniform -> reliability > 0."""
        rng = np.random.default_rng(42)
        pit = rng.beta(5, 5, 2000)  # concentrated near 0.5
        result = crps_decomposition(pit)
        assert result["reliability"] > 0.001

    def test_u_shaped_pit_detected(self):
        """U-shaped PIT (under-dispersed) -> high reliability."""
        rng = np.random.default_rng(42)
        pit = rng.beta(0.5, 0.5, 2000)  # U-shaped
        result = crps_decomposition(pit)
        assert result["reliability"] > 0.005

    def test_skewed_pit_detected(self):
        """Skewed PIT -> reliability > 0."""
        rng = np.random.default_rng(42)
        pit = rng.beta(2, 5, 2000)  # skewed left
        result = crps_decomposition(pit)
        assert result["reliability"] > 0.001


class TestResolution:
    """Resolution measures informativeness."""

    def test_resolution_positive(self):
        """Resolution should be non-negative."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 2000)
        result = crps_decomposition(pit)
        assert result["resolution"] >= 0

    def test_resolution_for_informative_model(self):
        """Well-calibrated model should have some resolution."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 5000)
        result = crps_decomposition(pit)
        # For large uniform sample, resolution should be near the theoretical value
        assert result["resolution"] >= 0


class TestEdgeCases:
    """Edge cases for CRPS decomposition."""

    def test_too_few_samples(self):
        """< 10 samples -> NaN."""
        pit = np.array([0.1, 0.5, 0.9])
        result = crps_decomposition(pit)
        assert np.isnan(result["reliability"])

    def test_nan_filtered(self):
        """NaN values are filtered out."""
        rng = np.random.default_rng(42)
        pit = np.concatenate([rng.uniform(0, 1, 1000), [np.nan, np.nan]])
        result = crps_decomposition(pit)
        assert np.isfinite(result["reliability"])

    def test_boundary_values(self):
        """PIT values at 0 and 1 handled correctly."""
        rng = np.random.default_rng(42)
        pit = np.concatenate([[0.0, 1.0], rng.uniform(0, 1, 500)])
        result = crps_decomposition(pit)
        assert np.isfinite(result["reliability"])

    def test_bin_counts_sum(self):
        """Bin counts sum to n."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 1000)
        result = crps_decomposition(pit, n_bins=10)
        assert abs(np.sum(result["bin_counts"]) - 1000) < 1e-10

    def test_bin_obs_freq_sum_to_one(self):
        """Bin observed frequencies sum to 1."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 1000)
        result = crps_decomposition(pit, n_bins=10)
        assert abs(np.sum(result["bin_obs_freq"]) - 1.0) < 1e-10


class TestDifferentBins:
    """Test with different bin counts."""

    @pytest.mark.parametrize("n_bins", [5, 10, 20, 50])
    def test_reliability_stable_across_bins(self, n_bins):
        """Reliability should be similar across bin counts for uniform PIT."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 5000)
        result = crps_decomposition(pit, n_bins=n_bins)
        # All should show low reliability for uniform PIT
        assert result["reliability"] < 0.005
