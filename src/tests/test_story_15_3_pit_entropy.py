"""
Story 15.3 – PIT Entropy as Calibration Diagnostic
====================================================
Verify pit_entropy() detects well-calibrated, peaked, and U-shaped PIT.
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

from models.numba_wrappers import pit_entropy


class TestUniformPITEntropy:
    """Uniform PIT has entropy ratio near 1.0."""

    def test_perfect_uniform(self):
        """Perfectly uniform PIT -> entropy ratio ~ 1.0."""
        pit = np.linspace(0.025, 0.975, 10000)
        result = pit_entropy(pit)
        assert abs(result["entropy_ratio"] - 1.0) < 0.05
        assert result["well_calibrated"]

    def test_large_sample_uniform(self):
        """Large uniform sample -> well calibrated."""
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 50000)
        result = pit_entropy(pit)
        assert result["well_calibrated"]

    def test_entropy_uniform_value(self):
        """Entropy of uniform = log(B) = log(20) ~ 2.996."""
        pit = np.linspace(0.025, 0.975, 10000)
        result = pit_entropy(pit, n_bins=20)
        assert abs(result["entropy_uniform"] - np.log(20)) < 1e-10


class TestPeakedPIT:
    """Peaked PIT detected via low entropy ratio."""

    def test_peaked_low_ratio(self):
        """Peaked at 0.5 -> ratio < 0.95."""
        rng = np.random.default_rng(42)
        pit = np.clip(rng.normal(0.5, 0.05, 5000), 0.01, 0.99)
        result = pit_entropy(pit)
        assert result["entropy_ratio"] < 0.90
        assert not result["well_calibrated"]


class TestUShapedPIT:
    """U-shaped PIT has entropy close to uniform (it's still spread)."""

    def test_u_shaped(self):
        """U-shaped from Beta(0.5, 0.5) -> entropy lower than uniform."""
        rng = np.random.default_rng(42)
        pit = rng.beta(0.5, 0.5, 5000)
        result = pit_entropy(pit)
        # U-shaped concentrates at edges -> some bins get more, entropy lower
        assert result["entropy_ratio"] < 1.0


class TestEdgeCases:
    """Edge cases for pit_entropy."""

    def test_empty_pit(self):
        result = pit_entropy(np.array([]))
        assert result["entropy"] == 0.0
        assert not result["well_calibrated"]

    def test_single_value(self):
        result = pit_entropy(np.array([0.5]))
        assert result["entropy_ratio"] < 0.5  # all in one bin

    def test_nan_handling(self):
        pit = np.array([0.1, 0.3, np.nan, 0.7, 0.9])
        result = pit_entropy(pit)
        assert result["entropy"] > 0

    def test_custom_bins(self):
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 10000)
        r10 = pit_entropy(pit, n_bins=10)
        r50 = pit_entropy(pit, n_bins=50)
        assert abs(r10["entropy_uniform"] - np.log(10)) < 1e-10
        assert abs(r50["entropy_uniform"] - np.log(50)) < 1e-10


class TestBinProperties:
    """Bin counts and probabilities are correct."""

    def test_bin_counts_sum(self):
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 1000)
        result = pit_entropy(pit)
        assert result["bin_counts"].sum() == 1000

    def test_bin_probs_sum_to_one(self):
        rng = np.random.default_rng(42)
        pit = rng.uniform(0, 1, 1000)
        result = pit_entropy(pit)
        assert abs(result["bin_probs"].sum() - 1.0) < 1e-10
