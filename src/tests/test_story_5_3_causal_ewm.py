"""
Story 5.3: Causal EWM Location Correction with Bias Prevention
================================================================
EWM correction uses only past info (lag-1 indexing). No look-ahead.
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

from models.gaussian import causal_ewm_location_correction


class TestCausalEWM:
    """Acceptance criteria for Story 5.3."""

    def test_strict_causality(self):
        """AC1: correction[t] uses only innovations[0..t-1], not innovations[t]."""
        rng = np.random.default_rng(42)
        innovations = rng.normal(0, 0.01, 200)

        corr = causal_ewm_location_correction(innovations, ewm_lambda=0.96)

        # correction[0] should be 0 (no past data)
        assert corr[0] == 0.0

        # Manually verify correction[1] = alpha * innovations[0]
        alpha = 1.0 - 0.96
        expected_1 = 0.96 * 0.0 + alpha * innovations[0]
        assert abs(corr[1] - expected_1) < 1e-15

        # Verify correction[2] = lambda * (alpha * innovations[0]) + alpha * innovations[1]
        expected_2 = 0.96 * expected_1 + alpha * innovations[1]
        assert abs(corr[2] - expected_2) < 1e-15

    def test_known_bias_correction(self):
        """AC2: Inject known bias, verify EWM corrects it within 30 steps."""
        n = 100
        bias = 0.005
        innovations = np.full(n, bias)  # Constant bias

        corr = causal_ewm_location_correction(innovations, ewm_lambda=0.90)

        # After 30 steps, correction should be close to 0.005
        assert abs(corr[30] - bias) < 0.001, f"corr[30]={corr[30]}, expected ~{bias}"

    def test_no_nans_first_10(self):
        """AC3: First 10 observations (EWM not yet stable) produce no NaN."""
        innovations = np.random.default_rng(42).normal(0, 0.01, 50)
        corr = causal_ewm_location_correction(innovations, ewm_lambda=0.96)
        assert not np.any(np.isnan(corr[:10])), "NaN in first 10 corrections"

    def test_disabled_returns_zeros(self):
        """AC4: ewm_lambda=0 disables correction (all zeros)."""
        innovations = np.random.default_rng(42).normal(0, 0.01, 100)
        corr = causal_ewm_location_correction(innovations, ewm_lambda=0.0)
        assert np.all(corr == 0.0)

    def test_correction_reduces_bias(self):
        """AC5: Correction applied to biased predictions reduces mean error."""
        rng = np.random.default_rng(42)
        n = 500
        true_bias = 0.003
        innovations = rng.normal(true_bias, 0.02, n)

        corr = causal_ewm_location_correction(innovations, ewm_lambda=0.96)

        # After warmup, corrected innovations should have smaller bias
        warmup = 50
        raw_bias = abs(float(np.mean(innovations[warmup:])))
        corrected_innovations = innovations[warmup:] - corr[warmup:]
        corrected_bias = abs(float(np.mean(corrected_innovations)))

        assert corrected_bias < raw_bias, \
            f"corrected_bias={corrected_bias:.6f} >= raw_bias={raw_bias:.6f}"

    def test_output_shape(self):
        """AC6: Output has same shape as input."""
        for n in [10, 100, 500]:
            innovations = np.zeros(n)
            corr = causal_ewm_location_correction(innovations, ewm_lambda=0.96)
            assert len(corr) == n
