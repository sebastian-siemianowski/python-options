"""
Test Story 3.5: Tuning Validation Gate.

Validates:
  1. NaN parameter caught
  2. Extreme q caught and clipped
  3. All models fail -> conservative prior
  4. Valid params pass
  5. Inf caught
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from calibration.tune_validator import (
    validate_model_params,
    validate_tune_result,
    get_conservative_prior,
    PARAM_BOUNDS,
    CONSERVATIVE_PRIOR,
)


class TestValidationGate(unittest.TestCase):
    """Tests for parameter validation gate."""

    def test_valid_params_pass(self):
        """Normal params should pass."""
        params = {"q": 1e-5, "c": 1.0, "phi": 0.98, "nu": 8.0, "bic": -5000}
        result = validate_model_params("test_model", params)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.warnings), 0)

    def test_nan_caught(self):
        """NaN parameter should fail validation."""
        params = {"q": float('nan'), "c": 1.0, "phi": 0.98}
        result = validate_model_params("test_model", params)
        self.assertFalse(result.passed)
        self.assertTrue(any("NaN" in w for w in result.warnings))

    def test_inf_caught(self):
        """Inf parameter should fail validation."""
        params = {"q": float('inf'), "c": 1.0, "phi": 0.98}
        result = validate_model_params("test_model", params)
        self.assertFalse(result.passed)

    def test_extreme_q_clipped(self):
        """Extremely small q should be clipped."""
        params = {"q": 1e-15, "c": 1.0, "phi": 0.98}
        result = validate_model_params("test_model", params)
        self.assertTrue(result.passed)  # Passes but with warning
        self.assertIn("q", result.clipped_params)
        self.assertEqual(result.clipped_params["q"], PARAM_BOUNDS["q"][0])

    def test_none_params_fail(self):
        """None params should fail."""
        result = validate_model_params("test_model", None)
        self.assertFalse(result.passed)

    def test_positive_bic_warned(self):
        """Positive BIC should be warned (usually means bad fit)."""
        params = {"q": 1e-5, "bic": 1000}
        result = validate_model_params("test_model", params)
        self.assertTrue(any("bic" in w for w in result.warnings))

    def test_validate_tune_result(self):
        """Validate full tune result structure."""
        tune_result = {
            "global": {
                "q": 1e-5,
                "c": 1.0,
                "phi": 0.98,
                "models": {
                    "model_a": {"q": 1e-5, "c": 1.0, "phi": 0.98},
                    "model_b": {"q": float('nan'), "c": 1.0},
                }
            }
        }
        summary = validate_tune_result(tune_result)
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertFalse(summary["all_failed"])

    def test_all_models_fail(self):
        """All models fail -> all_failed=True."""
        tune_result = {
            "global": {
                "models": {
                    "model_a": {"q": float('nan')},
                    "model_b": None,
                }
            }
        }
        summary = validate_tune_result(tune_result)
        self.assertTrue(summary["all_failed"])

    def test_conservative_prior(self):
        """Conservative prior has valid params."""
        prior = get_conservative_prior()
        for key in ["q", "c", "phi", "nu"]:
            self.assertIn(key, prior)
        result = validate_model_params("conservative_prior", prior)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
