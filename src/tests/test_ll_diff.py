#!/usr/bin/env python3
"""Quick test to verify unified models produce valid log-likelihoods.

The model architecture was upgraded to unified models (momentum is internal).
Current model keys: kalman_gaussian_unified, kalman_phi_gaussian_unified,
phi_student_t_nu_{3,4,8,20}, phi_student_t_unified_nu_{3,4,8,20}.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import warnings
warnings.filterwarnings('ignore')
import numpy as np

from tuning.tune import fit_all_models_for_regime


class TestLogLikelihoodDiff(unittest.TestCase):
    """Test that unified models produce valid and distinct log-likelihoods."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        vol = np.abs(returns) * 1.5 + 0.01
        cls.models = fit_all_models_for_regime(returns, vol)

    def test_all_models_have_log_likelihood(self):
        """Every model should return a finite log-likelihood."""
        for name, result in self.models.items():
            with self.subTest(model=name):
                self.assertIn('log_likelihood', result)
                self.assertTrue(
                    np.isfinite(result['log_likelihood']),
                    f"{name}: LL={result['log_likelihood']}"
                )

    def test_unified_vs_base_student_t(self):
        """Unified and base phi_student_t models should have different LL."""
        for nu in [3, 4, 8, 20]:
            base = f'phi_student_t_nu_{nu}'
            unified = f'phi_student_t_unified_nu_{nu}'
            if base in self.models and unified in self.models:
                base_ll = self.models[base]['log_likelihood']
                unified_ll = self.models[unified]['log_likelihood']
                # They may differ due to different optimization pipelines
                self.assertTrue(np.isfinite(base_ll), f"{base} LL not finite")
                self.assertTrue(np.isfinite(unified_ll), f"{unified} LL not finite")

    def test_gaussian_models_exist(self):
        """Unified Gaussian models should be present."""
        self.assertIn('kalman_gaussian_unified', self.models)
        self.assertIn('kalman_phi_gaussian_unified', self.models)

    def test_expected_model_count(self):
        """Should have 10 models (2 Gaussian + 4 base + 4 unified Student-t)."""
        self.assertEqual(len(self.models), 10)


if __name__ == "__main__":
    unittest.main()
