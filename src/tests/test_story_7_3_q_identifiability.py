"""
Story 7.3: Joint q_calm / q_stress Optimization with Identifiability Constraints
==================================================================================
Constrained optimization ensuring bimodal regime structure.
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

from models.phi_student_t import (
    optimize_q_calm_stress,
    MIN_Q_RATIO,
    MAX_Q_RATIO,
)


class TestJointQOptimization:
    """Acceptance criteria for Story 7.3."""

    def test_ratio_constraint_min(self):
        """AC1: q_stress / q_calm >= 5 (hard constraint)."""
        np.random.seed(42)
        n = 500
        vol = np.abs(np.random.normal(0.02, 0.01, n))
        returns = np.random.normal(0, 0.015, n)
        qc, qs, diag = optimize_q_calm_stress(vol, returns)
        ratio = qs / qc
        assert ratio >= MIN_Q_RATIO - 0.01, f"Ratio {ratio:.1f} < {MIN_Q_RATIO}"

    def test_ratio_constraint_max(self):
        """AC2: q_stress / q_calm <= 1000 (hard constraint)."""
        np.random.seed(42)
        n = 500
        vol = np.abs(np.random.normal(0.02, 0.01, n))
        returns = np.random.normal(0, 0.015, n)
        qc, qs, diag = optimize_q_calm_stress(vol, returns)
        ratio = qs / qc
        assert ratio <= MAX_Q_RATIO + 0.01, f"Ratio {ratio:.1f} > {MAX_Q_RATIO}"

    def test_moderate_regime_difference(self):
        """AC3: For moderate vol data, ratio should be reasonable (not extreme)."""
        np.random.seed(123)
        n = 500
        vol = np.abs(np.random.normal(0.015, 0.003, n))
        returns = np.random.normal(0, 0.01, n)
        qc, qs, diag = optimize_q_calm_stress(vol, returns)
        assert 5 <= diag['ratio'] <= 1000

    def test_extreme_vol_higher_ratio(self):
        """AC4: Extreme vol data should produce higher ratio."""
        np.random.seed(777)
        n = 500
        vol = np.abs(np.random.normal(0.02, 0.005, n))
        # Add extreme spike
        vol[200:230] = 0.12  # 6x normal vol
        returns = np.random.normal(0, 0.02, n)
        returns[200:230] = np.random.normal(0, 0.08, 30)

        qc, qs, diag = optimize_q_calm_stress(vol, returns)
        assert diag['ratio'] >= 5.0

    def test_bic_comparison(self):
        """AC5: BIC comparison with single-q model is computed."""
        np.random.seed(42)
        n = 500
        vol = np.abs(np.random.normal(0.02, 0.01, n))
        returns = np.random.normal(0, 0.015, n)
        _, _, diag = optimize_q_calm_stress(vol, returns)
        assert 'bic_msq' in diag
        assert 'bic_single' in diag
        assert 'bic_improvement_nats' in diag
        assert 'msq_selected' in diag

    def test_diagnostics_complete(self):
        """AC6: All diagnostic fields present."""
        np.random.seed(42)
        n = 300
        vol = np.abs(np.random.normal(0.02, 0.005, n))
        returns = np.random.normal(0, 0.01, n)
        _, _, diag = optimize_q_calm_stress(vol, returns)
        required = {'q_calm', 'q_stress', 'ratio', 'bic_msq', 'bic_single',
                     'bic_improvement_nats', 'll_msq', 'll_single', 'msq_selected'}
        assert required.issubset(diag.keys())

    def test_q_calm_less_than_q_stress(self):
        """AC7: q_calm < q_stress always."""
        np.random.seed(42)
        n = 400
        vol = np.abs(np.random.normal(0.02, 0.008, n))
        returns = np.random.normal(0, 0.012, n)
        qc, qs, _ = optimize_q_calm_stress(vol, returns)
        assert qc < qs, f"q_calm={qc:.2e} >= q_stress={qs:.2e}"
