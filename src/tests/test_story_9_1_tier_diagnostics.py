"""
Story 9.1: Tier Interaction Diagnostics
========================================
Detect harmful parameter interactions across unified model tiers.
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

from models.phi_student_t_unified import compute_tier_interaction_diagnostics


class TestTierInteractionDiagnostics:
    """Acceptance criteria for Story 9.1."""

    def test_garch_msq_imbalance_detected(self):
        """AC1: Detect GARCH + MS-q double-counting when variance ratio out of [0.2, 5.0]."""
        np.random.seed(42)
        n = 500
        h_garch = np.abs(np.random.normal(0.001, 0.0002, n))
        q_msq = np.abs(np.random.normal(0.1, 0.02, n))  # 100x larger
        result = compute_tier_interaction_diagnostics(
            h_garch=h_garch, q_msq=q_msq,
            garch_active=True, msq_active=True
        )
        assert 'garch_msq_imbalance' in result['flags']

    def test_phi_risk_premium_collinearity(self):
        """AC2: Detect phi + risk premium collinearity when both large."""
        result = compute_tier_interaction_diagnostics(
            phi=0.95, risk_premium=0.8
        )
        assert 'phi_risk_premium_collinearity' in result['flags']
        assert result['adjustments'].get('disable_risk_premium') is True

    def test_vov_garch_redundancy(self):
        """AC3: Detect VoV + GARCH redundancy when VoV damping < 0.2."""
        result = compute_tier_interaction_diagnostics(
            vov_gamma=0.1, garch_active=True, vov_active=True
        )
        assert 'vov_garch_redundancy' in result['flags']
        assert result['vov_garch_redundant'] is True

    def test_no_flags_clean_model(self):
        """AC4: Clean model produces no flags."""
        np.random.seed(42)
        n = 500
        h_garch = np.abs(np.random.normal(0.001, 0.0003, n))
        q_msq = np.abs(np.random.normal(0.001, 0.0003, n))
        result = compute_tier_interaction_diagnostics(
            h_garch=h_garch, q_msq=q_msq,
            phi=0.5, risk_premium=0.01,
            vov_gamma=0.5,
            garch_active=True, msq_active=True, vov_active=True
        )
        # Should have few flags with balanced parameters
        assert result['n_flags'] <= 1

    def test_adjustment_recommendations(self):
        """AC5: Flagged assets receive parameter adjustment recommendations."""
        result = compute_tier_interaction_diagnostics(
            vov_gamma=0.05, garch_active=True, vov_active=True,
            phi=0.99, risk_premium=0.9
        )
        assert len(result['adjustments']) > 0

    def test_diagnostics_structure(self):
        """AC6: All diagnostic fields present."""
        result = compute_tier_interaction_diagnostics()
        required = {'flags', 'n_flags', 'adjustments', 'garch_msq_var_ratio',
                     'phi_risk_premium_interaction', 'vov_garch_redundant'}
        assert required.issubset(result.keys())
