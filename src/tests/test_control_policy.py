#!/usr/bin/env python3
"""
Test suite for Control Policy module.

Counter-Proposal v1.0 Implementation Tests.
"""
import pytest
import numpy as np
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.control_policy import (
    EscalationDecision,
    CalibrationDiagnostics,
    ControlPolicy,
    AdaptiveRefinementConfig,
    TuningAuditRecord,
    EscalationStatistics,
    DECISION_NAMES,
    DEFAULT_CONTROL_POLICY,
    DEFAULT_REFINEMENT_CONFIG,
    verify_control_policy_architecture,
)


class TestEscalationDecision:
    """Test EscalationDecision enum."""
    
    def test_all_decisions_defined(self):
        """Verify all expected decisions exist."""
        decisions = list(EscalationDecision)
        assert EscalationDecision.HOLD_CURRENT in decisions
        assert EscalationDecision.REFINE_NU in decisions
        assert EscalationDecision.APPLY_MIXTURE in decisions
        assert EscalationDecision.FALLBACK_GH in decisions
        assert EscalationDecision.FALLBACK_TVVM in decisions
        assert EscalationDecision.REJECT_ASSET in decisions
    
    def test_decision_names(self):
        """Verify decision names for audit trail."""
        for decision in EscalationDecision:
            assert decision in DECISION_NAMES


class TestCalibrationDiagnostics:
    """Test immutable CalibrationDiagnostics."""
    
    @pytest.fixture
    def sample_diagnostics(self):
        """Create sample diagnostics."""
        return CalibrationDiagnostics(
            asset='TEST',
            pit_ks_pvalue=0.03,
            ks_statistic=0.15,
            excess_kurtosis=7.5,
            skewness=-0.6,
            current_nu=6.0,
            regime_id=1,
            bic_current=1500.0,
            n_observations=500,
            realized_volatility=0.20,
        )
    
    def test_immutability(self, sample_diagnostics):
        """Test that diagnostics are frozen."""
        with pytest.raises(Exception):  # FrozenInstanceError
            sample_diagnostics.pit_ks_pvalue = 0.5
    
    def test_is_severe(self, sample_diagnostics):
        """Test severe threshold detection."""
        assert not sample_diagnostics.is_severe  # 0.03 >= 0.01
        
        severe = CalibrationDiagnostics(
            asset='TEST', pit_ks_pvalue=0.005, ks_statistic=0.2,
            excess_kurtosis=5.0, skewness=0.0, current_nu=None,
            regime_id=1, bic_current=1500.0, n_observations=500,
            realized_volatility=0.15,
        )
        assert severe.is_severe
    
    def test_is_warning(self, sample_diagnostics):
        """Test warning threshold detection."""
        assert sample_diagnostics.is_warning  # 0.03 < 0.05
    
    def test_is_fat_tailed(self, sample_diagnostics):
        """Test fat tail detection."""
        assert sample_diagnostics.is_fat_tailed  # 7.5 > 6.0
    
    def test_is_skewed(self, sample_diagnostics):
        """Test skewness detection."""
        assert sample_diagnostics.is_skewed  # |−0.6| > 0.5
        assert sample_diagnostics.is_left_skewed
        assert not sample_diagnostics.is_right_skewed
    
    def test_to_dict(self, sample_diagnostics):
        """Test export for audit trail."""
        d = sample_diagnostics.to_dict()
        assert d['asset'] == 'TEST'
        assert d['is_warning'] == True
        assert d['is_skewed'] == True


class TestControlPolicy:
    """Test ControlPolicy decision making."""
    
    @pytest.fixture
    def policy(self):
        """Create default policy."""
        return ControlPolicy()
    
    @pytest.fixture
    def calibrated_diagnostics(self):
        """Diagnostics that pass calibration."""
        return CalibrationDiagnostics(
            asset='GOOD', pit_ks_pvalue=0.15, ks_statistic=0.08,
            excess_kurtosis=3.0, skewness=0.1, current_nu=8.0,
            regime_id=1, bic_current=1400.0, n_observations=600,
            realized_volatility=0.18,
        )
    
    @pytest.fixture
    def warning_diagnostics(self):
        """Diagnostics with warning level."""
        return CalibrationDiagnostics(
            asset='WARN', pit_ks_pvalue=0.03, ks_statistic=0.12,
            excess_kurtosis=5.0, skewness=0.2, current_nu=6.0,
            regime_id=1, bic_current=1500.0, n_observations=500,
            realized_volatility=0.20,
        )
    
    @pytest.fixture
    def severe_skewed_diagnostics(self):
        """Diagnostics with severe miscalibration + skew."""
        return CalibrationDiagnostics(
            asset='SEVERE', pit_ks_pvalue=0.005, ks_statistic=0.20,
            excess_kurtosis=9.0, skewness=-0.8, current_nu=4.0,
            regime_id=3, bic_current=1600.0, n_observations=400,
            realized_volatility=0.30,
        )
    
    def test_hold_current_when_calibrated(self, policy, calibrated_diagnostics):
        """No escalation needed when calibration passes."""
        decision = policy.decide(calibrated_diagnostics, [])
        assert decision == EscalationDecision.HOLD_CURRENT
    
    def test_refine_nu_on_warning(self, policy, warning_diagnostics):
        """First escalation should be ν refinement."""
        decision = policy.decide(warning_diagnostics, [])
        assert decision == EscalationDecision.REFINE_NU
    
    def test_gh_on_severe_skewed(self, policy, severe_skewed_diagnostics):
        """GH fallback for severe + skewed."""
        decision = policy.decide(severe_skewed_diagnostics, [])
        assert decision == EscalationDecision.FALLBACK_GH
    
    def test_escalation_budget(self, policy, warning_diagnostics):
        """Escalation budget prevents cascade."""
        # Fill up the budget
        history = [
            EscalationDecision.REFINE_NU,
            EscalationDecision.APPLY_MIXTURE,
        ]
        
        decision = policy.decide(warning_diagnostics, history)
        assert decision == EscalationDecision.HOLD_CURRENT
    
    def test_regime_conditioned_budget(self, policy):
        """Crisis regime has higher escalation budget."""
        crisis_diag = CalibrationDiagnostics(
            asset='CRISIS', pit_ks_pvalue=0.02, ks_statistic=0.15,
            excess_kurtosis=8.0, skewness=0.3, current_nu=6.0,
            regime_id=4,  # CRISIS regime
            bic_current=1500.0, n_observations=500,
            realized_volatility=0.40,
        )
        
        # In crisis, should allow more escalation
        budget = policy.max_escalations_by_regime[4]
        assert budget >= 3  # Crisis allows more


class TestTrustPenalty:
    """Test trust penalty computation."""
    
    @pytest.fixture
    def policy(self):
        return ControlPolicy()
    
    @pytest.fixture
    def diagnostics(self):
        return CalibrationDiagnostics(
            asset='TEST', pit_ks_pvalue=0.03, ks_statistic=0.12,
            excess_kurtosis=5.0, skewness=0.2, current_nu=6.0,
            regime_id=1, bic_current=1500.0, n_observations=500,
            realized_volatility=0.20,
        )
    
    def test_gaussian_minimal_penalty(self, policy, diagnostics):
        """Gaussian should have minimal model penalty."""
        penalty = policy.compute_trust_penalty('gaussian', 1, diagnostics)
        assert penalty < 0.15  # Low model penalty + some calibration penalty
    
    def test_gh_high_penalty(self, policy, diagnostics):
        """GH fallback should have highest model penalty."""
        penalty = policy.compute_trust_penalty('gh', 1, diagnostics)
        assert penalty >= 0.20  # High model penalty
    
    def test_penalty_bounded(self, policy, diagnostics):
        """Total penalty should be bounded."""
        for model in ['gaussian', 'phi_student_t_nu_4', 'gh', 'tvvm']:
            for regime in range(5):
                penalty = policy.compute_trust_penalty(model, regime, diagnostics)
                assert penalty <= policy.max_total_penalty
    
    def test_effective_trust_computation(self, policy, diagnostics):
        """Test effective trust with audit decomposition."""
        trust, audit = policy.compute_effective_trust(
            base_trust=0.80,
            model_type='phi_student_t_nu_6',
            regime_id=1,
            diagnostics=diagnostics,
        )
        
        assert 0.0 <= trust <= 1.0
        assert 'model_penalty' in audit
        assert 'regime_penalty' in audit
        assert audit['effective_trust'] == trust


class TestAdaptiveRefinementConfig:
    """Test adaptive flatness threshold."""
    
    @pytest.fixture
    def config(self):
        return AdaptiveRefinementConfig()
    
    def test_regime_conditioning(self, config):
        """Crisis regime should have higher threshold."""
        threshold_normal = config.get_threshold(
            regime_id=1, n_observations=1000, realized_volatility=0.15
        )
        threshold_crisis = config.get_threshold(
            regime_id=4, n_observations=1000, realized_volatility=0.15
        )
        
        assert threshold_crisis > threshold_normal
    
    def test_information_scaling(self, config):
        """More observations should allow stricter threshold."""
        threshold_small = config.get_threshold(
            regime_id=1, n_observations=500, realized_volatility=0.15
        )
        threshold_large = config.get_threshold(
            regime_id=1, n_observations=2000, realized_volatility=0.15
        )
        
        # Larger sample → threshold increases (stricter)
        assert threshold_large >= threshold_small
    
    def test_volatility_adjustment(self, config):
        """High volatility should increase threshold (more tolerance)."""
        threshold_low_vol = config.get_threshold(
            regime_id=1, n_observations=1000, realized_volatility=0.10
        )
        threshold_high_vol = config.get_threshold(
            regime_id=1, n_observations=1000, realized_volatility=0.40
        )
        
        assert threshold_high_vol > threshold_low_vol


class TestEscalationStatistics:
    """Test escalation statistics tracking."""
    
    def test_record_decision(self):
        """Test decision counting."""
        stats = EscalationStatistics()
        
        stats.record_decision(EscalationDecision.HOLD_CURRENT)
        stats.record_decision(EscalationDecision.HOLD_CURRENT)
        stats.record_decision(EscalationDecision.REFINE_NU)
        
        assert stats.hold_current_count == 2
        assert stats.refine_nu_count == 1
    
    def test_escalation_rate(self):
        """Test escalation rate computation."""
        stats = EscalationStatistics()
        stats.total_assets = 10
        
        stats.hold_current_count = 7
        stats.refine_nu_count = 2
        stats.apply_mixture_count = 1
        
        assert stats.escalation_rate == 0.3  # 3/10
    
    def test_trust_statistics(self):
        """Test trust value tracking."""
        stats = EscalationStatistics()
        
        stats.record_trust(0.80)
        stats.record_trust(0.60)
        stats.record_trust(0.70)
        
        assert abs(stats.mean_trust - 0.70) < 0.01


class TestArchitectureVerification:
    """Test architecture verification checks."""
    
    def test_all_checks_pass(self):
        """All architecture checks should pass with default config."""
        checks = verify_control_policy_architecture()
        
        for check_name, passed in checks.items():
            assert passed, f"Check failed: {check_name}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
