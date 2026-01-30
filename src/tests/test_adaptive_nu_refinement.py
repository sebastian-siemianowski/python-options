#!/usr/bin/env python3
"""
Tests for adaptive ν refinement module.

Tests cover:
- Detection criterion (needs_nu_refinement)
- Refinement candidate selection
- Likelihood flatness detection
- Full refinement workflow
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adaptive_nu_refinement import (
    AdaptiveNuConfig,
    DEFAULT_ADAPTIVE_NU_CONFIG,
    NuRefinementResult,
    AdaptiveNuRefiner,
    needs_nu_refinement,
    get_refinement_candidates,
    is_nu_likelihood_flat,
    is_phi_t_model,
    analyze_calibration_failures,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def default_config():
    """Default adaptive ν configuration."""
    return AdaptiveNuConfig()


@pytest.fixture
def sample_phi_t_result_boundary_12():
    """Sample result for φ-T model at boundary ν=12 with PIT failure."""
    return {
        'asset': 'TEST_ASSET',
        'model': 'φ-T(ν=12)',
        'noise_model': 'phi_student_t_nu_12',
        'nu': 12.0,
        'pit_ks_pvalue': 0.02,  # Below 0.05 threshold
        'bic': 1500.0,
        'model_comparison': {
            'phi_student_t_nu_8': {'ll': -745.3, 'bic': 1510.0},
            'phi_student_t_nu_12': {'ll': -745.0, 'bic': 1500.0},  # Best, flat (diff=0.3 < 1.0)
            'phi_student_t_nu_20': {'ll': -745.5, 'bic': 1520.0},
        }
    }


@pytest.fixture
def sample_phi_t_result_boundary_20():
    """Sample result for φ-T model at boundary ν=20 with PIT failure."""
    return {
        'asset': 'TEST_ASSET_20',
        'model': 'φ-T(ν=20)',
        'noise_model': 'phi_student_t_nu_20',
        'nu': 20.0,
        'pit_ks_pvalue': 0.03,
        'bic': 1480.0,
        'model_comparison': {
            'phi_student_t_nu_12': {'ll': -739.2, 'bic': 1490.0},
            'phi_student_t_nu_20': {'ll': -739.0, 'bic': 1480.0},  # Best, flat (diff=0.2)
        }
    }


@pytest.fixture
def sample_phi_t_result_passing():
    """Sample result for φ-T model with passing PIT."""
    return {
        'asset': 'TEST_ASSET_PASS',
        'model': 'φ-T(ν=8)',
        'noise_model': 'phi_student_t_nu_8',
        'nu': 8.0,
        'pit_ks_pvalue': 0.15,  # Passing
        'bic': 1450.0,
        'model_comparison': {
            'phi_student_t_nu_8': {'ll': -720.0, 'bic': 1450.0},
            'phi_student_t_nu_12': {'ll': -735.0, 'bic': 1480.0},
        }
    }


@pytest.fixture
def sample_gaussian_result():
    """Sample result for Gaussian model."""
    return {
        'asset': 'TEST_GAUSSIAN',
        'model': 'Gaussian',
        'noise_model': 'kalman_gaussian',
        'nu': None,
        'pit_ks_pvalue': 0.02,
        'bic': 1550.0,
    }


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test configuration handling."""
    
    def test_default_config_values(self, default_config):
        """Verify default configuration values."""
        assert default_config.enabled is True
        assert default_config.pit_threshold == 0.05
        assert 12.0 in default_config.boundary_nu_values
        assert 20.0 in default_config.boundary_nu_values
        assert default_config.likelihood_flatness_threshold == 1.0
    
    def test_refinement_candidates_nu_12(self, default_config):
        """Check refinement candidates for ν=12."""
        candidates = default_config.refinement_candidates.get(12.0)
        assert candidates is not None
        assert 10.0 in candidates
        assert 14.0 in candidates
    
    def test_refinement_candidates_nu_20_asymmetric(self, default_config):
        """Check asymmetric refinement for ν=20 (downward only)."""
        candidates = default_config.refinement_candidates.get(20.0)
        assert candidates is not None
        assert 16.0 in candidates
        # Should NOT include values above 20
        assert all(c < 20.0 for c in candidates)
    
    def test_config_to_dict(self, default_config):
        """Test config serialization."""
        d = default_config.to_dict()
        assert d['enabled'] is True
        assert d['pit_threshold'] == 0.05
        assert '12.0' in d['refinement_candidates'] or 12.0 in d['refinement_candidates']


# =============================================================================
# DETECTION TESTS
# =============================================================================

class TestDetection:
    """Test refinement detection criteria."""
    
    def test_needs_refinement_boundary_12_pit_failure(
        self, sample_phi_t_result_boundary_12, default_config
    ):
        """φ-T at ν=12 with PIT failure should need refinement."""
        assert needs_nu_refinement(sample_phi_t_result_boundary_12, default_config)
    
    def test_needs_refinement_boundary_20_pit_failure(
        self, sample_phi_t_result_boundary_20, default_config
    ):
        """φ-T at ν=20 with PIT failure should need refinement."""
        assert needs_nu_refinement(sample_phi_t_result_boundary_20, default_config)
    
    def test_no_refinement_for_passing_pit(
        self, sample_phi_t_result_passing, default_config
    ):
        """Passing PIT should not trigger refinement."""
        assert not needs_nu_refinement(sample_phi_t_result_passing, default_config)
    
    def test_no_refinement_for_gaussian(
        self, sample_gaussian_result, default_config
    ):
        """Gaussian models should not be refined."""
        assert not needs_nu_refinement(sample_gaussian_result, default_config)
    
    def test_no_refinement_for_non_boundary_nu(self, default_config):
        """Non-boundary ν values should not trigger refinement."""
        result = {
            'model': 'φ-T(ν=8)',
            'nu': 8.0,
            'pit_ks_pvalue': 0.02,
            'model_comparison': {},
        }
        assert not needs_nu_refinement(result, default_config)
    
    def test_no_refinement_when_disabled(
        self, sample_phi_t_result_boundary_12
    ):
        """Refinement should not trigger when disabled."""
        config = AdaptiveNuConfig(enabled=False)
        assert not needs_nu_refinement(sample_phi_t_result_boundary_12, config)


class TestLikelihoodFlatness:
    """Test likelihood flatness detection."""
    
    def test_flat_likelihood(self):
        """Small LL difference should be classified as flat."""
        result = {
            'model_comparison': {
                'phi_student_t_nu_8': {'ll': -740.0},
                'phi_student_t_nu_12': {'ll': -740.5},  # Difference = 0.5 < 1.0
            }
        }
        assert is_nu_likelihood_flat(result, threshold=1.0)
    
    def test_steep_likelihood(self):
        """Large LL difference should not be classified as flat."""
        result = {
            'model_comparison': {
                'phi_student_t_nu_8': {'ll': -740.0},
                'phi_student_t_nu_12': {'ll': -745.0},  # Difference = 5.0 > 1.0
            }
        }
        assert not is_nu_likelihood_flat(result, threshold=1.0)
    
    def test_insufficient_models(self):
        """With < 2 models, should default to flat (allow refinement)."""
        result = {
            'model_comparison': {
                'phi_student_t_nu_12': {'ll': -740.0},
            }
        }
        assert is_nu_likelihood_flat(result, threshold=1.0)


class TestModelTypeDetection:
    """Test φ-t model type detection."""
    
    def test_phi_t_with_greek_prefix(self, default_config):
        """Model names with φ-T prefix should be detected."""
        assert is_phi_t_model('φ-T(ν=12)', default_config)
        assert is_phi_t_model('φ-T(ν=4)', default_config)
    
    def test_phi_t_with_ascii_prefix(self, default_config):
        """Model names with phi_student_t prefix should be detected."""
        assert is_phi_t_model('phi_student_t_nu_12', default_config)
        assert is_phi_t_model('phi_student_t_nu_20', default_config)
    
    def test_gaussian_not_phi_t(self, default_config):
        """Gaussian models should not be detected as φ-t."""
        assert not is_phi_t_model('Gaussian', default_config)
        assert not is_phi_t_model('kalman_gaussian', default_config)
    
    def test_phi_gaussian_not_phi_t(self, default_config):
        """φ-Gaussian (not Student-t) should not be detected."""
        # Note: depends on config; default only allows Student-t
        assert not is_phi_t_model('φ-Gaussian', default_config)


# =============================================================================
# REFINEMENT CANDIDATE TESTS
# =============================================================================

class TestRefinementCandidates:
    """Test refinement candidate selection."""
    
    def test_candidates_for_nu_12(self, default_config):
        """ν=12 should have candidates [10, 14]."""
        candidates = get_refinement_candidates(12.0, default_config)
        assert 10.0 in candidates
        assert 14.0 in candidates
    
    def test_candidates_for_nu_20_asymmetric(self, default_config):
        """ν=20 should only have downward candidates [16]."""
        candidates = get_refinement_candidates(20.0, default_config)
        assert 16.0 in candidates
        assert len(candidates) == 1
    
    def test_no_candidates_for_interior_nu(self, default_config):
        """Interior ν values should have no candidates."""
        assert get_refinement_candidates(8.0, default_config) == []
        assert get_refinement_candidates(6.0, default_config) == []


# =============================================================================
# REFINER TESTS
# =============================================================================

class TestAdaptiveNuRefiner:
    """Test the refinement engine."""
    
    def test_identify_candidates(
        self,
        sample_phi_t_result_boundary_12,
        sample_phi_t_result_passing,
        sample_gaussian_result,
        default_config
    ):
        """Refiner should correctly identify candidates."""
        calibration_results = {
            'ASSET_1': sample_phi_t_result_boundary_12,
            'ASSET_2': sample_phi_t_result_passing,
            'ASSET_3': sample_gaussian_result,
        }
        
        refiner = AdaptiveNuRefiner(default_config)
        candidates = refiner.identify_candidates(calibration_results)
        
        assert 'ASSET_1' in candidates
        assert 'ASSET_2' not in candidates
        assert 'ASSET_3' not in candidates
    
    def test_refine_single_asset_with_mock(
        self,
        sample_phi_t_result_boundary_12,
        default_config
    ):
        """Test refinement with mocked fit function."""
        refiner = AdaptiveNuRefiner(default_config)
        
        # Mock fit function that returns improving results
        def mock_fit(asset, nu):
            if nu == 10.0:
                return {'bic': 1490.0, 'pit_ks_pvalue': 0.08, 'll': -742.0}
            elif nu == 14.0:
                return {'bic': 1495.0, 'pit_ks_pvalue': 0.04, 'll': -744.0}
            return {'bic': 1500.0, 'pit_ks_pvalue': 0.02, 'll': -746.0}
        
        result = refiner.refine_single_asset(
            'TEST_ASSET',
            sample_phi_t_result_boundary_12,
            mock_fit
        )
        
        assert result.refinement_attempted is True
        assert result.improvement_achieved is True
        assert result.nu_final == 10.0  # Best BIC
        assert result.bic_after == 1490.0
    
    def test_refine_single_asset_no_improvement(
        self,
        sample_phi_t_result_boundary_12,
        default_config
    ):
        """Test refinement when no improvement found."""
        refiner = AdaptiveNuRefiner(default_config)
        
        # Mock fit function that returns worse results
        def mock_fit(asset, nu):
            return {'bic': 1600.0, 'pit_ks_pvalue': 0.01, 'll': -800.0}
        
        result = refiner.refine_single_asset(
            'TEST_ASSET',
            sample_phi_t_result_boundary_12,
            mock_fit
        )
        
        assert result.refinement_attempted is True
        assert result.improvement_achieved is False
        assert result.nu_final == 12.0  # Original ν retained
    
    def test_get_summary(self, default_config):
        """Test summary generation."""
        refiner = AdaptiveNuRefiner(default_config)
        
        # Add some mock results
        refiner.refinement_log = [
            NuRefinementResult(
                asset='A', refinement_attempted=True, nu_original=12.0,
                nu_candidates_tested=[10.0, 14.0], nu_final=10.0,
                improvement_achieved=True, pit_before=0.02, pit_after=0.08,
                bic_before=1500.0, bic_after=1490.0
            ),
            NuRefinementResult(
                asset='B', refinement_attempted=True, nu_original=20.0,
                nu_candidates_tested=[16.0], nu_final=20.0,
                improvement_achieved=False, pit_before=0.03, pit_after=0.03,
                bic_before=1480.0, bic_after=1480.0
            ),
        ]
        
        summary = refiner.get_summary()
        
        assert summary['total_candidates'] == 2
        assert summary['refinement_attempted'] == 2
        assert summary['improvement_achieved'] == 1
        assert summary['improvement_rate'] == 0.5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with real-like data."""
    
    def test_full_workflow(self, default_config):
        """Test complete refinement workflow."""
        # Create calibration results
        calibration_results = {
            'AAPL': {
                'model': 'φ-T(ν=12)',
                'nu': 12.0,
                'pit_ks_pvalue': 0.02,
                'bic': 1500.0,
                'model_comparison': {
                    'phi_student_t_nu_8': {'ll': -741.2},  # Flat: diff < 1.0
                    'phi_student_t_nu_12': {'ll': -741.0},
                }
            },
            'GOOGL': {
                'model': 'φ-T(ν=8)',
                'nu': 8.0,
                'pit_ks_pvalue': 0.15,  # Passing
                'bic': 1450.0,
            },
        }
        
        # Mock fit function
        def mock_fit(asset, nu):
            return {'bic': 1495.0, 'pit_ks_pvalue': 0.06, 'll': -744.0}
        
        refiner = AdaptiveNuRefiner(default_config)
        results = refiner.run_refinement(calibration_results, mock_fit)
        
        # Only AAPL should be refined
        assert 'AAPL' in results
        assert 'GOOGL' not in results
        
        # Check summary
        summary = refiner.get_summary()
        assert summary['total_candidates'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])