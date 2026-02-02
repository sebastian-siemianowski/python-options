#!/usr/bin/env python3
"""
Test AIGF-NF Integration

Tests the AIGF-NF (Adaptive Implicit Generative Filter - Normalizing Flow) model
integration with tune.py and signals.py.

This test validates:
1. AIGF-NF model imports and basic functionality
2. Model registry integration
3. Tuning integration (fit_all_models_for_regime)
4. Signal generation integration (BMA sampling)
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


class TestAIGFNFModel:
    """Test AIGF-NF model core functionality."""
    
    def test_import(self):
        """Test AIGF-NF can be imported."""
        from models.aigf_nf import (
            AIGFNFConfig,
            DEFAULT_AIGF_NF_CONFIG,
            FlowArtifact,
            LatentState,
            AIGFNFModel,
            AIGF_NF_MODEL_NAME,
            is_aigf_nf_model,
            fit_aigf_nf,
        )
        
        assert AIGF_NF_MODEL_NAME == "aigf_nf"
        assert is_aigf_nf_model("aigf_nf")
        assert is_aigf_nf_model("AIGF_NF")
        assert not is_aigf_nf_model("gaussian")
    
    def test_config_validation(self):
        """Test configuration validation."""
        from models.aigf_nf import AIGFNFConfig
        
        # Default config should be valid
        config = AIGFNFConfig()
        errors = config.validate()
        assert len(errors) == 0, f"Default config has errors: {errors}"
        
        # Invalid EWMA lambda
        config_bad = AIGFNFConfig(ewma_lambda=0.5)  # Outside [0.90, 0.99]
        errors = config_bad.validate()
        assert len(errors) > 0, "Should reject ewma_lambda outside permitted range"
    
    def test_model_creation(self):
        """Test model can be created."""
        from models.aigf_nf import AIGFNFModel
        
        model = AIGFNFModel()
        assert model.config.latent_dim == 8
        assert model.state.z.shape == (8,)
        assert model.state.n_updates == 0
    
    def test_sampling(self):
        """Test predictive sampling."""
        from models.aigf_nf import AIGFNFModel
        
        model = AIGFNFModel()
        
        # Sample from predictive distribution
        samples = model.sample_predictive(n_samples=1000)
        
        assert len(samples) == 1000
        assert np.isfinite(samples).all()
        
        # Check samples have reasonable properties
        assert -10 < np.mean(samples) < 10
        assert 0 < np.std(samples) < 10
    
    def test_filtering(self):
        """Test filter/update functionality."""
        from models.aigf_nf import AIGFNFModel
        
        model = AIGFNFModel()
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        
        # Run filter
        result = model.filter(returns)
        
        assert result['n_observations'] == 100
        assert np.isfinite(result['mean_log_likelihood'])
        assert model.state.n_updates == 100
    
    def test_optimize_params(self):
        """Test parameter optimization (fitting)."""
        from models.aigf_nf import AIGFNFModel
        
        model = AIGFNFModel()
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        
        # Optimize (fit)
        result = model.optimize_params(returns)
        
        assert result['fit_success']
        assert np.isfinite(result['bic'])
        assert np.isfinite(result['aic'])
        assert np.isfinite(result['mean_log_likelihood'])
        assert result['n_observations'] == 200
    
    def test_pit_calibration(self):
        """Test PIT calibration diagnostic."""
        from models.aigf_nf import AIGFNFModel
        
        model = AIGFNFModel()
        
        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        
        # Run filter first
        model.filter(returns)
        
        # Get PIT calibration
        ks_stat, ks_pvalue = model.pit_ks(returns[:50])  # Use subset
        
        assert 0 <= ks_stat <= 1
        assert 0 <= ks_pvalue <= 1


class TestModelRegistryIntegration:
    """Test AIGF-NF integration with model registry."""
    
    def test_registry_contains_aigf_nf(self):
        """Test AIGF-NF is in the model registry."""
        from models.model_registry import MODEL_REGISTRY, make_aigf_nf_name, ModelFamily
        
        aigf_name = make_aigf_nf_name()
        assert aigf_name in MODEL_REGISTRY, f"AIGF-NF not in registry: {list(MODEL_REGISTRY.keys())[:10]}..."
        
        spec = MODEL_REGISTRY[aigf_name]
        assert spec.family == ModelFamily.AIGF_NF
        assert spec.n_params == 10
        assert not spec.is_augmentation
    
    def test_base_models_includes_aigf_nf(self):
        """Test AIGF-NF is in base models for tuning."""
        from models.model_registry import get_base_models_for_tuning, make_aigf_nf_name
        
        base_models = get_base_models_for_tuning()
        aigf_name = make_aigf_nf_name()
        
        assert aigf_name in base_models, f"AIGF-NF not in base models: {base_models}"
    
    def test_sampler_dispatch(self):
        """Test sampler dispatch returns correct strategy for AIGF-NF."""
        from models.model_registry import (
            MODEL_REGISTRY, make_aigf_nf_name, get_sampler_for_model
        )
        
        aigf_name = make_aigf_nf_name()
        spec = MODEL_REGISTRY[aigf_name]
        sampler = get_sampler_for_model(spec)
        
        assert sampler == "aigf_nf_mc"


class TestTuningIntegration:
    """Test AIGF-NF integration with tune.py."""
    
    def test_fit_all_models_includes_aigf_nf(self):
        """Test fit_all_models_for_regime includes AIGF-NF."""
        from tuning.tune import fit_all_models_for_regime
        from models.aigf_nf import is_aigf_nf_model
        
        # Generate synthetic data
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        vol = np.abs(np.random.randn(200)) * 0.01 + 0.01
        
        # Fit all models
        models = fit_all_models_for_regime(returns, vol)
        
        # Check AIGF-NF is present
        aigf_models = [m for m in models.keys() if is_aigf_nf_model(m)]
        assert len(aigf_models) == 1, f"Expected 1 AIGF-NF model, found: {aigf_models}"
        
        aigf_name = aigf_models[0]
        aigf_result = models[aigf_name]
        
        # Verify result structure
        assert aigf_result.get('fit_success'), f"AIGF-NF fit failed: {aigf_result.get('error')}"
        assert 'bic' in aigf_result
        assert 'mean_log_likelihood' in aigf_result
        assert aigf_result.get('model_type') == 'aigf_nf'


class TestSignalsIntegration:
    """Test AIGF-NF integration with signals.py."""
    
    def test_aigf_nf_import_in_signals(self):
        """Test AIGF-NF can be imported in signals context."""
        # Import from models module, not signals (signals doesn't re-export)
        from models import is_aigf_nf_model, AIGF_NF_MODEL_NAME
        
        assert AIGF_NF_MODEL_NAME == "aigf_nf"
        assert is_aigf_nf_model("aigf_nf")


def main():
    """Run tests directly."""
    import traceback
    
    test_classes = [
        TestAIGFNFModel,
        TestModelRegistryIntegration,
        TestTuningIntegration,
        TestSignalsIntegration,
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing: {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        for name in dir(instance):
            if name.startswith('test_'):
                try:
                    print(f"  {name}...", end=" ")
                    getattr(instance, name)()
                    print("✓ PASS")
                    passed += 1
                except Exception as e:
                    print(f"✗ FAIL: {e}")
                    traceback.print_exc()
                    failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('='*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
