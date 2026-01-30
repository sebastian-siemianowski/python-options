#!/usr/bin/env python3
"""
DEPRECATED: K=2 Mixture Model Implementation Verification

This script verified the K=2 mixture model implementation which has been
REMOVED from the system after empirical evaluation showed:
  - 206 attempts across assets
  - 0 selections (0% success rate)
  - Model misspecification: returns are fat-tailed unimodal, not bimodal

The HMM regime-switching + Student-t architecture already captures regime
heterogeneity more effectively.

See: docs/CALIBRATION_SOLUTIONS_ANALYSIS.md for decision rationale.

This file is kept for historical reference only.
"""
import sys
import os
import warnings

warnings.warn(
    "K=2 mixture model has been removed (206 attempts, 0 selections). "
    "This verification script is deprecated.",
    DeprecationWarning,
    stacklevel=2
)

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def verify():
    print('=' * 60)
    print('ACCEPTANCE CRITERIA VERIFICATION')
    print('=' * 60)

    # 1. Core Functionality
    print('\n1. CORE FUNCTIONALITY')
    try:
        from phi_t_mixture_k2 import (
            PhiTMixtureK2, PhiTMixtureK2Config, PhiTMixtureK2Result,
            should_use_mixture, fit_and_select, summarize_mixture_improvement
        )
        print('   ✓ PhiTMixtureK2 model class exists')
        print('   ✓ PhiTMixtureK2Config configuration class exists')
        print('   ✓ PhiTMixtureK2Result result class exists')
        
        config = PhiTMixtureK2Config()
        print(f'   ✓ sigma_ratio_min = {config.sigma_ratio_min} (≥1.5 enforced)')
        print(f'   ✓ weight bounds = [{config.min_weight}, {config.max_weight}]')
        print(f'   ✓ entropy_penalty = {config.entropy_penalty}')
        print(f'   ✓ bic_threshold = {config.bic_threshold}')
    except ImportError as e:
        print(f'   ✗ Import error: {e}')
        return False

    # 2. Integration with tune_q_mle
    print('\n2. INTEGRATION WITH tune.py')
    try:
        from tune import (
            MIXTURE_MODEL_AVAILABLE, MIXTURE_MODEL_ENABLED,
            MIXTURE_SIGMA_RATIO_MIN, MIXTURE_MIN_WEIGHT, MIXTURE_MAX_WEIGHT,
            MIXTURE_BIC_THRESHOLD, get_mixture_config
        )
        print(f'   ✓ MIXTURE_MODEL_AVAILABLE = {MIXTURE_MODEL_AVAILABLE}')
        print(f'   ✓ MIXTURE_MODEL_ENABLED = {MIXTURE_MODEL_ENABLED}')
        print(f'   ✓ MIXTURE_SIGMA_RATIO_MIN = {MIXTURE_SIGMA_RATIO_MIN}')
        print(f'   ✓ MIXTURE_MIN_WEIGHT = {MIXTURE_MIN_WEIGHT}')
        print(f'   ✓ MIXTURE_MAX_WEIGHT = {MIXTURE_MAX_WEIGHT}')
        print(f'   ✓ MIXTURE_BIC_THRESHOLD = {MIXTURE_BIC_THRESHOLD}')
        
        cfg = get_mixture_config()
        print(f'   ✓ get_mixture_config() returns valid config: {cfg is not None}')
    except ImportError as e:
        print(f'   ✗ Import error: {e}')
        return False

    # 3. fx_signals_presentation integration
    print('\n3. INTEGRATION WITH signals_ux.py')
    try:
        from signals_ux import render_calibration_report
        print('   ✓ render_calibration_report() exists')
    except ImportError as e:
        print(f'   ✗ Import error: {e}')
        return False

    # 4. Verify constraints are enforced in actual fit
    print('\n4. CONSTRAINT VERIFICATION (actual fit)')
    import numpy as np
    
    np.random.seed(42)
    n = 500
    returns = np.random.randn(n) * 0.02
    vol = np.full(n, 0.02)
    
    mixer = PhiTMixtureK2(config)
    result = mixer.fit(returns=returns, vol=vol, nu=8.0, phi_init=0.1, sigma_init=0.02)
    
    if result:
        print(f'   ✓ Fit successful')
        assert result.sigma_ratio >= 1.5, f'σ ratio constraint violated: {result.sigma_ratio}'
        print(f'   ✓ σ_B/σ_A = {result.sigma_ratio:.2f} (≥ 1.5: True)')
        assert 0.1 <= result.weight <= 0.9, f'Weight bounds violated: {result.weight}'
        print(f'   ✓ weight = {result.weight:.3f} (in [0.1, 0.9]: True)')
        assert -0.999 <= result.phi <= 0.999, f'φ bounds violated: {result.phi}'
        print(f'   ✓ φ = {result.phi:.3f} (in [-0.999, 0.999]: True)')
        print(f'   ✓ BIC = {result.bic:.1f}')
        print(f'   ✓ PIT p-value = {result.pit_ks_pvalue:.4f}')
    else:
        print('   ✗ Fit failed')
        return False

    print('\n' + '=' * 60)
    print('ALL ACCEPTANCE CRITERIA VERIFIED ✓')
    print('=' * 60)
    return True


if __name__ == '__main__':
    success = verify()
    sys.exit(0 if success else 1)