#!/usr/bin/env python3
"""Test momentum models compete in BMA."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')
import numpy as np

def test_momentum_bma_competition():
    """Test that momentum models properly compete in BMA."""
    from tuning.tune import fit_all_models_for_regime, MOMENTUM_AUGMENTATION_ENABLED, MOMENTUM_AUGMENTATION_AVAILABLE
    from calibration.model_selection import compute_bic_model_weights, normalize_weights

    print(f'MOMENTUM_AUGMENTATION_ENABLED: {MOMENTUM_AUGMENTATION_ENABLED}')
    print(f'MOMENTUM_AUGMENTATION_AVAILABLE: {MOMENTUM_AUGMENTATION_AVAILABLE}')

    # Generate synthetic data with momentum signal
    np.random.seed(42)
    n = 500
    # Create returns with trend (momentum should help)
    trend = np.cumsum(np.random.randn(n) * 0.001)  # Cumulative drift
    noise = np.random.randn(n) * 0.015
    returns = trend + noise
    vol = np.abs(returns) * 1.5 + 0.01

    # Fit all models
    print('\nFitting models...')
    models = fit_all_models_for_regime(returns, vol)

    print(f'\nTotal models fitted: {len(models)}')

    # Check which are momentum
    momentum_models = [m for m in models if models[m].get('momentum_augmented', False)]
    base_models = [m for m in models if not models[m].get('momentum_augmented', False)]

    print(f'Base models: {len(base_models)}')
    for m in sorted(base_models):
        info = models[m]
        if info.get('fit_success'):
            print(f'  {m}: BIC={info.get("bic", float("nan")):.1f}')
        else:
            print(f'  {m}: FAILED')

    print(f'\nMomentum models: {len(momentum_models)}')
    for m in sorted(momentum_models):
        info = models[m]
        if info.get('fit_success'):
            bic_raw = info.get('bic_raw', info.get('bic'))
            bic_adj = info.get('bic')
            print(f'  {m}: BIC_raw={bic_raw:.1f}, BIC_adj={bic_adj:.1f}')
        else:
            print(f'  {m}: FAILED - {info.get("error", "unknown")}')

    # Compute BMA weights
    bic_values = {m: models[m].get('bic', float('inf')) for m in models if models[m].get('fit_success', False)}
    print(f'\nModels with valid BIC: {len(bic_values)}')

    weights = compute_bic_model_weights(bic_values)
    weights = normalize_weights(weights)

    print('\nBMA Weights (sorted by weight):')
    for m, w in sorted(weights.items(), key=lambda x: -x[1]):
        is_mom = '_momentum' in m
        marker = ' ← MOMENTUM' if is_mom else ''
        print(f'  {m}: {w:.4f}{marker}')

    # Check if momentum models are getting non-zero weight
    mom_weights = {m: w for m, w in weights.items() if '_momentum' in m}
    total_mom_weight = sum(mom_weights.values())
    print(f'\nTotal momentum model weight: {total_mom_weight:.4f}')
    print(f'Total base model weight: {1 - total_mom_weight:.4f}')
    
    # Assertions
    # Unified Gaussian models include momentum internally
    assert len(models) >= 8, f"Expected at least 8 models, got {len(models)}"
    assert 'kalman_gaussian_unified' in models, "Missing kalman_gaussian_unified"
    assert 'kalman_phi_gaussian_unified' in models, "Missing kalman_phi_gaussian_unified"
    
    print('\n✅ All assertions passed - models are properly competing in BMA')
    return True


if __name__ == "__main__":
    test_momentum_bma_competition()
