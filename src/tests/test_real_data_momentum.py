#!/usr/bin/env python3
"""Test momentum model creation with real data."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')
import numpy as np

from ingestion.data_utils import fetch_px
from tuning.tune import fit_all_models_for_regime, MOMENTUM_AUGMENTATION_ENABLED, MOMENTUM_AUGMENTATION_AVAILABLE
from calibration.model_selection import compute_bic_model_weights, normalize_weights

print(f'MOMENTUM_AUGMENTATION_ENABLED: {MOMENTUM_AUGMENTATION_ENABLED}')
print(f'MOMENTUM_AUGMENTATION_AVAILABLE: {MOMENTUM_AUGMENTATION_AVAILABLE}')

# Load cached price data for AAPL
print('\nLoading AAPL price data...')
import pandas as pd
price_file = 'src/data/prices/AAPL.csv'
try:
    df = pd.read_csv(price_file, index_col=0, parse_dates=True)
    if 'Close' in df.columns:
        px = df['Close']
    elif 'Adj Close' in df.columns:
        px = df['Adj Close']
    else:
        px = df.iloc[:, 0]
    print(f'Loaded {len(px)} price points')
except Exception as e:
    px = None
    print(f'Error: {e}')
if px is not None and len(px) > 100:
    returns = px.pct_change().dropna().values
    vol = np.abs(returns) * 1.5 + 0.01
    
    print(f'Data points: {len(returns)}')
    print('Fitting models...')
    models = fit_all_models_for_regime(returns, vol)
    
    print(f'\nTotal models: {len(models)}')
    mom_models = [m for m in models if '_momentum' in m]
    base_models = [m for m in models if '_momentum' not in m]
    print(f'Base models: {len(base_models)}')
    print(f'Momentum models: {len(mom_models)}')
    
    # Show BIC comparison
    print('\n' + '='*60)
    print('BIC Comparison (base vs momentum):')
    print('='*60)
    for base in ['kalman_gaussian', 'kalman_phi_gaussian', 'phi_student_t_nu_8']:
        mom = base + '_momentum'
        if base in models and mom in models:
            base_info = models[base]
            mom_info = models[mom]
            base_bic = base_info.get('bic', float('inf'))
            mom_bic = mom_info.get('bic', float('inf'))
            base_ll = base_info.get('log_likelihood', float('nan'))
            mom_ll = mom_info.get('log_likelihood', float('nan'))
            winner = 'BASE' if base_bic < mom_bic else 'MOMENTUM'
            print(f'\n  {base}:')
            print(f'    Base:     LL={base_ll:.1f}, BIC={base_bic:.1f}')
            print(f'    Momentum: LL={mom_ll:.1f}, BIC={mom_bic:.1f}')
            print(f'    Winner:   {winner}')
    
    # Compute BMA weights
    print('\n' + '='*60)
    print('BMA Weights:')
    print('='*60)
    bic_values = {m: models[m].get('bic', float('inf')) for m in models if models[m].get('fit_success', False)}
    weights = compute_bic_model_weights(bic_values)
    weights = normalize_weights(weights)
    
    for m, w in sorted(weights.items(), key=lambda x: -x[1])[:10]:
        is_mom = '_momentum' in m
        print(f'  {m}: {w:.4f} {"← MOMENTUM" if is_mom else ""}')
    
    mom_total = sum(w for m, w in weights.items() if '_momentum' in m)
    print(f'\nTotal momentum weight: {mom_total:.4f}')
else:
    print('Failed to fetch data')
