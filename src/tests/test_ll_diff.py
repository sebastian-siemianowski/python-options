#!/usr/bin/env python3
"""Quick test to verify momentum models have different log-likelihoods."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings('ignore')
import numpy as np

from tuning.tune import fit_all_models_for_regime

np.random.seed(42)
returns = np.random.randn(500) * 0.02
vol = np.abs(returns) * 1.5 + 0.01

models = fit_all_models_for_regime(returns, vol)

print("Log-Likelihood Comparison:")
print("=" * 60)
for base_name in ['kalman_gaussian', 'kalman_phi_gaussian', 'phi_student_t_nu_20']:
    mom_name = base_name + '_momentum'
    base_ll = models[base_name]['log_likelihood']
    mom_ll = models[mom_name]['log_likelihood']
    same = abs(base_ll - mom_ll) < 0.001
    print(f"{base_name}:")
    print(f"  Base:     LL = {base_ll:.4f}")
    print(f"  Momentum: LL = {mom_ll:.4f}")
    print(f"  Same LL:  {same}")
    print()
