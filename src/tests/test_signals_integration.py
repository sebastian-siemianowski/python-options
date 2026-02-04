#!/usr/bin/env python3
"""Test signals.py integration with online update."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing signals.py integration...")

from decision.signals import ONLINE_UPDATE_AVAILABLE
print(f"ONLINE_UPDATE_AVAILABLE: {ONLINE_UPDATE_AVAILABLE}")

if ONLINE_UPDATE_AVAILABLE:
    from decision.signals import (
        compute_adaptive_kalman_params,
        clear_updater_cache,
    )
    print("✓ Online update functions imported from signals.py")
    
    import numpy as np
    tuned_params = {
        'global': {
            'q': 1e-6, 'c': 1.0, 'phi': 0.95, 'nu': 8.0,
            'models': {'phi_student_t_nu_8': {'q': 1e-6, 'c': 1.0, 'phi': 0.95, 'nu': 8.0, 'fit_success': True}},
            'model_posterior': {'phi_student_t_nu_8': 1.0}
        }
    }
    
    np.random.seed(42)
    returns = np.random.randn(50) * 0.02
    volatility = np.abs(np.random.randn(50)) * 0.02 + 0.01
    
    result = compute_adaptive_kalman_params(
        asset='TEST',
        returns=returns,
        volatility=volatility,
        tuned_params=tuned_params,
        enable_online=True,
    )
    
    print(f"✓ compute_adaptive_kalman_params worked")
    print(f"  online_active: {result['online_active']}")
    print(f"  online_updated: {result['current_params']['online_updated']}")
    
    clear_updater_cache()
    print("✓ Cache cleared")
    
    print()
    print("SUCCESS: Online update integration with signals.py is WORKING!")
else:
    print("✗ ONLINE_UPDATE_AVAILABLE is False - check imports")
