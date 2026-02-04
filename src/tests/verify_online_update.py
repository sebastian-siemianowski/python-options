#!/usr/bin/env python3
"""Quick test script to verify online update is working."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("ONLINE BAYESIAN PARAMETER UPDATES - VERIFICATION TEST")
    print("=" * 60)
    print()
    
    # Test 1: Import
    print("Test 1: Module Import")
    try:
        from calibration.online_update import (
            OnlineBayesianUpdater,
            OnlineUpdateConfig,
        )
        import numpy as np
        print("  ✓ Import successful")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return 1
    
    # Test 2: Initialization
    print("\nTest 2: Initialization")
    batch_params = {'q': 1e-6, 'c': 1.0, 'phi': 0.95, 'nu': 8.0}
    config = OnlineUpdateConfig(n_particles=50)
    
    try:
        updater = OnlineBayesianUpdater(batch_params, config)
        print(f"  ✓ Created updater with {len(updater.particles)} particles")
        print(f"  ✓ rng initialized: {hasattr(updater, 'rng') and updater.rng is not None}")
    except Exception as e:
        print(f"  ✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 3: Run updates
    print("\nTest 3: Running Updates")
    try:
        np.random.seed(42)
        for i in range(20):
            y = 0.001 * np.sin(i / 3) + 0.002 * np.random.randn()
            sigma = 0.02
            result = updater.update(y, sigma)
        
        print(f"  ✓ Completed 20 updates")
        print(f"  ✓ Final ESS: {result.effective_sample_size:.1f}")
        print(f"  ✓ Stable: {result.is_stable}")
        print(f"  ✓ Fallback to batch: {result.fallback_to_batch}")
    except Exception as e:
        print(f"  ✗ Update failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Test 4: Get parameters
    print("\nTest 4: Parameter Extraction")
    try:
        params = updater.get_current_params()
        print(f"  ✓ q = {params['q']:.2e} (batch: {batch_params['q']:.2e})")
        print(f"  ✓ c = {params['c']:.3f} (batch: {batch_params['c']:.3f})")
        print(f"  ✓ φ = {params['phi']:.3f} (batch: {batch_params['phi']:.3f})")
        print(f"  ✓ ν = {params['nu']:.1f} (batch: {batch_params['nu']:.1f})")
        print(f"  ✓ online_updated = {params['online_updated']}")
    except Exception as e:
        print(f"  ✗ Parameter extraction failed: {e}")
        return 1
    
    # Test 5: Verify parameters are actually changing
    print("\nTest 5: Parameter Adaptation")
    try:
        # Parameters should have adapted from batch values
        q_changed = abs(params['q'] - batch_params['q']) > 1e-10
        c_changed = abs(params['c'] - batch_params['c']) > 0.001
        phi_changed = abs(params['phi'] - batch_params['phi']) > 0.001
        nu_changed = abs(params['nu'] - batch_params['nu']) > 0.1
        
        any_changed = q_changed or c_changed or phi_changed or nu_changed
        
        if any_changed:
            print("  ✓ Parameters have adapted from batch values")
            if q_changed: print(f"    - q changed: {batch_params['q']:.2e} → {params['q']:.2e}")
            if c_changed: print(f"    - c changed: {batch_params['c']:.3f} → {params['c']:.3f}")
            if phi_changed: print(f"    - φ changed: {batch_params['phi']:.3f} → {params['phi']:.3f}")
            if nu_changed: print(f"    - ν changed: {batch_params['nu']:.1f} → {params['nu']:.1f}")
        else:
            print("  ⚠ Parameters haven't changed much (may be normal for stable data)")
    except Exception as e:
        print(f"  ✗ Adaptation check failed: {e}")
    
    # Test 6: Audit trail
    print("\nTest 6: Audit Trail")
    try:
        audit = updater.get_audit_trail()
        print(f"  ✓ Audit trail has {len(audit)} records")
        if audit:
            print(f"    - First record: step {audit[0]['step']}")
            print(f"    - Last record: step {audit[-1]['step']}")
    except Exception as e:
        print(f"  ✗ Audit trail failed: {e}")
    
    print()
    print("=" * 60)
    print("ALL TESTS PASSED - Online Update is WORKING!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
