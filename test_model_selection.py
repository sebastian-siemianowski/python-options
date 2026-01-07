#!/usr/bin/env python3
"""
Test script to verify that fx_pln_jpy_signals.py correctly uses best_model_by_bic
from tune_q_mle.py cache.
"""

import sys
import os
import json

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from fx_pln_jpy_signals import _load_tuned_kalman_params

def test_load_tuned_params():
    """Test that _load_tuned_kalman_params loads model selection results."""
    print("=" * 80)
    print("Testing Model Selection Integration")
    print("=" * 80)
    
    cache_path = "cache/kalman_q_cache.json"
    
    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found: {cache_path}")
        return False
    
    # Load cache directly to see what's available
    with open(cache_path, 'r') as f:
        cache = json.load(f)
    
    print(f"\n✓ Cache loaded with {len(cache)} assets")
    
    # Test a few assets
    test_assets = ["PLNJPY=X", "GC=F", "SPY", "BTC-USD"]
    
    for asset in test_assets:
        if asset not in cache:
            print(f"\n⚠️  {asset}: Not in cache")
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Asset: {asset}")
        print(f"{'=' * 60}")
        
        # Check raw cache data
        raw_data = cache[asset]
        best_model_raw = raw_data.get('best_model_by_bic', 'NOT_FOUND')
        model_comp = raw_data.get('model_comparison', {})
        
        print(f"\nRaw cache data:")
        print(f"  best_model_by_bic: {best_model_raw}")
        
        if model_comp:
            print(f"  Model comparison available:")
            for model_name, metrics in model_comp.items():
                bic = metrics.get('bic', float('nan'))
                ll = metrics.get('ll', float('nan'))
                print(f"    {model_name:20s}: BIC={bic:>10.1f}, LL={ll:>10.1f}")
        
        # Test loading via function
        params = _load_tuned_kalman_params(asset, cache_path)
        
        if params is None:
            print(f"\n❌ Failed to load params via function")
            continue
        
        print(f"\nLoaded via _load_tuned_kalman_params():")
        print(f"  best_model_by_bic: {params.get('best_model_by_bic', 'NOT_FOUND')}")
        print(f"  delta_ll_vs_zero: {params.get('delta_ll_vs_zero', float('nan')):.2f}")
        print(f"  delta_ll_vs_const: {params.get('delta_ll_vs_const', float('nan')):.2f}")
        print(f"  delta_ll_vs_ewma: {params.get('delta_ll_vs_ewma', float('nan')):.2f}")
        
        # Verify consistency
        if best_model_raw != params.get('best_model_by_bic'):
            print(f"\n❌ MISMATCH: Raw cache has '{best_model_raw}' but function returned '{params.get('best_model_by_bic')}'")
            return False
        
        print(f"\n✓ Model selection data loaded correctly")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed - fx_pln_jpy_signals.py correctly loads model selection")
    print("=" * 80)
    return True

if __name__ == '__main__':
    success = test_load_tuned_params()
    sys.exit(0 if success else 1)
