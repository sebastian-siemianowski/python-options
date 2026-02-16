#!/usr/bin/env python3
"""Quick test for HAR volatility and enhanced mixture weights."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np

def test_har_volatility():
    """Test HAR volatility computation."""
    from calibration.realized_volatility import (
        compute_har_volatility, 
        compute_hybrid_volatility_har,
        HAR_WEIGHT_DAILY,
        HAR_WEIGHT_WEEKLY,
        HAR_WEIGHT_MONTHLY,
    )
    
    np.random.seed(42)
    n = 252
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + np.random.randn(n) * 0.005)
    high = np.maximum(close, open_) * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = np.minimum(close, open_) * (1 - np.abs(np.random.randn(n)) * 0.01)
    
    # Test HAR volatility with GK
    har_vol, har_diag = compute_har_volatility(
        open_=open_, high=high, low=low, close=close,
        use_gk=True
    )
    print(f'HAR-GK vol mean: {np.nanmean(har_vol):.6f}')
    print(f'HAR estimator: {har_diag["estimator"]}')
    print(f'HAR weights: {har_diag["weights"]}')
    
    # Test hybrid HAR
    vol, estimator = compute_hybrid_volatility_har(
        open_=open_, high=high, low=low, close=close,
        use_har=True
    )
    print(f'Hybrid HAR estimator: {estimator}')
    print('✅ HAR volatility test PASSED')
    return True


def test_market_conditioning():
    """Test market conditioning layer."""
    from calibration.market_conditioning import (
        compute_vix_nu_adjustment,
        VIX_KAPPA_DEFAULT,
        NU_MIN_FLOOR,
    )
    
    # Test VIX conditioning
    result = compute_vix_nu_adjustment(nu_base=8.0, vix_value=25.0)
    print(f'VIX conditioning: nu_base=8.0, VIX=25 -> nu_adjusted={result.nu_adjusted:.2f}')
    
    # Test extreme VIX
    result_extreme = compute_vix_nu_adjustment(nu_base=12.0, vix_value=50.0)
    print(f'Extreme VIX: nu_base=12.0, VIX=50 -> nu_adjusted={result_extreme.nu_adjusted:.2f}')
    
    # Verify floor is respected
    assert result_extreme.nu_adjusted >= NU_MIN_FLOOR, f"Nu below floor: {result_extreme.nu_adjusted}"
    print('✅ Market conditioning test PASSED')
    return True


def test_enhanced_mixture_filter():
    """Test enhanced mixture weight dynamics."""
    from models.phi_student_t import PhiStudentTDriftModel
    
    np.random.seed(42)
    n = 200
    returns = np.random.randn(n) * 0.02
    vol = np.abs(returns) + 0.01
    vol = np.convolve(vol, np.ones(5)/5, mode='same')  # Smooth
    
    # Test enhanced mixture filter
    mu, P, ll = PhiStudentTDriftModel.filter_phi_mixture_enhanced(
        returns=returns,
        vol=vol,
        q=1e-6,
        c=1.0,
        phi=0.1,
        nu_calm=12.0,
        nu_stress=4.0,
        w_base=0.5,
        a_shock=1.0,
        b_vol_accel=0.5,
        c_momentum=0.3,
    )
    
    print(f'Enhanced mixture filter: LL={ll:.2f}, mu_range=[{mu.min():.4f}, {mu.max():.4f}]')
    assert np.isfinite(ll), "Log-likelihood not finite"
    assert len(mu) == n, "Output length mismatch"
    print('✅ Enhanced mixture filter test PASSED')
    return True


def test_phi_student_t_constants():
    """Verify new constants are defined."""
    from models.phi_student_t import (
        MIXTURE_WEIGHT_A_SHOCK,
        MIXTURE_WEIGHT_B_VOL_ACCEL,
        MIXTURE_WEIGHT_C_MOMENTUM,
    )
    
    print(f'MIXTURE_WEIGHT_A_SHOCK: {MIXTURE_WEIGHT_A_SHOCK}')
    print(f'MIXTURE_WEIGHT_B_VOL_ACCEL: {MIXTURE_WEIGHT_B_VOL_ACCEL}')
    print(f'MIXTURE_WEIGHT_C_MOMENTUM: {MIXTURE_WEIGHT_C_MOMENTUM}')
    print('✅ Constants test PASSED')
    return True


def test_models_init_exports():
    """Verify models/__init__.py exports new constants."""
    from models import (
        MIXTURE_WEIGHT_A_SHOCK,
        MIXTURE_WEIGHT_B_VOL_ACCEL,
        MIXTURE_WEIGHT_C_MOMENTUM,
    )
    
    assert MIXTURE_WEIGHT_A_SHOCK == 1.0
    assert MIXTURE_WEIGHT_B_VOL_ACCEL == 0.5
    assert MIXTURE_WEIGHT_C_MOMENTUM == 0.3
    print('✅ Models __init__ exports test PASSED')
    return True


def test_tune_imports():
    """Verify tune.py imports work correctly."""
    # This will fail fast if any imports are broken
    from tuning.tune import (
        ENHANCED_MIXTURE_ENABLED,
        MARKET_CONDITIONING_AVAILABLE,
        HAR_VOLATILITY_AVAILABLE,
    )
    print(f'ENHANCED_MIXTURE_ENABLED: {ENHANCED_MIXTURE_ENABLED}')
    print(f'MARKET_CONDITIONING_AVAILABLE: {MARKET_CONDITIONING_AVAILABLE}')
    print(f'HAR_VOLATILITY_AVAILABLE: {HAR_VOLATILITY_AVAILABLE}')
    print('✅ Tune.py imports test PASSED')
    return True


def test_signals_imports():
    """Verify signals.py imports work correctly."""
    from decision.signals import (
        ENHANCED_MIXTURE_ENABLED,
        ENHANCED_MIXTURE_AVAILABLE,
    )
    print(f'signals.py ENHANCED_MIXTURE_ENABLED: {ENHANCED_MIXTURE_ENABLED}')
    print(f'signals.py ENHANCED_MIXTURE_AVAILABLE: {ENHANCED_MIXTURE_AVAILABLE}')
    print('✅ Signals.py imports test PASSED')
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Student-t Model Enhancements (February 2026)")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        ("HAR Volatility", test_har_volatility),
        ("Market Conditioning", test_market_conditioning),
        ("Enhanced Mixture Filter", test_enhanced_mixture_filter),
        ("Constants", test_phi_student_t_constants),
        ("Models Init Exports", test_models_init_exports),
        ("Tune.py Imports", test_tune_imports),
        ("Signals.py Imports", test_signals_imports),
    ]
    
    for name, test_fn in tests:
        print(f"\n{len([t for t in tests if t[0] <= name])}. Testing {name}...")
        try:
            test_fn()
        except Exception as e:
            print(f'❌ {name} test FAILED: {e}')
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
