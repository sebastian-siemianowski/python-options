#!/usr/bin/env python3
"""
test_signal_labeling.py

Unit tests for signal labeling helper functions.
Tests compute_dynamic_thresholds and apply_confirmation_logic for Level-7 code quality.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals import compute_dynamic_thresholds, apply_confirmation_logic


def test_compute_dynamic_thresholds_symmetric():
    """Test threshold computation with zero skew (symmetric case)."""
    print("Test 1: Symmetric thresholds (zero skew)")
    
    result = compute_dynamic_thresholds(
        skew=0.0,
        regime_meta={"method": "threshold_fallback", "vol_regime": 1.0},
        sig_H=0.10,
        med_vol_last=0.10,
        H=21
    )
    
    print(f"  Buy threshold: {result['buy_thr']:.4f}")
    print(f"  Sell threshold: {result['sell_thr']:.4f}")
    print(f"  Uncertainty: {result['uncertainty']:.4f}")
    
    # Assertions
    assert 0.55 <= result['buy_thr'] <= 0.70, "Buy threshold out of range"
    assert 0.30 <= result['sell_thr'] <= 0.45, "Sell threshold out of range"
    assert result['buy_thr'] > result['sell_thr'], "Buy threshold should be > sell threshold"
    assert result['buy_thr'] - result['sell_thr'] >= 0.12, "Minimum separation not met"
    
    print("  ✓ PASS\n")
    return result


def test_compute_dynamic_thresholds_negative_skew():
    """Test threshold computation with negative skew (crash risk)."""
    print("Test 2: Negative skew (crash risk)")
    
    result = compute_dynamic_thresholds(
        skew=-1.0,  # Strong negative skew
        regime_meta={"method": "threshold_fallback", "vol_regime": 1.0},
        sig_H=0.10,
        med_vol_last=0.10,
        H=21
    )
    
    print(f"  Buy threshold: {result['buy_thr']:.4f}")
    print(f"  Sell threshold: {result['sell_thr']:.4f}")
    print(f"  Skew adjustment: {result['skew_adjustment']:.4f}")
    
    # With negative skew, thresholds should shift upward (more conservative)
    symmetric_result = test_compute_dynamic_thresholds_symmetric()
    
    assert result['buy_thr'] >= symmetric_result['buy_thr'], "Negative skew should raise buy threshold"
    assert result['sell_thr'] >= symmetric_result['sell_thr'], "Negative skew should raise sell threshold"
    
    print("  ✓ PASS\n")
    return result


def test_compute_dynamic_thresholds_positive_skew():
    """Test threshold computation with positive skew (rally potential)."""
    print("Test 3: Positive skew (rally potential)")
    
    result = compute_dynamic_thresholds(
        skew=1.0,  # Strong positive skew
        regime_meta={"method": "threshold_fallback", "vol_regime": 1.0},
        sig_H=0.10,
        med_vol_last=0.10,
        H=21
    )
    
    print(f"  Buy threshold: {result['buy_thr']:.4f}")
    print(f"  Sell threshold: {result['sell_thr']:.4f}")
    print(f"  Skew adjustment: {result['skew_adjustment']:.4f}")
    
    # With positive skew, thresholds should shift downward (more aggressive)
    symmetric_result = test_compute_dynamic_thresholds_symmetric()
    
    assert result['buy_thr'] <= symmetric_result['buy_thr'], "Positive skew should lower buy threshold"
    assert result['sell_thr'] <= symmetric_result['sell_thr'], "Positive skew should lower sell threshold"
    
    print("  ✓ PASS\n")
    return result


def test_compute_dynamic_thresholds_high_uncertainty():
    """Test threshold computation with high uncertainty (regime entropy)."""
    print("Test 4: High uncertainty (widened thresholds)")
    
    # High entropy regime (max uncertainty)
    import math
    result_high = compute_dynamic_thresholds(
        skew=0.0,
        regime_meta={
            "method": "hmm_posterior",
            "probabilities": {"calm": 0.33, "trending": 0.33, "crisis": 0.34}  # Max entropy
        },
        sig_H=0.20,  # High forecast vol
        med_vol_last=0.10,  # vs low historical vol
        H=21
    )
    
    # Low uncertainty
    result_low = compute_dynamic_thresholds(
        skew=0.0,
        regime_meta={
            "method": "hmm_posterior",
            "probabilities": {"calm": 0.95, "trending": 0.03, "crisis": 0.02}  # Low entropy
        },
        sig_H=0.10,
        med_vol_last=0.10,
        H=21
    )
    
    print(f"  High uncertainty - Buy: {result_high['buy_thr']:.4f}, Sell: {result_high['sell_thr']:.4f}")
    print(f"  Low uncertainty - Buy: {result_low['buy_thr']:.4f}, Sell: {result_low['sell_thr']:.4f}")
    print(f"  High U: {result_high['uncertainty']:.4f}, Low U: {result_low['uncertainty']:.4f}")
    
    # High uncertainty should widen thresholds
    assert result_high['buy_thr'] >= result_low['buy_thr'], "High uncertainty should raise buy threshold"
    assert result_high['sell_thr'] <= result_low['sell_thr'], "High uncertainty should lower sell threshold"
    assert result_high['uncertainty'] > result_low['uncertainty'], "High entropy should produce higher uncertainty"
    
    print("  ✓ PASS\n")
    return result_high, result_low


def test_apply_confirmation_logic_buy():
    """Test confirmation logic for BUY signal."""
    print("Test 5: Confirmation logic - BUY signal")
    
    # Both days above buy threshold with hysteresis
    label = apply_confirmation_logic(
        p_smoothed_now=0.62,
        p_smoothed_prev=0.61,
        p_raw=0.63,
        pos_strength=0.35,
        buy_thr=0.58,
        sell_thr=0.42,
        edge=0.50,
        edge_floor=0.10
    )
    
    print(f"  Label: {label}")
    assert label == "BUY", f"Expected BUY, got {label}"
    
    print("  ✓ PASS\n")
    return label


def test_apply_confirmation_logic_strong_buy():
    """Test confirmation logic for STRONG BUY signal."""
    print("Test 6: Confirmation logic - STRONG BUY signal")
    
    # High conviction (p_raw > 0.66) and strong position (>0.30)
    label = apply_confirmation_logic(
        p_smoothed_now=0.68,
        p_smoothed_prev=0.67,
        p_raw=0.70,  # High raw probability
        pos_strength=0.40,  # Strong Kelly fraction
        buy_thr=0.58,
        sell_thr=0.42,
        edge=0.80,
        edge_floor=0.10
    )
    
    print(f"  Label: {label}")
    assert label == "STRONG BUY", f"Expected STRONG BUY, got {label}"
    
    print("  ✓ PASS\n")
    return label


def test_apply_confirmation_logic_hold():
    """Test confirmation logic for HOLD signal."""
    print("Test 7: Confirmation logic - HOLD signal")
    
    # Probability in neutral zone
    label = apply_confirmation_logic(
        p_smoothed_now=0.52,
        p_smoothed_prev=0.51,
        p_raw=0.53,
        pos_strength=0.20,
        buy_thr=0.58,
        sell_thr=0.42,
        edge=0.25,
        edge_floor=0.10
    )
    
    print(f"  Label: {label}")
    assert label == "HOLD", f"Expected HOLD, got {label}"
    
    print("  ✓ PASS\n")
    return label


def test_apply_confirmation_logic_edge_floor():
    """Test confirmation logic with edge floor constraint."""
    print("Test 8: Confirmation logic - Edge floor constraint")
    
    # Edge below floor should force HOLD regardless of probability
    label = apply_confirmation_logic(
        p_smoothed_now=0.65,  # Would normally be BUY
        p_smoothed_prev=0.64,
        p_raw=0.66,
        pos_strength=0.35,
        buy_thr=0.58,
        sell_thr=0.42,
        edge=0.05,  # Below edge_floor
        edge_floor=0.10
    )
    
    print(f"  Label: {label} (edge {0.05:.2f} < floor {0.10:.2f})")
    assert label == "HOLD", f"Expected HOLD due to edge floor, got {label}"
    
    print("  ✓ PASS\n")
    return label


def test_apply_confirmation_logic_no_confirmation():
    """Test confirmation logic without 2-day confirmation."""
    print("Test 9: Confirmation logic - No 2-day confirmation")
    
    # Only today above threshold, yesterday below
    label = apply_confirmation_logic(
        p_smoothed_now=0.62,  # Above buy threshold
        p_smoothed_prev=0.50,  # Below buy threshold
        p_raw=0.63,
        pos_strength=0.25,
        buy_thr=0.58,
        sell_thr=0.42,
        edge=0.40,
        edge_floor=0.10
    )
    
    print(f"  Label: {label} (no 2-day confirmation)")
    assert label == "HOLD", f"Expected HOLD without 2-day confirmation, got {label}"
    
    print("  ✓ PASS\n")
    return label


def test_apply_confirmation_logic_sell():
    """Test confirmation logic for SELL signal."""
    print("Test 10: Confirmation logic - SELL signal")
    
    # Both days below sell threshold with hysteresis
    label = apply_confirmation_logic(
        p_smoothed_now=0.38,
        p_smoothed_prev=0.37,
        p_raw=0.36,
        pos_strength=0.30,
        buy_thr=0.58,
        sell_thr=0.42,
        edge=-0.45,
        edge_floor=0.10
    )
    
    print(f"  Label: {label}")
    assert label == "SELL", f"Expected SELL, got {label}"
    
    print("  ✓ PASS\n")
    return label


def test_edge_cases():
    """Test edge cases: NaN, Inf, extreme values."""
    print("Test 11: Edge cases")
    
    # NaN skew should default to zero
    result = compute_dynamic_thresholds(
        skew=float('nan'),
        regime_meta={"method": "threshold_fallback", "vol_regime": 1.0},
        sig_H=0.10,
        med_vol_last=0.10,
        H=21
    )
    assert 0.55 <= result['buy_thr'] <= 0.70, "NaN skew should be handled"
    print("  ✓ NaN skew handled correctly")
    
    # Extreme skew should be clipped
    result = compute_dynamic_thresholds(
        skew=10.0,  # Extreme value
        regime_meta={"method": "threshold_fallback", "vol_regime": 1.0},
        sig_H=0.10,
        med_vol_last=0.10,
        H=21
    )
    assert 0.55 <= result['buy_thr'] <= 0.70, "Extreme skew should be clipped"
    print("  ✓ Extreme skew clipped correctly")
    
    # Zero volatility edge case
    result = compute_dynamic_thresholds(
        skew=0.0,
        regime_meta={"method": "threshold_fallback", "vol_regime": 1.0},
        sig_H=0.0,  # Zero volatility
        med_vol_last=0.10,
        H=21
    )
    assert 0.55 <= result['buy_thr'] <= 0.70, "Zero volatility should be handled"
    print("  ✓ Zero volatility handled correctly")
    
    print("  ✓ PASS\n")


def main():
    """Run all tests."""
    print("=" * 80)
    print("SIGNAL LABELING UNIT TESTS")
    print("Testing Level-7 modular helper functions")
    print("=" * 80 + "\n")
    
    try:
        test_compute_dynamic_thresholds_symmetric()
        test_compute_dynamic_thresholds_negative_skew()
        test_compute_dynamic_thresholds_positive_skew()
        test_compute_dynamic_thresholds_high_uncertainty()
        test_apply_confirmation_logic_buy()
        test_apply_confirmation_logic_strong_buy()
        test_apply_confirmation_logic_hold()
        test_apply_confirmation_logic_edge_floor()
        test_apply_confirmation_logic_no_confirmation()
        test_apply_confirmation_logic_sell()
        test_edge_cases()
        
        print("=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
