"""Test PIT conditional penalty implementation."""
import sys
sys.path.insert(0, 'src')

import numpy as np
from tuning.diagnostics import (
    compute_regime_aware_model_weights, 
    PIT_CATASTROPHIC_THRESHOLD, 
    PIT_CATASTROPHIC_PENALTY,
    REGIME_SCORING_WEIGHTS
)

# Test data
bic_values = {
    "model_A": -1000.0,  # Best BIC
    "model_B": -950.0,   # Moderate
    "model_C": -800.0,   # Worst
}

hyvarinen_scores = {
    "model_A": 500.0,   # Moderate Hyv
    "model_B": 600.0,   # Best Hyv (higher = better)
    "model_C": 400.0,   # Worst Hyv
}

crps_values = {
    "model_A": 0.02,   # Best CRPS
    "model_B": 0.03,   # Moderate
    "model_C": 0.05,   # Worst
}

# Scenario 1: All models have good PIT (no penalty)
pit_good = {
    "model_A": 0.15,  # Good
    "model_B": 0.08,  # Marginal but above threshold
    "model_C": 0.25,  # Good
}

weights1, meta1 = compute_regime_aware_model_weights(
    bic_values, hyvarinen_scores, crps_values, pit_good, regime=0
)

print("=== Scenario 1: All models have good PIT (p > 0.01) ===")
print(f"REGIME_SCORING_WEIGHTS (regime=0): {REGIME_SCORING_WEIGHTS[0]}")
print(f"Weights used: {meta1['weights_used']}")
print(f"Model weights: {weights1}")
print(f"PIT penalties applied: {meta1['pit_penalty_applied']}")
assert all(p == 0.0 for p in meta1['pit_penalty_applied'].values()), "No penalty should be applied"
print("✓ No penalty applied correctly\n")

# Scenario 2: One model has catastrophic PIT
pit_bad = {
    "model_A": 0.001,  # CATASTROPHIC - below 0.01
    "model_B": 0.08,   # Marginal but OK
    "model_C": 0.25,   # Good
}

weights2, meta2 = compute_regime_aware_model_weights(
    bic_values, hyvarinen_scores, crps_values, pit_bad, regime=0
)

print("=== Scenario 2: model_A has catastrophic PIT (p=0.001 < 0.01) ===")
print(f"Model weights: {weights2}")
print(f"PIT penalties applied: {meta2['pit_penalty_applied']}")
assert meta2['pit_penalty_applied']['model_A'] == PIT_CATASTROPHIC_PENALTY, "model_A should have penalty"
assert meta2['pit_penalty_applied']['model_B'] == 0.0, "model_B should have no penalty"
assert meta2['pit_penalty_applied']['model_C'] == 0.0, "model_C should have no penalty"
print(f"✓ Only model_A penalized with {PIT_CATASTROPHIC_PENALTY}\n")

# Verify model_A gets lower weight after penalty
print(f"model_A weight with good PIT: {weights1['model_A']:.4f}")
print(f"model_A weight with bad PIT:  {weights2['model_A']:.4f}")
assert weights2['model_A'] < weights1['model_A'], "model_A should have lower weight after penalty"
print("✓ model_A weight correctly reduced\n")

# Scenario 3: Edge case - PIT exactly at threshold
pit_edge = {
    "model_A": 0.01,   # Exactly at threshold - NO penalty (< threshold, not <=)
    "model_B": 0.009,  # Just below - PENALTY
    "model_C": 0.011,  # Just above - NO penalty
}

weights3, meta3 = compute_regime_aware_model_weights(
    bic_values, hyvarinen_scores, crps_values, pit_edge, regime=0
)

print("=== Scenario 3: Edge cases at threshold (0.01) ===")
print(f"PIT values: model_A=0.01, model_B=0.009, model_C=0.011")
print(f"PIT penalties: {meta3['pit_penalty_applied']}")
# model_A: 0.01 is NOT < 0.01, so no penalty
# model_B: 0.009 < 0.01, so penalty
# model_C: 0.011 is NOT < 0.01, so no penalty
assert meta3['pit_penalty_applied']['model_A'] == 0.0, "0.01 should NOT get penalty (not strictly less than)"
assert meta3['pit_penalty_applied']['model_B'] == PIT_CATASTROPHIC_PENALTY, "0.009 should get penalty"
assert meta3['pit_penalty_applied']['model_C'] == 0.0, "0.011 should NOT get penalty"
print("✓ Edge cases handled correctly\n")

print("=" * 50)
print("ALL TESTS PASSED ✓")
print("=" * 50)
print(f"\nSummary:")
print(f"  PIT_CATASTROPHIC_THRESHOLD = {PIT_CATASTROPHIC_THRESHOLD}")
print(f"  PIT_CATASTROPHIC_PENALTY = {PIT_CATASTROPHIC_PENALTY}")
print(f"  REGIME_SCORING_WEIGHTS contains 3-tuples (BIC, Hyv, CRPS)")
print(f"  PIT is NOT a weight - only a conditional penalty when p < {PIT_CATASTROPHIC_THRESHOLD}")
