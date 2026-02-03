#!/usr/bin/env python3
"""Test script for PIT penalty module."""
import sys
sys.path.insert(0, 'src')

from calibration.pit_penalty import (
    compute_pit_violation_severity,
    compute_pit_penalty,
    compute_model_pit_penalty,
    apply_pit_penalties_to_weights,
    PIT_EXIT_THRESHOLD,
    PIT_CRITICAL_THRESHOLDS,
)

print("=" * 60)
print("PIT PENALTY MODULE TEST")
print("=" * 60)

# Test 1: No violation (p > threshold)
v = compute_pit_violation_severity(0.5, 0.01)
print(f'\nTest 1 - No violation: V = {v} (expected 0.0)')
assert v == 0.0, "Test 1 failed"

# Test 2: Violation (p < threshold)
v = compute_pit_violation_severity(0.001, 0.01)
print(f'Test 2 - Violation: V = {v:.3f} (expected 0.9)')
assert abs(v - 0.9) < 0.001, "Test 2 failed"

# Test 3: Penalty for violation
p = compute_pit_penalty(0.9, 5.0)
print(f'Test 3 - Penalty: P = {p:.4f} (expected ~0.011)')
assert p < 0.02, "Test 3 failed"

# Test 4: Full model penalty computation
result = compute_model_pit_penalty(
    model_name='phi_student_t_nu_8',
    pit_pvalue=0.001,
    regime=0,  # LOW_VOL_TREND
    n_samples=100,
)
print(f'\nTest 4 - Model penalty:')
print(f'  is_violated: {result.is_violated}')
print(f'  violation_severity: {result.violation_severity:.3f}')
print(f'  effective_penalty: {result.effective_penalty:.4f}')
print(f'  triggers_exit: {result.triggers_exit}')
assert result.is_violated, "Test 4a failed"
assert result.triggers_exit, "Test 4b failed - should trigger EXIT"

# Test 5: No EXIT for mild violation
result_mild = compute_model_pit_penalty(
    model_name='phi_student_t_nu_8',
    pit_pvalue=0.02,  # Above some thresholds
    regime=4,  # CRISIS (threshold = 0.05)
    n_samples=100,
)
print(f'\nTest 5 - Mild violation (crisis regime):')
print(f'  is_violated: {result_mild.is_violated}')
print(f'  effective_penalty: {result_mild.effective_penalty:.4f}')
print(f'  triggers_exit: {result_mild.triggers_exit}')
assert not result_mild.is_violated, "Test 5 failed - should not be violated in crisis"

# Test 6: Apply penalties to weights
print(f'\nTest 6 - Apply penalties to weights:')
raw_weights = {'model_a': 0.4, 'model_b': 0.4, 'model_c': 0.2}
pit_pvalues = {'model_a': 0.5, 'model_b': 0.001, 'model_c': 0.1}
adjusted, report = apply_pit_penalties_to_weights(
    raw_weights, pit_pvalues, regime=0, n_samples=100
)
print(f'  Raw weights: {raw_weights}')
print(f'  PIT p-values: {pit_pvalues}')
print(f'  Adjusted weights: {dict((k, round(v, 3)) for k, v in adjusted.items())}')
print(f'  Selection diverged: {report.selection_diverged}')
print(f'  N violated: {report.n_violated}')

# model_b should be heavily penalized
assert adjusted['model_b'] < raw_weights['model_b'], "Test 6 failed - penalty not applied"

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print(f"\nEXIT threshold: {PIT_EXIT_THRESHOLD}")
print(f"Critical thresholds by regime: {dict(PIT_CRITICAL_THRESHOLDS)}")
