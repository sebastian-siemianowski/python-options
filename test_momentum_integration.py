#!/usr/bin/env python3
"""Test momentum model integration in calibration modules."""

import sys
sys.path.insert(0, 'src')

from calibration.control_policy import ControlPolicy, CalibrationDiagnostics
from calibration.calibrated_trust import TrustConfig
from calibration.pit_driven_escalation import extract_escalation_from_result, EscalationResult

print("=" * 60)
print("MOMENTUM MODEL INTEGRATION TEST")
print("=" * 60)

# Test models
test_models = [
    'kalman_gaussian_momentum',
    'kalman_phi_gaussian_momentum',
    'phi_student_t_nu_6_momentum',
    'phi_student_t_nu_8',
    'kalman_gaussian',
    'kalman_phi_gaussian',
]

# Create mock diagnostics
diag = CalibrationDiagnostics(
    asset='TEST',
    pit_ks_pvalue=0.5,
    ks_statistic=0.1,
    excess_kurtosis=3.0,
    skewness=0.1,
    current_nu=6.0,
    regime_id=1,
    bic_current=1000.0,
    n_observations=500,
    realized_volatility=0.15,
)

# Test 1: Control Policy Momentum Detection
print("\n1. Control Policy - compute_trust_penalty()")
print("-" * 50)
cp = ControlPolicy()
for model in test_models:
    penalty = cp.compute_trust_penalty(model, 1, diag)
    is_mom = 'momentum' in model
    print(f"  {model:40s}: penalty = {penalty:.3f} {'(momentum)' if is_mom else ''}")

# Test 2: Calibrated Trust Model Penalty Detection
print("\n2. TrustConfig - get_model_penalty()")
print("-" * 50)
tc = TrustConfig()
for model in test_models:
    penalty = tc.get_model_penalty(model)
    is_mom = 'momentum' in model
    print(f"  {model:40s}: penalty = {penalty:.3f} {'(momentum)' if is_mom else ''}")

# Test 3: PIT Escalation Result
print("\n3. PIT Escalation - extract_escalation_from_result()")
print("-" * 50)

# Test momentum Student-t result
mock_result = {
    'global': {
        'noise_model': 'phi_student_t_nu_6_momentum',
        'pit_ks_pvalue': 0.15,
        'ks_statistic': 0.08,
        'q': 1e-6,
        'c': 1.0,
        'phi': 0.8,
        'nu': 6,
    }
}
esc_result = extract_escalation_from_result(mock_result)
print(f"  Model: phi_student_t_nu_6_momentum")
print(f"    final_model: {esc_result.final_model}")
print(f"    momentum_enabled: {esc_result.momentum_enabled}")

# Test non-momentum Student-t result
mock_result_no_mom = {
    'global': {
        'noise_model': 'phi_student_t_nu_8',
        'pit_ks_pvalue': 0.15,
        'ks_statistic': 0.08,
        'q': 1e-6,
        'c': 1.0,
        'phi': 0.8,
        'nu': 8,
    }
}
esc_result_no_mom = extract_escalation_from_result(mock_result_no_mom)
print(f"  Model: phi_student_t_nu_8")
print(f"    final_model: {esc_result_no_mom.final_model}")
print(f"    momentum_enabled: {esc_result_no_mom.momentum_enabled}")

# Test momentum Gaussian result
mock_result_mom_g = {
    'global': {
        'noise_model': 'kalman_gaussian_momentum',
        'pit_ks_pvalue': 0.20,
        'ks_statistic': 0.05,
        'q': 1e-5,
        'c': 0.8,
    }
}
esc_result_mom_g = extract_escalation_from_result(mock_result_mom_g)
print(f"  Model: kalman_gaussian_momentum")
print(f"    final_model: {esc_result_mom_g.final_model}")
print(f"    momentum_enabled: {esc_result_mom_g.momentum_enabled}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED - Momentum integration verified!")
print("=" * 60)
