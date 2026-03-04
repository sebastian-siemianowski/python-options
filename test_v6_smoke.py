"""v6.0 Calibration Pipeline - Integration Smoke Test."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

# Test 1: Import all v6.0 kernels
from decision.signals_calibration_numba import (
    crps_student_t_nb, crps_student_t_mean_nb,
    emos_crps_student_t_objective_nb,
    beta_focal_nll_objective_nb,
    temperature_scaling_nll_nb,
    apply_isotonic_beta_blend_nb,
    brier_decomposition_nb,
    expanding_cv_fold_indices_nb,
)
print("1. All v6.0 Numba kernels imported OK")

# Test 2: Student-t CRPS
mu = np.array([1.0, 2.0], dtype=np.float64)
sig = np.array([1.0, 1.5], dtype=np.float64)
y = np.array([0.5, 3.0], dtype=np.float64)
nu = np.array([5.0, 5.0], dtype=np.float64)
crps = crps_student_t_nb(mu, sig, y, nu)
assert all(c > 0 for c in crps), f"CRPS must be positive: {crps}"
print(f"2. Student-t CRPS: {crps}")

crps_mean = crps_student_t_mean_nb(mu, sig, y, nu)
assert crps_mean > 0, f"Mean CRPS must be positive: {crps_mean}"
print(f"3. Student-t CRPS mean: {crps_mean:.4f}")

# Test 3: Focal loss
ln_p = np.log(np.array([0.6, 0.4], dtype=np.float64))
ln_1mp = np.log(1.0 - np.array([0.6, 0.4], dtype=np.float64))
yb = np.array([1.0, 0.0], dtype=np.float64)
w = np.ones(2, dtype=np.float64)
focal = beta_focal_nll_objective_nb(1.0, 1.0, 0.0, ln_p, ln_1mp, yb, w, 0.01, 2.0)
assert np.isfinite(focal) and focal > 0, f"Focal NLL invalid: {focal}"
print(f"4. Focal NLL: {focal:.4f}")

# Test 4: Temperature scaling
logits = np.array([0.5, -0.5], dtype=np.float64)
tnll = temperature_scaling_nll_nb(1.0, logits, yb, w)
assert np.isfinite(tnll) and tnll > 0, f"Temp NLL invalid: {tnll}"
print(f"5. Temperature NLL: {tnll:.4f}")

# Test 5: 5-param EMOS Student-t
pred = np.array([1.0, 2.0, -1.0], dtype=np.float64)
spred = np.array([0.5, 0.8, 0.6], dtype=np.float64)
actual = np.array([0.8, 2.5, -0.5], dtype=np.float64)
w3 = np.ones(3, dtype=np.float64)
emos_val = emos_crps_student_t_objective_nb(
    0.0, 1.0, 0.0, 1.0, 5.0,
    pred, spred, actual, w3,
    0.01, 0.01, 1.0, 30.0,
)
assert np.isfinite(emos_val) and emos_val > 0, f"EMOS-t obj invalid: {emos_val}"
print(f"6. EMOS Student-t objective: {emos_val:.4f}")

# Test 6: Import calibration module
from decision.signals_calibration import (
    CALIBRATION_VERSION,
    _fit_emos_student_t,
    _fit_temperature_scaling,
    _crps_student_t,
    _apply_temperature,
)
assert CALIBRATION_VERSION == "6.0", f"Version mismatch: {CALIBRATION_VERSION}"
print(f"7. Calibration version: {CALIBRATION_VERSION}")

# Test 7: _fit_emos_student_t
emos = _fit_emos_student_t(
    predicted=np.random.randn(30) * 2,
    actual=np.random.randn(30) * 3,
    sigma_pred=np.abs(np.random.randn(30)) + 0.5,
    nu_hat=np.full(30, 8.0),
    n_eval=30,
)
assert emos["type"] == "emos", f"Wrong type: {emos}"
assert "nu" in emos, f"Missing nu in EMOS: {emos}"
print(f"8. Student-t EMOS params: a={emos['a']}, b={emos['b']}, nu={emos['nu']}")

# Test 8: Temperature scaling
T = _fit_temperature_scaling(
    p_ups=np.random.uniform(0.3, 0.7, 30),
    actual_ups=np.random.binomial(1, 0.5, 30).astype(float),
)
assert 0.1 <= T <= 5.0, f"Temperature out of range: {T}"
print(f"9. Temperature: T={T:.3f}")

# Test 9: _apply_temperature
p_cal = _apply_temperature(0.6, 1.5)
assert 0.0 <= p_cal <= 1.0, f"p_cal out of range: {p_cal}"
print(f"10. apply_temperature(0.6, T=1.5) = {p_cal:.4f}")

# Test 10: _crps_student_t
crps_arr = _crps_student_t(
    mu=np.array([0.0, 1.0]),
    sigma=np.array([1.0, 2.0]),
    y=np.array([0.5, 0.5]),
    nu=np.array([5.0, 5.0]),
)
assert all(c > 0 for c in crps_arr), f"CRPS must be positive: {crps_arr}"
print(f"11. _crps_student_t: {crps_arr}")

# Test 11: Brier decomposition
cal_p = np.array([0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float64)
actual_u = np.array([0.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float64)
rel, res, unc = brier_decomposition_nb(cal_p, actual_u, 5)
assert 0 <= unc <= 0.25, f"Uncertainty out of range: {unc}"
print(f"12. Brier decomposition: REL={rel:.4f}, RES={res:.4f}, UNC={unc:.4f}")

# Test 12: Expanding CV
# Returns: (train_end_1, val_end_1, train_end_2, val_end_2, train_end_3, val_end_3)
# All folds have train starting at 0
te1, ve1, te2, ve2, te3, ve3 = expanding_cv_fold_indices_nb(100)
assert te1 < ve1 and te2 < ve2 and te3 < ve3, "Train must end before val end"
assert te1 < te2 < te3, "Train sizes must be expanding"
assert ve3 == 100, f"Last val must end at N: {ve3}"
print(f"13. CV folds: train_ends=[{te1},{te2},{te3}], val_ends=[{ve1},{ve2},{ve3}]")

print("\n=== ALL 13 SMOKE TESTS PASSED ===")
