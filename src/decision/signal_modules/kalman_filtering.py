from __future__ import annotations
"""
Kalman filtering core: innovation weighting, gain monitoring, and the main
_kalman_filter_drift() implementation.

Extracted from signals.py (Story 6.4). Contains the complete Kalman filter
with Student-t robust weighting, RV-Q/GAS-Q adaptive process noise,
Rauch-Tippett backward smoothing, and gain stall detection.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -- path setup so "from models..." and "from calibration..." work ----------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from scipy.stats import norm as scipy_norm

# Pull in all public symbols from signal_modules (feature flags, filter funcs, etc.)
from decision.signal_modules.config import *  # noqa: F403
from decision.signal_modules.volatility_imports import *  # noqa: F403

# Explicit private-name imports from sibling modules
from decision.signal_modules.parameter_loading import (
    _safe_get_nested, _load_tuned_kalman_params, _select_regime_params,
    is_student_t_family_model_name,
)
from decision.signal_modules.kalman_diagnostics import (
    _test_innovation_whiteness, _compute_kalman_log_likelihood,
    _compute_kalman_log_likelihood_heteroskedastic,
)

from ingestion.data_utils import _ensure_float_series


# =============================================================================
# KALMAN GAIN MONITORING AND ADAPTIVE RESET (Story 1.2)
# =============================================================================
# When Kalman gain K_t collapses below K_MIN_THRESHOLD for GAIN_STALL_WINDOW
# consecutive bars, the filter ignores >99% of new observations. This makes
# it unable to adapt when the market regime changes.
#
# The adaptive reset inflates P (state variance) at the stall point, which
# increases K_t and allows the filter to relearn from new observations.
#
# Anti-oscillation: maximum MAX_RESETS_PER_WINDOW resets per 252-bar window,
# with RESET_COOLDOWN_BARS bars between resets.
#
# K_t = P_pred / (P_pred + R_t)   [Gaussian]
# K_t = (nu/(nu+3)) * P_pred / (P_pred + R_t)   [Student-t]
# =============================================================================
K_MIN_THRESHOLD = 0.005           # Below this, filter is effectively stalled
GAIN_STALL_WINDOW = 10            # Consecutive bars below threshold to trigger reset
RESET_INFLATION_FACTOR = 10.0     # P_new = P_old * this factor
MAX_RESETS_PER_WINDOW = 3         # Maximum resets per 252-bar rolling window
RESET_COOLDOWN_BARS = 20          # Minimum bars between resets
GAIN_RESET_WINDOW_SIZE = 252      # Rolling window size (1 trading year)


# =============================================================================
# INNOVATION-WEIGHTED DRIFT AGGREGATION (Story 1.5)
# =============================================================================
# Large standardized innovations (z_t = innov / sqrt(S_t)) carry more information
# about regime changes than noise. Innovation weighting amplifies K_effective for
# surprising observations:
#
#   w(z_t) = min(1 + alpha * (|z_t| - 1)^+, w_max)
#   K_effective = K_t * w(z_t)
#
# This trades Kalman optimality guarantees (which assume Gaussian noise) for
# practical utility: faster response to earnings surprises and regime shifts.
# P_max bound prevents feedback divergence.
# =============================================================================
IW_ALPHA_UP = 0.3        # Sensitivity for positive surprises
IW_ALPHA_DOWN = 0.4      # Sensitivity for negative surprises (slightly higher)
IW_W_MAX = 2.0           # Maximum weight cap (conservative)
IW_P_MAX_MULT = 10.0     # P_max = this * P_steady_state


def innovation_weight(z_t: float, alpha_up: float = IW_ALPHA_UP,
                      alpha_down: float = IW_ALPHA_DOWN,
                      w_max: float = IW_W_MAX) -> float:
    """
    Compute innovation weight from standardized innovation z_t.

    w(z_t) = min(1 + alpha * (|z_t| - 1)^+, w_max)

    Asymmetric: negative surprises (z_t < 0) use alpha_down.
    Non-surprising observations (|z_t| <= 1) get w = 1.0 (no change).
    """
    abs_z = abs(z_t)
    if abs_z <= 1.0:
        return 1.0
    alpha = alpha_down if z_t < 0 else alpha_up
    return min(1.0 + alpha * (abs_z - 1.0), w_max)


def _compute_kalman_gain_from_filtered(
    P_filtered: np.ndarray,
    sigma: np.ndarray,
    phi: float,
    q: float,
    c: float,
    nu: Optional[float] = None,
) -> np.ndarray:
    """
    Compute K_t post-hoc from filtered P and observation noise.

    K_t = P_pred_t / (P_pred_t + R_t)  [Gaussian]
    K_t = (nu/(nu+3)) * P_pred_t / (P_pred_t + R_t)  [Student-t]

    where P_pred_t = phi^2 * P_{t-1} + q and R_t = c * sigma_t^2
    """
    T = len(P_filtered)
    K = np.zeros(T, dtype=float)
    phi_sq = phi * phi

    for t in range(T):
        if t == 0:
            P_prev = P_filtered[0]
        else:
            P_prev = P_filtered[t - 1]

        P_pred = phi_sq * P_prev + q
        R_t = max(c * sigma[t] * sigma[t], 1e-12)
        S_t = max(P_pred + R_t, 1e-12)
        K_t = P_pred / S_t

        if nu is not None and nu > 3.0:
            K_t *= nu / (nu + 3.0)

        K[t] = K_t

    return K


def _apply_gain_monitoring_reset(
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    K_gain: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    phi: float,
    q: float,
    c: float,
    nu: Optional[float] = None,
) -> dict:
    """
    Detect Kalman gain stalls and apply P-inflation resets.

    When K_t < K_MIN_THRESHOLD for GAIN_STALL_WINDOW consecutive bars,
    inflate P and re-run the filter forward from the stall point.

    Returns dict with:
        - reset_count: number of resets applied
        - reset_indices: list of bar indices where resets occurred
        - K_gain_pre_reset: original K_t before any resets
        - gain_stall_detected: bool
    """
    T = len(mu_filtered)
    if T < GAIN_STALL_WINDOW + RESET_COOLDOWN_BARS:
        return {
            "reset_count": 0,
            "reset_indices": [],
            "K_gain_pre_reset": K_gain.copy(),
            "gain_stall_detected": False,
        }

    K_pre = K_gain.copy()
    phi_sq = phi * phi
    is_student_t = nu is not None and nu > 3.0
    nu_factor = (nu / (nu + 3.0)) if is_student_t else 1.0

    # Detect stall windows: K_t < threshold for N consecutive bars
    stall_mask = K_gain < K_MIN_THRESHOLD
    consecutive_stall = np.zeros(T, dtype=int)
    for t in range(T):
        if stall_mask[t]:
            consecutive_stall[t] = (consecutive_stall[t - 1] + 1) if t > 0 else 1
        else:
            consecutive_stall[t] = 0

    # Find reset trigger points
    reset_indices = []
    last_reset = -RESET_COOLDOWN_BARS - 1  # Allow first reset immediately

    for t in range(GAIN_STALL_WINDOW, T):
        if consecutive_stall[t] >= GAIN_STALL_WINDOW:
            # Check cooldown
            if t - last_reset < RESET_COOLDOWN_BARS:
                continue

            # Check per-window maximum
            window_start = max(0, t - GAIN_RESET_WINDOW_SIZE)
            resets_in_window = sum(
                1 for ri in reset_indices if ri >= window_start
            )
            if resets_in_window >= MAX_RESETS_PER_WINDOW:
                continue

            reset_indices.append(t)
            last_reset = t

    if not reset_indices:
        return {
            "reset_count": 0,
            "reset_indices": [],
            "K_gain_pre_reset": K_pre,
            "gain_stall_detected": bool(np.any(consecutive_stall >= GAIN_STALL_WINDOW)),
        }

    # Apply resets: inflate P and re-run filter forward
    for ri in reset_indices:
        P_filtered[ri] = P_filtered[ri] * RESET_INFLATION_FACTOR

        # Re-run filter forward from reset point
        mu_t = mu_filtered[ri]
        P_t = P_filtered[ri]

        for t in range(ri + 1, T):
            # Check if we hit the next reset point (it will handle forward from there)
            if t in reset_indices:
                break

            mu_pred = phi * mu_t
            P_pred = phi_sq * P_t + q
            R_t = max(c * sigma[t] * sigma[t], 1e-12)
            S_t = max(P_pred + R_t, 1e-12)
            innov = y[t] - mu_pred
            K_t = nu_factor * P_pred / S_t

            if is_student_t:
                z_sq = (innov * innov) / S_t
                w_t = (nu + 1.0) / (nu + z_sq)
                mu_t = mu_pred + K_t * w_t * innov
                P_t = max((1.0 - w_t * K_t) * P_pred, 1e-12)
            else:
                mu_t = mu_pred + K_t * innov
                P_t = max((1.0 - K_t) * P_pred, 1e-12)

            mu_filtered[t] = mu_t
            P_filtered[t] = P_t
            K_gain[t] = K_t

    return {
        "reset_count": len(reset_indices),
        "reset_indices": reset_indices,
        "K_gain_pre_reset": K_pre,
        "gain_stall_detected": True,
    }


def _kalman_filter_drift(
    ret: pd.Series, 
    vol: pd.Series, 
    q: Optional[float] = None, 
    optimize_q: bool = True, 
    asset_symbol: Optional[str] = None,
    enable_online_updates: bool = True,
) -> Dict[str, pd.Series]:
    """
    Kalman filter for time-varying drift estimation using pre-tuned parameters.
    
    Args:
        ret: Returns series
        vol: Volatility series
        q: Override process noise (if None, use tuned)
        optimize_q: Legacy flag (kept for API compatibility)
        asset_symbol: Asset symbol for loading tuned parameters
        enable_online_updates: Deprecated, kept for API compatibility
        
    Returns:
        Dictionary with filtered drift estimates and diagnostics
    """
    ret_clean = _ensure_float_series(ret).dropna()
    vol_clean = _ensure_float_series(vol).reindex(ret_clean.index).dropna()
    df = pd.concat([ret_clean, vol_clean], axis=1, join='inner').dropna()
    if len(df) < 50:
        return {}
    df.columns = ['ret', 'vol']
    y = df['ret'].values.astype(float)
    sigma = df['vol'].values.astype(float)
    idx = df.index

    tuned_params = None
    if asset_symbol is not None:
        tuned_params = _load_tuned_kalman_params(asset_symbol)
    
    noise_model = (tuned_params or {}).get('noise_model', 'gaussian')
    is_rv_q = noise_model.startswith('rv_q_')
    requires_phi = 'phi' in noise_model or is_student_t_family_model_name(noise_model) or is_rv_q
    is_student_t = is_student_t_family_model_name(noise_model) or (is_rv_q and 'student_t' in noise_model)

    # =========================================================================
    # PARAMETER EXTRACTION: Batch-tuned parameters from cache
    # =========================================================================
    
    # φ is structural
    phi_used = (tuned_params or {}).get('phi')
    
    if requires_phi:
        if phi_used is None or not np.isfinite(phi_used):
            raise ValueError("phi required by selected model but missing from tuning cache")
        phi_used = float(phi_used)
    else:
        phi_used = 1.0

    # q: explicit arg > tuned
    if q is not None:
        q_used = q
    else:
        q_used = (tuned_params or {}).get('q')
    
    # c: from tuned
    c_used = (tuned_params or {}).get('c')
    
    # nu: from tuned (only for Student-t)
    if is_student_t:
        nu_used = (tuned_params or {}).get('nu')
    else:
        nu_used = None
    
    # =========================================================================
    # MARKET CONDITIONING: VIX-Based ν Adjustment (February 2026)
    # =========================================================================
    # When market is stressed (high VIX), reduce ν to increase tail heaviness.
    # This makes the model more conservative during volatile periods.
    # ν_adjusted = max(ν_min, ν_base - κ × VIX_normalized)
    # =========================================================================
    vix_nu_adjustment_applied = False
    nu_original = nu_used
    
    if is_student_t and nu_used is not None:
        vix_adjustment_enabled = (tuned_params or {}).get('vix_nu_adjustment_enabled', False)
        if vix_adjustment_enabled:
            try:
                from calibration.market_conditioning import compute_vix_nu_adjustment
                vix_result = compute_vix_nu_adjustment(nu_base=float(nu_used))
                if vix_result.adjustment_applied:
                    nu_used = vix_result.nu_adjusted
                    vix_nu_adjustment_applied = True
            except Exception as vix_err:
                # Silently fallback to original nu on any error
                pass

    if q_used is None or not np.isfinite(q_used) or q_used <= 0:
        return {}
    if c_used is None or not np.isfinite(c_used) or c_used <= 0:
        raise ValueError("Observation scale c required by tuned model but missing/invalid")
    if is_student_t and (nu_used is None or not np.isfinite(nu_used)):
        raise ValueError("Student-t model selected but ν missing from tuning cache")

    obs_scale = float(c_used)
    q_scalar = float(q_used)

    T = len(y)
    mu_filtered = np.zeros(T, dtype=float)
    P_filtered = np.zeros(T, dtype=float)
    K_gain = np.zeros(T, dtype=float)
    innovations = np.zeros(T, dtype=float)
    innovation_vars = np.zeros(T, dtype=float)

    # =========================================================================
    # GAS-Q SCORE-DRIVEN PARAMETER DYNAMICS (February 2026)
    # =========================================================================
    # Check if GAS-Q augmentation is enabled in tuned params.
    # When gas_q_augmented=True, q evolves dynamically via score-driven updates.
    # =========================================================================
    gas_q_augmented = False
    gas_q_result = None
    q_t_series = None
    
    if GAS_Q_AVAILABLE and tuned_params is not None:
        gas_q_augmented = tuned_params.get('gas_q_augmented', False)
        gas_q_params = tuned_params.get('gas_q_params', {})
        
        if gas_q_augmented and gas_q_params:
            try:
                # Build GAS-Q config from tuned params
                gas_config = GASQConfig(
                    omega=gas_q_params.get('omega', q_scalar * 0.1),
                    alpha=gas_q_params.get('alpha', 0.1),
                    beta=gas_q_params.get('beta', 0.5),
                )
                
                # Run GAS-Q filter to get dynamic q_t series
                if is_student_t and nu_used is not None:
                    gas_q_result = gas_q_filter_student_t(
                        y=y,
                        sigma=sigma,
                        phi=phi_used,
                        c=obs_scale,
                        nu=nu_used,
                        gas_config=gas_config,
                        q_init=q_scalar,
                    )
                else:
                    gas_q_result = gas_q_filter_gaussian(
                        y=y,
                        sigma=sigma,
                        phi=phi_used,
                        c=obs_scale,
                        gas_config=gas_config,
                        q_init=q_scalar,
                    )
                
                if gas_q_result is not None and hasattr(gas_q_result, 'q_t'):
                    q_t_series = gas_q_result.q_t
                    # Use pre-computed filter results from GAS-Q
                    mu_filtered = gas_q_result.mu_filtered
                    P_filtered = gas_q_result.P_filtered
                    K_gain = gas_q_result.K_gain if hasattr(gas_q_result, 'K_gain') else np.zeros(T)
                    innovations = gas_q_result.innovations if hasattr(gas_q_result, 'innovations') else np.zeros(T)
                    innovation_vars = gas_q_result.innovation_vars if hasattr(gas_q_result, 'innovation_vars') else np.zeros(T)
                    log_likelihood = gas_q_result.log_likelihood if hasattr(gas_q_result, 'log_likelihood') else 0.0
                    
            except Exception as gas_e:
                # Graceful fallback to static q if GAS-Q fails
                if os.getenv("DEBUG"):
                    print(f"GAS-Q filter failed, using static q: {gas_e}")
                gas_q_augmented = False
                gas_q_result = None

    # =========================================================================
    # RV-ADAPTIVE PROCESS NOISE DISPATCH (Tune.md Epic 1)
    # =========================================================================
    # When the BMA-selected model is an RV-Q variant, dispatch to the
    # proactive RV-adaptive filter instead of static q or GAS-Q.
    #   q_t = q_base * exp(gamma * delta_log(sigma_t^2))
    # Responds to volatility changes immediately (not reactively).
    # =========================================================================
    rv_q_result = None
    if RV_Q_AVAILABLE and is_rv_q and gas_q_result is None and tuned_params is not None:
        rv_q_base = tuned_params.get('q_base')
        rv_gamma = tuned_params.get('gamma')
        if rv_q_base is not None and rv_gamma is not None:
            try:
                rv_config = RVAdaptiveQConfig(
                    q_base=float(rv_q_base),
                    gamma=float(rv_gamma),
                )
                if is_student_t and nu_used is not None:
                    rv_q_result = rv_adaptive_q_filter_student_t(
                        y, sigma, obs_scale, phi_used, float(nu_used),
                        config=rv_config,
                    )
                else:
                    rv_q_result = rv_adaptive_q_filter_gaussian(
                        y, sigma, obs_scale, phi_used,
                        config=rv_config,
                    )
                if rv_q_result is not None:
                    mu_filtered = rv_q_result.mu_filtered
                    P_filtered = rv_q_result.P_filtered
                    log_likelihood = rv_q_result.log_likelihood
                    # Recompute innovations from filtered state
                    for t in range(T):
                        mu_pred = phi_used * (mu_filtered[t - 1] if t > 0 else 0.0)
                        innovations[t] = y[t] - mu_pred
                        R_t = max(obs_scale * (sigma[t] ** 2), 1e-12)
                        P_pred = (phi_used ** 2) * (P_filtered[t - 1] if t > 0 else 1.0) + rv_q_result.q_path[t]
                        S_t = max(P_pred + R_t, 1e-12)
                        innovation_vars[t] = S_t
                        K_gain[t] = P_pred / S_t
            except Exception as rv_q_err:
                if os.getenv("DEBUG"):
                    print(f"RV-Q filter failed, falling through to standard: {rv_q_err}")
                rv_q_result = None

    # =========================================================================
    # ENHANCED STUDENT-T MODELS (February 2026)
    # =========================================================================
    # Check for enhanced Student-t variants:
    #   - Unified: phi_student_t_unified_nu_* (combines all enhancements)
    #   - Vol-of-Vol (VoV): gamma_vov in tuned params
    #   - Two-Piece: nu_left and nu_right in tuned params
    #   - Mixture: nu_calm and nu_stress in tuned params
    #   - MS-q: Markov-Switching process noise (q_calm, q_stress)
    # These use specialized filter implementations from PhiStudentTDriftModel.
    # =========================================================================
    enhanced_result = None
    enhanced_model_type = None
    ms_q_diagnostics = None
    
    # Check for Unified Gaussian model (February 2026 - ν-free Calibration Pipeline)
    if tuned_params is not None and tuned_params.get('gaussian_unified') and gas_q_result is None:
        try:
            from models.gaussian import GaussianDriftModel, GaussianUnifiedConfig
            import numpy as _np_sig
            
            # Build GaussianUnifiedConfig from tuned params (including momentum + GAS-Q)
            g_unified_config = GaussianUnifiedConfig(
                q=float(tuned_params.get('q', 1e-6)),
                c=float(tuned_params.get('c', 1.0)),
                phi=float(tuned_params.get('phi', 0.0)),
                variance_inflation=float(tuned_params.get('variance_inflation', 1.0)),
                mu_drift=float(tuned_params.get('mu_drift', 0.0)),
                garch_omega=float(tuned_params.get('garch_omega', 0.0)),
                garch_alpha=float(tuned_params.get('garch_alpha', 0.0)),
                garch_beta=float(tuned_params.get('garch_beta', 0.0)),
                garch_leverage=float(tuned_params.get('garch_leverage', 0.0)),
                garch_unconditional_var=float(tuned_params.get('garch_unconditional_var', 1e-4)),
                crps_ewm_lambda=float(tuned_params.get('crps_ewm_lambda', 0.0)),
                crps_sigma_shrinkage=float(tuned_params.get('crps_sigma_shrinkage', 1.0)),
                calibrated_gw=float(tuned_params.get('calibrated_gw', 0.0)),
                calibrated_lambda_rho=float(tuned_params.get('calibrated_lambda_rho', 0.985)),
                calibrated_beta_probit_corr=float(tuned_params.get('calibrated_beta_probit_corr', 1.0)),
                # Momentum integration (Stage 1.5)
                momentum_weight=float(tuned_params.get('momentum_weight', 0.0)),
                # GAS-Q adaptive process noise (Stage 4.5)
                gas_q_omega=float(tuned_params.get('gas_q_omega', 0.0)),
                gas_q_alpha=float(tuned_params.get('gas_q_alpha', 0.0)),
                gas_q_beta=float(tuned_params.get('gas_q_beta', 0.0)),
            )
            
            # Reconstruct momentum signal at inference time (deterministic given returns)
            _gu_mom_signal = None
            if g_unified_config.momentum_enabled:
                try:
                    from models.momentum_augmented import compute_momentum_features, compute_momentum_signal
                    _gu_mom_feats = compute_momentum_features(y)
                    _gu_mom_signal = compute_momentum_signal(_gu_mom_feats)
                except Exception:
                    _gu_mom_signal = None
            
            # Run filter with momentum augmentation (if active)
            mu_g, P_g, mu_pred_g, S_pred_g, ll_g = GaussianDriftModel._filter_phi_with_momentum(
                y, sigma, g_unified_config.q, g_unified_config.c, g_unified_config.phi,
                _gu_mom_signal, g_unified_config.momentum_weight
            )
            mu_filtered = mu_g
            P_filtered = P_g
            log_likelihood = ll_g
            enhanced_result = True
            enhanced_model_type = 'gaussian_unified'
            
            # Get calibrated variance from filter_and_calibrate
            try:
                _pit_gu, _pit_p_gu, _sigma_cal_gu, _, _calib_diag_gu = \
                    GaussianDriftModel.filter_and_calibrate(
                        y, sigma, g_unified_config, train_frac=0.7,
                        momentum_signal=_gu_mom_signal
                    )
                _n_train_gu = int(len(y) * 0.7)
                if len(_sigma_cal_gu) == len(y) - _n_train_gu and _np_sig.all(_sigma_cal_gu > 0):
                    _S_cal_full_gu = _np_sig.copy(S_pred_g)
                    _S_cal_full_gu[_n_train_gu:] = _sigma_cal_gu ** 2
                    P_filtered = _S_cal_full_gu
            except Exception:
                pass  # Keep raw P_filtered on failure
            
            if os.getenv("DEBUG"):
                print(f"Using unified Gaussian filter: φ={g_unified_config.phi:.3f}, "
                      f"β={g_unified_config.variance_inflation:.3f}, "
                      f"garch=({g_unified_config.garch_alpha:.3f},{g_unified_config.garch_beta:.3f}), "
                      f"gw={g_unified_config.calibrated_gw:.3f}, "
                      f"mom_w={g_unified_config.momentum_weight:.3f}, "
                      f"gas_q={'ON' if g_unified_config.gas_q_enabled else 'OFF'}")
        except Exception as gu_e:
            if os.getenv("DEBUG"):
                print(f"Unified Gaussian filter failed, falling back: {gu_e}")
            enhanced_result = None
    
    # Check for Unified Student-T model (February 2026 - Elite Architecture)
    if tuned_params is not None and tuned_params.get('unified_model') and not tuned_params.get('gaussian_unified') and gas_q_result is None:
        try:
            _use_improved_unified = (
                tuned_params.get('model_type') == 'phi_student_t_unified_improved'
                or tuned_params.get('implementation') == 'improved'
                or str(tuned_params.get('best_model', '')).startswith('phi_student_t_unified_improved_nu_')
                or str(noise_model).startswith('phi_student_t_unified_improved_nu_')
            )
            if _use_improved_unified:
                from models.phi_student_t_unified_improved import (
                    UnifiedStudentTConfig,
                    UnifiedPhiStudentTModel,
                )
            else:
                from models.phi_student_t_unified import UnifiedStudentTConfig, UnifiedPhiStudentTModel
            import numpy as _np_sig
            
            # Build unified config from tuned params with ALL evolved parameters
            # Includes: core, asymmetry, MS-q, VoV, calibration, GARCH,
            #           wavelet/DTCWT, Merton jump-diffusion, and data-driven bounds
            unified_config = UnifiedStudentTConfig(
                # Core parameters
                q=float(tuned_params.get('q', 1e-6)),
                c=float(tuned_params.get('c', 1.0)),
                phi=float(tuned_params.get('phi', 0.0)),
                nu_base=float(tuned_params.get('nu', 8)),
                # Asymmetric tails
                alpha_asym=float(tuned_params.get('alpha_asym', 0.0)),
                k_asym=float(tuned_params.get('k_asym', 1.0)),
                # Markov-switching process noise
                gamma_vov=float(tuned_params.get('gamma_vov', 0.3)),
                ms_sensitivity=float(tuned_params.get('ms_sensitivity', 2.0)),
                ms_ewm_lambda=float(tuned_params.get('ms_ewm_lambda', 0.0)),
                q_stress_ratio=float(tuned_params.get('q_stress_ratio', 10.0)),
                vov_damping=float(tuned_params.get('vov_damping', 0.3)),
                # Calibration parameters (February 2026)
                variance_inflation=float(tuned_params.get('variance_inflation', 1.0)),
                mu_drift=float(tuned_params.get('mu_drift', 0.0)),
                risk_premium_sensitivity=float(tuned_params.get('risk_premium_sensitivity', 0.0)),
                # Conditional skew dynamics (February 2026 - GAS Framework)
                skew_score_sensitivity=float(tuned_params.get('skew_score_sensitivity', 0.0)),
                skew_persistence=float(tuned_params.get('skew_persistence', 0.97)),
                # GARCH parameters (February 2026)
                garch_omega=float(tuned_params.get('garch_omega', 0.0)),
                garch_alpha=float(tuned_params.get('garch_alpha', 0.0)),
                garch_beta=float(tuned_params.get('garch_beta', 0.0)),
                garch_leverage=float(tuned_params.get('garch_leverage', 0.0)),
                garch_unconditional_var=float(tuned_params.get('garch_unconditional_var', 1e-4)),
                # Rough volatility memory (February 2026 - Gatheral-Jaisson-Rosenbaum)
                rough_hurst=float(tuned_params.get('rough_hurst', 0.0)),
                # Merton jump-diffusion parameters (February 2026)
                jump_intensity=float(tuned_params.get('jump_intensity', 0.0)),
                jump_variance=float(tuned_params.get('jump_variance', 0.0)),
                jump_sensitivity=float(tuned_params.get('jump_sensitivity', 1.0)),
                jump_mean=float(tuned_params.get('jump_mean', 0.0)),
                # Data-driven bounds
                c_min=float(tuned_params.get('c_min', 0.01)),
                c_max=float(tuned_params.get('c_max', 10.0)),
                q_min=float(tuned_params.get('q_min', 1e-8)),
                # CRPS-optimal EWM location correction (February 2026)
                crps_ewm_lambda=float(tuned_params.get('crps_ewm_lambda', 0.0)),
                # Heston-DLSV leverage and mean reversion (February 2026)
                rho_leverage=float(tuned_params.get('rho_leverage', 0.0)),
                kappa_mean_rev=float(tuned_params.get('kappa_mean_rev', 0.0)),
                theta_long_var=float(tuned_params.get('theta_long_var', 0.0)),
                crps_sigma_shrinkage=float(tuned_params.get('crps_sigma_shrinkage', 1.0)),
                # CRPS-enhancement: vol-of-vol noise, asymmetric df, regime switching
                sigma_eta=float(tuned_params.get('sigma_eta', 0.0)),
                t_df_asym=float(tuned_params.get('t_df_asym', 0.0)),
                regime_switch_prob=float(tuned_params.get('regime_switch_prob', 0.0)),
                # GARCH-Kalman reconciliation + Q_t coupling + location bias (February 2026)
                garch_kalman_weight=float(tuned_params.get('garch_kalman_weight', 0.0)),
                q_vol_coupling=float(tuned_params.get('q_vol_coupling', 0.0)),
                loc_bias_var_coeff=float(tuned_params.get('loc_bias_var_coeff', 0.0)),
                loc_bias_drift_coeff=float(tuned_params.get('loc_bias_drift_coeff', 0.0)),
                # v7.8: Elite MC enhancements
                leverage_dynamic_decay=float(tuned_params.get('leverage_dynamic_decay', 0.0)),
                liq_stress_coeff=float(tuned_params.get('liq_stress_coeff', 0.0)),
                entropy_sigma_lambda=float(tuned_params.get('entropy_sigma_lambda', 0.0)),
                # Stage 6: pre-calibrated walk-forward CV params
                calibrated_gw=float(tuned_params.get('calibrated_gw', 0.50)),
                calibrated_nu_pit=float(tuned_params.get('calibrated_nu_pit', 0.0)),
                calibrated_beta_probit_corr=float(tuned_params.get('calibrated_beta_probit_corr', 1.0)),
                calibrated_lambda_rho=float(tuned_params.get('calibrated_lambda_rho', 0.985)),
                calibrated_nu_crps=float(tuned_params.get('calibrated_nu_crps', 0.0)),
            )
            
            # Run unified filter
            mu_u, P_u, mu_pred_u, S_pred_u, ll_u = UnifiedPhiStudentTModel.filter_phi_unified(
                y, sigma, unified_config
            )
            mu_filtered = mu_u
            P_filtered = P_u
            log_likelihood = ll_u
            enhanced_result = True
            enhanced_model_type = 'unified'
            
            # ─────────────────────────────────────────────────────────
            # Calibrated variance (February 2026)
            # Use filter_and_calibrate to get GARCH-blended, β-corrected
            # predictive variance. This is the same calibrated sigma used
            # for PIT testing, ensuring signal generation uses the most
            # accurate variance estimate.
            # ─────────────────────────────────────────────────────────
            try:
                _pit_sig, _pit_p_sig, _sigma_cal_sig, _, _calib_diag_sig = \
                    UnifiedPhiStudentTModel.filter_and_calibrate(
                        y, sigma, unified_config, train_frac=0.7
                    )
                _n_train_sig = int(len(y) * 0.7)
                _nu_eff_sig = _calib_diag_sig.get('nu_effective', unified_config.nu_base)
                # Store calibrated sigma for downstream signal generation
                if len(_sigma_cal_sig) == len(y) - _n_train_sig and _np_sig.all(_sigma_cal_sig > 0):
                    # Extend P_filtered with calibrated variance for test period
                    _S_cal_full = _np_sig.copy(S_pred_u)
                    _S_cal_full[_n_train_sig:] = _sigma_cal_sig ** 2 * _nu_eff_sig / max(_nu_eff_sig - 2, 0.1)
                    P_filtered = _S_cal_full
            except Exception:
                pass  # Keep raw P_filtered on failure
            
            if os.getenv("DEBUG"):
                jump_active = unified_config.jump_intensity > 1e-6 and unified_config.jump_variance > 1e-12
                gjr_active = unified_config.garch_leverage > 1e-6
                rough_active = unified_config.rough_hurst > 0.01
                rp_active = abs(getattr(unified_config, 'risk_premium_sensitivity', 0.0)) > 1e-4
                skew_active = getattr(unified_config, 'skew_score_sensitivity', 0.0) > 1e-6
                _impl_label = "improved unified" if _use_improved_unified else "unified"
                print(f"Using {_impl_label} Student-T filter: ν={unified_config.nu_base}, "
                      f"α={unified_config.alpha_asym:.3f}, γ_vov={unified_config.gamma_vov:.2f}, "
                      f"β={unified_config.variance_inflation:.3f}, "
                      f"garch=({unified_config.garch_alpha:.3f},{unified_config.garch_beta:.3f}), "
                      f"GJR_γ={'%.3f' % unified_config.garch_leverage if gjr_active else 'OFF'}, "
                      f"H={'%.3f' % unified_config.rough_hurst if rough_active else 'OFF'}, "
                      f"λ₁={'%.3f' % unified_config.risk_premium_sensitivity if rp_active else 'OFF'}, "
                      f"κ_skew={'%.4f' % unified_config.skew_score_sensitivity if skew_active else 'OFF'}, "
                      f"jump={'ON' if jump_active else 'OFF'}"
                      + (f" p₀={unified_config.jump_intensity:.3f} σ²_J={unified_config.jump_variance:.5f}" if jump_active else ""))
        except Exception as unified_e:
            if os.getenv("DEBUG"):
                print(f"Unified filter failed, falling back: {unified_e}")
            enhanced_result = None
    
    if ENHANCED_STUDENT_T_AVAILABLE and tuned_params is not None and gas_q_result is None:
        # Check for Vol-of-Vol enhancement
        if tuned_params.get('vov_enhanced') and tuned_params.get('gamma_vov') is not None:
            try:
                gamma_vov = float(tuned_params.get('gamma_vov'))
                nu_vov = float(tuned_params.get('nu', 8))  # Default ν if not specified
                mu_vov, P_vov, ll_vov = PhiStudentTDriftModel.filter_phi_vov(
                    y, sigma, q_scalar, obs_scale, phi_used, nu_vov, gamma_vov
                )
                mu_filtered = mu_vov
                P_filtered = P_vov
                log_likelihood = ll_vov
                enhanced_result = True
                enhanced_model_type = 'vov'
            except Exception as vov_e:
                if os.getenv("DEBUG"):
                    print(f"VoV filter failed, using standard: {vov_e}")
                enhanced_result = None
        
        # Check for Two-Piece asymmetric tails
        elif tuned_params.get('two_piece') and tuned_params.get('nu_left') is not None:
            try:
                nu_left = float(tuned_params.get('nu_left'))
                nu_right = float(tuned_params.get('nu_right'))
                mu_2p, P_2p, ll_2p = PhiStudentTDriftModel.filter_phi_two_piece(
                    y, sigma, q_scalar, obs_scale, phi_used, nu_left, nu_right
                )
                mu_filtered = mu_2p
                P_filtered = P_2p
                log_likelihood = ll_2p
                enhanced_result = True
                enhanced_model_type = 'two_piece'
            except Exception as tp_e:
                if os.getenv("DEBUG"):
                    print(f"Two-Piece filter failed, using standard: {tp_e}")
                enhanced_result = None
        
        # Check for Two-Component Mixture
        elif tuned_params.get('mixture_model') and tuned_params.get('nu_calm') is not None:
            try:
                nu_calm = float(tuned_params.get('nu_calm'))
                nu_stress = float(tuned_params.get('nu_stress'))
                w_base = float(tuned_params.get('w_base', 0.5))
                # Use enhanced mixture if enabled (multi-factor weight dynamics)
                if ENHANCED_MIXTURE_ENABLED and ENHANCED_MIXTURE_AVAILABLE:
                    mu_mix, P_mix, ll_mix = PhiStudentTDriftModel.filter_phi_mixture_enhanced(
                        y, sigma, q_scalar, obs_scale, phi_used, nu_calm, nu_stress, w_base,
                        a_shock=MIXTURE_WEIGHT_A_SHOCK,
                        b_vol_accel=MIXTURE_WEIGHT_B_VOL_ACCEL,
                        c_momentum=MIXTURE_WEIGHT_C_MOMENTUM,
                    )
                else:
                    mu_mix, P_mix, ll_mix = PhiStudentTDriftModel.filter_phi_mixture(
                        y, sigma, q_scalar, obs_scale, phi_used, nu_calm, nu_stress, w_base
                    )
                mu_filtered = mu_mix
                P_filtered = P_mix
                log_likelihood = ll_mix
                enhanced_result = True
                enhanced_model_type = 'mixture'
            except Exception as mix_e:
                if os.getenv("DEBUG"):
                    print(f"Mixture filter failed, using standard: {mix_e}")
                enhanced_result = None
        
        # Check for Markov-Switching Process Noise (MS-q) — February 2026
        elif tuned_params.get('ms_q_augmented') and MS_Q_AVAILABLE and MS_Q_ENABLED:
            try:
                q_calm = float(tuned_params.get('q_calm', MS_Q_CALM_DEFAULT))
                q_stress = float(tuned_params.get('q_stress', MS_Q_STRESS_DEFAULT))
                nu_ms = float(tuned_params.get('nu', 8))
                
                # Use MS-q filter with proactive regime-switching q
                mu_ms, P_ms, ll_ms, q_t, p_stress = filter_phi_ms_q(
                    y, sigma, obs_scale, phi_used, nu_ms,
                    q_calm=q_calm, q_stress=q_stress,
                    sensitivity=MS_Q_SENSITIVITY,
                    threshold=MS_Q_THRESHOLD,
                )
                mu_filtered = mu_ms
                P_filtered = P_ms
                log_likelihood = ll_ms
                enhanced_result = True
                enhanced_model_type = 'ms_q'
                # Store MS-q diagnostics for downstream use
                ms_q_diagnostics = {
                    'q_t_mean': float(np.mean(q_t)),
                    'q_t_std': float(np.std(q_t)),
                    'p_stress_mean': float(np.mean(p_stress)),
                    'p_stress_max': float(np.max(p_stress)),
                    'q_calm': q_calm,
                    'q_stress': q_stress,
                }
            except Exception as ms_q_e:
                if os.getenv("DEBUG"):
                    print(f"MS-q filter failed, using standard: {ms_q_e}")
                enhanced_result = None

    mu_t = 0.0
    P_t = 1.0
    log_likelihood_init = 0.0 if gas_q_result is None and rv_q_result is None else log_likelihood

    # Only run static filter if neither GAS-Q, RV-Q, nor enhanced models were used
    if gas_q_result is None and rv_q_result is None and enhanced_result is None:
        log_likelihood = log_likelihood_init
        # P_max bound for innovation weighting (Story 1.5)
        # Estimate steady-state P from q and obs_scale * typical vol^2
        _P_ss_est = max(q_scalar * 100.0, 1e-6)
        _P_max = IW_P_MAX_MULT * _P_ss_est

        for t in range(T):
            mu_pred = phi_used * mu_t
            P_pred = (phi_used ** 2) * P_t + q_scalar
            R_t = float(max(obs_scale * (sigma[t] ** 2), 1e-12))
            innov = y[t] - mu_pred
            S_t = float(max(P_pred + R_t, 1e-12))

            if is_student_t:
                nu_adj = min(nu_used / (nu_used + 3.0), 1.0)
                K_t = nu_adj * P_pred / S_t
            else:
                K_t = P_pred / S_t

            # Innovation weighting (Story 1.5): boost K_t for surprising obs
            z_t = innov / max(math.sqrt(S_t), 1e-8)
            w_iw = innovation_weight(z_t)
            K_eff = min(K_t * w_iw, 0.99)  # Cap to prevent K >= 1

            mu_t = mu_pred + K_eff * innov
            P_t = float(max((1.0 - K_eff) * P_pred, 1e-12))
            P_t = min(P_t, _P_max)  # P_max bound (Story 1.5)

            mu_filtered[t] = mu_t
            P_filtered[t] = P_t
            K_gain[t] = K_eff
            innovations[t] = innov
            innovation_vars[t] = S_t

            if is_student_t:
                try:
                    ll_t = StudentTDriftModel.logpdf(innov, nu_used, 0.0, math.sqrt(S_t))
                    if np.isfinite(ll_t):
                        log_likelihood += ll_t
                except Exception:
                    pass
            else:
                try:
                    ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
                    if np.isfinite(ll_t):
                        log_likelihood += ll_t
                except Exception:
                    pass

    kalman_gain_mean = float(np.mean(K_gain))
    kalman_gain_recent = float(K_gain[-1]) if len(K_gain) > 0 else float('nan')

    # =========================================================================
    # KALMAN GAIN MONITORING AND ADAPTIVE RESET (Story 1.2)
    # =========================================================================
    # For enhanced/GAS-Q paths, K_gain may be all zeros since the external
    # filter functions don't return it. Compute post-hoc from P_filtered.
    # Then detect gain stalls and apply P-inflation resets.
    # =========================================================================
    gain_reset_info = {
        "reset_count": 0,
        "reset_indices": [],
        "gain_stall_detected": False,
    }

    if T >= GAIN_STALL_WINDOW + RESET_COOLDOWN_BARS:
        # Compute K_t post-hoc if not already populated (enhanced/GAS-Q paths)
        if np.all(K_gain == 0) and T > 1:
            K_gain[:] = _compute_kalman_gain_from_filtered(
                P_filtered, sigma, phi_used, q_scalar, obs_scale,
                nu=float(nu_used) if nu_used is not None else None,
            )
            kalman_gain_mean = float(np.mean(K_gain))
            kalman_gain_recent = float(K_gain[-1])

        # Apply gain monitoring and adaptive reset
        gain_reset_info = _apply_gain_monitoring_reset(
            mu_filtered, P_filtered, K_gain,
            y, sigma,
            phi=phi_used, q=q_scalar, c=obs_scale,
            nu=float(nu_used) if nu_used is not None else None,
        )

        if gain_reset_info["reset_count"] > 0:
            # Recompute diagnostics after reset
            kalman_gain_mean = float(np.mean(K_gain))
            kalman_gain_recent = float(K_gain[-1])

    # =========================================================================
    # ONLINE c UPDATE (Story 2.3) — Adaptive obs noise between re-tuning
    # =========================================================================
    # c_t+1 = c_t + eta * (v_t^2/R_t - 1) with decaying learning rate.
    # Updates c using the innovations from the filter run above.
    # The final c_online replaces obs_scale for forecast interval computation.
    # =========================================================================
    _online_c_applied = False
    _online_c_final = obs_scale
    if ONLINE_C_UPDATE_AVAILABLE and T >= 30:
        try:
            _oc_result = run_online_c_update(
                returns=y,
                mu_filtered=mu_filtered,
                vol=sigma,
                c_init=obs_scale,
                phi=phi_used,
            )
            if _oc_result is not None and _oc_result.c_final > 0 and np.isfinite(_oc_result.c_final):
                _online_c_final = _oc_result.c_final
                obs_scale = _online_c_final
                _online_c_applied = True
        except Exception:
            pass

    # =========================================================================
    # TWSC SCALE CORRECTION — REAL MODEL IMPROVEMENT (March 2026)
    # =========================================================================
    # Apply the Tail-Weighted Scale Correction factor from tuning to P_filtered.
    # This corrects systematic scale bias identified during calibration:
    #   P_filtered_corrected = P_filtered × twsc_scale_factor²
    #
    # The scale factor was computed during tuning as the geometric mean of
    # EWMA-tracked tail exceedance inflation factors. When model variance
    # is systematically too tight, twsc_scale_factor > 1 → wider variance
    # → more conservative position sizing and better calibrated MC samples.
    #
    # Impact chain: P_filtered → P_t_mc → MC simulation → EU/sizing → PnL
    # =========================================================================
    _twsc_applied = False
    _twsc_factor = 1.0
    if tuned_params is not None:
        _global_cal = (tuned_params.get('global') or {}).get('calibration_params', {})
        _twsc_factor = _global_cal.get('twsc_scale_factor', 1.0)
        if isinstance(_twsc_factor, (int, float)) and np.isfinite(_twsc_factor):
            _twsc_factor = float(np.clip(_twsc_factor, 0.5, 3.0))  # Safety bounds
            if abs(_twsc_factor - 1.0) > 0.01:  # Only apply if meaningful
                P_filtered = P_filtered * (_twsc_factor ** 2)
                _twsc_applied = True

    return {
        "mu_kf_filtered": pd.Series(mu_filtered, index=idx, name="mu_kf_filtered"),
        "mu_kf_smoothed": pd.Series(mu_filtered, index=idx, name="mu_kf_smoothed"),  # no RTS smoothing here
        "var_kf_filtered": pd.Series(P_filtered, index=idx, name="var_kf_filtered"),
        "var_kf_smoothed": pd.Series(P_filtered, index=idx, name="var_kf_smoothed"),
        "kalman_gain": pd.Series(K_gain, index=idx, name="kalman_gain"),
        "innovations": pd.Series(innovations, index=idx, name="innovations"),
        "innovation_vars": pd.Series(innovation_vars, index=idx, name="innovation_vars"),
        "log_likelihood": float(log_likelihood),
        "process_noise_var": float(q_scalar),
        "n_obs": int(T),
        "kalman_gain_mean": kalman_gain_mean,
        "kalman_gain_recent": kalman_gain_recent,
        "kalman_heteroskedastic_mode": False,
        "kalman_c_optimal": obs_scale,
        "online_c_applied": _online_c_applied,
        "online_c_final": _online_c_final,
        "phi_used": float(phi_used) if phi_used is not None and np.isfinite(phi_used) else None,
        "kalman_noise_model": noise_model,
        "kalman_nu": float(nu_used) if nu_used is not None else None,
        # =========================================================================
        # KALMAN GAIN MONITORING AND ADAPTIVE RESET (Story 1.2)
        # =========================================================================
        "gain_reset_count": gain_reset_info["reset_count"],
        "gain_reset_indices": gain_reset_info["reset_indices"],
        "gain_stall_detected": gain_reset_info["gain_stall_detected"],
        # =========================================================================
        # GAS-Q SCORE-DRIVEN PARAMETER DYNAMICS DIAGNOSTICS (February 2026)
        # =========================================================================
        # When gas_q_augmented=True, process noise q evolves dynamically via
        # score-driven updates: q_t = omega + alpha * s_{t-1} + beta * q_{t-1}
        # This allows adaptive filtering during volatility regime changes.
        # =========================================================================
        "gas_q_augmented": gas_q_augmented,
        "gas_q_result": gas_q_result,
        "q_t_series": pd.Series(q_t_series, index=idx, name="q_t") if q_t_series is not None else None,
        # =========================================================================
        # ENHANCED STUDENT-T MODEL DIAGNOSTICS (February 2026)
        # =========================================================================
        # When enhanced models are used, this tracks which variant was applied:
        #   - 'vov': Vol-of-Vol adjustment to observation noise
        #   - 'two_piece': Asymmetric tails (νL ≠ νR)
        #   - 'mixture': Two-component mixture (νcalm + νstress)
        #   - 'ms_q': Markov-Switching process noise
        # =========================================================================
        "enhanced_student_t_active": enhanced_result is not None,
        "enhanced_student_t_type": enhanced_model_type,
        "ms_q_diagnostics": ms_q_diagnostics,
        # =========================================================================
        # VIX-BASED ν ADJUSTMENT DIAGNOSTICS (February 2026)
        # =========================================================================
        # When market is stressed (high VIX), ν is reduced to increase tail heaviness.
        # ν_adjusted = max(ν_min, ν_base - κ × VIX_normalized)
        # =========================================================================
        "vix_nu_adjustment_applied": vix_nu_adjustment_applied,
        "nu_original": float(nu_original) if nu_original is not None else None,
        "nu_adjusted": float(nu_used) if nu_used is not None else None,
        # =========================================================================
        # CALIBRATION CORRECTIONS (March 2026) — REAL MODEL IMPROVEMENT
        # =========================================================================
        # Parameters from tuning's AD correction pipeline, applied here for
        # real predictive improvement (not just cosmetic AD p-value inflation).
        #
        # twsc_scale_applied: Whether scale correction was applied to P_filtered
        # twsc_scale_factor: The scale factor used (>1 = model was too tight)
        # calibration_params: Full dict of correction params for downstream use:
        #   - gpd_left_xi, gpd_right_xi: GPD tail shape params (→ ν adjustment)
        #   - isotonic_x_knots, isotonic_y_knots: Transport map (→ p_up calibration)
        #   - nu_effective: GPD-implied effective ν (→ MC tail heaviness)
        # =========================================================================
        "twsc_scale_applied": _twsc_applied,
        "twsc_scale_factor": float(_twsc_factor),
        "calibration_params": (tuned_params.get('global') or {}).get('calibration_params', {}) if tuned_params else {},
    }
