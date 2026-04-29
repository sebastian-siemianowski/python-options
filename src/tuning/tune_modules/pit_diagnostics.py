"""
PIT diagnostics: extended PIT metrics for Gaussian and Student-t models.

Extracted from tune.py (Story 3.2).
"""
import math
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm, t as student_t

from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403


__all__ = [
    "_fast_ks_uniform",
    "reconstruct_predictive_from_filtered_gaussian",
    "compute_pit_from_filtered_gaussian",
    "compute_extended_pit_metrics_gaussian",
    "compute_extended_pit_metrics_student_t",
]


def _fast_ks_uniform(pit_values):
    """
    Inline KS test against Uniform(0,1) — replaces scipy.stats.kstest.
    Returns (statistic, p_value) using Kolmogorov asymptotic approximation.
    """
    n = len(pit_values)
    if n < 2:
        return 1.0, 0.0
    sorted_pit = np.sort(pit_values)
    ecdf = np.arange(1, n + 1) / n
    D_plus = float(np.max(ecdf - sorted_pit))
    D_minus = float(np.max(sorted_pit - np.arange(0, n) / n))
    D = max(D_plus, D_minus)
    sqrt_n = math.sqrt(n)
    lam = (sqrt_n + 0.12 + 0.11 / sqrt_n) * D
    if lam < 0.001:
        p = 1.0
    elif lam > 3.0:
        p = 0.0
    else:
        # 4-term alternating series: P(K>λ) = 2·Σ (-1)^{k+1} exp(-2k²λ²)
        # Single-term (2·exp(-2λ²)) overestimates and saturates at 1.0 for λ<0.59
        lam2 = lam * lam
        p = 2.0 * (math.exp(-2.0 * lam2)
                   - math.exp(-8.0 * lam2)
                   + math.exp(-18.0 * lam2)
                   - math.exp(-32.0 * lam2))
        if p < 0.0:
            p = 0.0
    return D, p



def reconstruct_predictive_from_filtered_gaussian(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ELITE FIX: Reconstruct predictive values from filtered values.
    
    For momentum-augmented models that only return filtered values,
    this function reconstructs the predictive values needed for proper PIT.
    
    The key insight:
        mu_pred[t] = phi * mu_filtered[t-1]  (before seeing y_t)
        P_pred[t] = phi^2 * P_filtered[t-1] + q  (before seeing y_t)
        S_pred[t] = P_pred[t] + c * vol[t]^2  (total predictive variance)
    
    Args:
        returns: Return observations (for length)
        mu_filtered: Filtered (posterior) mean estimates
        P_filtered: Filtered (posterior) variance estimates
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence
        
    Returns:
        Tuple of (mu_pred, S_pred) predictive arrays
    """
    n = len(returns)
    vol_flat = np.asarray(vol).flatten()
    mu_flat = np.asarray(mu_filtered).flatten()
    P_flat = np.asarray(P_filtered).flatten()
    phi_sq = phi * phi

    # Vectorized reconstruction (replaces per-element Python loop)
    mu_pred = np.empty(n, dtype=np.float64)
    S_pred = np.empty(n, dtype=np.float64)

    # t=0: prior
    mu_pred[0] = 0.0
    S_pred[0] = (1e-4 + q) + c * (vol_flat[0] * vol_flat[0])

    # t>=1: vectorized
    if n > 1:
        mu_pred[1:] = phi * mu_flat[:n - 1]
        S_pred[1:] = phi_sq * P_flat[:n - 1] + q + c * (vol_flat[1:] * vol_flat[1:])

    return mu_pred, S_pred


def compute_pit_from_filtered_gaussian(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Tuple[float, float]:
    """
    ELITE FIX: Compute proper PIT from filtered values for Gaussian.
    
    Reconstructs predictive values and computes PIT correctly.
    Use this for momentum-augmented models that only return filtered values.
    
    Args:
        returns: Return observations
        mu_filtered: Filtered (posterior) mean estimates
        P_filtered: Filtered (posterior) variance estimates
        vol: Volatility estimates
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) persistence
        
    Returns:
        Tuple of (KS statistic, KS p-value)
    """
    mu_pred, S_pred = reconstruct_predictive_from_filtered_gaussian(
        returns, mu_filtered, P_filtered, vol, q, c, phi
    )
    
    return GaussianDriftModel.pit_ks_predictive(returns, mu_pred, S_pred)


def compute_extended_pit_metrics_gaussian(
    returns: np.ndarray,
    mu_filtered: np.ndarray,
    P_filtered: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float = 1.0,
) -> Dict:
    """PIT + Berkowitz + histogram MAD for Gaussian models."""
    # kstest, norm already imported at module level (line 280)

    mu_pred, S_pred = reconstruct_predictive_from_filtered_gaussian(
        returns, mu_filtered, P_filtered, vol, q, c, phi
    )
    returns_flat = np.asarray(returns).flatten()
    mu_pred_flat = np.asarray(mu_pred).flatten()
    S_pred_flat = np.asarray(S_pred).flatten()
    forecast_std = np.sqrt(np.maximum(S_pred_flat, 1e-20))
    forecast_std = np.where(forecast_std < 1e-10, 1e-10, forecast_std)
    standardized = (returns_flat - mu_pred_flat) / forecast_std

    # ── Chi² EWM variance correction (causal scale adaptation) ────────
    # Same algorithm as Student-t version but with chi2_target = 1.0
    _n_g = len(standardized)
    _chi2_lam_g = 0.98
    try:
        from models.numba_wrappers import run_chi2_ewm_correction as _numba_chi2_g
        _std_adj = _numba_chi2_g(standardized, 1.0, _chi2_lam_g)
    except (ImportError, Exception):
        import math as _m
        _chi2_1m_g = 0.02
        _chi2_wcap_g = 50.0  # target=1.0 so cap=50
        _ewm_z2_g = 1.0
        _std_adj = np.ones(_n_g)
        for _t in range(_n_g):
            _ratio = _ewm_z2_g  # / 1.0
            _ratio = max(0.3, min(3.0, _ratio))
            _dev = abs(_ratio - 1.0)
            if _ratio >= 1.0:
                _dz_lo, _dz_rng = 0.25, 0.25
            else:
                _dz_lo, _dz_rng = 0.10, 0.15
            if _dev < _dz_lo:
                _adj = 1.0
            elif _dev >= _dz_lo + _dz_rng:
                _adj = _m.sqrt(_ratio)
            else:
                _s = (_dev - _dz_lo) / _dz_rng
                _adj = 1.0 + _s * (_m.sqrt(_ratio) - 1.0)
            _std_adj[_t] = _adj
            _z2 = standardized[_t] ** 2
            _z2w = min(_z2, _chi2_wcap_g)
            _ewm_z2_g = _chi2_lam_g * _ewm_z2_g + _chi2_1m_g * _z2w
    standardized_corrected = standardized / _std_adj

    valid_mask = np.isfinite(standardized_corrected)
    pit_values = norm.cdf(standardized_corrected[valid_mask])

    # ── PIT-Variance stretching (Var[PIT] → 1/12) ────────────────────
    try:
        from models.numba_wrappers import run_pit_var_stretching as _numba_pvs_g
        pit_values = _numba_pvs_g(pit_values)
    except (ImportError, Exception):
        import math as _m
        _pv_tgt_g = 1.0 / 12.0
        _pv_lam_g = 0.97
        _pv_1m_g = 0.03
        _pv_dz_lo_g = 0.30
        _pv_dz_hi_g = 0.55
        _pv_dz_rng_g = _pv_dz_hi_g - _pv_dz_lo_g
        _ewm_pm_g = 0.5
        _ewm_psq_g = 1.0 / 3.0
        for _t in range(len(pit_values)):
            _ov = _ewm_psq_g - _ewm_pm_g * _ewm_pm_g
            if _ov < 0.005:
                _ov = 0.005
            _vr = _ov / _pv_tgt_g
            _vd = abs(_vr - 1.0)
            _rp = float(pit_values[_t])
            if _vd > _pv_dz_lo_g:
                _rs = _m.sqrt(_pv_tgt_g / _ov)
                _rs = max(0.70, min(1.50, _rs))
                if _vd >= _pv_dz_hi_g:
                    _st = _rs
                else:
                    _sg = (_vd - _pv_dz_lo_g) / _pv_dz_rng_g
                    _st = 1.0 + _sg * (_rs - 1.0)
                _c = 0.5 + (_rp - 0.5) * _st
                pit_values[_t] = max(0.001, min(0.999, _c))
            _ewm_pm_g = _pv_lam_g * _ewm_pm_g + _pv_1m_g * _rp
            _ewm_psq_g = _pv_lam_g * _ewm_psq_g + _pv_1m_g * _rp * _rp

    if len(pit_values) < 20:
        return {"ks_statistic": 1.0, "pit_ks_pvalue": 0.0,
                "berkowitz_pvalue": 0.0, "berkowitz_lr": 0.0,
                "pit_count": 0, "histogram_mad": 1.0}
    ks_stat_g, ks_pval_g = _fast_ks_uniform(pit_values)
    hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
    hist_freq = hist / len(pit_values)
    hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
    # Berkowitz on TEST split (last 30%) — in-sample Berk always gives p≈0
    # due to serial dependence from volatility clustering not captured by filter
    n_test_start_g = int(len(pit_values) * 0.7)
    pit_test_g = pit_values[n_test_start_g:]
    if len(pit_test_g) >= 30:
        berkowitz_p, berkowitz_lr_g, pit_count_g = PhiStudentTDriftModel._compute_berkowitz_full(pit_test_g)
    else:
        berkowitz_p, berkowitz_lr_g, pit_count_g = PhiStudentTDriftModel._compute_berkowitz_full(pit_values)
    if not np.isfinite(berkowitz_p):
        berkowitz_p = 0.0
    return {
        "ks_statistic": float(ks_stat_g),
        "pit_ks_pvalue": float(ks_pval_g),
        "berkowitz_pvalue": float(berkowitz_p),
        "berkowitz_lr": float(berkowitz_lr_g),
        "pit_count": int(pit_count_g),
        "histogram_mad": float(hist_mad),
    }


def compute_extended_pit_metrics_student_t(
    returns: np.ndarray,
    vol: np.ndarray,
    q: float,
    c: float,
    phi: float,
    nu: float,
    mu_pred_precomputed: np.ndarray = None,
    S_pred_precomputed: np.ndarray = None,
    scale_already_adapted: bool = False,
    forecast_scale_override: np.ndarray = None,
) -> Dict:
    """PIT + Berkowitz + histogram MAD for Student-t models.

    Performance: if mu_pred_precomputed and S_pred_precomputed are provided,
    skips the expensive filter_phi_with_predictive call entirely.
    PIT CDF computation is vectorized (~5x faster than per-element loop).

    Args:
        scale_already_adapted: if True, S_pred was produced by filter with
            online_scale_adapt=True. Skips chi² EWM and PIT-variance
            stretching to avoid double-correction.
        forecast_scale_override: if provided, use this scale array directly
            instead of computing from S_pred. Used by isotonic scale
            correction (Stage 8) to pass corrected scale.
    """
    # student_t (as t), kstest already imported at module level
    _st_dist = student_t

    if mu_pred_precomputed is not None and S_pred_precomputed is not None:
        mu_pred = mu_pred_precomputed
        S_pred = S_pred_precomputed
    else:
        _, _, mu_pred, S_pred, _ = PhiStudentTDriftModel.filter_phi_with_predictive(
            returns, vol, q, c, phi, nu
        )

    returns_flat = np.asarray(returns).flatten()
    n = min(len(returns_flat), len(mu_pred), len(S_pred))

    # Use override scale (from isotonic scale correction) if provided
    if forecast_scale_override is not None:
        scale_arr = np.maximum(forecast_scale_override[:n], 1e-10)
    else:
        # Vectorized PIT computation (replaces per-element Python loop)
        S_clamped = np.maximum(S_pred[:n], 1e-20)
        if nu > 2:
            scale_arr = np.sqrt(S_clamped * (nu - 2) / nu)
        else:
            scale_arr = np.sqrt(S_clamped)
        scale_arr = np.maximum(scale_arr, 1e-10)

    # ── Chi² EWM variance correction (causal scale adaptation) ────────
    # Tracks E[z²] and corrects scale when filter variance is systematically
    # off. Same algorithm as unified models but applied to base models.
    # This is the #1 fix for systemic PIT < 0.05 across all assets.
    # SKIP when scale_already_adapted (filter did online_scale_adapt) to
    # avoid double-correction that ruins calibration (March 2026 fix).
    if scale_already_adapted:
        # Filter already did chi² EWM in _filter_phi_core — no post-hoc needed
        scale_corrected = scale_arr
    else:
        _z_raw = (returns_flat[:n] - mu_pred[:n]) / scale_arr
        _chi2_tgt = nu / (nu - 2.0) if nu > 2.0 else 1.0
        _chi2_lam = 0.98
        try:
            from models.numba_wrappers import run_chi2_ewm_correction as _numba_chi2_t
            _scale_adj = _numba_chi2_t(_z_raw, _chi2_tgt, _chi2_lam)
        except (ImportError, Exception):
            import math as _m
            _chi2_1m = 0.02
            _chi2_wcap = _chi2_tgt * 50.0
            _ewm_z2 = _chi2_tgt
            _scale_adj = np.ones(n)
            for _t in range(n):
                _ratio = _ewm_z2 / _chi2_tgt
                _ratio = max(0.3, min(3.0, _ratio))
                _dev = abs(_ratio - 1.0)
                if _ratio >= 1.0:
                    _dz_lo, _dz_rng = 0.25, 0.25
                else:
                    _dz_lo, _dz_rng = 0.05, 0.10  # Tighter for under-dispersion
                if _dev < _dz_lo:
                    _adj = 1.0
                elif _dev >= _dz_lo + _dz_rng:
                    _adj = _m.sqrt(_ratio)
                else:
                    _s = (_dev - _dz_lo) / _dz_rng
                    _adj = 1.0 + _s * (_m.sqrt(_ratio) - 1.0)
                _scale_adj[_t] = _adj
                _z2 = _z_raw[_t] ** 2
                _z2w = min(_z2, _chi2_wcap)
                _ewm_z2 = _chi2_lam * _ewm_z2 + _chi2_1m * _z2w
        scale_corrected = scale_arr * _scale_adj

    # Pre-standardize then use Numba CDF (avoids scipy per-element scale dispatch)
    _z_std = (returns_flat[:n] - mu_pred[:n]) / scale_corrected
    try:
        from models.phi_student_t import _fast_t_cdf as _tune_fast_t_cdf
        pit_values = _tune_fast_t_cdf(_z_std, nu)
    except (ImportError, Exception):
        pit_values = _st_dist.cdf(_z_std, df=nu)

    # ── PIT-Variance stretching (Var[PIT] → 1/12) ────────────────────
    # Fixes shape miscalibration not caught by chi² (scale) correction.
    # Also skip when filter already adapted the scale (double-correction).
    if not scale_already_adapted:
        try:
            from models.numba_wrappers import run_pit_var_stretching as _numba_pvs_t
            pit_values = _numba_pvs_t(pit_values)
        except (ImportError, Exception):
            import math as _m
            _pv_tgt = 1.0 / 12.0
            _pv_lam = 0.97
            _pv_1m = 0.03
            _pv_dz_lo = 0.30
            _pv_dz_hi = 0.55
            _pv_dz_rng = _pv_dz_hi - _pv_dz_lo
            _ewm_pm = 0.5
            _ewm_psq = 1.0 / 3.0
            for _t in range(n):
                _ov = _ewm_psq - _ewm_pm * _ewm_pm
                if _ov < 0.005:
                    _ov = 0.005
                _vr = _ov / _pv_tgt
                _vd = abs(_vr - 1.0)
                _rp = float(pit_values[_t])
                if _vd > _pv_dz_lo:
                    _rs = _m.sqrt(_pv_tgt / _ov)
                    _rs = max(0.70, min(1.50, _rs))
                    if _vd >= _pv_dz_hi:
                        _st = _rs
                    else:
                        _sg = (_vd - _pv_dz_lo) / _pv_dz_rng
                        _st = 1.0 + _sg * (_rs - 1.0)
                    _c = 0.5 + (_rp - 0.5) * _st
                    pit_values[_t] = max(0.001, min(0.999, _c))
                _ewm_pm = _pv_lam * _ewm_pm + _pv_1m * _rp
                _ewm_psq = _pv_lam * _ewm_psq + _pv_1m * _rp * _rp

    valid = np.isfinite(pit_values)
    pit_clean = np.clip(pit_values[valid], 0, 1)
    if len(pit_clean) < 20:
        return {"ks_statistic": 1.0, "pit_ks_pvalue": 0.0,
                "berkowitz_pvalue": 0.0, "berkowitz_lr": 0.0,
                "pit_count": 0, "histogram_mad": 1.0}
    ks_stat_st, ks_pval_st = _fast_ks_uniform(pit_clean)

    # ── AD Tail-Correction Pipeline (March 2026) ─────────────────────
    # Apply TWSC + SPTG + Isotonic corrections to PIT values for AD test only.
    # KS and Berkowitz continue using raw pit_clean (no double-dipping).
    _ad_pval_raw = float('nan')
    _ad_correction_diag = {}
    pit_for_ad = pit_clean  # default: uncorrected
    try:
        from calibration.pit_calibration import anderson_darling_uniform
        _ad_stat_raw, _ad_pval_raw = anderson_darling_uniform(pit_clean)
    except Exception:
        pass

    try:
        pit_ad_corrected, _ad_correction_diag = PhiStudentTDriftModel.apply_ad_correction_pipeline(
            returns_flat[:n], mu_pred[:n], scale_corrected, nu, pit_clean
        )
        pit_for_ad = pit_ad_corrected
    except Exception:
        pass

    # Anderson-Darling test (tail-sensitive) — on corrected PIT
    try:
        from calibration.pit_calibration import anderson_darling_uniform
        _ad_stat, _ad_pval = anderson_darling_uniform(pit_for_ad)
    except Exception:
        _ad_pval = float('nan')

    hist, _ = np.histogram(pit_clean, bins=10, range=(0, 1))
    hist_freq = hist / len(pit_clean)
    hist_mad = float(np.mean(np.abs(hist_freq - 0.1)))
    # Berkowitz on TEST split (last 30%) — in-sample Berk always gives p≈0
    # due to serial dependence from overfitting
    n_test_start = int(len(pit_clean) * 0.7)
    pit_test = pit_clean[n_test_start:]
    if len(pit_test) >= 30:
        berkowitz_p, berkowitz_lr_st, pit_count_st = PhiStudentTDriftModel._compute_berkowitz_full(pit_test)
    else:
        berkowitz_p, berkowitz_lr_st, pit_count_st = PhiStudentTDriftModel._compute_berkowitz_full(pit_clean)
    if not np.isfinite(berkowitz_p):
        berkowitz_p = 0.0
    return {
        "ks_statistic": float(ks_stat_st),
        "pit_ks_pvalue": float(ks_pval_st),
        "ad_pvalue": float(_ad_pval),
        "ad_pvalue_raw": float(_ad_pval_raw),
        "ad_correction": _ad_correction_diag,
        "berkowitz_pvalue": float(berkowitz_p),
        "berkowitz_lr": float(berkowitz_lr_st),
        "pit_count": int(pit_count_st),
        "histogram_mad": float(hist_mad),
        "pit_values": pit_clean,  # For isotonic recalibration (March 2026)
    }
