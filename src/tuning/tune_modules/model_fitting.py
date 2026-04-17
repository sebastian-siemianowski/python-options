"""
Model fitting: fit_all_models_for_regime, elite diagnostics, q/c/phi MLE.

Extracted from tune.py (Story 4.1). This is the computational heart of the
tuning pipeline -- the 1,040-line model fitting loop that orchestrates all
14+ model variants across different noise distributions.
"""
import math
import os
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import norm, t as student_t

from tuning.tune_modules.config import *  # noqa: F401,F403
from tuning.tune_modules.utilities import *  # noqa: F401,F403


__all__ = [
    "optimize_q_c_phi_mle",
    "compute_elite_diagnostics",
    "format_elite_status",
    "fit_all_models_for_regime",
]


def optimize_q_c_phi_mle(
    returns: np.ndarray,
    vol: np.ndarray,
    train_frac: float = 0.7,
    q_min: float = 1e-10,
    q_max: float = 1e-1,
    c_min: float = 0.3,
    c_max: float = 3.0,
    phi_min: float = -0.999,
    phi_max: float = 0.999,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0
) -> Tuple[float, float, float, float, Dict]:
    """Delegate φ-Gaussian optimization to PhiGaussianDriftModel for modularity."""
    return PhiGaussianDriftModel.optimize_params(
        returns=returns,
        vol=vol,
        train_frac=train_frac,
        q_min=q_min,
        q_max=q_max,
        c_min=c_min,
        c_max=c_max,
        phi_min=phi_min,
        phi_max=phi_max,
        prior_log_q_mean=prior_log_q_mean,
        prior_lambda=prior_lambda,
    )


# =============================================================================
# ELITE TUNING DIAGNOSTICS HELPER (v2.0 - February 2026)
# =============================================================================
# Computes elite tuning diagnostics after parameter optimization.
# 
# TOP 0.001% UPGRADES:
#   1. DIRECTIONAL CURVATURE: φ-q coupling penalized 2× more than benign couplings
#   2. RIDGE DETECTION: Distinguishes flat basins from narrow ridges
#   3. ASYMMETRIC COHERENCE: Drift penalized harder than oscillation
#   4. PURE FRAGILITY: No calibration leakage into tuning
#
# INSTITUTIONAL ALIGNMENT:
#   Top funds don't just prefer flat optima; they explicitly reject optima
#   that are only flat along fragile directions.
# =============================================================================

def compute_elite_diagnostics(
    objective_fn,
    optimal_params: np.ndarray,
    optimal_value: float,
    bounds: List[Tuple[float, float]],
    param_names: Optional[List[str]] = None,
    fold_optimal_params: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive elite tuning diagnostics (v2.0).
    
    This function computes stability metrics that distinguish top 0.001% funds:
    - Curvature penalty with directional weighting (φ-q coupling danger)
    - Ridge vs basin detection via joint perturbations
    - Drift vs noise decomposition in cross-fold coherence
    - Pure parameter fragility index (no calibration leakage)
    
    Args:
        objective_fn: Objective function to evaluate (neg penalized LL)
        optimal_params: Optimized parameter vector
        optimal_value: Objective value at optimum
        bounds: Parameter bounds [(lo, hi), ...]
        param_names: Optional parameter names for diagnostics
        fold_optimal_params: Optional list of per-fold optimal params
        
    Returns:
        Dictionary with elite diagnostics including:
        - condition_number: Hessian condition number
        - curvature_penalty: Soft penalty for sharp optima
        - directional_curvature_penalty: Weighted by coupling danger
        - plateau_score: Width of acceptable region
        - basin_score: 1.0 = flat basin, 0.0 = narrow ridge
        - is_ridge_optimum: True if ridge detected (dangerous)
        - drift_component: Variance from persistent drift
        - noise_component: Variance from random oscillation
        - fragility_index: Composite fragility score
        - fragility_warning: True if fragility > threshold
        - fragility_components: Per-dimension breakdown
    """
    if not ELITE_TUNING_AVAILABLE or not ELITE_TUNING_ENABLED:
        return {'elite_tuning_enabled': False}
    
    diagnostics = {
        'elite_tuning_version': '2.0',
        'elite_tuning_enabled': True,
    }
    
    n_params = len(optimal_params)
    param_ranges = np.array([b[1] - b[0] for b in bounds])
    
    # Get elite tuning config
    try:
        config = create_elite_tuning_config(ELITE_TUNING_PRESET)
    except Exception:
        config = EliteTuningConfig()
    
    # =========================================================================
    # 1. HESSIAN-BASED CURVATURE ANALYSIS (with directional weighting)
    # =========================================================================
    try:
        H = compute_hessian_finite_diff(
            objective_fn, optimal_params, 
            config.hessian_epsilon, bounds
        )
        
        if config.enable_directional_curvature:
            # v2.0: Directional curvature - dangerous couplings penalized more
            penalty, cond_num, eigenvalues, coupling_fragility = compute_directional_curvature_penalty(
                H, config.max_condition_number
            )
            diagnostics['directional_curvature_penalty'] = float(penalty)
            diagnostics['dangerous_coupling_fragility'] = coupling_fragility
        else:
            # Legacy: Standard curvature penalty
            penalty, cond_num, eigenvalues = compute_curvature_penalty(
                H, config.max_condition_number
            )
        
        diagnostics['condition_number'] = float(cond_num) if np.isfinite(cond_num) else None
        diagnostics['curvature_penalty'] = float(penalty)
        diagnostics['eigenvalues'] = eigenvalues.tolist() if len(eigenvalues) > 0 else []
        
    except Exception as e:
        diagnostics['condition_number'] = None
        diagnostics['curvature_penalty'] = 0.0
        diagnostics['curvature_error'] = str(e)
    
    # =========================================================================
    # 2. CONNECTED PLATEAU ANALYSIS (basin vs ridge detection)
    # =========================================================================
    try:
        if config.enable_ridge_detection:
            # v2.0: Detect ridges via joint perturbations
            plateau_width, plateau_score, is_isolated, basin_score, survival_rate = evaluate_connected_plateau(
                objective_fn,
                optimal_params,
                optimal_value,
                bounds,
                config.plateau_acceptance_ratio,
                n_joint_perturbations=config.n_joint_perturbations,
                ridge_threshold=config.ridge_threshold
            )
            diagnostics['is_ridge_optimum'] = basin_score < config.ridge_threshold
            diagnostics['basin_score'] = float(basin_score)
            diagnostics['joint_perturbation_survival_rate'] = float(survival_rate)
        else:
            # Legacy: Axis-aligned plateau only
            from tuning.elite_tuning import evaluate_plateau_width
            plateau_width, plateau_score, is_isolated = evaluate_plateau_width(
                objective_fn, optimal_params, optimal_value, bounds,
                config.plateau_acceptance_ratio
            )
            basin_score = plateau_score
            diagnostics['is_ridge_optimum'] = False
            diagnostics['basin_score'] = float(basin_score)
        
        diagnostics['plateau_score'] = float(plateau_score)
        diagnostics['plateau_width'] = plateau_width.tolist()
        diagnostics['is_isolated_optimum'] = bool(is_isolated)
        
    except Exception as e:
        diagnostics['plateau_score'] = 0.5
        diagnostics['plateau_width'] = []
        diagnostics['is_isolated_optimum'] = False
        diagnostics['basin_score'] = 1.0
        diagnostics['is_ridge_optimum'] = False
        diagnostics['plateau_error'] = str(e)
        basin_score = 1.0
        plateau_score = 0.5
    
    # =========================================================================
    # 3. CROSS-FOLD COHERENCE (with drift vs noise decomposition)
    # =========================================================================
    coherence_variance = np.array([])
    drift_ratio = 0.0
    
    if fold_optimal_params and len(fold_optimal_params) >= config.min_folds_for_coherence:
        try:
            if config.enable_drift_penalty:
                # v2.0: Asymmetric coherence - drift is worse than oscillation
                penalty, variance, drift_comp, noise_comp, drift_pen = compute_asymmetric_coherence_penalty(
                    fold_optimal_params,
                    param_ranges,
                    config.drift_penalty_multiplier
                )
                diagnostics['drift_component'] = drift_comp.tolist()
                diagnostics['noise_component'] = noise_comp.tolist()
                diagnostics['drift_penalty'] = float(drift_pen)
                
                # Compute drift ratio for fragility
                total_var = np.sum(drift_comp) + np.sum(noise_comp)
                if total_var > 1e-12:
                    drift_ratio = np.sum(drift_comp) / total_var
                diagnostics['drift_ratio'] = float(drift_ratio)
            else:
                # Legacy: Symmetric coherence
                penalty, variance = compute_coherence_penalty(
                    fold_optimal_params, param_ranges
                )
            
            coherence_variance = variance
            diagnostics['coherence_penalty'] = float(penalty)
            diagnostics['parameter_variance_across_folds'] = variance.tolist()
            diagnostics['n_folds_evaluated'] = len(fold_optimal_params)
            
        except Exception as e:
            diagnostics['coherence_error'] = str(e)
    
    # =========================================================================
    # 4. FRAGILITY INDEX (PURE PARAMETER FRAGILITY - v2.0)
    # =========================================================================
    try:
        fragility, components = compute_fragility_index(
            condition_number=diagnostics.get('condition_number', 1.0) or 1.0,
            coherence_variance=coherence_variance,
            plateau_width=np.array(diagnostics.get('plateau_width', [])),
            basin_score=basin_score,  # v2.0: ridge awareness
            drift_ratio=drift_ratio,  # v2.0: drift awareness
            # NO calibration_ks_stat - pure parameter fragility only
        )
        
        diagnostics['fragility_index'] = float(fragility)
        diagnostics['fragility_warning'] = fragility > config.fragility_threshold
        diagnostics['fragility_components'] = components
        diagnostics['fragility_threshold'] = config.fragility_threshold
        
    except Exception as e:
        diagnostics['fragility_index'] = 0.5
        diagnostics['fragility_warning'] = False
        diagnostics['fragility_error'] = str(e)
    
    return diagnostics


def format_elite_status(diagnostics: Dict[str, Any]) -> str:
    """
    Format elite tuning status for logging.
    
    Returns a compact string summarizing elite diagnostics.
    """
    if not diagnostics.get('elite_tuning_enabled', False):
        return "elite:disabled"
    
    parts = []
    
    # Fragility
    frag = diagnostics.get('fragility_index', 0)
    if frag > 0.5:
        parts.append(f"⚠frag={frag:.2f}")
    else:
        parts.append(f"frag={frag:.2f}")
    
    # Ridge warning
    if diagnostics.get('is_ridge_optimum', False):
        parts.append("⚠RIDGE")
    
    # Basin score
    basin = diagnostics.get('basin_score', 1.0)
    if basin < 0.5:
        parts.append(f"basin={basin:.2f}")
    
    # Condition number
    cond = diagnostics.get('condition_number')
    if cond and cond > 1e6:
        parts.append(f"κ={cond:.1e}")
    
    return " | ".join(parts) if parts else "elite:ok"


# =============================================================================
# MODEL SELECTION FUNCTIONS — IMPORTED FROM src/calibration/model_selection.py
# =============================================================================
# The following functions are imported at the top of this file:
#   - compute_bic_model_weights_from_scores()
#   - compute_bic_model_weights()
#   - apply_temporal_smoothing()
#   - normalize_weights()
#   - DEFAULT_MODEL_SELECTION_METHOD
#   - DEFAULT_BIC_WEIGHT
#   - DEFAULT_TEMPORAL_ALPHA
# =============================================================================


def fit_all_models_for_regime(
    returns: np.ndarray,
    vol: np.ndarray,
    prior_log_q_mean: float = -6.0,
    prior_lambda: float = 1.0,
    prices: np.ndarray = None,  # Added for MR integration (February 2026)
    regime_labels: np.ndarray = None,  # Added for regime-adaptive blending
    asset: str = None,  # FIX #4: Asset symbol for c-bounds detection
    gk_c_prior_value: float = None,  # Story 2.2: GK-informed c prior
) -> Dict[str, Dict]:
    """
    Fit ALL candidate model classes for a single regime's data.

    For each model m, computes:
        - Tuned parameters θ_{r,m}
        - Full log-likelihood
        - BIC, AIC
        - PIT calibration diagnostics

    DISCRETE ν GRID FOR STUDENT-T:
    We fit separate Student-t sub-models for each ν in STUDENT_T_NU_GRID.
    Each sub-model participates independently in BMA, eliminating ν-σ identifiability issues.

    Args:
        returns: Regime-specific returns
        vol: Regime-specific volatility
        prior_log_q_mean: Prior mean for log10(q)
        prior_lambda: Regularization strength
        prices: Price array for MR equilibrium estimation (optional)
        regime_labels: Regime labels for adaptive blending (optional)
        asset: Asset symbol for c-bounds detection (FIX #4, optional)

    Returns:
        Dictionary with fitted models:
        {
            "kalman_gaussian": {...},
            "kalman_phi_gaussian": {...},
            "phi_student_t_nu_4": {...},
            "phi_student_t_nu_6": {...},
            "phi_student_t_nu_8": {...},
            "phi_student_t_nu_12": {...},
            "phi_student_t_nu_20": {...},
        }
    """
    n_obs = len(returns)
    models = {}

    # =========================================================================
    # Phi Lower Bound from Autocorrelation (Story 1.8)
    # =========================================================================
    # Compute empirical lag-1 autocorrelation of returns and derive a
    # principled lower bound for phi.  If autocorrelation is positive,
    # phi should be positive — preventing the anti-persistent phi values
    # that signals.py currently floors at 0.0 post-hoc.
    #
    # phi_floor = max(0, acf_1 * PHI_ACF_SCALE)
    # =========================================================================
    _acf_1 = 0.0
    _phi_acf_floor = 0.0
    if n_obs > 10:
        _r = np.asarray(returns).flatten()
        _acf_corr = np.corrcoef(_r[:-1], _r[1:])
        if _acf_corr.shape == (2, 2) and np.isfinite(_acf_corr[0, 1]):
            _acf_1 = float(_acf_corr[0, 1])
        _phi_acf_floor = max(0.0, _acf_1 * PHI_ACF_SCALE)

    # =========================================================================
    # FILTER RESULT CACHE (February 2026 Performance Optimization)
    # =========================================================================
    # Cache filter results to avoid redundant filter runs when creating
    # momentum/enhanced variants from base models.
    #
    # KEY: (model_name, q, c, phi) -> (mu, P, ll, lfo_cv_score)
    # 
    # PERFORMANCE IMPACT: ~30% speedup by avoiding redundant filter calls
    # =========================================================================
    _filter_cache = {}

    def _cache_key(model_name: str, q: float, c: float, phi: float) -> str:
        """Generate unique cache key for filter results."""
        return f"{model_name}_{q:.8e}_{c:.6f}_{phi:.6f}"

    # Optionally, you can import or define run_student_t_filter_with_lfo_cv if available
    try:
        from models.phi_student_t import run_student_t_filter_with_lfo_cv
    except ImportError:
        run_student_t_filter_with_lfo_cv = None

    # Import Gaussian LFO-CV wrapper for Phase 1 Gaussian models
    try:
        from models.numba_wrappers import run_gaussian_filter_with_lfo_cv
    except ImportError:
        run_gaussian_filter_with_lfo_cv = None

    def _get_cached_or_filter(
        model_name: str, 
        q: float, 
        c: float, 
        phi: float, 
        nu: float,
        with_lfo_cv: bool = True,
    ) -> tuple:
        """Get filter results from cache or compute and cache them."""
        key = _cache_key(model_name, q, c, phi)

        if key in _filter_cache:
            return _filter_cache[key]

        # Determine if this is a Gaussian model (nu is None or model name indicates it)
        _is_gaussian = (nu is None or "gaussian" in model_name.lower())

        # Use fused LFO-CV filter when available (100-200x faster than pure Python)
        if with_lfo_cv:
            if _is_gaussian and run_gaussian_filter_with_lfo_cv is not None:
                try:
                    result = run_gaussian_filter_with_lfo_cv(
                        returns, vol, q, c, phi,
                        lfo_start_frac=LFO_CV_MIN_TRAIN_FRAC,
                    )
                    _filter_cache[key] = result
                    return result
                except Exception:
                    pass
            elif not _is_gaussian and run_student_t_filter_with_lfo_cv is not None:
                try:
                    result = run_student_t_filter_with_lfo_cv(
                        returns, vol, q, c, phi, nu,
                        lfo_start_frac=LFO_CV_MIN_TRAIN_FRAC
                    )
                    _filter_cache[key] = result
                    return result
                except Exception:
                    pass

        # Fallback to standard filter
        if _is_gaussian:
            from models.phi_gaussian import PhiGaussianDriftModel
            mu, P, ll = PhiGaussianDriftModel.filter(
                returns, vol, q, c, phi
            )
        else:
            mu, P, ll = PhiStudentTDriftModel.filter_phi(
                returns, vol, q, c, phi, nu
            )
        lfo_cv_score = float('-inf')  # Not computed in fallback
        result = (mu, P, ll, lfo_cv_score)
        _filter_cache[key] = result
        return result

    # =========================================================================
    # Model 1: Phi-Student-t with DISCRETE ν GRID
    # =========================================================================
    # Delegates fully to PhiStudentTDriftModel.tune_and_calibrate() which
    # encapsulates: MLE → filter → PIT/CRPS → GARCH blending → ν-refinement
    # → GAS ν → CRPS shrinkage → momentum augmentation → isotonic recal.
    # =========================================================================

    # VoV precomputation — shared across all ν (data-only, O(n))
    _vov_rolling = PhiStudentTDriftModel._precompute_vov(vol)
    _gamma_vov_fixed = 0.3

    # Pre-create momentum wrapper for internal CRPS comparison (if available)
    _st_momentum_wrapper = None
    if MOMENTUM_AUGMENTATION_ENABLED and MOMENTUM_AUGMENTATION_AVAILABLE:
        _st_momentum_wrapper = MomentumAugmentedDriftModel(DEFAULT_MOMENTUM_CONFIG)
        if prices is not None:
            _st_momentum_wrapper.precompute_signals(
                returns=returns, prices=prices, vol=vol,
                regime_labels=regime_labels, q=1e-6,
            )
        else:
            _st_momentum_wrapper.precompute_momentum(returns)

    _st_warm_start = None  # Cross-ν warm-starting (March 2026)
    for nu_fixed in STUDENT_T_NU_GRID:
        model_name = f"phi_student_t_nu_{nu_fixed}"
        try:
            models[model_name] = PhiStudentTDriftModel.tune_and_calibrate(
                returns, vol, nu_fixed,
                prior_log_q_mean=prior_log_q_mean,
                prior_lambda=prior_lambda,
                asset_symbol=asset,
                n_obs=n_obs,
                n_train=int(n_obs * 0.7),
                vov_rolling=_vov_rolling,
                gamma_vov=_gamma_vov_fixed,
                momentum_wrapper=_st_momentum_wrapper,
                prices=prices,
                regime_labels=regime_labels,
                warm_start_params=_st_warm_start,
            )
            # Extract optimised (q, c, φ) for warm-starting next ν
            _m = models[model_name]
            if _m.get("fit_success", False):
                _st_warm_start = (_m.get("q", 1e-5), _m.get("c", 1.0), _m.get("phi", 0.0))
        except Exception as e:
            models[model_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "crps": float('inf'),
                "nu": float(nu_fixed),
                "nu_fixed": True,
            }

    # =========================================================================
    # Model 1b: Continuous Nu Optimization (Story 1.5)
    # =========================================================================
    # Profile-likelihood refinement: hold (q, c, phi) at best discrete MLE,
    # optimise nu over [2.5, 60] via bounded Brent.  The discrete grid remains
    # the initialisation — this finds the true MLE between grid points.
    #
    # Mathematical justification:
    #   L_profile(nu) = L(q_hat, c_hat, phi_hat, nu)
    #   nu_mle = argmax_nu  L_profile(nu)
    #   se(nu) = 1 / sqrt( -d^2 log L_profile / d nu^2 |_{nu=nu_mle} )
    # =========================================================================
    _best_discrete_bic = float('inf')
    _best_discrete_name = None
    for _dn, _dm in models.items():
        if _dn.startswith("phi_student_t_nu_") and _dm.get("fit_success", False):
            _dbic = _dm.get("bic", float('inf'))
            if _dbic < _best_discrete_bic:
                _best_discrete_bic = _dbic
                _best_discrete_name = _dn

    if _best_discrete_name is not None:
        _best_dm = models[_best_discrete_name]
        _prof_q = float(_best_dm.get("q", 1e-5))
        _prof_c = float(_best_dm.get("c", 1.0))
        _prof_phi = float(_best_dm.get("phi", 0.0))
        _prof_nu_init = float(_best_dm.get("nu", 8.0))
        _prof_gamma_vov = float(_best_dm.get("gamma_vov", 0.0))

        # Profile log-likelihood: filter with fixed (q, c, phi), varying nu
        def _profile_neg_ll(nu_candidate):
            nu_c = float(np.clip(nu_candidate, NU_CONTINUOUS_BOUNDS[0], NU_CONTINUOUS_BOUNDS[1]))
            try:
                _, _, _, _, ll = PhiStudentTDriftModel.filter_phi_with_predictive(
                    returns, vol, _prof_q, _prof_c, _prof_phi, nu_c,
                    robust_wt=True,
                    gamma_vov=_prof_gamma_vov,
                    vov_rolling=_vov_rolling if _prof_gamma_vov > 1e-12 else None,
                )
                if not np.isfinite(ll):
                    return 1e12
                return -ll
            except Exception:
                return 1e12

        try:
            _nu_opt_result = minimize_scalar(
                _profile_neg_ll,
                bounds=NU_CONTINUOUS_BOUNDS,
                method='bounded',
                options={'xatol': 0.05, 'maxiter': 40},
            )
            _nu_mle = float(_nu_opt_result.x)

            # Standard error via central finite-difference Hessian
            _h = NU_SE_FD_STEP
            _ll_center = -_profile_neg_ll(_nu_mle)
            _ll_plus = -_profile_neg_ll(_nu_mle + _h)
            _ll_minus = -_profile_neg_ll(_nu_mle - _h)
            _d2ll = (_ll_plus - 2.0 * _ll_center + _ll_minus) / (_h * _h)
            if _d2ll < -1e-12:
                _nu_se = 1.0 / math.sqrt(-_d2ll)
            else:
                _nu_se = float('inf')

            # Profile LL at the optimum
            _ll_mle = -_profile_neg_ll(_nu_mle)

            # Only create the continuous model if it improves on discrete
            _n_params_mle = 4  # q, c, phi, nu (same param count as discrete)
            _bic_mle = -2.0 * _ll_mle + _n_params_mle * math.log(max(n_obs, 1))

            _nu_mle_name = "phi_student_t_nu_mle"

            # Copy best discrete model and update nu-related fields
            _mle_model = dict(_best_dm)
            _mle_model["nu"] = _nu_mle
            _mle_model["nu_mle"] = _nu_mle
            _mle_model["nu_se"] = min(_nu_se, 999.0)
            _mle_model["nu_discrete_init"] = _prof_nu_init
            _mle_model["nu_fixed"] = False
            _mle_model["log_likelihood"] = _ll_mle
            _mle_model["bic"] = _bic_mle
            _mle_model["fit_success"] = True
            _mle_model["continuous_nu_optimization"] = True

            models[_nu_mle_name] = _mle_model
        except Exception:
            pass  # Discrete grid models still available

    # =========================================================================
    # Model 2: UNIFIED Phi-Student-t (February 2026 - Elite Architecture)
    # =========================================================================
    # Consolidates 48+ model variants into 3 adaptive unified models.
    # Each unified model combines: VoV, Smooth Asymmetric ν, Probabilistic MS-q
    #
    # This replaces separate VoV/Two-Piece/Mixture/MS-q loops with a single
    # coherent architecture that auto-adapts to asset characteristics.
    #
    # Model naming: "phi_student_t_unified_nu_{nu}"
    # =========================================================================
    
    # Unified models use 3 ν flavours matching STUDENT_T_NU_GRID.
    # Internal Stage 5 re-optimizes ν across a finer grid [3..20],
    # so intermediate values (6, 10, 12, 15) are still explored —
    # we just don't spawn separate BMA models for them.
    UNIFIED_NU_GRID = [3, 4, 8, 20]
    n_params_unified = 14  # q, c, φ, γ_vov, ms_sensitivity, α_asym, ν, garch(3), rough_hurst, risk_premium, skew(2), jump(4-cond)
    
    from models.phi_student_t_unified import UnifiedPhiStudentTModel
    
    for nu_fixed in UNIFIED_NU_GRID:
        unified_name = f"phi_student_t_unified_nu_{nu_fixed}"
        
        try:
            # Staged optimization for unified model
            config, diagnostics = UnifiedPhiStudentTModel.optimize_params_unified(
                returns, vol, 
                nu_base=float(nu_fixed),
                train_frac=0.7, 
                asset_symbol=asset,
                gk_c_prior_value=gk_c_prior_value,
            )

            if not diagnostics.get("success", False):
                raise ValueError(f"Unified optimization failed: {diagnostics.get('error', 'Unknown')}")

            # Run unified filter to get predictive values
            mu_u, P_u, mu_pred_u, S_pred_u, ll_u = UnifiedPhiStudentTModel.filter_phi_unified(
                returns, vol, config
            )

            # PIT calibration using predictive distribution
            ks_u, pit_p_u, pit_metrics = UnifiedPhiStudentTModel.pit_ks_unified(
                returns, mu_pred_u, S_pred_u, config
            )

            # Information criteria — adjust param count if jump/rough layers active
            jump_active = getattr(config, 'jump_intensity', 0.0) > 1e-6 and getattr(config, 'jump_variance', 0.0) > 1e-12
            rough_active = getattr(config, 'rough_hurst', 0.0) > 0.01
            effective_n_params = n_params_unified + (4 if jump_active else 0) + (1 if rough_active else 0)
            aic_u = compute_aic(ll_u, effective_n_params)
            bic_u = compute_bic(ll_u, effective_n_params, n_obs)
            mean_ll_u = ll_u / max(n_obs, 1)

            # Hyvärinen and CRPS scores using calibrated predictive values
            # ─────────────────────────────────────────────────────────────
            # Use filter_and_calibrate to get the same calibrated sigma
            # and effective ν that the PIT test uses. This ensures tuning
            # optimises against the actual predictive distribution, not
            # the raw Kalman S_pred which ignores GARCH blending, β
            # recalibration, and ν refinement.
            # ─────────────────────────────────────────────────────────────
            n_train_u = int(n_obs * 0.7)
            # Default Berk/MAD/LR/count from raw PIT; overridden by filter_and_calibrate
            _calibrated_berk_u = pit_metrics.get("berkowitz_pvalue", 0.0)
            _calibrated_berk_lr_u = pit_metrics.get("berkowitz_lr", 0.0)
            _calibrated_pit_count_u = int(pit_metrics.get("pit_count", 0))
            _calibrated_mad_u = pit_metrics.get("histogram_mad", 1.0)
            _ad_p_u = float('nan')
            try:
                # ── PERFORMANCE: Reuse test-period evaluation from optimize_params_unified
                # instead of calling filter_and_calibrate a second time (saves ~1 full
                # pipeline pass per unified model: filter + GARCH + Stage 6 calibration)
                _calib_diag_u = diagnostics.get('test_calib_diag')
                if (_calib_diag_u is not None
                    and diagnostics.get('test_sigma') is not None
                    and diagnostics.get('test_pit_pvalue') is not None):
                    # Reuse cached results from optimize_params_unified
                    _sigma_cal_u = diagnostics['test_sigma']
                    _pit_p_u = diagnostics['test_pit_pvalue']
                    _pit_cal_u = diagnostics.get('test_pit_values')
                else:
                    # Fallback: call filter_and_calibrate (shouldn't normally happen)
                    _pit_cal_u, _pit_p_u, _sigma_cal_u, _, _calib_diag_u = \
                        UnifiedPhiStudentTModel.filter_and_calibrate(
                            returns, vol, config, train_frac=0.7
                        )
                # Use calibrated PIT p-value from filter_and_calibrate
                # (raw pit_ks_unified runs on full data without GARCH blending;
                #  filter_and_calibrate applies GARCH + beta recalibration on
                #  the test fold, matching what make pit-metals reports)
                if _pit_p_u is not None and np.isfinite(_pit_p_u):
                    pit_p_u = float(_pit_p_u)
                # Extract calibrated Berkowitz/MAD from GARCH path
                _bd = _calib_diag_u.get('berkowitz_pvalue')
                if _bd is not None and np.isfinite(_bd):
                    _calibrated_berk_u = float(_bd)
                _blr = _calib_diag_u.get('berkowitz_lr')
                if _blr is not None and np.isfinite(_blr):
                    _calibrated_berk_lr_u = float(_blr)
                _bpc = _calib_diag_u.get('pit_count')
                if _bpc is not None and _bpc > 0:
                    _calibrated_pit_count_u = int(_bpc)
                _md = _calib_diag_u.get('mad')
                if _md is not None and np.isfinite(_md):
                    _calibrated_mad_u = float(_md)
                _ad_p_u = _calib_diag_u.get('ad_pvalue', float('nan'))
                _nu_eff_u = _calib_diag_u.get('nu_effective', float(nu_fixed))
                returns_test_u = returns[n_train_u:]
                mu_pred_test_u = mu_pred_u[n_train_u:]
                
                if len(_sigma_cal_u) == len(returns_test_u) and np.all(_sigma_cal_u > 0):
                    forecast_scale_u = np.maximum(_sigma_cal_u, 1e-10)
                    nu_for_score = _nu_eff_u
                else:
                    if nu_fixed > 2:
                        forecast_scale_u = np.sqrt(S_pred_u[n_train_u:] * (nu_fixed - 2) / nu_fixed)
                    else:
                        forecast_scale_u = np.sqrt(S_pred_u[n_train_u:])
                    forecast_scale_u = np.maximum(forecast_scale_u, 1e-10)
                    nu_for_score = float(nu_fixed)
                
                mu_effective_u = mu_pred_test_u
            except Exception:
                # Fallback to raw predictions on full data
                if nu_fixed > 2:
                    forecast_scale_u = np.sqrt(S_pred_u * (nu_fixed - 2) / nu_fixed)
                else:
                    forecast_scale_u = np.sqrt(S_pred_u)
                forecast_scale_u = np.maximum(forecast_scale_u, 1e-10)
                mu_effective_u = mu_pred_u
                nu_for_score = float(nu_fixed)
                returns_test_u = returns

            hyvarinen_u = compute_hyvarinen_score_student_t(
                returns_test_u, mu_effective_u, forecast_scale_u, nu_for_score
            )
            crps_u = compute_crps_student_t_inline(
                returns_test_u, mu_effective_u, forecast_scale_u, nu_for_score
            )

            # =============================================================
            # STAGE U-H: Hansen Skew-t augmentation for unified model
            # =============================================================
            # Same CRPS-gated pattern as base phi_student_t (Stage 7.5).
            # Fits Hansen λ on training residuals, re-runs filter with
            # Hansen observation noise using the unified config's core
            # parameters (q, c, phi, nu).
            # =============================================================
            _u_hansen_activated = False
            _u_hansen_lambda = 0.0
            _u_hansen_diag = None
            try:
                if (forecast_scale_u is not None and len(forecast_scale_u) >= 50
                    and n_obs >= 100):
                    _fs_valid_u = forecast_scale_u[forecast_scale_u > 1e-12]
                    if len(_fs_valid_u) >= 50:
                        from models.hansen_skew_t import fit_hansen_skew_t_mle
                        # Compute standardised residuals for Hansen MLE
                        _resid_u = (returns_test_u - mu_effective_u) / np.maximum(forecast_scale_u, 1e-12)
                        _resid_train_u = _resid_u[:n_train_u] if len(_resid_u) > n_train_u else _resid_u[:len(_resid_u)//2]
                        _nu_h_u, _lam_h_u, _ll_h_u, _h_diag_u = fit_hansen_skew_t_mle(
                            _resid_train_u, nu_hint=nu_for_score)
                        if (_h_diag_u.get('fit_success', False) and
                            abs(_lam_h_u) > 0.01 and np.isfinite(_ll_h_u)):
                            from models.numba_wrappers import run_phi_hansen_skew_t_filter
                            mu_fh_u, P_fh_u, mu_ph_u, S_ph_u, ll_h_u = run_phi_hansen_skew_t_filter(
                                returns, vol,
                                float(config.q), float(config.c), float(config.phi),
                                nu_for_score,
                                hansen_lambda=_lam_h_u,
                                online_scale_adapt=True,
                                gamma_vov=float(getattr(config, 'gamma_vov', 0.0)),
                            )
                            if nu_for_score > 2:
                                _fs_h_u = np.sqrt(S_ph_u * (nu_for_score - 2) / nu_for_score)
                            else:
                                _fs_h_u = np.sqrt(S_ph_u)
                            _fs_h_u = np.maximum(_fs_h_u, 1e-10)
                            # CRPS on test fold
                            _crps_h_u = compute_crps_student_t_inline(
                                returns_test_u, mu_ph_u[n_train_u:] if len(mu_ph_u) > n_train_u else mu_ph_u,
                                _fs_h_u[n_train_u:] if len(_fs_h_u) > n_train_u else _fs_h_u,
                                nu_for_score)
                            _n_params_h_u = effective_n_params + 1
                            _bic_h_u = compute_bic(ll_h_u, _n_params_h_u, n_obs)
                            # CRPS gate
                            if (np.isfinite(_crps_h_u) and _crps_h_u < crps_u and
                                np.isfinite(_bic_h_u) and _bic_h_u < bic_u + 2.0):
                                _u_hansen_activated = True
                                _u_hansen_lambda = float(_lam_h_u)
                                _u_hansen_diag = {
                                    'nu_hansen': float(_nu_h_u),
                                    'lambda_hansen': float(_lam_h_u),
                                    'crps_before': float(crps_u),
                                    'crps_after': float(_crps_h_u),
                                    'bic_before': float(bic_u),
                                    'bic_after': float(_bic_h_u),
                                }
                                # Update scoring vars
                                mu_pred_u = mu_ph_u
                                S_pred_u = S_ph_u
                                forecast_scale_u = _fs_h_u
                                crps_u = _crps_h_u
                                ll_u = ll_h_u
                                mean_ll_u = ll_h_u / max(n_obs, 1)
                                effective_n_params = _n_params_h_u
                                bic_u = _bic_h_u
                                aic_u = compute_aic(ll_h_u, _n_params_h_u)
                                # Recompute diagnostics
                                _pit_h_u = compute_extended_pit_metrics_student_t(
                                    returns, vol,
                                    float(config.q), float(config.c), float(config.phi),
                                    nu_for_score,
                                    mu_pred_precomputed=mu_ph_u,
                                    S_pred_precomputed=S_ph_u,
                                    scale_already_adapted=True,
                                )
                                ks_u = _pit_h_u["ks_statistic"]
                                pit_p_u = _pit_h_u["pit_ks_pvalue"]
                                _ad_p_u = _pit_h_u.get("ad_pvalue", float('nan'))
                                _calibrated_berk_u = _pit_h_u["berkowitz_pvalue"]
                                _calibrated_mad_u = _pit_h_u["histogram_mad"]
                                hyvarinen_u = compute_hyvarinen_score_student_t(
                                    returns_test_u,
                                    mu_ph_u[n_train_u:] if len(mu_ph_u) > n_train_u else mu_ph_u,
                                    _fs_h_u[n_train_u:] if len(_fs_h_u) > n_train_u else _fs_h_u,
                                    nu_for_score)
            except Exception:
                pass

            # =============================================================
            # STAGE U-C: Contaminated Student-t augmentation for unified
            # =============================================================
            # Same CRPS-gated pattern as base phi_student_t (Stage 7.6).
            # Profile grid search over (ν_crisis, ε).
            # =============================================================
            _u_cst_activated = False
            _u_cst_nu_crisis = None
            _u_cst_epsilon = 0.0
            _u_cst_diag = None
            try:
                if n_obs >= 100 and nu_for_score >= 4:
                    _U_CST_NU_GRID = [3.0, 4.0, 5.0, 6.0]
                    _U_CST_EPS_GRID = [0.02, 0.05, 0.10]
                    _best_cst_crps_u = crps_u

                    _best_cst_combo_u = None
                    for _nc_u in _U_CST_NU_GRID:
                        if _nc_u >= nu_for_score:
                            continue
                        for _eps_u in _U_CST_EPS_GRID:
                            try:
                                from models.numba_wrappers import run_phi_cst_filter
                                mu_fc_u, P_fc_u, mu_pc_u, S_pc_u, ll_c_u = run_phi_cst_filter(
                                    returns, vol,
                                    float(config.q), float(config.c), float(config.phi),
                                    nu_normal=nu_for_score,
                                    nu_crisis=_nc_u,
                                    epsilon=_eps_u,
                                    online_scale_adapt=True,
                                    gamma_vov=float(getattr(config, 'gamma_vov', 0.0)),
                                )
                                if nu_for_score > 2:
                                    _fs_c_u = np.sqrt(S_pc_u * (nu_for_score - 2) / nu_for_score)
                                else:
                                    _fs_c_u = np.sqrt(S_pc_u)
                                _fs_c_u = np.maximum(_fs_c_u, 1e-10)
                                _crps_c_u = compute_crps_student_t_inline(
                                    returns_test_u,
                                    mu_pc_u[n_train_u:] if len(mu_pc_u) > n_train_u else mu_pc_u,
                                    _fs_c_u[n_train_u:] if len(_fs_c_u) > n_train_u else _fs_c_u,
                                    nu_for_score)
                                _n_params_c_u = effective_n_params + 2
                                _bic_c_u = compute_bic(ll_c_u, _n_params_c_u, n_obs)
                                if (np.isfinite(_crps_c_u) and _crps_c_u < _best_cst_crps_u and
                                    np.isfinite(_bic_c_u) and _bic_c_u < bic_u + 4.0):
                                    _best_cst_crps_u = _crps_c_u
                                    _best_cst_combo_u = (_nc_u, _eps_u, mu_pc_u, S_pc_u,
                                                         _fs_c_u, ll_c_u, _bic_c_u, _n_params_c_u)
                            except Exception:
                                continue

                    if _best_cst_combo_u is not None:
                        _nc_u, _eps_u, mu_pc_u, S_pc_u, _fs_c_u, ll_c_u, _bic_c_u, _n_params_c_u = _best_cst_combo_u
                        _u_cst_activated = True
                        _u_cst_nu_crisis = float(_nc_u)
                        _u_cst_epsilon = float(_eps_u)
                        _u_cst_diag = {
                            'nu_crisis': float(_nc_u),
                            'epsilon': float(_eps_u),
                            'crps_before': float(crps_u),
                            'crps_after': float(_best_cst_crps_u),
                            'bic_before': float(bic_u),
                            'bic_after': float(_bic_c_u),
                        }
                        # Update scoring vars
                        mu_pred_u = mu_pc_u
                        S_pred_u = S_pc_u
                        forecast_scale_u = _fs_c_u
                        crps_u = _best_cst_crps_u
                        ll_u = ll_c_u
                        mean_ll_u = ll_c_u / max(n_obs, 1)
                        effective_n_params = _n_params_c_u
                        bic_u = _bic_c_u
                        aic_u = compute_aic(ll_c_u, _n_params_c_u)
                        # Recompute diagnostics
                        _pit_c_u = compute_extended_pit_metrics_student_t(
                            returns, vol,
                            float(config.q), float(config.c), float(config.phi),
                            nu_for_score,
                            mu_pred_precomputed=mu_pc_u,
                            S_pred_precomputed=S_pc_u,
                            scale_already_adapted=True,
                        )
                        ks_u = _pit_c_u["ks_statistic"]
                        pit_p_u = _pit_c_u["pit_ks_pvalue"]
                        _ad_p_u = _pit_c_u.get("ad_pvalue", float('nan'))
                        _calibrated_berk_u = _pit_c_u["berkowitz_pvalue"]
                        _calibrated_mad_u = _pit_c_u["histogram_mad"]
                        hyvarinen_u = compute_hyvarinen_score_student_t(
                            returns_test_u,
                            mu_pc_u[n_train_u:] if len(mu_pc_u) > n_train_u else mu_pc_u,
                            _fs_c_u[n_train_u:] if len(_fs_c_u) > n_train_u else _fs_c_u,
                            nu_for_score)
            except Exception:
                pass

            models[unified_name] = {
                # Core parameters
                "q": float(config.q),
                "c": float(config.c),
                "phi": float(config.phi),
                "nu": float(nu_for_score),  # Use calibrated ν (from Stage 5/6), not grid ν
                "nu_grid": float(nu_fixed),  # Original grid value preserved for reference
                # Unified-specific parameters
                "gamma_vov": float(config.gamma_vov),
                "alpha_asym": float(config.alpha_asym),
                "k_asym": float(getattr(config, 'k_asym', 1.0)),
                "ms_sensitivity": float(config.ms_sensitivity),
                "ms_ewm_lambda": float(getattr(config, 'ms_ewm_lambda', 0.0)),
                "q_stress_ratio": float(config.q_stress_ratio),
                "vov_damping": float(getattr(config, 'vov_damping', 0.3)),
                # Calibration parameters (February 2026)
                "variance_inflation": float(getattr(config, 'variance_inflation', 1.0)),
                "mu_drift": float(getattr(config, 'mu_drift', 0.0)),
                "risk_premium_sensitivity": float(getattr(config, 'risk_premium_sensitivity', 0.0)),
                # Conditional skew dynamics (February 2026 - GAS Framework)
                "skew_score_sensitivity": float(getattr(config, 'skew_score_sensitivity', 0.0)),
                "skew_persistence": float(getattr(config, 'skew_persistence', 0.97)),
                # GARCH parameters (February 2026)
                "garch_omega": float(getattr(config, 'garch_omega', 0.0)),
                "garch_alpha": float(getattr(config, 'garch_alpha', 0.0)),
                "garch_beta": float(getattr(config, 'garch_beta', 0.0)),
                "garch_leverage": float(getattr(config, 'garch_leverage', 0.0)),
                "garch_unconditional_var": float(getattr(config, 'garch_unconditional_var', 1e-4)),
                # Rough volatility memory (February 2026 - Gatheral-Jaisson-Rosenbaum)
                "rough_hurst": float(getattr(config, 'rough_hurst', 0.0)),
                # Merton jump-diffusion parameters (February 2026)
                "jump_intensity": float(getattr(config, 'jump_intensity', 0.0)),
                "jump_variance": float(getattr(config, 'jump_variance', 0.0)),
                "jump_sensitivity": float(getattr(config, 'jump_sensitivity', 1.0)),
                "jump_mean": float(getattr(config, 'jump_mean', 0.0)),
                # Data-driven bounds
                "c_min": float(getattr(config, 'c_min', 0.01)),
                "c_max": float(getattr(config, 'c_max', 10.0)),
                "q_min": float(getattr(config, 'q_min', 1e-8)),
                # CRPS-optimal EWM location correction (February 2026)
                "crps_ewm_lambda": float(getattr(config, 'crps_ewm_lambda', 0.0)),
                # Heston-DLSV leverage and mean reversion (February 2026)
                "rho_leverage": float(getattr(config, 'rho_leverage', 0.0)),
                "kappa_mean_rev": float(getattr(config, 'kappa_mean_rev', 0.0)),
                "theta_long_var": float(getattr(config, 'theta_long_var', 0.0)),
                "crps_sigma_shrinkage": float(getattr(config, 'crps_sigma_shrinkage', 1.0)),
                # CRPS-enhancement: vol-of-vol noise, asymmetric df, regime switching
                "sigma_eta": float(getattr(config, 'sigma_eta', 0.0)),
                "t_df_asym": float(getattr(config, 't_df_asym', 0.0)),
                "regime_switch_prob": float(getattr(config, 'regime_switch_prob', 0.0)),
                # v7.8 elite MC enhancements: dynamic leverage, liquidity stress, entropy
                "leverage_dynamic_decay": float(getattr(config, 'leverage_dynamic_decay', 0.0)),
                "liq_stress_coeff": float(getattr(config, 'liq_stress_coeff', 0.0)),
                "entropy_sigma_lambda": float(getattr(config, 'entropy_sigma_lambda', 0.0)),
                # GARCH-Kalman reconciliation + Q_t coupling + location bias (February 2026)
                "garch_kalman_weight": float(getattr(config, 'garch_kalman_weight', 0.0)),
                "q_vol_coupling": float(getattr(config, 'q_vol_coupling', 0.0)),
                "loc_bias_var_coeff": float(getattr(config, 'loc_bias_var_coeff', 0.0)),
                "loc_bias_drift_coeff": float(getattr(config, 'loc_bias_drift_coeff', 0.0)),
                # Stage 6: pre-calibrated walk-forward CV params
                "calibrated_gw": float(getattr(config, 'calibrated_gw', 0.50)),
                "calibrated_nu_pit": float(getattr(config, 'calibrated_nu_pit', 0.0)),
                "calibrated_beta_probit_corr": float(getattr(config, 'calibrated_beta_probit_corr', 1.0)),
                "calibrated_lambda_rho": float(getattr(config, 'calibrated_lambda_rho', 0.985)),
                "calibrated_nu_crps": float(getattr(config, 'calibrated_nu_crps', 0.0)),
                # Scores
                "log_likelihood": float(ll_u),
                "mean_log_likelihood": float(mean_ll_u),
                "bic": float(bic_u),
                "aic": float(aic_u),
                "hyvarinen_score": float(hyvarinen_u),
                "crps": float(crps_u),
                "ks_statistic": float(ks_u),
                "pit_ks_pvalue": float(pit_p_u),
                "ad_pvalue": float(_ad_p_u),
                "ad_pvalue_raw": float(pit_metrics.get("ad_pvalue_raw", _ad_p_u) if isinstance(pit_metrics, dict) else _ad_p_u),
                "ad_correction": pit_metrics.get("ad_correction", {}) if isinstance(pit_metrics, dict) else {},
                "calibration_params": pit_metrics.get("ad_correction", {}).get("calibration_params", {}) if isinstance(pit_metrics, dict) else {},
                "pit_calibration_grade": pit_metrics.get("calibration_grade", "F"),
                "histogram_mad": float(_calibrated_mad_u),
                "berkowitz_pvalue": float(_calibrated_berk_u),
                "berkowitz_lr": float(_calibrated_berk_lr_u),
                "pit_count": int(_calibrated_pit_count_u),
                # Hansen/CST pipeline augmentations (March 2026)
                "hansen_activated": _u_hansen_activated,
                "hansen_lambda": float(_u_hansen_lambda),
                "hansen_diagnostics": _u_hansen_diag,
                "cst_activated": _u_cst_activated,
                "cst_nu_crisis": float(_u_cst_nu_crisis) if _u_cst_nu_crisis is not None else None,
                "cst_epsilon": float(_u_cst_epsilon),
                "cst_diagnostics": _u_cst_diag,
                # Metadata
                "fit_success": True,
                "unified_model": True,
                "degraded": diagnostics.get("degraded", False),
                "hessian_cond": diagnostics.get("hessian_cond", float('inf')),
                "n_params": int(effective_n_params),
                "nu_fixed": True,
                "model_type": "phi_student_t_unified",
            }

        except Exception as e:
            models[unified_name] = {
                "fit_success": False,
                "error": str(e),
                "bic": float('inf'),
                "aic": float('inf'),
                "hyvarinen_score": float('-inf'),
                "crps": float('inf'),
                "nu": float(nu_fixed),
                "unified_model": True,
                "nu_fixed": True,
            }

    # =========================================================================
    # Model 2b: UNIFIED Gaussian (Feb 2026 - nu-free Calibration Pipeline)
    # =========================================================================
    # Includes internal momentum integration (Stage 1.5) and GAS-Q adaptive
    # process noise (Stage 4.5), both with degradation guards.
    # Momentum and GAS-Q are tuned INSIDE optimize_params_unified — not as
    # separate external model variants.
    # =========================================================================
    n_params_gaussian_unified = 8  # base params; momentum/GAS-Q add 1+3 conditionally

    # Pre-compute momentum signal for unified Gaussian (reused for both phi modes)
    _gu_momentum_signal = None
    if MOMENTUM_AUGMENTATION_AVAILABLE and MOMENTUM_AUGMENTATION_ENABLED:
        try:
            _gu_mom_features = compute_momentum_features(returns)
            _gu_momentum_signal = compute_momentum_signal(_gu_mom_features)
        except Exception:
            _gu_momentum_signal = None

    for phi_mode, model_prefix in [(False, "kalman_gaussian_unified"),
                                    (True, "kalman_phi_gaussian_unified")]:
        try:
            g_config, g_diag = GaussianDriftModel.optimize_params_unified(
                returns, vol, phi_mode=phi_mode, train_frac=0.7,
                asset_symbol=asset, momentum_signal=_gu_momentum_signal,
                gk_c_prior_value=gk_c_prior_value)
            if not g_diag.get("success", False):
                raise ValueError("Gaussian unified: " + str(g_diag.get("error", "?")))

            # Reconstruct momentum signal for filter_and_calibrate
            _gu_mom_for_calib = _gu_momentum_signal if g_config.momentum_enabled else None

            _, _, mu_pred_gu, S_pred_gu, ll_gu = GaussianDriftModel._filter_phi_with_momentum(
                returns, vol, float(g_config.q), float(g_config.c), float(g_config.phi),
                _gu_mom_for_calib, float(g_config.momentum_weight))
            _pit_gu, _pit_p_gu, _sigma_gu, _crps_gu, _calib_gu = GaussianDriftModel.filter_and_calibrate(
                returns, vol, g_config, train_frac=0.7,
                momentum_signal=_gu_mom_for_calib,
                mu_pred_precomputed=mu_pred_gu,
                S_pred_precomputed=S_pred_gu,
                ll_precomputed=ll_gu)
            pit_p_gu = float(_pit_p_gu) if np.isfinite(_pit_p_gu) else 0.0
            _ad_p_gu = float(_calib_gu.get("ad_pvalue", float('nan')))
            _berk_p_gu = float(_calib_gu.get("berkowitz_pvalue", 0.0))
            _berk_lr_gu = float(_calib_gu.get("berkowitz_lr", 0.0))
            _pit_count_gu = int(_calib_gu.get("pit_count", 0))
            _mad_gu = float(_calib_gu.get("mad", 1.0))

            # Parameter count: base 8 + momentum (1) + GAS-Q (3) when active
            _eff_n_params_gu = n_params_gaussian_unified
            if g_config.momentum_enabled:
                _eff_n_params_gu += 1
            if g_config.gas_q_enabled:
                _eff_n_params_gu += 3

            aic_gu = compute_aic(ll_gu, _eff_n_params_gu)
            bic_gu = compute_bic(ll_gu, _eff_n_params_gu, n_obs)
            n_train_gu = int(n_obs * 0.7)
            returns_test_gu = returns[n_train_gu:]
            mu_eff_gu = _calib_gu.get("mu_effective", mu_pred_gu[n_train_gu:])
            if isinstance(mu_eff_gu, np.ndarray) and len(mu_eff_gu) == len(returns_test_gu):
                forecast_std_gu = np.maximum(_sigma_gu, 1e-10)
            else:
                forecast_std_gu = np.sqrt(np.maximum(S_pred_gu[n_train_gu:], 1e-20))
                mu_eff_gu = mu_pred_gu[n_train_gu:]
            hyvarinen_gu = compute_hyvarinen_score_gaussian(returns_test_gu, mu_eff_gu, forecast_std_gu)
            crps_final_gu = compute_crps_gaussian_inline(returns_test_gu, mu_eff_gu, forecast_std_gu)
            _ks_stat_gu = 1.0
            if hasattr(_pit_gu, "__len__") and len(_pit_gu) > 0:
                _ps_gu = np.sort(np.asarray(_pit_gu))
                _n_ps_gu = len(_ps_gu)
                _dp_gu = np.max(np.arange(1, _n_ps_gu + 1) / _n_ps_gu - _ps_gu)
                _dm_gu = np.max(_ps_gu - np.arange(0, _n_ps_gu) / _n_ps_gu)
                _ks_stat_gu = float(max(_dp_gu, _dm_gu))
            models[model_prefix] = {
                "q": float(g_config.q), "c": float(g_config.c), "phi": float(g_config.phi),
                "variance_inflation": float(g_config.variance_inflation),
                "mu_drift": float(g_config.mu_drift),
                "garch_omega": float(g_config.garch_omega),
                "garch_alpha": float(g_config.garch_alpha),
                "garch_beta": float(g_config.garch_beta),
                "garch_leverage": float(g_config.garch_leverage),
                "garch_unconditional_var": float(g_config.garch_unconditional_var),
                "crps_ewm_lambda": float(g_config.crps_ewm_lambda),
                "crps_sigma_shrinkage": float(g_config.crps_sigma_shrinkage),
                # v7.8 elite MC enhancements
                "leverage_dynamic_decay": float(getattr(g_config, 'leverage_dynamic_decay', 0.0)),
                "liq_stress_coeff": float(getattr(g_config, 'liq_stress_coeff', 0.0)),
                "entropy_sigma_lambda": float(getattr(g_config, 'entropy_sigma_lambda', 0.0)),
                "calibrated_gw": float(g_config.calibrated_gw),
                "calibrated_lambda_rho": float(g_config.calibrated_lambda_rho),
                "calibrated_beta_probit_corr": float(g_config.calibrated_beta_probit_corr),
                # Momentum integration (Stage 1.5)
                "momentum_weight": float(g_config.momentum_weight),
                "momentum_augmented": g_config.momentum_enabled,
                # GAS-Q adaptive process noise (Stage 4.5)
                "gas_q_omega": float(g_config.gas_q_omega),
                "gas_q_alpha": float(g_config.gas_q_alpha),
                "gas_q_beta": float(g_config.gas_q_beta),
                "gas_q_enabled": g_config.gas_q_enabled,
                # Scores
                "log_likelihood": float(ll_gu),
                "mean_log_likelihood": float(ll_gu / max(n_obs, 1)),
                "bic": float(bic_gu), "aic": float(aic_gu),
                "hyvarinen_score": float(hyvarinen_gu),
                "crps": float(crps_final_gu),
                "ks_statistic": _ks_stat_gu,
                "pit_ks_pvalue": float(pit_p_gu),
                "ad_pvalue": float(_ad_p_gu),
                "ad_pvalue_raw": float(_calib_gu.get("ad_pvalue_raw", _ad_p_gu) if isinstance(_calib_gu, dict) else _ad_p_gu),
                "ad_correction": _calib_gu.get("ad_correction", {}) if isinstance(_calib_gu, dict) else {},
                "calibration_params": _calib_gu.get("ad_correction", {}).get("calibration_params", {}) if isinstance(_calib_gu, dict) else {},
                "pit_calibration_grade": "A" if _mad_gu < 0.02 else ("B" if _mad_gu < 0.05 else ("C" if _mad_gu < 0.10 else "F")),
                "histogram_mad": float(_mad_gu),
                "berkowitz_pvalue": float(_berk_p_gu),
                "berkowitz_lr": float(_berk_lr_gu),
                "pit_count": int(_pit_count_gu),
                "fit_success": True, "unified_model": True,
                "gaussian_unified": True,
                "phi_mode": phi_mode,
                "n_params": _eff_n_params_gu,
                "model_type": "gaussian_unified",
            }
        except Exception as e:
            models[model_prefix] = {
                "fit_success": False, "error": str(e),
                "bic": float('inf'), "aic": float('inf'),
                "hyvarinen_score": float('-inf'), "crps": float('inf'),
                "unified_model": True, "gaussian_unified": True,
            }

    # =========================================================================
    # MOMENTUM AUGMENTED STUDENT-T — RETIRED AS SEPARATE MODELS (Feb 2026)
    # =========================================================================
    # Momentum augmentation is now an INTERNAL pipeline step within each
    # phi_student_t_nu_X base model (see Model 1 block above).
    # Activated ONLY if CRPS improves. No separate _momentum model names.
    #
    # Legacy Gaussian momentum models similarly retired — functionality
    # subsumed by unified Gaussian models with internal momentum + GAS-Q.
    # =========================================================================
    
    # =========================================================================
    # LEGACY MODEL GRIDS REMOVED (March 2026)
    # =========================================================================
    # GAS-Q Student-t, MS-q Student-t, VoV, Two-Piece, and Two-Component
    # Mixture grids were all gated by `not UNIFIED_STUDENT_T_ONLY` (never True
    # in production). Functionality is now incorporated internally by unified
    # models via Stages 7.x / U-x pipeline.
    # =========================================================================

    # =========================================================================
    # RV-Q ADAPTIVE PROCESS NOISE MODELS (Tune.md Story 1.3)
    # =========================================================================
    # Proactive q: q_t = q_base * exp(gamma * delta_log(vol^2))
    # Competes with static-q (unified) and GAS-Q via BMA.
    # Data decides which wins per asset/regime.
    # =========================================================================
    if RV_Q_AVAILABLE and RV_Q_ENABLED:
        try:
            # Get base parameters from best existing Gaussian model as starting point
            _rv_c_base = 1.0
            _rv_phi_base = 0.98
            for _mname, _mdata in models.items():
                if _mdata.get("fit_success") and "phi" in _mdata and "c" in _mdata:
                    _rv_c_base = float(_mdata["c"])
                    _rv_phi_base = float(_mdata.get("phi", 0.98))
                    break

            # --- RV-Q phi-Gaussian ---
            _rv_g_name = make_rv_q_phi_gaussian_name()
            try:
                _rv_g_fit = fit_rv_q_gaussian(returns, vol, _rv_c_base, _rv_phi_base)
                if _rv_g_fit.fit_success:
                    _rv_g_result = rv_adaptive_q_filter_gaussian(
                        returns, vol, _rv_c_base, _rv_phi_base,
                        RVAdaptiveQConfig(q_base=_rv_g_fit.q_base, gamma=_rv_g_fit.gamma)
                    )
                    _rv_g_mu = _rv_g_result.mu_filtered
                    _rv_g_P = _rv_g_result.P_filtered
                    _rv_g_ll = _rv_g_result.log_likelihood
                    _rv_g_n_params = 4  # q_base, gamma, c, phi
                    _rv_g_bic = compute_bic(_rv_g_ll, _rv_g_n_params, n_obs)
                    _rv_g_aic = compute_aic(_rv_g_ll, _rv_g_n_params)
                    _rv_g_n_train = int(n_obs * 0.7)
                    _rv_g_ret_test = returns[_rv_g_n_train:]
                    _rv_g_mu_test = _rv_g_mu[_rv_g_n_train:]
                    _rv_g_std_test = np.sqrt(np.maximum(_rv_g_P[_rv_g_n_train:] + (_rv_c_base * vol[_rv_g_n_train:])**2, 1e-20))
                    _rv_g_hyv = compute_hyvarinen_score_gaussian(_rv_g_ret_test, _rv_g_mu_test, _rv_g_std_test)
                    _rv_g_crps = compute_crps_gaussian_inline(_rv_g_ret_test, _rv_g_mu_test, _rv_g_std_test)
                    # PIT via extended metrics (returns dict)
                    _rv_g_calib = compute_extended_pit_metrics_gaussian(
                        returns, _rv_g_mu, _rv_g_P, vol, _rv_g_fit.q_base, _rv_c_base, _rv_phi_base)
                    _rv_g_mad = float(_rv_g_calib.get("histogram_mad", 1.0))
                    _rv_g_berk_p = float(_rv_g_calib.get("berkowitz_pvalue", 0.0))
                    _rv_g_berk_lr = float(_rv_g_calib.get("berkowitz_lr", 0.0))
                    _rv_g_pit_count = int(_rv_g_calib.get("pit_count", 0))
                    _rv_g_ad_p = float(_rv_g_calib.get("ad_pvalue", float('nan')))
                    _rv_g_ks = float(_rv_g_calib.get("ks_statistic", 1.0))
                    _rv_g_pit_p = float(_rv_g_calib.get("pit_ks_pvalue", 0.0))
                    models[_rv_g_name] = {
                        "q_base": float(_rv_g_fit.q_base), "gamma": float(_rv_g_fit.gamma),
                        "q": float(_rv_g_fit.q_base), "c": float(_rv_c_base), "phi": float(_rv_phi_base),
                        "log_likelihood": float(_rv_g_ll), "mean_log_likelihood": float(_rv_g_ll / max(n_obs, 1)),
                        "bic": float(_rv_g_bic), "aic": float(_rv_g_aic),
                        "hyvarinen_score": float(_rv_g_hyv), "crps": float(_rv_g_crps),
                        "ks_statistic": _rv_g_ks, "pit_ks_pvalue": _rv_g_pit_p,
                        "ad_pvalue": float(_rv_g_ad_p), "histogram_mad": float(_rv_g_mad),
                        "berkowitz_pvalue": float(_rv_g_berk_p), "berkowitz_lr": float(_rv_g_berk_lr),
                        "pit_count": int(_rv_g_pit_count),
                        "fit_success": True, "n_params": _rv_g_n_params,
                        "model_type": "rv_q", "rv_q_model": True,
                        "ll_improvement": float(_rv_g_fit.ll_improvement),
                        "delta_bic": float(_rv_g_fit.delta_bic),
                    }
            except Exception as e:
                models[_rv_g_name] = {
                    "fit_success": False, "error": str(e),
                    "bic": float('inf'), "crps": float('inf'),
                    "hyvarinen_score": float('-inf'), "model_type": "rv_q",
                }

            # --- RV-Q Student-t family (nu grid) ---
            for _rv_nu in STUDENT_T_NU_GRID:
                _rv_t_name = make_rv_q_student_t_name(_rv_nu)
                try:
                    _rv_t_fit = fit_rv_q_student_t(returns, vol, _rv_c_base, _rv_phi_base, float(_rv_nu))
                    if _rv_t_fit.fit_success:
                        _rv_t_result = rv_adaptive_q_filter_student_t(
                            returns, vol, _rv_c_base, _rv_phi_base, float(_rv_nu),
                            RVAdaptiveQConfig(q_base=_rv_t_fit.q_base, gamma=_rv_t_fit.gamma)
                        )
                        _rv_t_mu = _rv_t_result.mu_filtered
                        _rv_t_P = _rv_t_result.P_filtered
                        _rv_t_ll = _rv_t_result.log_likelihood
                        _rv_t_n_params = 5  # q_base, gamma, c, phi, nu
                        _rv_t_bic = compute_bic(_rv_t_ll, _rv_t_n_params, n_obs)
                        _rv_t_aic = compute_aic(_rv_t_ll, _rv_t_n_params)
                        _rv_t_n_train = int(n_obs * 0.7)
                        _rv_t_ret_test = returns[_rv_t_n_train:]
                        _rv_t_mu_test = _rv_t_mu[_rv_t_n_train:]
                        _rv_t_scale_test = np.sqrt(np.maximum(_rv_t_P[_rv_t_n_train:] + (_rv_c_base * vol[_rv_t_n_train:])**2, 1e-20))
                        _rv_t_hyv = compute_hyvarinen_score_student_t(_rv_t_ret_test, _rv_t_mu_test, _rv_t_scale_test, float(_rv_nu))
                        _rv_t_crps = compute_crps_student_t_inline(_rv_t_ret_test, _rv_t_mu_test, _rv_t_scale_test, float(_rv_nu))
                        # PIT via extended metrics (returns dict)
                        # Convert filtered -> predictive for Student-t PIT
                        _rv_t_mu_pred, _rv_t_S_pred = reconstruct_predictive_from_filtered_gaussian(
                            returns, _rv_t_mu, _rv_t_P, vol, _rv_t_fit.q_base, _rv_c_base, _rv_phi_base)
                        _rv_t_calib = compute_extended_pit_metrics_student_t(
                            returns, vol, _rv_t_fit.q_base, _rv_c_base, _rv_phi_base, float(_rv_nu),
                            mu_pred_precomputed=_rv_t_mu_pred, S_pred_precomputed=_rv_t_S_pred)
                        _rv_t_mad = float(_rv_t_calib.get("histogram_mad", 1.0))
                        _rv_t_berk_p = float(_rv_t_calib.get("berkowitz_pvalue", 0.0))
                        _rv_t_berk_lr = float(_rv_t_calib.get("berkowitz_lr", 0.0))
                        _rv_t_pit_count = int(_rv_t_calib.get("pit_count", 0))
                        _rv_t_ad_p = float(_rv_t_calib.get("ad_pvalue", float('nan')))
                        _rv_t_ks = float(_rv_t_calib.get("ks_statistic", 1.0))
                        _rv_t_pit_p = float(_rv_t_calib.get("pit_ks_pvalue", 0.0))
                        models[_rv_t_name] = {
                            "q_base": float(_rv_t_fit.q_base), "gamma": float(_rv_t_fit.gamma),
                            "q": float(_rv_t_fit.q_base), "c": float(_rv_c_base), "phi": float(_rv_phi_base),
                            "nu": float(_rv_nu),
                            "log_likelihood": float(_rv_t_ll), "mean_log_likelihood": float(_rv_t_ll / max(n_obs, 1)),
                            "bic": float(_rv_t_bic), "aic": float(_rv_t_aic),
                            "hyvarinen_score": float(_rv_t_hyv), "crps": float(_rv_t_crps),
                            "ks_statistic": _rv_t_ks, "pit_ks_pvalue": _rv_t_pit_p,
                            "ad_pvalue": float(_rv_t_ad_p), "histogram_mad": float(_rv_t_mad),
                            "berkowitz_pvalue": float(_rv_t_berk_p), "berkowitz_lr": float(_rv_t_berk_lr),
                            "pit_count": int(_rv_t_pit_count),
                            "fit_success": True, "n_params": _rv_t_n_params,
                            "model_type": "rv_q", "rv_q_model": True,
                            "ll_improvement": float(_rv_t_fit.ll_improvement),
                            "delta_bic": float(_rv_t_fit.delta_bic),
                        }
                except Exception as e:
                    models[_rv_t_name] = {
                        "fit_success": False, "error": str(e),
                        "bic": float('inf'), "crps": float('inf'),
                        "hyvarinen_score": float('-inf'), "model_type": "rv_q",
                    }
        except Exception:
            pass  # RV-Q fitting failed entirely, continue without it

    # =========================================================================
    # Phi ACF Lower Bound Enforcement (Story 1.8)
    # =========================================================================
    # Apply the autocorrelation-based phi floor to all fitted models.
    # Store acf_1 and phi_acf_floor in model diagnostics.
    # =========================================================================
    for _mn, _md in models.items():
        if not _md.get("fit_success", False):
            continue
        _md["acf_1"] = _acf_1
        _md["phi_acf_floor"] = _phi_acf_floor
        _phi_val = _md.get("phi")
        if _phi_val is not None and _phi_val < _phi_acf_floor:
            _md["phi_original_pre_acf_floor"] = _phi_val
            _md["phi"] = _phi_acf_floor

    return models


