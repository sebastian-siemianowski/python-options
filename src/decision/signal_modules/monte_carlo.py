from __future__ import annotations

"""Monte Carlo simulation engine for signal generation.

Extracted from signals.py - Story 7.1.
Contains: run_regime_specific_mc, run_unified_mc, MCDiagnostics,
          diagnose_mc_paths, compute_model_posteriors_from_combined_score,
          shift_features, make_features_views, _simulate_forward_paths.
"""

import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sys as _sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from decision.signal_modules.config import *  # noqa: F403
from decision.signal_modules.volatility_imports import *  # noqa: F403

def run_regime_specific_mc(
    regime: int,
    mu_t: float,
    P_t: float,
    phi: float,
    q: float,
    sigma2_step: float,
    H: int,
    n_paths: int = 5000,
    nu: Optional[float] = None,
    hansen_lambda: Optional[float] = None,
    # Contaminated Student-t parameters
    cst_nu_normal: Optional[float] = None,
    cst_nu_crisis: Optional[float] = None,
    cst_epsilon: Optional[float] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Run posterior predictive MC for a specific regime.

    This is a lightweight wrapper that generates r_samples for one regime
    using regime-specific parameters.

    Supports four noise distributions (in priority order):
    1. Contaminated Student-t (cst_nu_normal, cst_nu_crisis, cst_epsilon specified):
       Regime-dependent heavy tails: (1-ε)×t(ν_normal) + ε×t(ν_crisis)
    2. Hansen Skew-t (nu + hansen_lambda specified): Asymmetric heavy tails
    3. Student-t (nu specified): Heavy tails, symmetric
    4. Gaussian (default): Light tails, symmetric
    
    Contaminated Student-t model:
        p(r) = (1-ε) × t(r; ν_normal) + ε × t(r; ν_crisis)
        
    Where ε is the contamination probability (crisis mode), typically 5-15%.
    This captures the intuition: "Most of the time markets are normal, but
    occasionally we're in crisis mode with much heavier tails."

    Args:
        regime: Regime index (0-4)
        mu_t: Current drift estimate
        P_t: Drift posterior variance
        phi: AR(1) persistence
        q: Process noise variance
        sigma2_step: Per-step observation variance
        H: Forecast horizon
        n_paths: Number of MC paths
        nu: Degrees of freedom for Student-t (None for Gaussian)
        hansen_lambda: Hansen skewness parameter (None for symmetric)
        cst_nu_normal: Contaminated-t normal regime ν
        cst_nu_crisis: Contaminated-t crisis regime ν
        cst_epsilon: Contaminated-t crisis probability
        seed: Random seed

    Returns:
        Array of return samples
    """
    # Check for Contaminated Student-t (highest priority for Student-t family)
    use_contaminated_t = (
        CONTAMINATED_ST_AVAILABLE and
        cst_nu_normal is not None and
        cst_nu_crisis is not None and
        cst_epsilon is not None and
        cst_epsilon > 0.001
    )
    
    # =========================================================================
    # HANSEN SKEW-T DETECTION (asymmetric Student-t)
    # =========================================================================
    # If hansen_lambda is provided and non-trivial, use Hansen skew-t sampling
    # instead of symmetric Student-t. This is the CRITICAL fix - hansen_lambda
    # was previously accepted but IGNORED in sampling.
    #
    # Priority order:
    #   1. Contaminated Student-t (regime-dependent tails)
    #   2. Hansen Skew-t (asymmetric tails with fixed λ)
    #   3. Symmetric Student-t (heavy tails only)
    #   4. Gaussian (light tails)
    # =========================================================================
    use_hansen_skew_t = (
        HANSEN_SKEW_T_AVAILABLE and
        not use_contaminated_t and  # CST takes priority
        nu is not None and
        hansen_lambda is not None and
        abs(hansen_lambda) > 0.01  # Only use if λ is non-trivial
    )
    
    # Input validation
    mu_t = float(mu_t) if np.isfinite(mu_t) else 0.0
    P_t = float(max(P_t, 0.0)) if np.isfinite(P_t) else 0.0
    phi = float(phi) if np.isfinite(phi) else 1.0
    q = float(max(q, 0.0)) if np.isfinite(q) else 0.0
    sigma2_step = float(max(sigma2_step, 1e-12)) if np.isfinite(sigma2_step) else 1e-6
    H = int(max(H, 1))

    if nu is not None and not use_contaminated_t:
        if not np.isfinite(nu) or nu <= 2.0:
            nu = None
        else:
            nu = float(np.clip(nu, 2.1, 500.0))

    rng = np.random.default_rng(seed)

    # Pre-compute mixture variance for contaminated Student-t if needed
    # This MUST be computed before sampling to avoid "unbound local variable" error
    mixture_var = 1.0  # Default to 1.0 (standard normal variance)
    if use_contaminated_t:
        var_normal = cst_nu_normal / (cst_nu_normal - 2) if cst_nu_normal > 2 else 10.0
        var_crisis = cst_nu_crisis / (cst_nu_crisis - 2) if cst_nu_crisis > 2 else 10.0
        mixture_var = (1 - cst_epsilon) * var_normal + cst_epsilon * var_crisis
        mixture_var = max(mixture_var, 1e-10)  # Ensure positive

    # Sample drift posterior
    if P_t > 0:
        if use_contaminated_t:
            # Contaminated Student-t posterior for drift
            # With probability ε, use crisis ν; else use normal ν
            mu_samples = contaminated_student_t_rvs(
                size=n_paths,
                nu_normal=cst_nu_normal,
                nu_crisis=cst_nu_crisis,
                epsilon=cst_epsilon,
                mu=0.0,
                sigma=1.0,
                random_state=rng
            )
            # Scale to have variance = P_t
            # Contaminated-t has variance ≈ weighted average of component variances
            # For t(ν): Var = ν/(ν-2) for ν > 2
            var_normal = cst_nu_normal / (cst_nu_normal - 2) if cst_nu_normal > 2 else 10.0
            var_crisis = cst_nu_crisis / (cst_nu_crisis - 2) if cst_nu_crisis > 2 else 10.0
            mixture_var = (1 - cst_epsilon) * var_normal + cst_epsilon * var_crisis
            t_scale = np.sqrt(P_t / mixture_var)
            mu_paths = mu_t + t_scale * mu_samples
        elif nu is not None:
            # Student-t posterior for drift
            t_scale = math.sqrt(P_t * (nu - 2.0) / nu) if nu > 2.0 else math.sqrt(P_t)
            mu_paths = mu_t + t_scale * rng.standard_t(df=nu, size=n_paths)
        else:
            # Gaussian posterior for drift
            mu_paths = rng.normal(loc=mu_t, scale=math.sqrt(P_t), size=n_paths)
    else:
        mu_paths = np.full(n_paths, mu_t, dtype=float)

    # Propagate drift and accumulate noise
    cum_mu = np.zeros(n_paths, dtype=float)
    cum_eps = np.zeros(n_paths, dtype=float)

    q_std = math.sqrt(q) if q > 0 else 0.0
    sigma_step = math.sqrt(sigma2_step)

    for k in range(H):
        # --- Drift propagation: μ_{t+k+1} = φ·μ_{t+k} + η_{k+1} ---
        if q_std > 0:
            if use_contaminated_t:
                # Contaminated Student-t drift noise
                eta_samples = contaminated_student_t_rvs(
                    size=n_paths,
                    nu_normal=cst_nu_normal,
                    nu_crisis=cst_nu_crisis,
                    epsilon=cst_epsilon,
                    mu=0.0,
                    sigma=1.0,
                    random_state=rng
                )
                eta_scale = q_std / np.sqrt(mixture_var)
                eta = eta_scale * eta_samples
            elif nu is not None:
                eta_scale = q_std * math.sqrt((nu - 2.0) / nu) if nu > 2.0 else q_std
                eta = eta_scale * rng.standard_t(df=nu, size=n_paths)
            else:
                eta = rng.normal(loc=0.0, scale=q_std, size=n_paths)
        else:
            eta = np.zeros(n_paths, dtype=float)

        mu_paths = phi * mu_paths + eta
        cum_mu += mu_paths

        # --- Observation noise: ε_k ---
        if sigma_step > 0:
            if use_contaminated_t:
                # Contaminated Student-t observation noise
                eps_samples = contaminated_student_t_rvs(
                    size=n_paths,
                    nu_normal=cst_nu_normal,
                    nu_crisis=cst_nu_crisis,
                    epsilon=cst_epsilon,
                    mu=0.0,
                    sigma=1.0,
                    random_state=rng
                )
                eps_scale = sigma_step / np.sqrt(mixture_var)
                eps_k = eps_scale * eps_samples
            elif nu is not None:
                eps_scale = sigma_step * math.sqrt((nu - 2.0) / nu) if nu > 2.0 else sigma_step
                eps_k = eps_scale * rng.standard_t(df=nu, size=n_paths)
            else:
                eps_k = rng.normal(loc=0.0, scale=sigma_step, size=n_paths)
        else:
            eps_k = np.zeros(n_paths, dtype=float)

        cum_eps += eps_k

    return cum_mu + cum_eps


def run_unified_mc(
    mu_t: float,
    P_t: float,
    phi: float,
    q: float,
    sigma2_step: float,
    H_max: int,
    n_paths: int = 5000,
    nu: Optional[float] = None,
    use_garch: bool = False,
    garch_omega: float = 0.0,
    garch_alpha: float = 0.0,
    garch_beta: float = 0.0,
    jump_intensity: float = 0.0,
    jump_mean: float = 0.0,
    jump_std: float = 0.05,
    enable_jumps: bool = True,
    seed: Optional[int] = None,
    # Exotic distributions (fallback to Python path)
    hansen_lambda: Optional[float] = None,
    cst_nu_normal: Optional[float] = None,
    cst_nu_crisis: Optional[float] = None,
    cst_epsilon: Optional[float] = None,
    # v7.6: Enriched MC params from tuned Student-t / unified models
    garch_leverage: float = 0.0,
    variance_inflation: float = 1.0,
    mu_drift: float = 0.0,
    alpha_asym: float = 0.0,
    k_asym: float = 2.0,
    risk_premium_sensitivity: float = 0.0,
    # v7.7: Tier 2 — vol mean-reversion, CRPS shrinkage, MS process noise, rough vol
    kappa_mean_rev: float = 0.0,
    theta_long_var: float = 0.0,
    crps_sigma_shrinkage: float = 1.0,
    ms_sensitivity: float = 0.0,
    q_stress_ratio: float = 1.0,
    rough_hurst: float = 0.0,
    # v7.7: Tier 3 — vol-of-vol, asymmetric ν, regime switching, GAS skew, loc bias
    sigma_eta: float = 0.0,
    t_df_asym: float = 0.0,
    regime_switch_prob: float = 0.0,
    gamma_vov: float = 0.0,
    vov_damping: float = 0.0,
    skew_score_sensitivity: float = 0.0,
    skew_persistence: float = 0.97,
    loc_bias_var_coeff: float = 0.0,
    loc_bias_drift_coeff: float = 0.0,
    q_vol_coupling: float = 0.0,
    # v7.8: Elite MC enhancements — dynamic leverage, liquidity stress
    leverage_dynamic_decay: float = 0.0,
    liq_stress_coeff: float = 0.0,
    # Story 1.4: Dual-frequency drift propagation
    phi_slow: float = 0.0,
    mu_slow_0: float = 0.0,
    # Story 3.4: Asset-class-aware per-step return cap
    return_cap: float = 0.30,
) -> Dict[str, np.ndarray]:
    """Unified MC engine with GJR-GARCH + jumps + Student-t via Numba kernel.

    v7.7: Full Tier 2 + Tier 3 MC integration -- all tuned params from
    unified Student-t models now flow through to MC simulation:

    Tier 2:
      - kappa_mean_rev + theta_long_var (vol mean-reversion)
      - crps_sigma_shrinkage (CRPS-optimal sigma tightening)
      - ms_sensitivity + q_stress_ratio (MS process noise)
      - rough_hurst (fractional vol memory)

    Tier 3:
      - sigma_eta (vol-of-vol noise)
      - t_df_asym (static two-piece ν)
      - regime_switch_prob (observation variance switching)
      - gamma_vov + vov_damping (VoV observation noise)
      - skew_score_sensitivity + skew_persistence (GAS dynamic skew)
      - loc_bias_var_coeff + loc_bias_drift_coeff (location bias)
      - q_vol_coupling (process noise volatility coupling)

    v7.6: Enriched MC — reuses tuned Student-t / unified model parameters:
      - GJR-GARCH leverage (asymmetric variance response to neg shocks)
      - variance_inflation (calibrated predictive variance β)
      - mu_drift (systematic drift bias correction)
      - alpha_asym + k_asym (asymmetric tail thickness)
      - risk_premium_sensitivity (ICAPM variance-conditional drift)

    v7.0: Single MC engine that replaces both run_regime_specific_mc (constant vol,
    no jumps) and _simulate_forward_paths (GARCH+jumps but only used for display).

    Now both p_up and exp_ret come from the SAME distribution, eliminating
    the dual-MC incoherence that was the #1 root cause of poor CRPS.

    Returns cumulative returns for ALL horizons 1..H_max simultaneously,
    enabling multi-horizon extraction from a single MC run.

    For exotic distributions (Hansen Skew-t, Contaminated Student-t),
    falls back to the original Python path since these can't be compiled
    into the Numba kernel.

    Args:
        mu_t: Current drift estimate
        P_t: Drift posterior variance
        phi: AR(1) drift persistence
        q: Process noise variance
        sigma2_step: Per-step observation variance
        H_max: Maximum forecast horizon (all horizons 1..H_max produced)
        n_paths: Number of MC paths
        nu: Degrees of freedom for Student-t (None for Gaussian)
        use_garch: Whether to use GARCH(1,1) variance evolution
        garch_omega, garch_alpha, garch_beta: GARCH parameters
        jump_intensity: Poisson jump rate per step
        jump_mean, jump_std: Jump size distribution N(mu_J, sigma_J^2)
        enable_jumps: Whether to include jump-diffusion
        seed: Random seed
        hansen_lambda: Hansen skew-t parameter (Python fallback)
        cst_nu_normal, cst_nu_crisis, cst_epsilon: CST parameters (Python fallback)

    Returns:
        Dict with:
        - 'returns': (H_max, n_paths) cumulative log returns
        - 'volatility': (H_max, n_paths) per-step sigma
        - 'samples_at_H': callable that extracts 1D samples at horizon H
    """
    # Detect exotic distributions that need Python path
    use_cst = (
        cst_nu_normal is not None and cst_nu_crisis is not None and
        cst_epsilon is not None and cst_epsilon > 0.001
    )
    use_hansen = (
        hansen_lambda is not None and nu is not None and
        abs(hansen_lambda) > 0.01
    )
    use_exotic = use_cst or use_hansen

    # Input validation
    mu_t = float(mu_t) if np.isfinite(mu_t) else 0.0
    P_t = float(max(P_t, 0.0)) if np.isfinite(P_t) else 0.0
    phi = float(phi) if np.isfinite(phi) else 1.0
    q = float(max(q, 0.0)) if np.isfinite(q) else 0.0
    sigma2_step = float(max(sigma2_step, 1e-12)) if np.isfinite(sigma2_step) else 1e-6
    H_max = int(max(H_max, 1))

    if nu is not None and not use_exotic:
        if not np.isfinite(nu) or nu <= 2.0:
            nu = None
        else:
            nu = float(np.clip(nu, 2.1, 500.0))

    rng = np.random.default_rng(seed)

    # ========================================================================
    # EXOTIC DISTRIBUTIONS: Fall back to Python path
    # ========================================================================
    if use_exotic:
        # Use original run_regime_specific_mc but call it for H_max
        # and reconstruct multi-horizon structure
        # This preserves Hansen/CST sampling accuracy
        cum_out = np.zeros((H_max, n_paths), dtype=np.float64)
        vol_out = np.zeros((H_max, n_paths), dtype=np.float64)

        sigma_step_val = math.sqrt(sigma2_step)

        # Sample initial drift from posterior
        if P_t > 0:
            if nu is not None and nu > 2.0:
                t_sc = math.sqrt(P_t * (nu - 2.0) / nu)
                mu_paths = mu_t + t_sc * rng.standard_t(df=nu, size=n_paths)
            else:
                mu_paths = rng.normal(loc=mu_t, scale=math.sqrt(P_t), size=n_paths)
        else:
            mu_paths = np.full(n_paths, mu_t, dtype=np.float64)

        q_std = math.sqrt(q) if q > 0 else 0.0
        h_t = np.full(n_paths, max(sigma2_step, 1e-8), dtype=np.float64)
        # v7.9: Dynamic GARCH variance cap — prevent Student-t + GARCH explosion
        _h_dyn_cap = max(25.0 * max(sigma2_step, 1e-8), 0.005)

        # Story 1.4: Dual-frequency drift — split into fast + slow
        _use_dual_freq = phi_slow > 0.0 and abs(mu_slow_0) > 0.0
        if _use_dual_freq:
            _mu_slow_paths = np.full(n_paths, mu_slow_0, dtype=np.float64)

        for t_step in range(H_max):
            sigma_t = np.sqrt(np.maximum(h_t, 1e-12))
            vol_out[t_step, :] = sigma_t

            # Observation noise — use actual Hansen/CST distributions
            if use_hansen and nu is not None and nu > 2.0:
                # Hansen skew-t: asymmetric tails via λ
                eps_sc = sigma_t * math.sqrt((nu - 2.0) / nu)
                eps = hansen_skew_t_rvs(
                    size=n_paths, nu=nu, lambda_=hansen_lambda,
                    random_state=rng
                ) * eps_sc
            elif use_cst and cst_nu_normal is not None and cst_nu_normal > 2.0:
                # Contaminated Student-t: mixture of normal + crisis tails
                eps_sc = sigma_t * math.sqrt((cst_nu_normal - 2.0) / cst_nu_normal)
                eps = contaminated_student_t_rvs(
                    size=n_paths, nu_normal=cst_nu_normal,
                    nu_crisis=cst_nu_crisis, epsilon=cst_epsilon,
                    mu=0.0, sigma=1.0, random_state=rng
                ) * eps_sc
            elif nu is not None and nu > 2.0:
                eps_sc = sigma_t * math.sqrt((nu - 2.0) / nu)
                eps = eps_sc * rng.standard_t(df=nu, size=n_paths)
            else:
                eps = sigma_t * rng.standard_normal(size=n_paths)

            e_t = eps

            # Jump component
            jump = np.zeros(n_paths, dtype=np.float64)
            if enable_jumps and jump_intensity > 0:
                n_jumps = rng.poisson(lam=jump_intensity, size=n_paths)
                for pidx in range(n_paths):
                    if n_jumps[pidx] > 0:
                        jsizes = rng.normal(loc=jump_mean, scale=jump_std, size=int(n_jumps[pidx]))
                        jump[pidx] = float(np.sum(jsizes))

            r_t = mu_paths + e_t + jump
            # Story 1.4: Add slow drift component to returns
            if _use_dual_freq:
                r_t = r_t + _mu_slow_paths
            # Story 3.4: Asset-class-aware per-step return cap
            r_t = np.clip(r_t, -return_cap, return_cap)
            if t_step == 0:
                cum_out[t_step, :] = r_t
            else:
                cum_out[t_step, :] = cum_out[t_step - 1, :] + r_t

            # GARCH evolution
            if use_garch:
                h_t = garch_omega + garch_alpha * (e_t ** 2) + garch_beta * h_t
                h_t = np.clip(h_t, 1e-12, _h_dyn_cap)

            # Drift evolution
            if q_std > 0:
                eta = rng.normal(loc=0.0, scale=q_std, size=n_paths)
            else:
                eta = np.zeros(n_paths, dtype=np.float64)
            mu_paths = phi * mu_paths + eta
            # Story 1.4: Slow drift deterministic decay
            if _use_dual_freq:
                _mu_slow_paths = phi_slow * _mu_slow_paths

        return {'returns': cum_out, 'volatility': vol_out}

    # ========================================================================
    # NUMBA PATH: Student-t or Gaussian with GARCH + jumps
    # ========================================================================
    try:
        from models.numba_kernels import unified_mc_simulate_kernel
    except ImportError:
        # Fallback import path
        try:
            import importlib
            nk = importlib.import_module('src.models.numba_kernels')
            unified_mc_simulate_kernel = nk.unified_mc_simulate_kernel
        except Exception:
            unified_mc_simulate_kernel = None

    nu_val = nu if nu is not None else 200.0  # >100 treated as Gaussian

    # Sample initial drift from posterior (in Python, before Numba kernel)
    if P_t > 0:
        if nu is not None and nu > 2.0 and nu < 100.0:
            t_sc = math.sqrt(P_t * (nu - 2.0) / nu)
            mu_start = mu_t + t_sc * rng.standard_t(df=nu, size=n_paths)
        else:
            mu_start = rng.normal(loc=mu_t, scale=math.sqrt(P_t), size=n_paths)
    else:
        mu_start = np.full(n_paths, mu_t, dtype=np.float64)

    # For the Numba kernel, we pass the mean of initial drift draws
    # The kernel handles drift evolution from there
    mu_now_for_kernel = float(np.mean(mu_start))

    # Pre-generate all random numbers (Numba kernel uses pre-generated arrays)
    z_normals = rng.standard_normal(size=(H_max, n_paths)).astype(np.float64)
    z_drift = rng.standard_normal(size=(H_max, n_paths)).astype(np.float64)
    z_jump_uniform = rng.uniform(size=(H_max, n_paths)).astype(np.float64)
    z_jump_normal = rng.standard_normal(size=(H_max, n_paths)).astype(np.float64)

    # Generate chi2(nu)/nu draws for Student-t
    if nu is not None and nu > 2.0 and nu < 100.0:
        # chi2(nu) = sum of nu standard_normal^2
        # For efficiency, use gamma distribution: chi2(nu) ~ Gamma(nu/2, 2)
        chi2_draws = rng.gamma(shape=nu_val / 2.0, scale=2.0, size=(H_max, n_paths))
        z_chi2 = (chi2_draws / nu_val).astype(np.float64)
    else:
        z_chi2 = np.ones((H_max, n_paths), dtype=np.float64)

    # Allocate output arrays
    cum_out = np.zeros((H_max, n_paths), dtype=np.float64)
    vol_out = np.zeros((H_max, n_paths), dtype=np.float64)

    h0 = float(max(sigma2_step, 1e-8))

    if unified_mc_simulate_kernel is not None:
        # v7.7: Precompute fractional weights for rough vol (outside Numba)
        frac_weights_arr = np.empty(0, dtype=np.float64)
        if rough_hurst > 0.001:
            d = rough_hurst - 0.5  # d < 0 for H < 0.5 (rough)
            max_lag = 50
            fw = np.zeros(max_lag, dtype=np.float64)
            fw[0] = 1.0
            for k_idx in range(1, max_lag):
                fw[k_idx] = fw[k_idx - 1] * (k_idx - 1 - d) / k_idx
            # Normalize to sum to 1
            fw_sum = np.sum(np.abs(fw))
            if fw_sum > 1e-12:
                fw = fw / fw_sum
            frac_weights_arr = fw

        # Use Numba kernel (v7.7: with all Tier 2+3 params)
        unified_mc_simulate_kernel(
            n_paths, H_max, mu_now_for_kernel, h0,
            phi, q, nu_val,
            use_garch, garch_omega, garch_alpha, garch_beta,
            jump_intensity, jump_mean, jump_std, enable_jumps,
            z_normals, z_chi2, z_drift,
            z_jump_uniform, z_jump_normal,
            cum_out, vol_out,
            # v7.6: Enriched MC params from tuned models
            garch_leverage,
            variance_inflation,
            mu_drift,
            alpha_asym,
            k_asym,
            risk_premium_sensitivity,
            # v7.7: Tier 2 params
            kappa_mean_rev,
            theta_long_var,
            crps_sigma_shrinkage,
            ms_sensitivity,
            q_stress_ratio,
            rough_hurst,
            frac_weights_arr,
            # v7.7: Tier 3 params
            sigma_eta,
            t_df_asym,
            regime_switch_prob,
            gamma_vov,
            vov_damping,
            skew_score_sensitivity,
            skew_persistence,
            loc_bias_var_coeff,
            loc_bias_drift_coeff,
            q_vol_coupling,
            # v7.8: Elite MC enhancements
            leverage_dynamic_decay,
            liq_stress_coeff,
            # Story 1.4: Dual-frequency drift
            phi_slow,
            mu_slow_0,
            # Story 3.4: Asset-class-aware return cap
            return_cap,
        )
        # Each path's returns are shifted by (mu_start[p] - mu_now_for_kernel)
        # This preserves the drift posterior spread while using Numba for the loop
        drift_offset = mu_start - mu_now_for_kernel
        for t_step in range(H_max):
            cum_out[t_step, :] += drift_offset * (t_step + 1)
    else:
        # Pure Python fallback (same logic as kernel, v7.7: all Tier 2+3 params)
        # v7.7: Apply variance_inflation, crps_sigma_shrinkage, and mu_drift
        h0_cal = h0 * variance_inflation * crps_sigma_shrinkage
        # v7.9: Dynamic GARCH variance cap
        _h_dyn_cap_py = max(25.0 * h0_cal, 0.005)
        mu_t_arr = mu_start.copy() + mu_drift
        h_t = np.full(n_paths, h0_cal, dtype=np.float64)
        # Story 1.4: Dual-frequency drift — slow component
        _use_dual_freq_py = phi_slow > 0.0 and abs(mu_slow_0) > 0.0
        _mu_slow_arr = np.full(n_paths, mu_slow_0, dtype=np.float64) if _use_dual_freq_py else None
        use_asym = (nu is not None and nu > 2.0 and nu < 100.0 and abs(alpha_asym) > 1e-8)
        use_rp = abs(risk_premium_sensitivity) > 1e-10
        use_kappa_py = kappa_mean_rev > 0.001 and theta_long_var > 1e-12
        use_ms_q_py = ms_sensitivity > 0.01 and q_stress_ratio > 1.01
        use_sigma_eta_py = sigma_eta > 0.005
        use_regime_sw_py = regime_switch_prob > 0.005
        use_loc_bias_py = abs(loc_bias_var_coeff) > 1e-6 or abs(loc_bias_drift_coeff) > 1e-6

        # Per-path state: regime switching stress prob
        p_stress_obs_arr = np.full(n_paths, 0.1, dtype=np.float64)
        # MS-q vol EMA state
        vol_ema_arr = h_t.copy()

        for t_step in range(H_max):
            sigma_t = np.sqrt(np.maximum(h_t, 1e-12))
            vol_out[t_step, :] = sigma_t

            if nu is not None and nu > 2.0 and nu < 100.0:
                raw_t = z_normals[t_step] / np.sqrt(np.maximum(z_chi2[t_step], 1e-8))
                if use_asym:
                    # v7.6: Asymmetric tail thickness per-sample
                    nu_eff = nu * (1.0 + alpha_asym * np.tanh(k_asym * raw_t))
                    nu_eff = np.clip(nu_eff, 2.5, 200.0)
                    t_var_eff = nu_eff / (nu_eff - 2.0)
                    eps = raw_t / np.sqrt(t_var_eff)
                else:
                    t_var_val = nu / (nu - 2.0)
                    eps = raw_t / math.sqrt(t_var_val)
            else:
                eps = z_normals[t_step]

            e_t = sigma_t * eps

            jump = np.zeros(n_paths, dtype=np.float64)
            if enable_jumps and jump_intensity > 0:
                mask = z_jump_uniform[t_step] < jump_intensity
                jump[mask] = jump_mean + jump_std * z_jump_normal[t_step, mask]

            # v7.6: Variance-conditional risk premium
            rp = risk_premium_sensitivity * h_t if use_rp else 0.0

            # v7.7: Location bias correction
            loc_bias = np.zeros(n_paths, dtype=np.float64)
            if use_loc_bias_py:
                if abs(loc_bias_var_coeff) > 1e-6 and theta_long_var > 1e-12:
                    loc_bias += loc_bias_var_coeff * (h_t - theta_long_var)
                if abs(loc_bias_drift_coeff) > 1e-6:
                    sign_mu = np.sign(mu_t_arr)
                    loc_bias += loc_bias_drift_coeff * sign_mu * np.sqrt(np.abs(mu_t_arr))

            r_t = mu_t_arr + rp + loc_bias + e_t + jump
            # Story 1.4: Add slow drift component
            if _use_dual_freq_py:
                r_t = r_t + _mu_slow_arr
            # Story 3.4: Asset-class-aware per-step return cap
            r_t = np.clip(r_t, -return_cap, return_cap)
            if t_step == 0:
                cum_out[t_step, :] = r_t
            else:
                cum_out[t_step, :] = cum_out[t_step - 1, :] + r_t

            # v7.7: GJR-GARCH with all Tier 2+3 enhancements
            if use_garch:
                e2 = e_t ** 2
                h_t = garch_omega + garch_alpha * e2 + garch_beta * h_t
                if garch_leverage > 1e-8:
                    h_t += garch_leverage * e2 * (e_t < 0).astype(np.float64)

                # Variance mean-reversion
                if use_kappa_py:
                    h_t = (1.0 - kappa_mean_rev) * h_t + kappa_mean_rev * theta_long_var

                # Vol-of-vol noise
                if use_sigma_eta_py:
                    z_std = np.abs(e_t) / np.maximum(sigma_t, 1e-8)
                    excess = np.maximum(z_std - 1.5, 0.0)
                    h_t += sigma_eta * excess * excess * h_t

                # Regime switching on observation variance
                if use_regime_sw_py:
                    z_rs = np.abs(e_t) / np.maximum(sigma_t, 1e-8)
                    ind_stress = (z_rs > 2.0).astype(np.float64)
                    p_stress_obs_arr = (1.0 - regime_switch_prob) * p_stress_obs_arr + regime_switch_prob * ind_stress
                    h_t *= (1.0 + p_stress_obs_arr * (math.sqrt(q_stress_ratio) - 1.0))

                h_t = np.clip(h_t, 1e-12, _h_dyn_cap_py)

            # v7.7: Drift evolution with MS process noise
            if use_ms_q_py:
                ewm_alpha_ms = 0.05
                vol_ema_arr = (1.0 - ewm_alpha_ms) * vol_ema_arr + ewm_alpha_ms * h_t
                vol_z = np.where(vol_ema_arr > 1e-12,
                                 (h_t - vol_ema_arr) / np.maximum(np.sqrt(vol_ema_arr), 1e-8),
                                 0.0)
                p_stress_ms = 1.0 / (1.0 + np.exp(-ms_sensitivity * vol_z))
                q_t = (1.0 - p_stress_ms) * q + p_stress_ms * q * q_stress_ratio
                drift_sigma_t = np.sqrt(np.maximum(q_t, 0.0))
                mu_t_arr = phi * mu_t_arr + drift_sigma_t * z_drift[t_step]
            else:
                drift_eta = z_drift[t_step] * math.sqrt(q) if q > 0 else np.zeros(n_paths)
                mu_t_arr = phi * mu_t_arr + drift_eta
            # Story 1.4: Slow drift deterministic decay
            if _use_dual_freq_py:
                _mu_slow_arr = phi_slow * _mu_slow_arr

    return {'returns': cum_out, 'volatility': vol_out}


# ─── Story 3.6: MC Path Diagnostics and Anomaly Detection ──────────────
MC_EXTREME_THRESHOLD = 5.0      # |cum_return| > 5.0 ≈ >500% return
MC_EXTREME_WARN_FRAC = 0.10    # warn + trim when >10% paths are extreme
MC_TRIM_PERCENTILE = 2.5       # trim at 2.5th/97.5th percentiles


class MCDiagnostics:
    """Lightweight diagnostics container for MC path quality."""

    __slots__ = ("n_paths", "n_nan", "n_extreme", "extreme_frac",
                 "median", "mean", "trimmed_mean", "mean_median_gap",
                 "used_trimmed")

    def __init__(self, n_paths: int, n_nan: int, n_extreme: int,
                 extreme_frac: float, median: float, mean: float,
                 trimmed_mean: float, mean_median_gap: float,
                 used_trimmed: bool):
        self.n_paths = n_paths
        self.n_nan = n_nan
        self.n_extreme = n_extreme
        self.extreme_frac = extreme_frac
        self.median = median
        self.mean = mean
        self.trimmed_mean = trimmed_mean
        self.mean_median_gap = mean_median_gap
        self.used_trimmed = used_trimmed

    def to_dict(self) -> Dict[str, float]:
        return {
            "n_paths": self.n_paths,
            "n_nan": self.n_nan,
            "n_extreme": self.n_extreme,
            "extreme_frac": round(self.extreme_frac, 6),
            "median": round(self.median, 8),
            "mean": round(self.mean, 8),
            "trimmed_mean": round(self.trimmed_mean, 8),
            "mean_median_gap": round(self.mean_median_gap, 6),
            "used_trimmed": self.used_trimmed,
        }


def diagnose_mc_paths(cum_out: np.ndarray, H: int,
                      asset_name: str = "",
                      extreme_threshold: float = MC_EXTREME_THRESHOLD,
                      warn_frac: float = MC_EXTREME_WARN_FRAC) -> MCDiagnostics:
    """Diagnose MC path quality at horizon H.

    Args:
        cum_out: (H_max, n_paths) cumulative log-return array
        H: target horizon (1-indexed)
        asset_name: for warning messages
        extreme_threshold: |return| above this is flagged
        warn_frac: fraction above which trimmed stats are used

    Returns:
        MCDiagnostics with counts, statistics, and trim flag.
    """
    if H < 1 or cum_out.ndim != 2 or H > cum_out.shape[0]:
        return MCDiagnostics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    final_returns = cum_out[H - 1, :]
    n_paths = final_returns.shape[0]
    if n_paths == 0:
        return MCDiagnostics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    # NaN / Inf detection
    finite_mask = np.isfinite(final_returns)
    n_nan = int(np.sum(~finite_mask))
    clean = final_returns[finite_mask]

    if len(clean) == 0:
        if n_nan > 0:
            warnings.warn(
                f"[MC DIAG] {asset_name} H={H}: ALL {n_paths} paths are NaN/Inf",
                RuntimeWarning,
            )
        return MCDiagnostics(n_paths, n_nan, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    # Extreme path detection
    n_extreme = int(np.sum(np.abs(clean) > extreme_threshold))
    extreme_frac = n_extreme / len(clean) if len(clean) > 0 else 0.0

    median_val = float(np.median(clean))
    mean_val = float(np.mean(clean))

    # Trimmed mean: exclude top/bottom 2.5% when extreme fraction is high
    used_trimmed = extreme_frac > warn_frac
    if used_trimmed:
        lo = np.percentile(clean, MC_TRIM_PERCENTILE)
        hi = np.percentile(clean, 100.0 - MC_TRIM_PERCENTILE)
        trimmed = clean[(clean >= lo) & (clean <= hi)]
        trimmed_mean = float(np.mean(trimmed)) if len(trimmed) > 0 else mean_val
        warnings.warn(
            f"[MC DIAG] {asset_name} H={H}: {n_extreme}/{len(clean)} "
            f"({extreme_frac:.1%}) extreme paths (|r|>{extreme_threshold}). "
            f"Using trimmed mean={trimmed_mean:.6f} vs raw mean={mean_val:.6f}",
            RuntimeWarning,
        )
    else:
        trimmed_mean = mean_val

    if n_nan > 0:
        warnings.warn(
            f"[MC DIAG] {asset_name} H={H}: {n_nan}/{n_paths} NaN/Inf paths excluded",
            RuntimeWarning,
        )

    mean_median_gap = abs(mean_val - median_val) / max(abs(median_val), 1e-6)

    return MCDiagnostics(
        n_paths=n_paths,
        n_nan=n_nan,
        n_extreme=n_extreme,
        extreme_frac=extreme_frac,
        median=median_val,
        mean=mean_val,
        trimmed_mean=trimmed_mean,
        mean_median_gap=mean_median_gap,
        used_trimmed=used_trimmed,
    )


# Temperature must match tune.py's DEFAULT_ENTROPY_LAMBDA = 0.05
# Using T=1.0 would flatten posteriors by 20×, making model selection near-uniform.
DEFAULT_POSTERIOR_TEMPERATURE = 0.05


def compute_model_posteriors_from_combined_score(
    models: Dict[str, Dict],
    temperature: float = DEFAULT_POSTERIOR_TEMPERATURE,
    min_weight_fraction: float = 0.01,
    epsilon: float = 1e-10,
) -> Tuple[Dict[str, float], Dict]:
    """
    Convert combined scores into normalized posterior weights with entropy floor.

    This is the EPISTEMIC WEIGHTING step that ensures Hyvärinen scores
    directly influence signal generation.

    The combined_score is the entropy-regularized standardized score:
        combined_score = w_crps * CRPS_std + w_pit * PIT_dev_std + w_berk * Berk_std
                       + w_tail * Tail_std + w_mad * MAD_std + w_ad * AD_dev_std

    Lower combined_score = better model.

    To get normalized posteriors we use softmax over NEGATED scores:
        p(m) = exp(-combined_score_m / T) / Σ_k exp(-combined_score_k / T)

    An entropy floor is applied to prevent belief collapse:
        w_m = max(w_m, min_weight_fraction / n_models)

    This ensures dominated models retain some probability mass, preventing
    overconfident allocations during regime transitions.

    Args:
        models: Dictionary mapping model_name -> model_params dict
                Each model_params must have 'combined_score'
        temperature: Softmax temperature (1.0 = standard, <1 = sharper, >1 = smoother)
        min_weight_fraction: Minimum total mass to uniform (0.01 = 1%)
        epsilon: Small constant to prevent zero weights

    Returns:
        Tuple of:
        - Dictionary mapping model_name -> posterior weight p(m)
        - Metadata dict
    """
    metadata = {
        "method": "combined",
        "temperature": temperature,
        "min_weight_fraction": min_weight_fraction,
    }

    # Extract valid models with combined scores
    valid_models = {}
    for model_name, model_params in models.items():
        if not isinstance(model_params, dict):
            continue
        if not model_params.get('fit_success', True):
            continue

        combined_score = model_params.get('combined_score')
        if combined_score is not None and np.isfinite(combined_score):
            valid_models[model_name] = combined_score

    if not valid_models:
        return {}, metadata

    # Convert to arrays for softmax
    model_names = list(valid_models.keys())
    scores = np.array([valid_models[m] for m in model_names])
    n_models = len(model_names)

    # Softmax over NEGATED scores (lower score = better = higher weight)
    # With numerical stabilization
    neg_scores = -scores / temperature
    neg_scores = neg_scores - neg_scores.max()  # Numerical stability

    weights = np.exp(neg_scores)
    weights = np.maximum(weights, epsilon)
    weights = weights / weights.sum()

    # =========================================================================
    # ENTROPY FLOOR: Prevent belief collapse
    # =========================================================================
    # Ensure each model has at least min_weight_fraction / n_models weight.
    # This prevents overconfident allocations during regime transitions or
    # when models happen to agree on similar scores.
    # =========================================================================
    min_weight_per_model = min_weight_fraction / max(n_models, 1)
    weights = np.maximum(weights, min_weight_per_model)
    weights = weights / weights.sum()  # Re-normalize after floor

    return dict(zip(model_names, weights)), metadata







# -------------------------
# Backtest-safe feature view (no look-ahead)
# -------------------------

def shift_features(feats: Dict[str, pd.Series], lag: int = 1) -> Dict[str, pd.Series]:
    """Return a copy of features dict with time-series shifted by `lag` days to remove look-ahead.
    Use for backtesting so that features at date t only use information available up to t−lag.
    Only series keys are shifted; scalar/meta entries are passed through.
    """
    if feats is None:
        return {}
    lag = int(max(0, lag))
    if lag == 0:
        return dict(feats)
    keys_to_shift = {
        # drifts
        "mu", "mu_post", "mu_blend", "mu_kf", "mu_final",
        # vols and regimes
        "vol_fast", "vol_slow", "vol", "vol_regime",
        # trend/momentum/stretch
        "sma200", "trend_z", "z5", "mom21", "mom63", "mom126", "mom252",
        # tails
        "skew", "nu",
        # base series for reference
        "ret",
    }
    shifted: Dict[str, pd.Series] = {}
    for k, v in feats.items():
        if isinstance(v, pd.Series) and k in keys_to_shift:
            try:
                shifted[k] = v.shift(lag)
            except Exception:
                shifted[k] = v
        else:
            # pass-through (px and any non-shifted helper)
            shifted[k] = v
    return shifted


def make_features_views(feats: Dict[str, pd.Series]) -> Dict[str, Dict[str, pd.Series]]:
    """Convenience wrapper to expose both live and backtest-safe views.
    - live: unshifted (as-of) features suitable for real-time use
    - bt:   shifted by 1 day (no look-ahead) for backtesting
    """
    return {
        "live": feats,
        "bt": shift_features(feats, lag=1),
    }


# =============================================================================
# NOTE: Legacy single-model MC functions removed
# =============================================================================
# The following functions were removed as they assume flat parameters (q, phi, nu)
# and are not compatible with the new BMA architecture:
#
#   - edge_for_horizon() - analytic z-score approximation (not used)
#   - posterior_predictive_mc_probability() - single-model MC (replaced by BMA)
#   - compute_expected_utility() - single-model EU (EU now computed inline from BMA samples)
#
# The BMA path uses:
#   - run_regime_specific_mc() - per-regime MC with model-specific params
#   - bayesian_model_average_mc() - full BMA mixture over regimes and models
#   - Inline EU computation in latest_signals() from r_samples
# =============================================================================


def _simulate_forward_paths(feats: Dict[str, pd.Series], H_max: int, n_paths: int = 3000, phi: float = 0.95, kappa: float = 1e-4, process_noise_q: float = 0.0, return_cap: float = 0.30) -> Dict[str, np.ndarray]:
    """Monte-Carlo forward simulation of cumulative log returns and volatility over 1..H_max.
    - Drift evolves as AR(1): mu_{t+1} = phi * mu_t + eta_t,  eta ~ N(0, q)
    - Volatility evolves via GARCH(1,1) when available; else held constant.
    - Innovations are Student-t with global df (nu_hat) scaled to unit variance.
    - Jump-diffusion (Merton model): captures discontinuous gap risk via rare large moves.

    Pillar 1 integration: Drift uncertainty from Kalman filter (var_kf) is propagated
    into process noise q, widening forecast confidence intervals when drift is uncertain.

    Level-7 parameter uncertainty: if PARAM_UNC environment variable is set to
    'sample' (default) and garch_params contains a covariance matrix, we sample
    (omega, alpha, beta) per path from N(theta_hat, Cov) with constraints, which
    widens confidence during regime shifts and narrows during stability.

    Stochastic volatility: Tracks full h_t (variance) trajectories across paths,
    enabling posterior uncertainty bands for volatility forecasts.

    Level-7 jump-diffusion: Merton model adds discontinuous jumps to capture gap risk:
        dS/S = μ dt + σ dW + J dN
    Where:
        - dW: continuous Brownian motion (Student-t innovations)
        - dN: Poisson process with intensity λ (jump arrival rate)
        - J: jump size ~ N(μ_J, σ_J²) (typically negative for crash risk)
    Jump parameters calibrated from historical returns: count large moves (>3σ) as jumps.

    Returns:
        Dictionary with:
            - 'returns': array of shape (H_max, n_paths) with cumulative log returns
            - 'volatility': array of shape (H_max, n_paths) with volatility (sigma_t = sqrt(h_t))
    """
    # Inputs at 'now'
    ret_idx = feats.get("ret", pd.Series(dtype=float)).index
    if ret_idx is None or len(ret_idx) == 0:
        return {
            'returns': np.zeros((H_max, n_paths), dtype=float),
            'volatility': np.zeros((H_max, n_paths), dtype=float)
        }
    mu_series = feats.get("mu_post")
    if not isinstance(mu_series, pd.Series) or mu_series.empty:
        mu_series = feats.get("mu")
    vol_series = feats.get("vol")
    if not isinstance(vol_series, pd.Series) or vol_series.empty or not isinstance(mu_series, pd.Series) or mu_series.empty:
        return {
            'returns': np.zeros((H_max, n_paths), dtype=float),
            'volatility': np.zeros((H_max, n_paths), dtype=float)
        }
    mu_now = float(mu_series.iloc[-1]) if len(mu_series) else 0.0
    vol_now = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
    vol_now = float(max(vol_now, 1e-6))

    # Pillar 1: Extract Kalman drift uncertainty for proper uncertainty propagation
    var_kf_series = feats.get("var_kf")
    if isinstance(var_kf_series, pd.Series) and not var_kf_series.empty:
        var_kf_now = float(var_kf_series.iloc[-1])
        var_kf_now = float(max(var_kf_now, 0.0))
    else:
        var_kf_now = 0.0

    # Tail parameter (global nu) with posterior uncertainty
    nu_hat_series = feats.get("nu_hat")
    nu_info = feats.get("nu_info", {})

    if isinstance(nu_hat_series, pd.Series) and not nu_hat_series.empty:
        nu_hat = float(nu_hat_series.iloc[-1])
    else:
        nu_hat, _ = _tail2("nu", 50.0)
        if not np.isfinite(nu_hat):
            nu_hat = 50.0
    nu_hat = float(np.clip(nu_hat, 4.5, 500.0))

    # Extract standard error for ν (Tier 2: posterior parameter variance)
    se_nu = None
    if isinstance(nu_info, dict) and "se_nu" in nu_info:
        se_nu_val = nu_info.get("se_nu", float("nan"))
        if np.isfinite(se_nu_val) and se_nu_val > 0:
            se_nu = float(se_nu_val)

    # Determine if ν sampling is enabled (Tier 2: propagate tail parameter uncertainty)
    nu_sample_mode = os.getenv("NU_SAMPLE", "true").strip().lower() == "true"

    # Sample ν per path if uncertainty available and sampling enabled
    if nu_sample_mode and se_nu is not None and se_nu > 0:
        rng = np.random.default_rng()
        # Sample from N(nu_hat, se_nu²) and clip to valid range
        nu_samples = rng.normal(loc=nu_hat, scale=se_nu, size=n_paths)
        nu_samples = np.clip(nu_samples, 4.5, 500.0)
    else:
        # Use point estimate for all paths
        nu_samples = np.full(n_paths, nu_hat, dtype=float)

    # GARCH params
    garch_params = feats.get("garch_params", {}) or {}
    use_garch = isinstance(garch_params, dict) and all(k in garch_params for k in ("omega", "alpha", "beta"))

    # Determine parameter uncertainty mode
    param_unc_mode = os.getenv("PARAM_UNC", "sample").strip().lower()
    if param_unc_mode not in ("none", "sample"):
        param_unc_mode = "sample"

    # Build per-path parameters (possibly sampled)
    if use_garch:
        base_theta = np.array([
            float(max(garch_params.get("omega", 0.0), 1e-12)),
            float(np.clip(garch_params.get("alpha", 0.0), 0.0, 0.999)),
            float(np.clip(garch_params.get("beta", 0.0), 0.0, 0.999)),
        ], dtype=float)
        cov = garch_params.get("cov")
        if isinstance(cov, list):
            try:
                cov = np.array(cov, dtype=float)
            except Exception:
                cov = None
        # Sample theta per path if enabled and covariance available
        if (param_unc_mode == "sample") and (cov is not None) and np.shape(cov) == (3, 3):
            rng = np.random.default_rng()
            try:
                thetas = rng.multivariate_normal(mean=base_theta, cov=cov, size=n_paths).astype(float)
            except Exception:
                # Fall back to eigen-decomposition sampling with small regularization
                try:
                    eigvals, eigvecs = np.linalg.eigh(0.5*(cov+cov.T) + 1e-12*np.eye(3))
                    eigvals = np.clip(eigvals, 0.0, None)
                    z = rng.normal(size=(n_paths, 3)) * np.sqrt(eigvals)
                    thetas = (z @ eigvecs.T) + base_theta
                except Exception:
                    thetas = np.tile(base_theta, (n_paths, 1))
            # Enforce constraints; replace invalid draws with base_theta
            omega_s = thetas[:, 0]
            alpha_s = thetas[:, 1]
            beta_s  = thetas[:, 2]
            # Fix obvious violations
            omega_s = np.maximum(omega_s, 1e-12)
            alpha_s = np.clip(alpha_s, 0.0, 0.999)
            beta_s  = np.clip(beta_s, 0.0, 0.999)
            # Enforce alpha+beta < 0.999 by shrinking both toward base proportionally
            ab = alpha_s + beta_s
            viol = ab >= 0.999
            if np.any(viol):
                # target sum slightly below 1
                target = 0.998
                scale = target / np.maximum(ab[viol], 1e-12)
                alpha_s[viol] *= scale
                beta_s[viol] *= scale
            omega_paths = omega_s
            alpha_paths = alpha_s
            beta_paths = beta_s
        else:
            omega_paths = np.full(n_paths, base_theta[0], dtype=float)
            alpha_paths = np.full(n_paths, base_theta[1], dtype=float)
            beta_paths  = np.full(n_paths, base_theta[2], dtype=float)
    else:
        omega_paths = np.zeros(n_paths, dtype=float)
        alpha_paths = np.zeros(n_paths, dtype=float)
        beta_paths  = np.zeros(n_paths, dtype=float)

    # Drift evolution noise: use process noise q from tune cache (March 2026)
    # Previously used P_t (posterior variance ~1e-4) which is 100x too large.
    # P_t is filtering uncertainty about today's state, not how drift evolves.
    # q (~1e-6) is the actual state-transition noise from the Kalman model.
    if process_noise_q > 0:
        drift_unc_now = process_noise_q
    else:
        # Fallback: use var_kf but cap it to avoid excess diffusion
        drift_unc_now = min(max(var_kf_now, 1e-10), 1e-5)

    h0 = vol_now ** 2

    # Level-7 Jump-Diffusion: Calibrate jump parameters from historical returns
    # Detect large moves (>3σ) as empirical jumps to estimate:
    #   - λ (jump intensity): frequency of jumps per day
    #   - μ_J (jump mean): average jump size
    #   - σ_J (jump std): volatility of jump sizes
    jump_intensity = 0.0
    jump_mean = 0.0
    jump_std = 0.05
    enable_jumps = os.getenv("ENABLE_JUMPS", "true").strip().lower() == "true"

    if enable_jumps:
        try:
            # Get historical returns for calibration
            ret_hist = feats.get("ret", pd.Series(dtype=float))
            vol_hist = feats.get("vol", pd.Series(dtype=float))

            if isinstance(ret_hist, pd.Series) and isinstance(vol_hist, pd.Series) and len(ret_hist) >= 252:
                # Align returns and volatility
                df_jump = pd.concat([ret_hist, vol_hist], axis=1, join='inner').dropna()
                if len(df_jump) >= 252:
                    df_jump.columns = ['ret', 'vol']

                    # Identify jumps: returns that exceed 3σ threshold (outliers)
                    # Standardize returns by conditional volatility
                    z_scores = df_jump['ret'] / df_jump['vol']
                    jump_threshold = 3.0
                    jump_mask = np.abs(z_scores) > jump_threshold

                    n_jumps = int(np.sum(jump_mask))
                    n_days = len(df_jump)

                    if n_jumps > 0:
                        # Jump intensity: λ = frequency of jumps per day
                        jump_intensity = float(n_jumps / n_days)

                        # Jump sizes: extract returns on jump days
                        jump_returns = df_jump.loc[jump_mask, 'ret'].values

                        # Jump mean and std (typically negative mean for crash risk)
                        jump_mean = float(np.mean(jump_returns))
                        jump_std = float(np.std(jump_returns))

                        # Floor jump std to avoid degenerate case
                        jump_std = float(max(jump_std, 0.01))
                    else:
                        # No historical jumps detected: use symmetric defaults
                        # (no evidence of asymmetric crashes → don't inject one)
                        jump_intensity = 0.01  # ~2.5 jumps per year
                        jump_mean = 0.0  # symmetric: no directional bias
                        jump_std = 0.05
        except Exception:
            # Fallback to symmetric defaults if calibration fails
            jump_intensity = 0.01
            jump_mean = 0.0  # symmetric: no directional bias
            jump_std = 0.05

    # Initialize state arrays (vectorized across paths)
    cum = np.zeros((H_max, n_paths), dtype=float)
    vol_paths = np.zeros((H_max, n_paths), dtype=float)  # Track volatility (sigma_t) at each horizon
    mu_t = np.full(n_paths, mu_now, dtype=float)
    h_t = np.full(n_paths, max(h0, 1e-8), dtype=float)
    # v7.9: Dynamic GARCH variance cap
    _h_dyn_cap_sim = max(25.0 * max(h0, 1e-8), 0.005)

    rng = np.random.default_rng()

    for t in range(H_max):
        # Student-t shocks standardized to unit variance (continuous component)
        # Tier 2: Use path-specific ν samples for proper tail parameter uncertainty
        # Draw Student-t per path with its own degrees of freedom
        z = np.zeros(n_paths, dtype=float)
        for path_idx in range(n_paths):
            nu_path = nu_samples[path_idx]
            # Draw from Student-t with df=nu_path and scale to unit variance
            z_raw = rng.standard_t(df=nu_path)
            # Variance of t(ν) is ν/(ν-2) for ν>2
            if nu_path > 2.0:
                t_var_path = nu_path / (nu_path - 2.0)
                t_scale_path = math.sqrt(t_var_path)
                z[path_idx] = float(z_raw / t_scale_path)
            else:
                # Edge case: use raw draw for very low ν (shouldn't happen with clipping)
                z[path_idx] = float(z_raw)

        eps = z
        sigma_t = np.sqrt(np.maximum(h_t, 1e-12))
        e_t = sigma_t * eps

        # Level-7 Jump-Diffusion: Add discontinuous jump component
        # Merton model: dS/S = μ dt + σ dW + J dN
        jump_component = np.zeros(n_paths, dtype=float)
        if enable_jumps and jump_intensity > 0:
            # Poisson arrivals: number of jumps in this time step
            # For daily data, dt=1, so intensity per step = jump_intensity
            n_jumps = rng.poisson(lam=jump_intensity, size=n_paths)

            # For paths with jumps, draw jump sizes from N(μ_J, σ_J²)
            # Total jump = sum of all jumps in this step (if multiple)
            for path_idx in range(n_paths):
                if n_jumps[path_idx] > 0:
                    # Draw jump sizes (log returns)
                    jump_sizes = rng.normal(loc=jump_mean, scale=jump_std, size=int(n_jumps[path_idx]))
                    jump_component[path_idx] = float(np.sum(jump_sizes))

        # Total return: continuous (drift + diffusion) + jumps
        r_t = mu_t + e_t + jump_component
        # Story 3.4: Asset-class-aware per-step return cap
        r_t = np.clip(r_t, -return_cap, return_cap)

        # Accumulate log return
        if t == 0:
            cum[t, :] = r_t
        else:
            cum[t, :] = cum[t-1, :] + r_t
        # Store volatility at this horizon (stochastic volatility tracking)
        vol_paths[t, :] = sigma_t
        # Evolve volatility via GARCH or hold constant on fallback
        if use_garch:
            h_t = omega_paths + alpha_paths * (e_t ** 2) + beta_paths * h_t
            h_t = np.clip(h_t, 1e-12, _h_dyn_cap_sim)
        # Evolve drift via AR(1) using posterior drift uncertainty only
        eta = rng.normal(loc=0.0, scale=math.sqrt(drift_unc_now), size=n_paths)
        mu_t = phi * mu_t + eta

    return {
        'returns': cum,
        'volatility': vol_paths
    }


