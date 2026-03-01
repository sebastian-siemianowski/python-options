"""
===============================================================================
GAUSSIAN DRIFT MODEL — Kalman Filter with Gaussian Observation Noise
===============================================================================

Implements a local-level state-space model for drift estimation:

    State equation:    μ_t = μ_{t-1} + w_t,  w_t ~ N(0, q)
    Observation:       r_t = μ_t + ε_t,      ε_t ~ N(0, c·σ²_t)

Parameters:
    q: Process noise variance (drift evolution uncertainty)
    c: Observation noise scale (multiplier on EWMA variance)

The model is estimated via cross-validated MLE with Bayesian regularization.

Unified Gaussian Pipeline (February 2026)
=========================================
GaussianUnifiedConfig + optimize_params_unified + filter_and_calibrate
brings Gaussian models to the same calibration quality as unified Student-t
by adding GJR-GARCH variance dynamics, EWM location correction,
walk-forward calibration, and proper Berkowitz testing — while keeping
Gaussian observation noise (no ν parameter).

    optimize_params_unified — Stage Dependency Chain
    ================================================
    Stage 1    (q, c, φ)       Base Kalman: process noise q, observation scale c,
                                persistence φ. L-BFGS-B with state regularization.
    Stage 1.5  (mom_w)         Momentum injection weight via CRPS grid search.
                                Degradation guard: 0.0 always competes.
    Stage 2    (β)             Variance inflation via PIT MAD grid search.
    Stage 3    (GARCH)         GJR-GARCH(1,1) on Kalman innovations.
    Stage 4    (EWM λ, σ_s)   Causal EWM location correction + CRPS σ shrinkage.
    Stage 4.5  (ω, α, β_gas)  GAS-Q adaptive process noise (Creal-Koopman-Lucas).
                                Degradation guard: must improve CRPS by >1%.
    Stage 5    (gw, λ_ρ, β_p) Walk-forward calibration: GARCH blend weight,
                                EWM decay, probit-variance correction.

    Momentum and GAS-Q are tuned INSIDE optimize_params_unified, making the
    legacy standalone models (kalman_gaussian_momentum, kalman_phi_gaussian_momentum,
    kalman_gaussian_momentum+GAS-Q) redundant.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest, norm

# Numba wrappers for JIT-compiled filters (optional performance enhancement)
try:
    from .numba_wrappers import (
        is_numba_available,
        run_gaussian_filter,
        run_phi_gaussian_filter,
        is_cv_kernel_available,
        run_gaussian_cv_test_fold,
    )
    _USE_NUMBA = is_numba_available()
    _USE_CV_KERNEL = is_cv_kernel_available()
except ImportError:
    _USE_NUMBA = False
    _USE_CV_KERNEL = False
    run_gaussian_filter = None
    run_phi_gaussian_filter = None
    run_gaussian_cv_test_fold = None

# Pre-computed constants
_LOG_2PI = math.log(2.0 * math.pi)
_LOG10_C_TARGET = math.log10(0.9)


def _fast_ks_statistic(pit_values):
    """Lightweight KS statistic against Uniform(0,1) without scipy overhead."""
    n = len(pit_values)
    if n == 0:
        return 1.0
    pit_sorted = np.sort(pit_values)
    d_plus = np.max(np.arange(1, n + 1) / n - pit_sorted)
    d_minus = np.max(pit_sorted - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


# =============================================================================
# GAUSSIAN UNIFIED CONFIG
# =============================================================================

class GaussianUnifiedConfig:
    """
    Configuration for Unified Gaussian Model.

    Mirrors UnifiedStudentTConfig but with Gaussian-only fields:
      1. Core Kalman: q, c, φ
      2. Variance inflation β for PIT calibration
      3. GJR-GARCH(1,1) on innovations for dynamic variance
      4. EWM causal location correction
      5. CRPS σ shrinkage
      6. Pre-calibrated walk-forward params (Stage 5)

    No ν parameter — observation noise is Gaussian throughout.
    """

    # Core Kalman
    q: float = 1e-6
    c: float = 1.0
    phi: float = 0.0

    # Variance inflation β: S_cal = S_pred × β
    variance_inflation: float = 1.0

    # Mean drift correction: μ_pred += mu_drift
    mu_drift: float = 0.0

    # GJR-GARCH(1,1) on innovations
    garch_omega: float = 0.0
    garch_alpha: float = 0.0
    garch_beta: float = 0.0
    garch_leverage: float = 0.0
    garch_unconditional_var: float = 1e-4

    # Causal EWM location correction: λ ∈ (0.90, 0.97), 0 = disabled
    crps_ewm_lambda: float = 0.0

    # CRPS σ shrinkage: σ_crps = σ × α, α ∈ [0.5, 1.0]
    crps_sigma_shrinkage: float = 1.0

    # Pre-calibrated walk-forward CV params (read-only at inference)
    calibrated_gw: float = 0.0           # GARCH blend weight
    calibrated_lambda_rho: float = 0.985  # EWM decay
    calibrated_beta_probit_corr: float = 1.0  # Probit β correction

    # Data-driven bounds (from auto_configure)
    q_min: float = 1e-8
    c_min: float = 0.01
    c_max: float = 10.0

    def __init__(self, **kwargs):
        self.q = 1e-6
        self.c = 1.0
        self.phi = 0.0
        self.variance_inflation = 1.0
        self.mu_drift = 0.0
        self.garch_omega = 0.0
        self.garch_alpha = 0.0
        self.garch_beta = 0.0
        self.garch_leverage = 0.0
        self.garch_unconditional_var = 1e-4
        self.crps_ewm_lambda = 0.0
        self.crps_sigma_shrinkage = 1.0
        self.calibrated_gw = 0.0
        self.calibrated_lambda_rho = 0.985
        self.calibrated_beta_probit_corr = 1.0
        self.momentum_weight = 0.0
        self.gas_q_omega = 0.0
        self.gas_q_alpha = 0.0
        self.gas_q_beta = 0.0
        self.q_min = 1e-8
        self.c_min = 0.01
        self.c_max = 10.0
        self.momentum_lookbacks = [5, 10, 20, 60]
        # Chi-squared + PIT-var calibration (March 2026 — asset-adaptive)
        self.chisq_ewm_lambda = 0.98     # Default ~35d half-life
        self.pit_var_lambda = 0.97        # PIT-var EWM decay
        self.pit_var_dz_lo = 0.30         # Dead-zone: start correcting
        self.pit_var_dz_hi = 0.55         # Dead-zone: full correction
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def momentum_enabled(self) -> bool:
        return abs(self.momentum_weight) > 1e-10

    @property
    def gas_q_enabled(self) -> bool:
        return (abs(self.gas_q_alpha) > 1e-10 or abs(self.gas_q_beta) > 1e-10)

    @classmethod
    def auto_configure(cls, returns: np.ndarray, vol: np.ndarray) -> 'GaussianUnifiedConfig':
        """Data-driven initial bounds for Gaussian unified model."""
        config = cls()
        ret_var = float(np.var(returns))
        vol_var_median = float(np.median(vol ** 2))

        config.q_min = max(1e-10, 0.001 * vol_var_median)
        config.c_min = max(0.01, 0.1 * ret_var / (vol_var_median + 1e-12))
        config.c_max = min(10.0, 10.0 * ret_var / (vol_var_median + 1e-12))
        if config.c_min >= config.c_max:
            config.c_min, config.c_max = 0.1, 5.0

        return config


class GaussianDriftModel:
    """Encapsulates Gaussian Kalman drift model logic for modular reuse."""

    @staticmethod
    def filter(returns: np.ndarray, vol: np.ndarray, q: float, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Optimized Kalman filter for drift estimation.
        
        Performance optimizations (February 2026):
        - Pre-compute R array once
        - Pre-compute log_2pi constant
        - Use np.empty instead of np.zeros
        - Ensure contiguous array access
        """
        n = len(returns)
        
        # Convert to contiguous float64 arrays once
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)

        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, 'item') else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, 'item') else float(c)
        
        # Pre-compute constants
        log_2pi = np.log(2 * np.pi)
        
        # Pre-compute R array (vectorized)
        R = c_val * (vol * vol)

        mu = 0.0
        P = 1e-4

        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = mu
            P_pred = P + q_val

            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12

            K = P_pred / S
            innovation = returns[t] - mu_pred

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            # Inlined log-likelihood
            log_likelihood += -0.5 * (log_2pi + np.log(S) + (innovation * innovation) / S)

        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def filter_phi(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Kalman filter with persistent/mean-reverting drift μ_t = φ μ_{t-1} + w_t.
        
        Uses Numba JIT-compiled kernel when available (10-50× speedup).
        """
        # Try Numba-accelerated version first
        if _USE_NUMBA:
            try:
                return run_phi_gaussian_filter(returns, vol, q, c, phi)
            except Exception:
                pass  # Fall through to Python implementation
        
        return GaussianDriftModel._filter_phi_python(returns, vol, q, c, phi)
    
    @staticmethod
    def _filter_phi_python(returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pure Python implementation of φ-Gaussian filter (for fallback and testing)."""
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        log_likelihood = 0.0

        for t in range(n):
            mu_pred = phi_val * mu
            P_pred = (phi_val ** 2) * P + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            R = c_val * (vol_scalar ** 2)

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            K = P_pred / S

            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            P = float(max(P, 1e-12))

            mu_filtered[t] = mu
            P_filtered[t] = P

            log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)

        return mu_filtered, P_filtered, float(log_likelihood)

    @staticmethod
    def filter_phi_with_predictive(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        ELITE FIX: φ-Gaussian filter returning PREDICTIVE values for proper PIT.
        
        Same as filter_phi but also returns:
            mu_pred[t] = φ × μ_{t-1}     (BEFORE seeing y_t)
            S_pred[t] = P_pred + R_t      (BEFORE seeing y_t)
        
        For proper PIT computation.
        """
        n = len(returns)
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        mu_pred_arr = np.zeros(n)  # PREDICTIVE mean
        S_pred_arr = np.zeros(n)   # PREDICTIVE variance
        log_likelihood = 0.0

        for t in range(n):
            # Prediction step (BEFORE seeing y_t)
            mu_pred = phi_val * mu
            P_pred = (phi_val ** 2) * P + q_val

            vol_t = vol[t]
            vol_scalar = float(vol_t) if np.ndim(vol_t) == 0 else float(vol_t.item())
            R = c_val * (vol_scalar ** 2)

            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12
            
            # Store PREDICTIVE values
            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S

            ret_t = returns[t]
            r_val = float(ret_t) if np.ndim(ret_t) == 0 else float(ret_t.item())
            innovation = r_val - mu_pred

            # Update step (AFTER seeing y_t)
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            P = float(max(P, 1e-12))

            mu_filtered[t] = mu
            P_filtered[t] = P

            log_likelihood += -0.5 * (np.log(2 * np.pi * S) + (innovation ** 2) / S)

        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)

    @staticmethod
    def pit_ks_predictive(
        returns: np.ndarray,
        mu_pred: np.ndarray,
        S_pred: np.ndarray,
    ) -> Tuple[float, float]:
        """
        ELITE FIX: Proper PIT/KS using PREDICTIVE distribution for Gaussian.
        
        For Gaussian, scale = sqrt(S_pred) (no adjustment needed).
        """
        returns_flat = np.asarray(returns).flatten()
        mu_pred_flat = np.asarray(mu_pred).flatten()
        S_pred_flat = np.asarray(S_pred).flatten()
        
        forecast_std = np.sqrt(np.maximum(S_pred_flat, 1e-20))
        forecast_std = np.where(forecast_std < 1e-10, 1e-10, forecast_std)
        
        standardized = (returns_flat - mu_pred_flat) / forecast_std
        
        valid_mask = np.isfinite(standardized)
        if not np.any(valid_mask):
            return 1.0, 0.0
        
        standardized_clean = standardized[valid_mask]
        pit_values = norm.cdf(standardized_clean)
        
        if len(pit_values) < 2:
            return 1.0, 0.0
        
        ks_result = kstest(pit_values, 'uniform')
        return float(ks_result.statistic), float(ks_result.pvalue)

    @staticmethod
    def filter_augmented(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        exogenous_input: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Gaussian filter with exogenous input (phi=1).
        
        STATE-EQUATION INTEGRATION (Elite Upgrade - February 2026):
        Delegates to filter_phi_augmented with phi=1.0.
        """
        return GaussianDriftModel.filter_phi_augmented(
            returns, vol, q, c, phi=1.0, exogenous_input=exogenous_input
        )

    @staticmethod
    def filter_phi_augmented(
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float = 1.0,
        exogenous_input: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        φ-Gaussian filter with exogenous input in state equation.
        
        STATE-EQUATION INTEGRATION (Elite Upgrade - February 2026):
            μ_t = φ × μ_{t-1} + u_t + w_t
            r_t = μ_t + ε_t,  ε_t ~ N(0, c×σ²)
        
        EXPERT VALIDATED: mu_pred includes u_t for coherent likelihood.
        
        Args:
            returns: Array of returns
            vol: Array of volatility estimates
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            exogenous_input: Array of u_t values (α×MOM - β×MR)
            
        Returns:
            Tuple of (mu_filtered, P_filtered, log_likelihood)
        """
        n = len(returns)
        
        # Convert to contiguous float64 arrays
        returns = np.ascontiguousarray(returns.flatten(), dtype=np.float64)
        vol = np.ascontiguousarray(vol.flatten(), dtype=np.float64)
        
        q_val = float(q) if np.ndim(q) == 0 else float(q.item()) if hasattr(q, "item") else float(q)
        c_val = float(c) if np.ndim(c) == 0 else float(c.item()) if hasattr(c, "item") else float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        phi_sq = phi_val * phi_val
        
        # Pre-compute constants
        log_2pi = np.log(2 * np.pi)
        
        # Pre-compute R array
        R = c_val * (vol * vol)
        
        # Allocate output arrays
        mu_filtered = np.empty(n, dtype=np.float64)
        P_filtered = np.empty(n, dtype=np.float64)
        
        # State initialization
        mu = 0.0
        P = 1e-4
        log_likelihood = 0.0
        
        # Main filter loop with exogenous input
        for t in range(n):
            # Exogenous input (KEY: injected into state equation)
            u_t = exogenous_input[t] if exogenous_input is not None and t < len(exogenous_input) else 0.0
            
            # Prediction step INCLUDES exogenous input
            mu_pred = phi_val * mu + u_t
            P_pred = phi_sq * P + q_val
            
            # Observation update
            S = P_pred + R[t]
            if S <= 1e-12:
                S = 1e-12
            
            innovation = returns[t] - mu_pred
            K = P_pred / S
            
            # State update
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12
            
            # Store filtered values
            mu_filtered[t] = mu
            P_filtered[t] = P
            
            # Log-likelihood (coherent with u_t)
            log_likelihood += -0.5 * (log_2pi + np.log(S) + (innovation * innovation) / S)
        
        return mu_filtered, P_filtered, float(log_likelihood)

    # =========================================================================
    # UNIFIED GAUSSIAN PIPELINE (February 2026)
    # =========================================================================

    @classmethod
    def optimize_params_unified(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        phi_mode: bool = True,
        train_frac: float = 0.7,
        asset_symbol: str = None,
        momentum_signal: np.ndarray = None,
    ) -> Tuple['GaussianUnifiedConfig', Dict]:
        """
        Staged optimization for unified Gaussian Kalman filter model.

        Orchestrates 5 stages in sequence, threading results forward.
        Each stage freezes upstream parameters and optimizes <= 3 new ones.

        Stage chain:
          1 (q,c,φ) → 2 (β) → 3 (GARCH) → 4 (EWM λ, σ_s) → 5 (gw, λ_ρ, β_p)

        Args:
            returns: Return series
            vol: EWMA/GK volatility series
            phi_mode: If True, estimate φ (AR(1) drift). If False, φ=1 (random walk).
            train_frac: Fraction for train/test split
            asset_symbol: Asset symbol for diagnostics

        Returns:
            Tuple of (GaussianUnifiedConfig, diagnostics_dict)
        """
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()

        config = GaussianUnifiedConfig.auto_configure(returns, vol)

        # ── ASSET-CLASS CALIBRATION PROFILE (March 2026) ──
        # Apply asset-class-specific chi² and PIT-var parameters.
        # Mirrors phi_student_t.py profile lookup.
        _GAUSSIAN_PROFILES = {
            'metals_gold':    {'chisq_ewm_lambda': 0.985, 'pit_var_lambda': 0.975, 'pit_var_dz_lo': 0.35, 'pit_var_dz_hi': 0.60},
            'metals_silver':  {'chisq_ewm_lambda': 0.96,  'pit_var_lambda': 0.95,  'pit_var_dz_lo': 0.25, 'pit_var_dz_hi': 0.50},
            'metals_other':   {'chisq_ewm_lambda': 0.97,  'pit_var_lambda': 0.96,  'pit_var_dz_lo': 0.30, 'pit_var_dz_hi': 0.55},
            'high_vol_equity':{'chisq_ewm_lambda': 0.94,  'pit_var_lambda': 0.94,  'pit_var_dz_lo': 0.20, 'pit_var_dz_hi': 0.45},
            'forex':          {'chisq_ewm_lambda': 0.99,  'pit_var_lambda': 0.98,  'pit_var_dz_lo': 0.35, 'pit_var_dz_hi': 0.60},
        }
        _gaussian_asset_class = None
        if asset_symbol is not None:
            _sym = asset_symbol.strip().upper()
            # Import asset-class detection from phi_student_t
            try:
                from models.phi_student_t import _detect_asset_class
                _gaussian_asset_class = _detect_asset_class(_sym)
            except ImportError:
                # Inline fallback detection
                _METALS_GOLD = {'GC=F', 'GLD', 'IAU', 'XAUUSD=X'}
                _METALS_SILVER = {'SI=F', 'SLV', 'XAGUSD=X'}
                _METALS_OTHER = {'HG=F', 'PL=F', 'PA=F', 'COPX', 'PPLT'}
                _HIGH_VOL = {'MSTR', 'AMZE', 'RCAT', 'SMCI', 'RGTI', 'QBTS', 'BKSY',
                             'SPCE', 'ABTC', 'BZAI', 'BNZI', 'AIRI',
                             'ESLT', 'QS', 'QUBT', 'PACB', 'APLM', 'NVTS',
                             'ACHR', 'GORO', 'USAS', 'APLT', 'ONDS', 'GPUS'}
                if _sym in _METALS_GOLD: _gaussian_asset_class = 'metals_gold'
                elif _sym in _METALS_SILVER: _gaussian_asset_class = 'metals_silver'
                elif _sym in _METALS_OTHER: _gaussian_asset_class = 'metals_other'
                elif _sym in _HIGH_VOL: _gaussian_asset_class = 'high_vol_equity'
                elif _sym.endswith('=X'): _gaussian_asset_class = 'forex'

            if _gaussian_asset_class and _gaussian_asset_class in _GAUSSIAN_PROFILES:
                _g_prof = _GAUSSIAN_PROFILES[_gaussian_asset_class]
                config.chisq_ewm_lambda = _g_prof.get('chisq_ewm_lambda', 0.98)
                config.pit_var_lambda = _g_prof.get('pit_var_lambda', 0.97)
                config.pit_var_dz_lo = _g_prof.get('pit_var_dz_lo', 0.30)
                config.pit_var_dz_hi = _g_prof.get('pit_var_dz_hi', 0.55)

        n = len(returns)
        n_train = int(n * train_frac)
        returns_train = returns[:n_train]
        vol_train = vol[:n_train]

        # ── STAGE 1: Base params (q, c, φ) ──
        s1 = cls._gaussian_stage_1(returns_train, vol_train, n_train, config, phi_mode)
        if not s1['success']:
            config.q = 1e-6
            config.c = 1.0
            config.phi = 0.0 if phi_mode else 1.0
            return config, {"stage": 0, "success": False, "error": "Stage 1 failed"}

        q_opt, c_opt, phi_opt = s1['q'], s1['c'], s1['phi']
        config.q = q_opt
        config.c = c_opt
        config.phi = phi_opt

        # -- STAGE 1.5: Momentum injection weight --
        _mom_sig_train = None
        _momentum_weight = 0.0
        try:
            if momentum_signal is not None and len(momentum_signal) >= n_train:
                _mom_sig_train = momentum_signal[:n_train]
            else:
                from models.momentum_augmented import compute_momentum_features, compute_momentum_signal
                _mom_feats = compute_momentum_features(returns_train)
                _mom_sig_train = compute_momentum_signal(_mom_feats)
            s15 = cls._gaussian_stage_1_5_momentum(
                returns_train, vol_train, n_train,
                q_opt, c_opt, phi_opt, _mom_sig_train)
            _momentum_weight = s15['momentum_weight']
            config.momentum_weight = _momentum_weight
        except Exception:
            config.momentum_weight = 0.0

        # Get predictive values for downstream stages (with momentum if active)
        _, _, mu_pred_train, S_pred_train, _ = cls._filter_phi_with_momentum(
            returns_train, vol_train, q_opt, c_opt, phi_opt,
            _mom_sig_train, _momentum_weight)
        innovations_train = returns_train - mu_pred_train

        # ── STAGE 2: Variance inflation β ──
        beta_opt = cls._gaussian_stage_2_variance_inflation(
            returns_train, mu_pred_train, S_pred_train, n_train)
        config.variance_inflation = beta_opt

        # Mean drift correction
        mu_drift = float(np.mean(innovations_train))
        config.mu_drift = mu_drift

        # ── STAGE 3: GJR-GARCH on innovations ──
        garch = cls._gaussian_stage_3_garch(innovations_train, mu_drift, n_train)
        config.garch_omega = garch['garch_omega']
        config.garch_alpha = garch['garch_alpha']
        config.garch_beta = garch['garch_beta']
        config.garch_leverage = garch['garch_leverage']
        config.garch_unconditional_var = garch['unconditional_var']

        # ── STAGE 4: EWM λ + CRPS σ shrinkage ──
        s4 = cls._gaussian_stage_4_ewm_and_shrinkage(
            returns_train, vol_train, n_train,
            q_opt, c_opt, phi_opt, beta_opt, mu_drift, garch)
        config.crps_ewm_lambda = s4['crps_ewm_lambda']
        config.crps_sigma_shrinkage = s4['crps_sigma_shrinkage']

        # ── STAGE 4.5: GAS-Q adaptive process noise ──
        try:
            s45 = cls._gaussian_stage_4_5_gas_q(
                returns_train, vol_train, n_train,
                q_opt, c_opt, phi_opt, beta_opt, mu_drift, garch,
                _mom_sig_train, _momentum_weight)
            config.gas_q_omega = s45['gas_q_omega']
            config.gas_q_alpha = s45['gas_q_alpha']
            config.gas_q_beta = s45['gas_q_beta']
        except Exception:
            config.gas_q_omega = 0.0
            config.gas_q_alpha = 0.0
            config.gas_q_beta = 0.0

        # ── STAGE 5: Walk-forward calibration (ν-free) ──
        s5 = cls._gaussian_stage_5_calibration(
            returns, vol, config, train_frac)
        config.calibrated_gw = s5['calibrated_gw']
        config.calibrated_lambda_rho = s5['calibrated_lambda_rho']
        config.calibrated_beta_probit_corr = s5['calibrated_beta_probit_corr']

        diagnostics = {
            "success": True,
            "stage": 5,
            "q": q_opt, "c": c_opt, "phi": phi_opt,
            "variance_inflation": beta_opt,
            "mu_drift": mu_drift,
            "momentum_weight": float(config.momentum_weight),
            "momentum_enabled": config.momentum_enabled,
            "garch_alpha": garch['garch_alpha'],
            "garch_beta": garch['garch_beta'],
            "garch_leverage": garch['garch_leverage'],
            "gas_q_omega": float(config.gas_q_omega),
            "gas_q_alpha": float(config.gas_q_alpha),
            "gas_q_beta": float(config.gas_q_beta),
            "gas_q_enabled": config.gas_q_enabled,
            "crps_ewm_lambda": s4['crps_ewm_lambda'],
            "crps_sigma_shrinkage": s4['crps_sigma_shrinkage'],
            "calibrated_gw": s5['calibrated_gw'],
            "calibrated_lambda_rho": s5['calibrated_lambda_rho'],
            "calibrated_beta_probit_corr": s5['calibrated_beta_probit_corr'],
            "phi_mode": phi_mode,
            "asset_symbol": asset_symbol,
        }

        return config, diagnostics

    @classmethod
    def filter_and_calibrate(cls, returns, vol, config, train_frac=0.7, momentum_signal=None):
        """
        Honest PIT + CRPS for unified Gaussian. All params from training config.

        Mirrors PhiStudentTDriftModel.filter_and_calibrate but uses Gaussian CDF
        (no ν parameter). Delegates to GARCH path when GARCH params are active.

        Returns (pit_values, pit_pvalue, sigma_crps, crps, diagnostics).
        """
        returns = np.asarray(returns).flatten()
        vol = np.asarray(vol).flatten()
        n = len(returns)
        n_train = int(n * train_frac)
        n_test = n - n_train

        q = float(config.q)
        c = float(config.c)
        phi = float(config.phi)
        mom_w = float(getattr(config, 'momentum_weight', 0.0))

        # Use momentum-augmented filter when momentum is active
        _, _, mu_pred, S_pred, ll = cls._filter_phi_with_momentum(
            returns, vol, q, c, phi, momentum_signal, mom_w)

        variance_inflation = float(getattr(config, 'variance_inflation', 1.0))
        use_garch = (getattr(config, 'garch_alpha', 0.0) > 0 or
                     getattr(config, 'garch_beta', 0.0) > 0)

        returns_test = returns[n_train:]
        mu_pred_test = mu_pred[n_train:]
        S_pred_test = S_pred[n_train:]

        if use_garch:
            innovations = returns - mu_pred
            h_garch_full = cls._compute_garch_variance_gaussian(innovations, config)
            h_garch = h_garch_full[n_train:]

            pit_values, sigma, mu_effective, S_calibrated = cls._gaussian_pit_garch_path(
                returns, mu_pred, S_pred, h_garch_full, config, n_train, n_test)
        else:
            S_calibrated = S_pred_test * variance_inflation
            sigma = np.sqrt(np.maximum(S_calibrated, 1e-20))
            sigma = np.maximum(sigma, 1e-10)
            z = (returns_test - mu_pred_test) / sigma
            pit_values = np.clip(norm.cdf(z), 0.001, 0.999)
            mu_effective = mu_pred_test

        pit_pvalue = float(kstest(pit_values, 'uniform').pvalue)
        # Anderson-Darling test (tail-sensitive complement to KS)
        try:
            from calibration.pit_calibration import anderson_darling_uniform
            _ad_stat, _ad_pvalue = anderson_darling_uniform(pit_values)
        except Exception:
            _ad_pvalue = float('nan')
        hist, _ = np.histogram(pit_values, bins=10, range=(0, 1))
        mad = float(np.mean(np.abs(hist / n_test - 0.1)))

        # Berkowitz on test split
        berkowitz_pvalue, berkowitz_lr, berkowitz_n_pit = cls._compute_berkowitz_full(pit_values)

        # CRPS
        _crps_shrink = float(np.clip(float(getattr(config, 'crps_sigma_shrinkage', 1.0)), 0.30, 1.0))
        sigma_crps = sigma * _crps_shrink
        try:
            from tuning.diagnostics import compute_crps_gaussian_inline
            crps = compute_crps_gaussian_inline(returns_test, mu_effective, sigma_crps)
        except Exception:
            crps = float('nan')

        diagnostics = {
            'pit_pvalue': pit_pvalue,
            'ad_pvalue': _ad_pvalue,
            'berkowitz_pvalue': float(berkowitz_pvalue),
            'berkowitz_lr': float(berkowitz_lr),
            'pit_count': int(berkowitz_n_pit),
            'mad': mad,
            'crps': crps,
            'log_likelihood': ll,
            'n_train': n_train,
            'n_test': n_test,
            'variance_inflation': variance_inflation,
            'crps_shrink': _crps_shrink,
            'mu_effective': mu_effective,
            'calibrated_gw': float(getattr(config, 'calibrated_gw', 0.0)),
        }
        return pit_values, pit_pvalue, sigma_crps, crps, diagnostics

    # =========================================================================
    # MOMENTUM-AUGMENTED FILTER
    # =========================================================================

    @classmethod
    def _filter_phi_with_momentum(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        momentum_signal: np.ndarray = None,
        momentum_weight: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        φ-Gaussian filter with optional momentum injection into state equation.

        When momentum is active:
            μ_pred[t] = φ·μ_{t-1} + w·mom[t]·σ_t

        Degrades gracefully to filter_phi_with_predictive when momentum_weight=0
        or momentum_signal is None.

        Returns (mu_filtered, P_filtered, mu_pred, S_pred, log_likelihood).
        """
        if momentum_signal is None or abs(momentum_weight) < 1e-10:
            return cls.filter_phi_with_predictive(returns, vol, q, c, phi)

        n = len(returns)
        q_val = float(q)
        c_val = float(c)
        phi_val = float(np.clip(phi, -0.999, 0.999))
        mom_w = float(momentum_weight)

        mu = 0.0
        P = 1e-4
        mu_filtered = np.zeros(n)
        P_filtered = np.zeros(n)
        mu_pred_arr = np.zeros(n)
        S_pred_arr = np.zeros(n)
        log_likelihood = 0.0

        phi_sq = phi_val * phi_val
        log_2pi = math.log(2.0 * math.pi)

        for t in range(n):
            # Momentum-augmented prediction
            mom_adj = mom_w * float(momentum_signal[t]) * float(vol[t])
            mu_pred = phi_val * mu + mom_adj
            P_pred = phi_sq * P + q_val

            vol_t = float(vol[t])
            R = c_val * (vol_t * vol_t)
            S = P_pred + R
            if S <= 1e-12:
                S = 1e-12

            mu_pred_arr[t] = mu_pred
            S_pred_arr[t] = S

            innovation = float(returns[t]) - mu_pred
            K = P_pred / S
            mu = mu_pred + K * innovation
            P = (1.0 - K) * P_pred
            if P < 1e-12:
                P = 1e-12

            mu_filtered[t] = mu
            P_filtered[t] = P

            log_likelihood += -0.5 * (log_2pi + math.log(S) + (innovation * innovation) / S)

        return mu_filtered, P_filtered, mu_pred_arr, S_pred_arr, float(log_likelihood)

    # =========================================================================
    # UNIFIED STAGE METHODS
    # =========================================================================

    @classmethod
    def _gaussian_stage_1_5_momentum(cls, returns_train, vol_train, n_train,
                                      q, c, phi, momentum_signal):
        """
        Stage 1.5: Momentum injection weight via CRPS grid search.

        Tests momentum_weight ∈ {0.0, 0.05, 0.10, 0.15}.
        Degradation guard: 0.0 always competes. Only activates momentum
        if it improves validation-fold CRPS vs no-momentum baseline.

        Returns dict with 'momentum_weight'.
        """
        from scipy.stats import norm as _norm

        # Validation fold: last 30% of training data
        n_val = max(50, n_train // 3)
        val_start = n_train - n_val

        WEIGHT_GRID = [0.0, 0.05, 0.10, 0.15]
        best_crps = float('inf')
        best_weight = 0.0

        for w in WEIGHT_GRID:
            try:
                _, _, mu_pred, S_pred, _ = cls._filter_phi_with_momentum(
                    returns_train, vol_train, q, c, phi, momentum_signal, w)

                # Compute CRPS on validation fold
                crps_sum = 0.0
                for t in range(val_start, n_train):
                    sig = math.sqrt(max(S_pred[t], 1e-20))
                    if sig < 1e-10:
                        sig = 1e-10
                    z = (returns_train[t] - mu_pred[t]) / sig
                    crps_t = sig * (z * (2 * _norm.cdf(z) - 1)
                                    + 2 * _norm.pdf(z)
                                    - 1.0 / math.sqrt(math.pi))
                    crps_sum += crps_t
                avg_crps = crps_sum / n_val

                if avg_crps < best_crps:
                    best_crps = avg_crps
                    best_weight = w
            except Exception:
                continue

        return {'momentum_weight': best_weight}

    @classmethod
    def _gaussian_stage_4_5_gas_q(cls, returns_train, vol_train, n_train,
                                   q, c, phi, beta, mu_drift, garch,
                                   momentum_signal, momentum_weight):
        """
        Stage 4.5: GAS-Q adaptive process noise estimation.

        Optimizes (ω, α, β_gas) via concentrated likelihood from gas_q.py.
        Degradation guard: only enables GAS-Q if validation-fold CRPS
        improves by > 1% relative to static q baseline.

        Returns dict with gas_q_omega, gas_q_alpha, gas_q_beta, gas_q_enabled.
        """
        DISABLED = {
            'gas_q_omega': 0.0, 'gas_q_alpha': 0.0, 'gas_q_beta': 0.0,
            'gas_q_enabled': False
        }

        if n_train < 200:
            return DISABLED

        try:
            from models.gas_q import optimize_gas_q_params, gas_q_filter_gaussian
        except ImportError:
            try:
                from .gas_q import optimize_gas_q_params, gas_q_filter_gaussian
            except ImportError:
                return DISABLED

        try:
            gas_config, gas_diag = optimize_gas_q_params(
                returns_train, vol_train, c, phi, nu=None, train_frac=0.7)

            if not gas_diag.get('fit_success', False):
                return DISABLED

            # ── Degradation guard: compare CRPS on validation fold ──
            from scipy.stats import norm as _norm
            n_val = max(50, n_train // 3)
            val_start = n_train - n_val

            # Baseline CRPS (static q)
            _, _, mu_base, S_base, _ = cls._filter_phi_with_momentum(
                returns_train, vol_train, q, c, phi,
                momentum_signal, momentum_weight)

            def _fold_crps(mu_p, S_p):
                total = 0.0
                for t in range(val_start, n_train):
                    sig = math.sqrt(max(S_p[t], 1e-20))
                    if sig < 1e-10:
                        sig = 1e-10
                    z = (returns_train[t] - mu_p[t]) / sig
                    total += sig * (z * (2 * _norm.cdf(z) - 1)
                                    + 2 * _norm.pdf(z)
                                    - 1.0 / math.sqrt(math.pi))
                return total / n_val

            crps_baseline = _fold_crps(mu_base, S_base)

            # GAS-Q CRPS
            gas_result = gas_q_filter_gaussian(
                returns_train, vol_train, c, phi, gas_config)

            if gas_result is None or not hasattr(gas_result, 'mu_filtered'):
                return DISABLED

            # Build S_pred from GAS-Q filter (P + R)
            P_gas = gas_result.P_filtered
            R = c * (vol_train ** 2)
            S_gas = P_gas + R

            crps_gas = _fold_crps(gas_result.mu_filtered, S_gas)

            # Gate: must improve CRPS by >1%
            if crps_baseline > 0 and crps_gas < crps_baseline * 0.99:
                return {
                    'gas_q_omega': float(gas_config.omega),
                    'gas_q_alpha': float(gas_config.alpha),
                    'gas_q_beta': float(gas_config.beta),
                    'gas_q_enabled': True,
                }

            return DISABLED

        except Exception:
            return DISABLED

    @classmethod
    def _gaussian_stage_1(cls, returns_train, vol_train, n_train, config, phi_mode):
        """Stage 1: Base params (q, c, φ) via CV MLE with state regularization."""
        _returns_r = np.ascontiguousarray(returns_train.flatten(), dtype=np.float64)
        _vol_sq = np.ascontiguousarray((vol_train ** 2).flatten(), dtype=np.float64)

        ret_std = float(np.std(_returns_r))
        vol_var_median = float(np.median(_vol_sq))

        q_min = max(config.q_min, 1e-8)
        _log_q_min = np.log10(q_min)
        _log_q_max = np.log10(1e-1)
        _log_c_min = np.log10(max(config.c_min, 0.01))
        _log_c_max = np.log10(min(config.c_max, 10.0))
        phi_min = -0.999 if phi_mode else 0.999
        phi_max = 0.999

        # Adaptive prior
        vol_mean = float(np.mean(np.sqrt(_vol_sq)))
        vol_cv = float(np.std(np.sqrt(_vol_sq))) / (vol_mean + 1e-12)
        prior_mean = -5.0 + (0.5 if vol_cv > 0.5 else (-0.3 if vol_cv < 0.2 else 0.0))

        # Rolling folds
        min_train = min(max(60, int(n_train * 0.4)), max(n_train - 5, 1))
        test_window = min(max(20, int(n_train * 0.1)), max(n_train - min_train, 5))
        folds = []
        train_end = min_train
        while train_end + test_window <= n_train:
            test_end = min(train_end + test_window, n_train)
            if test_end - train_end >= 20:
                folds.append((0, train_end, train_end, test_end))
            train_end += test_window
        if not folds:
            split = int(n_train * 0.7)
            folds = [(0, split, split, n_train)]

        _mlog = math.log
        _msqrt = math.sqrt

        # Try Numba-accelerated CV test fold kernel
        try:
            from models.numba_wrappers import run_phi_gaussian_cv_test_fold as _numba_cv_fold
            _use_numba_cv = True
        except (ImportError, Exception):
            _use_numba_cv = False

        # Pre-allocate standardized residuals buffer for Numba kernel
        _std_buf = np.zeros(n_train, dtype=np.float64) if _use_numba_cv else None

        def neg_cv_ll(params):
            if phi_mode:
                log_q, log_c, phi = params
            else:
                log_q, log_c = params
                phi = 1.0
            q = 10 ** log_q
            c = 10 ** log_c
            phi = float(np.clip(phi, -0.999, 0.999))
            phi_sq = phi * phi

            if q <= 0 or c <= 0 or not math.isfinite(q) or not math.isfinite(c):
                return 1e12

            total_ll = 0.0
            total_obs = 0

            for ts, te, vs, ve in folds:
                try:
                    mu_f, P_f, _ = cls.filter_phi(_returns_r[ts:te], vol_train[ts:te], q, c, phi)
                    mu_p = float(mu_f[-1])
                    P_p = float(P_f[-1])

                    if _use_numba_cv:
                        ll_fold, n_obs_fold, _ = _numba_cv_fold(
                            _returns_r, _vol_sq, q, c, phi,
                            mu_p, P_p, vs, ve,
                            _std_buf, 0, 0,
                        )
                        total_ll += ll_fold
                        total_obs += n_obs_fold
                    else:
                        ll_fold = 0.0
                        for t in range(vs, ve):
                            mu_p = phi * mu_p
                            P_p = phi_sq * P_p + q
                            R = c * _vol_sq[t]
                            inn = _returns_r[t] - mu_p
                            fv = P_p + R
                            if fv > 1e-12:
                                ll_fold += -0.5 * (_LOG_2PI + _mlog(fv) + inn * inn / fv)
                            S = P_p + R
                            K = P_p / S if S > 1e-12 else 0.0
                            mu_p = mu_p + K * inn
                            P_p = (1.0 - K) * P_p
                        total_ll += ll_fold
                        total_obs += (ve - vs)
                except Exception:
                    continue

            if total_obs == 0:
                return 1e12

            avg_ll = total_ll / total_obs
            ps = 1.0 / max(total_obs, 100)
            prior_q = -1.0 * ps * (log_q - prior_mean) ** 2
            prior_c = -0.1 * ps * (log_c - _LOG10_C_TARGET) ** 2

            # State regularization: prevent φ→1/q→0 collapse
            phi_near = max(0.0, abs(phi) - 0.95) ** 2
            q_small = max(0.0, -7.0 - log_q) ** 2
            state_reg = -500.0 * (phi_near + q_small)

            val = avg_ll + prior_q + prior_c + state_reg
            return -val if math.isfinite(val) else 1e12

        # Grid search
        lq_grid = np.linspace(max(_log_q_min, prior_mean - 1.5),
                              min(_log_q_max, prior_mean + 1.5), 4)
        lc_grid = np.array([_log_c_min, 0.0, _log_c_max * 0.6])
        phi_grid = np.array([-0.3, 0.0, 0.3]) if phi_mode else np.array([1.0])

        best_val = 1e20
        best_lq, best_lc, best_ph = prior_mean, 0.0, 0.0

        for lq in lq_grid:
            for lc in lc_grid:
                for ph in phi_grid:
                    try:
                        p = np.array([lq, lc, ph]) if phi_mode else np.array([lq, lc])
                        v = neg_cv_ll(p)
                        if v < best_val:
                            best_val = v
                            best_lq, best_lc, best_ph = lq, lc, ph
                    except Exception:
                        continue

        # L-BFGS-B refinement
        if phi_mode:
            bounds = [(_log_q_min, _log_q_max), (_log_c_min, _log_c_max), (phi_min, phi_max)]
            starts = [
                np.array([best_lq, best_lc, best_ph]),
                np.array([prior_mean, 0.0, 0.0]),
            ]
        else:
            bounds = [(_log_q_min, _log_q_max), (_log_c_min, _log_c_max)]
            starts = [
                np.array([best_lq, best_lc]),
                np.array([prior_mean, 0.0]),
            ]

        best_result = None
        best_fun = best_val
        for x0 in starts:
            try:
                res = minimize(neg_cv_ll, x0, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 100, 'ftol': 1e-8})
                if res.fun < best_fun:
                    best_fun = res.fun
                    best_result = res
            except Exception:
                continue

        if best_result is not None:
            if phi_mode:
                lq, lc, ph = best_result.x
            else:
                lq, lc = best_result.x
                ph = 1.0
        else:
            lq, lc, ph = best_lq, best_lc, best_ph

        return {
            'success': True,
            'q': 10 ** lq,
            'c': 10 ** lc,
            'phi': float(np.clip(ph, -0.999, 0.999)),
            'log_q': lq,
        }

    @staticmethod
    def _gaussian_stage_2_variance_inflation(returns_train, mu_pred_train, S_pred_train, n_train):
        """Stage 2: Variance inflation β via PIT MAD grid search on training data."""
        n_val = max(50, n_train // 3)
        val_start = n_train - n_val

        best_beta, best_mad = 1.0, 1.0
        for beta in [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.50]:
            S_cal = S_pred_train[val_start:] * beta
            sigma = np.sqrt(np.maximum(S_cal, 1e-20))
            sigma = np.maximum(sigma, 1e-10)
            z = (returns_train[val_start:] - mu_pred_train[val_start:]) / sigma
            pit = np.clip(norm.cdf(z), 0.001, 0.999)
            hist, _ = np.histogram(pit, bins=10, range=(0, 1))
            mad = float(np.mean(np.abs(hist / len(pit) - 0.1)))
            if mad < best_mad:
                best_mad = mad
                best_beta = beta

        return best_beta

    @staticmethod
    def _gaussian_stage_3_garch(innovations_train, mu_drift, n_train):
        """
        Stage 3: GJR-GARCH(1,1) on Kalman innovations.

        h_t = ω + α·ε²_{t-1} + γ_lev·ε²_{t-1}·I(ε<0) + β·h_{t-1}

        Reuses the same estimation logic as PhiStudentTDriftModel._stage_5c.
        """
        inn = innovations_train - mu_drift
        sq = inn ** 2
        uvar = float(np.var(inn))
        garch_leverage = 0.0

        if n_train > 100:
            sq_c = sq - uvar
            denom = np.sum(sq_c[:-1] ** 2)
            if denom > 1e-12:
                ga = float(np.sum(sq_c[1:] * sq_c[:-1]) / denom)
                ga = np.clip(ga, 0.02, 0.25)
            else:
                ga = 0.08

            neg_ind = (inn[:-1] < 0).astype(np.float64)
            n_neg = max(int(np.sum(neg_ind)), 1)
            n_pos = max(int(np.sum(1.0 - neg_ind)), 1)
            mean_sq_neg = float(np.sum(sq[1:] * neg_ind) / n_neg)
            mean_sq_pos = float(np.sum(sq[1:] * (1.0 - neg_ind)) / n_pos)

            if mean_sq_pos > 1e-12:
                lev_ratio = mean_sq_neg / mean_sq_pos
            else:
                lev_ratio = 1.0

            if lev_ratio > 1.0:
                garch_leverage = float(np.clip(ga * (lev_ratio - 1.0), 0.0, 0.20))

            gb = 0.97 - ga - garch_leverage / 2.0
            gb = float(np.clip(gb, 0.70, 0.95))

            total_p = ga + garch_leverage / 2.0 + gb
            if total_p >= 0.99:
                gb = 0.98 - ga - garch_leverage / 2.0
                gb = max(gb, 0.5)

            go = uvar * (1 - ga - garch_leverage / 2.0 - gb)
            go = max(go, 1e-10)
        else:
            go = uvar * 0.05
            ga = 0.08
            gb = 0.87

        return {
            'garch_omega': float(go),
            'garch_alpha': float(ga),
            'garch_beta': float(gb),
            'garch_leverage': float(garch_leverage),
            'unconditional_var': float(uvar),
        }

    @classmethod
    def _gaussian_stage_4_ewm_and_shrinkage(cls, returns_train, vol_train, n_train,
                                             q, c, phi, beta, mu_drift, garch):
        """Stage 4: EWM location correction λ + CRPS σ shrinkage."""
        crps_ewm_lambda = 0.0
        crps_sigma_shrinkage = 1.0

        if n_train < 150:
            return {'crps_ewm_lambda': crps_ewm_lambda,
                    'crps_sigma_shrinkage': crps_sigma_shrinkage}

        _, _, mu_pred, S_pred, _ = cls.filter_phi_with_predictive(
            returns_train, vol_train, q, c, phi)
        innovations = returns_train - mu_pred

        # Compute GARCH variance for blending
        h = np.zeros(n_train)
        h[0] = garch['unconditional_var']
        sq = innovations ** 2
        neg = (innovations < 0).astype(np.float64)
        go, ga, gb, gl = garch['garch_omega'], garch['garch_alpha'], garch['garch_beta'], garch['garch_leverage']

        for t in range(1, n_train):
            h[t] = max(go + ga * sq[t-1] + gl * sq[t-1] * neg[t-1] + gb * h[t-1], 1e-12)

        # Test EWM λ candidates on last 30% of training
        n_test_inner = max(50, n_train // 3)
        test_start = n_train - n_test_inner

        best_crps = float('inf')
        from scipy.stats import norm as _norm

        _inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
        _S_test = S_pred[test_start:n_train] * beta
        _sig_base = np.sqrt(np.maximum(_S_test, 1e-20))
        _sig_base = np.maximum(_sig_base, 1e-10)
        _ret_test = returns_train[test_start:n_train]

        for lam in [0.0, 0.92, 0.94, 0.96, 0.98]:
            # Build mu_eff array with EWM correction
            mu_eff_arr = np.empty(n_test_inner)
            ewm_mu = float(np.mean(innovations[:test_start]))
            alpha_e = 1.0 - lam if lam > 0 else 0.0
            for t in range(test_start, n_train):
                mu_eff_arr[t - test_start] = mu_pred[t] + ewm_mu + mu_drift
                if lam > 0:
                    ewm_mu = lam * ewm_mu + alpha_e * innovations[t]

            # Vectorized CRPS computation
            z_arr = (_ret_test - mu_eff_arr) / _sig_base
            cdf_arr = _norm.cdf(z_arr)
            pdf_arr = _norm.pdf(z_arr)
            crps_arr = _sig_base * (z_arr * (2 * cdf_arr - 1) + 2 * pdf_arr - _inv_sqrt_pi)
            avg_crps = float(np.mean(crps_arr))
            if avg_crps < best_crps:
                best_crps = avg_crps
                crps_ewm_lambda = lam

        # CRPS σ shrinkage: golden section on [0.7, 1.0]
        best_shrink = 1.0
        best_crps_s = best_crps
        for s in [0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
            # Build mu_eff array with best EWM lambda
            mu_eff_arr = np.empty(n_test_inner)
            ewm_mu = float(np.mean(innovations[:test_start]))
            alpha_e = 1.0 - crps_ewm_lambda if crps_ewm_lambda > 0 else 0.0
            for t in range(test_start, n_train):
                mu_eff_arr[t - test_start] = mu_pred[t] + ewm_mu + mu_drift
                if crps_ewm_lambda > 0:
                    ewm_mu = crps_ewm_lambda * ewm_mu + alpha_e * innovations[t]

            # Vectorized CRPS with shrinkage
            sig_s = _sig_base * s
            z_arr = (_ret_test - mu_eff_arr) / sig_s
            cdf_arr = _norm.cdf(z_arr)
            pdf_arr = _norm.pdf(z_arr)
            crps_arr = sig_s * (z_arr * (2 * cdf_arr - 1) + 2 * pdf_arr - _inv_sqrt_pi)
            avg_crps = float(np.mean(crps_arr))
            if avg_crps < best_crps_s:
                best_crps_s = avg_crps
                best_shrink = s

        return {'crps_ewm_lambda': crps_ewm_lambda,
                'crps_sigma_shrinkage': best_shrink}

    @classmethod
    def _gaussian_stage_5_calibration(cls, returns, vol, config, train_frac=0.7):
        """
        Stage 5: Walk-forward calibration (ν-free Gaussian variant).

        Searches over (gw, λ_ρ) grid — no ν dimension since Gaussian.
        Pre-computes calibration params read by filter_and_calibrate.
        """
        D = {'calibrated_gw': 0.0, 'calibrated_lambda_rho': 0.985,
             'calibrated_beta_probit_corr': 1.0}

        ret = np.asarray(returns).flatten()
        vl = np.asarray(vol).flatten()
        n = len(ret)
        nt = int(n * train_frac)

        use_garch = (getattr(config, 'garch_alpha', 0.0) > 0 or
                     getattr(config, 'garch_beta', 0.0) > 0)
        if not use_garch or nt < 150:
            return D

        q, c, phi = float(config.q), float(config.c), float(config.phi)
        _, _, mp, sp, _ = cls.filter_phi_with_predictive(ret, vl, q, c, phi)
        inn = ret - mp

        h = cls._compute_garch_variance_gaussian(inn, config)
        ht = h[:nt]
        it = inn[:nt]
        st = sp[:nt]

        _msqrt = math.sqrt
        _norm_cdf = norm.cdf

        # Try Numba-accelerated score fold kernel
        try:
            from models.numba_wrappers import run_gaussian_score_fold as _numba_score_fold
            _use_numba_score = True
        except (ImportError, Exception):
            _use_numba_score = False

        def _score_fold(Sb, ee, ve, lam):
            """Run EWM PIT for one fold, return (ks_p_approx, mad)."""
            nv = ve - ee
            if nv < 20:
                return 0.0, 1.0

            if _use_numba_score:
                init_em = float(np.mean(it[:ee]))
                init_en = float(np.mean(it[:ee] ** 2))
                init_ed = float(np.mean(Sb[:ee]))
                return _numba_score_fold(
                    it, Sb, ee, ve, lam,
                    init_em, init_en, init_ed,
                )

            em = float(np.mean(it[:ee]))
            en = float(np.mean(it[:ee] ** 2))
            ed = float(np.mean(Sb[:ee]))
            lm1 = 1.0 - lam

            zv = np.empty(nv)
            for tv in range(nv):
                ix = ee + tv
                bv = en / (ed + 1e-12)
                if bv < 0.2: bv = 0.2
                elif bv > 5.0: bv = 5.0
                iv = it[ix] - em
                Sv = Sb[ix] * bv
                s = _msqrt(Sv) if Sv > 0 else 1e-10
                if s < 1e-10: s = 1e-10
                zv[tv] = iv / s
                em = lam * em + lm1 * it[ix]
                en = lam * en + lm1 * (it[ix] ** 2)
                ed = lam * ed + lm1 * Sb[ix]

            pv = np.clip(_norm_cdf(zv), 0.001, 0.999)
            # KS approximation
            ps = np.sort(pv)
            dp = np.max(np.arange(1, nv+1) / nv - ps)
            dm = np.max(ps - np.arange(0, nv) / nv)
            D_ks = max(dp, dm)
            sq_n = math.sqrt(nv)
            lam_ks = (sq_n + 0.12 + 0.11 / sq_n) * D_ks
            if lam_ks < 0.001: kp = 1.0
            elif lam_ks > 3.0: kp = 0.0
            else: kp = min(2.0 * math.exp(-2.0 * lam_ks * lam_ks), 1.0)

            hi, _ = np.histogram(pv, bins=10, range=(0, 1))
            md = float(np.mean(np.abs(hi / nv - 0.1)))
            return kp, md

        # Build folds
        s5f = float(getattr(config, 'crps_ewm_lambda', 0.0))
        lam = float(np.clip(max(s5f, 0.975), 0.975, 0.995)) if s5f >= 0.50 else 0.985

        nf = 1 if nt < 200 else (2 if nt < 400 else 3)
        fs = nt // (nf + 1)

        GW_GRID = [0.0, 0.15, 0.30, 0.50, 0.65, 0.80, 0.95, 1.0]
        LAM_GRID = [0.975, 0.980, 0.985, 0.990]

        bs = -1.0
        bg, bl = 0.0, 0.985

        for gw in GW_GRID:
            Sb = (1 - gw) * st + gw * ht
            for lam_c in LAM_GRID:
                total_score = 0.0
                total_folds = 0
                for fi in range(nf):
                    ee = (fi + 1) * fs
                    ve = min((fi + 2) * fs, nt)
                    if ve <= ee:
                        continue
                    kp, md = _score_fold(Sb, ee, ve, lam_c)
                    # Combined score: KS p-value × (1 - mad_penalty)
                    mp_ = max(0, 1 - md / 0.05)
                    score = kp * mp_
                    total_score += score
                    total_folds += 1

                if total_folds > 0:
                    avg_score = total_score / total_folds
                    if avg_score > bs:
                        bs = avg_score
                        bg = gw
                        bl = lam_c

        return {
            'calibrated_gw': bg,
            'calibrated_lambda_rho': bl,
            'calibrated_beta_probit_corr': 1.0,  # No ν-based correction for Gaussian
        }

    # =========================================================================
    # CALIBRATION HELPERS
    # =========================================================================

    @staticmethod
    def _compute_garch_variance_gaussian(innovations, config):
        """
        GJR-GARCH(1,1) variance for Gaussian model.

        Runs continuously from t=0 to avoid cold-start at train/test boundary.
        """
        n = len(innovations)
        sq = innovations ** 2
        neg = (innovations < 0).astype(np.float64)

        go = float(getattr(config, 'garch_omega', 0.0))
        ga = float(getattr(config, 'garch_alpha', 0.0))
        gb = float(getattr(config, 'garch_beta', 0.0))
        gl = float(getattr(config, 'garch_leverage', 0.0))
        gu = float(getattr(config, 'garch_unconditional_var', 1e-4))

        h = np.zeros(n)
        h[0] = gu
        for t in range(1, n):
            ht = go + ga * sq[t-1] + gl * sq[t-1] * neg[t-1] + gb * h[t-1]
            h[t] = max(ht, 1e-12)

        return h

    @classmethod
    def _gaussian_pit_garch_path(cls, returns, mu_pred, S_pred, h_garch_full,
                                  config, n_train, n_test):
        """
        PIT via GARCH-blended adaptive EWM for Gaussian model.

        Reads Stage 5 pre-calibrated params from config.
        Uses Gaussian CDF (no ν parameter). Includes AR(1) probit whitening.

        Returns (pit_values, sigma, mu_effective, S_calibrated).
        """
        returns_test = returns[n_train:]
        mu_pred_test = mu_pred[n_train:]
        S_pred_test = S_pred[n_train:]
        h_garch = h_garch_full[n_train:]

        _best_gw = float(config.calibrated_gw)
        _best_lam = float(config.calibrated_lambda_rho)
        _beta_corr = float(config.calibrated_beta_probit_corr)
        _1m_lam = 1.0 - _best_lam
        _msqrt = math.sqrt

        # EWM warm-start from training data
        inn_train = returns[:n_train] - mu_pred[:n_train]
        S_bt = (1 - _best_gw) * S_pred[:n_train] + _best_gw * h_garch_full[:n_train]

        _cal_start = int(n_train * 0.6)
        _ewm_mu = float(np.mean(inn_train))
        _ewm_num = float(np.mean(inn_train ** 2))
        _ewm_den = float(np.mean(S_bt))
        for _t in range(_cal_start):
            _ewm_mu = _best_lam * _ewm_mu + _1m_lam * inn_train[_t]
            _ewm_num = _best_lam * _ewm_num + _1m_lam * (inn_train[_t] ** 2)
            _ewm_den = _best_lam * _ewm_den + _1m_lam * S_bt[_t]

        # Training probit PITs for whitening warm-start
        _n_cal = n_train - _cal_start
        _zcal = np.empty(_n_cal)
        _em, _en, _ed = _ewm_mu, _ewm_num, _ewm_den
        for _t in range(_n_cal):
            _idx = _cal_start + _t
            _bv = _en / (_ed + 1e-12) * _beta_corr
            if _bv < 0.2: _bv = 0.2
            elif _bv > 5.0: _bv = 5.0
            _inn = inn_train[_idx] - _em
            _S_cv = S_bt[_idx] * _bv
            _sig = _msqrt(_S_cv) if _S_cv > 0 else 1e-10
            if _sig < 1e-10: _sig = 1e-10
            _zcal[_t] = _inn / _sig
            _em = _best_lam * _em + _1m_lam * inn_train[_idx]
            _en = _best_lam * _en + _1m_lam * (inn_train[_idx] ** 2)
            _ed = _best_lam * _ed + _1m_lam * S_bt[_idx]

        _pit_train_cal = np.clip(norm.cdf(_zcal), 0.001, 0.999)

        # ── DOMAIN-MATCHED TRAINING PIT CORRECTIONS (March 2026) ────
        # Apply the SAME chi² and PIT-var corrections to training PITs
        # that will be applied to test PITs. Prevents isotonic domain shift.
        # ────────────────────────────────────────────────────────────
        _CHI2_LAMBDA_TR = float(getattr(config, 'chisq_ewm_lambda', 0.98))
        _chi2_target_tr = 1.0  # Gaussian: E[z²] = 1
        _CHI2_WINSOR_CAP_TR = _chi2_target_tr * 50.0
        _CHI2_1M_TR = 1.0 - _CHI2_LAMBDA_TR
        _ewm_z2_tr = _chi2_target_tr

        # Chi² correct training z-values
        _zcal_corrected = np.empty_like(_zcal)
        for _t in range(len(_zcal)):
            _ratio_tr = _ewm_z2_tr / _chi2_target_tr
            if _ratio_tr < 0.3: _ratio_tr = 0.3
            elif _ratio_tr > 3.0: _ratio_tr = 3.0
            _dev_tr = abs(_ratio_tr - 1.0)
            if _ratio_tr >= 1.0:
                _dz_lo_tr = 0.25; _dz_rng_tr = 0.25
            else:
                _dz_lo_tr = 0.10; _dz_rng_tr = 0.15
            if _dev_tr < _dz_lo_tr:
                _adj_tr = 1.0
            elif _dev_tr >= _dz_lo_tr + _dz_rng_tr:
                _adj_tr = _msqrt(_ratio_tr)
            else:
                _str_tr = (_dev_tr - _dz_lo_tr) / _dz_rng_tr
                _adj_tr = 1.0 + _str_tr * (_msqrt(_ratio_tr) - 1.0)
            _zcal_corrected[_t] = _zcal[_t] / _adj_tr
            _raw_z2_tr = _zcal[_t] * _zcal[_t]
            _raw_z2_w_tr = _raw_z2_tr if _raw_z2_tr < _CHI2_WINSOR_CAP_TR else _CHI2_WINSOR_CAP_TR
            _ewm_z2_tr = _CHI2_LAMBDA_TR * _ewm_z2_tr + _CHI2_1M_TR * _raw_z2_w_tr

        _pit_train_cal = np.clip(norm.cdf(_zcal_corrected), 0.001, 0.999)

        # PIT-var correction on training PITs
        _PIT_VAR_LAMBDA_TR = float(getattr(config, 'pit_var_lambda', 0.97))
        _PIT_VAR_1M_TR = 1.0 - _PIT_VAR_LAMBDA_TR
        _PIT_VAR_DZ_LO_TR = float(getattr(config, 'pit_var_dz_lo', 0.30))
        _PIT_VAR_DZ_HI_TR = float(getattr(config, 'pit_var_dz_hi', 0.55))
        _PIT_VAR_DZ_RANGE_TR = _PIT_VAR_DZ_HI_TR - _PIT_VAR_DZ_LO_TR
        _ewm_pit_m_tr = 0.5
        _ewm_pit_sq_tr = 1.0 / 3.0
        for _t in range(len(_pit_train_cal)):
            _obs_var_tr = _ewm_pit_sq_tr - _ewm_pit_m_tr * _ewm_pit_m_tr
            if _obs_var_tr < 0.005: _obs_var_tr = 0.005
            _var_ratio_tr = _obs_var_tr / (1.0 / 12.0)
            _var_dev_tr = abs(_var_ratio_tr - 1.0)
            _raw_pit_tr = _pit_train_cal[_t]
            if _var_dev_tr > _PIT_VAR_DZ_LO_TR:
                _raw_stretch_tr = _msqrt((1.0 / 12.0) / _obs_var_tr)
                if _raw_stretch_tr < 0.70: _raw_stretch_tr = 0.70
                elif _raw_stretch_tr > 1.50: _raw_stretch_tr = 1.50
                if _var_dev_tr >= _PIT_VAR_DZ_HI_TR:
                    _stretch_tr = _raw_stretch_tr
                else:
                    _s_tr = (_var_dev_tr - _PIT_VAR_DZ_LO_TR) / _PIT_VAR_DZ_RANGE_TR
                    _stretch_tr = 1.0 + _s_tr * (_raw_stretch_tr - 1.0)
                _corrected_tr = 0.5 + (_raw_pit_tr - 0.5) * _stretch_tr
                if _corrected_tr < 0.001: _corrected_tr = 0.001
                elif _corrected_tr > 0.999: _corrected_tr = 0.999
                _pit_train_cal[_t] = _corrected_tr
            _ewm_pit_m_tr = _PIT_VAR_LAMBDA_TR * _ewm_pit_m_tr + _PIT_VAR_1M_TR * _raw_pit_tr
            _ewm_pit_sq_tr = _PIT_VAR_LAMBDA_TR * _ewm_pit_sq_tr + _PIT_VAR_1M_TR * _raw_pit_tr * _raw_pit_tr
        _pit_train_cal = np.clip(_pit_train_cal, 0.001, 0.999)
        # ── END DOMAIN-MATCHED TRAINING PIT CORRECTIONS ─────────────

        _z_probit_cal = norm.ppf(_pit_train_cal)
        _z_probit_cal = _z_probit_cal[np.isfinite(_z_probit_cal)]

        # Init test EWM
        _ewm_mu_t = float(np.mean(inn_train))
        _ewm_num_t = float(np.mean(inn_train ** 2))
        _ewm_den_t = float(np.mean(S_bt))

        _S_blend = (1 - _best_gw) * S_pred_test + _best_gw * h_garch
        inn_test = returns_test - mu_pred_test
        sq_inn = inn_test ** 2

        _z_test = np.empty(n_test)
        sigma = np.empty(n_test)
        mu_effective = np.empty(n_test)

        for _t in range(n_test):
            _bv = _ewm_num_t / (_ewm_den_t + 1e-12) * _beta_corr
            if _bv < 0.2: _bv = 0.2
            elif _bv > 5.0: _bv = 5.0
            _S_cal = _S_blend[_t] * _bv
            _inn = inn_test[_t] - _ewm_mu_t
            mu_effective[_t] = mu_pred_test[_t] + _ewm_mu_t
            _sig = _msqrt(_S_cal) if _S_cal > 0 else 1e-10
            if _sig < 1e-10: _sig = 1e-10
            sigma[_t] = _sig
            _z_test[_t] = _inn / _sig

            _ewm_mu_t = _best_lam * _ewm_mu_t + _1m_lam * inn_test[_t]
            _ewm_num_t = _best_lam * _ewm_num_t + _1m_lam * sq_inn[_t]
            _ewm_den_t = _best_lam * _ewm_den_t + _1m_lam * _S_blend[_t]

        # Gaussian CDF — ν-free
        # (Defer CDF until after chi-squared correction below)

        # ── CHI-SQUARED VARIANCE CORRECTION (Causal — Gaussian) ─────
        #
        # For Gaussian, E[z²] = 1.0. If the EWM β correction leaves
        # residual variance miscalibration, z² will deviate from 1.0.
        # Same algorithm as Student-t path, with target = 1.0.
        # Asset-adaptive λ from config (March 2026).
        # ────────────────────────────────────────────────────────────
        _CHI2_LAMBDA = float(getattr(config, 'chisq_ewm_lambda', 0.98))
        _CHI2_MIN_RATIO = 0.3
        _CHI2_MAX_RATIO = 3.0
        _CHI2_DZ_LO_WIDE = 0.25
        _CHI2_DZ_HI_WIDE = 0.50
        _CHI2_DZ_RANGE_WIDE = _CHI2_DZ_HI_WIDE - _CHI2_DZ_LO_WIDE
        _CHI2_DZ_LO_NARROW = 0.10
        _CHI2_DZ_HI_NARROW = 0.25
        _CHI2_DZ_RANGE_NARROW = _CHI2_DZ_HI_NARROW - _CHI2_DZ_LO_NARROW
        _chi2_target = 1.0  # Gaussian: E[z²] = 1
        _CHI2_WINSOR_MULT = 50.0
        _chi2_winsor_cap = _chi2_target * _CHI2_WINSOR_MULT
        _CHI2_1M = 1.0 - _CHI2_LAMBDA

        # Warm-start from training z² values
        _ewm_z2 = _chi2_target
        for _t in range(len(_zcal)):
            _z2_raw = _zcal[_t] * _zcal[_t]
            _z2_w = _z2_raw if _z2_raw < _chi2_winsor_cap else _chi2_winsor_cap
            _ewm_z2 = _CHI2_LAMBDA * _ewm_z2 + _CHI2_1M * _z2_w

        # Apply causal correction to test z values
        for _t in range(n_test):
            _ratio = _ewm_z2 / _chi2_target
            if _ratio < _CHI2_MIN_RATIO:
                _ratio = _CHI2_MIN_RATIO
            elif _ratio > _CHI2_MAX_RATIO:
                _ratio = _CHI2_MAX_RATIO

            _deviation = abs(_ratio - 1.0)
            if _ratio >= 1.0:
                _dz_lo = _CHI2_DZ_LO_WIDE
                _dz_range = _CHI2_DZ_RANGE_WIDE
            else:
                _dz_lo = _CHI2_DZ_LO_NARROW
                _dz_range = _CHI2_DZ_RANGE_NARROW

            if _deviation < _dz_lo:
                _adj = 1.0
            elif _deviation >= _dz_lo + _dz_range:
                _adj = _msqrt(_ratio)
            else:
                _strength = (_deviation - _dz_lo) / _dz_range
                _adj_raw = _msqrt(_ratio)
                _adj = 1.0 + _strength * (_adj_raw - 1.0)

            _z_test[_t] /= _adj
            sigma[_t] *= _adj
            _raw_z = _z_test[_t] * _adj
            _raw_z2 = _raw_z * _raw_z
            _raw_z2_w = _raw_z2 if _raw_z2 < _chi2_winsor_cap else _chi2_winsor_cap
            _ewm_z2 = _CHI2_LAMBDA * _ewm_z2 + _CHI2_1M * _raw_z2_w

        # Now compute Gaussian PIT after chi-squared correction
        pit_values = np.clip(norm.cdf(_z_test), 0.001, 0.999)

        # ── RANDOMIZED PIT FOR STALE OBSERVATIONS (Czado et al. 2009)
        _STALE_RETURN_THRESHOLD = 1e-10
        _STALE_EWM_LAMBDA = 0.97
        _STALE_ACTIVATION = 0.05
        _GOLDEN_RATIO = 1.6180339887498949

        _returns_train = returns[:n_train]
        _p_stale_train = float(np.mean(np.abs(_returns_train) < _STALE_RETURN_THRESHOLD))

        if _p_stale_train > _STALE_ACTIVATION:
            _p_stale = _p_stale_train
            for _t in range(n_test):
                _is_stale_t = abs(returns_test[_t]) < _STALE_RETURN_THRESHOLD
                if _is_stale_t and _p_stale > _STALE_ACTIVATION:
                    _F_lo = (1.0 - _p_stale) / 2.0
                    _F_hi = 0.5 + _p_stale / 2.0
                    _v_t = (_t * _GOLDEN_RATIO) % 1.0
                    _pit_rand = _F_lo + _v_t * (_F_hi - _F_lo)
                    pit_values[_t] = max(0.001, min(0.999, _pit_rand))
                _stale_ind = 1.0 if _is_stale_t else 0.0
                _p_stale = _STALE_EWM_LAMBDA * _p_stale + (1.0 - _STALE_EWM_LAMBDA) * _stale_ind

        # ── PIT VARIANCE RECALIBRATION (Beta(a,a) analog) ───────────
        _PIT_VAR_TARGET = 1.0 / 12.0
        _PIT_VAR_LAMBDA = float(getattr(config, 'pit_var_lambda', 0.97))
        _PIT_VAR_1M = 1.0 - _PIT_VAR_LAMBDA
        _PIT_VAR_DZ_LO = float(getattr(config, 'pit_var_dz_lo', 0.30))
        _PIT_VAR_DZ_HI = float(getattr(config, 'pit_var_dz_hi', 0.55))
        _PIT_VAR_DZ_RANGE = _PIT_VAR_DZ_HI - _PIT_VAR_DZ_LO
        _PIT_VAR_MIN_STRETCH = 0.70
        _PIT_VAR_MAX_STRETCH = 1.50

        _ewm_pit_m = 0.5
        _ewm_pit_sq = 1.0 / 3.0
        for _t in range(len(_pit_train_cal)):
            _p = float(_pit_train_cal[_t])
            _ewm_pit_m = _PIT_VAR_LAMBDA * _ewm_pit_m + _PIT_VAR_1M * _p
            _ewm_pit_sq = _PIT_VAR_LAMBDA * _ewm_pit_sq + _PIT_VAR_1M * _p * _p

        for _t in range(n_test):
            _obs_var = _ewm_pit_sq - _ewm_pit_m * _ewm_pit_m
            if _obs_var < 0.005:
                _obs_var = 0.005

            _var_ratio = _obs_var / _PIT_VAR_TARGET
            _var_dev = abs(_var_ratio - 1.0)
            _raw_pit = pit_values[_t]

            if _var_dev > _PIT_VAR_DZ_LO:
                _raw_stretch = _msqrt(_PIT_VAR_TARGET / _obs_var)
                if _raw_stretch < _PIT_VAR_MIN_STRETCH:
                    _raw_stretch = _PIT_VAR_MIN_STRETCH
                elif _raw_stretch > _PIT_VAR_MAX_STRETCH:
                    _raw_stretch = _PIT_VAR_MAX_STRETCH

                if _var_dev >= _PIT_VAR_DZ_HI:
                    _stretch = _raw_stretch
                else:
                    _str = (_var_dev - _PIT_VAR_DZ_LO) / _PIT_VAR_DZ_RANGE
                    _stretch = 1.0 + _str * (_raw_stretch - 1.0)

                _corrected = 0.5 + (_raw_pit - 0.5) * _stretch
                if _corrected < 0.001:
                    _corrected = 0.001
                elif _corrected > 0.999:
                    _corrected = 0.999
                pit_values[_t] = _corrected

            _ewm_pit_m = _PIT_VAR_LAMBDA * _ewm_pit_m + _PIT_VAR_1M * _raw_pit
            _ewm_pit_sq = _PIT_VAR_LAMBDA * _ewm_pit_sq + _PIT_VAR_1M * _raw_pit * _raw_pit

        pit_values = np.clip(pit_values, 0.001, 0.999)

        # AR(1) probit whitening
        if _best_lam > 0:
            _z_probit = norm.ppf(np.clip(pit_values, 0.0001, 0.9999))
            _z_white = np.zeros(n_test)
            _z_white[0] = _z_probit[0]

            _ewm_cross, _ewm_sq = 0.0, 1.0
            if len(_z_probit_cal) > 2:
                for _t in range(1, len(_z_probit_cal)):
                    _ewm_cross = _best_lam * _ewm_cross + _1m_lam * _z_probit_cal[_t-1] * (_z_probit_cal[_t-2] if _t > 1 else 0.0)
                    _ewm_sq = _best_lam * _ewm_sq + _1m_lam * _z_probit_cal[_t-1] ** 2

            for _t in range(1, n_test):
                _ewm_cross = _best_lam * _ewm_cross + _1m_lam * _z_probit[_t-1] * (_z_probit[_t-2] if _t > 1 else (_z_probit_cal[-1] if len(_z_probit_cal) > 0 else 0.0))
                _ewm_sq = _best_lam * _ewm_sq + _1m_lam * _z_probit[_t-1] ** 2
                _rho_t = (_ewm_cross / _ewm_sq) if _ewm_sq > 0.1 else 0.0
                _rho_t = max(-0.3, min(0.3, _rho_t))

                if abs(_rho_t) > 0.01:
                    _z_white[_t] = (_z_probit[_t] - _rho_t * _z_probit[_t-1]) / _msqrt(max(1 - _rho_t * _rho_t, 0.5))
                else:
                    _z_white[_t] = _z_probit[_t]

            pit_values = np.clip(norm.cdf(_z_white), 0.001, 0.999)

        # ── ISOTONIC RECALIBRATION (Kuleshov et al. 2018) ───────────
        _ISO_BLEND_ALPHA = 0.4  # Conservative: 40% isotonic, 60% identity
        try:
            from calibration.isotonic_recalibration import IsotonicRecalibrator
            if len(_pit_train_cal) >= 50:
                _iso_recal = IsotonicRecalibrator()
                _iso_result = _iso_recal.fit(_pit_train_cal)
                if _iso_result.fit_success and not _iso_result.is_identity:
                    _pit_iso = _iso_recal.transform(pit_values)
                    pit_values = (1.0 - _ISO_BLEND_ALPHA) * pit_values + _ISO_BLEND_ALPHA * _pit_iso
                    pit_values = np.clip(pit_values, 0.001, 0.999)
        except Exception:
            pass  # Graceful degradation

        return pit_values, sigma, mu_effective, _S_blend

    @staticmethod
    def _compute_berkowitz_full(pit_values):
        """
        Berkowitz (2001) LR test for Gaussian unified models.

        H0: Φ^{-1}(PIT) ~ N(0,1) iid vs H1: AR(1).
        Returns (p_value, lr_statistic, n_pit).
        """
        try:
            from scipy.stats import chi2
            z = norm.ppf(np.clip(pit_values, 0.0001, 0.9999))
            z = z[np.isfinite(z)]
            n_z = len(z)
            if n_z <= 20:
                return (float('nan'), 0.0, n_z)
            mu_hat = float(np.mean(z))
            var_hat = float(np.var(z, ddof=0))
            z_c = z - mu_hat
            denom = np.sum(z_c[:-1] ** 2)
            rho_hat = float(np.clip(np.sum(z_c[1:] * z_c[:-1]) / denom, -0.99, 0.99)) if denom > 1e-12 else 0.0
            ll_null = -0.5 * n_z * np.log(2 * np.pi) - 0.5 * np.sum(z ** 2)
            sigma_sq = max(var_hat * (1 - rho_hat ** 2) if abs(rho_hat) < 0.99 else var_hat * 0.01, 1e-6)
            ll_alt = -0.5 * np.log(2 * np.pi * var_hat) - 0.5 * (z[0] - mu_hat) ** 2 / var_hat
            for t in range(1, n_z):
                resid = z[t] - (mu_hat + rho_hat * (z[t-1] - mu_hat))
                ll_alt += -0.5 * np.log(2 * np.pi * sigma_sq) - 0.5 * resid ** 2 / sigma_sq
            lr_stat = float(max(2 * (ll_alt - ll_null), 0))
            p_value = float(1 - chi2.cdf(lr_stat, df=3))
            return (p_value, lr_stat, n_z)
        except Exception:
            return (float('nan'), 0.0, 0)

    @classmethod
    def optimize_params(
        cls,
        returns: np.ndarray,
        vol: np.ndarray,
        train_frac: float = 0.7,
        q_min: float = 1e-10,
        q_max: float = 1e-1,
        c_min: float = 0.3,
        c_max: float = 3.0,
        prior_log_q_mean: float = -6.0,
        prior_lambda: float = 1.0
    ) -> Tuple[float, float, float, Dict]:
        """Jointly optimize (q, c) via maximum likelihood with enhanced Bayesian regularization."""
        n = len(returns)

        ret_p005 = np.percentile(returns, 0.5)
        ret_p995 = np.percentile(returns, 99.5)
        returns_robust = np.clip(returns, ret_p005, ret_p995)

        ret_std = float(np.std(returns_robust))
        ret_mean = float(np.mean(returns_robust))
        vol_mean = float(np.mean(vol))
        vol_std = float(np.std(vol))

        if vol_mean > 0:
            vol_cv = vol_std / vol_mean
        else:
            vol_cv = 0.0

        if ret_std > 0:
            rv_ratio = abs(ret_mean) / ret_std
        else:
            rv_ratio = 0.0

        if vol_cv > 0.5 or rv_ratio > 0.15:
            adaptive_prior_mean = prior_log_q_mean + 0.5
            adaptive_lambda = prior_lambda * 0.5
        elif vol_cv < 0.2 and rv_ratio < 0.05:
            adaptive_prior_mean = prior_log_q_mean - 0.3
            adaptive_lambda = prior_lambda * 1.5
        else:
            adaptive_prior_mean = prior_log_q_mean
            adaptive_lambda = prior_lambda

        min_train = min(max(60, int(n * 0.4)), max(n - 5, 1))
        test_window = min(max(20, int(n * 0.1)), max(n - min_train, 5))

        fold_splits = []
        train_end = min_train
        while train_end + test_window <= n:
            test_end = min(train_end + test_window, n)
            if test_end - train_end >= 20:
                fold_splits.append((0, train_end, train_end, test_end))
            train_end += test_window

        if not fold_splits:
            split_idx = int(n * train_frac)
            fold_splits = [(0, split_idx, split_idx, n)]

        # Pre-compute arrays for inner loop (avoid per-element np.ndim checks)
        _returns_r = np.ascontiguousarray(returns_robust.flatten(), dtype=np.float64)
        _vol_sq = np.ascontiguousarray((vol * vol).flatten(), dtype=np.float64)
        _mlog = math.log
        _msqrt = math.sqrt

        def negative_penalized_ll_cv(params: np.ndarray) -> float:
            log_q, log_c = params
            q = 10 ** log_q
            c = 10 ** log_c

            if q <= 0 or c <= 0 or not math.isfinite(q) or not math.isfinite(c):
                return 1e12

            total_ll_oos = 0.0
            total_obs = 0
            # Pre-allocated array for standardized residuals (avoids list append overhead)
            std_buf = np.empty(1000, dtype=np.float64)
            std_count = 0

            for train_start, train_end, test_start, test_end in fold_splits:
                try:
                    ret_train = _returns_r[train_start:train_end]
                    vol_train = vol[train_start:train_end]

                    if len(ret_train) < 3:
                        continue

                    mu_filt_train, P_filt_train, _ = cls.filter(ret_train, vol_train, q, c)

                    mu_pred = float(mu_filt_train[-1])
                    P_pred = float(P_filt_train[-1])

                    if _USE_CV_KERNEL:
                        ll_fold, n_fold, n_written = run_gaussian_cv_test_fold(
                            _returns_r, _vol_sq, q, c,
                            mu_pred, P_pred,
                            test_start, test_end,
                            std_buf, std_count, 1000,
                        )
                        total_ll_oos += ll_fold
                        total_obs += n_fold
                        std_count += n_written
                    else:
                        ll_fold = 0.0

                        for t in range(test_start, test_end):
                            P_pred = P_pred + q

                            R = c * _vol_sq[t]
                            innovation = _returns_r[t] - mu_pred
                            forecast_var = P_pred + R

                            if forecast_var > 1e-12:
                                ll_fold += -0.5 * (_LOG_2PI + _mlog(forecast_var) + (innovation * innovation) / forecast_var)
                                if std_count < 1000:
                                    std_buf[std_count] = innovation / _msqrt(forecast_var)
                                    std_count += 1

                            S_total = P_pred + R
                            K = P_pred / S_total if S_total > 1e-12 else 0.0
                            mu_pred = mu_pred + K * innovation
                            P_pred = (1.0 - K) * P_pred

                        total_ll_oos += ll_fold
                        total_obs += (test_end - test_start)

                except Exception:
                    continue

            if total_obs == 0:
                return 1e12

            avg_ll_oos = total_ll_oos / max(total_obs, 1)

            calibration_penalty = 0.0
            if std_count >= 30:
                try:
                    pit_values = norm.cdf(std_buf[:std_count])
                    ks_stat = _fast_ks_statistic(pit_values)

                    if ks_stat > 0.05:
                        calibration_penalty = -50.0 * ((ks_stat - 0.05) ** 2)

                        if ks_stat > 0.10:
                            calibration_penalty -= 100.0 * (ks_stat - 0.10)

                        if ks_stat > 0.15:
                            calibration_penalty -= 200.0 * (ks_stat - 0.15)
                except Exception:
                    pass

            prior_scale = 1.0 / max(total_obs, 100)
            log_prior_q = -adaptive_lambda * prior_scale * (log_q - adaptive_prior_mean) ** 2
            log_prior_c = -0.1 * prior_scale * (log_c - _LOG10_C_TARGET) ** 2

            penalized_ll = avg_ll_oos + log_prior_q + log_prior_c + calibration_penalty

            if not math.isfinite(penalized_ll):
                return 1e12

            return -penalized_ll

        log_q_min = np.log10(q_min)
        log_q_max = np.log10(q_max)
        log_c_min = np.log10(c_min)
        log_c_max = np.log10(c_max)

        log_q_grid = np.linspace(
            max(log_q_min, adaptive_prior_mean - 2.0),
            min(log_q_max, adaptive_prior_mean + 2.0),
            8)

        log_c_grid = np.linspace(log_c_min, log_c_max, 8)

        best_neg_ll = float('inf')
        best_log_q_grid = adaptive_prior_mean
        best_log_c_grid = np.log10(0.9)

        for lq in log_q_grid:
            for lc in log_c_grid:
                try:
                    neg_ll = negative_penalized_ll_cv(np.array([lq, lc]))
                    if neg_ll < best_neg_ll:
                        best_neg_ll = neg_ll
                        best_log_q_grid = lq
                        best_log_c_grid = lc
                except Exception:
                    continue

        grid_best_q = 10 ** best_log_q_grid
        grid_best_c = 10 ** best_log_c_grid

        bounds = [(log_q_min, log_q_max), (log_c_min, log_c_max)]

        best_result = None
        best_fun = float('inf')

        start_points = [
            np.array([best_log_q_grid, best_log_c_grid]),
            np.array([adaptive_prior_mean, np.log10(0.9)]),
            np.array([best_log_q_grid - 0.5, best_log_c_grid]),
            np.array([best_log_q_grid + 0.5, best_log_c_grid]),
            np.array([best_log_q_grid, best_log_c_grid + 0.15]),
        ]

        for x0 in start_points:
            try:
                result = minimize(
                    negative_penalized_ll_cv,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-7}
                )

                if result.fun < best_fun:
                    best_fun = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is not None and best_result.success:
            log_q_opt, log_c_opt = best_result.x
            q_optimal = 10 ** log_q_opt
            c_optimal = 10 ** log_c_opt
            ll_optimal = -best_result.fun
        else:
            q_optimal = grid_best_q
            c_optimal = grid_best_c
            ll_optimal = -best_neg_ll

        diagnostics = {
            'grid_best_q': float(grid_best_q),
            'grid_best_c': float(grid_best_c),
            'refined_best_q': float(q_optimal),
            'refined_best_c': float(c_optimal),
            'prior_applied': adaptive_lambda > 0,
            'prior_log_q_mean': float(adaptive_prior_mean),
            'prior_lambda': float(adaptive_lambda),
            'vol_cv': float(vol_cv),
            'rv_ratio': float(rv_ratio),
            'ret_mean': float(ret_mean),
            'ret_std': float(ret_std),
            'n_folds': int(len(fold_splits)),
            'adaptive_regularization': True,
            'robust_optimization': True,
            'winsorized': True,
            'optimization_successful': best_result is not None and (best_result.success if best_result else False)
        }

        return q_optimal, c_optimal, ll_optimal, diagnostics
