"""
===============================================================================
MOMENTUM STUDENT-T REGIME-COUPLED — Regime-Aware Tail Dynamics Model
===============================================================================

Student-t with regime-aware momentum processing:
    - Different ν per regime (pre-specified, not estimated)
    - Momentum signal weighted by regime confidence
    - Crisis regime uses heaviest tails regardless of momentum

Key Innovation: Regime assignment influences both ν selection AND
momentum weighting. Crisis regimes override momentum signals.

Regime ν Assignment:
    - LOW_VOL_TREND (0): ν=12 (near-Gaussian, trending is stable)
    - HIGH_VOL_TREND (1): ν=6 (moderate tails)
    - LOW_VOL_RANGE (2): ν=8 (stable ranging)
    - HIGH_VOL_RANGE (3): ν=4 (heavy tails, volatility clustering)
    - CRISIS_JUMP (4): ν=3 (extreme tails, momentum ignored)

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Dict, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize

from .base import BaseExperimentalModel


class MomentumStudentTRegimeCoupled(BaseExperimentalModel):
    """
    Regime-Coupled Momentum Student-t Model.
    
    Key Innovation: Regime assignment influences both ν selection AND
    momentum weighting. Crisis regimes override momentum signals.
    
    Regime ν Assignment:
        - LOW_VOL_TREND (0): ν=12 (near-Gaussian, trending is stable)
        - HIGH_VOL_TREND (1): ν=6 (moderate tails)
        - LOW_VOL_RANGE (2): ν=8 (stable ranging)
        - HIGH_VOL_RANGE (3): ν=4 (heavy tails, volatility clustering)
        - CRISIS_JUMP (4): ν=3 (extreme tails, momentum ignored)
    """
    
    # Regime-specific degrees of freedom
    REGIME_NU = {
        0: 12.0,  # LOW_VOL_TREND
        1: 6.0,   # HIGH_VOL_TREND
        2: 8.0,   # LOW_VOL_RANGE
        3: 4.0,   # HIGH_VOL_RANGE
        4: 3.0,   # CRISIS_JUMP
    }
    
    # Regime-specific momentum weights
    REGIME_MOMENTUM_WEIGHT = {
        0: 1.0,   # Full momentum in low vol trend
        1: 0.7,   # Reduced in high vol trend
        2: 0.5,   # Moderate in ranging
        3: 0.3,   # Low weight in high vol range
        4: 0.0,   # No momentum in crisis (model dominates)
    }
    
    def __init__(self):
        """Initialize MomentumStudentTRegimeCoupled model."""
        self._momentum_cache = None
        self._regime_cache = None
    
    def assign_regimes(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
    ) -> np.ndarray:
        """
        Assign regimes based on volatility and drift.
        
        Uses same logic as tune.py for consistency.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            
        Returns:
            Array of regime labels (0-4)
        """
        n = len(returns)
        regimes = np.zeros(n, dtype=int)
        
        # Rolling statistics
        window = 60
        
        for t in range(window, n):
            recent_vol = vol[t]
            vol_median = np.median(vol[max(0, t-252):t])
            drift = np.mean(returns[t-window:t])
            
            is_trending = abs(drift) > 0.0005  # ~12.5% annualized
            is_high_vol = recent_vol > 1.3 * vol_median
            is_low_vol = recent_vol < 0.85 * vol_median
            is_crisis = recent_vol > 2.0 * vol_median
            
            if is_crisis:
                regimes[t] = 4
            elif is_low_vol and is_trending:
                regimes[t] = 0
            elif is_high_vol and is_trending:
                regimes[t] = 1
            elif is_low_vol and not is_trending:
                regimes[t] = 2
            else:
                regimes[t] = 3
        
        self._regime_cache = regimes
        return regimes
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        regimes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run regime-coupled filter.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            regimes: Optional pre-computed regime labels
            
        Returns:
            (mu, P, log_likelihood): Drift estimates, variance, total LL
        """
        from scipy.special import gammaln
        
        n = len(returns)
        
        if regimes is None:
            regimes = self.assign_regimes(returns, vol)
        
        mu = np.zeros(n)
        P = np.zeros(n)
        
        mu[0] = 0.0
        P[0] = 1e-4
        
        log_likelihood = 0.0
        
        for t in range(1, n):
            regime = regimes[t]
            nu_t = self.REGIME_NU.get(regime, 8.0)
            
            # Prediction
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            
            # Observation noise
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            
            # Update
            innovation = returns[t] - mu_pred
            S = P_pred + sigma_obs**2
            K = P_pred / S if S > 0 else 0
            
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
            
            # Student-t log-likelihood
            if S > 0:
                z = innovation / np.sqrt(S)
                ll_t = (
                    gammaln((nu_t + 1) / 2) 
                    - gammaln(nu_t / 2)
                    - 0.5 * np.log(nu_t * np.pi * S)
                    - ((nu_t + 1) / 2) * np.log(1 + z**2 / nu_t)
                )
                log_likelihood += ll_t
        
        return mu, P, log_likelihood
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        init_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit model parameters.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            init_params: Initial parameter guesses
            
        Returns:
            Dictionary with fitted parameters and diagnostics
        """
        init = init_params or {}
        q0 = init.get("q", 1e-6)
        c0 = init.get("c", 1.0)
        phi0 = init.get("phi", 0.95)
        
        regimes = self.assign_regimes(returns, vol)
        
        def neg_ll(params):
            q, c, phi = params
            if q <= 0 or c <= 0 or not (-1 < phi < 1):
                return 1e10
            try:
                _, _, ll = self.filter(returns, vol, q, c, phi, regimes)
                return -ll
            except:
                return 1e10
        
        result = minimize(
            neg_ll,
            x0=[q0, c0, phi0],
            method='L-BFGS-B',
            bounds=[(1e-10, 1e-2), (0.1, 5.0), (-0.99, 0.99)],
        )
        
        q_opt, c_opt, phi_opt = result.x
        _, _, final_ll = self.filter(returns, vol, q_opt, c_opt, phi_opt, regimes)
        
        n = len(returns)
        n_params = 3  # Only q, c, phi (ν per regime is fixed)
        bic = -2 * final_ll + n_params * np.log(n)
        
        return {
            "q": q_opt,
            "c": c_opt,
            "phi": phi_opt,
            "regime_nu": self.REGIME_NU,
            "log_likelihood": final_ll,
            "bic": bic,
            "n_observations": n,
            "n_params": n_params,
            "success": result.success,
        }
