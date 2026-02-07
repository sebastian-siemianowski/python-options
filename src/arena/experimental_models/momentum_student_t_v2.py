"""
===============================================================================
MOMENTUM STUDENT-T V2 — Adaptive Tail Coupling Model
===============================================================================

Upgraded Student-t with adaptive tail coupling:
    - Tail heaviness (ν) adapts based on momentum persistence
    - Strong momentum → lighter tails (trending, lower uncertainty)
    - Weak momentum → heavier tails (ranging, higher uncertainty)
    - Implements "momentum confidence scaling" concept

Key Innovation: ν adapts based on momentum signal strength.

The intuition is that strong momentum signals indicate regime clarity,
which should reduce tail uncertainty. Conversely, weak momentum 
indicates regime ambiguity, requiring heavier tails.

Parameters:
    q: Process noise variance
    c: Observation noise scale
    phi: AR(1) drift persistence
    nu_base: Base degrees of freedom (e.g., 8)
    nu_range: Range for ν adaptation (e.g., 4)
    alpha: Momentum-to-ν coupling strength (0-1)
    
Effective ν:
    ν_eff = nu_base + alpha * nu_range * |momentum_signal|
    
When momentum is strong: ν increases → lighter tails
When momentum is weak: ν stays at base → heavier tails

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize

from .base import BaseExperimentalModel


class MomentumStudentTV2(BaseExperimentalModel):
    """
    Upgraded Momentum Student-t with Adaptive Tail Coupling (v2.0).
    
    Key Innovation: ν adapts based on momentum signal strength.
    
    The intuition is that strong momentum signals indicate regime clarity,
    which should reduce tail uncertainty. Conversely, weak momentum 
    indicates regime ambiguity, requiring heavier tails.
    """
    
    def __init__(
        self,
        nu_base: float = 6.0,
        nu_range: float = 8.0,
        alpha: float = 0.5,
        momentum_lookbacks: List[int] = None,
    ):
        """
        Initialize MomentumStudentTV2 model.
        
        Args:
            nu_base: Base degrees of freedom (minimum ν)
            nu_range: Range for ν adaptation
            alpha: Momentum-to-ν coupling strength (0-1)
            momentum_lookbacks: List of lookback windows for momentum
        """
        self.nu_base = nu_base
        self.nu_range = nu_range
        self.alpha = alpha
        self.momentum_lookbacks = momentum_lookbacks or [5, 10, 20]
        
        self._momentum_cache: Optional[np.ndarray] = None
    
    def compute_momentum_signal(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute normalized momentum signal.
        
        Args:
            returns: Log returns array
            
        Returns:
            Normalized momentum signal in [0, 1]
        """
        n = len(returns)
        momentum = np.zeros(n)
        
        for lb in self.momentum_lookbacks:
            if lb >= n:
                continue
            
            # Cumulative return over lookback
            for t in range(lb, n):
                momentum[t] += np.sum(returns[t-lb:t])
        
        # Normalize to [0, 1] based on signal strength
        if len(self.momentum_lookbacks) > 0:
            momentum /= len(self.momentum_lookbacks)
        
        # Z-score normalization
        valid = momentum[20:]  # Skip warmup
        if len(valid) > 0:
            mu = np.mean(valid)
            sigma = np.std(valid)
            if sigma > 1e-10:
                momentum = (momentum - mu) / sigma
        
        # Clip and transform to [0, 1]
        momentum = np.clip(np.abs(momentum), 0, 3) / 3.0
        
        self._momentum_cache = momentum
        return momentum
    
    def get_effective_nu(self, t: int) -> float:
        """
        Get effective degrees of freedom at time t.
        
        ν_eff = nu_base + alpha * nu_range * momentum_strength
        
        Args:
            t: Time index
            
        Returns:
            Effective ν at time t
        """
        if self._momentum_cache is None or t >= len(self._momentum_cache):
            return self.nu_base
        
        momentum_strength = self._momentum_cache[t]
        nu_eff = self.nu_base + self.alpha * self.nu_range * momentum_strength
        
        # Clamp to valid range
        return np.clip(nu_eff, 3.0, 50.0)
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run Kalman filter with adaptive ν.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            
        Returns:
            (mu, P, log_likelihood): Drift estimates, variance, total LL
        """
        from scipy.special import gammaln
        
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        
        # Compute momentum signal
        self.compute_momentum_signal(returns)
        
        # Initialize
        mu[0] = 0.0
        P[0] = 1e-4
        
        log_likelihood = 0.0
        
        for t in range(1, n):
            # Prediction step
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            
            # Observation noise
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            
            # Adaptive ν
            nu_t = self.get_effective_nu(t)
            
            # Update step (Student-t likelihood)
            innovation = returns[t] - mu_pred
            S = P_pred + sigma_obs**2
            
            # Kalman gain (standard)
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
        Fit model parameters via MLE.
        
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
        
        def neg_ll(params):
            q, c, phi = params
            if q <= 0 or c <= 0 or not (-1 < phi < 1):
                return 1e10
            try:
                _, _, ll = self.filter(returns, vol, q, c, phi)
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
        _, _, final_ll = self.filter(returns, vol, q_opt, c_opt, phi_opt)
        
        n = len(returns)
        n_params = 5  # q, c, phi, nu_base, alpha (nu_range and alpha are hyperparams)
        bic = -2 * final_ll + n_params * np.log(n)
        
        return {
            "q": q_opt,
            "c": c_opt,
            "phi": phi_opt,
            "nu_base": self.nu_base,
            "nu_range": self.nu_range,
            "alpha": self.alpha,
            "log_likelihood": final_ll,
            "bic": bic,
            "n_observations": n,
            "n_params": n_params,
            "success": result.success,
        }
