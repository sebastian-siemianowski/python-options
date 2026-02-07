"""
===============================================================================
MULTI-HORIZON MODEL — Temporal Consistency Across Forecast Horizons
===============================================================================

Implements Professor Zhang Yifan's Solution 2: Multi-Horizon Objective Function

Key Insight: Current models optimize single-step-ahead prediction. This model
implements a multi-horizon objective that simultaneously fits 1-day, 5-day,
and 20-day forecasts with shared parameters.

Benefits:
    - Enforces temporal consistency
    - Reduces overfitting to daily noise
    - Better captures drift persistence
    - More robust parameter estimates

Objective:
    L = w₁*L₁ + w₅*L₅ + w₂₀*L₂₀
    
where L_h is the loss at horizon h, with weights reflecting importance.

Score: 86/100 — Addresses a real limitation in current architecture.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .base import BaseExperimentalModel


class MultiHorizonModel(BaseExperimentalModel):
    """
    Kalman filter optimized across multiple forecast horizons.
    
    Fits parameters to simultaneously minimize prediction error
    at 1-day, 5-day, and 20-day horizons.
    
    Parameters:
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) drift persistence
        horizons: List of forecast horizons
        horizon_weights: Weights for each horizon in objective
    """
    
    def __init__(
        self,
        horizons: List[int] = None,
        horizon_weights: List[float] = None,
        nu: float = 8.0,
    ):
        """
        Initialize MultiHorizonModel.
        
        Args:
            horizons: Forecast horizons in days (default: [1, 5, 20])
            horizon_weights: Weights for each horizon (default: equal)
            nu: Degrees of freedom for Student-t
        """
        self.horizons = horizons or [1, 5, 20]
        
        if horizon_weights is None:
            # Equal weights by default
            self.horizon_weights = [1.0 / len(self.horizons)] * len(self.horizons)
        else:
            # Normalize weights
            total = sum(horizon_weights)
            self.horizon_weights = [w / total for w in horizon_weights]
        
        self.nu = nu
    
    def multi_step_predict(
        self,
        mu_t: float,
        P_t: float,
        phi: float,
        q: float,
        h: int,
    ) -> Tuple[float, float]:
        """
        Compute h-step ahead prediction.
        
        For AR(1) process: E[μ_{t+h}|t] = φ^h * μ_t
                          Var[μ_{t+h}|t] = P_t * φ^{2h} + q * (1 - φ^{2h})/(1 - φ²)
        
        Args:
            mu_t: Current state estimate
            P_t: Current state variance
            phi: AR(1) coefficient
            q: Process noise
            h: Forecast horizon
            
        Returns:
            (mu_pred, P_pred): h-step ahead mean and variance
        """
        phi_h = phi ** h
        mu_pred = phi_h * mu_t
        
        if abs(phi) < 0.9999:
            # Geometric series for variance
            var_factor = (1 - phi ** (2 * h)) / (1 - phi ** 2)
        else:
            var_factor = h
        
        P_pred = P_t * phi ** (2 * h) + q * var_factor
        
        return mu_pred, P_pred
    
    def compute_horizon_loss(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
        horizon: int,
    ) -> float:
        """
        Compute prediction loss at a specific horizon.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            q, c, phi: Model parameters
            horizon: Forecast horizon
            
        Returns:
            Average squared prediction error at this horizon
        """
        n = len(returns)
        
        if n <= horizon + 60:
            return 1e10
        
        # Run filter
        mu = np.zeros(n)
        P = np.zeros(n)
        
        mu[0] = 0.0
        P[0] = 1e-4
        
        for t in range(1, n):
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            K = P_pred / S if S > 0 else 0
            
            innovation = returns[t] - mu_pred
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
        
        # Compute h-step ahead predictions and errors
        losses = []
        
        for t in range(60, n - horizon):
            # Predict h steps ahead
            mu_h, P_h = self.multi_step_predict(mu[t], P[t], phi, q, horizon)
            
            # Cumulative return over horizon
            actual_return = np.sum(returns[t+1:t+1+horizon])
            
            # Predicted cumulative return (sum of h one-step predictions)
            # For AR(1), cumulative prediction is sum of geometric series
            if abs(phi) < 0.9999:
                predicted_return = mu[t] * phi * (1 - phi**horizon) / (1 - phi)
            else:
                predicted_return = mu[t] * phi * horizon
            
            # Prediction error
            sigma_h = c * np.mean(vol[t+1:t+1+horizon]) * np.sqrt(horizon)
            error = (actual_return - predicted_return) / max(sigma_h, 1e-6)
            
            losses.append(error ** 2)
        
        return np.mean(losses) if losses else 1e10
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run filter and compute multi-horizon objective.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            q, c, phi: Model parameters
            
        Returns:
            (mu, P, combined_loss): State estimates and combined loss
        """
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        
        mu[0] = 0.0
        P[0] = 1e-4
        
        log_likelihood = 0.0
        
        for t in range(1, n):
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            K = P_pred / S if S > 0 else 0
            
            innovation = returns[t] - mu_pred
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
            
            # Student-t log-likelihood for 1-step
            if S > 0:
                z = innovation / np.sqrt(S)
                ll_t = (
                    gammaln((self.nu + 1) / 2) 
                    - gammaln(self.nu / 2)
                    - 0.5 * np.log(self.nu * np.pi * S)
                    - ((self.nu + 1) / 2) * np.log(1 + z**2 / self.nu)
                )
                log_likelihood += ll_t
        
        # Compute losses at each horizon
        horizon_losses = []
        for h, w in zip(self.horizons, self.horizon_weights):
            loss_h = self.compute_horizon_loss(returns, vol, q, c, phi, h)
            horizon_losses.append(w * loss_h)
        
        combined_loss = sum(horizon_losses)
        
        # Return likelihood for compatibility (negated combined loss)
        return mu, P, log_likelihood - 0.1 * combined_loss * n
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        init_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit model parameters using multi-horizon objective.
        
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
        n_params = 3
        bic = -2 * final_ll + n_params * np.log(n)
        
        # Compute per-horizon losses for diagnostics
        horizon_diagnostics = {}
        for h in self.horizons:
            loss_h = self.compute_horizon_loss(returns, vol, q_opt, c_opt, phi_opt, h)
            horizon_diagnostics[f"loss_h{h}"] = loss_h
        
        return {
            "q": q_opt,
            "c": c_opt,
            "phi": phi_opt,
            "nu": self.nu,
            "horizons": self.horizons,
            "horizon_weights": self.horizon_weights,
            "log_likelihood": final_ll,
            "bic": bic,
            "n_observations": n,
            "n_params": n_params,
            "success": result.success,
            **horizon_diagnostics,
        }
