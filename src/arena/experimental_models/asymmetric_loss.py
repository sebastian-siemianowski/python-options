"""
===============================================================================
ASYMMETRIC LOSS MODEL — Downside-Weighted Prediction
===============================================================================

Implements Professor Wei Chen's Solution 1: Asymmetric Loss Function Integration

Key Insight: Hedge funds care about tail risk asymmetrically. The standard
symmetric likelihood treats +5σ and -5σ equally, which is economically naive.

This model uses an asymmetric loss function that penalizes downside prediction
errors more heavily than upside errors.

Loss Function:
    L(y, μ, σ) = 
        α * (y - μ)²/σ²  if y < μ  (downside: penalize more)
        (y - μ)²/σ²      if y ≥ μ  (upside: standard)

where α > 1 is the asymmetry parameter (e.g., α = 2 means 2x penalty for
underpredicting negative returns).

Score: 78/100 — Sound principle but may introduce optimization instability.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize

from .base import BaseExperimentalModel


class AsymmetricLossModel(BaseExperimentalModel):
    """
    Kalman filter with asymmetric loss function.
    
    Penalizes downside prediction errors more heavily, reflecting
    the asymmetric risk preferences of hedge fund portfolios.
    
    Parameters:
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) drift persistence
        alpha: Asymmetry parameter (>1 means heavier downside penalty)
    """
    
    def __init__(
        self,
        alpha: float = 2.0,
        nu: float = 8.0,
    ):
        """
        Initialize AsymmetricLossModel.
        
        Args:
            alpha: Asymmetry parameter (downside multiplier)
            nu: Degrees of freedom for Student-t base distribution
        """
        self.alpha = alpha
        self.nu = nu
    
    def asymmetric_loss(
        self,
        y: float,
        mu: float,
        sigma: float,
    ) -> float:
        """
        Compute asymmetric loss for a single observation.
        
        Args:
            y: Actual observation
            mu: Predicted mean
            sigma: Predicted std dev
            
        Returns:
            Asymmetric loss value
        """
        z = (y - mu) / sigma
        
        if y < mu:
            # Downside: heavier penalty
            return self.alpha * z ** 2
        else:
            # Upside: standard penalty
            return z ** 2
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run Kalman filter with asymmetric loss scoring.
        
        The filter itself is standard, but the scoring uses
        asymmetric loss instead of symmetric log-likelihood.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            q: Process noise variance
            c: Observation noise scale
            phi: AR(1) persistence
            
        Returns:
            (mu, P, negative_loss): Drift estimates, variance, negative asymmetric loss
        """
        from scipy.special import gammaln
        
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        
        mu[0] = 0.0
        P[0] = 1e-4
        
        total_loss = 0.0
        
        for t in range(1, n):
            # Prediction step
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            
            # Observation noise
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            
            # Update step
            innovation = returns[t] - mu_pred
            S = P_pred + sigma_obs**2
            K = P_pred / S if S > 0 else 0
            
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
            
            # Asymmetric loss (instead of log-likelihood)
            sigma_total = np.sqrt(S)
            loss_t = self.asymmetric_loss(returns[t], mu_pred, sigma_total)
            total_loss += loss_t
        
        # Return negative loss (for maximization)
        return mu, P, -total_loss
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        init_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit model parameters by minimizing asymmetric loss.
        
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
        
        def neg_score(params):
            q, c, phi = params
            if q <= 0 or c <= 0 or not (-1 < phi < 1):
                return 1e10
            try:
                _, _, score = self.filter(returns, vol, q, c, phi)
                return -score  # Minimize negative of score
            except:
                return 1e10
        
        result = minimize(
            neg_score,
            x0=[q0, c0, phi0],
            method='L-BFGS-B',
            bounds=[(1e-10, 1e-2), (0.1, 5.0), (-0.99, 0.99)],
        )
        
        q_opt, c_opt, phi_opt = result.x
        _, _, final_score = self.filter(returns, vol, q_opt, c_opt, phi_opt)
        
        n = len(returns)
        n_params = 4  # q, c, phi, alpha
        
        # Compute pseudo-BIC (using asymmetric loss instead of likelihood)
        bic = 2 * (-final_score) + n_params * np.log(n)
        
        return {
            "q": q_opt,
            "c": c_opt,
            "phi": phi_opt,
            "alpha": self.alpha,
            "nu": self.nu,
            "asymmetric_loss": -final_score,
            "log_likelihood": final_score,  # Pseudo-likelihood
            "bic": bic,
            "n_observations": n,
            "n_params": n_params,
            "success": result.success,
        }
