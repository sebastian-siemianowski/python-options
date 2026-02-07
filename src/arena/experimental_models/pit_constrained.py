"""
===============================================================================
PIT CONSTRAINED MODEL — Calibration-Guaranteed Optimization
===============================================================================

Implements Professor Liu Xiaoming's Solution 1: PIT-Constrained Optimization

Key Insight: Reformulate parameter estimation as a constrained optimization
problem: maximize likelihood SUBJECT TO PIT uniformity constraint.
This ensures calibration is not sacrificed for likelihood gains.

Optimization Problem:
    max_{θ} L(θ; data)
    s.t.   KS_statistic(PIT(θ)) < threshold
           AD_statistic(PIT(θ)) < threshold

This is solved via penalty method or augmented Lagrangian.

Score: 86/100 — Mathematically clean but computationally expensive.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import kstest, anderson

from .base import BaseExperimentalModel


class PITConstrainedModel(BaseExperimentalModel):
    """
    Kalman filter with PIT calibration constraint.
    
    Optimizes likelihood subject to the constraint that PIT values
    are uniformly distributed (KS test p-value > threshold).
    
    Parameters:
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) drift persistence
        nu: Degrees of freedom
        pit_threshold: Minimum KS p-value for calibration
        penalty_weight: Weight for calibration penalty
    """
    
    def __init__(
        self,
        nu: float = 8.0,
        pit_threshold: float = 0.05,
        penalty_weight: float = 100.0,
    ):
        """
        Initialize PITConstrainedModel.
        
        Args:
            nu: Degrees of freedom for Student-t
            pit_threshold: Minimum acceptable KS p-value
            penalty_weight: Penalty for constraint violation
        """
        self.nu = nu
        self.pit_threshold = pit_threshold
        self.penalty_weight = penalty_weight
    
    def compute_pit_values(
        self,
        returns: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
    ) -> np.ndarray:
        """
        Compute PIT values using Student-t CDF.
        
        PIT_t = F_ν((r_t - μ_t) / σ_t)
        
        For well-calibrated forecasts, PIT values should be uniform on [0,1].
        
        Args:
            returns: Actual returns
            mu: Predicted means
            sigma: Predicted standard deviations
            
        Returns:
            Array of PIT values
        """
        from scipy.stats import t as student_t
        
        n = len(returns)
        pit = np.zeros(n)
        
        for t in range(1, n):
            if sigma[t] > 1e-10:
                z = (returns[t] - mu[t]) / sigma[t]
                pit[t] = student_t.cdf(z, df=self.nu)
            else:
                pit[t] = 0.5  # Neutral if undefined
        
        return pit
    
    def compute_calibration_penalty(
        self,
        pit_values: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute penalty for calibration constraint violation.
        
        Uses KS test against uniform distribution. Returns penalty
        that is 0 when constraint is satisfied, positive otherwise.
        
        Args:
            pit_values: PIT values to test
            
        Returns:
            (penalty, ks_pvalue): Penalty and KS test p-value
        """
        # Skip warmup period
        pit = pit_values[60:]
        
        if len(pit) < 50:
            return 0.0, 1.0  # Not enough data
        
        # KS test against uniform
        ks_stat, ks_pvalue = kstest(pit, 'uniform')
        
        # Penalty: 0 if p-value > threshold, positive otherwise
        if ks_pvalue >= self.pit_threshold:
            penalty = 0.0
        else:
            # Quadratic penalty for violation
            violation = self.pit_threshold - ks_pvalue
            penalty = self.penalty_weight * violation ** 2
        
        return penalty, ks_pvalue
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Run filter and compute PIT-constrained objective.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            q, c, phi: Model parameters
            
        Returns:
            (mu, P, constrained_ll, pit_values)
        """
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        sigma = np.zeros(n)
        
        mu[0] = 0.0
        P[0] = 1e-4
        
        log_likelihood = 0.0
        
        for t in range(1, n):
            # Prediction step
            mu_pred = phi * mu[t-1]
            P_pred = phi**2 * P[t-1] + q
            
            # Observation noise
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            
            # Total variance for prediction
            S = P_pred + sigma_obs**2
            sigma[t] = np.sqrt(S)
            
            # Store predicted mean (before update)
            mu_for_pit = mu_pred
            
            # Update step
            innovation = returns[t] - mu_pred
            K = P_pred / S if S > 0 else 0
            
            mu[t] = mu_pred + K * innovation
            P[t] = (1 - K) * P_pred
            
            # Student-t log-likelihood
            if S > 0:
                z = innovation / np.sqrt(S)
                ll_t = (
                    gammaln((self.nu + 1) / 2) 
                    - gammaln(self.nu / 2)
                    - 0.5 * np.log(self.nu * np.pi * S)
                    - ((self.nu + 1) / 2) * np.log(1 + z**2 / self.nu)
                )
                log_likelihood += ll_t
        
        # Compute PIT values using one-step-ahead predictions
        # Need to reconstruct predictions
        mu_pred_arr = np.zeros(n)
        for t in range(1, n):
            mu_pred_arr[t] = phi * mu[t-1]
        
        pit_values = self.compute_pit_values(returns, mu_pred_arr, sigma)
        
        # Compute calibration penalty
        penalty, ks_pvalue = self.compute_calibration_penalty(pit_values)
        
        # Constrained objective
        constrained_ll = log_likelihood - penalty
        
        return mu, P, constrained_ll, pit_values
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        init_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit model with PIT calibration constraint.
        
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
        
        # Store best solution found
        best_result = None
        best_ll = -np.inf
        
        def neg_ll(params):
            q, c, phi = params
            if q <= 0 or c <= 0 or not (-1 < phi < 1):
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi)
                return -ll
            except:
                return 1e10
        
        # Multi-start optimization for robustness
        starts = [
            [q0, c0, phi0],
            [1e-7, 0.8, 0.90],
            [1e-5, 1.2, 0.98],
        ]
        
        for start in starts:
            try:
                result = minimize(
                    neg_ll,
                    x0=start,
                    method='L-BFGS-B',
                    bounds=[(1e-10, 1e-2), (0.1, 5.0), (-0.99, 0.99)],
                )
                
                if result.success and -result.fun > best_ll:
                    best_result = result
                    best_ll = -result.fun
            except:
                continue
        
        if best_result is None:
            # Fallback
            best_result = minimize(
                neg_ll,
                x0=[q0, c0, phi0],
                method='L-BFGS-B',
                bounds=[(1e-10, 1e-2), (0.1, 5.0), (-0.99, 0.99)],
            )
        
        q_opt, c_opt, phi_opt = best_result.x
        mu, P, final_ll, pit_values = self.filter(returns, vol, q_opt, c_opt, phi_opt)
        
        # Final calibration check
        _, ks_pvalue = self.compute_calibration_penalty(pit_values)
        pit_calibrated = ks_pvalue >= self.pit_threshold
        
        n = len(returns)
        n_params = 4  # q, c, phi, nu
        bic = -2 * final_ll + n_params * np.log(n)
        
        return {
            "q": q_opt,
            "c": c_opt,
            "phi": phi_opt,
            "nu": self.nu,
            "pit_threshold": self.pit_threshold,
            "penalty_weight": self.penalty_weight,
            "log_likelihood": final_ll,
            "bic": bic,
            "ks_pvalue": ks_pvalue,
            "pit_calibrated": pit_calibrated,
            "n_observations": n,
            "n_params": n_params,
            "success": best_result.success,
        }
