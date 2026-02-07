"""
===============================================================================
ENSEMBLE DISTILLATION MODEL — Knowledge Transfer from Standard Models
===============================================================================

Implements Professor Wei Chen's Solution 3: Ensemble Distillation from Standard Models

Key Insight: Use the BMA posterior from standard models as an informative prior
for experimental models. This "knowledge distillation" prevents experimental
models from diverging too far from proven baselines while allowing innovation
at the margins.

Mechanism:
    1. Compute BMA posterior from standard models
    2. Use posterior mean/variance as prior for experimental model
    3. Regularize experimental model toward standard ensemble
    4. Allow deviations where data strongly supports them

Prior Structure:
    p(θ_exp) ∝ N(θ_exp | θ_std_mean, λ * Σ_std)
    
where λ controls the strength of regularization toward standards.

Score: 82/100 — Elegant solution that respects existing architecture.

Author: Chinese Staff Professor - Elite Quant Systems
Date: February 2026
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from .base import BaseExperimentalModel


class EnsembleDistillationModel(BaseExperimentalModel):
    """
    Experimental model with knowledge distillation from standard ensemble.
    
    Uses BMA posterior from standard models as regularizing prior,
    preventing the experimental model from diverging too far while
    still allowing innovation.
    
    Parameters:
        q: Process noise variance
        c: Observation noise scale
        phi: AR(1) drift persistence
        lambda_reg: Regularization strength toward standard ensemble
        teacher_params: Parameters from teacher (standard) models
    """
    
    def __init__(
        self,
        lambda_reg: float = 1.0,
        teacher_params: Optional[Dict[str, Dict[str, float]]] = None,
        nu: float = 8.0,
    ):
        """
        Initialize EnsembleDistillationModel.
        
        Args:
            lambda_reg: Regularization strength (higher = closer to teacher)
            teacher_params: Dict of standard model parameters for distillation
            nu: Degrees of freedom for Student-t base
        """
        self.lambda_reg = lambda_reg
        self.nu = nu
        
        # Default teacher parameters (from typical standard model fits)
        self.teacher_params = teacher_params or {
            "kalman_gaussian_momentum": {"q": 1e-6, "c": 1.0, "phi": 0.95},
            "kalman_phi_gaussian_momentum": {"q": 1e-6, "c": 1.0, "phi": 0.90},
            "phi_student_t_nu_8_momentum": {"q": 1e-6, "c": 1.0, "phi": 0.92},
        }
        
        # Compute teacher ensemble mean and variance
        self._compute_teacher_ensemble()
    
    def _compute_teacher_ensemble(self) -> None:
        """Compute weighted mean and variance of teacher parameters."""
        if not self.teacher_params:
            self.teacher_mean = {"q": 1e-6, "c": 1.0, "phi": 0.92}
            self.teacher_var = {"q": 1e-12, "c": 0.1, "phi": 0.01}
            return
        
        # Equal weights for simplicity (could be BIC-weighted)
        n_teachers = len(self.teacher_params)
        
        param_values = {"q": [], "c": [], "phi": []}
        
        for params in self.teacher_params.values():
            for key in param_values:
                if key in params:
                    param_values[key].append(params[key])
        
        self.teacher_mean = {
            key: np.mean(vals) if vals else 0.0
            for key, vals in param_values.items()
        }
        
        self.teacher_var = {
            key: np.var(vals) + 1e-10 if len(vals) > 1 else 1e-6
            for key, vals in param_values.items()
        }
    
    def regularization_penalty(
        self,
        q: float,
        c: float,
        phi: float,
    ) -> float:
        """
        Compute regularization penalty toward teacher ensemble.
        
        Uses squared Mahalanobis distance from teacher mean.
        
        Args:
            q, c, phi: Current parameters
            
        Returns:
            Regularization penalty (to be added to negative log-likelihood)
        """
        penalty = 0.0
        
        # Log-scale for q (since it's very small)
        log_q = np.log(max(q, 1e-12))
        log_q_teacher = np.log(max(self.teacher_mean["q"], 1e-12))
        penalty += self.lambda_reg * (log_q - log_q_teacher) ** 2 / 1.0
        
        # Linear scale for c
        penalty += self.lambda_reg * (c - self.teacher_mean["c"]) ** 2 / max(self.teacher_var["c"], 0.01)
        
        # Linear scale for phi
        penalty += self.lambda_reg * (phi - self.teacher_mean["phi"]) ** 2 / max(self.teacher_var["phi"], 0.001)
        
        return penalty
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        q: float,
        c: float,
        phi: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run filter with regularization toward teacher ensemble.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            q, c, phi: Model parameters
            
        Returns:
            (mu, P, regularized_likelihood)
        """
        n = len(returns)
        mu = np.zeros(n)
        P = np.zeros(n)
        
        mu[0] = 0.0
        P[0] = 1e-4
        
        log_likelihood = 0.0
        
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
        
        # Apply regularization penalty (subtract from log-likelihood)
        penalty = self.regularization_penalty(q, c, phi)
        regularized_ll = log_likelihood - penalty
        
        return mu, P, regularized_ll
    
    def set_teacher_params(self, teacher_params: Dict[str, Dict[str, float]]) -> None:
        """
        Update teacher parameters (e.g., from latest standard model fits).
        
        Args:
            teacher_params: Dict of standard model name -> parameters
        """
        self.teacher_params = teacher_params
        self._compute_teacher_ensemble()
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        init_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit model with distillation regularization.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            init_params: Initial parameter guesses
            
        Returns:
            Dictionary with fitted parameters and diagnostics
        """
        # Start from teacher mean (informed initialization)
        init = init_params or {}
        q0 = init.get("q", self.teacher_mean.get("q", 1e-6))
        c0 = init.get("c", self.teacher_mean.get("c", 1.0))
        phi0 = init.get("phi", self.teacher_mean.get("phi", 0.92))
        
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
        
        # Compute unregularized likelihood for comparison
        penalty = self.regularization_penalty(q_opt, c_opt, phi_opt)
        unregularized_ll = final_ll + penalty
        
        n = len(returns)
        n_params = 4  # q, c, phi, lambda_reg
        bic = -2 * final_ll + n_params * np.log(n)
        
        # Compute deviation from teacher
        deviation_q = abs(np.log(q_opt) - np.log(self.teacher_mean["q"]))
        deviation_c = abs(c_opt - self.teacher_mean["c"])
        deviation_phi = abs(phi_opt - self.teacher_mean["phi"])
        
        return {
            "q": q_opt,
            "c": c_opt,
            "phi": phi_opt,
            "nu": self.nu,
            "lambda_reg": self.lambda_reg,
            "log_likelihood": final_ll,
            "unregularized_ll": unregularized_ll,
            "regularization_penalty": penalty,
            "bic": bic,
            "n_observations": n,
            "n_params": n_params,
            "success": result.success,
            "teacher_mean": self.teacher_mean,
            "deviation_q": deviation_q,
            "deviation_c": deviation_c,
            "deviation_phi": deviation_phi,
        }
