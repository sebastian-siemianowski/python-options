"""
===============================================================================
ONLINE BAYESIAN MODEL AVERAGING — Dynamic Posterior Over Model Space
===============================================================================

Implements Professor Liu Xiaoming's Solution 5: Online Bayesian Model Averaging

Key Insight: Instead of selecting one model, maintain a dynamic posterior
distribution over the model space. Let the data continuously update model
weights based on predictive performance.

Model Structure:
    p(M_k | D_{1:t}) ∝ p(r_t | M_k, D_{1:t-1}) * p(M_k | D_{1:t-1})
    
The predictive distribution is:
    p(r_{t+1} | D_{1:t}) = Σ_k p(M_k | D_{1:t}) * p(r_{t+1} | M_k, D_{1:t})

This is "model stacking" done online with proper Bayesian updating.

Score: 94/100 — Optimal model combination under uncertainty.

Author: Chinese Staff Professor Panel Implementation
Date: February 2026
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.special import gammaln

from .base import BaseExperimentalModel


class OnlineBayesianModelAvgModel(BaseExperimentalModel):
    """
    Online Bayesian Model Averaging with dynamic weights.
    
    Maintains posterior probabilities over a set of sub-models and
    updates them online as new data arrives.
    
    Sub-models:
        - Gaussian Kalman filter (baseline)
        - Student-t with varying nu (heavy tails)
        - High/Low persistence variants
    
    The posterior is updated using predictive likelihoods:
        w_k^{t+1} ∝ w_k^t * p(r_t | M_k)
    """
    
    def __init__(self, forgetting_factor: float = 0.99):
        """
        Initialize OnlineBayesianModelAvgModel.
        
        Args:
            forgetting_factor: Discount factor for past observations (0.95-1.0)
                             Values < 1 allow model weights to adapt over time
        """
        self.forgetting_factor = forgetting_factor
        
        # Define sub-model specifications
        self.sub_models = [
            {'name': 'gaussian_stable', 'q': 1e-7, 'c': 0.9, 'phi': 0.2, 'nu': 100},
            {'name': 'gaussian_responsive', 'q': 1e-5, 'c': 1.1, 'phi': 0.0, 'nu': 100},
            {'name': 'student_t_8', 'q': 1e-6, 'c': 1.0, 'phi': 0.1, 'nu': 8},
            {'name': 'student_t_5', 'q': 1e-6, 'c': 1.0, 'phi': 0.1, 'nu': 5},
            {'name': 'student_t_4', 'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'nu': 4},
            {'name': 'high_persistence', 'q': 1e-7, 'c': 1.0, 'phi': 0.5, 'nu': 8},
            {'name': 'crisis_mode', 'q': 1e-5, 'c': 1.5, 'phi': 0.0, 'nu': 4},
        ]
        self.n_models = len(self.sub_models)
    
    def student_t_logpdf(
        self,
        x: float,
        mu: float,
        sigma: float,
        nu: float,
    ) -> float:
        """Compute Student-t log-pdf."""
        if sigma <= 0:
            return -1e10
        z = (x - mu) / sigma
        if nu > 50:  # Approximately Gaussian
            return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * z**2
        else:
            return (
                gammaln((nu + 1) / 2)
                - gammaln(nu / 2)
                - 0.5 * np.log(nu * np.pi * sigma**2)
                - ((nu + 1) / 2) * np.log(1 + z**2 / nu)
            )
    
    def filter_with_weights(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """
        Run online BMA filter.
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            
        Returns:
            (mu_bma, sigma_bma, total_ll, weight_history)
        """
        n = len(returns)
        
        # Initialize model states and weights
        mu = np.zeros((n, self.n_models))
        P = np.zeros((n, self.n_models))
        weights = np.ones(self.n_models) / self.n_models
        weight_history = np.zeros((n, self.n_models))
        
        # BMA outputs
        mu_bma = np.zeros(n)
        sigma_bma = np.zeros(n)
        
        # Initialize
        for k in range(self.n_models):
            mu[0, k] = 0.0
            P[0, k] = 1e-4
        
        weight_history[0] = weights.copy()
        total_ll = 0.0
        
        for t in range(1, n):
            pred_liks = np.zeros(self.n_models)
            mu_pred_all = np.zeros(self.n_models)
            sigma_pred_all = np.zeros(self.n_models)
            
            # Run each sub-model
            for k, model in enumerate(self.sub_models):
                q = model['q']
                c = model['c']
                phi = model['phi']
                nu = model['nu']
                
                # Prediction
                mu_pred = phi * mu[t-1, k]
                P_pred = phi**2 * P[t-1, k] + q
                
                sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
                S = P_pred + sigma_obs**2
                sigma_pred = np.sqrt(S)
                
                mu_pred_all[k] = mu_pred
                sigma_pred_all[k] = sigma_pred
                
                # Predictive likelihood
                pred_liks[k] = self.student_t_logpdf(
                    returns[t], mu_pred, sigma_pred, nu
                )
                
                # Kalman update
                innovation = returns[t] - mu_pred
                K = P_pred / S if S > 0 else 0
                mu[t, k] = mu_pred + K * innovation
                P[t, k] = (1 - K) * P_pred
            
            # Update weights using predictive likelihoods
            # Apply forgetting factor for adaptivity
            log_weights = np.log(weights + 1e-10) * self.forgetting_factor + pred_liks
            
            # Normalize (log-sum-exp for stability)
            max_log = np.max(log_weights)
            weights = np.exp(log_weights - max_log)
            weights = weights / (np.sum(weights) + 1e-10)
            
            weight_history[t] = weights.copy()
            
            # BMA predictions (weighted average)
            mu_bma[t] = np.sum(weights * mu_pred_all)
            # Variance: E[σ²] + Var[μ] (total variance formula)
            var_within = np.sum(weights * sigma_pred_all**2)
            var_between = np.sum(weights * (mu_pred_all - mu_bma[t])**2)
            sigma_bma[t] = np.sqrt(var_within + var_between)
            
            # Marginal likelihood (for scoring)
            total_ll += max_log + np.log(np.sum(np.exp(log_weights - max_log)))
        
        return mu_bma, sigma_bma, total_ll, weight_history
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        init_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit online BMA model (no explicit parameter optimization needed).
        
        Args:
            returns: Log returns
            vol: EWMA volatility
            init_params: Not used (model is self-tuning)
            
        Returns:
            Dictionary with fitted parameters and diagnostics
        """
        import time
        start_time = time.time()
        
        mu_bma, sigma_bma, total_ll, weight_history = self.filter_with_weights(
            returns, vol
        )
        
        n = len(returns)
        # Effective parameters: weighted average of sub-model parameters
        n_params = 3  # Approximately, since it's a weighted combination
        bic = -2 * total_ll + n_params * np.log(n)
        aic = -2 * total_ll + 2 * n_params
        
        # Final model weights
        final_weights = weight_history[-1]
        
        # Best model and its parameters
        best_model_idx = np.argmax(final_weights)
        best_model = self.sub_models[best_model_idx]
        
        # Average parameters (weighted)
        avg_q = np.sum(final_weights * np.array([m['q'] for m in self.sub_models]))
        avg_c = np.sum(final_weights * np.array([m['c'] for m in self.sub_models]))
        avg_phi = np.sum(final_weights * np.array([m['phi'] for m in self.sub_models]))
        avg_nu = np.sum(final_weights * np.array([m['nu'] for m in self.sub_models]))
        
        fit_time_ms = (time.time() - start_time) * 1000
        
        return {
            "final_weights": dict(zip([m['name'] for m in self.sub_models], 
                                      final_weights.tolist())),
            "best_model": best_model['name'],
            "best_model_weight": float(final_weights[best_model_idx]),
            "forgetting_factor": self.forgetting_factor,
            "n_models": self.n_models,
            "log_likelihood": total_ll,
            "bic": bic,
            "aic": aic,
            "n_observations": n,
            "n_params": n_params,
            "success": True,
            "fit_time_ms": fit_time_ms,
            "fit_params": {
                "q": avg_q,
                "c": avg_c,
                "phi": avg_phi,
                "nu": avg_nu,
            },
        }
