"""
===============================================================================
MODEL 6: MULTI-SCALE WAVELET KALMAN
===============================================================================
Professor Zhang Yifan (95/100)

Decomposes returns into multiple frequency bands using wavelets,
filters each band separately with optimized parameters, then recombines.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from scipy.optimize import minimize

from .base import BaseExperimentalModel


class WaveletKalmanModel(BaseExperimentalModel):
    """Multi-scale wavelet decomposition with Kalman filtering."""
    
    def __init__(self, n_scales: int = 3):
        self.n_scales = n_scales
    
    def haar_decompose(self, signal: np.ndarray) -> list:
        """Simple Haar wavelet decomposition."""
        approximations = [signal]
        details = []
        
        current = signal.copy()
        for _ in range(self.n_scales):
            n = len(current)
            if n < 2:
                break
            n_half = n // 2
            approx = np.zeros(n_half)
            detail = np.zeros(n_half)
            
            for i in range(n_half):
                approx[i] = (current[2*i] + current[2*i + 1]) / np.sqrt(2)
                detail[i] = (current[2*i] - current[2*i + 1]) / np.sqrt(2)
            
            approximations.append(approx)
            details.append(detail)
            current = approx
        
        return approximations, details
    
    def filter_scale(
        self,
        signal: np.ndarray,
        vol: np.ndarray,
        q: float, c: float, phi: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Filter a single scale."""
        n = len(signal)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        P = 1e-4
        state = 0.0
        log_likelihood = 0.0
        
        # Downsample vol to match signal length
        if len(vol) > n:
            vol_scale = vol[::len(vol)//n][:n]
        else:
            vol_scale = vol
        
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            v = vol_scale[t] if t < len(vol_scale) and vol_scale[t] > 0 else 0.01
            sigma_obs = c * v
            S = P_pred + sigma_obs**2
            
            mu[t] = mu_pred
            sigma[t] = np.sqrt(S)
            
            if t < n:
                innovation = signal[t] - mu_pred
                K = P_pred / S if S > 0 else 0
                state = mu_pred + K * innovation
                P = (1 - K) * P_pred
                
                if S > 1e-10:
                    ll_t = -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
                    log_likelihood += ll_t
        
        return mu, sigma, log_likelihood
    
    def filter(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        params_list: list,  # [(q, c, phi) for each scale]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        n = len(returns)
        
        # Decompose
        approxs, details = self.haar_decompose(returns)
        
        # Filter each scale
        total_ll = 0.0
        filtered_approxs = []
        filtered_details = []
        
        for i, approx in enumerate(approxs[1:], 1):
            if i-1 < len(params_list):
                q, c, phi = params_list[i-1]
            else:
                q, c, phi = 1e-6, 1.0, 0.0
            mu, _, ll = self.filter_scale(approx, vol, q, c, phi)
            filtered_approxs.append(mu)
            total_ll += ll
        
        for i, detail in enumerate(details):
            if i < len(params_list):
                q, c, phi = params_list[i]
            else:
                q, c, phi = 1e-6, 1.0, 0.0
            mu, _, ll = self.filter_scale(detail, vol, q, c, 0.0)  # phi=0 for details
            filtered_details.append(mu)
            total_ll += ll
        
        # Reconstruct (simplified - just use original scale prediction)
        mu_final = np.zeros(n)
        sigma_final = np.ones(n) * 0.01
        
        # Use filtered approximation at finest scale
        if filtered_approxs:
            finest = filtered_approxs[0]
            # Upsample
            for i in range(min(len(finest)*2, n)):
                mu_final[i] = finest[i//2] if i//2 < len(finest) else 0
        
        # Add detail contributions
        for detail in filtered_details:
            for i in range(min(len(detail)*2, n)):
                if i//2 < len(detail):
                    mu_final[i] += detail[i//2] * 0.1
        
        # Recompute sigma from original scale
        _, sigma_final, _ = self.filter_scale(returns, vol, params_list[0][0], params_list[0][1], params_list[0][2])
        
        return mu_final, sigma_final, total_ll
    
    def fit(
        self,
        returns: np.ndarray,
        vol: np.ndarray,
        init_params: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        import time
        start_time = time.time()
        
        # Optimize parameters for each scale
        def neg_ll(params):
            params_list = []
            for i in range(self.n_scales):
                q = params[i*3]
                c = params[i*3 + 1]
                phi = params[i*3 + 2]
                if q <= 0 or c <= 0 or not (-0.99 < phi < 0.99):
                    return 1e10
                params_list.append((q, c, phi))
            try:
                _, _, ll = self.filter(returns, vol, params_list)
                return -ll
            except:
                return 1e10
        
        x0 = []
        bounds = []
        for _ in range(self.n_scales):
            x0.extend([1e-6, 1.0, 0.0])
            bounds.extend([(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5)])
        
        result = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
        
        params_list = []
        for i in range(self.n_scales):
            params_list.append((result.x[i*3], result.x[i*3+1], result.x[i*3+2]))
        
        _, _, final_ll = self.filter(returns, vol, params_list)
        
        n = len(returns)
        n_params = 3 * self.n_scales
        bic = -2 * final_ll + n_params * np.log(n - 60)
        
        return {
            "scale_params": params_list,
            "log_likelihood": final_ll, "bic": bic,
            "aic": -2 * final_ll + 2 * n_params,
            "n_observations": n, "n_params": n_params,
            "success": result.success,
            "fit_time_ms": (time.time() - start_time) * 1000,
            "fit_params": {"q": params_list[0][0], "c": params_list[0][1], "phi": params_list[0][2]},
        }
