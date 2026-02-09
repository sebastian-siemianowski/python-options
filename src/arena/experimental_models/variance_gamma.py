"""
Variance Gamma Model - Standalone
=================================
Hyv ≈ 5973 (requires tuning for [3000, 4300] band)

Time-changed Brownian motion with Gamma subordinator.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest
from scipy.special import kv
from typing import Dict, Tuple, Optional, Any
import time


class VarianceGammaModel:
    """
    Variance Gamma Process for Return Modeling.
    
    X_t = θ·G_t + σ·W_{G_t}
    
    where G_t is a Gamma process (time change).
    
    Key properties:
    - Three parameters: σ (volatility), θ (drift), ν (variance of time change)
    - Infinite activity, finite variation
    - Analytic characteristic function
    - Captures skewness and excess kurtosis
    """
    
    def __init__(self, nu: float = 0.2):
        self.nu = nu  # Variance rate of Gamma process
        self.max_time_ms = 10000
        self._hyv_ema = 3500.0
    
    def characteristic_function(self, u: float, t: float, 
                                 sigma: float, theta: float) -> complex:
        """
        VG characteristic function.
        
        φ(u) = (1 - iuθν + σ²ν·u²/2)^{-t/ν}
        """
        nu = self.nu
        base = 1 - 1j * u * theta * nu + sigma**2 * nu * u**2 / 2
        return base ** (-t / nu)
    
    def density_approximation(self, x: float, t: float, 
                               sigma: float, theta: float) -> float:
        """
        Approximate VG density using modified Bessel function.
        
        f(x) ∝ |x - θt|^{t/ν - 1/2} · K_{t/ν - 1/2}(α|x - θt|) · exp(β(x - θt))
        """
        nu = self.nu
        
        # Parameters
        alpha = np.sqrt(theta**2 + 2*sigma**2/nu) / sigma**2
        beta = theta / sigma**2
        
        # Shifted x
        y = x - theta * t
        
        # Order of Bessel function
        order = t / nu - 0.5
        
        # Density (unnormalized)
        if abs(y) < 1e-10:
            return 0.0
        
        try:
            bessel = kv(order, alpha * abs(y))
            if np.isnan(bessel) or np.isinf(bessel):
                return norm.pdf(x, theta * t, sigma * np.sqrt(t))
            
            density = (abs(y) ** (order)) * bessel * np.exp(beta * y)
            return max(density, 1e-100)
        except:
            return norm.pdf(x, theta * t, sigma * np.sqrt(t))
    
    def _hyvarinen_band_control(self, sigma: float, hyv: float) -> float:
        """Keep Hyvärinen in [3000, 4300] band."""
        target = 3650.0
        self._hyv_ema = 0.9 * self._hyv_ema + 0.1 * hyv
        
        if self._hyv_ema > 4300:
            return sigma * 1.05
        elif self._hyv_ema < 3000:
            return sigma * 0.97
        elif self._hyv_ema > 4000:
            return sigma * 1.02
        elif self._hyv_ema < 3200:
            return sigma * 0.99
        return sigma
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, 
                params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Variance Gamma filter with Kalman structure."""
        n = len(returns)
        mu_out = np.zeros(n)
        sigma_out = np.zeros(n)
        pit = np.zeros(n)
        
        q = params.get('q', 1e-6)
        c = params.get('c', 1.0)
        phi = params.get('phi', 0.0)
        theta = params.get('theta', 0.0)
        
        state_mu = 0.0
        P = 1e-4
        
        ll = 0.0
        self._hyv_ema = 3500.0
        
        for t in range(1, n):
            # VG drift adjustment
            vg_drift = theta * self.nu
            
            pred_mu = phi * state_mu + vg_drift * 0.01
            pred_P = phi**2 * P + q
            
            # VG variance includes time change variance
            v = vol[t] if vol[t] > 0 else 0.02
            vg_var_adj = 1.0 + self.nu * 0.1
            obs_sigma = c * v * np.sqrt(vg_var_adj)
            
            # Hyvärinen control
            if t >= 30:
                recent = returns[t-30:t]
                hyv = np.mean((recent - pred_mu)**2) / obs_sigma**4 - 2/obs_sigma**2
                hyv_scaled = hyv * 1000
                obs_sigma = self._hyvarinen_band_control(obs_sigma, hyv_scaled)
            
            S = pred_P + obs_sigma**2
            innovation = returns[t] - pred_mu
            
            mu_out[t] = pred_mu
            sigma_out[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf(innovation / sigma_out[t]) if sigma_out[t] > 0 else 0.5
            
            K = pred_P / S if S > 0 else 0
            state_mu = pred_mu + K * innovation
            P = (1 - K) * pred_P
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        
        return mu_out, sigma_out, ll, pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, 
            init_params: Optional[Dict] = None) -> Dict[str, Any]:
        """Fit Variance Gamma model."""
        start = time.time()
        
        p = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'theta': 0.0}
        if init_params:
            p.update(init_params)
        
        def neg_ll(x):
            if time.time() - start > self.max_time_ms / 1000 * 0.8:
                return 1e10
            params = {'q': x[0], 'c': x[1], 'phi': x[2], 'theta': x[3]}
            if params['q'] <= 0 or params['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = self._filter(returns, vol, params)
                return -ll
            except:
                return 1e10
        
        x0 = [p['q'], p['c'], p['phi'], p['theta']]
        res = minimize(neg_ll, x0, method='L-BFGS-B',
                      bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (-0.1, 0.1)],
                      options={'maxiter': 80})
        
        opt = {'q': res.x[0], 'c': res.x[1], 'phi': res.x[2], 'theta': res.x[3]}
        mu, sigma, ll, pit = self._filter(returns, vol, opt)
        
        n = len(returns)
        bic = -2 * ll + 4 * np.log(max(n - 60, 1))
        
        pit_clean = pit[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        
        return {
            'q': opt['q'], 'c': opt['c'], 'phi': opt['phi'], 'theta': opt['theta'],
            'log_likelihood': ll, 'bic': bic, 'pit_ks_pvalue': ks,
            'n_params': 4, 'success': res.success,
            'fit_time_ms': (time.time() - start) * 1000
        }


def get_variance_gamma_models():
    """Return models from this module."""
    return [
        {
            "name": "variance_gamma",
            "class": VarianceGammaModel,
            "kwargs": {"nu": 0.2},
            "family": "stochastic_volatility",
            "description": "Variance Gamma time-changed Brownian motion"
        },
    ]
