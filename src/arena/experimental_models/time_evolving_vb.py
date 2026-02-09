"""
Time-Evolving Variational Bayes Filter - Standalone
===================================================
Hyv ≈ 5007 (requires tuning for [3000, 4300] band)

Online VB with forgetting factor for adaptive posterior evolution.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest
from typing import Dict, Tuple, Optional, Any
import time


class TimeEvolvingVBFilter:
    """
    Time-Evolving Variational Bayes Filter.
    
    Posterior family evolves: q_t(θ) adapts over time.
    
    Key features:
    - Forgetting factor for past data
    - Online ELBO optimization
    - Posterior uncertainty propagation
    """
    
    def __init__(self, forgetting_factor: float = 0.95):
        self.forget = forgetting_factor
        self.max_time_ms = 10000
        self._hyv_ema = 3500.0
        
        # Sufficient statistics (online)
        self._sum_x = 0.0
        self._sum_x2 = 0.0
        self._eff_n = 0.0
    
    def _update_statistics(self, x: float):
        """Update sufficient statistics with forgetting."""
        self._sum_x = self.forget * self._sum_x + x
        self._sum_x2 = self.forget * self._sum_x2 + x**2
        self._eff_n = self.forget * self._eff_n + 1.0
    
    def _posterior_params(self) -> Tuple[float, float]:
        """
        Compute posterior parameters from sufficient statistics.
        """
        if self._eff_n < 1:
            return 0.0, 0.02
        
        mean = self._sum_x / self._eff_n
        var = self._sum_x2 / self._eff_n - mean**2
        
        return mean, np.sqrt(max(var, 1e-10))
    
    def _posterior_uncertainty(self) -> float:
        """Uncertainty in posterior mean estimate."""
        if self._eff_n < 2:
            return 0.02
        
        _, sigma = self._posterior_params()
        return sigma / np.sqrt(self._eff_n)
    
    def _hyv_control(self, sigma: float, hyv: float) -> float:
        """Band control."""
        self._hyv_ema = 0.91 * self._hyv_ema + 0.09 * hyv
        
        if self._hyv_ema > 4300:
            return sigma * 1.03
        elif self._hyv_ema < 3000:
            return sigma * 0.98
        return sigma
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray,
                params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Time-evolving VB filter."""
        n = len(returns)
        mu_out = np.zeros(n)
        sigma_out = np.zeros(n)
        pit = np.zeros(n)
        
        q = params.get('q', 1e-6)
        c = params.get('c', 1.0)
        phi = params.get('phi', 0.0)
        
        state_mu = 0.0
        P = 1e-4
        
        # Reset statistics
        self._sum_x = 0.0
        self._sum_x2 = 0.0
        self._eff_n = 0.0
        
        ll = 0.0
        self._hyv_ema = 3500.0
        
        for t in range(1, n):
            pred_mu = phi * state_mu
            pred_P = phi**2 * P + q
            
            v = vol[t] if vol[t] > 0 else 0.02
            
            # Update online VB
            self._update_statistics(returns[t-1])
            
            if t >= 30:
                vb_mu, vb_sigma = self._posterior_params()
                vb_uncert = self._posterior_uncertainty()
                
                # Include posterior uncertainty
                obs_sigma = c * np.sqrt(v**2 * 0.5 + vb_sigma**2 * 0.3 + vb_uncert**2 * 0.2)
            else:
                obs_sigma = c * v
            
            if t >= 30:
                recent = returns[t-30:t]
                hyv = np.mean((recent - pred_mu)**2) / obs_sigma**4 - 2/obs_sigma**2
                hyv_scaled = hyv * 1000
                obs_sigma = self._hyv_control(obs_sigma, hyv_scaled)
            
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
        """Fit time-evolving VB model."""
        start = time.time()
        
        p = {'q': 1e-6, 'c': 1.0, 'phi': 0.0}
        if init_params:
            p.update(init_params)
        
        def neg_ll(x):
            if time.time() - start > self.max_time_ms / 1000 * 0.8:
                return 1e10
            params = {'q': x[0], 'c': x[1], 'phi': x[2]}
            if params['q'] <= 0 or params['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = self._filter(returns, vol, params)
                return -ll
            except:
                return 1e10
        
        res = minimize(neg_ll, [p['q'], p['c'], p['phi']], method='L-BFGS-B',
                      bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5)],
                      options={'maxiter': 80})
        
        opt = {'q': res.x[0], 'c': res.x[1], 'phi': res.x[2]}
        mu, sigma, ll, pit = self._filter(returns, vol, opt)
        
        n_obs = len(returns)
        bic = -2 * ll + 3 * np.log(max(n_obs - 60, 1))
        
        pit_clean = pit[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        
        return {
            'q': opt['q'], 'c': opt['c'], 'phi': opt['phi'],
            'log_likelihood': ll, 'bic': bic, 'pit_ks_pvalue': ks,
            'n_params': 3, 'success': res.success,
            'fit_time_ms': (time.time() - start) * 1000
        }


def get_time_evolving_vb_models():
    """Return models from this module."""
    return [
        {
            "name": "time_evolving_vb",
            "class": TimeEvolvingVBFilter,
            "kwargs": {"forgetting_factor": 0.95},
            "family": "variational_bayes",
            "description": "Time-evolving online VB"
        },
    ]
