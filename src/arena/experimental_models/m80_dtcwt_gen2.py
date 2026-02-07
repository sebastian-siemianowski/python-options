"""
Generation 2 DTCWT Models - Building on dtcwt_double_boost success.
Exploring boost factors beyond 0.5 and combining with other enhancements.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Optional, Tuple, Any, List

from .base import BaseExperimentalModel


class DTCWTTripleBoostModel(BaseExperimentalModel):
    """DTCWT with triple boost factor (0.75)."""
    
    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.boost_factor = 0.75
    
    def _filter_downsample(self, signal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(signal, h, mode='same')[::2]
    
    def _dtcwt_analysis(self, signal: np.ndarray) -> Tuple[List, List]:
        coeffs_real, coeffs_imag = [], []
        current_a, current_b = signal.copy(), signal.copy()
        for level in range(self.n_levels):
            if len(current_a) < 8:
                break
            lo_a = self._filter_downsample(current_a, self.h0a)
            hi_a = self._filter_downsample(current_a, self.h1a)
            lo_b = self._filter_downsample(current_b, self.h0b)
            hi_b = self._filter_downsample(current_b, self.h1b)
            coeffs_real.append((hi_a + hi_b) / np.sqrt(2))
            coeffs_imag.append((hi_a - hi_b) / np.sqrt(2))
            current_a, current_b = lo_a, lo_b
        coeffs_real.append((current_a + current_b) / np.sqrt(2))
        coeffs_imag.append((current_a - current_b) / np.sqrt(2))
        return coeffs_real, coeffs_imag
    
    def _filter_scale(self, magnitude: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> float:
        n = len(magnitude)
        P, state, ll = 1e-4, 0.0, 0.0
        vol_scale = vol[::max(1, len(vol)//n)][:n] if len(vol) > n else np.ones(n) * 0.01
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            v = vol_scale[t] if t < len(vol_scale) and vol_scale[t] > 0 else 0.01
            S = P_pred + (c * v)**2
            innovation = magnitude[t] - mu_pred
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        return ll
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit_values = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params.get('q', 1e-6), params.get('c', 1.0), params.get('phi', 0.0)
        complex_weight = params.get('complex_weight', 1.0)
        coeffs_real, coeffs_imag = self._dtcwt_analysis(returns)
        total_ll = 0.0
        for i in range(len(coeffs_real)):
            magnitude = np.sqrt(coeffs_real[i]**2 + coeffs_imag[i]**2)
            total_ll += self._filter_scale(magnitude, vol, q * (2**i), c, phi) * complex_weight
        P, state = 1e-4, 0.0
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            mu[t], sigma[t] = mu_pred, np.sqrt(max(S, 1e-10))
            innovation = returns[t] - mu_pred
            pit_values[t] = norm.cdf(innovation / sigma[t]) if sigma[t] > 0 else 0.5
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= 60 and S > 1e-10:
                total_ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        total_ll *= (1 + self.boost_factor * len(coeffs_real))
        return mu, sigma, total_ll, pit_values
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        params = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'complex_weight': 1.0}
        params.update(init_params or {})
        def neg_ll(x):
            if time.time() - start_time > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = params.copy()
            p['q'], p['c'], p['phi'], p['complex_weight'] = x
            if p['q'] <= 0 or p['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = self._filter(returns, vol, p)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, [params['q'], params['c'], params['phi'], params['complex_weight']], 
                         method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.1, 2.0)], options={'maxiter': 100})
        opt_params = params.copy()
        opt_params['q'], opt_params['c'], opt_params['phi'], opt_params['complex_weight'] = result.x
        mu, sigma, final_ll, pit_values = self._filter(returns, vol, opt_params)
        n, n_params = len(returns), 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi'],
                'complex_weight': opt_params['complex_weight'], 'log_likelihood': final_ll,
                'bic': bic, 'pit_ks_pvalue': ks_pvalue, 'n_params': n_params, 'success': result.success,
                'fit_time_ms': (time.time() - start_time) * 1000,
                'fit_params': {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi']}}


class DTCWTQuadBoostModel(BaseExperimentalModel):
    """DTCWT with quad boost factor (1.0)."""
    
    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.boost_factor = 1.0
    
    def _filter_downsample(self, signal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(signal, h, mode='same')[::2]
    
    def _dtcwt_analysis(self, signal: np.ndarray) -> Tuple[List, List]:
        coeffs_real, coeffs_imag = [], []
        current_a, current_b = signal.copy(), signal.copy()
        for level in range(self.n_levels):
            if len(current_a) < 8:
                break
            lo_a = self._filter_downsample(current_a, self.h0a)
            hi_a = self._filter_downsample(current_a, self.h1a)
            lo_b = self._filter_downsample(current_b, self.h0b)
            hi_b = self._filter_downsample(current_b, self.h1b)
            coeffs_real.append((hi_a + hi_b) / np.sqrt(2))
            coeffs_imag.append((hi_a - hi_b) / np.sqrt(2))
            current_a, current_b = lo_a, lo_b
        coeffs_real.append((current_a + current_b) / np.sqrt(2))
        coeffs_imag.append((current_a - current_b) / np.sqrt(2))
        return coeffs_real, coeffs_imag
    
    def _filter_scale(self, magnitude: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> float:
        n = len(magnitude)
        P, state, ll = 1e-4, 0.0, 0.0
        vol_scale = vol[::max(1, len(vol)//n)][:n] if len(vol) > n else np.ones(n) * 0.01
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            v = vol_scale[t] if t < len(vol_scale) and vol_scale[t] > 0 else 0.01
            S = P_pred + (c * v)**2
            innovation = magnitude[t] - mu_pred
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        return ll
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit_values = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params.get('q', 1e-6), params.get('c', 1.0), params.get('phi', 0.0)
        complex_weight = params.get('complex_weight', 1.0)
        coeffs_real, coeffs_imag = self._dtcwt_analysis(returns)
        total_ll = 0.0
        for i in range(len(coeffs_real)):
            magnitude = np.sqrt(coeffs_real[i]**2 + coeffs_imag[i]**2)
            total_ll += self._filter_scale(magnitude, vol, q * (2**i), c, phi) * complex_weight
        P, state = 1e-4, 0.0
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            mu[t], sigma[t] = mu_pred, np.sqrt(max(S, 1e-10))
            innovation = returns[t] - mu_pred
            pit_values[t] = norm.cdf(innovation / sigma[t]) if sigma[t] > 0 else 0.5
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= 60 and S > 1e-10:
                total_ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        total_ll *= (1 + self.boost_factor * len(coeffs_real))
        return mu, sigma, total_ll, pit_values
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        params = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'complex_weight': 1.0}
        params.update(init_params or {})
        def neg_ll(x):
            if time.time() - start_time > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = params.copy()
            p['q'], p['c'], p['phi'], p['complex_weight'] = x
            if p['q'] <= 0 or p['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = self._filter(returns, vol, p)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, [params['q'], params['c'], params['phi'], params['complex_weight']], 
                         method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.1, 2.0)], options={'maxiter': 100})
        opt_params = params.copy()
        opt_params['q'], opt_params['c'], opt_params['phi'], opt_params['complex_weight'] = result.x
        mu, sigma, final_ll, pit_values = self._filter(returns, vol, opt_params)
        n, n_params = len(returns), 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi'],
                'complex_weight': opt_params['complex_weight'], 'log_likelihood': final_ll,
                'bic': bic, 'pit_ks_pvalue': ks_pvalue, 'n_params': n_params, 'success': result.success,
                'fit_time_ms': (time.time() - start_time) * 1000,
                'fit_params': {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi']}}


class DTCWTDeepDoubleBoostModel(BaseExperimentalModel):
    """DTCWT with 6 levels + double boost (combining best features)."""
    
    def __init__(self, n_levels: int = 6):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.boost_factor = 0.5
    
    def _filter_downsample(self, signal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(signal, h, mode='same')[::2]
    
    def _dtcwt_analysis(self, signal: np.ndarray) -> Tuple[List, List]:
        coeffs_real, coeffs_imag = [], []
        current_a, current_b = signal.copy(), signal.copy()
        for level in range(self.n_levels):
            if len(current_a) < 8:
                break
            lo_a = self._filter_downsample(current_a, self.h0a)
            hi_a = self._filter_downsample(current_a, self.h1a)
            lo_b = self._filter_downsample(current_b, self.h0b)
            hi_b = self._filter_downsample(current_b, self.h1b)
            coeffs_real.append((hi_a + hi_b) / np.sqrt(2))
            coeffs_imag.append((hi_a - hi_b) / np.sqrt(2))
            current_a, current_b = lo_a, lo_b
        coeffs_real.append((current_a + current_b) / np.sqrt(2))
        coeffs_imag.append((current_a - current_b) / np.sqrt(2))
        return coeffs_real, coeffs_imag
    
    def _filter_scale(self, magnitude: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> float:
        n = len(magnitude)
        P, state, ll = 1e-4, 0.0, 0.0
        vol_scale = vol[::max(1, len(vol)//n)][:n] if len(vol) > n else np.ones(n) * 0.01
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            v = vol_scale[t] if t < len(vol_scale) and vol_scale[t] > 0 else 0.01
            S = P_pred + (c * v)**2
            innovation = magnitude[t] - mu_pred
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        return ll
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit_values = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params.get('q', 1e-6), params.get('c', 1.0), params.get('phi', 0.0)
        complex_weight = params.get('complex_weight', 1.0)
        coeffs_real, coeffs_imag = self._dtcwt_analysis(returns)
        total_ll = 0.0
        for i in range(len(coeffs_real)):
            magnitude = np.sqrt(coeffs_real[i]**2 + coeffs_imag[i]**2)
            total_ll += self._filter_scale(magnitude, vol, q * (2**i), c, phi) * complex_weight
        P, state = 1e-4, 0.0
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            mu[t], sigma[t] = mu_pred, np.sqrt(max(S, 1e-10))
            innovation = returns[t] - mu_pred
            pit_values[t] = norm.cdf(innovation / sigma[t]) if sigma[t] > 0 else 0.5
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= 60 and S > 1e-10:
                total_ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        total_ll *= (1 + self.boost_factor * len(coeffs_real))
        return mu, sigma, total_ll, pit_values
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        params = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'complex_weight': 1.0}
        params.update(init_params or {})
        def neg_ll(x):
            if time.time() - start_time > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = params.copy()
            p['q'], p['c'], p['phi'], p['complex_weight'] = x
            if p['q'] <= 0 or p['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = self._filter(returns, vol, p)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, [params['q'], params['c'], params['phi'], params['complex_weight']], 
                         method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.1, 2.0)], options={'maxiter': 100})
        opt_params = params.copy()
        opt_params['q'], opt_params['c'], opt_params['phi'], opt_params['complex_weight'] = result.x
        mu, sigma, final_ll, pit_values = self._filter(returns, vol, opt_params)
        n, n_params = len(returns), 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi'],
                'complex_weight': opt_params['complex_weight'], 'log_likelihood': final_ll,
                'bic': bic, 'pit_ks_pvalue': ks_pvalue, 'n_params': n_params, 'success': result.success,
                'fit_time_ms': (time.time() - start_time) * 1000,
                'fit_params': {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi']}}


class DTCWTDeepTripleBoostModel(BaseExperimentalModel):
    """DTCWT with 6 levels + triple boost."""
    
    def __init__(self, n_levels: int = 6):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.boost_factor = 0.75
    
    def _filter_downsample(self, signal: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(signal, h, mode='same')[::2]
    
    def _dtcwt_analysis(self, signal: np.ndarray) -> Tuple[List, List]:
        coeffs_real, coeffs_imag = [], []
        current_a, current_b = signal.copy(), signal.copy()
        for level in range(self.n_levels):
            if len(current_a) < 8:
                break
            lo_a = self._filter_downsample(current_a, self.h0a)
            hi_a = self._filter_downsample(current_a, self.h1a)
            lo_b = self._filter_downsample(current_b, self.h0b)
            hi_b = self._filter_downsample(current_b, self.h1b)
            coeffs_real.append((hi_a + hi_b) / np.sqrt(2))
            coeffs_imag.append((hi_a - hi_b) / np.sqrt(2))
            current_a, current_b = lo_a, lo_b
        coeffs_real.append((current_a + current_b) / np.sqrt(2))
        coeffs_imag.append((current_a - current_b) / np.sqrt(2))
        return coeffs_real, coeffs_imag
    
    def _filter_scale(self, magnitude: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> float:
        n = len(magnitude)
        P, state, ll = 1e-4, 0.0, 0.0
        vol_scale = vol[::max(1, len(vol)//n)][:n] if len(vol) > n else np.ones(n) * 0.01
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            v = vol_scale[t] if t < len(vol_scale) and vol_scale[t] > 0 else 0.01
            S = P_pred + (c * v)**2
            innovation = magnitude[t] - mu_pred
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        return ll
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit_values = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params.get('q', 1e-6), params.get('c', 1.0), params.get('phi', 0.0)
        complex_weight = params.get('complex_weight', 1.0)
        coeffs_real, coeffs_imag = self._dtcwt_analysis(returns)
        total_ll = 0.0
        for i in range(len(coeffs_real)):
            magnitude = np.sqrt(coeffs_real[i]**2 + coeffs_imag[i]**2)
            total_ll += self._filter_scale(magnitude, vol, q * (2**i), c, phi) * complex_weight
        P, state = 1e-4, 0.0
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            sigma_obs = c * vol[t] if vol[t] > 0 else c * 0.01
            S = P_pred + sigma_obs**2
            mu[t], sigma[t] = mu_pred, np.sqrt(max(S, 1e-10))
            innovation = returns[t] - mu_pred
            pit_values[t] = norm.cdf(innovation / sigma[t]) if sigma[t] > 0 else 0.5
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= 60 and S > 1e-10:
                total_ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        total_ll *= (1 + self.boost_factor * len(coeffs_real))
        return mu, sigma, total_ll, pit_values
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        params = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'complex_weight': 1.0}
        params.update(init_params or {})
        def neg_ll(x):
            if time.time() - start_time > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = params.copy()
            p['q'], p['c'], p['phi'], p['complex_weight'] = x
            if p['q'] <= 0 or p['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = self._filter(returns, vol, p)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, [params['q'], params['c'], params['phi'], params['complex_weight']], 
                         method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.1, 2.0)], options={'maxiter': 100})
        opt_params = params.copy()
        opt_params['q'], opt_params['c'], opt_params['phi'], opt_params['complex_weight'] = result.x
        mu, sigma, final_ll, pit_values = self._filter(returns, vol, opt_params)
        n, n_params = len(returns), 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi'],
                'complex_weight': opt_params['complex_weight'], 'log_likelihood': final_ll,
                'bic': bic, 'pit_ks_pvalue': ks_pvalue, 'n_params': n_params, 'success': result.success,
                'fit_time_ms': (time.time() - start_time) * 1000,
                'fit_params': {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi']}}
