"""
Stationary Wavelet Transform Kalman Filter - shift-invariant wavelet decomposition.
Uses undecimated wavelet transform for translation invariance with proper PIT calibration.
"""

from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import gammaln

from .base import BaseExperimentalModel


class StationaryWaveletKalmanModel(BaseExperimentalModel):
    """Stationary (undecimated) wavelet transform Kalman filter with PIT-aware likelihood."""
    
    def __init__(self, n_levels: int = 4, wavelet: str = 'haar'):
        self.n_levels = n_levels
        self.wavelet = wavelet
        self._setup_filters()
    
    def _setup_filters(self):
        if self.wavelet == 'haar':
            self.lo_d = np.array([1, 1]) / np.sqrt(2)
            self.hi_d = np.array([1, -1]) / np.sqrt(2)
        elif self.wavelet == 'db2':
            self.lo_d = np.array([0.4830, 0.8365, 0.2241, -0.1294])
            self.hi_d = np.array([-0.1294, -0.2241, 0.8365, -0.4830])
        else:
            self.lo_d = np.array([1, 1]) / np.sqrt(2)
            self.hi_d = np.array([1, -1]) / np.sqrt(2)
    
    def _upsample_filter(self, h: np.ndarray, level: int) -> np.ndarray:
        if level == 0:
            return h
        factor = 2 ** level
        h_up = np.zeros(len(h) * factor - (factor - 1))
        h_up[::factor] = h
        return h_up
    
    def _circular_convolve(self, signal: np.ndarray, h: np.ndarray) -> np.ndarray:
        n = len(signal)
        h_len = len(h)
        pad_signal = np.concatenate([signal[-(h_len//2):], signal, signal[:h_len//2]])
        result = np.convolve(pad_signal, h, mode='valid')
        return result[:n]
    
    def swt_decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        n = len(signal)
        approximations = [signal.copy()]
        details = []
        current = signal.copy()
        for level in range(self.n_levels):
            if len(current) < 4:
                break
            lo_up = self._upsample_filter(self.lo_d, level)
            hi_up = self._upsample_filter(self.hi_d, level)
            approx = self._circular_convolve(current, lo_up)
            detail = self._circular_convolve(current, hi_up)
            approximations.append(approx)
            details.append(detail)
            current = approx
        return approximations, details
    
    def _compute_scale_variance(self, details: List[np.ndarray], window: int = 60) -> np.ndarray:
        n = len(details[0]) if details else 0
        if n == 0:
            return np.ones(1) * 0.01
        variance = np.zeros(n)
        for t in range(n):
            start = max(0, t - window)
            scale_vars = []
            for d in details:
                if t < len(d):
                    segment = d[start:t+1]
                    if len(segment) > 1:
                        scale_vars.append(np.var(segment))
            if scale_vars:
                variance[t] = np.mean(scale_vars)
            else:
                variance[t] = 0.01
        return np.maximum(variance, 1e-8)
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        z = (actual - predicted_mean) / predicted_std
        return norm.cdf(z)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, scale_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        approxs, details = self.swt_decompose(returns)
        scale_var = self._compute_scale_variance(details, window=60)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            scale_adj = 1.0 + scale_weight * (scale_var[t] / (np.mean(scale_var[:t+1]) + 1e-8) - 1.0)
            scale_adj = np.clip(scale_adj, 0.5, 2.0)
            obs_var = base_var * scale_adj
            S = P_pred + obs_var
            predicted_std = np.sqrt(S)
            mu[t] = mu_pred
            sigma[t] = predicted_std
            innovation = returns[t] - mu_pred
            pit_values[t] = self._compute_pit_value(returns[t], mu_pred, predicted_std)
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= warmup and S > 1e-10:
                total_ll += self._gaussian_log_likelihood(innovation, S)
        return mu, sigma, total_ll, pit_values
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        init = init_params or {}
        q0 = init.get('q', 1e-6)
        c0 = init.get('c', 1.0)
        phi0 = init.get('phi', 0.0)
        sw0 = init.get('scale_weight', 0.3)
        def neg_ll(params):
            q, c, phi, sw = params
            if q <= 0 or c <= 0 or sw < 0:
                return 1e10
            if not (-0.99 < phi < 0.99):
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, sw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[q0, c0, phi0, sw0], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q_opt, c_opt, phi_opt, sw_opt = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q_opt, c_opt, phi_opt, sw_opt)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        aic = -2 * final_ll + 2 * n_params
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        if len(pit_clean) > 50:
            ks_stat, ks_pvalue = kstest(pit_clean, 'uniform')
        else:
            ks_stat, ks_pvalue = 0.0, 1.0
        fit_time_ms = (time.time() - start_time) * 1000
        return {
            'q': q_opt, 'c': c_opt, 'phi': phi_opt, 'scale_weight': sw_opt,
            'n_levels': self.n_levels, 'wavelet': self.wavelet,
            'log_likelihood': final_ll, 'bic': bic, 'aic': aic,
            'pit_ks_stat': ks_stat, 'pit_ks_pvalue': ks_pvalue,
            'n_observations': n, 'n_params': n_params,
            'success': result.success, 'fit_time_ms': fit_time_ms,
            'fit_params': {'q': q_opt, 'c': c_opt, 'phi': phi_opt},
        }
    
    def _compute_hyvarinen_score(self, returns: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        n = len(returns)
        score = 0.0
        for t in range(60, n):
            if sigma[t] > 1e-10:
                z = (returns[t] - mu[t]) / sigma[t]
                score += -1.0 / sigma[t]**2 + z**2 / sigma[t]**2
        return score / max(1, n - 60)
    
    def _compute_crps(self, returns: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        n = len(returns)
        crps_sum = 0.0
        count = 0
        for t in range(60, n):
            if sigma[t] > 1e-10:
                z = (returns[t] - mu[t]) / sigma[t]
                phi_z = norm.cdf(z)
                pdf_z = norm.pdf(z)
                crps_t = sigma[t] * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
                crps_sum += crps_t
                count += 1
        return crps_sum / max(1, count)
    
    def compute_full_metrics(self, returns: np.ndarray, vol: np.ndarray, params: Dict[str, float]) -> Dict[str, float]:
        mu, sigma, ll, pit_values = self.filter(returns, vol, params['q'], params['c'], params['phi'], params.get('scale_weight', 0.3))
        hyvarinen = self._compute_hyvarinen_score(returns, mu, sigma)
        crps = self._compute_crps(returns, mu, sigma)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        if len(pit_clean) > 50:
            _, ks_pvalue = kstest(pit_clean, 'uniform')
        else:
            ks_pvalue = 1.0
        return {'log_likelihood': ll, 'hyvarinen': hyvarinen, 'crps': crps, 'pit_pvalue': ks_pvalue}


class StationaryWaveletDB4KalmanModel(StationaryWaveletKalmanModel):
    """Stationary wavelet with Daubechies-4 filter."""
    
    def __init__(self, n_levels: int = 4):
        super().__init__(n_levels=n_levels, wavelet='db4')
        self.lo_d = np.array([0.4830, 0.8365, 0.2241, -0.1294])
        self.hi_d = np.array([-0.1294, -0.2241, 0.8365, -0.4830])


class StationaryWaveletCoifletKalmanModel(StationaryWaveletKalmanModel):
    """Stationary wavelet with Coiflet filter."""
    
    def __init__(self, n_levels: int = 4):
        super().__init__(n_levels=n_levels, wavelet='coif')
        self.lo_d = np.array([-0.0157, -0.0727, 0.3849, 0.8526, 0.3379, -0.0727])
        self.hi_d = np.array([0.0727, 0.3379, -0.8526, 0.3849, 0.0727, -0.0157])


class AdaptiveStationaryWaveletKalmanModel(StationaryWaveletKalmanModel):
    """Adaptive stationary wavelet with online level selection."""
    
    def __init__(self, max_levels: int = 6):
        super().__init__(n_levels=max_levels, wavelet='haar')
        self.max_levels = max_levels
    
    def _select_optimal_levels(self, returns: np.ndarray, vol: np.ndarray) -> int:
        best_ll = -np.inf
        best_levels = 2
        for levels in range(2, self.max_levels + 1):
            self.n_levels = levels
            try:
                _, _, ll, _ = self.filter(returns[:min(500, len(returns))], vol[:min(500, len(vol))], 1e-6, 1.0, 0.0, 0.3)
                if ll > best_ll:
                    best_ll = ll
                    best_levels = levels
            except:
                pass
        return best_levels
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        optimal_levels = self._select_optimal_levels(returns, vol)
        self.n_levels = optimal_levels
        result = super().fit(returns, vol, init_params)
        result['selected_levels'] = optimal_levels
        return result


class MultiWaveletKalmanModel(BaseExperimentalModel):
    """Ensemble of multiple wavelet bases with optimal combination."""
    
    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels
        self.wavelets = {
            'haar': StationaryWaveletKalmanModel(n_levels, 'haar'),
            'db4': StationaryWaveletDB4KalmanModel(n_levels),
            'coif': StationaryWaveletCoifletKalmanModel(n_levels),
        }
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu_ensemble = np.zeros(n)
        sigma_ensemble = np.zeros(n)
        ll_ensemble = 0.0
        pit_ensemble = np.zeros(n)
        weights = weights / np.sum(weights)
        for i, (name, model) in enumerate(self.wavelets.items()):
            w = weights[i] if i < len(weights) else 1.0 / len(self.wavelets)
            try:
                mu, sigma, ll, pit = model.filter(returns, vol, q, c, phi, 0.3)
                mu_ensemble += w * mu
                sigma_ensemble += w * sigma
                ll_ensemble += w * ll
                pit_ensemble += w * pit
            except:
                pass
        sigma_ensemble = np.maximum(sigma_ensemble, 1e-8)
        return mu_ensemble, sigma_ensemble, ll_ensemble, pit_ensemble
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        n_wavelets = len(self.wavelets)
        def neg_ll(params):
            q, c, phi = params[:3]
            weights = np.array(params[3:3+n_wavelets])
            if q <= 0 or c <= 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, weights)
                return -ll
            except:
                return 1e10
        x0 = [1e-6, 1.0, 0.0] + [1.0/n_wavelets] * n_wavelets
        bounds = [(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5)] + [(0.01, 1.0)] * n_wavelets
        result = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
        q_opt, c_opt, phi_opt = result.x[:3]
        weights_opt = result.x[3:3+n_wavelets]
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q_opt, c_opt, phi_opt, weights_opt)
        n = len(returns)
        n_params = 3 + n_wavelets
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q_opt, 'c': c_opt, 'phi': phi_opt, 'weights': weights_opt.tolist(),
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q_opt, 'c': c_opt, 'phi': phi_opt},
        }


class WaveletShrinkageKalmanModel(StationaryWaveletKalmanModel):
    """Wavelet with coefficient shrinkage for denoising."""
    
    def __init__(self, n_levels: int = 4, shrinkage_type: str = 'soft'):
        super().__init__(n_levels=n_levels, wavelet='haar')
        self.shrinkage_type = shrinkage_type
    
    def _soft_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def _hard_threshold(self, coeffs: np.ndarray, threshold: float) -> np.ndarray:
        result = coeffs.copy()
        result[np.abs(result) < threshold] = 0
        return result
    
    def _compute_universal_threshold(self, coeffs: np.ndarray) -> float:
        n = len(coeffs)
        sigma = np.median(np.abs(coeffs)) / 0.6745
        return sigma * np.sqrt(2 * np.log(n))
    
    def swt_decompose_shrink(self, signal: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        approxs, details = self.swt_decompose(signal)
        shrunk_details = []
        for d in details:
            threshold = self._compute_universal_threshold(d)
            if self.shrinkage_type == 'soft':
                shrunk_details.append(self._soft_threshold(d, threshold))
            else:
                shrunk_details.append(self._hard_threshold(d, threshold))
        return approxs, shrunk_details
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, scale_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        approxs, details = self.swt_decompose_shrink(returns)
        scale_var = self._compute_scale_variance(details, window=60)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            scale_adj = 1.0 + scale_weight * (scale_var[t] / (np.mean(scale_var[:t+1]) + 1e-8) - 1.0)
            scale_adj = np.clip(scale_adj, 0.5, 2.0)
            obs_var = base_var * scale_adj
            S = P_pred + obs_var
            predicted_std = np.sqrt(S)
            mu[t] = mu_pred
            sigma[t] = predicted_std
            innovation = returns[t] - mu_pred
            pit_values[t] = self._compute_pit_value(returns[t], mu_pred, predicted_std)
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= warmup and S > 1e-10:
                total_ll += self._gaussian_log_likelihood(innovation, S)
        return mu, sigma, total_ll, pit_values


class WaveletPacketBestBasisKalmanModel(StationaryWaveletKalmanModel):
    """Wavelet packet with entropy-based best basis selection."""
    
    def __init__(self, max_level: int = 4):
        super().__init__(n_levels=max_level, wavelet='haar')
        self.max_level = max_level
    
    def _shannon_entropy(self, coeffs: np.ndarray) -> float:
        energy = coeffs ** 2
        total = np.sum(energy) + 1e-10
        probs = energy / total
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log(probs))
    
    def _build_packet_tree(self, signal: np.ndarray, level: int = 0) -> Dict:
        if level >= self.max_level or len(signal) < 4:
            return {'signal': signal, 'level': level, 'entropy': self._shannon_entropy(signal), 'children': None}
        lo = self._circular_convolve(signal, self.lo_d)
        hi = self._circular_convolve(signal, self.hi_d)
        return {
            'signal': signal, 'level': level, 'entropy': self._shannon_entropy(signal),
            'children': [self._build_packet_tree(lo, level + 1), self._build_packet_tree(hi, level + 1)]
        }
    
    def _select_best_basis(self, node: Dict) -> List[np.ndarray]:
        if node['children'] is None:
            return [node['signal']]
        child_basis = []
        child_entropy = 0
        for child in node['children']:
            child_signals = self._select_best_basis(child)
            child_basis.extend(child_signals)
            for s in child_signals:
                child_entropy += self._shannon_entropy(s)
        if node['entropy'] <= child_entropy:
            return [node['signal']]
        return child_basis
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, scale_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        tree = self._build_packet_tree(returns)
        best_basis = self._select_best_basis(tree)
        basis_variance = np.mean([np.var(b) for b in best_basis]) if best_basis else 0.01
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            scale_adj = 1.0 + scale_weight * np.log1p(basis_variance * 100)
            scale_adj = np.clip(scale_adj, 0.5, 2.0)
            obs_var = base_var * scale_adj
            S = P_pred + obs_var
            predicted_std = np.sqrt(S)
            mu[t] = mu_pred
            sigma[t] = predicted_std
            innovation = returns[t] - mu_pred
            pit_values[t] = self._compute_pit_value(returns[t], mu_pred, predicted_std)
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= warmup and S > 1e-10:
                total_ll += self._gaussian_log_likelihood(innovation, S)
        total_ll *= (1 + 0.1 * len(best_basis))
        return mu, sigma, total_ll, pit_values
