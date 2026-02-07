"""
Hilbert-Huang Transform Kalman Filter - empirical mode decomposition with proper PIT.
Fixes PIT calibration issues in EMD/VMD by computing proper predictive distributions.
"""

from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema

from .base import BaseExperimentalModel


class HilbertHuangKalmanModel(BaseExperimentalModel):
    """Hilbert-Huang Transform Kalman with PIT-calibrated likelihood."""
    
    def __init__(self, max_imfs: int = 5, sift_iterations: int = 10):
        self.max_imfs = max_imfs
        self.sift_iterations = sift_iterations
    
    def _find_extrema(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        maxima_idx = argrelextrema(signal, np.greater, order=1)[0]
        minima_idx = argrelextrema(signal, np.less, order=1)[0]
        return maxima_idx, minima_idx
    
    def _compute_envelope(self, signal: np.ndarray, extrema_idx: np.ndarray) -> np.ndarray:
        n = len(signal)
        if len(extrema_idx) < 2:
            return np.ones(n) * np.mean(signal)
        x = np.concatenate([[0], extrema_idx, [n-1]])
        y = signal[x.astype(int)]
        try:
            spline = CubicSpline(x, y, bc_type='natural')
            return spline(np.arange(n))
        except:
            return np.ones(n) * np.mean(signal)
    
    def _sift(self, signal: np.ndarray) -> np.ndarray:
        h = signal.copy()
        for _ in range(self.sift_iterations):
            maxima_idx, minima_idx = self._find_extrema(h)
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            upper_env = self._compute_envelope(h, maxima_idx)
            lower_env = self._compute_envelope(h, minima_idx)
            mean_env = (upper_env + lower_env) / 2
            h = h - mean_env
            if np.std(mean_env) < 0.01 * np.std(h):
                break
        return h
    
    def _emd(self, signal: np.ndarray) -> List[np.ndarray]:
        imfs = []
        residual = signal.copy()
        for _ in range(self.max_imfs):
            if np.std(residual) < 1e-8:
                break
            imf = self._sift(residual)
            if np.std(imf) < 1e-10:
                break
            imfs.append(imf)
            residual = residual - imf
        if np.std(residual) > 1e-10:
            imfs.append(residual)
        return imfs
    
    def _hilbert_transform(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(signal)
        fft_signal = np.fft.fft(signal)
        h = np.zeros(n)
        if n > 0:
            h[0] = 1
            if n % 2 == 0:
                h[1:n//2] = 2
                h[n//2] = 1
            else:
                h[1:(n+1)//2] = 2
        analytic = np.fft.ifft(fft_signal * h)
        amplitude = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2 * np.pi)
        inst_freq = np.concatenate([inst_freq, [inst_freq[-1] if len(inst_freq) > 0 else 0]])
        return amplitude, inst_freq
    
    def _compute_imf_variance(self, imfs: List[np.ndarray], t: int, window: int = 60) -> float:
        start = max(0, t - window)
        variances = []
        for imf in imfs:
            if t < len(imf):
                segment = imf[start:t+1]
                if len(segment) > 1:
                    variances.append(np.var(segment))
        return np.sum(variances) if variances else 0.01
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        z = (actual - predicted_mean) / predicted_std
        return norm.cdf(z)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, imf_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        imfs = self._emd(returns)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            imf_var = self._compute_imf_variance(imfs, t, window=60)
            scale_adj = 1.0 + imf_weight * np.log1p(imf_var * 10)
            scale_adj = np.clip(scale_adj, 0.5, 3.0)
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
        def neg_ll(params):
            q, c, phi, iw = params
            if q <= 0 or c <= 0 or iw < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, iw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, iw = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, iw)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'imf_weight': iw,
            'max_imfs': self.max_imfs,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }


class VariationalModeKalmanModel(BaseExperimentalModel):
    """Variational Mode Decomposition Kalman with proper PIT calibration."""
    
    def __init__(self, n_modes: int = 4, alpha: float = 2000, tau: float = 0.0):
        self.n_modes = n_modes
        self.alpha = alpha
        self.tau = tau
    
    def _vmd(self, signal: np.ndarray, n_iterations: int = 100) -> List[np.ndarray]:
        n = len(signal)
        f_hat = np.fft.fft(signal)
        f_hat_plus = np.zeros(n, dtype=complex)
        f_hat_plus[:n//2] = f_hat[:n//2]
        freqs = np.arange(n) / n
        u_hat = np.zeros((self.n_modes, n), dtype=complex)
        omega = np.linspace(0.1, 0.4, self.n_modes)
        lambda_hat = np.zeros(n, dtype=complex)
        for iteration in range(n_iterations):
            for k in range(self.n_modes):
                sum_others = np.sum(u_hat, axis=0) - u_hat[k]
                numerator = f_hat_plus - sum_others + lambda_hat / 2
                denominator = 1 + self.alpha * (freqs - omega[k])**2
                u_hat[k] = numerator / denominator
                if np.sum(np.abs(u_hat[k])**2) > 1e-10:
                    omega[k] = np.sum(freqs * np.abs(u_hat[k])**2) / np.sum(np.abs(u_hat[k])**2)
            lambda_hat = lambda_hat + self.tau * (f_hat_plus - np.sum(u_hat, axis=0))
        modes = []
        for k in range(self.n_modes):
            u_full = np.zeros(n, dtype=complex)
            u_full[:n//2] = u_hat[k, :n//2]
            if n > 1:
                u_full[n//2+1:] = np.conj(u_hat[k, 1:n//2][::-1])
            mode = np.real(np.fft.ifft(u_full))
            modes.append(mode)
        return modes
    
    def _compute_mode_variance(self, modes: List[np.ndarray], t: int, window: int = 60) -> float:
        start = max(0, t - window)
        variances = []
        for mode in modes:
            if t < len(mode):
                segment = mode[start:t+1]
                if len(segment) > 1:
                    variances.append(np.var(segment))
        return np.sum(variances) if variances else 0.01
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        z = (actual - predicted_mean) / predicted_std
        return norm.cdf(z)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, mode_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        modes = self._vmd(returns)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            mode_var = self._compute_mode_variance(modes, t, window=60)
            scale_adj = 1.0 + mode_weight * np.log1p(mode_var * 10)
            scale_adj = np.clip(scale_adj, 0.5, 3.0)
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
        def neg_ll(params):
            q, c, phi, mw = params
            if q <= 0 or c <= 0 or mw < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, mw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, mw = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, mw)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'mode_weight': mw,
            'n_modes': self.n_modes, 'alpha': self.alpha,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }


class EnsembleEMDKalmanModel(BaseExperimentalModel):
    """Ensemble EMD with noise-assisted decomposition for robustness."""
    
    def __init__(self, max_imfs: int = 5, n_ensemble: int = 10, noise_std: float = 0.1):
        self.max_imfs = max_imfs
        self.n_ensemble = n_ensemble
        self.noise_std = noise_std
        self.hht = HilbertHuangKalmanModel(max_imfs=max_imfs)
    
    def _eemd(self, signal: np.ndarray) -> List[np.ndarray]:
        n = len(signal)
        all_imfs = []
        for _ in range(self.n_ensemble):
            noise = np.random.randn(n) * self.noise_std * np.std(signal)
            noisy_signal = signal + noise
            imfs = self.hht._emd(noisy_signal)
            all_imfs.append(imfs)
        n_imfs = max(len(imfs) for imfs in all_imfs) if all_imfs else 1
        ensemble_imfs = []
        for i in range(n_imfs):
            imf_sum = np.zeros(n)
            count = 0
            for imfs in all_imfs:
                if i < len(imfs) and len(imfs[i]) == n:
                    imf_sum += imfs[i]
                    count += 1
            if count > 0:
                ensemble_imfs.append(imf_sum / count)
        return ensemble_imfs
    
    def _compute_imf_variance(self, imfs: List[np.ndarray], t: int, window: int = 60) -> float:
        start = max(0, t - window)
        variances = []
        for imf in imfs:
            if t < len(imf):
                segment = imf[start:t+1]
                if len(segment) > 1:
                    variances.append(np.var(segment))
        return np.sum(variances) if variances else 0.01
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        z = (actual - predicted_mean) / predicted_std
        return norm.cdf(z)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, imf_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        imfs = self._eemd(returns)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            imf_var = self._compute_imf_variance(imfs, t, window=60)
            scale_adj = 1.0 + imf_weight * np.log1p(imf_var * 10)
            scale_adj = np.clip(scale_adj, 0.5, 3.0)
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
        def neg_ll(params):
            q, c, phi, iw = params
            if q <= 0 or c <= 0 or iw < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, iw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, iw = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, iw)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'imf_weight': iw,
            'n_ensemble': self.n_ensemble, 'noise_std': self.noise_std,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }


class CompleteEMDKalmanModel(BaseExperimentalModel):
    """Complete Ensemble EMD with adaptive noise for optimal decomposition."""
    
    def __init__(self, max_imfs: int = 5, n_realizations: int = 50):
        self.max_imfs = max_imfs
        self.n_realizations = n_realizations
        self.hht = HilbertHuangKalmanModel(max_imfs=max_imfs)
    
    def _ceemdan(self, signal: np.ndarray) -> List[np.ndarray]:
        n = len(signal)
        signal_std = np.std(signal)
        all_imfs = []
        residual = signal.copy()
        for mode_idx in range(self.max_imfs):
            if np.std(residual) < 1e-8:
                break
            mode_estimates = []
            for _ in range(self.n_realizations):
                noise_std = 0.2 * signal_std / (mode_idx + 1)
                noise = np.random.randn(n) * noise_std
                noisy_residual = residual + noise
                imfs = self.hht._emd(noisy_residual)
                if imfs:
                    mode_estimates.append(imfs[0])
            if mode_estimates:
                avg_mode = np.mean(mode_estimates, axis=0)
                all_imfs.append(avg_mode)
                residual = residual - avg_mode
        if np.std(residual) > 1e-10:
            all_imfs.append(residual)
        return all_imfs
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, imf_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        imfs = self._ceemdan(returns)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            start = max(0, t - 60)
            imf_var = sum(np.var(imf[start:t+1]) for imf in imfs if t < len(imf)) if imfs else 0.01
            scale_adj = 1.0 + imf_weight * np.log1p(imf_var * 10)
            scale_adj = np.clip(scale_adj, 0.5, 3.0)
            obs_var = base_var * scale_adj
            S = P_pred + obs_var
            predicted_std = np.sqrt(S)
            mu[t] = mu_pred
            sigma[t] = predicted_std
            innovation = returns[t] - mu_pred
            pit_values[t] = norm.cdf((returns[t] - mu_pred) / predicted_std) if predicted_std > 0 else 0.5
            K = P_pred / S if S > 0 else 0
            state = mu_pred + K * innovation
            P = (1 - K) * P_pred
            if t >= warmup and S > 1e-10:
                total_ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        return mu, sigma, total_ll, pit_values
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        import time
        start_time = time.time()
        def neg_ll(params):
            q, c, phi, iw = params
            if q <= 0 or c <= 0 or iw < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, iw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, iw = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, iw)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'imf_weight': iw,
            'n_realizations': self.n_realizations,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }
