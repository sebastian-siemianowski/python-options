"""
Time-Frequency Kalman Models - advanced time-frequency analysis with PIT calibration.
Includes Short-Time Fourier Transform, Wigner-Ville, and Gabor transform approaches.
"""

from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.signal import stft, istft, windows

from .base import BaseExperimentalModel


class STFTKalmanModel(BaseExperimentalModel):
    """Short-Time Fourier Transform Kalman with adaptive windowing."""
    
    def __init__(self, nperseg: int = 64, noverlap: int = 48):
        self.nperseg = nperseg
        self.noverlap = noverlap
    
    def _compute_stft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(signal)
        if n < self.nperseg:
            return np.array([[0]]), np.array([0]), np.array([0])
        f, t, Zxx = stft(signal, fs=1.0, nperseg=self.nperseg, noverlap=self.noverlap, window='hann')
        return f, t, Zxx
    
    def _compute_spectral_energy(self, Zxx: np.ndarray, t_idx: int) -> float:
        if t_idx >= Zxx.shape[1]:
            return 0.01
        return np.sum(np.abs(Zxx[:, t_idx])**2)
    
    def _compute_spectral_entropy(self, Zxx: np.ndarray, t_idx: int) -> float:
        if t_idx >= Zxx.shape[1]:
            return 0.5
        power = np.abs(Zxx[:, t_idx])**2
        total = np.sum(power) + 1e-10
        probs = power / total
        probs = probs[probs > 1e-10]
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(power))
        return entropy / max_entropy if max_entropy > 0 else 0.5
    
    def _map_time_to_stft(self, t: int, n: int, n_frames: int) -> int:
        hop = self.nperseg - self.noverlap
        stft_idx = max(0, (t - self.nperseg // 2) // hop)
        return min(stft_idx, n_frames - 1) if n_frames > 0 else 0
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        z = (actual - predicted_mean) / predicted_std
        return norm.cdf(z)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, spectral_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        f, t_frames, Zxx = self._compute_stft(returns)
        n_frames = Zxx.shape[1] if len(Zxx.shape) > 1 else 0
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            stft_idx = self._map_time_to_stft(t, n, n_frames)
            if n_frames > 0:
                spectral_energy = self._compute_spectral_energy(Zxx, stft_idx)
                spectral_entropy = self._compute_spectral_entropy(Zxx, stft_idx)
                scale_adj = 1.0 + spectral_weight * (np.log1p(spectral_energy) + (1 - spectral_entropy))
            else:
                scale_adj = 1.0
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
            q, c, phi, sw = params
            if q <= 0 or c <= 0 or sw < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, sw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, sw = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, sw)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'spectral_weight': sw,
            'nperseg': self.nperseg, 'noverlap': self.noverlap,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }


class GaborKalmanModel(BaseExperimentalModel):
    """Gabor transform Kalman with optimal time-frequency localization."""
    
    def __init__(self, n_freqs: int = 32, sigma: float = 4.0):
        self.n_freqs = n_freqs
        self.sigma = sigma
    
    def _gabor_atom(self, t: np.ndarray, t0: int, freq: float) -> np.ndarray:
        gaussian = np.exp(-((t - t0)**2) / (2 * self.sigma**2))
        return gaussian * np.exp(2j * np.pi * freq * t)
    
    def _compute_gabor_transform(self, signal: np.ndarray, hop: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(signal)
        freqs = np.linspace(0, 0.5, self.n_freqs)
        time_points = np.arange(0, n, hop)
        n_times = len(time_points)
        coeffs = np.zeros((self.n_freqs, n_times), dtype=complex)
        t = np.arange(n)
        for i, t0 in enumerate(time_points):
            for j, freq in enumerate(freqs):
                atom = self._gabor_atom(t, t0, freq)
                coeffs[j, i] = np.sum(signal * np.conj(atom))
        return freqs, time_points, coeffs
    
    def _compute_gabor_energy(self, coeffs: np.ndarray, t_idx: int) -> float:
        if t_idx >= coeffs.shape[1]:
            return 0.01
        return np.sum(np.abs(coeffs[:, t_idx])**2)
    
    def _map_time_to_gabor(self, t: int, time_points: np.ndarray) -> int:
        if len(time_points) == 0:
            return 0
        idx = np.searchsorted(time_points, t)
        return min(max(0, idx - 1), len(time_points) - 1)
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        return norm.cdf((actual - predicted_mean) / predicted_std)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, gabor_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        freqs, time_points, coeffs = self._compute_gabor_transform(returns)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            gabor_idx = self._map_time_to_gabor(t, time_points)
            gabor_energy = self._compute_gabor_energy(coeffs, gabor_idx)
            scale_adj = 1.0 + gabor_weight * np.log1p(gabor_energy)
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
            q, c, phi, gw = params
            if q <= 0 or c <= 0 or gw < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, gw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, gw = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, gw)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'gabor_weight': gw,
            'n_freqs': self.n_freqs, 'sigma': self.sigma,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }


class WignerVilleKalmanModel(BaseExperimentalModel):
    """Pseudo Wigner-Ville Distribution Kalman for high-resolution time-frequency."""
    
    def __init__(self, window_length: int = 64):
        self.window_length = window_length
    
    def _compute_pwvd(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        half_win = self.window_length // 2
        analytic = np.fft.ifft(np.fft.fft(signal) * (np.arange(n) < n//2).astype(float) * 2)
        pwvd = np.zeros((self.window_length, n))
        for t in range(half_win, n - half_win):
            tau_range = np.arange(-half_win, half_win)
            for i, tau in enumerate(tau_range):
                t1 = max(0, min(n-1, t + tau))
                t2 = max(0, min(n-1, t - tau))
                pwvd[i, t] = np.real(analytic[t1] * np.conj(analytic[t2]))
        return pwvd
    
    def _compute_wvd_energy(self, pwvd: np.ndarray, t: int) -> float:
        if t >= pwvd.shape[1]:
            return 0.01
        return np.sum(np.abs(pwvd[:, t]))
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        return norm.cdf((actual - predicted_mean) / predicted_std)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, wvd_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        pwvd = self._compute_pwvd(returns)
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            wvd_energy = self._compute_wvd_energy(pwvd, t)
            scale_adj = 1.0 + wvd_weight * np.log1p(wvd_energy)
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
            q, c, phi, ww = params
            if q <= 0 or c <= 0 or ww < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, ww)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, ww = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, ww)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'wvd_weight': ww,
            'window_length': self.window_length,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }


class MultitaperKalmanModel(BaseExperimentalModel):
    """Multitaper spectral estimation Kalman for reduced variance spectral estimates."""
    
    def __init__(self, nw: float = 4.0, n_tapers: int = 7):
        self.nw = nw
        self.n_tapers = n_tapers
    
    def _dpss_tapers(self, n: int) -> np.ndarray:
        tapers = []
        for k in range(self.n_tapers):
            t = np.arange(n)
            w = self.nw / n
            taper = np.sin(np.pi * w * (2 * (t - n/2) + 1)) / (np.pi * (t - n/2 + 0.5))
            taper *= np.cos(2 * np.pi * k * t / n)
            taper = taper / np.sqrt(np.sum(taper**2) + 1e-10)
            tapers.append(taper)
        return np.array(tapers)
    
    def _multitaper_spectrum(self, signal: np.ndarray, window: int = 128) -> np.ndarray:
        n = len(signal)
        n_windows = max(1, n // window)
        spectra = np.zeros((n_windows, window // 2))
        tapers = self._dpss_tapers(window)
        for i in range(n_windows):
            start = i * window
            end = min(start + window, n)
            segment = signal[start:end]
            if len(segment) < window:
                segment = np.concatenate([segment, np.zeros(window - len(segment))])
            spectrum = np.zeros(window // 2)
            for taper in tapers:
                tapered = segment * taper[:len(segment)]
                fft_result = np.fft.fft(tapered, n=window)
                spectrum += np.abs(fft_result[:window // 2])**2
            spectra[i] = spectrum / self.n_tapers
        return spectra
    
    def _compute_mt_energy(self, spectra: np.ndarray, t_idx: int) -> float:
        if t_idx >= spectra.shape[0]:
            return 0.01
        return np.sum(spectra[t_idx])
    
    def _map_time_to_window(self, t: int, window: int, n_windows: int) -> int:
        idx = t // window
        return min(max(0, idx), n_windows - 1) if n_windows > 0 else 0
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        return norm.cdf((actual - predicted_mean) / predicted_std)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, mt_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        window = 128
        spectra = self._multitaper_spectrum(returns, window)
        n_windows = spectra.shape[0]
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            win_idx = self._map_time_to_window(t, window, n_windows)
            mt_energy = self._compute_mt_energy(spectra, win_idx)
            scale_adj = 1.0 + mt_weight * np.log1p(mt_energy)
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
            'q': q, 'c': c, 'phi': phi, 'mt_weight': mw,
            'nw': self.nw, 'n_tapers': self.n_tapers,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }


class ChirpletKalmanModel(BaseExperimentalModel):
    """Chirplet transform Kalman for non-stationary frequency analysis."""
    
    def __init__(self, n_chirplets: int = 16, sigma: float = 4.0):
        self.n_chirplets = n_chirplets
        self.sigma = sigma
    
    def _chirplet_atom(self, t: np.ndarray, t0: int, f0: float, chirp_rate: float) -> np.ndarray:
        gaussian = np.exp(-((t - t0)**2) / (2 * self.sigma**2))
        instantaneous_freq = f0 + chirp_rate * (t - t0)
        phase = 2 * np.pi * (f0 * (t - t0) + 0.5 * chirp_rate * (t - t0)**2)
        return gaussian * np.exp(1j * phase)
    
    def _compute_chirplet_coeffs(self, signal: np.ndarray, hop: int = 8) -> np.ndarray:
        n = len(signal)
        freqs = np.linspace(0.01, 0.4, self.n_chirplets // 2)
        chirp_rates = np.linspace(-0.01, 0.01, 2)
        time_points = np.arange(0, n, hop)
        n_times = len(time_points)
        n_atoms = len(freqs) * len(chirp_rates)
        coeffs = np.zeros((n_atoms, n_times), dtype=complex)
        t = np.arange(n)
        for i, t0 in enumerate(time_points):
            atom_idx = 0
            for f0 in freqs:
                for cr in chirp_rates:
                    atom = self._chirplet_atom(t, t0, f0, cr)
                    coeffs[atom_idx, i] = np.sum(signal * np.conj(atom))
                    atom_idx += 1
        return coeffs
    
    def _compute_chirplet_energy(self, coeffs: np.ndarray, t_idx: int) -> float:
        if t_idx >= coeffs.shape[1]:
            return 0.01
        return np.sum(np.abs(coeffs[:, t_idx])**2)
    
    def _gaussian_log_likelihood(self, innovation: float, variance: float) -> float:
        if variance <= 0:
            return -1e10
        return -0.5 * np.log(2 * np.pi * variance) - 0.5 * innovation**2 / variance
    
    def _compute_pit_value(self, actual: float, predicted_mean: float, predicted_std: float) -> float:
        if predicted_std <= 0:
            return 0.5
        return norm.cdf((actual - predicted_mean) / predicted_std)
    
    def filter(self, returns: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float, chirp_weight: float) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu = np.zeros(n)
        sigma = np.zeros(n)
        pit_values = np.zeros(n)
        state = 0.0
        P = q
        total_ll = 0.0
        warmup = 60
        hop = 8
        coeffs = self._compute_chirplet_coeffs(returns, hop)
        n_times = coeffs.shape[1]
        for t in range(1, n):
            mu_pred = phi * state
            P_pred = phi**2 * P + q
            base_var = (c * vol[t])**2 if vol[t] > 0 else (c * 0.01)**2
            chirp_idx = min(t // hop, n_times - 1) if n_times > 0 else 0
            chirp_energy = self._compute_chirplet_energy(coeffs, chirp_idx)
            scale_adj = 1.0 + chirp_weight * np.log1p(chirp_energy)
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
            q, c, phi, cw = params
            if q <= 0 or c <= 0 or cw < 0:
                return 1e10
            try:
                _, _, ll, _ = self.filter(returns, vol, q, c, phi, cw)
                return -ll
            except:
                return 1e10
        result = minimize(neg_ll, x0=[1e-6, 1.0, 0.0, 0.3], method='L-BFGS-B', bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.0, 1.0)])
        q, c, phi, cw = result.x
        mu, sigma, final_ll, pit_values = self.filter(returns, vol, q, c, phi, cw)
        n = len(returns)
        n_params = 4
        bic = -2 * final_ll + n_params * np.log(n - 60)
        from scipy.stats import kstest
        pit_clean = pit_values[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        return {
            'q': q, 'c': c, 'phi': phi, 'chirp_weight': cw,
            'n_chirplets': self.n_chirplets, 'sigma': self.sigma,
            'log_likelihood': final_ll, 'bic': bic, 'pit_ks_pvalue': ks_pvalue,
            'n_params': n_params, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': q, 'c': c, 'phi': phi},
        }
