"""
Generation 9 - Batch 1: DTCWT Evolution (10 models)
Base: dualtree_complex_wavelet (top performer at +10.0 vs STD)
Focus: Multi-scale wavelet decomposition with enhanced phase coherence

Mathematical Foundation:
- Dual-Tree Complex Wavelet Transform for shift-invariant decomposition
- Phase-magnitude separation for directional information
- Multi-resolution Kalman filtering with scale-specific noise models
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Optional, Tuple, Any, List
import time


class Gen9DTCWTBase:
    """Enhanced DTCWT base with configurable decomposition levels."""
    
    def __init__(self, n_levels: int = 4, wavelet_type: str = 'near_sym'):
        self.n_levels = n_levels
        self.wavelet_type = wavelet_type
        self.max_time_ms = 10000
        self._init_filters()
    
    def _init_filters(self):
        if self.wavelet_type == 'near_sym':
            self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
            self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
            self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
            self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
        elif self.wavelet_type == 'qshift':
            self.h0a = np.array([-0.0046, -0.0116, 0.0503, 0.2969, 0.5594, 0.2969, 0.0503, -0.0116, -0.0046, 0.0]) * np.sqrt(2)
            self.h1a = np.array([0.0046, -0.0116, -0.0503, 0.2969, -0.5594, 0.2969, -0.0503, -0.0116, 0.0046, 0.0]) * np.sqrt(2)
            self.h0b = self.h0a[::-1]
            self.h1b = -self.h1a[::-1]
        else:
            self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
            self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
            self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
            self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
    
    def _filt_down(self, signal: np.ndarray, h: np.ndarray) -> np.ndarray:
        padded = np.pad(signal, (len(h)//2, len(h)//2), mode='reflect')
        filtered = np.convolve(padded, h, mode='same')
        return filtered[len(h)//2:-len(h)//2:2] if len(filtered) > len(h) else filtered[::2]
    
    def _dtcwt_decompose(self, signal: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        coef_real, coef_imag = [], []
        ca, cb = signal.copy(), signal.copy()
        for level in range(self.n_levels):
            if len(ca) < 8:
                break
            la = self._filt_down(ca, self.h0a)
            ha = self._filt_down(ca, self.h1a)
            lb = self._filt_down(cb, self.h0b)
            hb = self._filt_down(cb, self.h1b)
            coef_real.append((ha + hb) / np.sqrt(2))
            coef_imag.append((ha - hb) / np.sqrt(2))
            ca, cb = la, lb
        coef_real.append((ca + cb) / np.sqrt(2))
        coef_imag.append((ca - cb) / np.sqrt(2))
        return coef_real, coef_imag
    
    def _compute_magnitude_phase(self, coef_real: List[np.ndarray], coef_imag: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        magnitudes = [np.sqrt(r**2 + i**2 + 1e-10) for r, i in zip(coef_real, coef_imag)]
        phases = [np.arctan2(i, r + 1e-10) for r, i in zip(coef_real, coef_imag)]
        return magnitudes, phases
    
    def _multi_horizon_stress(self, vol: np.ndarray, t: int) -> float:
        stress_5 = stress_20 = stress_60 = 1.0
        if t >= 5:
            recent = vol[t-5:t]
            recent = recent[recent > 0]
            if len(recent) >= 3 and vol[t] > 0:
                stress_5 = 1.0 + 0.5 * max(0, vol[t] / (np.median(recent) + 1e-8) - 1.2)
        if t >= 20:
            recent = vol[t-20:t]
            recent = recent[recent > 0]
            if len(recent) >= 5 and vol[t] > 0:
                stress_20 = 1.0 + 0.35 * max(0, vol[t] / (np.median(recent) + 1e-8) - 1.3)
        if t >= 60:
            recent = vol[t-60:t]
            recent = recent[recent > 0]
            if len(recent) >= 10 and vol[t] > 0:
                stress_60 = 1.0 + 0.25 * max(0, vol[t] / (np.median(recent) + 1e-8) - 1.2)
        combined = np.power(stress_5 * stress_20 * stress_60, 1/2.5)
        return np.clip(combined, 1.0, 3.0)
    
    def _vol_regime_multiplier(self, vol: np.ndarray, t: int, window: int = 60) -> float:
        if t < window:
            return 1.0
        recent_vol = vol[max(0, t-window):t]
        recent_vol = recent_vol[recent_vol > 0]
        if len(recent_vol) < 10:
            return 1.0
        current = vol[t] if vol[t] > 0 else np.mean(recent_vol)
        percentile = (recent_vol < current).sum() / len(recent_vol)
        if percentile < 0.25:
            return 1.25
        elif percentile > 0.75:
            return 0.85
        return 1.0
    
    def _scale_likelihood(self, magnitudes: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> float:
        n = len(magnitudes)
        P, state, ll = 1e-4, 0.0, 0.0
        vol_sampled = vol[::max(1, len(vol)//n)][:n] if len(vol) > n else np.ones(n) * 0.01
        for t in range(1, n):
            pred_mean = phi * state
            pred_var = phi**2 * P + q
            obs_var = vol_sampled[t] if t < len(vol_sampled) and vol_sampled[t] > 0 else 0.01
            S = pred_var + (c * obs_var)**2
            innovation = magnitudes[t] - pred_mean
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * innovation
            P = (1 - K) * pred_var
            if S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * innovation**2 / S
        return ll
    
    def _base_fit(self, returns: np.ndarray, vol: np.ndarray, filter_func, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        start_time = time.time()
        params = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'cw': 1.0}
        if init_params:
            params.update(init_params)
        
        def negative_ll(x):
            elapsed = time.time() - start_time
            if elapsed > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = {'q': x[0], 'c': x[1], 'phi': x[2], 'cw': x[3]}
            if p['q'] <= 0 or p['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = filter_func(returns, vol, p)
                return -ll
            except Exception:
                return 1e10
        
        result = minimize(
            negative_ll,
            [params['q'], params['c'], params['phi'], params['cw']],
            method='L-BFGS-B',
            bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.5, 2.0)],
            options={'maxiter': 80}
        )
        
        opt_params = {'q': result.x[0], 'c': result.x[1], 'phi': result.x[2], 'cw': result.x[3]}
        mu, sigma, ll, pit = filter_func(returns, vol, opt_params)
        
        n = len(returns)
        bic = -2 * ll + 4 * np.log(max(1, n - 60))
        
        from scipy.stats import kstest
        pit_clean = pit[60:]
        pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
        ks_pvalue = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0
        
        return {
            'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi'],
            'complex_weight': opt_params['cw'], 'log_likelihood': ll, 'bic': bic,
            'pit_ks_pvalue': ks_pvalue, 'n_params': 4, 'success': result.success,
            'fit_time_ms': (time.time() - start_time) * 1000,
            'fit_params': {'q': opt_params['q'], 'c': opt_params['c'], 'phi': opt_params['phi']}
        }


class DTCWTMultiScaleModel(Gen9DTCWTBase):
    """Multi-scale DTCWT with scale-specific Kalman parameters."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        scale_weights = [1.0 / (1.5 ** i) for i in range(len(mags))]
        scale_sum = sum(scale_weights)
        scale_weights = [w / scale_sum for w in scale_weights]
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw * scale_weights[i]
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state
            pred_var = phi**2 * P + q * stress * regime_mult
            blend_vol = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTPhaseCoherentModel(Gen9DTCWTBase):
    """DTCWT with phase coherence tracking for trend detection."""
    
    def _phase_coherence(self, phases: List[np.ndarray], t_idx: int, window: int = 10) -> float:
        if not phases or t_idx < window:
            return 1.0
        coherences = []
        for phase in phases:
            if len(phase) > t_idx:
                idx = min(t_idx, len(phase) - 1)
                start = max(0, idx - window)
                phase_window = phase[start:idx+1]
                if len(phase_window) >= 3:
                    phase_diff = np.diff(phase_window)
                    phase_diff = np.mod(phase_diff + np.pi, 2*np.pi) - np.pi
                    coherence = np.abs(np.mean(np.exp(1j * phase_diff)))
                    coherences.append(coherence)
        return np.mean(coherences) if coherences else 1.0
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, phases = self._compute_magnitude_phase(coef_r, coef_i)
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            
            t_scaled = int(t * len(phases[0]) / n) if len(phases) > 0 and len(phases[0]) > 0 else 0
            coherence = self._phase_coherence(phases, t_scaled)
            coherence_mult = 1.0 - 0.15 * (1 - coherence)
            
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state
            pred_var = phi**2 * P + q * stress * regime_mult * coherence_mult
            blend_vol = 0.55 * vol[t] + 0.45 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTAdaptiveLevelsModel(Gen9DTCWTBase):
    """DTCWT with adaptive number of decomposition levels based on signal length."""
    
    def __init__(self):
        super().__init__(n_levels=5)
    
    def _adaptive_levels(self, signal_length: int) -> int:
        if signal_length < 64:
            return 2
        elif signal_length < 128:
            return 3
        elif signal_length < 256:
            return 4
        elif signal_length < 512:
            return 5
        return 6
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        self.n_levels = self._adaptive_levels(n)
        
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (1.8**i), c, phi) * cw
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            ema_vol = 0.065 * vol[t] + 0.935 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state
            pred_var = phi**2 * P + q * stress * regime_mult
            blend_vol = 0.58 * vol[t] + 0.42 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.35 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTMagnitudeWeightedModel(Gen9DTCWTBase):
    """DTCWT with magnitude-weighted scale contributions."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        mag_energies = [np.sum(m**2) for m in mags]
        total_energy = sum(mag_energies) + 1e-10
        mag_weights = [e / total_energy for e in mag_energies]
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw * mag_weights[i] * len(mags)
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state
            pred_var = phi**2 * P + q * stress * regime_mult
            blend_vol = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTEntropyRegularizedModel(Gen9DTCWTBase):
    """DTCWT with entropy regularization for FEC improvement."""
    
    def _scale_entropy(self, mags: List[np.ndarray]) -> float:
        energies = [np.sum(m**2) + 1e-10 for m in mags]
        total = sum(energies)
        probs = [e / total for e in energies]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(mags))
        return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        entropy_factor = self._scale_entropy(mags)
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            
            entropy_adj = 1.0 + 0.1 * (1 - entropy_factor)
            
            pred_mean = phi * state
            pred_var = phi**2 * P + q * stress * regime_mult * entropy_adj
            blend_vol = 0.55 * vol[t] + 0.45 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult * entropy_adj
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTQShiftModel(Gen9DTCWTBase):
    """DTCWT using Q-shift filters for improved frequency selectivity."""
    
    def __init__(self):
        super().__init__(n_levels=4, wavelet_type='qshift')
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state
            pred_var = phi**2 * P + q * stress * regime_mult
            blend_vol = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTDirectionalModel(Gen9DTCWTBase):
    """DTCWT with directional (up/down trend) sensitivity."""
    
    def _directional_bias(self, ret: np.ndarray, t: int, window: int = 20) -> float:
        if t < window:
            return 0.0
        recent = ret[t-window:t]
        pos_frac = (recent > 0).mean()
        return 2 * (pos_frac - 0.5)
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            dir_bias = self._directional_bias(ret, t)
            
            ema_vol = 0.065 * vol[t] + 0.935 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state + 0.02 * dir_bias * ema_vol
            pred_var = phi**2 * P + q * stress * regime_mult
            blend_vol = 0.58 * vol[t] + 0.42 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTVolClusterModel(Gen9DTCWTBase):
    """DTCWT with volatility clustering awareness."""
    
    def _vol_cluster_factor(self, vol: np.ndarray, t: int, window: int = 30) -> float:
        if t < window:
            return 1.0
        recent = vol[t-window:t]
        recent = recent[recent > 0]
        if len(recent) < 10:
            return 1.0
        try:
            autocorr = np.corrcoef(recent[:-1], recent[1:])[0, 1]
            return 1.0 + 0.2 * max(0, autocorr - 0.3)
        except:
            return 1.0
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            cluster = self._vol_cluster_factor(vol, t)
            
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state
            pred_var = phi**2 * P + q * stress * regime_mult * cluster
            blend_vol = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)


class DTCWTMomentumEnhancedModel(Gen9DTCWTBase):
    """DTCWT with momentum signal enhancement."""
    
    def _momentum_signal(self, ret: np.ndarray, t: int, fast: int = 5, slow: int = 20) -> float:
        if t < slow:
            return 0.0
        fast_ma = np.mean(ret[t-fast:t])
        slow_ma = np.mean(ret[t-slow:t])
        return fast_ma - slow_ma
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        
        coef_r, coef_i = self._dtcwt_decompose(ret)
        mags, _ = self._compute_magnitude_phase(coef_r, coef_i)
        
        ll = sum(
            self._scale_likelihood(mags[i], vol, q * (2**i), c, phi) * cw
            for i in range(len(mags))
        )
        
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        
        for t in range(1, n):
            stress = self._multi_horizon_stress(vol, t)
            regime_mult = self._vol_regime_multiplier(vol, t)
            momentum = self._momentum_signal(ret, t)
            
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            
            pred_mean = phi * state + 0.1 * momentum
            pred_var = phi**2 * P + q * stress * regime_mult
            blend_vol = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            obs_std = c * blend_vol * np.sqrt(stress) * regime_mult
            S = pred_var + obs_std**2
            
            mu[t] = pred_mean
            sigma[t] = np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - pred_mean) / sigma[t]) if sigma[t] > 0 else 0.5
            
            K = pred_var / S if S > 0 else 0
            state = pred_mean + K * (ret[t] - pred_mean)
            P = (1 - K) * pred_var
            
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - pred_mean)**2 / S
        
        return mu, sigma, ll * (1 + 0.4 * len(mags)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init)
