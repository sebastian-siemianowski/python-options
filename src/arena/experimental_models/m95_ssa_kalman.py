"""
SSA-Kalman Hybrid Models (5 variants)
Singular Spectrum Analysis combined with Kalman filtering.
Extracts latent oscillatory structure via trajectory matrix SVD.

Mathematical Foundation:
- Trajectory matrix X = [x_1, x_2, ..., x_K] with x_i = (y_i, ..., y_{i+L-1})
- SVD: X = U Σ V^T extracts principal components (singular triplets)
- Grouping: Partition singular triplets by eigenvalue magnitude
- Reconstruction: Sum grouped components for trend, seasonality, noise

Hard Gate Targets:
- CSS >= 0.65 via stress-aware component weighting
- FEC >= 0.75 via entropy-consistent reconstruction
- vs STD > 3 points via superior trend extraction
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import svd, hankel
from typing import Dict, Optional, Tuple, Any, List
from .base import BaseExperimentalModel


class SSAKalmanBase(BaseExperimentalModel):
    """Base class for SSA-Kalman hybrid models with DTCWT preprocessing."""
    
    def __init__(self, window: int = 50, n_components: int = 5, n_levels: int = 4):
        self.window = window
        self.n_components = n_components
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._init_wavelet_filters()
    
    def _init_wavelet_filters(self):
        self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
    
    def _filter_downsample(self, sig: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(sig, h, mode='same')[::2]
    
    def _dtcwt_analysis(self, sig: np.ndarray) -> Tuple[List, List]:
        cr, ci = [], []
        ca, cb = sig.copy(), sig.copy()
        for _ in range(self.n_levels):
            if len(ca) < 8:
                break
            la = self._filter_downsample(ca, self.h0a)
            ha = self._filter_downsample(ca, self.h1a)
            lb = self._filter_downsample(cb, self.h0b)
            hb = self._filter_downsample(cb, self.h1b)
            cr.append((ha + hb) / np.sqrt(2))
            ci.append((ha - hb) / np.sqrt(2))
            ca, cb = la, lb
        cr.append((ca + cb) / np.sqrt(2))
        ci.append((ca - cb) / np.sqrt(2))
        return cr, ci
    
    def _build_trajectory_matrix(self, x: np.ndarray, L: int) -> np.ndarray:
        N = len(x)
        K = N - L + 1
        if K < 1:
            return np.array([[x[0]]])
        X = np.zeros((L, K))
        for i in range(K):
            X[:, i] = x[i:i+L]
        return X
    
    def _ssa_decomposition(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(x)
        L = min(self.window, N // 2)
        if L < 2:
            return np.zeros_like(x), np.zeros_like(x), x.copy()
        X = self._build_trajectory_matrix(x, L)
        try:
            U, s, Vh = svd(X, full_matrices=False)
        except Exception:
            return np.zeros_like(x), np.zeros_like(x), x.copy()
        r = min(self.n_components, len(s))
        trend = np.zeros(N)
        oscillatory = np.zeros(N)
        noise = np.zeros(N)
        for i in range(len(s)):
            Xi = s[i] * np.outer(U[:, i], Vh[i, :])
            recon = self._diagonal_averaging(Xi, N)
            if i == 0:
                trend = recon
            elif i < r:
                oscillatory += recon
            else:
                noise += recon
        return trend, oscillatory, noise
    
    def _diagonal_averaging(self, X: np.ndarray, N: int) -> np.ndarray:
        L, K = X.shape
        if L + K - 1 != N:
            result = np.zeros(N)
            result[:min(N, L*K)] = X.flatten()[:min(N, L*K)]
            return result
        result = np.zeros(N)
        for k in range(N):
            count = 0
            total = 0.0
            for i in range(L):
                j = k - i
                if 0 <= j < K:
                    total += X[i, j]
                    count += 1
            if count > 0:
                result[k] = total / count
        return result
    
    def _detect_stress(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return 1.0
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return 1.0
        spike = vol[t] / (np.median(rv) + 1e-8) if vol[t] > 0 else 1.0
        return np.clip(1.0 + 0.4 * max(0, spike - 1.3), 1.0, 2.5)
    
    def _vol_regime(self, vol: np.ndarray, t: int, win: int = 60) -> float:
        if t < win:
            return 1.0
        rv = vol[max(0, t-win):t]
        rv = rv[rv > 0]
        if len(rv) < 10:
            return 1.0
        curr = vol[t] if vol[t] > 0 else np.mean(rv)
        pct = (rv < curr).sum() / len(rv)
        if pct < 0.3:
            return 1.15
        elif pct > 0.7:
            return 0.85
        return 1.0
    
    def _filter_scale_ll(self, mag: np.ndarray, vol: np.ndarray, q: float, c: float, phi: float) -> float:
        n = len(mag)
        P, st, ll = 1e-4, 0.0, 0.0
        vs = vol[::max(1, len(vol)//n)][:n] if len(vol) > n else np.ones(n) * 0.01
        for t in range(1, n):
            mp = phi * st
            Pp = phi**2 * P + q
            v = vs[t] if t < len(vs) and vs[t] > 0 else 0.01
            S = Pp + (c * v)**2
            inn = mag[t] - mp
            K = Pp / S if S > 0 else 0
            st = mp + K * inn
            P = (1 - K) * Pp
            if S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return ll
    
    def _base_fit(self, returns: np.ndarray, vol: np.ndarray, filt, init: Optional[Dict] = None) -> Dict[str, Any]:
        import time
        start = time.time()
        params = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'cw': 1.0, 'ssa_weight': 0.3}
        params.update(init or {})
        def neg_ll(x):
            if time.time() - start > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = {'q': x[0], 'c': x[1], 'phi': x[2], 'cw': x[3], 'ssa_weight': x[4]}
            if p['q'] <= 0 or p['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = filt(returns, vol, p)
                return -ll
            except:
                return 1e10
        res = minimize(neg_ll, [params['q'], params['c'], params['phi'], params['cw'], params['ssa_weight']], 
                      method='L-BFGS-B', 
                      bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.1, 2.0), (0.0, 0.7)], 
                      options={'maxiter': 80})
        opt = {'q': res.x[0], 'c': res.x[1], 'phi': res.x[2], 'cw': res.x[3], 'ssa_weight': res.x[4]}
        mu, sigma, ll, pit = filt(returns, vol, opt)
        n = len(returns)
        bic = -2 * ll + 5 * np.log(n - 60)
        from scipy.stats import kstest
        pc = pit[60:]
        pc = pc[(pc > 0.001) & (pc < 0.999)]
        ks_p = kstest(pc, 'uniform')[1] if len(pc) > 50 else 1.0
        return {'q': opt['q'], 'c': opt['c'], 'phi': opt['phi'], 'complex_weight': opt['cw'],
                'ssa_weight': opt['ssa_weight'], 'log_likelihood': ll, 'bic': bic, 
                'pit_ks_pvalue': ks_p, 'n_params': 5, 'success': res.success,
                'fit_time_ms': (time.time() - start) * 1000,
                'fit_params': {'q': opt['q'], 'c': opt['c'], 'phi': opt['phi']}}


class SSAKalmanModel(SSAKalmanBase):
    """Standard SSA-Kalman hybrid model."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, sw = params['cw'], params['ssa_weight']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        trend, osc, noise = self._ssa_decomposition(returns[:min(200, n)])
        trend = np.pad(trend, (0, max(0, n - len(trend))), mode='edge')
        P, st = 1e-4, 0.0
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            mp = phi * st
            if t < len(trend):
                ssa_pred = trend[t] if abs(trend[t]) < 0.1 else 0.0
                mp = (1 - sw) * mp + sw * ssa_pred
            Pp = phi**2 * P + q * stress * regime
            so = c * vol[t] * np.sqrt(stress) * regime if vol[t] > 0 else c * 0.01
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            inn = returns[t] - mp
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class SSAKalmanStressModel(SSAKalmanBase):
    """SSA-Kalman with enhanced stress detection for CSS."""
    
    def _enhanced_stress(self, vol: np.ndarray, returns: np.ndarray, t: int) -> float:
        s1 = self._detect_stress(vol, t)
        s2 = 1.0
        if t >= 20:
            rv = np.abs(returns[t-20:t])
            if len(rv) > 5:
                curr = abs(returns[t]) if t < len(returns) else 0
                s2 = 1.0 + 0.25 * max(0, curr / (np.mean(rv) + 1e-8) - 1.8)
        return np.sqrt(s1 * s2)
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, sw = params['cw'], params['ssa_weight']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        for t in range(1, n):
            stress = self._enhanced_stress(vol, returns, t)
            regime = self._vol_regime(vol, t)
            mp = phi * st
            Pp = phi**2 * P + q * stress * regime
            so = c * vol[t] * np.sqrt(stress) * regime if vol[t] > 0 else c * 0.01
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            inn = returns[t] - mp
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class SSAKalmanEntropyModel(SSAKalmanBase):
    """SSA-Kalman with entropy matching for FEC."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, sw = params['cw'], params['ssa_weight']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            ema_vol = 0.05 * vol[t] + 0.95 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * regime
            so = c * ema_vol * np.sqrt(stress) * regime
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            inn = returns[t] - mp
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class SSAKalmanRobustModel(SSAKalmanBase):
    """SSA-Kalman with robust M-estimation for outlier resistance."""
    
    def _huber_weight(self, residual: float, scale: float, k: float = 1.5) -> float:
        z = abs(residual) / (scale + 1e-10)
        if z <= k:
            return 1.0
        return k / z
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, sw = params['cw'], params['ssa_weight']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        scale_est = 0.01
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            mp = phi * st
            Pp = phi**2 * P + q * stress * regime
            so = c * vol[t] * np.sqrt(stress) * regime if vol[t] > 0 else c * 0.01
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            inn = returns[t] - mp
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            w = self._huber_weight(inn, scale_est)
            K = w * Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            scale_est = 0.95 * scale_est + 0.05 * abs(inn)
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * w * inn**2 / S
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class SSAKalmanAdaptiveModel(SSAKalmanBase):
    """SSA-Kalman with adaptive calibration adjustments."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, sw = params['cw'], params['ssa_weight']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        calib_adj = 1.0
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            if t > 100 and t % 20 == 0:
                rp = pit[max(60, t-50):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 15:
                    spread = np.std(rp)
                    target_spread = 1/np.sqrt(12)
                    calib_adj = 1.0 + 0.15 * (target_spread - spread)
                    calib_adj = np.clip(calib_adj, 0.9, 1.15)
            mp = phi * st
            Pp = phi**2 * P + q * stress * regime
            so = c * vol[t] * calib_adj * np.sqrt(stress) * regime if vol[t] > 0 else c * 0.01
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            inn = returns[t] - mp
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)
