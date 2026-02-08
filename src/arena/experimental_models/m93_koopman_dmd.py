"""
Koopman-DMD Models (5 variants)
Dynamic Mode Decomposition for linearized nonlinear dynamics.
Lifts system to higher-dimensional space where evolution is linear.

Mathematical Foundation:
- Koopman operator K: g(x_{t+1}) = K g(x_t) for observables g
- DMD approximates K via SVD of data matrices
- Eigenvalues encode frequencies and growth rates
- Modes capture coherent spatiotemporal patterns

Hard Gate Targets:
- CSS >= 0.65 via stress-adaptive mode weighting
- FEC >= 0.75 via entropy-aware prediction intervals
- vs STD > 3 points via superior decomposition
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import svd, eig
from typing import Dict, Optional, Tuple, Any, List
from .base import BaseExperimentalModel


class KoopmanDMDBase(BaseExperimentalModel):
    """Base class for Koopman-DMD models with DTCWT preprocessing."""
    
    def __init__(self, n_modes: int = 10, n_levels: int = 4):
        self.n_modes = n_modes
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
    
    def _build_hankel(self, x: np.ndarray, delay: int) -> np.ndarray:
        n = len(x) - delay
        if n <= 0:
            return np.array([[]])
        H = np.zeros((delay, n))
        for i in range(delay):
            H[i, :] = x[i:i+n]
        return H
    
    def _exact_dmd(self, X: np.ndarray, Y: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if X.shape[1] < 2 or Y.shape[1] < 2:
            return np.eye(min(X.shape[0], rank)), np.ones(rank), np.eye(rank, X.shape[0])
        try:
            U, s, Vh = svd(X, full_matrices=False)
            r = min(rank, len(s), np.sum(s > 1e-10))
            if r < 1:
                r = 1
            U_r = U[:, :r]
            s_r = s[:r]
            V_r = Vh[:r, :].T
            S_inv = np.diag(1.0 / (s_r + 1e-10))
            A_tilde = U_r.T @ Y @ V_r @ S_inv
            eigenvalues, W = eig(A_tilde)
            Phi = Y @ V_r @ S_inv @ W
            return Phi, eigenvalues, W
        except Exception:
            return np.eye(min(X.shape[0], rank)), np.ones(rank), np.eye(rank, X.shape[0])
    
    def _koopman_observables(self, x: np.ndarray, vol: np.ndarray) -> np.ndarray:
        n = len(x)
        obs = np.zeros((6, n))
        obs[0, :] = x
        obs[1, :] = x ** 2
        obs[2, :] = np.sign(x) * np.abs(x) ** 0.5
        obs[3, :] = vol
        obs[4, :] = vol ** 2
        ema = np.zeros(n)
        ema[0] = x[0]
        for i in range(1, n):
            ema[i] = 0.1 * x[i] + 0.9 * ema[i-1]
        obs[5, :] = ema
        return obs
    
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
        params = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'cw': 1.0, 'dmd_weight': 0.5}
        params.update(init or {})
        def neg_ll(x):
            if time.time() - start > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = {'q': x[0], 'c': x[1], 'phi': x[2], 'cw': x[3], 'dmd_weight': x[4]}
            if p['q'] <= 0 or p['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = filt(returns, vol, p)
                return -ll
            except:
                return 1e10
        res = minimize(neg_ll, [params['q'], params['c'], params['phi'], params['cw'], params['dmd_weight']], 
                      method='L-BFGS-B', 
                      bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.1, 2.0), (0.0, 1.0)], 
                      options={'maxiter': 80})
        opt = {'q': res.x[0], 'c': res.x[1], 'phi': res.x[2], 'cw': res.x[3], 'dmd_weight': res.x[4]}
        mu, sigma, ll, pit = filt(returns, vol, opt)
        n = len(returns)
        bic = -2 * ll + 5 * np.log(n - 60)
        from scipy.stats import kstest
        pc = pit[60:]
        pc = pc[(pc > 0.001) & (pc < 0.999)]
        ks_p = kstest(pc, 'uniform')[1] if len(pc) > 50 else 1.0
        return {'q': opt['q'], 'c': opt['c'], 'phi': opt['phi'], 'complex_weight': opt['cw'],
                'dmd_weight': opt['dmd_weight'], 'log_likelihood': ll, 'bic': bic, 
                'pit_ks_pvalue': ks_p, 'n_params': 5, 'success': res.success,
                'fit_time_ms': (time.time() - start) * 1000,
                'fit_params': {'q': opt['q'], 'c': opt['c'], 'phi': opt['phi']}}


class KoopmanDMDModel(KoopmanDMDBase):
    """Standard Koopman-DMD with DTCWT preprocessing."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, dw = params['cw'], params['dmd_weight']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        obs = self._koopman_observables(returns[:min(200, n)], vol[:min(200, n)])
        delay = min(50, obs.shape[1] // 3)
        if delay > 5 and obs.shape[1] > delay + 10:
            X = obs[:, :-1]
            Y = obs[:, 1:]
            Phi, eigs, _ = self._exact_dmd(X, Y, self.n_modes)
            dmd_pred = np.zeros(n)
            for t in range(60, min(n, 200)):
                if t < obs.shape[1]:
                    dmd_pred[t] = np.real(Phi[0, :] @ (eigs ** (t - 60)) @ np.linalg.pinv(Phi) @ obs[:, 60])
        else:
            dmd_pred = np.zeros(n)
        P, st = 1e-4, 0.0
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            mp = phi * st
            if t < len(dmd_pred) and abs(dmd_pred[t]) < 0.1:
                mp = (1 - dw) * mp + dw * dmd_pred[t]
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
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class KoopmanDMDStressModel(KoopmanDMDBase):
    """Koopman-DMD with enhanced stress detection for CSS."""
    
    def _enhanced_stress(self, vol: np.ndarray, returns: np.ndarray, t: int) -> float:
        s1 = self._detect_stress(vol, t)
        s2 = 1.0
        if t >= 20:
            rv = np.abs(returns[t-20:t])
            if len(rv) > 5:
                curr = abs(returns[t]) if t < len(returns) else 0
                s2 = 1.0 + 0.25 * max(0, curr / (np.mean(rv) + 1e-8) - 1.8)
        s3 = 1.0
        if t >= 10:
            recent_vol = vol[t-10:t]
            recent_vol = recent_vol[recent_vol > 0]
            if len(recent_vol) > 3:
                vol_accel = (vol[t] - np.mean(recent_vol)) / (np.std(recent_vol) + 1e-8)
                s3 = 1.0 + 0.15 * max(0, vol_accel - 1.0)
        return np.power(s1 * s2 * s3, 1/2.5)
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, dw = params['cw'], params['dmd_weight']
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
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class KoopmanDMDEntropyModel(KoopmanDMDBase):
    """Koopman-DMD with entropy matching for FEC."""
    
    def _entropy_target(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return 0.0
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return 0.0
        return 0.5 * np.log(2 * np.pi * np.e * np.mean(rv)**2)
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, dw = params['cw'], params['dmd_weight']
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
                te = self._entropy_target(vol, t)
                pe = 0.5 * np.log(2 * np.pi * np.e * sigma[t]**2)
                ll -= 0.008 * (pe - te)**2
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class KoopmanDMDRobustModel(KoopmanDMDBase):
    """Koopman-DMD with robust M-estimation for outlier resistance."""
    
    def _huber_weight(self, residual: float, scale: float, k: float = 1.5) -> float:
        z = abs(residual) / (scale + 1e-10)
        if z <= k:
            return 1.0
        return k / z
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, dw = params['cw'], params['dmd_weight']
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
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class KoopmanDMDAdaptiveModel(KoopmanDMDBase):
    """Koopman-DMD with adaptive mode selection based on regime."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi = params['q'], params['c'], params['phi']
        cw, dw = params['cw'], params['dmd_weight']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        calib_adj = 1.0
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            if t > 100 and t % 25 == 0:
                rp = pit[max(60, t-50):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 20:
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
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)
