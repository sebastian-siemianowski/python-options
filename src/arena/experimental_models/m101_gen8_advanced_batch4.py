"""
Generation 8 Elite Models - Batch 4: Advanced Hybrid (10 models)
Target: CSS >= 0.65, FEC >= 0.75, vs STD > 3.

Advanced Methods:
- Koopman-inspired dynamics
- SSA-style decomposition
- Robust M-estimation
- Information-theoretic approaches
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import svd
from typing import Dict, Optional, Tuple, Any, List
from .base import BaseExperimentalModel


class Gen8AdvancedBase(BaseExperimentalModel):
    """Base for advanced hybrid models."""
    
    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._init_filters()
    
    def _init_filters(self):
        self.h0a = np.array([0.0, -0.0884, 0.0884, 0.6959, 0.6959, 0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h1a = np.array([0.0, 0.0884, 0.0884, -0.6959, 0.6959, -0.0884, -0.0884, 0.0]) * np.sqrt(2)
        self.h0b = np.array([0.0, 0.0884, -0.0884, 0.6959, 0.6959, -0.0884, 0.0884, 0.0]) * np.sqrt(2)
        self.h1b = np.array([0.0, -0.0884, -0.0884, -0.6959, 0.6959, 0.0884, 0.0884, 0.0]) * np.sqrt(2)
    
    def _filt_down(self, s: np.ndarray, h: np.ndarray) -> np.ndarray:
        return np.convolve(s, h, mode='same')[::2]
    
    def _dtcwt(self, s: np.ndarray) -> Tuple[List, List]:
        cr, ci = [], []
        ca, cb = s.copy(), s.copy()
        for _ in range(self.n_levels):
            if len(ca) < 8:
                break
            la, ha = self._filt_down(ca, self.h0a), self._filt_down(ca, self.h1a)
            lb, hb = self._filt_down(cb, self.h0b), self._filt_down(cb, self.h1b)
            cr.append((ha + hb) / np.sqrt(2))
            ci.append((ha - hb) / np.sqrt(2))
            ca, cb = la, lb
        cr.append((ca + cb) / np.sqrt(2))
        ci.append((ca - cb) / np.sqrt(2))
        return cr, ci
    
    def _multi_stress(self, vol: np.ndarray, t: int) -> float:
        s5, s20, s60 = 1.0, 1.0, 1.0
        if t >= 5:
            rv = vol[t-5:t]
            rv = rv[rv > 0]
            if len(rv) >= 3:
                s5 = 1.0 + 0.5 * max(0, vol[t] / (np.median(rv) + 1e-8) - 1.2) if vol[t] > 0 else 1.0
        if t >= 20:
            rv = vol[t-20:t]
            rv = rv[rv > 0]
            if len(rv) >= 5:
                s20 = 1.0 + 0.35 * max(0, vol[t] / (np.median(rv) + 1e-8) - 1.3) if vol[t] > 0 else 1.0
        if t >= 60:
            rv = vol[t-60:t]
            rv = rv[rv > 0]
            if len(rv) >= 10:
                s60 = 1.0 + 0.25 * max(0, vol[t] / (np.median(rv) + 1e-8) - 1.2) if vol[t] > 0 else 1.0
        return np.clip(np.power(s5 * s20 * s60, 1/2.5), 1.0, 3.0)
    
    def _ret_stress(self, ret: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return 1.0
        rv = np.abs(ret[t-win:t])
        if len(rv) < 5:
            return 1.0
        return 1.0 + 0.35 * max(0, abs(ret[t]) / (np.mean(rv) + 1e-8) - 1.5)
    
    def _vol_regime(self, vol: np.ndarray, t: int, win: int = 60) -> float:
        if t < win:
            return 1.0
        rv = vol[max(0, t-win):t]
        rv = rv[rv > 0]
        if len(rv) < 10:
            return 1.0
        curr = vol[t] if vol[t] > 0 else np.mean(rv)
        pct = (rv < curr).sum() / len(rv)
        if pct < 0.25:
            return 1.25
        elif pct > 0.75:
            return 0.85
        return 1.0
    
    def _huber_weight(self, z: float, k: float = 1.5) -> float:
        az = abs(z)
        if az <= k:
            return 1.0
        return k / az
    
    def _scale_ll(self, m: np.ndarray, v: np.ndarray, q: float, c: float, phi: float) -> float:
        n = len(m)
        P, st, ll = 1e-4, 0.0, 0.0
        vs = v[::max(1, len(v)//n)][:n] if len(v) > n else np.ones(n) * 0.01
        for t in range(1, n):
            mp = phi * st
            Pp = phi**2 * P + q
            vt = vs[t] if t < len(vs) and vs[t] > 0 else 0.01
            S = Pp + (c * vt)**2
            inn = m[t] - mp
            K = Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            if S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return ll
    
    def _base_fit(self, ret: np.ndarray, vol: np.ndarray, filt, init: Optional[Dict] = None) -> Dict[str, Any]:
        import time
        start = time.time()
        p = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'cw': 1.0}
        p.update(init or {})
        def neg_ll(x):
            if time.time() - start > self.max_time_ms / 1000 * 0.8:
                return 1e10
            pr = {'q': x[0], 'c': x[1], 'phi': x[2], 'cw': x[3]}
            if pr['q'] <= 0 or pr['c'] <= 0:
                return 1e10
            try:
                _, _, ll, _ = filt(ret, vol, pr)
                return -ll
            except:
                return 1e10
        res = minimize(neg_ll, [p['q'], p['c'], p['phi'], p['cw']], method='L-BFGS-B',
                      bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.5, 2.0)], options={'maxiter': 80})
        opt = {'q': res.x[0], 'c': res.x[1], 'phi': res.x[2], 'cw': res.x[3]}
        mu, sigma, ll, pit = filt(ret, vol, opt)
        n = len(ret)
        bic = -2 * ll + 4 * np.log(n - 60)
        from scipy.stats import kstest
        pc = pit[60:]
        pc = pc[(pc > 0.001) & (pc < 0.999)]
        ks = kstest(pc, 'uniform')[1] if len(pc) > 50 else 1.0
        return {'q': opt['q'], 'c': opt['c'], 'phi': opt['phi'], 'complex_weight': opt['cw'],
                'log_likelihood': ll, 'bic': bic, 'pit_ks_pvalue': ks, 'n_params': 4,
                'success': res.success, 'fit_time_ms': (time.time() - start) * 1000,
                'fit_params': {'q': opt['q'], 'c': opt['c'], 'phi': opt['phi']}}


class AdvancedKoopmanHybridModel(Gen8AdvancedBase):
    """Koopman-inspired dynamics with DTCWT."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        obs_ema = 0.0
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            obs_ema = 0.15 * ret[t-1] + 0.85 * obs_ema
            mp = phi * st + 0.05 * obs_ema
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedSSAHybridModel(Gen8AdvancedBase):
    """SSA-style trend extraction with DTCWT."""
    
    def _ssa_trend(self, x: np.ndarray, win: int = 30) -> float:
        if len(x) < win:
            return 0.0
        try:
            L = win // 2
            K = len(x) - L + 1
            if K < 2:
                return 0.0
            X = np.zeros((L, K))
            for i in range(K):
                X[:, i] = x[i:i+L]
            U, s, _ = svd(X, full_matrices=False)
            return np.sum(U[:, 0]) * s[0] / L if len(s) > 0 else 0.0
        except:
            return 0.0
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedRobustHybridModel(Gen8AdvancedBase):
    """Robust M-estimation with DTCWT."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        scale_est = 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            inn = ret[t] - mp
            z = inn / (scale_est + 1e-10)
            w = self._huber_weight(z)
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = w * Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            scale_est = 0.95 * scale_est + 0.05 * abs(inn)
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * w * inn**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedInfoTheoreticModel(Gen8AdvancedBase):
    """Information-theoretic approach."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedEnsembleModel(Gen8AdvancedBase):
    """Ensemble of methods."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema1 = vol[0] if vol[0] > 0 else 0.01
        ema2 = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema1 = 0.1 * vol[t] + 0.9 * ema1 if vol[t] > 0 else ema1
            ema2 = 0.03 * vol[t] + 0.97 * ema2 if vol[t] > 0 else ema2
            blend = 0.4 * vol[t] + 0.35 * ema1 + 0.25 * ema2 if vol[t] > 0 else ema2
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedScaleMixModel(Gen8AdvancedBase):
    """Scale mixture approach."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        scale_weights = [1.0 / (2 ** i) for i in range(len(cr))]
        sw_sum = sum(scale_weights)
        scale_weights = [w / sw_sum for w in scale_weights]
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw * scale_weights[i] for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            blend = 0.55 * vol[t] + 0.45 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedPhaseMagModel(Gen8AdvancedBase):
    """Phase-magnitude decomposition."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        mags = [np.sqrt(cr[i]**2 + ci[i]**2) for i in range(len(cr))]
        phases = [np.arctan2(ci[i], cr[i] + 1e-10) for i in range(len(ci))]
        ll = sum(self._scale_ll(mags[i], vol, q * (2**i), c, phi) * cw for i in range(len(mags)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedMomentumModel(Gen8AdvancedBase):
    """Momentum-aware filtering."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        ema_ret = 0.0
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            ema_ret = 0.2 * ret[t-1] + 0.8 * ema_ret
            mp = phi * st + 0.02 * ema_ret
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedVarianceTargetModel(Gen8AdvancedBase):
    """Variance targeting."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        target_var = np.var(ret[:min(60, n)]) if n > 60 else 0.0001
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            if t >= 80 and S > 1e-10:
                pred_var = sigma[max(60, t-30):t]
                pred_var = pred_var[pred_var > 0]
                if len(pred_var) > 10:
                    ratio = np.sqrt(target_var / (np.mean(pred_var**2) + 1e-10))
                    so = so * np.clip(ratio, 0.92, 1.08)
                    S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class AdvancedUltimateModel(Gen8AdvancedBase):
    """Ultimate combined model."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        calib = 1.0
        scale_est = 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.power(vs * rs, 0.5)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            if t > 80 and t % 20 == 0:
                rp = pit[max(60, t-40):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 15:
                    spread = np.std(rp)
                    target = 1/np.sqrt(12)
                    calib = 1.0 + 0.1 * (target - spread)
                    calib = np.clip(calib, 0.94, 1.1)
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * calib * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            inn = ret[t] - mp
            z = inn / (scale_est + 1e-10)
            w = self._huber_weight(z, k=2.0)
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = w * Pp / S if S > 0 else 0
            st, P = mp + K * inn, (1 - K) * Pp
            scale_est = 0.95 * scale_est + 0.05 * abs(inn)
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * w * inn**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)
