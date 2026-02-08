"""
Generation 8 Elite Models - Batch 3: Combined CSS+FEC (10 models)
Target: Both CSS >= 0.65 AND FEC >= 0.75 simultaneously.

Mathematical Foundation:
- Dual targeting: optimize for both calibration stability and entropy consistency
- Hybrid stress: combine volatility, return, and regime stress factors
- Adaptive smoothing: balance responsiveness with stability
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Optional, Tuple, Any, List
from .base import BaseExperimentalModel


class Gen8CombinedBase(BaseExperimentalModel):
    """Base for combined CSS+FEC models."""
    
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


class CombinedDualTargetModel(Gen8CombinedBase):
    """Dual target for CSS and FEC simultaneously."""
    
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
            ema_vol = 0.08 * vol[t] + 0.92 * ema_vol if vol[t] > 0 else ema_vol
            blend = 0.7 * vol[t] + 0.3 * ema_vol if vol[t] > 0 else ema_vol
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


class CombinedAdaptiveBalanceModel(Gen8CombinedBase):
    """Balance stress and entropy adaptively."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        calib = 1.0
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            if t > 80 and t % 20 == 0:
                rp = pit[max(60, t-40):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 15:
                    spread = np.std(rp)
                    target = 1/np.sqrt(12)
                    calib = 1.0 + 0.12 * (target - spread)
                    calib = np.clip(calib, 0.92, 1.12)
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * calib * np.sqrt(stress) * rm
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


class CombinedHybridStressModel(Gen8CombinedBase):
    """Hybrid of vol, return, and regime stress."""
    
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
            rm = self._vol_regime(vol, t)
            stress = np.power(vs * rs * (rm ** 0.5), 1/2.5)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
            blend = 0.65 * vol[t] + 0.35 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress
            so = c * blend * np.sqrt(stress)
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


class CombinedRobustBlendModel(Gen8CombinedBase):
    """Robust blending for stability."""
    
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
            ema2 = 0.1 * ema1 + 0.9 * ema2
            blend = 0.5 * vol[t] + 0.3 * ema1 + 0.2 * ema2 if vol[t] > 0 else ema2
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


class CombinedSmoothedStressModel(Gen8CombinedBase):
    """Smoothed stress for stability."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        ema_stress = 1.0
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            raw_stress = np.sqrt(vs * rs)
            ema_stress = 0.15 * raw_stress + 0.85 * ema_stress
            rm = self._vol_regime(vol, t)
            ema_vol = 0.08 * vol[t] + 0.92 * ema_vol if vol[t] > 0 else ema_vol
            blend = 0.65 * vol[t] + 0.35 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * ema_stress * rm
            so = c * blend * np.sqrt(ema_stress) * rm
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


class CombinedRegimeAwareModel(Gen8CombinedBase):
    """Full regime awareness."""
    
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
            rm = self._vol_regime(vol, t)
            regime_mult = 1.0
            if rm > 1.1:
                regime_mult = 1.15
            elif rm < 0.9:
                regime_mult = 0.95
            ema_vol = 0.06 * vol[t] + 0.94 * ema_vol if vol[t] > 0 else ema_vol
            blend = 0.6 * vol[t] + 0.4 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * vs * regime_mult
            so = c * blend * np.sqrt(vs) * regime_mult
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


class CombinedCalibConstrainedModel(Gen8CombinedBase):
    """Calibration constrained optimization."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        calib = 1.0
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rs = self._ret_stress(ret, t)
            stress = np.sqrt(vs * rs)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.08 * vol[t] + 0.92 * ema_vol if vol[t] > 0 else ema_vol
            if t > 100 and t % 25 == 0:
                rp = pit[max(60, t-50):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 20:
                    low = (rp < 0.1).mean()
                    high = (rp > 0.9).mean()
                    if low > 0.15 or high > 0.15:
                        calib = min(calib + 0.025, 1.15)
                    elif low < 0.08 and high < 0.08:
                        calib = max(calib - 0.015, 0.95)
            blend = 0.65 * vol[t] + 0.35 * ema_vol if vol[t] > 0 else ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * blend * calib * np.sqrt(stress) * rm
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


class CombinedMultiScaleModel(Gen8CombinedBase):
    """Multi-scale stress and smoothing."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema5 = vol[0] if vol[0] > 0 else 0.01
        ema20 = vol[0] if vol[0] > 0 else 0.01
        ema60 = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            vs = self._multi_stress(vol, t)
            rm = self._vol_regime(vol, t)
            ema5 = 0.2 * vol[t] + 0.8 * ema5 if vol[t] > 0 else ema5
            ema20 = 0.05 * vol[t] + 0.95 * ema20 if vol[t] > 0 else ema20
            ema60 = 0.02 * vol[t] + 0.98 * ema60 if vol[t] > 0 else ema60
            blend = 0.4 * ema5 + 0.35 * ema20 + 0.25 * ema60
            mp = phi * st
            Pp = phi**2 * P + q * vs * rm
            so = c * blend * np.sqrt(vs) * rm
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


class CombinedOptimalModel(Gen8CombinedBase):
    """Optimal balance between CSS and FEC."""
    
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
            stress = np.power(vs * rs, 0.45)
            rm = self._vol_regime(vol, t)
            ema_vol = 0.07 * vol[t] + 0.93 * ema_vol if vol[t] > 0 else ema_vol
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


class CombinedEliteModel(Gen8CombinedBase):
    """Elite combined model."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        calib = 1.0
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
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.4 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)
