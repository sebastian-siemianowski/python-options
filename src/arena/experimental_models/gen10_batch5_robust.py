"""
Generation 10 - Batch 5: Robust Estimation Models (10 models)
Focus: Outlier resistance via robust M-estimation

Mathematical Foundation:
- Huber, Tukey, and Hampel M-estimators
- Winsorized and trimmed mean estimators
- MAD and IQR based robust scale estimation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Optional, Tuple, Any, List
import time


class Gen10RobustBase:
    """Base class for robust estimation models."""
    
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
    
    def _stress(self, vol: np.ndarray, t: int) -> float:
        horizons = [(5, 0.35), (15, 0.25), (30, 0.20), (60, 0.20)]
        stress = 1.0
        for h, w in horizons:
            if t >= h:
                rv = vol[t-h:t]
                rv = rv[rv > 0]
                if len(rv) >= max(3, h//5) and vol[t] > 0:
                    ratio = vol[t] / (np.median(rv) + 1e-8)
                    stress *= 1.0 + w * max(0, ratio - 1.15)
        return np.clip(np.power(stress, 0.5), 1.0, 3.5)
    
    def _vol_regime(self, vol: np.ndarray, t: int) -> float:
        if t < 60:
            return 1.0
        rv = vol[t-60:t]
        rv = rv[rv > 0]
        if len(rv) < 10:
            return 1.0
        curr = vol[t] if vol[t] > 0 else np.mean(rv)
        pct = (rv < curr).sum() / len(rv)
        if pct < 0.2:
            return 1.35
        elif pct > 0.8:
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


class RobustHuberModel(Gen10RobustBase):
    """Huber M-estimator based robust model."""
    
    def _huber_weight(self, x: float, k: float = 1.345) -> float:
        if abs(x) <= k:
            return 1.0
        return k / abs(x)
    
    def _robust_vol(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return vol[t] if vol[t] > 0 else 0.01
        med = np.median(rv)
        mad = np.median(np.abs(rv - med)) * 1.4826 + 1e-10
        z = (rv - med) / mad
        weights = np.array([self._huber_weight(zi) for zi in z])
        weighted_mean = np.sum(rv * weights) / np.sum(weights)
        curr = vol[t] if vol[t] > 0 else weighted_mean
        return 0.7 * curr + 0.3 * weighted_mean
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.08 * rv + 0.92 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
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


class RobustTukeyModel(Gen10RobustBase):
    """Tukey's biweight M-estimator based model."""
    
    def _tukey_weight(self, x: float, c: float = 4.685) -> float:
        if abs(x) <= c:
            return (1 - (x / c)**2)**2
        return 0.0
    
    def _robust_vol(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return vol[t] if vol[t] > 0 else 0.01
        med = np.median(rv)
        mad = np.median(np.abs(rv - med)) * 1.4826 + 1e-10
        z = (rv - med) / mad
        weights = np.array([self._tukey_weight(zi) for zi in z])
        if np.sum(weights) > 0:
            weighted_mean = np.sum(rv * weights) / np.sum(weights)
        else:
            weighted_mean = med
        curr = vol[t] if vol[t] > 0 else weighted_mean
        return 0.7 * curr + 0.3 * weighted_mean
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.08 * rv + 0.92 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
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


class RobustWinsorizedModel(Gen10RobustBase):
    """Winsorized mean based robust model."""
    
    def _robust_vol(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return vol[t] if vol[t] > 0 else 0.01
        q05, q95 = np.percentile(rv, [5, 95])
        winsorized = np.clip(rv, q05, q95)
        winsor_mean = np.mean(winsorized)
        curr = vol[t] if vol[t] > 0 else winsor_mean
        return 0.7 * curr + 0.3 * winsor_mean
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
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


class RobustTrimmedModel(Gen10RobustBase):
    """Trimmed mean based robust model."""
    
    def _robust_vol(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return vol[t] if vol[t] > 0 else 0.01
        sorted_rv = np.sort(rv)
        trim = max(1, len(sorted_rv) // 10)
        trimmed_mean = np.mean(sorted_rv[trim:-trim]) if len(sorted_rv) > 2*trim else np.mean(sorted_rv)
        curr = vol[t] if vol[t] > 0 else trimmed_mean
        return 0.7 * curr + 0.3 * trimmed_mean
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
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


class RobustMADModel(Gen10RobustBase):
    """Median Absolute Deviation based robust model."""
    
    def _robust_vol(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return vol[t] if vol[t] > 0 else 0.01
        med = np.median(rv)
        mad = np.median(np.abs(rv - med)) * 1.4826
        curr = vol[t] if vol[t] > 0 else med
        if mad > 0 and abs(curr - med) > 2.5 * mad:
            return med + np.sign(curr - med) * 2 * mad
        return 0.6 * curr + 0.4 * med
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
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


class RobustIQRModel(Gen10RobustBase):
    """Interquartile Range based robust model."""
    
    def _robust_vol(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return vol[t] if vol[t] > 0 else 0.01
        q25, q50, q75 = np.percentile(rv, [25, 50, 75])
        iqr = q75 - q25
        curr = vol[t] if vol[t] > 0 else q50
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        if curr < lower or curr > upper:
            return q50
        return 0.65 * curr + 0.35 * q50
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
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


class RobustAdaptiveModel(Gen10RobustBase):
    """Adaptive robust estimation combining multiple methods."""
    
    def _robust_vol(self, vol: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return vol[t] if vol[t] > 0 else 0.01
        rv = vol[t-win:t]
        rv = rv[rv > 0]
        if len(rv) < 5:
            return vol[t] if vol[t] > 0 else 0.01
        med = np.median(rv)
        mad = np.median(np.abs(rv - med)) * 1.4826
        q25, q75 = np.percentile(rv, [25, 75])
        iqr = q75 - q25
        curr = vol[t] if vol[t] > 0 else med
        if mad > 0:
            z_mad = abs(curr - med) / mad
        else:
            z_mad = 0
        if iqr > 0:
            z_iqr = abs(curr - med) / (0.7413 * iqr)
        else:
            z_iqr = 0
        if z_mad > 3 or z_iqr > 3:
            return med
        elif z_mad > 2 or z_iqr > 2:
            return 0.5 * curr + 0.5 * med
        return 0.75 * curr + 0.25 * med
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
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


class RobustExponentialModel(Gen10RobustBase):
    """Exponentially weighted robust estimation."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        ema_med = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            if t >= 10:
                rv = vol[t-10:t]
                rv = rv[rv > 0]
                if len(rv) >= 5:
                    med = np.median(rv)
                    mad = np.median(np.abs(rv - med)) * 1.4826
                    ema_med = 0.15 * med + 0.85 * ema_med
                    curr = vol[t] if vol[t] > 0 else ema_med
                    if mad > 0 and abs(curr - ema_med) > 2.5 * mad:
                        robust_curr = ema_med + np.sign(curr - ema_med) * 2 * mad
                    else:
                        robust_curr = curr
                else:
                    robust_curr = vol[t] if vol[t] > 0 else 0.01
            else:
                robust_curr = vol[t] if vol[t] > 0 else 0.01
            ema_vol = 0.08 * robust_curr + 0.92 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * robust_curr + 0.45 * ema_vol
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


class RobustHybridModel(Gen10RobustBase):
    """Hybrid robust estimation model."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            if t >= 20:
                rv = vol[t-20:t]
                rv = rv[rv > 0]
                if len(rv) >= 10:
                    med = np.median(rv)
                    q25, q75 = np.percentile(rv, [25, 75])
                    mad = np.median(np.abs(rv - med)) * 1.4826
                    iqr = q75 - q25
                    robust_scale = min(mad, 0.7413 * iqr) if mad > 0 and iqr > 0 else (mad or 0.7413 * iqr or 0.01)
                    curr = vol[t] if vol[t] > 0 else med
                    if robust_scale > 0 and abs(curr - med) > 2.5 * robust_scale:
                        robust_curr = med + np.sign(curr - med) * 2 * robust_scale
                    else:
                        robust_curr = curr
                else:
                    robust_curr = vol[t] if vol[t] > 0 else 0.01
            else:
                robust_curr = vol[t] if vol[t] > 0 else 0.01
            ema_vol = 0.07 * robust_curr + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * robust_curr + 0.45 * ema_vol
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


class RobustCombinedEliteModel(Gen10RobustBase):
    """Combined elite robust model."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        ema_med = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._stress(vol, t)
            rm = self._vol_regime(vol, t)
            if t >= 15:
                rv = vol[t-15:t]
                rv = rv[rv > 0]
                if len(rv) >= 8:
                    med = np.median(rv)
                    mad = np.median(np.abs(rv - med)) * 1.4826
                    q05, q95 = np.percentile(rv, [5, 95])
                    ema_med = 0.12 * med + 0.88 * ema_med
                    curr = vol[t] if vol[t] > 0 else ema_med
                    curr_clipped = np.clip(curr, q05, q95)
                    if mad > 0 and abs(curr_clipped - ema_med) > 2 * mad:
                        robust_curr = ema_med + np.sign(curr_clipped - ema_med) * 1.5 * mad
                    else:
                        robust_curr = curr_clipped
                else:
                    robust_curr = vol[t] if vol[t] > 0 else 0.01
            else:
                robust_curr = vol[t] if vol[t] > 0 else 0.01
            ema_vol = 0.065 * robust_curr + 0.935 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * robust_curr + 0.45 * ema_vol
            so = c * blend * np.sqrt(stress) * rm
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.42 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)
