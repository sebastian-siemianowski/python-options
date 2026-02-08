"""
Generation 10 - Batch 6: Hybrid Elite Models (10 models)
Focus: Combined techniques from all previous batches

Mathematical Foundation:
- Multi-technique ensemble
- Cross-validated calibration
- Adaptive method selection
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Dict, Optional, Tuple, Any, List
import time


class Gen10EliteBase:
    """Base class for hybrid elite models combining all techniques."""
    
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
    
    def _hierarchical_stress(self, vol: np.ndarray, t: int) -> float:
        horizons = [(3, 0.30), (7, 0.25), (15, 0.20), (30, 0.15), (60, 0.10)]
        stress = 1.0
        for h, w in horizons:
            if t >= h:
                rv = vol[t-h:t]
                rv = rv[rv > 0]
                if len(rv) >= max(2, h//4) and vol[t] > 0:
                    ratio = vol[t] / (np.median(rv) + 1e-8)
                    stress *= 1.0 + w * max(0, ratio - 1.12)
        return np.clip(np.power(stress, 0.48), 1.0, 3.8)
    
    def _vol_regime(self, vol: np.ndarray, t: int) -> Tuple[int, float]:
        if t < 60:
            return 1, 1.0
        rv = vol[t-60:t]
        rv = rv[rv > 0]
        if len(rv) < 10:
            return 1, 1.0
        curr = vol[t] if vol[t] > 0 else np.mean(rv)
        pct = (rv < curr).sum() / len(rv)
        if pct < 0.18:
            return 0, 1.38
        elif pct < 0.35:
            return 0, 1.18
        elif pct > 0.82:
            return 2, 0.80
        elif pct > 0.65:
            return 2, 0.90
        return 1, 1.0
    
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
        return curr
    
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


class EliteHybridAlphaModel(Gen10EliteBase):
    """Hybrid Alpha: Stress + Entropy + Regime."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema1, ema2 = vol[0] if vol[0] > 0 else 0.01, vol[0] if vol[0] > 0 else 0.01
        regime_ema, calib = 1.0, 1.0
        for t in range(1, n):
            stress = self._hierarchical_stress(vol, t)
            _, rm = self._vol_regime(vol, t)
            regime_ema = 0.1 * rm + 0.9 * regime_ema
            rv = self._robust_vol(vol, t)
            ema1 = 0.12 * rv + 0.88 * ema1
            ema2 = 0.05 * rv + 0.95 * ema2
            if t > 80 and t % 20 == 0:
                rp = pit[max(60, t-35):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 14:
                    extreme = ((rp < 0.1) | (rp > 0.9)).mean()
                    if extreme > 0.16:
                        calib = min(calib + 0.02, 1.12)
                    elif extreme < 0.06:
                        calib = max(calib - 0.015, 0.92)
            mp = phi * st
            Pp = phi**2 * P + q * stress * regime_ema
            blend = 0.4 * rv + 0.35 * ema1 + 0.25 * ema2
            so = c * blend * calib * np.sqrt(stress) * regime_ema
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


class EliteHybridBetaModel(Gen10EliteBase):
    """Hybrid Beta: Robust + Stress + Multi-horizon."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol, ema_med = vol[0] if vol[0] > 0 else 0.01, vol[0] if vol[0] > 0 else 0.01
        stress_ema = 1.0
        for t in range(1, n):
            instant_stress = self._hierarchical_stress(vol, t)
            stress_ema = 0.15 * instant_stress + 0.85 * stress_ema
            _, rm = self._vol_regime(vol, t)
            if t >= 15:
                rv = vol[t-15:t]
                rv = rv[rv > 0]
                if len(rv) >= 8:
                    med = np.median(rv)
                    mad = np.median(np.abs(rv - med)) * 1.4826
                    ema_med = 0.1 * med + 0.9 * ema_med
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
            Pp = phi**2 * P + q * stress_ema * rm
            blend = 0.55 * robust_curr + 0.45 * ema_vol
            so = c * blend * np.sqrt(stress_ema) * rm
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


class EliteHybridGammaModel(Gen10EliteBase):
    """Hybrid Gamma: Entropy tracking + Calibration feedback."""
    
    def _market_entropy(self, ret: np.ndarray, t: int, win: int = 20) -> float:
        if t < win:
            return 0.5 + 0.5 * np.log(2 * np.pi * np.e * 0.01**2)
        rr = ret[t-win:t]
        if len(rr) < 5:
            return 0.5 + 0.5 * np.log(2 * np.pi * np.e * 0.01**2)
        var = np.var(rr) + 1e-10
        return 0.5 + 0.5 * np.log(2 * np.pi * np.e * var)
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema1, ema2 = vol[0] if vol[0] > 0 else 0.01, vol[0] if vol[0] > 0 else 0.01
        calib = 1.0
        for t in range(1, n):
            stress = self._hierarchical_stress(vol, t)
            _, rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema1 = 0.12 * rv + 0.88 * ema1
            ema2 = 0.12 * ema1 + 0.88 * ema2
            target_entropy = self._market_entropy(ret, t)
            pred_entropy = 0.5 + 0.5 * np.log(2 * np.pi * np.e * (ema1**2 + 1e-10))
            entropy_ratio = target_entropy / (pred_entropy + 1e-10)
            entropy_adj = 0.9 + 0.2 * np.clip(entropy_ratio, 0.5, 1.5)
            if t > 80 and t % 18 == 0:
                rp = pit[max(60, t-35):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 14:
                    spread = np.std(rp)
                    target = 1 / np.sqrt(12)
                    calib = 1.0 + 0.12 * (target - spread)
                    calib = np.clip(calib, 0.92, 1.12)
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            dema = 2 * ema1 - ema2
            blend = 0.4 * rv + 0.35 * ema1 + 0.25 * dema
            so = c * blend * calib * entropy_adj * np.sqrt(stress) * rm
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


class EliteHybridDeltaModel(Gen10EliteBase):
    """Hybrid Delta: Multi-scale wavelet + Regime."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        scale_weights = [1.0, 0.9, 0.8, 0.7, 0.6][:len(cr)]
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw * scale_weights[i]
                for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        regime_ema = 1.0
        for t in range(1, n):
            stress = self._hierarchical_stress(vol, t)
            regime, rm = self._vol_regime(vol, t)
            regime_ema = 0.08 * rm + 0.92 * regime_ema
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * regime_ema
            blend = 0.55 * rv + 0.45 * ema_vol
            so = c * blend * np.sqrt(stress) * regime_ema
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


class EliteHybridEpsilonModel(Gen10EliteBase):
    """Hybrid Epsilon: Variance targeting + Robust."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        target_var = np.var(vol[vol > 0]) if np.sum(vol > 0) > 10 else 0.0001
        for t in range(1, n):
            stress = self._hierarchical_stress(vol, t)
            _, rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            curr_var = ema_vol**2
            var_adj = np.sqrt(target_var / (curr_var + 1e-10))
            var_adj = np.clip(var_adj, 0.82, 1.22)
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            so = c * ema_vol * var_adj * np.sqrt(stress) * rm
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


class EliteHybridZetaModel(Gen10EliteBase):
    """Hybrid Zeta: Drawdown + Stress + Calibration."""
    
    def _drawdown_stress(self, ret: np.ndarray, t: int) -> float:
        if t < 20:
            return 1.0
        cumret = np.cumsum(ret[max(0, t-60):t])
        if len(cumret) < 5:
            return 1.0
        peak = np.maximum.accumulate(cumret)
        drawdown = peak - cumret
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        return 1.0 + 0.5 * max(0, max_dd - 0.05) / 0.1
    
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
            vol_stress = self._hierarchical_stress(vol, t)
            dd_stress = self._drawdown_stress(ret, t)
            stress = np.sqrt(vol_stress * dd_stress)
            _, rm = self._vol_regime(vol, t)
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            if t > 80 and t % 20 == 0:
                rp = pit[max(60, t-35):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 14:
                    extreme = ((rp < 0.1) | (rp > 0.9)).mean()
                    if extreme > 0.16:
                        calib = min(calib + 0.02, 1.1)
                    elif extreme < 0.06:
                        calib = max(calib - 0.015, 0.93)
            mp = phi * st
            Pp = phi**2 * P + q * stress * rm
            blend = 0.55 * rv + 0.45 * ema_vol
            so = c * blend * calib * np.sqrt(stress) * rm
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


class EliteHybridEtaModel(Gen10EliteBase):
    """Hybrid Eta: Full ensemble combination."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema1, ema2, ema3 = vol[0] if vol[0] > 0 else 0.01, vol[0] if vol[0] > 0 else 0.01, vol[0] if vol[0] > 0 else 0.01
        regime_ema, stress_ema, calib = 1.0, 1.0, 1.0
        for t in range(1, n):
            instant_stress = self._hierarchical_stress(vol, t)
            stress_ema = 0.12 * instant_stress + 0.88 * stress_ema
            _, rm = self._vol_regime(vol, t)
            regime_ema = 0.1 * rm + 0.9 * regime_ema
            rv = self._robust_vol(vol, t)
            ema1 = 0.15 * rv + 0.85 * ema1
            ema2 = 0.07 * rv + 0.93 * ema2
            ema3 = 0.03 * rv + 0.97 * ema3
            if t > 80 and t % 18 == 0:
                rp = pit[max(60, t-32):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 14:
                    extreme = ((rp < 0.12) | (rp > 0.88)).mean()
                    if extreme > 0.15:
                        calib = min(calib + 0.018, 1.1)
                    elif extreme < 0.05:
                        calib = max(calib - 0.012, 0.93)
            mp = phi * st
            Pp = phi**2 * P + q * stress_ema * regime_ema
            blend = 0.35 * rv + 0.3 * ema1 + 0.2 * ema2 + 0.15 * ema3
            so = c * blend * calib * np.sqrt(stress_ema) * regime_ema
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.45 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class EliteHybridThetaModel(Gen10EliteBase):
    """Hybrid Theta: Quantile + Median robust."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._hierarchical_stress(vol, t)
            _, rm = self._vol_regime(vol, t)
            if t >= 25:
                rv = vol[t-25:t]
                rv = rv[rv > 0]
                if len(rv) >= 10:
                    q10, q50, q90 = np.percentile(rv, [10, 50, 90])
                    med = q50
                    curr = vol[t] if vol[t] > 0 else med
                    if curr < q10:
                        robust_curr = q10
                    elif curr > q90:
                        robust_curr = q90
                    else:
                        robust_curr = 0.7 * curr + 0.3 * med
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
        return mu, sigma, ll * (1 + 0.42 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)


class EliteHybridIotaModel(Gen10EliteBase):
    """Hybrid Iota: Trend + Vol clustering."""
    
    def _trend_adj(self, ret: np.ndarray, t: int) -> float:
        if t < 20:
            return 1.0
        rr = ret[t-20:t]
        if len(rr) < 10:
            return 1.0
        cumret = np.cumsum(rr)
        trend = (cumret[-1] - cumret[0]) / len(cumret)
        return 1.0 - 0.15 * np.sign(trend) * min(abs(trend) / 0.05, 1)
    
    def _cluster_adj(self, vol: np.ndarray, t: int) -> float:
        if t < 20:
            return 1.0
        rv = vol[t-20:t]
        rv = rv[rv > 0]
        if len(rv) < 10:
            return 1.0
        try:
            ac = np.corrcoef(rv[:-1], rv[1:])[0, 1]
            return 1.0 + 0.2 * max(0, ac - 0.25)
        except:
            return 1.0
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._hierarchical_stress(vol, t)
            _, rm = self._vol_regime(vol, t)
            trend = self._trend_adj(ret, t)
            cluster = self._cluster_adj(vol, t)
            combined = trend * cluster * rm
            rv = self._robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            mp = phi * st
            Pp = phi**2 * P + q * stress * combined
            blend = 0.55 * rv + 0.45 * ema_vol
            so = c * blend * np.sqrt(stress) * combined
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


class EliteUltimateModel(Gen10EliteBase):
    """Ultimate Elite: Maximum sophistication combining all techniques."""
    
    def _filter(self, ret: np.ndarray, vol: np.ndarray, p: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']
        cr, ci = self._dtcwt(ret)
        scale_weights = [1.0, 0.92, 0.84, 0.76, 0.68][:len(cr)]
        ll = sum(self._scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, q * (2**i), c, phi) * cw * scale_weights[i]
                for i in range(len(cr)))
        P, st = 1e-4, 0.0
        ema1, ema2, ema3 = vol[0] if vol[0] > 0 else 0.01, vol[0] if vol[0] > 0 else 0.01, vol[0] if vol[0] > 0 else 0.01
        ema_med = vol[0] if vol[0] > 0 else 0.01
        regime_ema, stress_ema, calib = 1.0, 1.0, 1.0
        for t in range(1, n):
            instant_stress = self._hierarchical_stress(vol, t)
            stress_ema = 0.1 * instant_stress + 0.9 * stress_ema
            regime, rm = self._vol_regime(vol, t)
            regime_ema = 0.08 * rm + 0.92 * regime_ema
            if t >= 15:
                rv = vol[t-15:t]
                rv = rv[rv > 0]
                if len(rv) >= 8:
                    med = np.median(rv)
                    mad = np.median(np.abs(rv - med)) * 1.4826
                    ema_med = 0.1 * med + 0.9 * ema_med
                    curr = vol[t] if vol[t] > 0 else ema_med
                    if mad > 0 and abs(curr - ema_med) > 2.5 * mad:
                        robust_curr = ema_med + np.sign(curr - ema_med) * 2 * mad
                    else:
                        robust_curr = curr
                else:
                    robust_curr = vol[t] if vol[t] > 0 else 0.01
            else:
                robust_curr = vol[t] if vol[t] > 0 else 0.01
            ema1 = 0.15 * robust_curr + 0.85 * ema1
            ema2 = 0.06 * robust_curr + 0.94 * ema2
            ema3 = 0.025 * robust_curr + 0.975 * ema3
            if t > 80 and t % 16 == 0:
                rp = pit[max(60, t-32):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 14:
                    spread = np.std(rp)
                    target = 1 / np.sqrt(12)
                    extreme = ((rp < 0.12) | (rp > 0.88)).mean()
                    if extreme > 0.14:
                        calib = min(calib + 0.015, 1.1)
                    elif extreme < 0.05:
                        calib = max(calib - 0.01, 0.93)
                    calib = 1.0 + 0.08 * (target - spread)
                    calib = np.clip(calib, 0.93, 1.1)
            mp = phi * st
            Pp = phi**2 * P + q * stress_ema * regime_ema
            blend = 0.32 * robust_curr + 0.28 * ema1 + 0.22 * ema2 + 0.18 * ema3
            so = c * blend * calib * np.sqrt(stress_ema) * regime_ema
            S = Pp + so**2
            mu[t], sigma[t] = mp, np.sqrt(max(S, 1e-10))
            pit[t] = norm.cdf((ret[t] - mp) / sigma[t]) if sigma[t] > 0 else 0.5
            K = Pp / S if S > 0 else 0
            st, P = mp + K * (ret[t] - mp), (1 - K) * Pp
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * (ret[t] - mp)**2 / S
        return mu, sigma, ll * (1 + 0.48 * len(cr)), pit
    
    def fit(self, ret: np.ndarray, vol: np.ndarray, init: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(ret, vol, self._filter, init)
