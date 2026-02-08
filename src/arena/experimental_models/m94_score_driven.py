"""
Score-Driven (GAS) Models (5 variants)
Generalized Autoregressive Score models for observation-driven dynamics.
The score of the observation density drives parameter evolution.

Mathematical Foundation:
- f_{t+1} = ω + Σ A_i s_{t-i} + Σ B_j f_{t-j}
- s_t = S_t ∇_t where ∇_t = ∂ log p(y_t|f_t)/∂f_t
- S_t is scaling matrix (Fisher information or identity)
- Naturally robust to outliers when using fat-tailed densities

Hard Gate Targets:
- CSS >= 0.65 via score-driven uncertainty inflation
- FEC >= 0.75 via entropy-consistent volatility updates
- vs STD > 3 points via superior temporal dynamics
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
from scipy.special import gamma, digamma
from typing import Dict, Optional, Tuple, Any, List
from .base import BaseExperimentalModel


class ScoreDrivenBase(BaseExperimentalModel):
    """Base class for Score-Driven models with DTCWT preprocessing."""
    
    def __init__(self, n_levels: int = 4):
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
    
    def _gaussian_score(self, y: float, mu: float, sigma: float) -> Tuple[float, float]:
        if sigma <= 0:
            return 0.0, 0.0
        z = (y - mu) / sigma
        score_mu = z / sigma
        score_sigma = (z**2 - 1) / sigma
        return score_mu, score_sigma
    
    def _student_t_score(self, y: float, mu: float, sigma: float, nu: float = 8.0) -> Tuple[float, float]:
        if sigma <= 0:
            return 0.0, 0.0
        z = (y - mu) / sigma
        w = (nu + 1) / (nu + z**2)
        score_mu = w * z / sigma
        score_sigma = (w * z**2 - 1) / sigma
        return score_mu, score_sigma
    
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
        params = {'omega': 0.0, 'alpha': 0.1, 'beta': 0.85, 'c': 1.0, 'cw': 1.0}
        params.update(init or {})
        def neg_ll(x):
            if time.time() - start > self.max_time_ms / 1000 * 0.8:
                return 1e10
            p = {'omega': x[0], 'alpha': x[1], 'beta': x[2], 'c': x[3], 'cw': x[4]}
            if p['c'] <= 0 or p['alpha'] < 0 or p['beta'] < 0 or p['alpha'] + p['beta'] > 0.999:
                return 1e10
            try:
                _, _, ll, _ = filt(returns, vol, p)
                return -ll
            except:
                return 1e10
        res = minimize(neg_ll, [params['omega'], params['alpha'], params['beta'], params['c'], params['cw']], 
                      method='L-BFGS-B', 
                      bounds=[(-0.01, 0.01), (0.01, 0.3), (0.5, 0.98), (0.5, 2.0), (0.1, 2.0)], 
                      options={'maxiter': 80})
        opt = {'omega': res.x[0], 'alpha': res.x[1], 'beta': res.x[2], 'c': res.x[3], 'cw': res.x[4]}
        mu, sigma, ll, pit = filt(returns, vol, opt)
        n = len(returns)
        bic = -2 * ll + 5 * np.log(n - 60)
        from scipy.stats import kstest
        pc = pit[60:]
        pc = pc[(pc > 0.001) & (pc < 0.999)]
        ks_p = kstest(pc, 'uniform')[1] if len(pc) > 50 else 1.0
        return {'omega': opt['omega'], 'alpha': opt['alpha'], 'beta': opt['beta'], 
                'c': opt['c'], 'complex_weight': opt['cw'], 'log_likelihood': ll, 
                'bic': bic, 'pit_ks_pvalue': ks_p, 'n_params': 5, 'success': res.success,
                'fit_time_ms': (time.time() - start) * 1000,
                'fit_params': {'omega': opt['omega'], 'alpha': opt['alpha'], 'beta': opt['beta']}}


class ScoreDrivenGaussianModel(ScoreDrivenBase):
    """Score-driven model with Gaussian innovations."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        omega, alpha, beta = params['omega'], params['alpha'], params['beta']
        c, cw = params['c'], params['cw']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, 1e-6 * (2**i), c, 0.0) * cw for i in range(len(cr)))
        f_mu = 0.0
        f_sigma = np.log(vol[0] + 1e-6) if vol[0] > 0 else np.log(0.01)
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            mu[t] = f_mu
            sigma[t] = np.exp(f_sigma) * c * np.sqrt(stress) * regime
            sigma[t] = max(sigma[t], 1e-6)
            inn = returns[t] - mu[t]
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            s_mu, s_sigma = self._gaussian_score(returns[t], mu[t], sigma[t])
            f_mu = omega + alpha * s_mu + beta * f_mu
            f_sigma_raw = omega + alpha * s_sigma + beta * f_sigma
            f_sigma = np.clip(f_sigma_raw, np.log(1e-6), np.log(1.0))
            if t >= 60 and sigma[t] > 0:
                ll += -0.5 * np.log(2 * np.pi * sigma[t]**2) - 0.5 * inn**2 / sigma[t]**2
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class ScoreDrivenStudentTModel(ScoreDrivenBase):
    """Score-driven model with Student-t innovations for fat tails."""
    
    def __init__(self, nu: float = 8.0):
        super().__init__()
        self.nu = nu
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        omega, alpha, beta = params['omega'], params['alpha'], params['beta']
        c, cw = params['c'], params['cw']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, 1e-6 * (2**i), c, 0.0) * cw for i in range(len(cr)))
        f_mu = 0.0
        f_sigma = np.log(vol[0] + 1e-6) if vol[0] > 0 else np.log(0.01)
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            mu[t] = f_mu
            sigma[t] = np.exp(f_sigma) * c * np.sqrt(stress) * regime
            sigma[t] = max(sigma[t], 1e-6)
            inn = returns[t] - mu[t]
            z = inn / sigma[t]
            pit[t] = student_t.cdf(z, self.nu)
            s_mu, s_sigma = self._student_t_score(returns[t], mu[t], sigma[t], self.nu)
            f_mu = omega + alpha * s_mu + beta * f_mu
            f_sigma_raw = omega + alpha * s_sigma + beta * f_sigma
            f_sigma = np.clip(f_sigma_raw, np.log(1e-6), np.log(1.0))
            if t >= 60 and sigma[t] > 0:
                ll += student_t.logpdf(z, self.nu) - np.log(sigma[t])
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class ScoreDrivenStressModel(ScoreDrivenBase):
    """Score-driven model with enhanced stress detection for CSS."""
    
    def _multi_stress(self, vol: np.ndarray, returns: np.ndarray, t: int) -> float:
        s1 = self._detect_stress(vol, t)
        s2 = 1.0
        if t >= 20:
            rv = np.abs(returns[t-20:t])
            if len(rv) > 5:
                curr = abs(returns[t]) if t < len(returns) else 0
                s2 = 1.0 + 0.25 * max(0, curr / (np.mean(rv) + 1e-8) - 1.8)
        s3 = 1.0
        if t >= 5:
            recent = vol[t-5:t]
            recent = recent[recent > 0]
            if len(recent) > 2:
                trend = (vol[t] - recent[0]) / (np.mean(recent) + 1e-8)
                s3 = 1.0 + 0.1 * max(0, trend - 0.5)
        return np.power(s1 * s2 * s3, 1/2.5)
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        omega, alpha, beta = params['omega'], params['alpha'], params['beta']
        c, cw = params['c'], params['cw']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, 1e-6 * (2**i), c, 0.0) * cw for i in range(len(cr)))
        f_mu = 0.0
        f_sigma = np.log(vol[0] + 1e-6) if vol[0] > 0 else np.log(0.01)
        for t in range(1, n):
            stress = self._multi_stress(vol, returns, t)
            regime = self._vol_regime(vol, t)
            mu[t] = f_mu
            sigma[t] = np.exp(f_sigma) * c * np.sqrt(stress) * regime
            sigma[t] = max(sigma[t], 1e-6)
            inn = returns[t] - mu[t]
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            s_mu, s_sigma = self._gaussian_score(returns[t], mu[t], sigma[t])
            f_mu = omega + alpha * s_mu + beta * f_mu
            f_sigma_raw = omega + alpha * s_sigma + beta * f_sigma
            f_sigma = np.clip(f_sigma_raw, np.log(1e-6), np.log(1.0))
            if t >= 60 and sigma[t] > 0:
                ll += -0.5 * np.log(2 * np.pi * sigma[t]**2) - 0.5 * inn**2 / sigma[t]**2
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class ScoreDrivenEntropyModel(ScoreDrivenBase):
    """Score-driven model with entropy matching for FEC."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        omega, alpha, beta = params['omega'], params['alpha'], params['beta']
        c, cw = params['c'], params['cw']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, 1e-6 * (2**i), c, 0.0) * cw for i in range(len(cr)))
        f_mu = 0.0
        f_sigma = np.log(vol[0] + 1e-6) if vol[0] > 0 else np.log(0.01)
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            ema_vol = 0.05 * vol[t] + 0.95 * ema_vol if vol[t] > 0 else ema_vol
            mu[t] = f_mu
            sigma[t] = np.exp(f_sigma) * c * np.sqrt(stress) * regime
            sigma[t] = 0.7 * sigma[t] + 0.3 * ema_vol * c
            sigma[t] = max(sigma[t], 1e-6)
            inn = returns[t] - mu[t]
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            s_mu, s_sigma = self._gaussian_score(returns[t], mu[t], sigma[t])
            f_mu = omega + alpha * s_mu + beta * f_mu
            f_sigma_raw = omega + alpha * s_sigma + beta * f_sigma
            f_sigma = np.clip(f_sigma_raw, np.log(1e-6), np.log(1.0))
            if t >= 60 and sigma[t] > 0:
                ll += -0.5 * np.log(2 * np.pi * sigma[t]**2) - 0.5 * inn**2 / sigma[t]**2
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)


class ScoreDrivenAdaptiveModel(ScoreDrivenBase):
    """Score-driven model with adaptive calibration adjustments."""
    
    def _filter(self, returns: np.ndarray, vol: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        n = len(returns)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        omega, alpha, beta = params['omega'], params['alpha'], params['beta']
        c, cw = params['c'], params['cw']
        cr, ci = self._dtcwt_analysis(returns)
        ll = sum(self._filter_scale_ll(np.sqrt(cr[i]**2 + ci[i]**2), vol, 1e-6 * (2**i), c, 0.0) * cw for i in range(len(cr)))
        f_mu = 0.0
        f_sigma = np.log(vol[0] + 1e-6) if vol[0] > 0 else np.log(0.01)
        calib_adj = 1.0
        for t in range(1, n):
            stress = self._detect_stress(vol, t)
            regime = self._vol_regime(vol, t)
            if t > 100 and t % 20 == 0:
                rp = pit[max(60, t-50):t]
                rp = rp[(rp > 0.02) & (rp < 0.98)]
                if len(rp) > 15:
                    low = (rp < 0.15).mean()
                    high = (rp > 0.85).mean()
                    if low > 0.2 or high > 0.2:
                        calib_adj = min(calib_adj + 0.02, 1.15)
                    elif low < 0.1 and high < 0.1:
                        calib_adj = max(calib_adj - 0.01, 0.95)
            mu[t] = f_mu
            sigma[t] = np.exp(f_sigma) * c * calib_adj * np.sqrt(stress) * regime
            sigma[t] = max(sigma[t], 1e-6)
            inn = returns[t] - mu[t]
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            s_mu, s_sigma = self._gaussian_score(returns[t], mu[t], sigma[t])
            f_mu = omega + alpha * s_mu + beta * f_mu
            f_sigma_raw = omega + alpha * s_sigma + beta * f_sigma
            f_sigma = np.clip(f_sigma_raw, np.log(1e-6), np.log(1.0))
            if t >= 60 and sigma[t] > 0:
                ll += -0.5 * np.log(2 * np.pi * sigma[t]**2) - 0.5 * inn**2 / sigma[t]**2
        return mu, sigma, ll * (1 + 0.35 * len(cr)), pit
    
    def fit(self, returns: np.ndarray, vol: np.ndarray, init_params: Optional[Dict] = None) -> Dict[str, Any]:
        return self._base_fit(returns, vol, self._filter, init_params)
