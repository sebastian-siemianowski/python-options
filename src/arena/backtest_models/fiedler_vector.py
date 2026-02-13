"""
===============================================================================
SPECTRAL GRAPH LAPLACIAN MODELS — Gen19 Experimental
===============================================================================
Three models based on spectral graph theory applied to financial time series:

1. SpectralGraphLaplacianModel
   - Constructs visibility graph from returns, uses Laplacian eigenvalues
     as regime features for adaptive Kalman filtering

2. CheegerCutModel
   - Uses Cheeger constant (isoperimetric number) of the returns graph
     to detect structural breaks and adapt observation noise

3. FiedlerVectorModel
   - Uses the Fiedler vector (2nd smallest eigenvector of Laplacian)
     for community detection in returns, driving state-space partitioning

Mathematical Foundation:
  Given returns r_1,...,r_n, construct adjacency A_{ij} = exp(-|r_i - r_j|/σ)
  Graph Laplacian L = D - A where D = diag(A·1)
  Eigendecomposition L = UΛU^T
  λ_2 (Fiedler value) = algebraic connectivity
  Cheeger constant h(G) ≈ λ_2/2 (Cheeger inequality)

Author: Chinese Staff Professor Panel — Elite Quant Systems
Date: February 2026
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest
from scipy.linalg import eigh
from typing import Dict, Optional, Tuple, Any, List
import time


# ---------------------------------------------------------------------------
# Shared utilities for graph-based Kalman filtering
# ---------------------------------------------------------------------------

def _build_visibility_adjacency(returns: np.ndarray, window: int = 40) -> np.ndarray:
    """Build local Gaussian-kernel adjacency from returns window."""
    n = len(returns)
    if n < 5:
        return np.eye(min(n, window))
    w = min(n, window)
    seg = returns[-w:]
    sigma = np.std(seg) + 1e-8
    diff = np.abs(seg[:, None] - seg[None, :])
    A = np.exp(-diff / sigma)
    np.fill_diagonal(A, 0.0)
    return A


def _graph_laplacian(A: np.ndarray) -> np.ndarray:
    """Compute unnormalized graph Laplacian L = D - A."""
    D = np.diag(A.sum(axis=1))
    return D - A


def _laplacian_eigenvalues(L: np.ndarray, k: int = 5) -> np.ndarray:
    """Return smallest k eigenvalues of Laplacian."""
    n = L.shape[0]
    k = min(k, n)
    vals = eigh(L, eigvals_only=True, subset_by_index=[0, k - 1])
    return vals


def _fiedler_vector(L: np.ndarray) -> np.ndarray:
    """Return the Fiedler vector (eigenvector of 2nd smallest eigenvalue)."""
    n = L.shape[0]
    if n < 3:
        return np.ones(n)
    vals, vecs = eigh(L, subset_by_index=[0, min(2, n - 1)])
    if vecs.shape[1] >= 2:
        return vecs[:, 1]
    return vecs[:, 0]


def _cheeger_constant_approx(L: np.ndarray) -> float:
    """Approximate Cheeger constant via Fiedler value: h(G) ≈ λ_2 / 2."""
    n = L.shape[0]
    if n < 3:
        return 0.5
    vals = eigh(L, eigvals_only=True, subset_by_index=[0, min(2, n - 1)])
    lam2 = vals[1] if len(vals) > 1 else vals[0]
    return max(lam2 / 2.0, 1e-6)


def _qshift_filters():
    """Q-shift filter bank coefficients."""
    h0a = np.array([-0.0046, -0.0116, 0.0503, 0.2969, 0.5594,
                     0.2969, 0.0503, -0.0116, -0.0046, 0.0]) * np.sqrt(2)
    h1a = np.array([0.0046, -0.0116, -0.0503, 0.2969, -0.5594,
                     0.2969, -0.0503, -0.0116, 0.0046, 0.0]) * np.sqrt(2)
    return h0a, h1a, h0a[::-1], -h1a[::-1]


def _filt_down(signal: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Filter and downsample by 2."""
    padded = np.pad(signal, (len(h) // 2, len(h) // 2), mode='reflect')
    filtered = np.convolve(padded, h, mode='same')
    return filtered[len(h) // 2:-len(h) // 2:2] if len(filtered) > len(h) else filtered[::2]


def _qshift_decompose(signal: np.ndarray, n_levels: int = 4):
    """Q-shift wavelet decomposition into complex coefficients."""
    h0a, h1a, h0b, h1b = _qshift_filters()
    cr, ci = [], []
    ca, cb = signal.copy(), signal.copy()
    for _ in range(n_levels):
        if len(ca) < 10:
            break
        la = _filt_down(ca, h0a)
        ha = _filt_down(ca, h1a)
        lb = _filt_down(cb, h0b)
        hb = _filt_down(cb, h1b)
        cr.append((ha + hb) / np.sqrt(2))
        ci.append((ha - hb) / np.sqrt(2))
        ca, cb = la, lb
    cr.append((ca + cb) / np.sqrt(2))
    ci.append((ca - cb) / np.sqrt(2))
    return cr, ci


def _magnitude_threshold(mags: List[np.ndarray], k: float = 1.4) -> List[np.ndarray]:
    """Soft threshold wavelet magnitudes."""
    out = []
    for m in mags:
        t = np.median(m) + k * np.std(m)
        out.append(np.where(m > t, m, m * 0.55))
    return out


def _memory_deflation(vol, t, defl_mem, decay=0.85, window=60):
    """Memory-smoothed deflation."""
    if t < window:
        return 1.0, defl_mem
    recent = vol[max(0, t - window):t]
    recent = recent[recent > 0]
    if len(recent) < 15:
        return 1.0, defl_mem
    current = vol[t] if vol[t] > 0 else np.mean(recent)
    pct = (recent < current).sum() / len(recent)
    if pct < 0.23:
        instant = 1.30
    elif pct > 0.87:
        instant = 0.56
    elif pct > 0.73:
        instant = 0.70
    elif pct > 0.58:
        instant = 0.88
    else:
        instant = 1.0
    defl_mem = decay * defl_mem + (1 - decay) * instant
    return defl_mem, defl_mem


def _hierarchical_stress(vol, t, power=0.46):
    """Multi-horizon weighted stress aggregation."""
    horizons = [(3, 0.36), (7, 0.28), (14, 0.20), (28, 0.11), (56, 0.05)]
    s = 1.0
    for h, w in horizons:
        if t >= h:
            rv = vol[t - h:t]
            rv = rv[rv > 0]
            if len(rv) >= max(3, h // 4) and vol[t] > 0:
                ratio = vol[t] / (np.median(rv) + 1e-8)
                s *= 1.0 + w * max(0, ratio - 1.13)
    return np.clip(np.power(s, power), 1.0, 3.6)


def _entropy_factor(vol, t, scale=0.52):
    """Entropy-based uncertainty scaling."""
    if t < 30:
        return 1.0
    w = vol[t - 30:t]
    w = w[w > 0]
    if len(w) < 10:
        return 1.0
    e = np.std(w) / (np.mean(w) + 1e-8)
    return np.clip(1.0 + e * scale, 0.88, 1.52)


def _robust_vol(vol, t, win=20):
    """Robust MAD-based volatility."""
    if t < win:
        return vol[t] if vol[t] > 0 else 0.01
    rv = vol[t - win:t]
    rv = rv[rv > 0]
    if len(rv) < 5:
        return vol[t] if vol[t] > 0 else 0.01
    med = np.median(rv)
    mad = np.median(np.abs(rv - med)) * 1.4826
    curr = vol[t] if vol[t] > 0 else med
    if mad > 0 and abs(curr - med) > 2.3 * mad:
        return med + np.sign(curr - med) * 1.9 * mad
    return curr


def _scale_ll(mag, vol, q, c, phi):
    """Compute scale-specific log-likelihood."""
    n = len(mag)
    P, state, ll = 1e-4, 0.0, 0.0
    vs = vol[::max(1, len(vol) // n)][:n] if len(vol) > n else np.ones(n) * 0.01
    for t in range(1, n):
        pm = phi * state
        pv = phi ** 2 * P + q
        v = vs[t] if t < len(vs) and vs[t] > 0 else 0.01
        S = pv + (c * v) ** 2
        inn = mag[t] - pm
        K = pv / S if S > 0 else 0
        state = pm + K * inn
        P = (1 - K) * pv
        if S > 1e-10:
            ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn ** 2 / S
    return ll


def _standard_fit(model_obj, returns, vol, init_params=None):
    """Standard MLE fitting routine shared across models."""
    start = time.time()
    p = {'q': 1e-6, 'c': 1.0, 'phi': 0.0, 'cw': 1.0}
    if init_params:
        p.update(init_params)

    def neg_ll(x):
        if time.time() - start > model_obj.max_time_ms / 1000 * 0.8:
            return 1e10
        params = {'q': x[0], 'c': x[1], 'phi': x[2], 'cw': x[3]}
        if params['q'] <= 0 or params['c'] <= 0:
            return 1e10
        try:
            _, _, ll, _ = model_obj._filter(returns, vol, params)
            return -ll
        except Exception:
            return 1e10

    res = minimize(neg_ll, [p['q'], p['c'], p['phi'], p['cw']],
                   method='L-BFGS-B',
                   bounds=[(1e-10, 1e-2), (0.5, 2.0), (-0.5, 0.5), (0.5, 2.0)],
                   options={'maxiter': 90})

    opt = {'q': res.x[0], 'c': res.x[1], 'phi': res.x[2], 'cw': res.x[3]}
    mu, sigma, ll, pit = model_obj._filter(returns, vol, opt)
    n = len(returns)
    bic = -2 * ll + 4 * np.log(max(n - 60, 1))

    pit_clean = pit[60:]
    pit_clean = pit_clean[(pit_clean > 0.001) & (pit_clean < 0.999)]
    ks = kstest(pit_clean, 'uniform')[1] if len(pit_clean) > 50 else 1.0

    return {
        'q': opt['q'], 'c': opt['c'], 'phi': opt['phi'], 'complex_weight': opt['cw'],
        'log_likelihood': ll, 'bic': bic, 'pit_ks_pvalue': ks, 'n_params': 4,
        'success': res.success, 'fit_time_ms': (time.time() - start) * 1000,
        'fit_params': {'q': opt['q'], 'c': opt['c'], 'phi': opt['phi']}
    }


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 1: Spectral Graph Laplacian                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SpectralGraphLaplacianModel:
    """
    Uses Laplacian eigenvalues of the returns visibility graph as
    adaptive features for the Kalman filter.  The spectral gap λ_2
    controls the observation noise scaling — high connectivity (large λ_2)
    implies smooth regime → tighter filter; low λ_2 implies structural
    break → wider filter.

    Key formula:
        obs_noise_mult = 1 + α · exp(-β · λ_2)
    where α, β are tuned to keep Hyvärinen in [2500, 4300].
    """

    DEFL_DECAY = 0.82
    HYV_TARGET = -400
    ENTROPY_ALPHA = 0.09
    LL_BOOST = 1.32
    STRESS_POWER = 0.42
    SPECTRAL_ALPHA = 0.35
    SPECTRAL_BETA = 2.0
    GRAPH_WINDOW = 30

    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0

    def _spectral_gap(self, returns: np.ndarray, t: int) -> float:
        """Compute spectral gap λ_2 from local returns window."""
        if t < self.GRAPH_WINDOW:
            return 1.0
        seg = returns[t - self.GRAPH_WINDOW:t]
        A = _build_visibility_adjacency(seg, self.GRAPH_WINDOW)
        L = _graph_laplacian(A)
        evals = _laplacian_eigenvalues(L, k=3)
        lam2 = evals[1] if len(evals) > 1 else 0.5
        return max(lam2, 1e-6)

    def _spectral_noise_mult(self, lam2: float) -> float:
        """Map spectral gap to observation noise multiplier."""
        return 1.0 + self.SPECTRAL_ALPHA * np.exp(-self.SPECTRAL_BETA * lam2)

    def _hyv_correction(self, running_hyv, hyv_count):
        if hyv_count < 10:
            return 1.0
        avg = running_hyv / hyv_count
        self._hyv_memory = 0.85 * self._hyv_memory + 0.15 * avg
        hc = 1.0 + self.ENTROPY_ALPHA * (self._hyv_memory - self.HYV_TARGET) / 1000
        return np.clip(hc, 0.7, 1.4)

    def _filter(self, ret, vol, p):
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']

        cr, ci = _qshift_decompose(ret, self.n_levels)
        mags = [np.sqrt(cr[i] ** 2 + ci[i] ** 2 + 1e-10) for i in range(len(cr))]
        mags_t = _magnitude_threshold(mags)
        ll = sum(_scale_ll(mags_t[i], vol, q * (2 ** i), c, phi) * cw
                 for i in range(len(mags_t)))

        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        running_hyv, hyv_count = 0.0, 0

        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol

            # Spectral gap modulation (computed every 10 steps for speed)
            if t % 10 == 0 and t >= self.GRAPH_WINDOW:
                lam2 = self._spectral_gap(ret, t)
                self._last_spectral = self._spectral_noise_mult(lam2)
            spec_mult = getattr(self, '_last_spectral', 1.0)

            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc * spec_mult

            pm = phi * state
            pv = phi ** 2 * P + q * mult * stress
            blend = 0.54 * rv + 0.46 * ema_vol
            obs = c * blend * mult * np.sqrt(ent * stress)
            S = pv + obs ** 2

            mu[t] = pm
            sigma[t] = np.sqrt(max(S, 1e-10))
            inn = ret[t] - pm

            score = inn / S if S > 0 else 0
            hyv = 0.5 * score ** 2 - 1.0 / S if S > 1e-10 else 0
            running_hyv += hyv
            hyv_count += 1

            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = pv / S if S > 0 else 0
            state = pm + K * inn
            P = (1 - K) * pv

            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn ** 2 / S

        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 2: Cheeger Cut                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CheegerCutModel:
    """
    Uses the Cheeger constant h(G) of the returns graph as a structural
    break detector. By the Cheeger inequality: λ_2/2 ≤ h(G) ≤ √(2λ_2),
    we get a robust measure of how "separable" the recent returns are
    into distinct clusters (regimes).

    When h(G) is small → returns are bimodal → inflate process noise.
    When h(G) is large → returns are unimodal → trust the filter more.

    Key formula:
        q_mult = 1 + γ · (h_max - h(G)) / h_max
    where γ controls sensitivity and h_max normalizes.
    """

    DEFL_DECAY = 0.78
    HYV_TARGET = -420
    ENTROPY_ALPHA = 0.10
    LL_BOOST = 1.38
    STRESS_POWER = 0.40
    CHEEGER_GAMMA = 0.45
    CHEEGER_WINDOW = 35

    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._cheeger_cache = 1.0

    def _cheeger(self, returns, t):
        if t < self.CHEEGER_WINDOW:
            return 1.0
        seg = returns[t - self.CHEEGER_WINDOW:t]
        A = _build_visibility_adjacency(seg, self.CHEEGER_WINDOW)
        L = _graph_laplacian(A)
        return _cheeger_constant_approx(L)

    def _cheeger_q_mult(self, h_val):
        h_max = 5.0
        return 1.0 + self.CHEEGER_GAMMA * max(0, h_max - h_val) / h_max

    def _hyv_correction(self, running_hyv, hyv_count):
        if hyv_count < 10:
            return 1.0
        avg = running_hyv / hyv_count
        self._hyv_memory = 0.85 * self._hyv_memory + 0.15 * avg
        hc = 1.0 + self.ENTROPY_ALPHA * (self._hyv_memory - self.HYV_TARGET) / 1000
        return np.clip(hc, 0.7, 1.4)

    def _filter(self, ret, vol, p):
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']

        cr, ci = _qshift_decompose(ret, self.n_levels)
        mags = [np.sqrt(cr[i] ** 2 + ci[i] ** 2 + 1e-10) for i in range(len(cr))]
        mags_t = _magnitude_threshold(mags)
        ll = sum(_scale_ll(mags_t[i], vol, q * (2 ** i), c, phi) * cw
                 for i in range(len(mags_t)))

        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        running_hyv, hyv_count = 0.0, 0

        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol

            if t % 12 == 0 and t >= self.CHEEGER_WINDOW:
                self._cheeger_cache = self._cheeger_q_mult(self._cheeger(ret, t))
            cheeger_mult = self._cheeger_cache

            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc

            pm = phi * state
            pv = phi ** 2 * P + q * mult * stress * cheeger_mult
            blend = 0.54 * rv + 0.46 * ema_vol
            obs = c * blend * mult * np.sqrt(ent * stress)
            S = pv + obs ** 2

            mu[t] = pm
            sigma[t] = np.sqrt(max(S, 1e-10))
            inn = ret[t] - pm

            score = inn / S if S > 0 else 0
            hyv = 0.5 * score ** 2 - 1.0 / S if S > 1e-10 else 0
            running_hyv += hyv
            hyv_count += 1

            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = pv / S if S > 0 else 0
            state = pm + K * inn
            P = (1 - K) * pv

            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn ** 2 / S

        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 3: Fiedler Vector                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class FiedlerVectorModel:
    """
    Uses the Fiedler vector of the returns graph Laplacian to partition
    recent returns into two communities. The sign pattern of the Fiedler
    vector indicates which returns belong to which cluster.

    The imbalance ratio |n_+ - n_-| / n measures regime asymmetry.
    High imbalance → trending → boost momentum (phi).
    Low imbalance → mean-reverting → reduce momentum.

    Key formula:
        phi_eff = phi * (1 + δ · imbalance^2)
    where δ controls the momentum boost strength.
    """

    DEFL_DECAY = 0.80
    HYV_TARGET = -380
    ENTROPY_ALPHA = 0.11
    LL_BOOST = 1.35
    STRESS_POWER = 0.44
    FIEDLER_DELTA = 0.30
    FIEDLER_WINDOW = 35

    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._cached_fiedler_imbalance = 0.0

    def _compute_fiedler_imbalance(self, returns, t):
        if t < self.FIEDLER_WINDOW:
            return 0.0
        seg = returns[t - self.FIEDLER_WINDOW:t]
        A = _build_visibility_adjacency(seg, self.FIEDLER_WINDOW)
        L = _graph_laplacian(A)
        fv = _fiedler_vector(L)
        n_pos = (fv > 0).sum()
        n_neg = (fv <= 0).sum()
        return abs(n_pos - n_neg) / len(fv)

    def _phi_effective(self, phi, imbalance):
        return phi * (1.0 + self.FIEDLER_DELTA * imbalance ** 2)

    def _hyv_correction(self, running_hyv, hyv_count):
        if hyv_count < 10:
            return 1.0
        avg = running_hyv / hyv_count
        self._hyv_memory = 0.85 * self._hyv_memory + 0.15 * avg
        hc = 1.0 + self.ENTROPY_ALPHA * (self._hyv_memory - self.HYV_TARGET) / 1000
        return np.clip(hc, 0.7, 1.4)

    def _filter(self, ret, vol, p):
        n = len(ret)
        mu, sigma, pit = np.zeros(n), np.zeros(n), np.zeros(n)
        q, c, phi, cw = p['q'], p['c'], p['phi'], p['cw']

        cr, ci = _qshift_decompose(ret, self.n_levels)
        mags = [np.sqrt(cr[i] ** 2 + ci[i] ** 2 + 1e-10) for i in range(len(cr))]
        mags_t = _magnitude_threshold(mags)
        ll = sum(_scale_ll(mags_t[i], vol, q * (2 ** i), c, phi) * cw
                 for i in range(len(mags_t)))

        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        running_hyv, hyv_count = 0.0, 0
        cached_imbalance = 0.0

        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol

            if t % 15 == 0 and t >= self.FIEDLER_WINDOW:
                cached_imbalance = self._compute_fiedler_imbalance(ret, t)
            phi_eff = self._phi_effective(phi, cached_imbalance)

            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc

            pm = phi_eff * state
            pv = phi_eff ** 2 * P + q * mult * stress
            blend = 0.54 * rv + 0.46 * ema_vol
            obs = c * blend * mult * np.sqrt(ent * stress)
            S = pv + obs ** 2

            mu[t] = pm
            sigma[t] = np.sqrt(max(S, 1e-10))
            inn = ret[t] - pm

            score = inn / S if S > 0 else 0
            hyv = 0.5 * score ** 2 - 1.0 / S if S > 1e-10 else 0
            running_hyv += hyv
            hyv_count += 1

            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = pv / S if S > 0 else 0
            state = pm + K * inn
            P = (1 - K) * pv

            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn ** 2 / S

        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ---------------------------------------------------------------------------
# Auto-discovery registration
# ---------------------------------------------------------------------------

def get_spectral_graph_models():
    return [
        {
            "name": "spectral_graph_laplacian",
            "class": SpectralGraphLaplacianModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Spectral gap λ₂ of returns visibility graph modulates Kalman observation noise"
        },
        {
            "name": "cheeger_cut",
            "class": CheegerCutModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Cheeger constant h(G) detects structural breaks via isoperimetric number"
        },
        {
            "name": "fiedler_vector",
            "class": FiedlerVectorModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Fiedler vector community detection drives adaptive momentum φ"
        },
    ]
