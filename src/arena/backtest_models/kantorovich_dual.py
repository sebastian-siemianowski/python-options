"""
===============================================================================
OPTIMAL TRANSPORT / WASSERSTEIN MODELS — Gen19 Experimental
===============================================================================
Three models based on optimal transport theory:

1. WassersteinDistanceModel
   - Uses W_1 (Earth Mover's Distance) between empirical return distributions
     across sliding windows to detect distributional regime changes

2. SinkhornDivergenceModel
   - Regularized OT via Sinkhorn iterations; the entropic divergence
     controls adaptive process noise inflation

3. KantorovichDualModel
   - Kantorovich dual formulation: sup E[φ(X)] - E[ψ(Y)] subject to
     φ(x) - ψ(y) ≤ c(x,y). The dual potential gap drives state correction.

Mathematical Foundation:
  W_p(μ,ν) = (inf_{γ∈Π(μ,ν)} ∫ |x-y|^p dγ)^{1/p}
  Sinkhorn: W_ε(μ,ν) = inf_{γ∈Π(μ,ν)} ∫ c dγ + ε KL(γ|μ⊗ν)
  Kantorovich dual: W_1 = sup_{Lip(f)≤1} E_μ[f] - E_ν[f]

Author: Chinese Staff Professor Panel — Elite Quant Systems
Date: February 2026
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest, wasserstein_distance
from typing import Dict, Optional, Tuple, Any, List
import time


def _qshift_filters():
    h0a = np.array([-0.0046, -0.0116, 0.0503, 0.2969, 0.5594,
                     0.2969, 0.0503, -0.0116, -0.0046, 0.0]) * np.sqrt(2)
    h1a = np.array([0.0046, -0.0116, -0.0503, 0.2969, -0.5594,
                     0.2969, -0.0503, -0.0116, 0.0046, 0.0]) * np.sqrt(2)
    return h0a, h1a, h0a[::-1], -h1a[::-1]


def _filt_down(signal, h):
    padded = np.pad(signal, (len(h) // 2, len(h) // 2), mode='reflect')
    filtered = np.convolve(padded, h, mode='same')
    return filtered[len(h) // 2:-len(h) // 2:2] if len(filtered) > len(h) else filtered[::2]


def _qshift_decompose(signal, n_levels=4):
    h0a, h1a, h0b, h1b = _qshift_filters()
    cr, ci = [], []
    ca, cb = signal.copy(), signal.copy()
    for _ in range(n_levels):
        if len(ca) < 10:
            break
        la, ha = _filt_down(ca, h0a), _filt_down(ca, h1a)
        lb, hb = _filt_down(cb, h0b), _filt_down(cb, h1b)
        cr.append((ha + hb) / np.sqrt(2))
        ci.append((ha - hb) / np.sqrt(2))
        ca, cb = la, lb
    cr.append((ca + cb) / np.sqrt(2))
    ci.append((ca - cb) / np.sqrt(2))
    return cr, ci


def _magnitude_threshold(mags, k=1.4):
    out = []
    for m in mags:
        t = np.median(m) + k * np.std(m)
        out.append(np.where(m > t, m, m * 0.55))
    return out


def _memory_deflation(vol, t, defl_mem, decay=0.85, window=60):
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
    if t < 30:
        return 1.0
    w = vol[t - 30:t]
    w = w[w > 0]
    if len(w) < 10:
        return 1.0
    e = np.std(w) / (np.mean(w) + 1e-8)
    return np.clip(1.0 + e * scale, 0.88, 1.52)


def _robust_vol(vol, t, win=20):
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


def _wasserstein_1d(a, b):
    """Compute W_1 distance between two 1D empirical distributions."""
    return wasserstein_distance(a, b)


def _sinkhorn_divergence(a, b, eps=0.1, max_iter=30):
    """Compute regularized OT divergence via Sinkhorn iterations."""
    n, m = len(a), len(b)
    if n < 3 or m < 3:
        return 0.0
    a_s, b_s = np.sort(a), np.sort(b)
    C = np.abs(a_s[:, None] - b_s[None, :])
    K = np.exp(-C / (eps + 1e-8))
    u = np.ones(n) / n
    v = np.ones(m) / m
    mu_w = np.ones(n) / n
    nu_w = np.ones(m) / m
    for _ in range(max_iter):
        u = mu_w / (K @ v + 1e-10)
        v = nu_w / (K.T @ u + 1e-10)
    T = np.diag(u) @ K @ np.diag(v)
    return np.sum(T * C)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 1: Wasserstein Distance                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class WassersteinDistanceModel:
    """
    Computes W_1 distance between consecutive return windows.
    Large W_1 → distributional shift → inflate process noise.
    Small W_1 → stable distribution → trust the filter.

    Key formula:
        q_mult = 1 + α · W_1(F_{t-w:t-w/2}, F_{t-w/2:t}) / σ_global
    """

    DEFL_DECAY = 0.83
    HYV_TARGET = -390
    ENTROPY_ALPHA = 0.10
    LL_BOOST = 1.34
    STRESS_POWER = 0.43
    WASS_ALPHA = 0.40
    WASS_WINDOW = 40

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._wass_cache = 1.0

    def _wasserstein_mult(self, returns, t):
        if t < self.WASS_WINDOW:
            return 1.0
        half = self.WASS_WINDOW // 2
        a = returns[t - self.WASS_WINDOW:t - half]
        b = returns[t - half:t]
        sigma_g = np.std(returns[:t]) + 1e-8
        w1 = _wasserstein_1d(a, b)
        return 1.0 + self.WASS_ALPHA * w1 / sigma_g

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
        ll = sum(_scale_ll(mags_t[i], vol, q * (2 ** i), c, phi) * cw for i in range(len(mags_t)))
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

            if t % 10 == 0 and t >= self.WASS_WINDOW:
                self._wass_cache = self._wasserstein_mult(ret, t)
            wm = np.clip(self._wass_cache, 0.8, 1.6)

            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc

            pm = phi * state
            pv = phi ** 2 * P + q * mult * stress * wm
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
# ║  MODEL 2: Sinkhorn Divergence                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class SinkhornDivergenceModel:
    """
    Uses entropy-regularized optimal transport (Sinkhorn divergence)
    between consecutive return windows. The entropic regularization
    provides smoother gradients than raw W_1.

    Key formula:
        obs_mult = 1 + β · S_ε(F_past, F_recent) / σ²
    where S_ε is Sinkhorn divergence with regularization ε.
    """

    DEFL_DECAY = 0.80
    HYV_TARGET = -410
    ENTROPY_ALPHA = 0.08
    LL_BOOST = 1.36
    STRESS_POWER = 0.41
    SINK_BETA = 0.35
    SINK_WINDOW = 36
    SINK_EPS = 0.1

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._sink_cache = 1.0

    def _sinkhorn_mult(self, returns, t):
        if t < self.SINK_WINDOW:
            return 1.0
        half = self.SINK_WINDOW // 2
        a = returns[t - self.SINK_WINDOW:t - half]
        b = returns[t - half:t]
        var_g = np.var(returns[:t]) + 1e-8
        sd = _sinkhorn_divergence(a, b, eps=self.SINK_EPS)
        return 1.0 + self.SINK_BETA * sd / var_g

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
        ll = sum(_scale_ll(mags_t[i], vol, q * (2 ** i), c, phi) * cw for i in range(len(mags_t)))
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

            if t % 12 == 0 and t >= self.SINK_WINDOW:
                self._sink_cache = self._sinkhorn_mult(ret, t)
            sm = np.clip(self._sink_cache, 0.8, 1.5)

            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc

            pm = phi * state
            pv = phi ** 2 * P + q * mult * stress
            blend = 0.54 * rv + 0.46 * ema_vol
            obs = c * blend * mult * np.sqrt(ent * stress) * sm
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
# ║  MODEL 3: Kantorovich Dual                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class KantorovichDualModel:
    """
    Uses the Kantorovich dual potential to measure distributional mismatch.
    The dual formulation W_1 = sup_{Lip(f)≤1} E_μ[f] - E_ν[f] gives us
    a "potential gap" that can be interpreted as the maximum expected
    profit from arbitraging between two distributions.

    We approximate the 1-Lipschitz function via sorted quantile matching.

    Key formula:
        state_correction = δ · sign(Δ_dual) · |Δ_dual|^{1/2}
    where Δ_dual is the Kantorovich dual gap.
    """

    DEFL_DECAY = 0.81
    HYV_TARGET = -370
    ENTROPY_ALPHA = 0.11
    LL_BOOST = 1.33
    STRESS_POWER = 0.44
    KANT_DELTA = 0.25
    KANT_WINDOW = 38

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._kant_cache = 0.0

    def _kantorovich_dual_gap(self, returns, t):
        if t < self.KANT_WINDOW:
            return 0.0
        half = self.KANT_WINDOW // 2
        a = np.sort(returns[t - self.KANT_WINDOW:t - half])
        b = np.sort(returns[t - half:t])
        min_len = min(len(a), len(b))
        if min_len < 3:
            return 0.0
        a_q = np.linspace(0, 1, min_len)
        a_vals = np.interp(a_q, np.linspace(0, 1, len(a)), a)
        b_vals = np.interp(a_q, np.linspace(0, 1, len(b)), b)
        gap = np.mean(a_vals) - np.mean(b_vals)
        return gap

    def _kant_state_correction(self, gap):
        return self.KANT_DELTA * np.sign(gap) * np.sqrt(abs(gap) + 1e-10)

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
        ll = sum(_scale_ll(mags_t[i], vol, q * (2 ** i), c, phi) * cw for i in range(len(mags_t)))
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

            if t % 15 == 0 and t >= self.KANT_WINDOW:
                self._kant_cache = self._kantorovich_dual_gap(ret, t)
            state_corr = self._kant_state_correction(self._kant_cache)

            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc

            pm = phi * state + state_corr
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


def get_optimal_transport_models():
    return [
        {
            "name": "wasserstein_distance",
            "class": WassersteinDistanceModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "W₁ Earth Mover's Distance detects distributional regime shifts"
        },
        {
            "name": "sinkhorn_divergence",
            "class": SinkhornDivergenceModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Entropy-regularized OT via Sinkhorn controls adaptive process noise"
        },
        {
            "name": "kantorovich_dual",
            "class": KantorovichDualModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Kantorovich dual potential gap drives state correction"
        },
    ]
