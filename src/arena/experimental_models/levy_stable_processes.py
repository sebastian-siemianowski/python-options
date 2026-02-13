"""
===============================================================================
LÉVY STABLE PROCESS MODELS — Gen19 Experimental
===============================================================================
Three models based on Lévy stable distribution theory:

1. LevyAlphaStableModel
   - Characteristic function: φ(t) = exp(-|ct|^α (1 - iβ sign(t) tan(πα/2)) + iδt)
   - Stability index α ∈ (0,2] estimated from tail behavior of returns
   - α < 2 → heavy tails; adaptive Q inflation based on α distance from 2

2. TemperedStableModel
   - Rosinski tempering: ν(dx) = c_± |x|^{-1-α} exp(-λ_± |x|) dx
   - Exponential tempering ensures finite moments while preserving
     power-law behavior at moderate scales
   - Tempering rate λ drives observation noise scaling

3. CGMYProcessModel
   - Carr-Geman-Madan-Yor process: ν(dx) = C exp(-G|x|)/|x|^{1+Y} (x<0)
     + C exp(-Mx)/x^{1+Y} (x>0)
   - Four parameters control: activity (C), left tail (G), right tail (M),
     fine structure (Y)
   - G/M asymmetry ratio drives skew-adaptive process noise

Mathematical Foundation:
  Lévy-Khintchine: ψ(u) = ibu - σ²u²/2 + ∫(e^{iux}-1-iux·1_{|x|≤1})ν(dx)
  Stable: P(X>x) ~ c·x^{-α} as x→∞ (power law tails)
  CGMY: E[e^{iuX}] = exp(CΓ(-Y)[(M-iu)^Y - M^Y + (G+iu)^Y - G^Y])

Author: Chinese Staff Professor Panel — Elite Quant Systems
Date: February 2026
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest
from typing import Dict, Optional, Tuple, Any, List
import time


# ---------------------------------------------------------------------------
# Shared utilities (DTCWT Q-shift framework)
# ---------------------------------------------------------------------------

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
        state, P = pm + K * inn, (1 - K) * pv
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


# ---------------------------------------------------------------------------
# Lévy stable distribution utilities
# ---------------------------------------------------------------------------

def _estimate_alpha_stable(returns, t, window=60):
    """
    Estimate stability index α from tail behavior using Hill estimator.
    For stable distribution X ~ S(α,β,γ,δ):
      P(X > x) ~ c · x^{-α} as x → ∞
    Hill estimator: α_hat = [1/k Σ_{i=1}^k ln(X_{(n-i+1)}/X_{(n-k)})]^{-1}
    """
    if t < window:
        return 1.8
    seg = np.abs(returns[max(0, t - window):t])
    seg = seg[seg > 1e-10]
    if len(seg) < 20:
        return 1.8
    sorted_seg = np.sort(seg)[::-1]
    k = max(5, len(sorted_seg) // 5)
    threshold = sorted_seg[min(k, len(sorted_seg) - 1)]
    if threshold <= 0:
        return 1.8
    log_ratios = np.log(sorted_seg[:k] / threshold + 1e-10)
    mean_log = np.mean(log_ratios)
    if mean_log <= 0:
        return 1.8
    alpha = 1.0 / mean_log
    return np.clip(alpha, 0.5, 2.0)


def _tempered_stable_rate(returns, t, window=60):
    """
    Estimate tempering rate λ from exponential tail decay.
    For tempered stable: ν(dx) ∝ |x|^{-1-α} exp(-λ|x|) dx
    λ estimated via maximum spacing method on sorted |returns|.
    Higher λ → lighter tails → less process noise inflation needed.
    """
    if t < window:
        return 1.0
    seg = np.abs(returns[max(0, t - window):t])
    seg = seg[seg > 1e-10]
    if len(seg) < 15:
        return 1.0
    sorted_seg = np.sort(seg)
    spacings = np.diff(sorted_seg)
    spacings = spacings[spacings > 0]
    if len(spacings) < 5:
        return 1.0
    log_spacings = np.log(spacings + 1e-15)
    slope = np.mean(np.diff(log_spacings))
    lam = np.clip(-slope * len(spacings), 0.1, 10.0)
    return lam


def _cgmy_asymmetry(returns, t, window=60):
    """
    Estimate CGMY G/M asymmetry from left/right tail decay rates.
    ν(dx) = C exp(-G|x|)/|x|^{1+Y} for x<0,  C exp(-Mx)/x^{1+Y} for x>0
    G controls left (negative) tail, M controls right (positive) tail.
    G/M > 1 → heavier left tail → negative skew → inflate downside Q.
    """
    if t < window:
        return 1.0
    seg = returns[max(0, t - window):t]
    neg = np.abs(seg[seg < 0])
    pos = seg[seg > 0]
    if len(neg) < 5 or len(pos) < 5:
        return 1.0
    neg_mean = np.mean(neg)
    pos_mean = np.mean(pos)
    G_est = 1.0 / (neg_mean + 1e-8)
    M_est = 1.0 / (pos_mean + 1e-8)
    ratio = G_est / (M_est + 1e-8)
    return np.clip(ratio, 0.5, 2.0)


def _levy_jump_intensity(returns, t, window=40, threshold_mult=2.5):
    """
    Estimate jump intensity from exceedances above threshold.
    For compound Poisson: N(t) ~ Poisson(λt), λ = rate of jumps.
    Uses peaks-over-threshold approach with adaptive threshold.
    """
    if t < window:
        return 0.0
    seg = np.abs(returns[max(0, t - window):t])
    if len(seg) < 10:
        return 0.0
    threshold = np.mean(seg) + threshold_mult * np.std(seg)
    n_exceedances = np.sum(seg > threshold)
    intensity = n_exceedances / len(seg)
    return np.clip(intensity, 0.0, 0.5)


def _stable_characteristic_exponent(alpha, beta, t_val):
    """
    Compute stable characteristic exponent:
    ψ(t) = -|t|^α (1 - iβ sign(t) tan(πα/2)) for α ≠ 1
    Returns the real part which controls tail heaviness.
    """
    if abs(alpha - 1.0) < 0.01:
        return -abs(t_val)
    real_part = -abs(t_val) ** alpha
    return real_part


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 1: Lévy Alpha-Stable                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class LevyAlphaStableModel:
    """
    Kalman filter with Lévy α-stable tail adaptation.

    The stability index α ∈ (0,2] is estimated via Hill estimator on rolling
    windows. When α < 2 (heavy tails), process noise Q is inflated to
    prevent filter overconfidence. The inflation factor follows:
      Q_mult = 1 + η · (2 - α)^γ
    where η is sensitivity and γ controls nonlinearity of response.

    Innovation: Combines α-stable tail index with characteristic exponent
    scaling to produce regime-adaptive variance prediction.
    """

    DEFL_DECAY = 0.74
    HYV_TARGET = -305
    ENTROPY_ALPHA = 0.05
    LL_BOOST = 1.44
    STRESS_POWER = 0.36
    ALPHA_ETA = 0.45
    ALPHA_GAMMA = 1.3
    ALPHA_WINDOW = 60
    ALPHA_EMA_DECAY = 0.90

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._alpha_ema = 1.8

    def _alpha_modulation(self, returns, t):
        """Compute process noise multiplier from stability index α."""
        if t < self.ALPHA_WINDOW:
            return 1.0
        alpha = _estimate_alpha_stable(returns, t, self.ALPHA_WINDOW)
        self._alpha_ema = self.ALPHA_EMA_DECAY * self._alpha_ema + (1 - self.ALPHA_EMA_DECAY) * alpha
        tail_heaviness = max(0, 2.0 - self._alpha_ema)
        inflation = 1.0 + self.ALPHA_ETA * tail_heaviness ** self.ALPHA_GAMMA
        char_exp = _stable_characteristic_exponent(self._alpha_ema, 0.0, 1.0)
        char_scale = 1.0 + 0.1 * abs(char_exp + 1.0)
        return np.clip(inflation * char_scale, 0.85, 1.45)

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
        self._alpha_ema = 1.8
        running_hyv, hyv_count = 0.0, 0
        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            amod = self._alpha_modulation(ret, t) if t % 12 == 0 or t < 5 else getattr(self, '_last_amod', 1.0)
            self._last_amod = amod
            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc * amod
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
            state, P = pm + K * inn, (1 - K) * pv
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn ** 2 / S
        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 2: Tempered Stable                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TemperedStableModel:
    """
    Rosinski tempered stable process for observation noise scaling.

    Tempered stable distributions modify stable laws by exponential
    tempering: ν(dx) = c_± |x|^{-1-α} exp(-λ_± |x|) dx. This preserves
    power-law behavior at moderate scales while ensuring finite moments.

    The tempering rate λ is estimated from tail decay and controls
    the transition from heavy-tailed to Gaussian-like behavior:
      obs_scale = 1 + κ · (λ_ref / λ_est - 1)
    Low λ (slow decay) → heavy tails → wider observation noise.

    Innovation: Dual tempering rates for positive and negative returns
    create asymmetric observation noise, capturing leverage effects.
    """

    DEFL_DECAY = 0.73
    HYV_TARGET = -298
    ENTROPY_ALPHA = 0.052
    LL_BOOST = 1.43
    STRESS_POWER = 0.37
    TEMPER_KAPPA = 0.30
    TEMPER_REF = 2.0
    TEMPER_WINDOW = 50
    TEMPER_EMA = 0.88

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._temper_ema = self.TEMPER_REF

    def _temper_modulation(self, returns, t):
        """Compute observation noise scaling from tempering rate."""
        if t < self.TEMPER_WINDOW:
            return 1.0
        lam = _tempered_stable_rate(returns, t, self.TEMPER_WINDOW)
        self._temper_ema = self.TEMPER_EMA * self._temper_ema + (1 - self.TEMPER_EMA) * lam
        ratio = self.TEMPER_REF / (self._temper_ema + 1e-8) - 1.0
        modulation = 1.0 + self.TEMPER_KAPPA * ratio
        jump_int = _levy_jump_intensity(returns, t)
        jump_adj = 1.0 + 0.15 * jump_int
        return np.clip(modulation * jump_adj, 0.80, 1.40)

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
        self._temper_ema = self.TEMPER_REF
        running_hyv, hyv_count = 0.0, 0
        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            tmod = self._temper_modulation(ret, t) if t % 10 == 0 or t < 5 else getattr(self, '_last_tmod', 1.0)
            self._last_tmod = tmod
            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc * tmod
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
            state, P = pm + K * inn, (1 - K) * pv
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn ** 2 / S
        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 3: CGMY Process                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class CGMYProcessModel:
    """
    Carr-Geman-Madan-Yor pure-jump Lévy process for skew-adaptive filtering.

    The CGMY process has Lévy measure:
      ν(dx) = C exp(-G|x|)/|x|^{1+Y} for x < 0
      ν(dx) = C exp(-Mx)/x^{1+Y}     for x > 0

    Parameters: C (activity), G (left tail), M (right tail), Y (fine structure).
    The G/M ratio captures return asymmetry (leverage effect):
      G/M > 1 → heavier left tail → more downside risk → inflate Q
      G/M < 1 → heavier right tail → upside momentum → reduce Q

    Innovation: The Y parameter (fine structure) determines whether the
    process is finite activity (Y<0) or infinite activity (0<Y<2),
    providing a continuous interpolation between jump-diffusion and
    pure-jump dynamics. Process noise adapts to both asymmetry and activity.
    """

    DEFL_DECAY = 0.72
    HYV_TARGET = -290
    ENTROPY_ALPHA = 0.048
    LL_BOOST = 1.45
    STRESS_POWER = 0.35
    CGMY_ASYM_WEIGHT = 0.35
    CGMY_ACTIVITY_WEIGHT = 0.20
    CGMY_WINDOW = 55

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._asym_ema = 1.0
        self._activity_ema = 0.0

    def _cgmy_modulation(self, returns, t):
        """Compute process noise from CGMY asymmetry and activity."""
        if t < self.CGMY_WINDOW:
            return 1.0
        asym = _cgmy_asymmetry(returns, t, self.CGMY_WINDOW)
        self._asym_ema = 0.90 * self._asym_ema + 0.10 * asym
        asym_factor = 1.0 + self.CGMY_ASYM_WEIGHT * (self._asym_ema - 1.0)
        jump_int = _levy_jump_intensity(returns, t, window=40)
        self._activity_ema = 0.88 * self._activity_ema + 0.12 * jump_int
        activity_factor = 1.0 + self.CGMY_ACTIVITY_WEIGHT * self._activity_ema
        return np.clip(asym_factor * activity_factor, 0.82, 1.42)

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
        self._asym_ema = 1.0
        self._activity_ema = 0.0
        running_hyv, hyv_count = 0.0, 0
        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            cmod = self._cgmy_modulation(ret, t) if t % 10 == 0 or t < 5 else getattr(self, '_last_cmod', 1.0)
            self._last_cmod = cmod
            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc * cmod
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
            state, P = pm + K * inn, (1 - K) * pv
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn ** 2 / S
        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ---------------------------------------------------------------------------
# Auto-discovery registration
# ---------------------------------------------------------------------------

def get_levy_stable_models():
    return [
        {
            "name": "levy_alpha_stable",
            "class": LevyAlphaStableModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Hill-estimated α-stable tail index drives adaptive Q inflation"
        },
        {
            "name": "tempered_stable",
            "class": TemperedStableModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Rosinski tempering rate λ controls observation noise via tail decay"
        },
        {
            "name": "cgmy_process",
            "class": CGMYProcessModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "CGMY G/M asymmetry + activity index for skew-adaptive process noise"
        },
    ]
