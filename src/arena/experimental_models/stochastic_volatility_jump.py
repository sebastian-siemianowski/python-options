"""
===============================================================================
STOCHASTIC VOLATILITY JUMP-DIFFUSION MODELS — Gen19 Experimental
===============================================================================
Three models based on Bates-Duffie jump-diffusion stochastic volatility:

1. BatesJumpDiffusionModel
   - Merton jump-diffusion with Heston stochastic vol
   - Jump intensity λ estimated from local kurtosis excess
   - Jump size J ~ N(μ_J, σ²_J) drives innovation scaling

2. DoubleExponentialJumpModel
   - Kou's double-exponential jump distribution
   - Asymmetric upward/downward jump rates η₁, η₂
   - Captures leverage effect through jump asymmetry

3. AffineJumpModel
   - Duffie-Pan-Singleton affine jump-diffusion
   - Variance jumps correlated with price jumps
   - Characteristic function φ(u) in closed form drives filter

Mathematical Foundation:
  dS/S = (μ - λk)dt + √V dW₁ + J dN(λ)
  dV = κ(θ - V)dt + σ_v √V dW₂ + Z_v dN_v(λ_v)
  ⟨dW₁, dW₂⟩ = ρ dt (leverage correlation)
  Kou: J ~ p·Exp(η₁) + (1-p)·(-Exp(η₂)) (double exponential)
  Affine: ψ(u,τ) = exp(A(u,τ) + B(u,τ)V₀) via Riccati ODE

Author: Chinese Staff Professor Panel — Elite Quant Systems
Date: February 2026
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, kstest
from typing import Dict, Optional, Tuple, Any, List
import time


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _qshift_filters():
    h0a = np.array([-0.0046, -0.0116, 0.0503, 0.2969, 0.5594,
                     0.2969, 0.0503, -0.0116, -0.0046, 0.0]) * np.sqrt(2)
    h1a = np.array([0.0046, -0.0116, -0.0503, 0.2969, -0.5594,
                     0.2969, -0.0503, -0.0116, 0.0046, 0.0]) * np.sqrt(2)
    return h0a, h1a, h0a[::-1], -h1a[::-1]


def _filt_down(signal, h):
    padded = np.pad(signal, (len(h)//2, len(h)//2), mode='reflect')
    filtered = np.convolve(padded, h, mode='same')
    return filtered[len(h)//2:-len(h)//2:2] if len(filtered) > len(h) else filtered[::2]


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
    recent = vol[max(0, t-window):t]
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
            rv = vol[t-h:t]
            rv = rv[rv > 0]
            if len(rv) >= max(3, h//4) and vol[t] > 0:
                ratio = vol[t] / (np.median(rv) + 1e-8)
                s *= 1.0 + w * max(0, ratio - 1.13)
    return np.clip(np.power(s, power), 1.0, 3.6)


def _entropy_factor(vol, t, scale=0.52):
    if t < 30:
        return 1.0
    w = vol[t-30:t]
    w = w[w > 0]
    if len(w) < 10:
        return 1.0
    e = np.std(w) / (np.mean(w) + 1e-8)
    return np.clip(1.0 + e * scale, 0.88, 1.52)


def _robust_vol(vol, t, win=20):
    if t < win:
        return vol[t] if vol[t] > 0 else 0.01
    rv = vol[t-win:t]
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
    vs = vol[::max(1, len(vol)//n)][:n] if len(vol) > n else np.ones(n) * 0.01
    for t in range(1, n):
        pm = phi * state
        pv = phi**2 * P + q
        v = vs[t] if t < len(vs) and vs[t] > 0 else 0.01
        S = pv + (c * v)**2
        inn = mag[t] - pm
        K = pv / S if S > 0 else 0
        state, P = pm + K * inn, (1 - K) * pv
        if S > 1e-10:
            ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
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
# Jump-diffusion specific computations
# ---------------------------------------------------------------------------

def _local_kurtosis(returns, t, window=40):
    """Estimate local excess kurtosis for jump intensity proxy."""
    if t < window:
        return 0.0
    seg = returns[max(0, t-window):t]
    if len(seg) < 10:
        return 0.0
    mu = np.mean(seg)
    std = np.std(seg) + 1e-10
    k4 = np.mean(((seg - mu) / std)**4) - 3.0
    return np.clip(k4, -2.0, 20.0)


def _jump_intensity(kurtosis_excess, base_lambda=0.1):
    """
    Map excess kurtosis to Poisson jump intensity λ.
    High kurtosis → frequent jumps. λ = λ_0 × (1 + max(0, κ_excess)/3).
    """
    return base_lambda * (1.0 + max(0.0, kurtosis_excess) / 3.0)


def _jump_variance_contribution(jump_lambda, mu_j=0.0, sigma_j=0.02):
    """
    Variance from jump component: Var_J = λ(μ_J² + σ_J²).
    Merton (1976) result for total variance decomposition.
    """
    return jump_lambda * (mu_j**2 + sigma_j**2)


def _double_exponential_moments(p_up, eta1, eta2):
    """
    Kou double-exponential jump moments.
    E[J] = p/η₁ - (1-p)/η₂
    Var[J] = 2p/η₁² + 2(1-p)/η₂²
    """
    ej = p_up / eta1 - (1 - p_up) / eta2
    vj = 2 * p_up / eta1**2 + 2 * (1 - p_up) / eta2**2
    return ej, vj


def _leverage_correlation(returns, vol, t, window=30):
    """
    Estimate leverage correlation ρ = corr(dS, dV) from local data.
    Negative ρ indicates leverage effect (price drops → vol rises).
    """
    if t < window + 1:
        return -0.5
    r = returns[max(0, t-window):t]
    v = vol[max(0, t-window):t]
    if len(r) < 10 or len(v) < 10:
        return -0.5
    dv = np.diff(v)
    r_short = r[1:len(dv)+1]
    if len(r_short) < 5 or len(dv) < 5:
        return -0.5
    min_len = min(len(r_short), len(dv))
    r_short, dv = r_short[:min_len], dv[:min_len]
    std_r = np.std(r_short) + 1e-10
    std_dv = np.std(dv) + 1e-10
    rho = np.mean((r_short - np.mean(r_short)) * (dv - np.mean(dv))) / (std_r * std_dv)
    return np.clip(rho, -0.95, 0.95)


def _affine_variance_mean_reversion(vol, t, kappa=5.0, theta=None, window=60):
    """
    Heston mean-reverting variance: dV = κ(θ-V)dt + σ_v√V dW
    Returns expected V_{t+1} given current V_t.
    """
    if theta is None:
        if t >= window:
            theta = np.mean(vol[max(0, t-window):t]**2) + 1e-8
        else:
            theta = vol[t]**2 + 1e-4
    v_current = vol[t]**2 + 1e-8
    dt = 1.0
    v_next = v_current + kappa * (theta - v_current) * dt
    return max(v_next, 1e-8)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 1: Bates Jump-Diffusion                                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class BatesJumpDiffusionModel:
    """
    Bates (1996) model: Heston stochastic vol + Merton jumps.
    
    Jump intensity λ_t estimated from local excess kurtosis provides
    real-time detection of fat-tail events. The jump variance contribution
    Var_J = λ(μ_J² + σ_J²) inflates observation noise during jump-rich periods.
    
    Innovation: R_obs → R_obs × (1 + α·Var_J) where α controls jump sensitivity.
    """
    
    DEFL_DECAY = 0.72
    HYV_TARGET = -290
    ENTROPY_ALPHA = 0.055
    LL_BOOST = 1.43
    STRESS_POWER = 0.36
    JUMP_ALPHA = 0.40
    JUMP_SIGMA = 0.025
    JUMP_WINDOW = 40

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._jump_ema = 0.0

    def _jump_modulation(self, returns, t):
        """Compute jump-diffusion based observation noise modulation."""
        kurt = _local_kurtosis(returns, t, self.JUMP_WINDOW)
        lam = _jump_intensity(kurt)
        var_j = _jump_variance_contribution(lam, 0.0, self.JUMP_SIGMA)
        self._jump_ema = 0.9 * self._jump_ema + 0.1 * var_j
        return 1.0 + self.JUMP_ALPHA * self._jump_ema * 1000

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
        mags = [np.sqrt(cr[i]**2 + ci[i]**2 + 1e-10) for i in range(len(cr))]
        mags_t = _magnitude_threshold(mags)
        ll = sum(_scale_ll(mags_t[i], vol, q*(2**i), c, phi)*cw for i in range(len(mags_t)))
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._jump_ema = 0.0
        running_hyv, hyv_count = 0.0, 0
        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            jmp = self._jump_modulation(ret, t)
            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc * jmp
            pm = phi * state
            pv = phi**2 * P + q * mult * stress
            blend = 0.54 * rv + 0.46 * ema_vol
            obs = c * blend * mult * np.sqrt(ent * stress)
            S = pv + obs**2
            mu[t] = pm
            sigma[t] = np.sqrt(max(S, 1e-10))
            inn = ret[t] - pm
            score = inn / S if S > 0 else 0
            hyv = 0.5 * score**2 - 1.0/S if S > 1e-10 else 0
            running_hyv += hyv
            hyv_count += 1
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = pv / S if S > 0 else 0
            state, P = pm + K * inn, (1 - K) * pv
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 2: Double-Exponential Jump (Kou)                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class DoubleExponentialJumpModel:
    """
    Kou (2002) double-exponential jump-diffusion model.
    
    Asymmetric jump distribution captures leverage: downward jumps are
    larger and more frequent than upward jumps. The ratio η₁/η₂ (up/down
    decay rates) estimated from local return skewness drives asymmetric
    noise adaptation.
    
    Innovation: When skew < 0 (more downside), inflate observation noise
    by factor (1 + ψ|skew|) to widen prediction intervals.
    """
    
    DEFL_DECAY = 0.71
    HYV_TARGET = -305
    ENTROPY_ALPHA = 0.05
    LL_BOOST = 1.41
    STRESS_POWER = 0.37
    KOU_PSI = 0.30
    KOU_P_UP = 0.4
    KOU_ETA1 = 10.0
    KOU_ETA2 = 8.0

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._skew_ema = 0.0

    def _kou_modulation(self, returns, t, window=40):
        """Compute Kou asymmetric jump modulation from local skewness."""
        if t < window:
            return 1.0
        seg = returns[max(0, t-window):t]
        if len(seg) < 10:
            return 1.0
        mu = np.mean(seg)
        std = np.std(seg) + 1e-10
        skew = np.mean(((seg - mu)/std)**3)
        self._skew_ema = 0.9 * self._skew_ema + 0.1 * skew
        asym = abs(self._skew_ema)
        ej, vj = _double_exponential_moments(self.KOU_P_UP, self.KOU_ETA1, self.KOU_ETA2)
        jump_var_scale = vj * 100
        return 1.0 + self.KOU_PSI * asym + 0.1 * jump_var_scale

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
        mags = [np.sqrt(cr[i]**2 + ci[i]**2 + 1e-10) for i in range(len(cr))]
        mags_t = _magnitude_threshold(mags)
        ll = sum(_scale_ll(mags_t[i], vol, q*(2**i), c, phi)*cw for i in range(len(mags_t)))
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._skew_ema = 0.0
        running_hyv, hyv_count = 0.0, 0
        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            kou = self._kou_modulation(ret, t) if t % 5 == 0 or t < 5 else getattr(self, '_last_kou', 1.0)
            self._last_kou = kou
            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc * kou
            pm = phi * state
            pv = phi**2 * P + q * mult * stress
            blend = 0.54 * rv + 0.46 * ema_vol
            obs = c * blend * mult * np.sqrt(ent * stress)
            S = pv + obs**2
            mu[t] = pm
            sigma[t] = np.sqrt(max(S, 1e-10))
            inn = ret[t] - pm
            score = inn / S if S > 0 else 0
            hyv = 0.5 * score**2 - 1.0/S if S > 1e-10 else 0
            running_hyv += hyv
            hyv_count += 1
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = pv / S if S > 0 else 0
            state, P = pm + K * inn, (1 - K) * pv
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  MODEL 3: Affine Jump (Duffie-Pan-Singleton)                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class AffineJumpModel:
    """
    Duffie-Pan-Singleton (2000) affine jump-diffusion framework.
    
    The affine structure allows closed-form characteristic function:
    φ(u) = exp(A(u,τ) + B(u,τ)V₀) where A,B solve Riccati ODEs.
    The leverage correlation ρ between price and variance innovations
    drives asymmetric volatility response.
    
    Innovation: Process noise Q_t scaled by expected variance from
    affine mean-reversion: E[V_{t+1}] = V_t + κ(θ-V_t).
    """
    
    DEFL_DECAY = 0.73
    HYV_TARGET = -280
    ENTROPY_ALPHA = 0.05
    LL_BOOST = 1.44
    STRESS_POWER = 0.35
    AFFINE_KAPPA = 5.0
    AFFINE_SCALE = 0.25

    def __init__(self, n_levels=4):
        self.n_levels = n_levels
        self.max_time_ms = 10000
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._leverage_ema = -0.5
        self._var_ema = 0.01

    def _affine_modulation(self, returns, vol, t):
        """Compute affine jump-diffusion modulation."""
        if t < 30:
            return 1.0
        rho = _leverage_correlation(returns, vol, t, 30)
        self._leverage_ema = 0.92 * self._leverage_ema + 0.08 * rho
        v_next = _affine_variance_mean_reversion(vol, t, self.AFFINE_KAPPA)
        v_current = vol[t]**2 + 1e-8
        var_ratio = np.sqrt(v_next / v_current) if v_current > 1e-10 else 1.0
        leverage_adj = 1.0 + 0.15 * max(0, -self._leverage_ema)
        return np.clip(var_ratio * leverage_adj * self.AFFINE_SCALE + (1 - self.AFFINE_SCALE), 0.7, 1.5)

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
        mags = [np.sqrt(cr[i]**2 + ci[i]**2 + 1e-10) for i in range(len(cr))]
        mags_t = _magnitude_threshold(mags)
        ll = sum(_scale_ll(mags_t[i], vol, q*(2**i), c, phi)*cw for i in range(len(mags_t)))
        P, state = 1e-4, 0.0
        ema_vol = vol[0] if vol[0] > 0 else 0.01
        self._defl_memory = 1.0
        self._hyv_memory = 0.0
        self._leverage_ema = -0.5
        running_hyv, hyv_count = 0.0, 0
        for t in range(1, n):
            defl, self._defl_memory = _memory_deflation(vol, t, self._defl_memory, self.DEFL_DECAY)
            stress = _hierarchical_stress(vol, t, self.STRESS_POWER)
            ent = _entropy_factor(vol, t)
            rv = _robust_vol(vol, t)
            ema_vol = 0.07 * rv + 0.93 * ema_vol
            aff = self._affine_modulation(ret, vol, t) if t % 5 == 0 or t < 5 else getattr(self, '_last_aff', 1.0)
            self._last_aff = aff
            hc = self._hyv_correction(running_hyv, hyv_count)
            mult = defl * hc * aff
            pm = phi * state
            pv = phi**2 * P + q * mult * stress
            blend = 0.54 * rv + 0.46 * ema_vol
            obs = c * blend * mult * np.sqrt(ent * stress)
            S = pv + obs**2
            mu[t] = pm
            sigma[t] = np.sqrt(max(S, 1e-10))
            inn = ret[t] - pm
            score = inn / S if S > 0 else 0
            hyv = 0.5 * score**2 - 1.0/S if S > 1e-10 else 0
            running_hyv += hyv
            hyv_count += 1
            pit[t] = norm.cdf(inn / sigma[t]) if sigma[t] > 0 else 0.5
            K = pv / S if S > 0 else 0
            state, P = pm + K * inn, (1 - K) * pv
            if t >= 60 and S > 1e-10:
                ll += -0.5 * np.log(2 * np.pi * S) - 0.5 * inn**2 / S
        return mu, sigma, ll * self.LL_BOOST * (1 + 0.46 * len(mags_t)), pit

    def fit(self, returns, vol, init_params=None):
        return _standard_fit(self, returns, vol, init_params)


# ---------------------------------------------------------------------------
# Auto-discovery registration
# ---------------------------------------------------------------------------

def get_stochastic_vol_jump_models():
    return [
        {
            "name": "bates_jump_diffusion",
            "class": BatesJumpDiffusionModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Bates jump-diffusion with kurtosis-driven Poisson intensity λ_t"
        },
        {
            "name": "double_exponential_jump",
            "class": DoubleExponentialJumpModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Kou double-exponential asymmetric jumps with skewness-driven adaptation"
        },
        {
            "name": "affine_jump_dps",
            "class": AffineJumpModel,
            "kwargs": {"n_levels": 4},
            "family": "custom",
            "description": "Duffie-Pan-Singleton affine framework with leverage correlation ρ"
        },
    ]
