"""
Vectorized Black-Scholes utilities and small quantitative helpers.
Separated from options.py to reduce code volume and allow reuse.
"""
from __future__ import annotations

import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import brentq

# -------------------- Black-Scholes (vectorized-capable) --------------------

def _as_array(x):
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return np.array(x, dtype=float)


def bsm_call_price(S, K, T, r, sigma):
    """BSM European call price. Accepts scalars or numpy arrays of same shape.
    Broadcasts as needed. Gracefully handles edge cases (T<=0 or sigma<=0) by returning intrinsic.
    """
    S = _as_array(S)
    K = _as_array(K)
    T = _as_array(T)
    r = float(r)
    sigma = _as_array(sigma)

    # Scalar fast-path to avoid mask assignment on 0-d arrays
    if S.ndim == 0 and K.ndim == 0 and T.ndim == 0 and sigma.ndim == 0:
        S0 = float(S); K0 = float(K); T0 = float(T); sig0 = float(sigma)
        intrinsic = max(S0 - K0, 0.0)
        if not (T0 > 0 and sig0 > 0 and S0 > 0 and K0 > 0):
            return float(intrinsic)
        sqrtT = math.sqrt(T0)
        d1 = (math.log(S0 / K0) + (r + 0.5 * sig0 * sig0) * T0) / (sig0 * sqrtT)
        d2 = d1 - sig0 * sqrtT
        return float(S0 * norm.cdf(d1) - K0 * math.exp(-r * T0) * norm.cdf(d2))

    # Intrinsic fallback for degenerate params
    intrinsic = np.maximum(S - K, 0.0)

    # Valid region mask
    m = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    out = np.array(intrinsic, dtype=float)
    if not np.any(m):
        return out

    Sm = S[m]; Km = K[m]; Tm = T[m]; sigm = sigma[m]
    sqrtT = np.sqrt(Tm)
    d1 = (np.log(Sm / Km) + (r + 0.5 * sigm * sigm) * Tm) / (sigm * sqrtT)
    d2 = d1 - sigm * sqrtT
    out[m] = Sm * norm.cdf(d1) - Km * np.exp(-r * Tm) * norm.cdf(d2)
    return out


def bsm_call_delta(S, K, T, r, sigma):
    S = _as_array(S); K = _as_array(K); T = _as_array(T); sigma = _as_array(sigma); r = float(r)
    # Scalar fast-path
    if S.ndim == 0 and K.ndim == 0 and T.ndim == 0 and sigma.ndim == 0:
        S0 = float(S); K0 = float(K); T0 = float(T); sig0 = float(sigma)
        if not (T0 > 0 and sig0 > 0 and S0 > 0 and K0 > 0):
            return float(1.0 if S0 > K0 else 0.0)
        sqrtT = math.sqrt(T0)
        d1 = (math.log(S0 / K0) + (r + 0.5 * sig0 * sig0) * T0) / (sig0 * sqrtT)
        return float(norm.cdf(d1))
    # Delta intrinsic approximation when invalid
    out = np.where(S > K, 1.0, 0.0).astype(float)
    m = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    if not np.any(m):
        return out
    Sm = S[m]; Km = K[m]; Tm = T[m]; sigm = sigma[m]
    sqrtT = np.sqrt(Tm)
    d1 = (np.log(Sm / Km) + (r + 0.5 * sigm * sigm) * Tm) / (sigm * sqrtT)
    out[m] = norm.cdf(d1)
    return out


def strike_for_target_delta(S, T, r, sigma, target_delta):
    """Solve scalar K such that call delta ~= target_delta using Brent. Scalar routine for stability."""
    S = float(S); T = float(T); sigma = float(sigma); target = float(target_delta); r = float(r)
    if not (0.01 <= target <= 0.99):
        target = 0.25

    def f(K):
        return float(bsm_call_delta(S, K, T, r, sigma)) - target

    try:
        return float(brentq(f, S * 0.5, S * 2.0, maxiter=200))
    except Exception:
        # heuristic fallback
        mny = max(0.01, min(0.2, 0.3 - target * 0.5))
        return float(S * (1.0 + mny))


def bsm_implied_vol(price, S, K, T, r):
    price = float(price); S = float(S); K = float(K); T = float(T); r = float(r)
    intrinsic = max(S - K, 0.0)
    price = float(max(price, intrinsic + 1e-8))

    def f(sig):
        return float(bsm_call_price(S, K, T, r, sig)) - price
    try:
        return float(brentq(f, 1e-6, 5.0, maxiter=300))
    except Exception:
        return float('nan')


def lognormal_prob_geq(S0, mu_ln, sigma_ln, threshold):
    if threshold <= 0:
        return 1.0
    z = (np.log(threshold) - mu_ln) / sigma_ln
    return 1.0 - float(norm.cdf(z))
