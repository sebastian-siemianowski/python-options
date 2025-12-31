#!/usr/bin/env python3
"""
fx_pln_jpy_signals_v3.py

Quant upgrades:
- multi-speed EWMA drift/vol (fast + slow blend)
- robust returns (winsorized)
- probability of positive return per horizon (p_up)
- t-stat style momentum (cumret / realized vol)
- shrinkage drift (toward 0, stronger in stressed vol regimes)
- clearer HOLD zone based on probability, not raw z

Notes:
- PLNJPY=X is JPY per PLN. BUY => long PLN vs JPY.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import t as student_t, norm
from rich.console import Console
import logging
import os

# HMM regime detection
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# Import presentation layer for display logic
from fx_signals_presentation import (
    render_detailed_signal_table,
    render_simplified_signal_table,
    render_multi_asset_summary_table,
    build_asset_display_label,
    extract_symbol_from_title,
    format_horizon_label,
    DETAILED_COLUMN_DESCRIPTIONS,
    SIMPLIFIED_COLUMN_DESCRIPTIONS,
)

# Import data utilities and helper functions
from fx_data_utils import (
    norm_cdf,
    _to_float,
    safe_last,
    winsorize,
    _download_prices,
    _resolve_display_name,
    _fetch_px_symbol,
    _fetch_with_fallback,
    fetch_px,
    fetch_usd_to_pln_exchange_rate,
    detect_quote_currency,
    _as_series,
    _ensure_float_series,
    _align_fx_asof,
    convert_currency_to_pln,
    convert_price_series_to_pln,
    _resolve_symbol_candidates,
)

# Suppress noisy yfinance download warnings (e.g., "1 Failed download: ...")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

PAIR = "PLNJPY=X"
DEFAULT_HORIZONS = [1, 3, 7, 21, 63, 126, 252]
NOTIONAL_PLN = 1_000_000  # for profit column

# Transaction-cost/slippage hurdle: minimum absolute edge required to act
# Can be overridden via environment variable EDGE_FLOOR (e.g., 0.10)
try:
    _edge_env = os.getenv("EDGE_FLOOR", "0.10")
    EDGE_FLOOR = float(_edge_env)
except Exception:
    EDGE_FLOOR = 0.10
# Clamp to a reasonable range to avoid misuse
EDGE_FLOOR = float(np.clip(EDGE_FLOOR, 0.0, 1.5))


@dataclass(frozen=True)
class Signal:
    horizon_days: int
    score: float          # edge in z units (mu_H/sigma_H with filters)
    p_up: float           # P(return>0)
    exp_ret: float        # expected log return over horizon
    ci_low: float         # lower bound of expected log return CI
    ci_high: float        # upper bound of expected log return CI
    profit_pln: float     # expected profit in PLN for NOTIONAL_PLN invested
    profit_ci_low_pln: float  # low CI bound for profit in PLN
    profit_ci_high_pln: float # high CI bound for profit in PLN
    position_strength: float  # fractional Kelly suggestion (0..1)
    regime: str               # detected regime label
    label: str                # BUY/HOLD/SELL or STRONG BUY/SELL





def fetch_px_asset(asset: str, start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    """
    Return a price series for the requested asset expressed in PLN terms when needed.
    - PLNJPY=X: native series (JPY per PLN); title indicates JPY per PLN.
    - Gold: try XAUUSD=X, GC=F, XAU=X; convert USD to PLN via USDPLN=X (or robust alternatives) → PLN per troy ounce.
    - Silver: try XAGUSD=X, SI=F, XAG=X; convert USD to PLN via USDPLN=X (or robust alternatives) → PLN per troy ounce.
    - Bitcoin (BTC-USD): convert USD to PLN via USDPLN → PLN per BTC.
    - MicroStrategy (MSTR): convert USD share price to PLN via USDPLN → PLN per share.
    - Generic equities/ETFs: fetch in native quote currency and convert to PLN via detected FX.
    Returns (px_series, title_suffix) where title_suffix describes units.
    """
    asset = asset.strip().upper()
    if asset == "PLNJPY=X":
        px = _fetch_px_symbol(asset, start, end)
        title = "Polish Zloty vs Japanese Yen (PLNJPY=X) — JPY per PLN"
        return px, title

    # Bitcoin in USD → PLN
    if asset in ("BTC-USD", "BTCUSD=X"):
        # Prefer robust USD path (avoid unreliable BTC-PLN tickers that 404)
        btc_px, _used = _fetch_with_fallback([asset] if asset == "BTC-USD" else [asset, "BTC-USD"], start, end)
        btc_px = _ensure_float_series(btc_px)
        # Use USD→PLN leg expanded to BTC date range and robust asof alignment
        usdpln_px = convert_currency_to_pln("USD", start, end, native_index=btc_px.index)
        usdpln_aligned = _align_fx_asof(btc_px, usdpln_px, max_gap_days=7)
        if usdpln_aligned.isna().all():
            usdpln_aligned = usdpln_px.reindex(btc_px.index).ffill().bfill()
        usdpln_aligned = _ensure_float_series(usdpln_aligned)
        # Direct vectorized conversion
        px_pln = (btc_px * usdpln_aligned).dropna()
        px_pln.name = "px"
        if px_pln.empty:
            raise RuntimeError("No overlap between BTC-USD and USDPLN data to compute PLN price")
        # Display name
        disp = "Bitcoin"
        return px_pln, f"{disp} (BTC-USD) — PLN per BTC"

    # MicroStrategy equity (USD) → PLN
    if asset == "MSTR":
        mstr_px = _fetch_px_symbol("MSTR", start, end)
        mstr_px = _ensure_float_series(mstr_px)
        # Use USD→PLN leg expanded to MSTR date range and robust asof alignment
        usdpln_px = convert_currency_to_pln("USD", start, end, native_index=mstr_px.index)
        usdpln_aligned = _align_fx_asof(mstr_px, usdpln_px, max_gap_days=7)
        if usdpln_aligned.isna().all():
            usdpln_aligned = usdpln_px.reindex(mstr_px.index).ffill().bfill()
        usdpln_aligned = _ensure_float_series(usdpln_aligned)
        # Direct vectorized conversion (ensure 1-D float Series)
        px_pln = (mstr_px * usdpln_aligned).dropna()
        px_pln.name = "px"
        if px_pln.empty:
            raise RuntimeError("No overlap between MSTR and USDPLN data to compute PLN price")
        disp = _resolve_display_name("MSTR") or "MicroStrategy"
        return px_pln, f"{disp} (MSTR) — PLN per share"

    # Metals in USD → convert to PLN
    if asset in ("XAUUSD=X", "GC=F", "XAU=X", "XAGUSD=X", "SI=F", "XAG=X"):
        if asset.startswith("XAU") or asset in ("GC=F", "XAU=X"):
            candidates = ["GC=F", "XAU=X", "XAUUSD=X"]
            if asset not in candidates:
                candidates = [asset] + candidates
            metal_px, used = _fetch_with_fallback(candidates, start, end)
            metal_px = _ensure_float_series(metal_px)
            metal_name = "Gold"
        else:
            candidates = ["SI=F", "XAG=X", "XAGUSD=X"]
            if asset not in candidates:
                candidates = [asset] + candidates
            metal_px, used = _fetch_with_fallback(candidates, start, end)
            metal_px = _ensure_float_series(metal_px)
            metal_name = "Silver"
        usdpln_px = fetch_usd_to_pln_exchange_rate(start, end)
        usdpln_aligned = usdpln_px.reindex(metal_px.index).ffill()
        df = pd.concat([metal_px, usdpln_aligned], axis=1).dropna()
        df.columns = ["metal_usd", "usdpln"]
        px_pln = (df["metal_usd"] * df["usdpln"]).rename("px")
        title = f"{metal_name} ({used}) — PLN per troy oz"
        if px_pln.empty:
            raise RuntimeError(f"No overlap between {metal_name} and USDPLN data to compute PLN price")
        return px_pln, title

    # Generic: resolve symbol candidates and convert to PLN using detected currency
    candidates = _resolve_symbol_candidates(asset)
    px_native = None
    used_sym = None
    last_err: Optional[Exception] = None
    for sym in candidates:
        try:
            s = _fetch_px_symbol(sym, start, end)
            px_native = s
            used_sym = sym
            break
        except Exception as e:
            last_err = e
            continue
    if px_native is None:
        raise last_err if last_err else RuntimeError(f"No data for {asset}")

    px_native = _ensure_float_series(px_native)
    qcy = detect_quote_currency(used_sym)
    px_pln, _ = convert_price_series_to_pln(px_native, qcy, start, end)
    if px_pln is None or px_pln.empty:
        raise RuntimeError(f"No overlapping FX data to convert {used_sym} to PLN")
    # Title with full name
    disp = _resolve_display_name(used_sym)
    title = f"{disp} ({used_sym}) — PLN per share"
    return px_pln, title


# -------------------------
# Features
# -------------------------

def _garch11_mle(ret: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
    """Estimate a GARCH(1,1) with normal errors via MLE.
    Returns (sigma_series, params_dict). Falls back by raising on failure.
    Model: r_t = mu_t + e_t, e_t ~ N(0, h_t), h_t = omega + alpha*e_{t-1}^2 + beta*h_{t-1}
    We estimate on de-meaned returns (mean 0) and treat residuals as r_t.

    Level-7: we also approximate parameter uncertainty by computing the observed
    information (numeric Hessian of the negative log-likelihood) at the optimum
    and inverting it to obtain an approximate covariance matrix for (omega,alpha,beta).
    """
    from scipy.optimize import minimize

    r = _ensure_float_series(ret).dropna().astype(float)
    if len(r) < 200:
        raise RuntimeError("Too few observations for stable GARCH(1,1) MLE (need >=200)")

    # De-mean for conditional variance fit
    r = r - r.mean()
    T = len(r)
    r2 = r.values**2
    var0 = float(np.nanvar(r.values)) if T > 1 else 1e-6

    # Parameter transform: ensure omega>0, alpha>=0, beta>=0, alpha+beta<0.999
    def nll(params):
        omega, alpha, beta = params
        # Hard penalties if constraints violated
        if omega <= 1e-12 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 0.999:
            return 1e12
        h = np.empty(T, dtype=float)
        # Initialize with unconditional variance to speed convergence
        try:
            h0 = omega / max(1e-12, 1.0 - alpha - beta)
            if not np.isfinite(h0) or h0 <= 0:
                h0 = var0
        except Exception:
            h0 = var0
        h[0] = max(1e-12, h0)
        for t in range(1, T):
            h[t] = omega + alpha * r2[t-1] + beta * h[t-1]
            if not np.isfinite(h[t]) or h[t] <= 0:
                h[t] = 1e-8
        # Normal likelihood (up to constant): 0.5*(log h_t + r_t^2/h_t)
        ll_terms = 0.5*(np.log(h) + r2 / h)
        if not np.all(np.isfinite(ll_terms)):
            return 1e12
        return float(np.sum(ll_terms))

    # Multiple starting points for robustness
    inits = [
        (0.1*var0*(1-0.1-0.8), 0.1, 0.8),
        (0.05*var0*(1-0.05-0.9), 0.05, 0.9),
        (0.2*var0*(1-0.15-0.7), 0.15, 0.7),
    ]
    best = (None, np.inf)
    best_params = None
    bounds = [(1e-12, 10.0*var0), (0.0, 0.999), (0.0, 0.999)]
    constraints = ({'type': 'ineq', 'fun': lambda p: 0.999 - (p[1] + p[2])},)

    for x0 in inits:
        try:
            res = minimize(nll, x0=np.array(x0, dtype=float), method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200})
            if res.success and res.fun < best[1]:
                best = (res, res.fun)
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        raise RuntimeError("GARCH(1,1) MLE failed to converge from all starts")

    omega, alpha, beta = [float(v) for v in best_params]

    # Rebuild conditional variance series with optimal params
    h = np.empty(T, dtype=float)
    try:
        h0 = omega / max(1e-12, 1.0 - alpha - beta)
        if not np.isfinite(h0) or h0 <= 0:
            h0 = var0
    except Exception:
        h0 = var0
    h[0] = max(1e-10, h0)
    for t in range(1, T):
        h[t] = omega + alpha * r2[t-1] + beta * h[t-1]
        if not np.isfinite(h[t]) or h[t] <= 0:
            h[t] = 1e-8
    sigma = np.sqrt(h)
    vol = pd.Series(sigma, index=r.index, name='vol_garch')

    # Approximate covariance of parameters via numeric Hessian of nll at optimum
    def _approx_hessian(x: np.ndarray) -> Optional[np.ndarray]:
        try:
            x = np.asarray(x, dtype=float)
            k = x.size
            H = np.zeros((k, k), dtype=float)
            # Step sizes scaled to parameter magnitudes
            eps_base = 1e-6
            h_vec = np.maximum(np.abs(x) * 1e-3, eps_base)
            f0 = nll(x)
            # Diagonal second derivatives
            for i in range(k):
                ei = np.zeros(k); ei[i] = h_vec[i]
                f_plus = nll(x + ei)
                f_minus = nll(x - ei)
                H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h_vec[i] ** 2)
            # Off-diagonals via mixed partials (central)
            for i in range(k):
                for j in range(i+1, k):
                    ei = np.zeros(k); ei[i] = h_vec[i]
                    ej = np.zeros(k); ej[j] = h_vec[j]
                    f_pp = nll(x + ei + ej)
                    f_pm = nll(x + ei - ej)
                    f_mp = nll(x - ei + ej)
                    f_mm = nll(x - ei - ej)
                    mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_vec[i] * h_vec[j])
                    H[i, j] = mixed
                    H[j, i] = mixed
            return H
        except Exception:
            return None

    cov = None
    se = None
    try:
        H = _approx_hessian(np.array([omega, alpha, beta], dtype=float))
        if H is not None:
            # Regularize slightly to improve conditioning
            lam = 1e-8
            H_reg = H + lam * np.eye(3)
            cov_try = np.linalg.pinv(H_reg)
            # Ensure symmetry and positive diagonals
            cov_try = 0.5 * (cov_try + cov_try.T)
            if np.all(np.isfinite(cov_try)) and np.all(np.diag(cov_try) >= 0):
                cov = cov_try
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        cov = None
        se = None

    # Compute final log-likelihood at optimum (negative nll)
    final_nll = float(best[1])
    final_ll = float(-final_nll)
    
    params: Dict[str, float] = {
        "omega": omega, 
        "alpha": alpha, 
        "beta": beta, 
        "converged": True,
        "log_likelihood": final_ll,
        "nll": final_nll,
        "n_obs": int(T),
        "aic": float(2.0 * 3 - 2.0 * final_ll),  # AIC = 2k - 2*ln(L), k=3 params
        "bic": float(3 * np.log(T) - 2.0 * final_ll),  # BIC = k*ln(n) - 2*ln(L)
    }
    if cov is not None:
        params["cov"] = cov.tolist()
    if se is not None:
        params["se_omega"], params["se_alpha"], params["se_beta"] = [float(x) for x in se]
    return vol, params


def _fit_student_nu_mle(z: pd.Series, min_n: int = 200, bounds: Tuple[float, float] = (4.5, 500.0)) -> Dict[str, float]:
    """Fit global Student-t degrees of freedom (nu) via MLE on standardized residuals z.
    - z should be approximately IID with unit scale (i.e., residuals divided by conditional sigma).
    - Returns a dict: {"nu_hat": float, "ll": float, "n": int, "converged": bool}.
    - On failure or insufficient data, returns a conservative default with converged=False.
    """
    from scipy.optimize import minimize

    if z is None or not isinstance(z, pd.Series) or z.empty:
        return {"nu_hat": 50.0, "ll": float("nan"), "n": 0, "converged": False}

    zz = pd.to_numeric(z, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    # Remove zeros that may indicate degenerate scaling (not necessary but harmless)
    zz = zz[np.isfinite(zz.values)]
    n = int(zz.shape[0])
    if n < max(50, min_n):
        # too short: near-normal default
        return {"nu_hat": 50.0, "ll": float("nan"), "n": n, "converged": False}

    x = zz.values.astype(float)

    def nll(nu_val: float) -> float:
        # Bound inside objective to avoid domain errors
        nu_b = float(np.clip(nu_val, bounds[0], bounds[1]))
        try:
            # Use scipy.stats.t logpdf with df=nu_b, loc=0, scale=1
            lp = student_t.logpdf(x, df=nu_b)
            if not np.all(np.isfinite(lp)):
                return 1e12
            return float(-np.sum(lp))
        except Exception:
            return 1e12

    # Multi-start initializations
    starts = [5.5, 8.0, 12.0, 20.0, 50.0, 100.0, 200.0]
    best = (None, np.inf)
    for s0 in starts:
        x0 = np.array([float(np.clip(s0, bounds[0], bounds[1]))], dtype=float)
        try:
            res = minimize(lambda v: nll(v[0]), x0=x0, method="L-BFGS-B", bounds=[bounds], options={"maxiter": 200})
            if res.success and res.fun < best[1]:
                best = (res, res.fun)
        except Exception:
            continue

    if best[0] is None:
        return {"nu_hat": 50.0, "ll": float("nan"), "n": n, "converged": False}

    nu_hat = float(np.clip(best[0].x[0], bounds[0], bounds[1]))
    ll = float(-best[1])
    return {"nu_hat": nu_hat, "ll": ll, "n": n, "converged": True}


def compute_features(px: pd.Series) -> Dict[str, pd.Series]:
    # Protect log conversion from garbage ticks and non-positive prices
    px = _ensure_float_series(px)
    px = px.replace([np.inf, -np.inf], np.nan).dropna()
    px = px[px > 0]

    log_px = np.log(px)
    ret = log_px.diff().dropna()
    ret = winsorize(ret, p=0.01)
    ret.name = "ret"

    # Multi-speed EWMA for drift and vol
    mu_fast = ret.ewm(span=21, adjust=False).mean()
    mu_slow = ret.ewm(span=126, adjust=False).mean()

    vol_fast = ret.ewm(span=21, adjust=False).std()
    vol_slow = ret.ewm(span=126, adjust=False).std()

    # Prefer GARCH(1,1) volatility via MLE; fallback to EWMA blend on failure
    try:
        vol_garch, garch_params = _garch11_mle(ret)
        # Align to ret index and name "vol" for downstream compatibility
        vol = vol_garch.reindex(ret.index).rename("vol")
        vol_source = "garch11"
    except Exception:
        # Blend vol: fast reacts, slow stabilizes
        vol = (0.6 * vol_fast + 0.4 * vol_slow).rename("vol")
        garch_params = {}
        vol_source = "ewma_fallback"
    # Robust global volatility floor to avoid feedback loops when vol collapses recently:
    # - Use a lagged expanding 10th percentile over the entire history (no look-ahead)
    # - Add a relative floor vs long-run median and a small absolute epsilon
    # - Provide an early-history fallback to ensure continuity
    MIN_HIST = 252
    LAG_DAYS = 21  # ~1 trading month lag to avoid immediate reaction to shocks
    abs_floor = 1e-6
    try:
        vol_lag = vol.shift(LAG_DAYS)
        # Expanding quantile and median computed on information up to t-LAG
        global_floor_series = vol_lag.expanding(MIN_HIST).quantile(0.10)
        long_med = vol_lag.expanding(MIN_HIST).median()
        rel_floor = 0.10 * long_med
        # Combine available floors at each timestamp
        floor_candidates = pd.concat([
            global_floor_series.rename("gf"),
            rel_floor.rename("rf")
        ], axis=1)
        floor_t = floor_candidates.max(axis=1)
        # Early history fallback (before MIN_HIST+LAG_DAYS)
        early_med = vol.rolling(63, min_periods=20).median()
        early_floor = np.maximum(0.10 * early_med, abs_floor)
        floor_t = floor_t.combine_first(early_floor)
        # Ensure absolute epsilon
        floor_t = np.maximum(floor_t, abs_floor)
        # Apply the floor index-wise
        vol = np.maximum(vol, floor_t)
    except Exception:
        # Fallback to a simple median-based floor if expanding quantile not available
        fallback_floor = np.maximum(vol.rolling(252, min_periods=63).median() * 0.10, abs_floor)
        vol = np.maximum(vol, fallback_floor)

    # Vol regime (relative to 1y median) — kept for diagnostics, not for shrinkage
    vol_med = vol.rolling(252).median()
    vol_regime = vol / vol_med

    # Drift from blended EWMA (fast/slow). With GARCH handling volatility coherently,
    # avoid additional shrinkage by volatility regime to prevent double-counting.
    mu_blend = 0.5 * mu_fast + 0.5 * mu_slow
    mu = mu_blend
    
    # Optional smoothing of drift to reduce whipsaws (backtest-safe: uses t-1 info)
    # mu_t = 0.7 * mu_blend_t + 0.3 * mu_{t-1}
    try:
        mu_smoothed = 0.7 * mu_blend + 0.3 * mu.shift(1)
        # Use smoothed where available; fallback to baseline mu
        mu = mu_smoothed.combine_first(mu)
    except Exception:
        # In case of alignment/type issues, keep baseline mu
        pass

    # Trend filter (200D z-distance) - kept for diagnostics
    sma200 = px.rolling(200).mean()
    trend_z = (px - sma200) / px.rolling(200).std()

    # HMM regime-aware drift estimation (replaces threshold-based shrinkage)
    # Fit preliminary HMM to get regime posteriors for drift adjustment
    hmm_result_prelim = fit_hmm_regimes(
        {"ret": ret, "vol": vol},
        n_states=3,
        random_seed=42
    )
    
    # Use HMM regime posteriors to weight drift estimates if available
    if hmm_result_prelim is not None and "posterior_probs" in hmm_result_prelim:
        try:
            posterior_probs = hmm_result_prelim["posterior_probs"]
            regime_means = hmm_result_prelim["means"][:, 0]  # drift component from each regime
            
            # Align posteriors with mu_blend index
            posterior_aligned = posterior_probs.reindex(mu_blend.index).ffill().fillna(0.333)
            
            # Regime-conditional drift: weight mu_blend by regime persistence
            # In calm regimes: trust sample drift more (less shrinkage)
            # In crisis regimes: shrink toward regime-learned drift (more conservative)
            regime_names = hmm_result_prelim["regime_names"]
            calm_idx = [k for k, v in regime_names.items() if v == "calm"]
            crisis_idx = [k for k, v in regime_names.items() if v == "crisis"]
            
            p_calm = posterior_aligned.iloc[:, calm_idx[0]].values if calm_idx else np.zeros(len(mu_blend))
            p_crisis = posterior_aligned.iloc[:, crisis_idx[0]].values if crisis_idx else np.zeros(len(mu_blend))
            
            # Shrinkage weight: high in crisis (shrink toward 0), low in calm (trust sample)
            shrinkage = 0.5 + 0.3 * p_crisis - 0.2 * p_calm
            shrinkage = np.clip(shrinkage, 0.2, 0.9)
            
            # Posterior drift with regime-aware shrinkage (no hard-coded thresholds)
            mu_post = pd.Series(
                (1.0 - shrinkage) * mu_blend.values + shrinkage * 0.0,  # shrink toward zero in crisis
                index=mu_blend.index
            )
            
        except Exception:
            # Fallback: use simple blend without shrinkage
            mu_post = mu_blend.copy()
    else:
        # HMM not available: use simple blend without threshold-based shrinkage
        mu_post = mu_blend.copy()
    
    # Robust fallback for NaNs
    mu_post = mu_post.fillna(mu_blend).fillna(0.0)

    # Short-term mean-reversion z (5d move over 1m vol)
    r5 = (log_px - log_px.shift(5))
    rv_1m = ret.rolling(21).std() * math.sqrt(5)
    z5 = r5 / rv_1m

    # Rolling skewness (directional asymmetry) and excess kurtosis (Fisher)
    skew = ret.rolling(252, min_periods=63).skew()
    # Optional stabilization: smooth skew to avoid warm-up swings when it first becomes defined
    try:
        skew_s = skew.ewm(span=30, adjust=False).mean()
    except Exception:
        skew_s = skew
    ex_kurt = ret.rolling(252, min_periods=63).kurt()  # normal ~ 0
    # Convert excess kurtosis to t degrees of freedom via: excess = 6/(nu-4) => nu = 4 + 6/excess
    # Handle near-zero/negative excess by mapping to large nu (approx normal)
    eps = 1e-6
    nu = 4.0 + 6.0 / ex_kurt.where(ex_kurt > eps, np.nan)
    nu = nu.fillna(1e6)  # ~normal
    # Clip degrees of freedom to a stable range to prevent extreme tail chaos in flash crashes
    nu = nu.clip(lower=4.5, upper=500.0)

    # Fit global Student-t tail parameter once via MLE on standardized residuals (Level-7 rule)
    try:
        # Residuals using posterior drift (more conservative) and current per-day vol
        mu_post_aligned = pd.Series(mu_post, index=ret.index).astype(float)
        vol_aligned = pd.Series(vol, index=ret.index).astype(float)
        resid = (ret - mu_post_aligned).replace([np.inf, -np.inf], np.nan)
        z_std = resid / vol_aligned.replace(0.0, np.nan)
        z_std = z_std.replace([np.inf, -np.inf], np.nan).dropna()
        nu_info = _fit_student_nu_mle(z_std, min_n=200, bounds=(4.5, 500.0))
        nu_hat = float(nu_info.get("nu_hat", 50.0))
    except Exception:
        nu_info = {"nu_hat": 50.0, "ll": float("nan"), "n": 0, "converged": False}
        nu_hat = 50.0

    # t-stat style momentum: cum return / realized vol over window
    def mom_t(days: int) -> pd.Series:
        cum = (log_px - log_px.shift(days))
        rv = ret.rolling(days).std() * math.sqrt(days)
        return cum / rv

    mom21 = mom_t(21)
    mom63 = mom_t(63)
    mom126 = mom_t(126)
    mom252 = mom_t(252)

    # Reuse HMM result from drift estimation (avoid duplicate fitting)
    hmm_result = hmm_result_prelim

    return {
        "px": px,
        "ret": ret,
        "mu": mu,
        "mu_post": mu_post,
        "mu_blend": mu_blend,
        "vol": vol,
        "vol_regime": vol_regime,
        "trend_z": trend_z,
        "z5": z5,
        "nu": nu,               # rolling, for diagnostics only
        "nu_hat": pd.Series([nu_hat], index=[ret.index[-1]]) if len(ret.index)>0 else pd.Series([nu_hat]),
        "nu_info": nu_info,     # dict metadata
        "skew": skew,
        "skew_s": skew_s,
        "mom21": mom21,
        "mom63": mom63,
        "mom126": mom126,
        "mom252": mom252,
        # meta (not series)
        "vol_source": vol_source,
        "garch_params": garch_params,
        # HMM regime detection
        "hmm_result": hmm_result,
    }


# -------------------------
# HMM Regime Detection (Formal Bayesian Inference)
# -------------------------

def fit_hmm_regimes(feats: Dict[str, pd.Series], n_states: int = 3, random_seed: int = 42) -> Optional[Dict]:
    """
    Fit a Hidden Markov Model with Gaussian emissions to detect market regimes.
    
    Each regime (state) has:
    - Its own μ (drift) dynamics captured by emission mean
    - Its own σ (volatility) dynamics captured by emission covariance
    - Persistence captured by transition matrix
    
    Args:
        feats: Feature dictionary from compute_features()
        n_states: Number of hidden states (default 3: calm, trending, crisis)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with HMM model, state sequence, and regime metadata, or None on failure
    """
    if not HMM_AVAILABLE:
        return None
        
    try:
        # Extract returns and volatility as observations
        ret = feats.get("ret", pd.Series(dtype=float))
        vol = feats.get("vol", pd.Series(dtype=float))
        
        if ret.empty or vol.empty:
            return None
            
        # Align and clean data
        df = pd.concat([ret, vol], axis=1, join="inner").dropna()
        if len(df) < 300:  # Need sufficient history for stable HMM
            return None
            
        df.columns = ["ret", "vol"]
        X = df.values  # Shape (T, 2): returns and volatility as features
        
        # Fit Gaussian HMM with full covariance (allows each state its own μ and σ)
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=random_seed,
            verbose=False
        )
        
        model.fit(X)
        
        # Infer hidden state sequence (Viterbi for most likely path)
        states = model.predict(X)
        
        # Posterior probabilities for each state at each time
        posteriors = model.predict_proba(X)
        
        # Identify regime characteristics from emission parameters
        means = model.means_  # Shape (n_states, 2): [drift, vol] per state
        covars = model.covars_  # Shape (n_states, 2, 2)
        transmat = model.transmat_  # Shape (n_states, n_states)
        
        # Label states by volatility level: calm < normal < crisis
        vol_means = means[:, 1]  # volatility component
        sorted_indices = np.argsort(vol_means)
        
        regime_names = {
            sorted_indices[0]: "calm",
            sorted_indices[1]: "trending" if n_states == 3 else "normal",
            sorted_indices[2]: "crisis" if n_states == 3 else "volatile"
        }
        
        # Build regime series aligned with returns index
        regime_series = pd.Series(
            [regime_names.get(s, f"state_{s}") for s in states],
            index=df.index,
            name="regime"
        )
        
        # Posterior probability series (one per state)
        posterior_df = pd.DataFrame(
            posteriors,
            index=df.index,
            columns=[regime_names.get(i, f"state_{i}") for i in range(n_states)]
        )
        
        # Compute log-likelihood and information criteria for model diagnostics
        try:
            log_likelihood = float(model.score(X))
            n_obs = int(len(X))
            # Count free parameters: n_states-1 for initial probs, n_states*(n_states-1) for transitions,
            # n_states*n_features for means, n_states*n_features*(n_features+1)/2 for full covariance
            n_features = X.shape[1]
            n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features + n_states * n_features * (n_features + 1) // 2
            aic = float(2.0 * n_params - 2.0 * log_likelihood)
            bic = float(n_params * np.log(n_obs) - 2.0 * log_likelihood)
        except Exception:
            log_likelihood = float("nan")
            n_obs = int(len(X))
            n_params = 0
            aic = float("nan")
            bic = float("nan")
        
        return {
            "model": model,
            "regime_series": regime_series,
            "posterior_probs": posterior_df,
            "states": states,
            "means": means,
            "covars": covars,
            "transmat": transmat,
            "regime_names": regime_names,
            "n_states": n_states,
            "log_likelihood": log_likelihood,
            "n_obs": n_obs,
            "n_params": n_params,
            "aic": aic,
            "bic": bic,
        }
        
    except Exception as e:
        # Silent fallback on HMM failure
        return None


def track_parameter_stability(ret: pd.Series, window_days: int = 252, step_days: int = 63) -> Dict[str, pd.DataFrame]:
    """
    Track GARCH parameter stability over time using rolling window estimation.
    
    Fits GARCH(1,1) on expanding windows to detect parameter drift.
    Returns time series of parameters, standard errors, and log-likelihoods.
    
    Args:
        ret: Returns series
        window_days: Minimum window size for initial fit
        step_days: Days between refits (trades off compute vs resolution)
        
    Returns:
        Dictionary with DataFrames tracking parameters over time
    """
    ret_clean = _ensure_float_series(ret).dropna()
    if len(ret_clean) < max(300, window_days):
        return {}
    
    # Time points to evaluate (start at window_days, step forward)
    dates = ret_clean.index
    eval_dates = []
    for i in range(window_days, len(dates), step_days):
        eval_dates.append(dates[i])
    
    if not eval_dates:
        return {}
    
    # Storage for parameter evolution
    records = []
    
    for eval_date in eval_dates:
        # Use expanding window up to eval_date
        window_ret = ret_clean.loc[:eval_date]
        
        # Try to fit GARCH
        try:
            _, params = _garch11_mle(window_ret)
            record = {
                "date": eval_date,
                "omega": params.get("omega", float("nan")),
                "alpha": params.get("alpha", float("nan")),
                "beta": params.get("beta", float("nan")),
                "se_omega": params.get("se_omega", float("nan")),
                "se_alpha": params.get("se_alpha", float("nan")),
                "se_beta": params.get("se_beta", float("nan")),
                "log_likelihood": params.get("log_likelihood", float("nan")),
                "aic": params.get("aic", float("nan")),
                "bic": params.get("bic", float("nan")),
                "n_obs": params.get("n_obs", 0),
                "converged": params.get("converged", False),
            }
            records.append(record)
        except Exception:
            # Skip windows where GARCH fails
            continue
    
    if not records:
        return {}
    
    df = pd.DataFrame(records).set_index("date")
    
    # Compute parameter drift statistics (rolling z-score of parameter changes)
    param_cols = ["omega", "alpha", "beta"]
    drift_stats = {}
    
    for col in param_cols:
        if col in df.columns:
            changes = df[col].diff()
            se_col = f"se_{col}"
            if se_col in df.columns:
                # Normalized change (z-score): change / standard error
                z_change = changes / df[se_col].replace(0, np.nan)
                drift_stats[f"{col}_drift_z"] = z_change
    
    drift_df = pd.DataFrame(drift_stats, index=df.index)
    
    return {
        "param_evolution": df,
        "param_drift": drift_df,
    }


def walk_forward_validation(px: pd.Series, train_days: int = 504, test_days: int = 21, horizons: List[int] = [1, 21, 63]) -> Dict[str, pd.DataFrame]:
    """
    Perform walk-forward out-of-sample testing to validate predictive power.
    
    Splits data into non-overlapping train/test windows, fits model on train,
    predicts on test, and tracks hit rates and prediction errors.
    
    Args:
        px: Price series
        train_days: Training window size (days)
        test_days: Test window size (days)
        horizons: Forecast horizons to test
        
    Returns:
        Dictionary with out-of-sample performance metrics
    """
    px_clean = _ensure_float_series(px).dropna()
    if len(px_clean) < train_days + test_days + max(horizons):
        return {}
    
    log_px = np.log(px_clean)
    dates = px_clean.index
    
    # Define walk-forward windows
    windows = []
    start_idx = 0
    while start_idx + train_days + test_days <= len(dates):
        train_end_idx = start_idx + train_days
        test_end_idx = min(train_end_idx + test_days, len(dates))
        
        train_dates = dates[start_idx:train_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]
        
        if len(test_dates) > 0:
            windows.append({
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
            })
        
        # Move forward by test_days (non-overlapping)
        start_idx = test_end_idx
    
    if not windows:
        return {}
    
    # Track predictions and outcomes for each horizon
    results = {h: [] for h in horizons}
    
    for window in windows:
        # Fit features on training data
        train_px = px_clean.loc[window["train_start"]:window["train_end"]]
        
        try:
            train_feats = compute_features(train_px)
            
            # Get predictions at end of training window
            mu_now = safe_last(train_feats.get("mu_post", pd.Series([0.0])))
            vol_now = safe_last(train_feats.get("vol", pd.Series([1.0])))
            
            if not np.isfinite(mu_now):
                mu_now = 0.0
            if not np.isfinite(vol_now) or vol_now <= 0:
                vol_now = 1.0
            
            # For each horizon, predict and measure actual outcome
            test_log_px = log_px.loc[window["test_start"]:window["test_end"]]
            train_end_log_px = float(log_px.loc[window["train_end"]])
            
            for H in horizons:
                # Predicted return over H days
                pred_ret_H = mu_now * H
                pred_sign = np.sign(pred_ret_H) if pred_ret_H != 0 else 0
                
                # Actual return H days forward from train_end
                try:
                    forward_idx = dates.get_loc(window["train_end"]) + H
                    if forward_idx < len(dates):
                        forward_date = dates[forward_idx]
                        actual_log_px = float(log_px.loc[forward_date])
                        actual_ret_H = actual_log_px - train_end_log_px
                        actual_sign = np.sign(actual_ret_H) if actual_ret_H != 0 else 0
                        
                        # Prediction error
                        pred_error = actual_ret_H - pred_ret_H
                        
                        # Direction hit (1 if signs match, 0 otherwise)
                        hit = 1 if (pred_sign * actual_sign > 0) else 0
                        
                        results[H].append({
                            "train_end": window["train_end"],
                            "forecast_date": forward_date,
                            "predicted_return": pred_ret_H,
                            "actual_return": actual_ret_H,
                            "prediction_error": pred_error,
                            "direction_hit": hit,
                        })
                except Exception:
                    continue
                    
        except Exception:
            continue
    
    # Aggregate results into DataFrames
    oos_metrics = {}
    for H in horizons:
        if results[H]:
            df = pd.DataFrame(results[H]).set_index("train_end")
            
            # Compute cumulative statistics
            hit_rate = df["direction_hit"].mean() if len(df) > 0 else float("nan")
            mean_error = df["prediction_error"].mean() if len(df) > 0 else float("nan")
            rmse = np.sqrt((df["prediction_error"] ** 2).mean()) if len(df) > 0 else float("nan")
            
            oos_metrics[f"H{H}"] = {
                "predictions": df,
                "hit_rate": float(hit_rate),
                "mean_error": float(mean_error),
                "rmse": float(rmse),
                "n_forecasts": len(df),
            }
    
    return oos_metrics


def compute_all_diagnostics(px: pd.Series, feats: Dict[str, pd.Series], enable_oos: bool = False) -> Dict:
    """
    Compute comprehensive diagnostics: log-likelihood monitoring, parameter stability, 
    and optionally out-of-sample tests.
    
    Args:
        px: Price series
        feats: Feature dictionary from compute_features
        enable_oos: If True, run expensive out-of-sample validation
        
    Returns:
        Dictionary with all diagnostic metrics
    """
    diagnostics = {}
    
    # 1. Log-likelihood monitoring from fitted models
    garch_params = feats.get("garch_params", {})
    if isinstance(garch_params, dict):
        diagnostics["garch_log_likelihood"] = garch_params.get("log_likelihood", float("nan"))
        diagnostics["garch_aic"] = garch_params.get("aic", float("nan"))
        diagnostics["garch_bic"] = garch_params.get("bic", float("nan"))
        diagnostics["garch_n_obs"] = garch_params.get("n_obs", 0)
    
    hmm_result = feats.get("hmm_result")
    if hmm_result is not None and isinstance(hmm_result, dict):
        diagnostics["hmm_log_likelihood"] = hmm_result.get("log_likelihood", float("nan"))
        diagnostics["hmm_aic"] = hmm_result.get("aic", float("nan"))
        diagnostics["hmm_bic"] = hmm_result.get("bic", float("nan"))
        diagnostics["hmm_n_obs"] = hmm_result.get("n_obs", 0)
    
    nu_info = feats.get("nu_info", {})
    if isinstance(nu_info, dict):
        diagnostics["student_t_log_likelihood"] = nu_info.get("ll", float("nan"))
        diagnostics["student_t_nu"] = nu_info.get("nu_hat", float("nan"))
        diagnostics["student_t_n_obs"] = nu_info.get("n", 0)
    
    # 2. Parameter stability tracking (expensive, only if enough data)
    ret = feats.get("ret", pd.Series(dtype=float))
    if not ret.empty and len(ret) >= 600:
        try:
            stability = track_parameter_stability(ret, window_days=252, step_days=126)
            if stability:
                diagnostics["parameter_stability"] = stability
                
                # Summary statistics: recent drift magnitude
                param_drift = stability.get("param_drift")
                if param_drift is not None and not param_drift.empty:
                    recent_drift = param_drift.tail(1)
                    for col in param_drift.columns:
                        val = safe_last(param_drift[col])
                        diagnostics[f"recent_{col}"] = float(val) if np.isfinite(val) else float("nan")
        except Exception:
            pass
    
    # 3. Out-of-sample tests (very expensive, optional)
    if enable_oos and not px.empty and len(px) >= 800:
        try:
            oos_metrics = walk_forward_validation(px, train_days=504, test_days=21, horizons=[1, 21, 63])
            if oos_metrics:
                diagnostics["out_of_sample"] = oos_metrics
                
                # Summary: hit rates for each horizon
                for horizon_key, metrics in oos_metrics.items():
                    if isinstance(metrics, dict):
                        hit_rate = metrics.get("hit_rate", float("nan"))
                        diagnostics[f"oos_{horizon_key}_hit_rate"] = float(hit_rate)
        except Exception:
            pass
    
    return diagnostics


def infer_current_regime(feats: Dict[str, pd.Series], hmm_result: Optional[Dict] = None) -> Tuple[str, Dict[str, float]]:
    """
    Infer the current market regime using posterior inference from HMM.
    
    Args:
        feats: Feature dictionary
        hmm_result: Result from fit_hmm_regimes(), or None to use threshold fallback
        
    Returns:
        Tuple of (regime_label, regime_metadata_dict)
        regime_label: "calm", "trending", "crisis", or threshold-based fallback
        regime_metadata: probabilities and diagnostics
    """
    # If HMM available and fitted, use posterior inference
    if hmm_result is not None and "regime_series" in hmm_result:
        try:
            regime_series = hmm_result["regime_series"]
            posterior_probs = hmm_result["posterior_probs"]
            
            if not regime_series.empty:
                current_regime = regime_series.iloc[-1]
                current_probs = posterior_probs.iloc[-1].to_dict()
                
                return str(current_regime), {
                    "method": "hmm_posterior",
                    "probabilities": current_probs,
                    "persistence": float(hmm_result["transmat"][hmm_result["states"][-1], hmm_result["states"][-1]]) if len(hmm_result["states"]) > 0 else 0.5,
                }
        except Exception:
            pass
    
    # Fallback to threshold-based regime detection (original logic)
    vol_regime = feats.get("vol_regime", pd.Series(dtype=float))
    trend_z = feats.get("trend_z", pd.Series(dtype=float))
    
    vr = safe_last(vol_regime) if not vol_regime.empty else float("nan")
    tz = safe_last(trend_z) if not trend_z.empty else float("nan")
    
    # Threshold-based classification
    if np.isfinite(vr) and vr > 1.8:
        if np.isfinite(tz) and tz > 0:
            label = "High-vol uptrend"
        elif np.isfinite(tz) and tz < 0:
            label = "High-vol downtrend"
        else:
            label = "crisis"  # Map to HMM-style label
    elif np.isfinite(vr) and vr < 0.85:
        if np.isfinite(tz) and tz > 0:
            label = "Calm uptrend"
        elif np.isfinite(tz) and tz < 0:
            label = "Calm downtrend"
        else:
            label = "calm"  # Map to HMM-style label
    elif np.isfinite(tz) and abs(tz) > 0.5:
        label = "trending"
    else:
        label = "Normal"
    
    return label, {
        "method": "threshold_fallback",
        "vol_regime": float(vr) if np.isfinite(vr) else None,
        "trend_z": float(tz) if np.isfinite(tz) else None,
    }


# -------------------------
# Backtest-safe feature view (no look-ahead)
# -------------------------

def shift_features(feats: Dict[str, pd.Series], lag: int = 1) -> Dict[str, pd.Series]:
    """Return a copy of features dict with time-series shifted by `lag` days to remove look-ahead.
    Use for backtesting so that features at date t only use information available up to t−lag.
    Only series keys are shifted; scalar/meta entries are passed through.
    """
    if feats is None:
        return {}
    lag = int(max(0, lag))
    if lag == 0:
        return dict(feats)
    keys_to_shift = {
        # drifts
        "mu", "mu_post", "mu_blend", "mu_kf", "mu_final",
        # vols and regimes
        "vol_fast", "vol_slow", "vol", "vol_regime",
        # trend/momentum/stretch
        "sma200", "trend_z", "z5", "mom21", "mom63", "mom126", "mom252",
        # tails
        "skew", "nu",
        # base series for reference
        "ret",
    }
    shifted: Dict[str, pd.Series] = {}
    for k, v in feats.items():
        if isinstance(v, pd.Series) and k in keys_to_shift:
            try:
                shifted[k] = v.shift(lag)
            except Exception:
                shifted[k] = v
        else:
            # pass-through (px and any non-shifted helper)
            shifted[k] = v
    return shifted


def make_features_views(feats: Dict[str, pd.Series]) -> Dict[str, Dict[str, pd.Series]]:
    """Convenience wrapper to expose both live and backtest-safe views.
    - live: unshifted (as-of) features suitable for real-time use
    - bt:   shifted by 1 day (no look-ahead) for backtesting
    """
    return {
        "live": feats,
        "bt": shift_features(feats, lag=1),
    }

# -------------------------
# Scoring
# -------------------------

def edge_for_horizon(mu: float, vol: float, H: int) -> float:
    # edge = mu_H / sigma_H in z units (analytic approximation)
    if not np.isfinite(mu) or not np.isfinite(vol) or vol <= 0:
        return 0.0
    mu_H = mu * H
    sig_H = vol * math.sqrt(H)
    return float(mu_H / sig_H)


def _simulate_forward_paths(feats: Dict[str, pd.Series], H_max: int, n_paths: int = 3000, phi: float = 0.95, kappa: float = 1e-4) -> np.ndarray:
    """Monte-Carlo forward simulation of cumulative log returns over 1..H_max.
    - Drift evolves as AR(1): mu_{t+1} = phi * mu_t + eta_t,  eta ~ N(0, q)
    - Volatility evolves via GARCH(1,1) when available; else held constant.
    - Innovations are Student-t with global df (nu_hat) scaled to unit variance.
    Level-7 parameter uncertainty: if PARAM_UNC environment variable is set to
    'sample' (default) and garch_params contains a covariance matrix, we sample
    (omega, alpha, beta) per path from N(theta_hat, Cov) with constraints, which
    widens confidence during regime shifts and narrows during stability.
    Returns an array of shape (H_max, n_paths) with cumulative log returns per horizon.
    """
    # Inputs at 'now'
    ret_idx = feats.get("ret", pd.Series(dtype=float)).index
    if ret_idx is None or len(ret_idx) == 0:
        return np.zeros((H_max, n_paths), dtype=float)
    mu_series = feats.get("mu_post")
    if not isinstance(mu_series, pd.Series) or mu_series.empty:
        mu_series = feats.get("mu")
    vol_series = feats.get("vol")
    if not isinstance(vol_series, pd.Series) or vol_series.empty or not isinstance(mu_series, pd.Series) or mu_series.empty:
        return np.zeros((H_max, n_paths), dtype=float)
    mu_now = float(mu_series.iloc[-1]) if len(mu_series) else 0.0
    vol_now = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
    vol_now = float(max(vol_now, 1e-6))

    # Tail parameter (global nu)
    nu_hat_series = feats.get("nu_hat")
    if isinstance(nu_hat_series, pd.Series) and not nu_hat_series.empty:
        nu = float(nu_hat_series.iloc[-1])
    else:
        nu = 50.0
    nu = float(np.clip(nu, 4.5, 500.0))
    # Scale to unit variance for Student-t innovations
    t_var = nu / (nu - 2.0) if nu > 2.0 else 1e6
    t_scale = math.sqrt(t_var)

    # GARCH params
    garch_params = feats.get("garch_params", {}) or {}
    use_garch = isinstance(garch_params, dict) and all(k in garch_params for k in ("omega", "alpha", "beta"))

    # Determine parameter uncertainty mode
    param_unc_mode = os.getenv("PARAM_UNC", "sample").strip().lower()
    if param_unc_mode not in ("none", "sample"):
        param_unc_mode = "sample"

    # Build per-path parameters (possibly sampled)
    if use_garch:
        base_theta = np.array([
            float(max(garch_params.get("omega", 0.0), 1e-12)),
            float(np.clip(garch_params.get("alpha", 0.0), 0.0, 0.999)),
            float(np.clip(garch_params.get("beta", 0.0), 0.0, 0.999)),
        ], dtype=float)
        cov = garch_params.get("cov")
        if isinstance(cov, list):
            try:
                cov = np.array(cov, dtype=float)
            except Exception:
                cov = None
        # Sample theta per path if enabled and covariance available
        if (param_unc_mode == "sample") and (cov is not None) and np.shape(cov) == (3, 3):
            rng = np.random.default_rng()
            try:
                thetas = rng.multivariate_normal(mean=base_theta, cov=cov, size=n_paths).astype(float)
            except Exception:
                # Fall back to eigen-decomposition sampling with small regularization
                try:
                    eigvals, eigvecs = np.linalg.eigh(0.5*(cov+cov.T) + 1e-12*np.eye(3))
                    eigvals = np.clip(eigvals, 0.0, None)
                    z = rng.normal(size=(n_paths, 3)) * np.sqrt(eigvals)
                    thetas = (z @ eigvecs.T) + base_theta
                except Exception:
                    thetas = np.tile(base_theta, (n_paths, 1))
            # Enforce constraints; replace invalid draws with base_theta
            omega_s = thetas[:, 0]
            alpha_s = thetas[:, 1]
            beta_s  = thetas[:, 2]
            # Fix obvious violations
            omega_s = np.maximum(omega_s, 1e-12)
            alpha_s = np.clip(alpha_s, 0.0, 0.999)
            beta_s  = np.clip(beta_s, 0.0, 0.999)
            # Enforce alpha+beta < 0.999 by shrinking both toward base proportionally
            ab = alpha_s + beta_s
            viol = ab >= 0.999
            if np.any(viol):
                # target sum slightly below 1
                target = 0.998
                scale = target / np.maximum(ab[viol], 1e-12)
                alpha_s[viol] *= scale
                beta_s[viol] *= scale
            omega_paths = omega_s
            alpha_paths = alpha_s
            beta_paths = beta_s
        else:
            omega_paths = np.full(n_paths, base_theta[0], dtype=float)
            alpha_paths = np.full(n_paths, base_theta[1], dtype=float)
            beta_paths  = np.full(n_paths, base_theta[2], dtype=float)
    else:
        omega_paths = np.zeros(n_paths, dtype=float)
        alpha_paths = np.zeros(n_paths, dtype=float)
        beta_paths  = np.zeros(n_paths, dtype=float)

    # Drift process noise variance q (relative to current variance)
    h0 = vol_now ** 2
    q = float(max(kappa * h0, 1e-10))

    # Initialize state arrays (vectorized across paths)
    cum = np.zeros((H_max, n_paths), dtype=float)
    mu_t = np.full(n_paths, mu_now, dtype=float)
    h_t = np.full(n_paths, max(h0, 1e-8), dtype=float)

    rng = np.random.default_rng()

    for t in range(H_max):
        # Student-t shocks standardized to unit variance
        z = rng.standard_t(df=nu, size=n_paths).astype(float)
        eps = z / t_scale
        sigma_t = np.sqrt(np.maximum(h_t, 1e-12))
        e_t = sigma_t * eps
        # accumulate log return: r_t = mu_t + e_t
        if t == 0:
            cum[t, :] = mu_t + e_t
        else:
            cum[t, :] = cum[t-1, :] + mu_t + e_t
        # Evolve volatility via GARCH or hold constant on fallback
        if use_garch:
            h_t = omega_paths + alpha_paths * (e_t ** 2) + beta_paths * h_t
            h_t = np.clip(h_t, 1e-12, 1e4)
        # Evolve drift via AR(1)
        eta = rng.normal(loc=0.0, scale=math.sqrt(q), size=n_paths)
        mu_t = phi * mu_t + eta

    return cum


def composite_edge(
    base_edge: float,
    trend_z: float,
    moms: List[float],
    vol_regime: float,
    z5: float,
) -> float:
    """Ensemble edge: blend trend-following and mean-reversion components.
    GARCH handles volatility dynamics; avoid extra regime dampening to prevent double-counting.
    """
    # Momentum confirmation: average tanh of t-momentum
    mom_terms = [np.tanh(m / 2.0) for m in moms if np.isfinite(m)]
    mom_align = float(np.mean(mom_terms)) if mom_terms else 0.0

    # Trend tilt (gentle)
    trend_tilt = float(np.tanh(trend_z / 2.0)) if np.isfinite(trend_z) else 0.0

    # TF component
    tf = base_edge + 0.30 * mom_align + 0.20 * trend_tilt

    # MR component: if z5 is very positive, expect mean-revert small negative edge; if very negative, mean-revert positive edge
    mr = float(-np.tanh(z5)) if np.isfinite(z5) else 0.0

    # Fixed blend (avoid vol_regime-driven dampening)
    w_tf, w_mr = 0.75, 0.25
    edge = w_tf * tf + w_mr * mr

    return float(edge)


def label_from_probability(p_up: float, pos_strength: float, buy_thr: float = 0.58, sell_thr: float = 0.42) -> str:
    """Map probability and position strength to label with customizable thresholds.
    - STRONG tiers require both probability and position_strength to be high.
    - buy_thr and sell_thr must satisfy sell_thr < 0.5 < buy_thr.
    """
    buy_thr = float(buy_thr)
    sell_thr = float(sell_thr)
    # Strong tiers (adjusted for half‑Kelly scaling: 0.30 instead of 0.60)
    if p_up >= max(0.66, buy_thr + 0.06) and pos_strength >= 0.30:
        return "STRONG BUY"
    if p_up <= min(0.34, sell_thr - 0.06) and pos_strength >= 0.30:
        return "STRONG SELL"
    # Base labels
    if p_up >= buy_thr:
        return "BUY"
    if p_up <= sell_thr:
        return "SELL"
    return "HOLD"


def latest_signals(feats: Dict[str, pd.Series], horizons: List[int], last_close: float, t_map: bool = True, ci: float = 0.68) -> Tuple[List[Signal], Dict[int, Dict[str, float]]]: 
    """Compute signals using regime‑aware priors, tail‑aware probability mapping, and
    anti‑snap logic (two‑day confirmation + hysteresis + smoothing) without extra flags.
    
    We build last‑two‑days estimates to avoid look‑ahead while adding stability.
    """
    idx = feats.get("px", pd.Series(dtype=float)).index
    if idx is None or len(idx) < 2:
        # Fallback to simple single‑day path
        idx = pd.DatetimeIndex(idx)
    last2 = idx[-2:] if len(idx) >= 2 else idx

    # Helper to safely fetch last/prev values from a Series
    def _tail2(series_key: str, default_val: float = np.nan) -> Tuple[float, float]:
        s = feats.get(series_key, None)
        if s is None or not isinstance(s, pd.Series) or s.empty:
            return (default_val, default_val)
        s2 = s.reindex(last2)
        vals = s2.to_numpy(dtype=float)
        if vals.size == 1:
            return (float(vals[-1]), float(vals[-1]))
        return (float(vals[-1]), float(vals[-2]))

    # Prefer posterior drift if available
    mu_now, mu_prev = _tail2("mu_post", 0.0)
    if not np.isfinite(mu_now):
        mu_now = 0.0
    if not np.isfinite(mu_prev):
        mu_prev = mu_now
    vol_now, vol_prev = _tail2("vol", np.nan)
    vol_reg_now, vol_reg_prev = _tail2("vol_regime", 1.0)
    trend_now, trend_prev = _tail2("trend_z", 0.0)
    z5_now, z5_prev = _tail2("z5", 0.0)
    # Use globally-fitted nu_hat (Level-7); fall back to rolling nu if missing
    nu_hat_series = feats.get("nu_hat")
    if isinstance(nu_hat_series, pd.Series) and not nu_hat_series.empty:
        nu_glob = float(nu_hat_series.iloc[-1])
    else:
        # fallback to last rolling nu
        nu_glob, _ = _tail2("nu", 50.0)
        if not np.isfinite(nu_glob):
            nu_glob = 50.0
    # Clip to safe range
    nu_glob = float(np.clip(nu_glob, 4.5, 500.0))
    # Prefer smoothed skew if available; fallback to raw skew; default neutral 0.0
    skew_now, skew_prev = _tail2("skew_s", np.nan)
    if not np.isfinite(skew_now) or not np.isfinite(skew_prev):
        skew_now_fallback, skew_prev_fallback = _tail2("skew", 0.0)
        if not np.isfinite(skew_now):
            skew_now = skew_now_fallback
        if not np.isfinite(skew_prev):
            skew_prev = skew_prev_fallback

    moms_now = [
        _tail2("mom21", 0.0)[0],
        _tail2("mom63", 0.0)[0],
        _tail2("mom126", 0.0)[0],
        _tail2("mom252", 0.0)[0],
    ]
    moms_prev = [
        _tail2("mom21", 0.0)[1],
        _tail2("mom63", 0.0)[1],
        _tail2("mom126", 0.0)[1],
        _tail2("mom252", 0.0)[1],
    ]

    # Mapping function that accepts per‑day skew/nu
    def map_prob(edge: float, nu_val: float, skew_val: float) -> float:
        if not np.isfinite(edge):
            return 0.5
        z = float(edge)
        # Base symmetric mapping (Student‑t preferred)
        if t_map and np.isfinite(nu_val) and 2.0 < nu_val < 1e9:
            try:
                base_p = float(student_t.cdf(z, df=float(nu_val)))
            except Exception:
                base_p = float(norm.cdf(z))
        else:
            base_p = float(norm.cdf(z))
        # Edgeworth asymmetry using realized skew and (optional) kurt proxy from nu
        g1 = float(np.clip(skew_val if np.isfinite(skew_val) else 0.0, -1.5, 1.5))
        if np.isfinite(nu_val) and nu_val > 4.5 and nu_val < 1e9:
            g2 = 6.0 / (float(nu_val) - 4.0)
        else:
            g2 = 0.0
        if g1 == 0.0 and g2 == 0.0:
            return float(np.clip(base_p, 0.001, 0.999))
        try:
            phi = float(norm.pdf(z))
            corr = (g1 / 6.0) * (1.0 - z * z) + (g2 / 24.0) * (z ** 3 - 3.0 * z) - (g1 ** 2 / 36.0) * (2.0 * z ** 3 - 5.0 * z)
            # Damp skew influence in extreme tails to stabilize mapping for |z|>~3
            try:
                damp = math.exp(-0.5 * z * z)
            except Exception:
                damp = 0.0
            p = base_p + phi * (corr * damp)
            return float(np.clip(p, 0.001, 0.999))
        except Exception:
            return float(np.clip(base_p, 0.001, 0.999))

    # Regime detection via HMM posterior inference (replaces threshold-based heuristics)
    hmm_result = feats.get("hmm_result")
    reg, regime_meta = infer_current_regime(feats, hmm_result)

    # CI quantile based on 'now'
    alpha = np.clip(ci, 1e-6, 0.999999)
    tail = 0.5 * (1 + alpha)
    if t_map and np.isfinite(nu_glob) and nu_glob > 2.0 and nu_glob < 1e9:
        try:
            z_star = float(student_t.ppf(tail, df=float(nu_glob)))
        except Exception:
            z_star = float(norm.ppf(tail))
    else:
        z_star = float(norm.ppf(tail))

    # Median vol for uncertainty component (use rolling median series if available)
    vol_series = feats.get("vol", pd.Series(dtype=float))
    try:
        med_vol_series = vol_series.rolling(252, min_periods=63).median()
        med_vol_last = safe_last(med_vol_series) if med_vol_series is not None else float('nan')
    except Exception:
        med_vol_last = float('nan')
    # Explicit, readable fallback to long-run belief anchor (global median over entire history)
    if not np.isfinite(med_vol_last) or med_vol_last <= 0:
        try:
            med_vol_last = float(np.nanmedian(np.asarray(vol_series.values, dtype=float)))
        except Exception:
            med_vol_last = float('nan')
    # Final guard: fall back to current vol (or 1.0) if global median is unavailable
    if not np.isfinite(med_vol_last) or med_vol_last <= 0:
        med_vol_last = vol_now if np.isfinite(vol_now) and vol_now > 0 else 1.0

    sigs: List[Signal] = []
    thresholds: Dict[int, Dict[str, float]] = {}

    # Regime-aware smoothing from HMM posterior uncertainty (replaces threshold-based)
    # Use regime persistence and uncertainty from posterior probabilities
    if regime_meta.get("method") == "hmm_posterior":
        # High persistence (diagonal transition prob) => low smoothing (trust signal)
        # Low persistence => high smoothing (reduce whipsaws)
        persistence = regime_meta.get("persistence", 0.5)
        alpha_edge = 0.30 + 0.25 * (1.0 - persistence)  # 0.30-0.55 range
        alpha_p = min(0.75, alpha_edge + 0.10)
    else:
        # Fallback for threshold-based regime
        alpha_edge = 0.40
        alpha_p = 0.50

    # Monte‑Carlo forward simulation to capture evolving drift/vol over horizons
    H_max = int(max(horizons) if horizons else 0)
    sims = _simulate_forward_paths(feats, H_max=H_max, n_paths=3000)

    for H in horizons:
        # Use simulation at horizon H (1‑indexed in description; here index H-1)
        if H <= 0 or H > sims.shape[0]:
            sim_H = np.zeros(3000, dtype=float)
        else:
            sim_H = sims[H-1, :]
        # Clean NaNs/Infs
        sim_H = np.asarray(sim_H, dtype=float)
        sim_H = sim_H[np.isfinite(sim_H)]
        if sim_H.size == 0:
            sim_H = np.zeros(3000, dtype=float)
        # Simulated moments and probability
        mH = float(np.mean(sim_H))
        vH = float(np.var(sim_H, ddof=1)) if sim_H.size > 1 else 0.0
        sH = float(math.sqrt(max(vH, 1e-12)))
        z_stat = float(mH / sH) if sH > 0 else 0.0
        p_now = float(np.mean(sim_H > 0.0))
        # For anti‑snap smoothing we need a previous probability; approximate with itself if unavailable
        p_prev = p_now
        p_s_prev = p_prev
        p_s_now = alpha_p * p_now + (1.0 - alpha_p) * p_prev

        # Base/composite edge built off z_stat to keep consistency
        base_now = z_stat
        base_prev = z_stat  # lacking prev sim, reuse now as stable default
        edge_prev = composite_edge(base_prev, trend_prev, moms_prev, vol_reg_prev, z5_prev)
        edge_now = composite_edge(base_now, trend_now, moms_now, vol_reg_now, z5_now)

        # Expected log return and CI from simulation (percentile CI)
        q = float(np.clip(ci, 1e-6, 0.999999))
        lo_q = (1.0 - q) / 2.0
        hi_q = 1.0 - lo_q
        try:
            ci_low = float(np.quantile(sim_H, lo_q))
            ci_high = float(np.quantile(sim_H, hi_q))
        except Exception:
            ci_low = mH - 1.0 * sH
            ci_high = mH + 1.0 * sH
        mu_H = mH  # use simulated mean directly
        sig_H = sH

        # Fractional Kelly sizing (half‑Kelly) using simulated mean/variance
        denom = vH if vH > 0 else (sig_H ** 2 if sig_H > 0 else 1.0)
        f_star = float(np.clip(mu_H / denom, -1.0, 1.0))
        pos_strength = float(min(1.0, 0.5 * abs(f_star)))

        # Dynamic thresholds (asymmetry via skew + uncertainty widening), based on 'now'
        g1 = float(np.clip(skew_now if np.isfinite(skew_now) else 0.0, -1.5, 1.5))
        base_buy, base_sell = 0.58, 0.42
        skew_delta = 0.02 * float(np.tanh(abs(g1) / 0.75))
        if g1 < 0:
            buy_thr = base_buy + skew_delta
            sell_thr = base_sell + skew_delta
        elif g1 > 0:
            buy_thr = base_buy - skew_delta
            sell_thr = base_sell - skew_delta
        else:
            buy_thr, sell_thr = base_buy, base_sell
        # Regime-based uncertainty (replaces threshold-based components)
        # Use HMM posterior entropy as uncertainty measure
        if regime_meta.get("method") == "hmm_posterior":
            probs = regime_meta.get("probabilities", {})
            # Shannon entropy of regime posteriors: high entropy = high uncertainty
            entropy = 0.0
            for p in probs.values():
                if p > 1e-12:
                    entropy -= p * np.log(p)
            # Normalize by max entropy (log(3) for 3 states)
            u_regime = float(np.clip(entropy / np.log(3.0), 0.0, 1.0))
        else:
            # Fallback: use vol_regime deviation if available
            u_regime = 0.5
            if np.isfinite(vol_reg_now):
                u_regime = float(np.clip(abs(vol_reg_now - 1.0) / 1.5, 0.0, 1.0))
        
        # Tail uncertainty from global Student-t fit
        if np.isfinite(nu_glob):
            nu_eff = float(np.clip(nu_glob, 3.0, 200.0))
            inv_nu = 1.0 / nu_eff
            inv_min, inv_max = 1.0 / 200.0, 1.0 / 3.0
            u_tail = float(np.clip((inv_nu - inv_min) / (inv_max - inv_min), 0.0, 1.0))
        else:
            u_tail = 0.0
        
        # Forecast uncertainty from simulated variance vs historical
        med_sig_H = (med_vol_last * math.sqrt(H)) if (np.isfinite(med_vol_last) and med_vol_last > 0) else sig_H
        ratio = float(sig_H / med_sig_H) if med_sig_H > 0 else 1.0
        u_sig = float(np.clip(ratio - 1.0, 0.0, 1.0))
        
        # Combined uncertainty: regime entropy dominates, tail and forecast refine
        U = float(np.clip(0.5 * u_regime + 0.3 * u_tail + 0.2 * u_sig, 0.0, 1.0))
        widen_delta = 0.04 * U
        buy_thr += widen_delta
        sell_thr -= widen_delta
        buy_thr = float(np.clip(buy_thr, 0.55, 0.70))
        sell_thr = float(np.clip(sell_thr, 0.30, 0.45))
        if buy_thr - sell_thr < 0.12:
            mid = 0.5
            sell_thr = min(sell_thr, mid - 0.06)
            buy_thr = max(buy_thr, mid + 0.06)

        thresholds[int(H)] = {"buy_thr": float(buy_thr), "sell_thr": float(sell_thr), "uncertainty": float(U), "edge_floor": float(EDGE_FLOOR)}

        # Hysteresis bands and 2‑day confirmation on smoothed probabilities
        buy_enter = buy_thr + 0.01
        sell_enter = sell_thr - 0.01
        label = "HOLD"
        if (p_s_prev >= buy_enter) and (p_s_now >= buy_enter):
            label = "BUY"
        elif (p_s_prev <= sell_enter) and (p_s_now <= sell_enter):
            label = "SELL"
        # Strong tiers based on current (unsmoothed) conviction and Kelly strength
        if p_now >= max(0.66, buy_thr + 0.06) and pos_strength >= 0.30:
            label = "STRONG BUY"
        if p_now <= min(0.34, sell_thr - 0.06) and pos_strength >= 0.30:
            label = "STRONG SELL"

        # Transaction-cost/slippage hurdle: if absolute edge is below EDGE_FLOOR, force HOLD
        edge_floor_applied = False
        try:
            if np.isfinite(edge_now) and abs(edge_now) < float(EDGE_FLOOR):
                label = "HOLD"
                edge_floor_applied = True
        except Exception:
            pass

        # CI bounds for expected log return
        ci_low = float(mu_H - z_star * sig_H)
        ci_high = float(mu_H + z_star * sig_H)

        # Convert expected log‑return to PLN profit for a 1,000,000 PLN notional
        exp_mult = float(np.exp(mu_H))
        ci_low_mult = float(np.exp(ci_low))
        ci_high_mult = float(np.exp(ci_high))
        profit_pln = float(NOTIONAL_PLN) * (exp_mult - 1.0)
        profit_ci_low_pln = float(NOTIONAL_PLN) * (ci_low_mult - 1.0)
        profit_ci_high_pln = float(NOTIONAL_PLN) * (ci_high_mult - 1.0)

        sigs.append(Signal(
            horizon_days=int(H),
            score=float(edge_now),
            p_up=float(p_now),
            exp_ret=float(mu_H),
            ci_low=ci_low,
            ci_high=ci_high,
            profit_pln=float(profit_pln),
            profit_ci_low_pln=float(profit_ci_low_pln),
            profit_ci_high_pln=float(profit_ci_high_pln),
            position_strength=float(pos_strength),
            regime=reg,
            label=label,
        ))

    return sigs, thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate signals across multiple horizons for PLN/JPY, Gold (PLN), Silver (PLN), Bitcoin (PLN), and MicroStrategy (PLN).")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--horizons", type=str, default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--assets", type=str, default="PLNJPY=X,GC=F,SI=F,BTC-USD,MSTR,NFLX,NVO,KTOS,RKLB,GOOO,GLDW,SGLP,GLDE,FACC,SLVI,MTX,IBKR,HOOD,RHEINMETALL,AMD,APPLE,AIRBUS,RENK,NORTHROP GRUMMAN,NVIDIA,TKMS AG & CO,MICROSOFT,UBER,VANGUARD S&P 500,THALES,TESLA,SAMSUNG", help="Comma-separated Yahoo symbols or friendly names. Metals, FX and USD/EUR/GBP/JPY/CAD/DKK/KRW assets are converted to PLN.")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--simple", action="store_true", help="Print an easy-to-read summary with simple explanations.")
    p.add_argument("--t_map", action="store_true", help="Use Student-t mapping based on realized kurtosis for probabilities (default on).")
    p.add_argument("--no_t_map", dest="t_map", action="store_false", help="Disable Student-t mapping; use Normal CDF.")
    p.add_argument("--ci", type=float, default=0.68, help="Two-sided confidence level for expected move bands (default 0.68 i.e., ~1-sigma).")
    # Caption controls for detailed view
    p.add_argument("--no_caption", action="store_true", help="Suppress the long column explanation caption in detailed tables.")
    p.add_argument("--force_caption", action="store_true", help="Force showing the caption for every detailed table.")
    # Diagnostics controls (Level-7 falsifiability)
    p.add_argument("--diagnostics", action="store_true", help="Enable full diagnostics: log-likelihood monitoring, parameter stability tracking, and out-of-sample tests (expensive).")
    p.add_argument("--diagnostics_lite", action="store_true", help="Enable lightweight diagnostics: log-likelihood monitoring and parameter stability (no OOS tests).")
    p.set_defaults(t_map=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})

    # Parse assets
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    all_blocks = []  # for JSON export
    csv_rows_simple = []  # for CSV simple export
    csv_rows_detailed = []  # for CSV detailed export
    summary_rows = []  # for summary table across assets

    caption_printed = False
    processed_syms = set()
    for asset in assets:
        try:
            px, title = fetch_px_asset(asset, args.start, args.end)
        except Exception as e:
            # Skip asset with a warning row (console)
            Console().print(f"[red]Warning:[/red] Failed to fetch {asset}: {e}")
            continue

        # De-duplicate by resolved symbol extracted from title (e.g., "Company (SYMBOL) — ...").
        # If not found, fall back to the asset token to avoid duplicates from identical inputs.
        canon = extract_symbol_from_title(title)
        if not canon:
            canon = asset.strip().upper()
        if canon in processed_syms:
            Console().print(f"[yellow]Skipping duplicate:[/yellow] {title} (from input '{asset}')")
            continue
        processed_syms.add(canon)

        feats = compute_features(px)
        last_close = _to_float(px.iloc[-1])
        sigs, thresholds = latest_signals(feats, horizons, last_close=last_close, t_map=args.t_map, ci=args.ci)
        
        # Compute diagnostics if requested (Level-7 falsifiability)
        diagnostics = {}
        if args.diagnostics or args.diagnostics_lite:
            enable_oos = args.diagnostics  # Full diagnostics include expensive OOS tests
            diagnostics = compute_all_diagnostics(px, feats, enable_oos=enable_oos)

        # Print table for this asset
        if args.simple:
            explanations = render_simplified_signal_table(asset, title, sigs, px, feats)
        else:
            # Determine caption policy for detailed view
            if args.force_caption:
                show_caption = True
            elif args.no_caption:
                show_caption = False
            else:
                show_caption = not caption_printed
            render_detailed_signal_table(asset, title, sigs, px, confidence_level=args.ci, used_student_t_mapping=args.t_map, show_caption=show_caption)
            caption_printed = caption_printed or show_caption
            explanations = []
        
        # Display diagnostics if computed (Level-7: model falsifiability)
        if diagnostics and (args.diagnostics or args.diagnostics_lite):
            from rich.table import Table
            console = Console()
            diag_table = Table(title=f"📊 Diagnostics for {asset} — Model Falsifiability Metrics")
            diag_table.add_column("Metric", justify="left", style="cyan")
            diag_table.add_column("Value", justify="right")
            
            # Log-likelihood monitoring
            if "garch_log_likelihood" in diagnostics:
                diag_table.add_row("GARCH(1,1) Log-Likelihood", f"{diagnostics['garch_log_likelihood']:.2f}")
                diag_table.add_row("GARCH(1,1) AIC", f"{diagnostics['garch_aic']:.2f}")
                diag_table.add_row("GARCH(1,1) BIC", f"{diagnostics['garch_bic']:.2f}")
            
            if "hmm_log_likelihood" in diagnostics:
                diag_table.add_row("HMM Regime Log-Likelihood", f"{diagnostics['hmm_log_likelihood']:.2f}")
                diag_table.add_row("HMM AIC", f"{diagnostics['hmm_aic']:.2f}")
                diag_table.add_row("HMM BIC", f"{diagnostics['hmm_bic']:.2f}")
            
            if "student_t_log_likelihood" in diagnostics:
                diag_table.add_row("Student-t Tail Log-Likelihood", f"{diagnostics['student_t_log_likelihood']:.2f}")
                diag_table.add_row("Student-t Degrees of Freedom (ν)", f"{diagnostics['student_t_nu']:.2f}")
            
            # Parameter stability (recent drift z-scores)
            drift_cols = [k for k in diagnostics.keys() if k.startswith("recent_") and k.endswith("_drift_z")]
            if drift_cols:
                diag_table.add_row("", "")  # spacer
                diag_table.add_row("[bold]Parameter Stability[/bold]", "[bold]Recent Drift (z-score)[/bold]")
                for col in drift_cols:
                    param_name = col.replace("recent_", "").replace("_drift_z", "")
                    val = diagnostics[col]
                    if np.isfinite(val):
                        color = "green" if abs(val) < 2.0 else ("yellow" if abs(val) < 3.0 else "red")
                        diag_table.add_row(f"  {param_name}", f"[{color}]{val:+.2f}[/{color}]")
            
            # Out-of-sample test results (if enabled)
            oos_keys = [k for k in diagnostics.keys() if k.startswith("oos_") and k.endswith("_hit_rate")]
            if oos_keys:
                diag_table.add_row("", "")  # spacer
                diag_table.add_row("[bold]Out-of-Sample Tests[/bold]", "[bold]Direction Hit Rate[/bold]")
                for key in oos_keys:
                    horizon_label = key.replace("oos_", "").replace("_hit_rate", "")
                    hit_rate = diagnostics[key]
                    if np.isfinite(hit_rate):
                        color = "green" if hit_rate >= 0.55 else ("yellow" if hit_rate >= 0.50 else "red")
                        diag_table.add_row(f"  {horizon_label}", f"[{color}]{hit_rate*100:.1f}%[/{color}]")
            
            console.print(diag_table)
            console.print("")  # blank line

        # Build summary row for this asset
        asset_label = build_asset_display_label(asset, title)
        horizon_signals = {
            int(s.horizon_days): {"label": s.label, "profit_pln": float(s.profit_pln)}
            for s in sigs
        }
        # Find nearest horizon label for sorting
        nearest_label = sigs[0].label if sigs else "HOLD"
        summary_rows.append({
            "asset_label": asset_label,
            "horizon_signals": horizon_signals,
            "nearest_label": nearest_label,
        })

        # Prepare JSON block
        block = {
            "symbol": asset,
            "title": title,
            "as_of": str(px.index[-1].date()),
            "last_close": last_close,
            "notional_pln": NOTIONAL_PLN,
            "signals": [s.__dict__ for s in sigs],
            "ci_level": args.ci,
            "ci_domain": "log_return",
            "profit_ci_domain": "arithmetic_pln",
            "probability_mapping": ("student_t" if args.t_map else "normal"),
            "nu_clip": {"min": 4.5, "max": 500.0},
            "edgeworth_damped": True,
            "kelly_rule": "half",
            "decision_thresholds": thresholds,
            # volatility modeling metadata
            "vol_source": feats.get("vol_source", "garch11"),
            "garch_params": feats.get("garch_params", {}),
            # tail modeling metadata (global ν)
            "tail_model": "student_t_global",
            "nu_hat": float(feats.get("nu_hat").iloc[-1]) if isinstance(feats.get("nu_hat"), pd.Series) and not feats.get("nu_hat").empty else 50.0,
            "nu_bounds": {"min": 4.5, "max": 500.0},
            "nu_info": feats.get("nu_info", {}),
        }
        
        # Add diagnostics to JSON if computed (Level-7 falsifiability)
        if diagnostics:
            # Filter out non-serializable objects (DataFrames) for JSON
            serializable_diagnostics = {}
            for k, v in diagnostics.items():
                if k in ("parameter_stability", "out_of_sample"):
                    # Skip raw DataFrames; summary metrics already in top-level diagnostics
                    continue
                if isinstance(v, (int, float, str, bool, type(None))):
                    serializable_diagnostics[k] = v
                elif isinstance(v, dict):
                    # Nested dicts are OK if they contain serializable values
                    serializable_diagnostics[k] = v
            block["diagnostics"] = serializable_diagnostics
        
        all_blocks.append(block)

        # Prepare CSV rows
        if args.simple:
            for i, s in enumerate(sigs):
                csv_rows_simple.append({
                    "asset": title,
                    "symbol": asset,
                    "timeframe": format_horizon_label(s.horizon_days),
                    "chance_up_pct": f"{s.p_up*100:.1f}",
                    "recommendation": s.label,
                    "why": explanations[i],
                })
        else:
            for s in sigs:
                row = s.__dict__.copy()
                row.update({
                    "asset": title,
                    "symbol": asset,
                })
                csv_rows_detailed.append(row)

    # After processing all assets, print a compact summary
    try:
        render_multi_asset_summary_table(summary_rows, horizons)
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not print summary table: {e}")

    # Exports
    if args.json:
        payload = {
            "assets": all_blocks,
            "column_descriptions": DETAILED_COLUMN_DESCRIPTIONS,
            "simple_column_descriptions": SIMPLIFIED_COLUMN_DESCRIPTIONS,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)

    if args.csv:
        if args.simple:
            df = pd.DataFrame(csv_rows_simple)
            df.to_csv(args.csv, index=False)
        else:
            if csv_rows_detailed:
                df = pd.DataFrame(csv_rows_detailed)
                rename_map = {
                    "horizon_days": "horizon_trading_days",
                    "score": "edge_z_risk_adjusted",
                    "p_up": "prob_up",
                    "exp_ret": "expected_log_return",
                    "ci_low": "ci_low_log",
                    "ci_high": "ci_high_log",
                    "profit_pln": "profit_pln_on_1m_pln",
                    "profit_ci_low_pln": "profit_ci_low_pln",
                    "profit_ci_high_pln": "profit_ci_high_pln",
                    "label": "signal",
                }
                df = df.rename(columns=rename_map)
                # ensure asset/symbol columns at the front
                cols = ["asset", "symbol"] + [c for c in df.columns if c not in ("asset", "symbol")]
                df = df[cols]
                df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
