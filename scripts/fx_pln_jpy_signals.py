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
from multiprocessing import Pool, cpu_count

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
    render_sector_summary_tables,
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
    DEFAULT_ASSET_UNIVERSE,
    get_default_asset_universe,
)

# Suppress noisy yfinance download warnings (e.g., "1 Failed download: ...")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

PAIR = "PLNJPY=X"
DEFAULT_HORIZONS = [1, 3, 7, 21, 63, 126, 252]
NOTIONAL_PLN = 1_000_000  # for profit column

# Sector mapping for summary grouping
SECTOR_MAP = {
    "FX / Commodities / Crypto": {
        "PLNJPY=X", "GC=F", "SI=F", "BTC-USD", "BTCUSD=X", "MSTR"
    },
    "Indices / Broad ETFs": {
        "SPY", "VOO", "GLD", "SLV", "SMH"
    },
    "Information Technology": {
        "AAPL", "ACN", "ADBE", "AMD", "AVGO", "CRM", "CSCO", "IBM", "INTC", "INTU", "MSFT", "NOW", "NVDA", "ORCL", "PLTR", "QCOM", "TXN", "GOOG", "GOOGL", "META", "NFLX", "AMZN"
    },
    "Health Care": {
        "ABBV", "ABT", "AMGN", "BMY", "CVS", "DHR", "GILD", "ISRG", "JNJ", "LLY", "MDT", "MRK", "NVO", "PFE", "TMO", "UNH"
    },
    "Financials": {
        "AIG", "AXP", "BAC", "BK", "BLK", "BRK.B", "C", "COF", "GS", "IBKR", "JPM", "MA", "MET", "MS", "PYPL", "SCHW", "USB", "V", "WFC", "HOOD"
    },
    "Consumer Discretionary": {
        "BKNG", "GM", "HD", "LOW", "MCD", "NKE", "SBUX", "TGT", "TSLA"
    },
    "Industrials": {
        "CAT", "DE", "EMR", "FDX", "MMM", "UBER", "UNP", "UPS"
    },
    "Defense & Aerospace": {
        "ACHR", "AIR", "AIRI", "AIRO", "AOUT", "ASTC", "ATI", "ATRO", "AVAV", "AXON", "AZ", "BA", "BAH", "BETA", "BWXT", "BYRN", "CACI", "CAE", "CDRE", "CODA", "CVU", "CW", "DCO", "DFSC", "DPRO", "DRS", "EH", "EMBJ", "ESLT", "EVEX", "EVTL", "FJET", "FLY", "FTAI", "GD", "GE", "GPUS", "HEI", "HEIA", "HII", "HOVR", "HWM", "HXL", "HON", "ISSC", "JOBY", "KITT", "KRMN", "KTOS", "LDOS", "LHX", "LMT", "LOAR", "LUNR", "MANT", "MNTS", "MOG.A", "MRCY", "MSA", "NOC", "NPK", "OPXS", "OSK", "PEW", "PKE", "PL", "POWW", "PRZO", "RCAT", "RDW", "RGR", "RKLB", "RTX", "SAIC", "SARO", "SATL", "SIDU", "SIF", "SKYH", "SPAI", "SPCE", "SPR", "SWBI", "TATT", "TDG", "TDY", "TXT", "VSAT", "VSEC", "VTSI", "VVX", "VWAV", "VOYG", "WWD", "RHM.DE", "AIR.PA", "HO.PA", "HAG.DE", "BA.L", "FACC.VI", "MTX.DE"
    },
    "Communication Services": {"CMCSA", "DIS", "T", "TMUS", "VZ"},
    "Consumer Staples": {"CL", "COST", "KO", "MDLZ", "MO", "PEP", "PG", "PM", "WMT"},
    "Energy": {"COP", "CVX", "XOM"},
    "Utilities": {"NEE", "SO"},
    "Real Estate": {"SPG"},
    "Materials": {"LIN", "NEM"},
    "Asian Tech & Manufacturing": {"005930.KS"},
    "VanEck ETFs": {"AFK", "ANGL", "CNXT", "EGPT", "FLTR", "GLIN", "MOTG", "IDX", "MLN", "NLR"},
}

# Transaction-cost/slippage hurdle: minimum absolute edge required to act
# Can be overridden via environment variable EDGE_FLOOR (e.g., 0.10)
try:
    _edge_env = os.getenv("EDGE_FLOOR", "0.10")
    EDGE_FLOOR = float(_edge_env)
except Exception:
    EDGE_FLOOR = 0.10
# Clamp to a reasonable range to avoid misuse
EDGE_FLOOR = float(np.clip(EDGE_FLOOR, 0.0, 1.5))

DEFAULT_CACHE_PATH = os.path.join("cache", "fx_plnjpy.json")

def get_sector(symbol: str) -> str:
    s = symbol.upper().strip()
    for sector, tickers in SECTOR_MAP.items():
        if s in tickers:
            return sector
    return "Unspecified"


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
    vol_mean: float       # mean volatility forecast (stochastic vol posterior)
    vol_ci_low: float     # lower bound of volatility CI
    vol_ci_high: float    # upper bound of volatility CI
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

    Level-7 Bayesian GARCH: 
    - We approximate parameter uncertainty by computing the observed information 
      (numeric Hessian of the negative log-likelihood) at the MLE optimum and 
      inverting it to obtain an approximate covariance matrix for (omega,alpha,beta).
    - In forward simulation (_simulate_forward_paths), parameters are sampled from 
      N(theta_hat, Cov) per path, propagating GARCH uncertainty into forecasts.
    - This Gaussian approximation is institution-grade and sufficient for Level-7.
    
    Future Level-8+ pathway (research frontier, not required):
    - Full Bayesian GARCH via HMC/NUTS posterior sampling (e.g., PyMC, Stan)
    - Priors enforcing stationarity: α + β < 1
    - Joint posterior over (ω, α, β) with proper uncertainty quantification
    - This would eliminate Gaussian approximation but requires MCMC infrastructure.
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
    - Returns a dict: {"nu_hat": float, "ll": float, "n": int, "converged": bool, "se_nu": float}.
    - On failure or insufficient data, returns a conservative default with converged=False.
    
    Tier 2 Enhancement: Posterior parameter variance tracking
    Computes standard error for ν via numeric Hessian (observed information matrix).
    This enables:
        ✔ Automatic conservatism during ν uncertainty
        ✔ ν sampling in Monte Carlo simulation
        ✔ Wider forecast intervals when tail parameter is uncertain
    """
    from scipy.optimize import minimize

    if z is None or not isinstance(z, pd.Series) or z.empty:
        return {"nu_hat": 50.0, "ll": float("nan"), "n": 0, "converged": False, "se_nu": float("nan")}

    zz = pd.to_numeric(z, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    # Remove zeros that may indicate degenerate scaling (not necessary but harmless)
    zz = zz[np.isfinite(zz.values)]
    n = int(zz.shape[0])
    if n < max(50, min_n):
        # too short: near-normal default
        return {"nu_hat": 50.0, "ll": float("nan"), "n": n, "converged": False, "se_nu": float("nan")}

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
        return {"nu_hat": 50.0, "ll": float("nan"), "n": n, "converged": False, "se_nu": float("nan")}

    nu_hat = float(np.clip(best[0].x[0], bounds[0], bounds[1]))
    ll = float(-best[1])
    
    # Compute standard error via numeric Hessian (observed information)
    # Hessian approximation: second derivative of negative log-likelihood
    se_nu = None
    try:
        # Finite difference approximation: d²NLL/dν²
        eps = max(0.01 * abs(nu_hat), 0.1)  # adaptive step size
        
        # Central difference for second derivative
        nll_0 = nll(nu_hat)
        nll_plus = nll(nu_hat + eps)
        nll_minus = nll(nu_hat - eps)
        
        # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h²
        d2_nll = (nll_plus - 2.0 * nll_0 + nll_minus) / (eps ** 2)
        
        # Standard error: sqrt(1 / observed_information)
        # observed_information = d²(-LL)/dν² = d²NLL/dν²
        if d2_nll > 1e-12:  # positive curvature (proper minimum)
            se_nu = float(np.sqrt(1.0 / d2_nll))
            # Sanity check: SE should be reasonable relative to estimate
            if se_nu > 10.0 * nu_hat or not np.isfinite(se_nu):
                se_nu = None
        else:
            se_nu = None
    except Exception:
        se_nu = None
    
    result = {
        "nu_hat": nu_hat,
        "ll": ll,
        "n": n,
        "converged": True,
        "se_nu": float(se_nu) if se_nu is not None else float("nan"),
    }
    
    return result


def _test_innovation_whiteness(innovations: np.ndarray, innovation_vars: np.ndarray, lags: int = 20) -> Dict[str, float]:
    """
    Test innovation whiteness using Ljung-Box test for autocorrelation.
    
    Refinement 3: Model adequacy via innovation whiteness testing.
    If innovations are not white noise (autocorrelated), the model may be misspecified.
    
    Args:
        innovations: Prediction errors from Kalman filter
        innovation_vars: Innovation variances (for standardization)
        lags: Number of lags to test
        
    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    try:
        # Standardize innovations by their predicted variance
        std_innovations = innovations / np.sqrt(np.maximum(innovation_vars, 1e-12))
        std_innovations = std_innovations[np.isfinite(std_innovations)]
        
        if len(std_innovations) < max(30, lags + 10):
            return {
                "ljung_box_statistic": float("nan"),
                "ljung_box_pvalue": float("nan"),
                "lags_tested": 0,
                "model_adequate": None,
                "note": "insufficient_data"
            }
        
        n = len(std_innovations)
        lags = min(lags, n // 5)  # conservative lag limit
        
        # Compute Ljung-Box statistic manually
        # Q = n(n+2) Σ(ρ_k² / (n-k)) for k=1..m
        # Under H0 (white noise), Q ~ χ²(m)
        
        # Compute autocorrelations
        acf_vals = []
        for lag in range(1, lags + 1):
            if lag >= n:
                break
            try:
                # Sample autocorrelation at lag k
                mean_innov = float(np.mean(std_innovations))
                numerator = float(np.sum((std_innovations[lag:] - mean_innov) * (std_innovations[:-lag] - mean_innov)))
                denominator = float(np.sum((std_innovations - mean_innov) ** 2))
                rho_k = numerator / denominator if abs(denominator) > 1e-12 else 0.0
                acf_vals.append(rho_k)
            except Exception:
                break
        
        if not acf_vals:
            return {
                "ljung_box_statistic": float("nan"),
                "ljung_box_pvalue": float("nan"),
                "lags_tested": 0,
                "model_adequate": None,
                "note": "acf_computation_failed"
            }
        
        # Ljung-Box statistic
        Q = 0.0
        m = len(acf_vals)
        for k, rho_k in enumerate(acf_vals, start=1):
            Q += (rho_k ** 2) / float(n - k)
        Q *= n * (n + 2)
        
        # Compute p-value using chi-squared distribution
        from scipy.stats import chi2
        pvalue = float(1.0 - chi2.cdf(Q, df=m))
        
        # Interpretation: reject H0 (white noise) if p < 0.05
        # model_adequate = True if we fail to reject (p >= 0.05)
        model_adequate = bool(pvalue >= 0.05)
        
        return {
            "ljung_box_statistic": float(Q),
            "ljung_box_pvalue": float(pvalue),
            "lags_tested": int(m),
            "model_adequate": model_adequate,
            "note": "pass" if model_adequate else "fail_autocorrelation_detected"
        }
        
    except Exception as e:
        return {
            "ljung_box_statistic": float("nan"),
            "ljung_box_pvalue": float("nan"),
            "lags_tested": 0,
            "model_adequate": None,
            "note": f"test_failed: {str(e)}"
        }


def _compute_kalman_log_likelihood(y: np.ndarray, sigma: np.ndarray, q: float) -> float:
    """
    Compute log-likelihood for Kalman filter with given process noise q.
    Used for q optimization via marginal likelihood maximization.
    
    Args:
        y: Observations (returns)
        sigma: Observation noise std (volatility) per time step
        q: Process noise variance to evaluate
        
    Returns:
        Total log-likelihood of observations under this q
    """
    T = len(y)
    if T < 2:
        return float('-inf')
    
    # Initialize
    mu_t = 0.0
    P_t = 1.0
    log_likelihood = 0.0
    
    for t in range(T):
        # Prediction
        mu_pred = mu_t
        P_pred = P_t + q
        
        # Observation variance
        R_t = float(max(sigma[t] ** 2, 1e-12))
        
        # Innovation
        innov = y[t] - mu_pred
        S_t = float(max(P_pred + R_t, 1e-12))
        
        # Log-likelihood contribution
        try:
            ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
            if np.isfinite(ll_t):
                log_likelihood += ll_t
        except Exception:
            pass
        
        # Update
        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * innov
        P_t = float(max((1.0 - K_t) * P_pred, 1e-12))
    
    return float(log_likelihood)


def _compute_kalman_log_likelihood_heteroskedastic(y: np.ndarray, sigma: np.ndarray, c: float) -> float:
    """
    Compute log-likelihood for Kalman filter with heteroskedastic process noise q_t = c * σ_t².
    
    This allows drift uncertainty to scale with market stress: higher volatility => more drift uncertainty.
    
    Args:
        y: Observations (returns)
        sigma: Observation noise std (volatility) per time step
        c: Scaling factor for heteroskedastic process noise (q_t = c * σ_t²)
        
    Returns:
        Total log-likelihood of observations under this c
    """
    T = len(y)
    if T < 2:
        return float('-inf')
    
    # Initialize
    mu_t = 0.0
    P_t = 1.0
    log_likelihood = 0.0
    
    for t in range(T):
        # Heteroskedastic process noise: q_t = c * σ_t²
        R_t = float(max(sigma[t] ** 2, 1e-12))
        q_t = float(max(c * R_t, 1e-12))
        
        # Prediction
        mu_pred = mu_t
        P_pred = P_t + q_t
        
        # Innovation
        innov = y[t] - mu_pred
        S_t = float(max(P_pred + R_t, 1e-12))
        
        # Log-likelihood contribution
        try:
            ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
            if np.isfinite(ll_t):
                log_likelihood += ll_t
        except Exception:
            pass
        
        # Update
        K_t = P_pred / S_t
        mu_t = mu_pred + K_t * innov
        P_t = float(max((1.0 - K_t) * P_pred, 1e-12))
    
    return float(log_likelihood)


def _estimate_regime_drift_priors(ret: pd.Series, vol: pd.Series) -> Optional[Dict[str, float]]:
    """
    Estimate regime-specific drift expectations E[μ_t | Regime=k] from historical data.
    
    Uses a quick HMM fit on returns to identify regimes, then computes mean return
    per regime as a simple proxy for regime-conditional drift.
    
    Args:
        ret: Returns series
        vol: Volatility series
        
    Returns:
        Dictionary with regime-specific drift priors, or None if estimation fails
    """
    if not HMM_AVAILABLE:
        return None
    
    try:
        # Align data
        df = pd.concat([ret, vol], axis=1, join='inner').dropna()
        if len(df) < 300:
            return None
        
        df.columns = ["ret", "vol"]
        X = df.values
        
        # Fit 3-state HMM
        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=42, verbose=False)
        model.fit(X)
        
        # Predict states
        states = model.predict(X)
        
        # Compute mean return per state
        regime_drifts = {}
        for state_idx in range(3):
            mask = (states == state_idx)
            if np.sum(mask) > 10:
                regime_drifts[state_idx] = float(np.mean(df.loc[mask, "ret"]))
            else:
                regime_drifts[state_idx] = 0.0
        
        # Identify regime names by volatility
        means = model.means_
        vol_means = means[:, 1]
        sorted_indices = np.argsort(vol_means)
        
        regime_map = {
            sorted_indices[0]: "calm",
            sorted_indices[1]: "trending",
            sorted_indices[2]: "crisis"
        }
        
        # Get current regime (last observation)
        current_state = states[-1]
        current_regime = regime_map.get(current_state, "calm")
        current_drift_prior = regime_drifts.get(current_state, 0.0)
        
        return {
            "current_regime": current_regime,
            "current_drift_prior": float(current_drift_prior),
            "regime_drifts": regime_drifts,
            "regime_map": regime_map,
        }
        
    except Exception:
        return None


def _load_tuned_kalman_params(asset_symbol: str, cache_path: str = "cache/kalman_q_cache.json") -> Optional[Dict]:
    """
    Load pre-tuned Kalman parameters from cache generated by tune_q_mle.py.
    
    Supports both Gaussian (q, c) and Student-t (q, c, ν) models with backwards compatibility.
    
    Args:
        asset_symbol: Asset symbol (e.g., "PLNJPY=X", "SPY")
        cache_path: Path to the tuned parameters cache file
        
    Returns:
        Dictionary with parameters if found, None otherwise. Contains:
        - 'q': Process noise variance
        - 'c': Observation variance scale
        - 'noise_model': "gaussian" or "student_t"
        - 'nu': Student-t degrees of freedom (if Student-t model)
        - Plus diagnostic metadata
    """
    try:
        if not os.path.exists(cache_path):
            return None
        
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        
        # Direct lookup
        if asset_symbol in cache:
            data = cache[asset_symbol]
            q_val = data.get('q')
            c_val = data.get('c', 1.0)
            
            # Validate basic parameters
            if q_val is not None and q_val > 0 and c_val > 0:
                # Load noise model (default to Gaussian for backwards compatibility)
                noise_model = data.get('noise_model', 'gaussian')
                nu_val = data.get('nu')
                
                result = {
                    'q': float(q_val),
                    'c': float(c_val),
                    'noise_model': noise_model,
                    'nu': float(nu_val) if nu_val is not None else None,
                    'source': 'tuned_cache',
                    'timestamp': data.get('timestamp'),
                    'delta_ll_vs_zero': data.get('delta_ll_vs_zero'),
                    'pit_ks_pvalue': data.get('pit_ks_pvalue'),
                    'bic': data.get('bic'),
                    'aic': data.get('aic'),
                    'best_model_by_bic': data.get('best_model_by_bic', 'kalman_drift'),
                    'model_comparison': data.get('model_comparison', {}),
                    'delta_ll_vs_const': data.get('delta_ll_vs_const'),
                    'delta_ll_vs_ewma': data.get('delta_ll_vs_ewma')
                }
                
                return result
        
        return None
        
    except Exception as e:
        if os.getenv('DEBUG'):
            print(f"Warning: Failed to load tuned params for {asset_symbol}: {e}")
        return None


def _kalman_filter_drift(ret: pd.Series, vol: pd.Series, q: Optional[float] = None, optimize_q: bool = True, asset_symbol: Optional[str] = None) -> Dict[str, pd.Series]:
    """
    Kalman filter for time-varying drift estimation with optional q optimization.
    
    State-space model:
        r_t = μ_t + ε_t,  ε_t ~ N(0, σ_t²) or t(ν, σ_t²)  (observation equation)
        μ_t = μ_{t-1} + η_t,  η_t ~ N(0, q_t)  (state transition, random walk)
    
    Level-7+ robust filtering: Student-t innovations option for heavy-tailed observations.
    When KALMAN_ROBUST_T=true, observation noise uses Student-t distribution with
    degrees of freedom ν, providing robustness to outliers and extreme market events.
    
    Where:
        r_t: observed return at time t
        μ_t: latent drift (hidden state)
        σ_t: conditional volatility from GARCH/EWMA
        q_t: drift evolution variance (process noise, possibly time-varying)
        ν: Student-t degrees of freedom (if robust mode enabled)
    
    Args:
        ret: Returns series
        vol: Conditional volatility series (from GARCH or EWMA)
        q: Process noise variance. If None, estimated via heuristic or optimization.
        optimize_q: If True and q is None, optimize q via marginal likelihood maximization.
        
    Returns:
        Dictionary with:
            - mu_kf_filtered: Forward-pass filtered drift estimates
            - mu_kf_smoothed: Backward-pass smoothed drift estimates (preferred)
            - var_kf_filtered: Forward-pass filtered drift variance
            - var_kf_smoothed: Backward-pass smoothed drift variance
            - kalman_gain: Kalman gain series (diagnostic)
            - log_likelihood: Total log-likelihood of observations
            - innovations: Prediction errors (diagnostic)
            - q_optimal: Optimized or heuristic q value used
            - q_heuristic: Baseline heuristic q for comparison
            - q_optimization_attempted: Whether q optimization was attempted
            - robust_t_mode: Whether Student-t innovations were used
            - nu_robust: Degrees of freedom for Student-t (if robust mode)
    """
    ret_clean = _ensure_float_series(ret).dropna()
    vol_clean = _ensure_float_series(vol).reindex(ret_clean.index).dropna()
    
    # Align series
    df = pd.concat([ret_clean, vol_clean], axis=1, join='inner').dropna()
    if len(df) < 50:
        # Not enough data for stable Kalman filtering
        return {}
    
    df.columns = ['ret', 'vol']
    y = df['ret'].values.astype(float)  # observations
    sigma = df['vol'].values.astype(float)  # observation noise std
    T = len(y)
    idx = df.index
    
    # Compute heuristic baseline q
    med_var = float(np.nanmedian(sigma ** 2))
    q_heuristic = 0.01 * med_var  # 1% of typical observation variance
    q_heuristic = float(max(q_heuristic, 1e-10))
    
    # Determine whether to use heteroskedastic process noise (Level-7 refinement)
    # q_t = c * σ_t² makes drift uncertainty adaptive to market stress
    use_heteroskedastic = os.getenv("KALMAN_HETEROSKEDASTIC", "true").strip().lower() == "true"
    
    # Determine q or c to use
    q_optimization_attempted = False
    c_optimal = None
    q_t_series = None  # Will store time-varying q_t if heteroskedastic
    tuned_params_source = None
    
    # Priority 1: Try to load pre-tuned parameters from cache (if asset_symbol provided)
    if q is None and asset_symbol is not None:
        tuned_params = _load_tuned_kalman_params(asset_symbol)
        if tuned_params is not None:
            q = tuned_params['q']
            c_optimal = tuned_params['c']
            tuned_params_source = 'cache'
            
            # For heteroskedastic mode, use cached c to build q_t series
            if use_heteroskedastic and c_optimal is not None:
                q_t_series = c_optimal * (sigma ** 2)
                q = float(np.mean(q_t_series))  # mean for reporting
            
            # Log cache usage for diagnostics
            if os.getenv('DEBUG'):
                print(f"  Using tuned params from cache: q={q:.2e}, c={c_optimal:.3f}")
    
    # Priority 2: Optimize if not provided and cache miss
    if q is None and optimize_q and T >= 252:
        q_optimization_attempted = True
        from scipy.optimize import minimize_scalar
        
        if use_heteroskedastic:
            # Optimize c for heteroskedastic process noise: q_t = c * σ_t²
            # c represents the ratio of drift uncertainty to observation uncertainty
            
            # Heuristic c: same as q_heuristic / med_var = 0.01
            c_heuristic = 0.01
            
            # Define negative log-likelihood as function of log(c)
            def neg_ll_log_c(log_c_val: float) -> float:
                c_trial = float(np.exp(log_c_val))
                try:
                    ll = _compute_kalman_log_likelihood_heteroskedastic(y, sigma, c_trial)
                    if not np.isfinite(ll):
                        return 1e12
                    return float(-ll)
                except Exception:
                    return 1e12
            
            # Search bounds in log-space: c in [0.0001, 1.0]
            log_c_min = np.log(0.0001)
            log_c_max = np.log(1.0)
            
            try:
                # Brent method (1D optimization without derivatives)
                result = minimize_scalar(
                    neg_ll_log_c,
                    bounds=(log_c_min, log_c_max),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                if result.success and np.isfinite(result.x):
                    c_optimal = float(np.exp(result.x))
                else:
                    c_optimal = c_heuristic
            except Exception:
                c_optimal = c_heuristic
            
            # Build time-varying q_t series
            q_t_series = c_optimal * (sigma ** 2)
            q = float(np.mean(q_t_series))  # mean for reporting
            
        else:
            # Optimize constant q via marginal likelihood maximization
            # Use Brent method in log-space for robust optimization over several orders of magnitude
            
            # Define negative log-likelihood as function of log(q) for numerical stability
            def neg_ll_log_q(log_q_val: float) -> float:
                q_trial = float(np.exp(log_q_val))
                try:
                    ll = _compute_kalman_log_likelihood(y, sigma, q_trial)
                    if not np.isfinite(ll):
                        return 1e12
                    return float(-ll)
                except Exception:
                    return 1e12
            
            # Search bounds in log-space: q in [0.0001*med_var, 1.0*med_var]
            log_q_min = np.log(max(0.0001 * med_var, 1e-12))
            log_q_max = np.log(max(1.0 * med_var, 1e-6))
            
            try:
                # Brent method (1D optimization without derivatives)
                result = minimize_scalar(
                    neg_ll_log_q,
                    bounds=(log_q_min, log_q_max),
                    method='bounded',
                    options={'xatol': 1e-6}
                )
                if result.success and np.isfinite(result.x):
                    q = float(np.exp(result.x))
                else:
                    q = q_heuristic
            except Exception:
                q = q_heuristic
                
    elif q is None:
        # Use heuristic if optimization disabled or insufficient data
        if use_heteroskedastic:
            c_optimal = 0.01
            q_t_series = c_optimal * (sigma ** 2)
            q = float(np.mean(q_t_series))
        else:
            q = q_heuristic
    else:
        q = float(max(q, 1e-10))
    
    # Level-7+ Robust Kalman filtering with Student-t innovations
    # When enabled, uses t-distribution for observation noise to handle outliers
    use_robust_t = os.getenv("KALMAN_ROBUST_T", "false").strip().lower() == "true"
    nu_robust = None
    
    if use_robust_t:
        # Estimate degrees of freedom for Student-t from standardized residuals
        # Use simple variance-based estimator: higher variance => lower nu (heavier tails)
        try:
            std_resid = y / np.maximum(sigma, 1e-12)
            std_resid = std_resid[np.isfinite(std_resid)]
            if len(std_resid) >= 100:
                # Robust estimation via method of moments
                # For Student-t: Var(X) = ν/(ν-2) for ν > 2
                # Sample excess variance relative to unit normal suggests tail heaviness
                sample_var = float(np.var(std_resid))
                if sample_var > 1.5:
                    # Heavier tails than normal: solve ν/(ν-2) = sample_var
                    # => ν = 2*sample_var / (sample_var - 1)
                    nu_est = 2.0 * sample_var / max(sample_var - 1.0, 0.1)
                    nu_robust = float(np.clip(nu_est, 4.5, 50.0))
                else:
                    # Light tails: use high nu (near-normal)
                    nu_robust = 30.0
            else:
                # Insufficient data: default to moderate heavy tails
                nu_robust = 10.0
        except Exception:
            nu_robust = 10.0
    
    # Initialize state and covariance
    # Prior: μ_0 ~ N(0, large variance) to represent initial uncertainty
    # Level-7+ Regime-aware prior: use regime-specific drift expectation if available
    mu_prior = 0.0
    P_prior = 1.0  # initial uncertainty
    regime_prior_info = None
    
    # Check if regime-aware initialization is enabled
    use_regime_prior = os.getenv("KALMAN_REGIME_PRIOR", "false").strip().lower() == "true"
    
    if use_regime_prior:
        # Estimate regime-specific drift priors from historical data
        regime_prior_info = _estimate_regime_drift_priors(ret_clean, vol_clean)
        if regime_prior_info is not None:
            # Use regime-conditional drift as prior mean
            mu_prior = regime_prior_info.get("current_drift_prior", 0.0)
            # Keep prior variance at 1.0 to allow filter to adapt quickly
    
    # Storage for forward pass
    mu_filtered = np.zeros(T, dtype=float)
    P_filtered = np.zeros(T, dtype=float)
    K_gain = np.zeros(T, dtype=float)
    innovations = np.zeros(T, dtype=float)
    innovation_vars = np.zeros(T, dtype=float)
    
    # Forward pass (Kalman filter)
    mu_t = mu_prior
    P_t = P_prior
    log_likelihood = 0.0
    
    for t in range(T):
        # Prediction step: project state forward
        # μ_{t|t-1} = μ_{t-1|t-1}  (random walk has no drift term in transition)
        # P_{t|t-1} = P_{t-1|t-1} + q_t
        # Use time-varying q_t if heteroskedastic, else constant q
        q_t = float(q_t_series[t]) if q_t_series is not None else q
        mu_pred = mu_t
        P_pred = P_t + q_t
        
        # Observation noise variance at time t
        R_t = sigma[t] ** 2
        R_t = float(max(R_t, 1e-12))  # floor
        
        # Innovation (prediction error)
        innov = y[t] - mu_pred
        
        # Innovation variance: S_t = H P_{t|t-1} H^T + R_t
        # With H=1 (observation matrix), S_t = P_pred + R_t
        S_t = P_pred + R_t
        S_t = float(max(S_t, 1e-12))
        
        # Kalman gain: K_t = P_{t|t-1} H^T S_t^{-1}
        # With H=1, K_t = P_pred / S_t
        K_t_base = P_pred / S_t
        
        # Robust Kalman: adaptive gain based on Student-t likelihood
        # Downweight outliers by adjusting gain based on innovation magnitude
        if use_robust_t and nu_robust is not None:
            # Robust weight: w_t = (ν + 1) / (ν + z_t²)
            # where z_t = innov / sqrt(S_t) is standardized innovation
            z_t_sq = (innov ** 2) / S_t
            w_t = (nu_robust + 1.0) / (nu_robust + z_t_sq)
            w_t = float(np.clip(w_t, 0.01, 1.0))  # bounded weight
            K_t = w_t * K_t_base  # adaptive gain
        else:
            K_t = K_t_base  # standard Kalman gain
        
        # Update step: refine state estimate with observation
        # μ_{t|t} = μ_{t|t-1} + K_t (y_t - μ_{t|t-1})
        mu_t = mu_pred + K_t * innov
        
        # Update covariance: P_{t|t} = (1 - K_t H) P_{t|t-1}
        # With H=1, P_{t|t} = (1 - K_t) P_pred
        # For robust case, use Joseph form for numerical stability
        if use_robust_t:
            # Joseph form: P_{t|t} = (I - K_t H)P_{t|t-1}(I - K_t H)^T + K_t R_t K_t^T
            # With H=1: P_{t|t} = (1-K_t)²P_pred + K_t²R_t
            P_t = (1.0 - K_t) ** 2 * P_pred + K_t ** 2 * R_t
        else:
            P_t = (1.0 - K_t) * P_pred
        P_t = float(max(P_t, 1e-12))  # ensure positive
        
        # Store filtered estimates
        mu_filtered[t] = mu_t
        P_filtered[t] = P_t
        K_gain[t] = K_t
        innovations[t] = innov
        innovation_vars[t] = S_t
        
        # Accumulate log-likelihood: ln p(y_t | y_{1:t-1})
        if use_robust_t and nu_robust is not None:
            # Student-t log-likelihood
            try:
                # Standardized innovation
                z_t = innov / np.sqrt(S_t)
                # Log-likelihood: log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(πνS_t) - ((ν+1)/2)*log(1 + z_t²/ν)
                from scipy.special import gammaln
                ll_t = (
                    gammaln((nu_robust + 1.0) / 2.0)
                    - gammaln(nu_robust / 2.0)
                    - 0.5 * np.log(np.pi * nu_robust * S_t)
                    - ((nu_robust + 1.0) / 2.0) * np.log(1.0 + (z_t ** 2) / nu_robust)
                )
                if np.isfinite(ll_t):
                    log_likelihood += float(ll_t)
            except Exception:
                pass
        else:
            # Gaussian log-likelihood
            try:
                ll_t = -0.5 * (np.log(2.0 * np.pi * S_t) + (innov ** 2) / S_t)
                if np.isfinite(ll_t):
                    log_likelihood += ll_t
            except Exception:
                pass
    
    # Backward pass (Rauch-Tung-Striebel smoother)
    # Smoothing uses all data (past and future) for refined estimates
    mu_smoothed = np.zeros(T, dtype=float)
    P_smoothed = np.zeros(T, dtype=float)
    
    # Initialize backward pass with last filtered estimate
    mu_smoothed[T-1] = mu_filtered[T-1]
    P_smoothed[T-1] = P_filtered[T-1]
    
    for t in range(T-2, -1, -1):
        # Smoother gain: C_t = P_{t|t} / P_{t+1|t}
        # Where P_{t+1|t} = P_filtered[t] + q_t (predicted covariance for next step)
        # Use time-varying q_t if heteroskedastic, else constant q
        q_t = float(q_t_series[t]) if q_t_series is not None else q
        P_pred_next = P_filtered[t] + q_t
        P_pred_next = float(max(P_pred_next, 1e-12))
        
        C_t = P_filtered[t] / P_pred_next
        
        # Smoothed state: μ_{t|T} = μ_{t|t} + C_t (μ_{t+1|T} - μ_{t+1|t})
        mu_pred_next = mu_filtered[t]  # prediction for t+1 from t (random walk)
        mu_smoothed[t] = mu_filtered[t] + C_t * (mu_smoothed[t+1] - mu_pred_next)
        
        # Smoothed covariance: P_{t|T} = P_{t|t} + C_t (P_{t+1|T} - P_{t+1|t}) C_t
        P_smoothed[t] = P_filtered[t] + C_t * (P_smoothed[t+1] - P_pred_next) * C_t
        P_smoothed[t] = float(max(P_smoothed[t], 1e-12))
    
    # Compute Kalman gain statistics for situational awareness
    kalman_gain_mean = float(np.mean(K_gain))
    kalman_gain_recent = float(K_gain[-1]) if len(K_gain) > 0 else float("nan")
    
    # Refinement 3: Innovation whiteness test (Ljung-Box)
    # Test if standardized innovations are white noise (model adequacy)
    innovation_whiteness = _test_innovation_whiteness(innovations, innovation_vars, lags=min(20, T // 5))
    
    # Build output series aligned with original index
    return {
        "mu_kf_filtered": pd.Series(mu_filtered, index=idx, name="mu_kf_filtered"),
        "mu_kf_smoothed": pd.Series(mu_smoothed, index=idx, name="mu_kf_smoothed"),
        "var_kf_filtered": pd.Series(P_filtered, index=idx, name="var_kf_filtered"),
        "var_kf_smoothed": pd.Series(P_smoothed, index=idx, name="var_kf_smoothed"),
        "kalman_gain": pd.Series(K_gain, index=idx, name="kalman_gain"),
        "innovations": pd.Series(innovations, index=idx, name="innovations"),
        "innovation_vars": pd.Series(innovation_vars, index=idx, name="innovation_vars"),
        "log_likelihood": float(log_likelihood),
        "process_noise_var": float(q),
        "n_obs": int(T),
        # Refinement 1: q optimization metadata
        "q_optimal": float(q),
        "q_heuristic": float(q_heuristic),
        "q_optimization_attempted": bool(q_optimization_attempted),
        # Refinement 2: Kalman gain statistics (situational awareness)
        "kalman_gain_mean": kalman_gain_mean,
        "kalman_gain_recent": kalman_gain_recent,
        # Refinement 3: Innovation whiteness test (model adequacy)
        "innovation_whiteness": innovation_whiteness,
        # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
        "kalman_heteroskedastic_mode": bool(use_heteroskedastic),
        "kalman_c_optimal": float(c_optimal) if c_optimal is not None else None,
        "kalman_q_t_mean": float(np.mean(q_t_series)) if q_t_series is not None else None,
        "kalman_q_t_std": float(np.std(q_t_series)) if q_t_series is not None else None,
        "kalman_q_t_min": float(np.min(q_t_series)) if q_t_series is not None else None,
        "kalman_q_t_max": float(np.max(q_t_series)) if q_t_series is not None else None,
        # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
        "kalman_robust_t_mode": bool(use_robust_t),
        "kalman_nu_robust": float(nu_robust) if nu_robust is not None else None,
        # Level-7+ Refinement: Regime-dependent drift priors
        "kalman_regime_prior_used": bool(use_regime_prior and regime_prior_info is not None),
        "kalman_regime_info": regime_prior_info if regime_prior_info is not None else {},
    }


def compute_features(px: pd.Series, asset_symbol: Optional[str] = None) -> Dict[str, pd.Series]:
    """
    Compute features from price series for signal generation.
    
    Args:
        px: Price series
        asset_symbol: Asset symbol (e.g., "PLNJPY=X") for loading tuned Kalman parameters
    
    Returns:
        Dictionary of computed features
    """
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

    # ========================================
    # Pillar 1: Model-Based Drift Estimation
    # ========================================
    # Use best model selected by BIC from tune_q_mle.py model comparison:
    # - zero_drift: μ = 0 (no predictable drift)
    # - constant_drift: μ = constant (fixed drift)
    # - ewma_drift: μ = EWMA of returns (adaptive)
    # - kalman_drift: μ from Kalman filter (state-space model)
    
    # Load tuned parameters and model selection results
    tuned_params = None
    if asset_symbol is not None:
        tuned_params = _load_tuned_kalman_params(asset_symbol)
    
    best_model = tuned_params.get('best_model_by_bic', 'kalman_drift') if tuned_params else 'kalman_drift'
    
    # Print model selection information for user transparency
    if asset_symbol and tuned_params:
        model_comparison = tuned_params.get('model_comparison', {})
        print(f"\n{'='*80}")
        print(f"📊 Model Selection for {asset_symbol}")
        print(f"{'='*80}")
        print(f"Selected Model: {best_model.upper().replace('_', ' ')}")
        
        if model_comparison:
            print(f"\nBIC Comparison (lower is better):")
            for model_name, metrics in sorted(model_comparison.items(), key=lambda x: x[1].get('bic', float('inf'))):
                bic = metrics.get('bic', float('nan'))
                ll = metrics.get('ll', float('nan'))
                n_params = metrics.get('n_params', 0)
                marker = " ← SELECTED" if model_name == best_model else ""
                print(f"  {model_name:20s}: BIC={bic:10.1f}, LL={ll:10.1f}, params={n_params}{marker}")
        
        # Show delta LL comparisons
        delta_ll_zero = tuned_params.get('delta_ll_vs_zero', float('nan'))
        delta_ll_const = tuned_params.get('delta_ll_vs_const', float('nan'))
        delta_ll_ewma = tuned_params.get('delta_ll_vs_ewma', float('nan'))
        
        print(f"\nLog-Likelihood Improvements (vs baselines):")
        print(f"  vs Zero-drift:     ΔLL = {delta_ll_zero:+7.2f}")
        print(f"  vs Constant-drift: ΔLL = {delta_ll_const:+7.2f}")
        print(f"  vs EWMA-drift:     ΔLL = {delta_ll_ewma:+7.2f}")
        
        # Explain the selection
        print(f"\nRationale:")
        if best_model == 'zero_drift':
            print(f"  • No predictable drift detected (μ = 0 is best fit)")
            print(f"  • Complex drift models do not improve fit enough to justify added parameters")
        elif best_model == 'constant_drift':
            print(f"  • Drift exists but is constant over time (μ = {ret.mean():.6f})")
            print(f"  • Time-varying drift models do not justify additional complexity")
        elif best_model == 'ewma_drift':
            print(f"  • Drift varies over time following recent return patterns")
            print(f"  • EWMA provides better fit than constant or Kalman state-space models")
        else:  # kalman_drift
            print(f"  • Drift evolves stochastically (state-space model justified)")
            print(f"  • Kalman filter provides best balance of fit and parsimony")
            
            # Show Kalman parameters if available
            noise_model = tuned_params.get('noise_model', 'gaussian')
            q_val = tuned_params.get('q', float('nan'))
            c_val = tuned_params.get('c', float('nan'))
            nu_val = tuned_params.get('nu')
            
            print(f"\n  Kalman Parameters:")
            print(f"    Noise Model: {noise_model}")
            print(f"    Process Noise (q): {q_val:.2e}")
            print(f"    Observation Scale (c): {c_val:.3f}")
            if nu_val is not None:
                print(f"    Student-t df (ν): {nu_val:.1f}")
        
        print(f"{'='*80}\n")
    
    # Apply drift estimation based on best model selection
    if best_model == 'zero_drift':
        # Zero-drift model: assume no predictable drift (μ = 0)
        mu_kf = pd.Series(0.0, index=ret.index, name="mu_zero")
        var_kf = pd.Series(0.0, index=ret.index)
        kalman_available = False
        kalman_metadata = {
            "model_selected": "zero_drift",
            "reason": "BIC comparison favored zero-drift baseline",
            "delta_ll_vs_zero": tuned_params.get('delta_ll_vs_zero', float('nan')) if tuned_params else float('nan')
        }
    
    elif best_model == 'constant_drift':
        # Constant-drift model: μ = mean(returns) for all t
        const_drift = float(ret.mean())
        mu_kf = pd.Series(const_drift, index=ret.index, name="mu_const")
        var_kf = pd.Series(0.0, index=ret.index)
        kalman_available = False
        kalman_metadata = {
            "model_selected": "constant_drift",
            "constant_drift_value": const_drift,
            "reason": "BIC comparison favored constant-drift baseline",
            "delta_ll_vs_const": tuned_params.get('delta_ll_vs_const', float('nan')) if tuned_params else float('nan')
        }
    
    elif best_model == 'ewma_drift':
        # EWMA-drift model: μ = EWMA of past returns
        mu_ewma = ret.ewm(span=21, adjust=False).mean()
        mu_kf = mu_ewma.rename("mu_ewma")
        var_kf = pd.Series(0.0, index=ret.index)
        kalman_available = False
        kalman_metadata = {
            "model_selected": "ewma_drift",
            "ewma_span": 21,
            "reason": "BIC comparison favored EWMA-drift baseline",
            "delta_ll_vs_ewma": tuned_params.get('delta_ll_vs_ewma', float('nan')) if tuned_params else float('nan')
        }
    
    else:
        # Kalman-drift model (default): full state-space estimation
        # Run Kalman filter on returns with GARCH/EWMA volatility
        # Uses pre-tuned parameters from cache if available, otherwise auto-estimates
        kf_result = _kalman_filter_drift(ret, vol, q=None, asset_symbol=asset_symbol)
        
        # Extract Kalman-filtered drift estimates
        if kf_result and "mu_kf_smoothed" in kf_result:
            # Use backward-smoothed estimates (uses all data, statistically optimal)
            mu_kf = kf_result["mu_kf_smoothed"]
            var_kf = kf_result["var_kf_smoothed"]
            kalman_available = True
            kalman_metadata = {
                "log_likelihood": kf_result.get("log_likelihood", float("nan")),
                "process_noise_var": kf_result.get("process_noise_var", float("nan")),
                "n_obs": kf_result.get("n_obs", 0),
                # Refinement 1: q optimization metadata
                "q_optimal": kf_result.get("q_optimal", float("nan")),
                "q_heuristic": kf_result.get("q_heuristic", float("nan")),
                "q_optimization_attempted": kf_result.get("q_optimization_attempted", False),
                # Refinement 2: Kalman gain statistics (situational awareness)
                "kalman_gain_mean": kf_result.get("kalman_gain_mean", float("nan")),
                "kalman_gain_recent": kf_result.get("kalman_gain_recent", float("nan")),
                # Refinement 3: Innovation whiteness test (model adequacy)
                "innovation_whiteness": kf_result.get("innovation_whiteness", {}),
                # Level-7 Refinement: Heteroskedastic process noise
                "kalman_heteroskedastic_mode": kf_result.get("heteroskedastic_mode", False),
                "kalman_c_optimal": kf_result.get("c_optimal"),
                "kalman_q_t_mean": kf_result.get("q_t_mean"),
                "kalman_q_t_std": kf_result.get("q_t_std"),
                "kalman_q_t_min": kf_result.get("q_t_min"),
                "kalman_q_t_max": kf_result.get("q_t_max"),
                # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
                "kalman_robust_t_mode": kf_result.get("robust_t_mode", False),
                "kalman_nu_robust": kf_result.get("nu_robust"),
                # Level-7+ Refinement: Regime-dependent drift priors
                "kalman_regime_prior_used": kf_result.get("regime_prior_used", False),
                "kalman_regime_info": kf_result.get("regime_prior_info", {}),
            }
        else:
            # Fallback: use EWMA blend if Kalman fails
            mu_blend = 0.5 * mu_fast + 0.5 * mu_slow
            mu_kf = mu_blend
            var_kf = pd.Series(0.0, index=mu_kf.index)  # no uncertainty quantified
            kalman_available = False
            kalman_metadata = {}
    
    # Trend filter (200D z-distance) - kept for diagnostics
    sma200 = px.rolling(200).mean()
    trend_z = (px - sma200) / px.rolling(200).std()

    # HMM regime detection (for regime-aware adjustments, not drift estimation)
    # Fit HMM to get regime posteriors
    hmm_result_prelim = fit_hmm_regimes(
        {"ret": ret, "vol": vol},
        n_states=3,
        random_seed=42
    )
    
    # Apply light regime-aware shrinkage to Kalman drift in extreme regimes
    # (Kalman already handles uncertainty; this adds regime-specific conservatism)
    if hmm_result_prelim is not None and "posterior_probs" in hmm_result_prelim:
        try:
            posterior_probs = hmm_result_prelim["posterior_probs"]
            
            # Align posteriors with mu_kf index
            posterior_aligned = posterior_probs.reindex(mu_kf.index).ffill().fillna(0.333)
            
            # Extract regime probabilities
            regime_names = hmm_result_prelim["regime_names"]
            calm_idx = [k for k, v in regime_names.items() if v == "calm"]
            crisis_idx = [k for k, v in regime_names.items() if v == "crisis"]
            
            p_calm = posterior_aligned.iloc[:, calm_idx[0]].values if calm_idx else np.zeros(len(mu_kf))
            p_crisis = posterior_aligned.iloc[:, crisis_idx[0]].values if crisis_idx else np.zeros(len(mu_kf))
            
            # Light shrinkage in crisis regimes (Kalman handles most uncertainty)
            # Shrink toward zero in extreme crisis to be conservative
            shrinkage = 0.3 * p_crisis  # 0-30% shrinkage based on crisis probability
            shrinkage = np.clip(shrinkage, 0.0, 0.5)
            
            # Final drift: Kalman estimate with regime-aware shrinkage
            mu_final = pd.Series(
                (1.0 - shrinkage) * mu_kf.values,  # shrink toward zero in crisis
                index=mu_kf.index,
                name="mu_final"
            )
            
        except Exception:
            # Fallback: use Kalman estimate without regime adjustment
            mu_final = mu_kf.copy()
    else:
        # HMM not available: use pure Kalman estimate
        mu_final = mu_kf.copy()
    
    # Robust fallback for NaNs
    mu_final = mu_final.fillna(0.0)
    
    # Legacy aliases for backward compatibility
    mu_blend = 0.5 * mu_fast + 0.5 * mu_slow  # kept for diagnostics
    mu_post = mu_final  # primary drift estimate
    mu = mu_final  # shorthand

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
        # Pillar 1: Kalman filter drift estimation
        "mu_kf": mu_kf if kalman_available else mu_blend,  # Kalman-filtered drift
        "var_kf": var_kf if kalman_available else pd.Series(0.0, index=ret.index),  # drift variance
        "mu_final": mu_final,  # final drift after regime adjustment
        "kalman_available": kalman_available,  # flag for diagnostics
        "kalman_metadata": kalman_metadata,  # log-likelihood, process noise, etc.
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
        df = pd.concat([ret, vol], axis=1, join='inner').dropna()
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


def compute_all_diagnostics(px: pd.Series, feats: Dict[str, pd.Series], enable_oos: bool = False, enable_pit_calibration: bool = False, enable_model_comparison: bool = False) -> Dict:
    """
    Compute comprehensive diagnostics: log-likelihood monitoring, parameter stability, 
    and optionally out-of-sample tests, PIT calibration verification, and structural model comparison.
    
    Args:
        px: Price series
        feats: Feature dictionary from compute_features
        enable_oos: If True, run expensive out-of-sample validation
        enable_pit_calibration: If True, run PIT calibration verification (expensive)
        enable_model_comparison: If True, run structural model comparison (AIC/BIC falsifiability)
        
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
    
    # Pillar 1: Kalman filter drift diagnostics (with refinements)
    kalman_metadata = feats.get("kalman_metadata", {})
    if isinstance(kalman_metadata, dict):
        diagnostics["kalman_log_likelihood"] = kalman_metadata.get("log_likelihood", float("nan"))
        diagnostics["kalman_process_noise_var"] = kalman_metadata.get("process_noise_var", float("nan"))
        diagnostics["kalman_n_obs"] = kalman_metadata.get("n_obs", 0)
        # Refinement 1: q optimization results
        diagnostics["kalman_q_optimal"] = kalman_metadata.get("q_optimal", float("nan"))
        diagnostics["kalman_q_heuristic"] = kalman_metadata.get("q_heuristic", float("nan"))
        diagnostics["kalman_q_optimization_attempted"] = kalman_metadata.get("q_optimization_attempted", False)
        # Refinement 2: Kalman gain (situational awareness)
        diagnostics["kalman_gain_mean"] = kalman_metadata.get("kalman_gain_mean", float("nan"))
        diagnostics["kalman_gain_recent"] = kalman_metadata.get("kalman_gain_recent", float("nan"))
        # Refinement 3: Innovation whiteness (model adequacy)
        innovation_whiteness = kalman_metadata.get("innovation_whiteness", {})
        if isinstance(innovation_whiteness, dict):
            diagnostics["innovation_ljung_box_statistic"] = innovation_whiteness.get("ljung_box_statistic", float("nan"))
            diagnostics["innovation_ljung_box_pvalue"] = innovation_whiteness.get("ljung_box_pvalue", float("nan"))
            diagnostics["innovation_model_adequate"] = innovation_whiteness.get("model_adequate", None)
            diagnostics["innovation_lags_tested"] = innovation_whiteness.get("lags_tested", 0)
        # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
        diagnostics["kalman_heteroskedastic_mode"] = kalman_metadata.get("heteroskedastic_mode", False)
        diagnostics["kalman_c_optimal"] = kalman_metadata.get("c_optimal")
        diagnostics["kalman_q_t_mean"] = kalman_metadata.get("q_t_mean")
        diagnostics["kalman_q_t_std"] = kalman_metadata.get("q_t_std")
        diagnostics["kalman_q_t_min"] = kalman_metadata.get("q_t_min")
        diagnostics["kalman_q_t_max"] = kalman_metadata.get("q_t_max")
        # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
        diagnostics["kalman_robust_t_mode"] = kalman_metadata.get("robust_t_mode", False)
        diagnostics["kalman_nu_robust"] = kalman_metadata.get("nu_robust")
        # Level-7+ Refinement: Regime-dependent drift priors
        diagnostics["kalman_regime_prior_used"] = kalman_metadata.get("regime_prior_used", False)
        regime_prior_info = kalman_metadata.get("regime_prior_info", {})
        if isinstance(regime_prior_info, dict) and regime_prior_info:
            diagnostics["kalman_regime_current"] = regime_prior_info.get("current_regime", "")
            diagnostics["kalman_regime_drift_prior"] = regime_prior_info.get("current_drift_prior", float("nan"))
    
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
        # Tier 2: Add standard error for posterior parameter variance tracking
        diagnostics["student_t_se_nu"] = nu_info.get("se_nu", float("nan"))
    
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
    
    # 4. PIT calibration verification (Level-7: probability calibration test)
    if enable_pit_calibration and not px.empty and len(px) >= 1000:
        try:
            from pit_calibration import run_pit_calibration_test
            
            # Run calibration test for key horizons
            calibration_results = run_pit_calibration_test(
                px=px,
                horizons=[1, 21, 63],
                n_bins=10,
                train_days=504,
                test_days=21,
                max_predictions=500
            )
            
            if calibration_results:
                diagnostics["pit_calibration"] = calibration_results
                
                # Summary: calibration status per horizon
                for horizon, metrics in calibration_results.items():
                    diagnostics[f"pit_H{horizon}_ece"] = metrics.expected_calibration_error
                    diagnostics[f"pit_H{horizon}_calibrated"] = metrics.calibrated
                    diagnostics[f"pit_H{horizon}_diagnosis"] = metrics.calibration_diagnosis
                    diagnostics[f"pit_H{horizon}_n_predictions"] = metrics.n_predictions
        except Exception as e:
            diagnostics["pit_calibration_error"] = str(e)
    
    # 5. Structural model comparison (Level-7: formal falsifiability via AIC/BIC)
    if enable_model_comparison:
        try:
            from model_comparison import run_all_comparisons
            
            # Get required inputs
            ret = feats.get("ret", pd.Series(dtype=float))
            vol = feats.get("vol", pd.Series(dtype=float))
            garch_params = feats.get("garch_params", {})
            nu_info = feats.get("nu_info", {})
            kalman_metadata = feats.get("kalman_metadata", {})
            
            if not ret.empty and not vol.empty:
                # Run all model comparisons
                comparison_results = run_all_comparisons(
                    returns=ret,
                    volatility=vol,
                    garch_params=garch_params if isinstance(garch_params, dict) else None,
                    student_t_params=nu_info if isinstance(nu_info, dict) else None,
                    kalman_metadata=kalman_metadata if isinstance(kalman_metadata, dict) else None,
                )
                
                diagnostics["model_comparison"] = comparison_results
                
                # Summary: winner per category
                for category, result in comparison_results.items():
                    if result is not None and hasattr(result, 'winner_aic'):
                        diagnostics[f"model_comparison_{category}_winner_aic"] = result.winner_aic
                        diagnostics[f"model_comparison_{category}_winner_bic"] = result.winner_bic
                        diagnostics[f"model_comparison_{category}_recommendation"] = result.recommendation
        except Exception as e:
            diagnostics["model_comparison_error"] = str(e)
    
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


def _simulate_forward_paths(feats: Dict[str, pd.Series], H_max: int, n_paths: int = 3000, phi: float = 0.95, kappa: float = 1e-4) -> Dict[str, np.ndarray]:
    """Monte-Carlo forward simulation of cumulative log returns and volatility over 1..H_max.
    - Drift evolves as AR(1): mu_{t+1} = phi * mu_t + eta_t,  eta ~ N(0, q)
    - Volatility evolves via GARCH(1,1) when available; else held constant.
    - Innovations are Student-t with global df (nu_hat) scaled to unit variance.
    - Jump-diffusion (Merton model): captures discontinuous gap risk via rare large moves.
    
    Pillar 1 integration: Drift uncertainty from Kalman filter (var_kf) is propagated
    into process noise q, widening forecast confidence intervals when drift is uncertain.
    
    Level-7 parameter uncertainty: if PARAM_UNC environment variable is set to
    'sample' (default) and garch_params contains a covariance matrix, we sample
    (omega, alpha, beta) per path from N(theta_hat, Cov) with constraints, which
    widens confidence during regime shifts and narrows during stability.
    
    Stochastic volatility: Tracks full h_t (variance) trajectories across paths,
    enabling posterior uncertainty bands for volatility forecasts.
    
    Level-7 jump-diffusion: Merton model adds discontinuous jumps to capture gap risk:
        dS/S = μ dt + σ dW + J dN
    Where:
        - dW: continuous Brownian motion (Student-t innovations)
        - dN: Poisson process with intensity λ (jump arrival rate)
        - J: jump size ~ N(μ_J, σ_J²) (typically negative for crash risk)
    Jump parameters calibrated from historical returns: count large moves (>3σ) as jumps.
    
    Returns:
        Dictionary with:
            - 'returns': array of shape (H_max, n_paths) with cumulative log returns
            - 'volatility': array of shape (H_max, n_paths) with volatility (sigma_t = sqrt(h_t))
    """
    # Inputs at 'now'
    ret_idx = feats.get("ret", pd.Series(dtype=float)).index
    if ret_idx is None or len(ret_idx) == 0:
        return {
            'returns': np.zeros((H_max, n_paths), dtype=float),
            'volatility': np.zeros((H_max, n_paths), dtype=float)
        }
    mu_series = feats.get("mu_post")
    if not isinstance(mu_series, pd.Series) or mu_series.empty:
        mu_series = feats.get("mu")
    vol_series = feats.get("vol")
    if not isinstance(vol_series, pd.Series) or vol_series.empty or not isinstance(mu_series, pd.Series) or mu_series.empty:
        return {
            'returns': np.zeros((H_max, n_paths), dtype=float),
            'volatility': np.zeros((H_max, n_paths), dtype=float)
        }
    mu_now = float(mu_series.iloc[-1]) if len(mu_series) else 0.0
    vol_now = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
    vol_now = float(max(vol_now, 1e-6))
    
    # Pillar 1: Extract Kalman drift uncertainty for proper uncertainty propagation
    var_kf_series = feats.get("var_kf")
    if isinstance(var_kf_series, pd.Series) and not var_kf_series.empty:
        var_kf_now = float(var_kf_series.iloc[-1])
        var_kf_now = float(max(var_kf_now, 0.0))
    else:
        var_kf_now = 0.0  # fallback if Kalman not available

    # Tail parameter (global nu) with posterior uncertainty
    nu_hat_series = feats.get("nu_hat")
    nu_info = feats.get("nu_info", {})
    
    if isinstance(nu_hat_series, pd.Series) and not nu_hat_series.empty:
        nu_hat = float(nu_hat_series.iloc[-1])
    else:
        # fallback to last rolling nu
        nu_hat, _ = _tail2("nu", 50.0)
        if not np.isfinite(nu_hat):
            nu_hat = 50.0
    # Clip to safe range
    nu_hat = float(np.clip(nu_hat, 4.5, 500.0))
    
    # Extract standard error for ν (Tier 2: posterior parameter variance)
    se_nu = None
    if isinstance(nu_info, dict) and "se_nu" in nu_info:
        se_nu_val = nu_info.get("se_nu", float("nan"))
        if np.isfinite(se_nu_val) and se_nu_val > 0:
            se_nu = float(se_nu_val)
    
    # Determine if ν sampling is enabled (Tier 2: propagate tail parameter uncertainty)
    nu_sample_mode = os.getenv("NU_SAMPLE", "true").strip().lower() == "true"
    
    # Sample ν per path if uncertainty available and sampling enabled
    if nu_sample_mode and se_nu is not None and se_nu > 0:
        rng = np.random.default_rng()
        # Sample from N(nu_hat, se_nu²) and clip to valid range
        nu_samples = rng.normal(loc=nu_hat, scale=se_nu, size=n_paths)
        nu_samples = np.clip(nu_samples, 4.5, 500.0)
    else:
        # Use point estimate for all paths
        nu_samples = np.full(n_paths, nu_hat, dtype=float)

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
    # Pillar 1: Incorporate Kalman drift uncertainty into process noise
    # This properly propagates drift estimation uncertainty into forecast confidence intervals
    h0 = vol_now ** 2
    q_baseline = kappa * h0  # baseline AR(1) process noise
    q_kalman = var_kf_now  # additional uncertainty from drift estimation
    
    # Combined process noise: baseline evolution + current drift uncertainty
    # When Kalman drift is uncertain, forecast intervals widen appropriately
    q = float(max(q_baseline + q_kalman, 1e-10))

    # Level-7 Jump-Diffusion: Calibrate jump parameters from historical returns
    # Detect large moves (>3σ outliers) as empirical jumps to estimate:
    #   - λ (jump intensity): frequency of jumps per day
    #   - μ_J (jump mean): average jump size
    #   - σ_J (jump std): volatility of jump sizes
    jump_intensity = 0.0
    jump_mean = 0.0
    jump_std = 0.05
    enable_jumps = os.getenv("ENABLE_JUMPS", "true").strip().lower() == "true"
    
    if enable_jumps:
        try:
            # Get historical returns for calibration
            ret_hist = feats.get("ret", pd.Series(dtype=float))
            vol_hist = feats.get("vol", pd.Series(dtype=float))
            
            if isinstance(ret_hist, pd.Series) and isinstance(vol_hist, pd.Series) and len(ret_hist) >= 252:
                # Align returns and volatility
                df_jump = pd.concat([ret_hist, vol_hist], axis=1, join='inner').dropna()
                if len(df_jump) >= 252:
                    df_jump.columns = ['ret', 'vol']
                    
                    # Identify jumps: returns that exceed 3σ threshold (outliers)
                    # Standardize returns by conditional volatility
                    z_scores = df_jump['ret'] / df_jump['vol']
                    jump_threshold = 3.0
                    jump_mask = np.abs(z_scores) > jump_threshold
                    
                    n_jumps = int(np.sum(jump_mask))
                    n_days = len(df_jump)
                    
                    if n_jumps > 0:
                        # Jump intensity: λ = frequency of jumps per day
                        jump_intensity = float(n_jumps / n_days)
                        
                        # Jump sizes: extract returns on jump days
                        jump_returns = df_jump.loc[jump_mask, 'ret'].values
                        
                        # Jump mean and std (typically negative mean for crash risk)
                        jump_mean = float(np.mean(jump_returns))
                        jump_std = float(np.std(jump_returns))
                        
                        # Floor jump std to avoid degenerate case
                        jump_std = float(max(jump_std, 0.01))
                    else:
                        # No historical jumps detected: use conservative defaults
                        jump_intensity = 0.01  # ~2.5 jumps per year
                        jump_mean = -0.02  # small negative bias (crash risk)
                        jump_std = 0.05
        except Exception:
            # Fallback to conservative defaults if calibration fails
            jump_intensity = 0.01
            jump_mean = -0.02
            jump_std = 0.05

    # Initialize state arrays (vectorized across paths)
    cum = np.zeros((H_max, n_paths), dtype=float)
    vol_paths = np.zeros((H_max, n_paths), dtype=float)  # Track volatility (sigma_t) at each horizon
    mu_t = np.full(n_paths, mu_now, dtype=float)
    h_t = np.full(n_paths, max(h0, 1e-8), dtype=float)

    rng = np.random.default_rng()

    for t in range(H_max):
        # Student-t shocks standardized to unit variance (continuous component)
        # Tier 2: Use path-specific ν samples for proper tail parameter uncertainty propagation
        # Draw Student-t per path with its own degrees of freedom
        z = np.zeros(n_paths, dtype=float)
        for path_idx in range(n_paths):
            nu_path = nu_samples[path_idx]
            # Draw from Student-t with df=nu_path and scale to unit variance
            z_raw = rng.standard_t(df=nu_path)
            # Variance of t(ν) is ν/(ν-2) for ν>2
            if nu_path > 2.0:
                t_var_path = nu_path / (nu_path - 2.0)
                t_scale_path = math.sqrt(t_var_path)
                z[path_idx] = float(z_raw / t_scale_path)
            else:
                # Edge case: use raw draw for very low ν (shouldn't happen with clipping)
                z[path_idx] = float(z_raw)
        
        eps = z
        sigma_t = np.sqrt(np.maximum(h_t, 1e-12))
        e_t = sigma_t * eps
        
        # Level-7 Jump-Diffusion: Add discontinuous jump component
        # Merton model: dS/S = μ dt + σ dW + J dN
        jump_component = np.zeros(n_paths, dtype=float)
        if enable_jumps and jump_intensity > 0:
            # Poisson arrivals: number of jumps in this time step
            # For daily data, dt=1, so intensity per step = jump_intensity
            n_jumps = rng.poisson(lam=jump_intensity, size=n_paths)
            
            # For paths with jumps, draw jump sizes from N(μ_J, σ_J²)
            # Total jump = sum of all jumps in this step (if multiple)
            for path_idx in range(n_paths):
                if n_jumps[path_idx] > 0:
                    # Draw jump sizes (log returns)
                    jump_sizes = rng.normal(loc=jump_mean, scale=jump_std, size=int(n_jumps[path_idx]))
                    jump_component[path_idx] = float(np.sum(jump_sizes))
        
        # Total return: continuous (drift + diffusion) + jumps
        r_t = mu_t + e_t + jump_component
        
        # Accumulate log return
        if t == 0:
            cum[t, :] = r_t
        else:
            cum[t, :] = cum[t-1, :] + r_t
        # Store volatility at this horizon (stochastic volatility tracking)
        vol_paths[t, :] = sigma_t
        # Evolve volatility via GARCH or hold constant on fallback
        if use_garch:
            h_t = omega_paths + alpha_paths * (e_t ** 2) + beta_paths * h_t
            h_t = np.clip(h_t, 1e-12, 1e4)
        # Evolve drift via AR(1)
        eta = rng.normal(loc=0.0, scale=math.sqrt(q), size=n_paths)
        mu_t = phi * mu_t + eta

    return {
        'returns': cum,
        'volatility': vol_paths
    }


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


def compute_dynamic_thresholds(
    skew: float,
    regime_meta: Dict[str, float],
    sig_H: float,
    med_vol_last: float,
    H: int
) -> Dict[str, float]:
    """
    Compute dynamic buy/sell thresholds with asymmetry and uncertainty adjustments.
    
    Level-7 modularization: Separates threshold computation from signal generation
    for better testability and maintainability.
    
    Args:
        skew: Return skewness (asymmetry measure)
        regime_meta: Regime detection metadata with method and probabilities
        sig_H: Forecast volatility at horizon H
        med_vol_last: Long-run median volatility
        H: Forecast horizon in days
        
    Returns:
        Dictionary with buy_thr, sell_thr, and uncertainty metrics
    """
    # Base thresholds
    base_buy, base_sell = 0.58, 0.42
    
    # Skew adjustment: shift thresholds based on return asymmetry
    g1 = float(np.clip(skew if np.isfinite(skew) else 0.0, -1.5, 1.5))
    skew_delta = 0.02 * float(np.tanh(abs(g1) / 0.75))
    
    if g1 < 0:  # Negative skew (crash risk)
        buy_thr = base_buy + skew_delta
        sell_thr = base_sell + skew_delta
    elif g1 > 0:  # Positive skew (rally potential)
        buy_thr = base_buy - skew_delta
        sell_thr = base_sell - skew_delta
    else:
        buy_thr, sell_thr = base_buy, base_sell
    
    # Regime-based uncertainty (HMM posterior entropy or vol regime deviation)
    if regime_meta.get("method") == "hmm_posterior":
        # Use Shannon entropy of regime posteriors as uncertainty measure
        probs = regime_meta.get("probabilities", {})
        entropy = 0.0
        for p in probs.values():
            if p > 1e-12:
                entropy -= p * np.log(p)
        # Normalize by max entropy (log(3) for 3 states)
        u_regime = float(np.clip(entropy / np.log(3.0), 0.0, 1.0))
    else:
        # Fallback: use vol_regime deviation if available
        vol_regime = regime_meta.get("vol_regime", 1.0)
        u_regime = float(np.clip(abs(vol_regime - 1.0) / 1.5, 0.0, 1.0)) if np.isfinite(vol_regime) else 0.5
    
    # Forecast uncertainty from realized vol vs historical
    med_sig_H = (med_vol_last * math.sqrt(H)) if (np.isfinite(med_vol_last) and med_vol_last > 0) else sig_H
    ratio = float(sig_H / med_sig_H) if med_sig_H > 0 else 1.0
    u_sig = float(np.clip(ratio - 1.0, 0.0, 1.0))
    
    # Combined uncertainty: regime entropy dominates, forecast uncertainty refines
    U = float(np.clip(0.5 * u_regime + 0.5 * u_sig, 0.0, 1.0))
    
    # Widen thresholds based on uncertainty
    widen_delta = 0.04 * U
    buy_thr += widen_delta
    sell_thr -= widen_delta
    
    # Clamp to reasonable ranges
    buy_thr = float(np.clip(buy_thr, 0.55, 0.70))
    sell_thr = float(np.clip(sell_thr, 0.30, 0.45))
    
    # Ensure minimum separation
    if buy_thr - sell_thr < 0.12:
        mid = 0.5
        sell_thr = min(sell_thr, mid - 0.06)
        buy_thr = max(buy_thr, mid + 0.06)
    
    return {
        "buy_thr": float(buy_thr),
        "sell_thr": float(sell_thr),
        "uncertainty": float(U),
        "u_regime": float(u_regime),
        "u_forecast": float(u_sig),
        "skew_adjustment": float(skew_delta),
    }


def apply_confirmation_logic(
    p_smoothed_now: float,
    p_smoothed_prev: float,
    p_raw: float,
    pos_strength: float,
    buy_thr: float,
    sell_thr: float,
    edge: float,
    edge_floor: float
) -> str:
    """
    Apply 2-day confirmation with hysteresis to reduce signal churn.
    
    Level-7 modularization: Separates confirmation logic from main signal flow.
    
    Args:
        p_smoothed_now: Smoothed probability (current)
        p_smoothed_prev: Smoothed probability (previous)
        p_raw: Raw probability without smoothing
        pos_strength: Position strength (Kelly fraction)
        buy_thr: Buy threshold
        sell_thr: Sell threshold
        edge: Composite edge score
        edge_floor: Minimum edge required to act
        
    Returns:
        Signal label: "STRONG BUY", "BUY", "HOLD", "SELL", or "STRONG SELL"
    """
    # Hysteresis bands (slightly wider than base thresholds)
    buy_enter = buy_thr + 0.01
    sell_enter = sell_thr - 0.01
    
    # Base label from 2-day confirmation (smoothed probabilities)
    label = "HOLD"
    if (p_smoothed_prev >= buy_enter) and (p_smoothed_now >= buy_enter):
        label = "BUY"
    elif (p_smoothed_prev <= sell_enter) and (p_smoothed_now <= sell_enter):
        label = "SELL"
    
    # Strong tiers based on raw conviction and Kelly strength
    if p_raw >= max(0.66, buy_thr + 0.06) and pos_strength >= 0.30:
        label = "STRONG BUY"
    if p_raw <= min(0.34, sell_thr - 0.06) and pos_strength >= 0.30:
        label = "STRONG SELL"
    
    # Transaction-cost hurdle: force HOLD if absolute edge below floor
    if np.isfinite(edge) and abs(edge) < float(edge_floor):
        label = "HOLD"
    
    return label


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
    sim_result = _simulate_forward_paths(feats, H_max=H_max, n_paths=3000)
    sims = sim_result['returns']
    vol_sims = sim_result['volatility']

    for H in horizons:
        # Use simulation at horizon H (1‑indexed in description; here index H-1)
        if H <= 0 or H > sims.shape[0]:
            sim_H = np.zeros(3000, dtype=float)
            vol_H = np.zeros(3000, dtype=float)
        else:
            sim_H = sims[H-1, :]
            vol_H = vol_sims[H-1, :]
        # Clean NaNs/Infs for returns
        sim_H = np.asarray(sim_H, dtype=float)
        sim_H = sim_H[np.isfinite(sim_H)]
        if sim_H.size == 0:
            sim_H = np.zeros(3000, dtype=float)
        # Clean NaNs/Infs for volatility
        vol_H = np.asarray(vol_H, dtype=float)
        vol_H = vol_H[np.isfinite(vol_H)]
        if vol_H.size == 0:
            vol_H = np.zeros(3000, dtype=float)
        # Simulated moments and probability
        mH = float(np.mean(sim_H))
        vH = float(np.var(sim_H, ddof=1)) if sim_H.size > 1 else 0.0
        sH = float(math.sqrt(max(vH, 1e-12)))
        z_stat = float(mH / sH) if sH > 0 else 0.0
        p_now = float(np.mean(sim_H > 0.0))
        
        # Stochastic volatility statistics (Level-7: full posterior uncertainty)
        vol_mean = float(np.mean(vol_H)) if vol_H.size > 0 else 0.0
        try:
            vol_ci_low = float(np.quantile(vol_H, lo_q))
            vol_ci_high = float(np.quantile(vol_H, hi_q))
        except Exception:
            vol_std = float(np.std(vol_H)) if vol_H.size > 1 else 0.0
            vol_ci_low = max(0.0, vol_mean - vol_std)
            vol_ci_high = vol_mean + vol_std
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

        # ========================================================================
        # Upgrade #2: Drift Confidence → Kelly Scaling
        # ========================================================================
        # Incorporate drift uncertainty (P_t) into Kelly denominator to prevent ruin
        # Add drift_weight based on model quality (ΔLL, PIT) to scale position size
        
        # Extract drift uncertainty (variance of drift estimate) from Kalman filter
        var_kf_series = feats.get("var_kf_smoothed", pd.Series(dtype=float))
        if isinstance(var_kf_series, pd.Series) and not var_kf_series.empty:
            P_t = float(var_kf_series.iloc[-1])  # Latest drift variance
        else:
            P_t = 0.0  # No drift uncertainty if Kalman not available
        
        # Ensure P_t is valid
        if not np.isfinite(P_t) or P_t < 0:
            P_t = 0.0
        
        # Kelly denominator: total variance = observation variance + drift uncertainty scaled by horizon
        # Original: denom = vH (observation variance over horizon H)
        # Upgraded: denom = vH + H * P_t (adds drift parameter uncertainty)
        denom_base = vH if vH > 0 else (sig_H ** 2 if sig_H > 0 else 1.0)
        denom = denom_base + float(H) * P_t
        
        # Compute drift_weight based on model quality metrics
        # - ΔLL < 0: drift model worse than zero-drift → weight = 0
        # - PIT p < 0.05: miscalibration warning → weight = 0.3
        # - Otherwise: well-calibrated drift → weight = 1.0
        kalman_metadata = feats.get("kalman_metadata", {})
        delta_ll = kalman_metadata.get("delta_ll_vs_zero", float("nan"))
        pit_pvalue = kalman_metadata.get("pit_ks_pvalue", float("nan"))
        
        drift_weight = 1.0  # Default: trust drift fully
        
        if np.isfinite(delta_ll) and delta_ll < 0:
            # Drift model performs worse than zero-drift baseline
            drift_weight = 0.0
        elif np.isfinite(pit_pvalue) and pit_pvalue < 0.05:
            # Calibration warning: model forecasts not well-calibrated
            drift_weight = 0.3
        
        # Fractional Kelly sizing (half‑Kelly) with drift confidence adjustment
        f_star = float(np.clip(mu_H / denom, -1.0, 1.0))
        pos_strength_raw = float(min(1.0, 0.5 * abs(f_star)))
        pos_strength = drift_weight * pos_strength_raw

        # Level-7 Modularization: Use helper function for dynamic thresholds
        # Enriched regime_meta with vol_regime for fallback path
        regime_meta_enriched = dict(regime_meta)
        regime_meta_enriched["vol_regime"] = vol_reg_now
        
        threshold_result = compute_dynamic_thresholds(
            skew=skew_now,
            regime_meta=regime_meta_enriched,
            sig_H=sig_H,
            med_vol_last=med_vol_last,
            H=H
        )
        
        buy_thr = threshold_result["buy_thr"]
        sell_thr = threshold_result["sell_thr"]
        U = threshold_result["uncertainty"]
        
        thresholds[int(H)] = {
            "buy_thr": float(buy_thr),
            "sell_thr": float(sell_thr),
            "uncertainty": float(U),
            "edge_floor": float(EDGE_FLOOR)
        }

        # Level-7 Modularization: Use helper function for confirmation logic
        label = apply_confirmation_logic(
            p_smoothed_now=p_s_now,
            p_smoothed_prev=p_s_prev,
            p_raw=p_now,
            pos_strength=pos_strength,
            buy_thr=buy_thr,
            sell_thr=sell_thr,
            edge=edge_now,
            edge_floor=EDGE_FLOOR
        )

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
            vol_mean=float(vol_mean),
            vol_ci_low=float(vol_ci_low),
            vol_ci_high=float(vol_ci_high),
            regime=reg,
            label=label,
        ))

    return sigs, thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate signals across multiple horizons for PLN/JPY, Gold (PLN), Silver (PLN), Bitcoin (PLN), and MicroStrategy (PLN).")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--horizons", type=str, default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--assets", type=str, default=",".join(DEFAULT_ASSET_UNIVERSE), help="Comma-separated Yahoo symbols or friendly names. Metals, FX and USD/EUR/GBP/JPY/CAD/DKK/KRW assets are converted to PLN.")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--cache-json", type=str, default=DEFAULT_CACHE_PATH, help="Path to auto-write cache JSON (default cache/fx_plnjpy.json)")
    p.add_argument("--from-cache", action="store_true", help="Render tables from cache JSON and skip computation")
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
    p.add_argument("--pit-calibration", action="store_true", help="Enable PIT calibration verification: tests if predicted probabilities match actual outcomes (Level-7 requirement, very expensive).")
    p.add_argument("--model-comparison", action="store_true", help="Enable structural model comparison: GARCH vs EWMA, Student-t vs Gaussian, Kalman vs EWMA using AIC/BIC (Level-7 falsifiability).")
    p.add_argument("--validate-kalman", action="store_true", help="🧪 Run Level-7 Kalman validation science: drift reasonableness, predictive likelihood improvement, PIT calibration, and stress-regime behavior analysis.")
    p.add_argument("--validation-plots", action="store_true", help="Generate diagnostic plots for Kalman validation (requires --validate-kalman).")
    p.add_argument("--failures-json", type=str, default=os.path.join(os.path.dirname(__file__), "fx_failures.json"), help="Where to write failure log (set to '' to disable)")
    p.set_defaults(t_map=True)
    return p.parse_args()


def process_single_asset(args_tuple: Tuple) -> Optional[Dict]:
    """
    Worker function to process a single asset in parallel.
    Only performs computation, no console output.
    
    Args:
        args_tuple: (asset, args, horizons)
        
    Returns:
        Dictionary with processed results or None if failed
    """
    asset, args, horizons = args_tuple
    
    try:
        # Fetch price data
        try:
            px, title = fetch_px_asset(asset, args.start, args.end)
        except Exception as e:
            return {
                "status": "error",
                "asset": asset,
                "error": str(e)
            }
        
        # De-duplicate by resolved symbol
        canon = extract_symbol_from_title(title)
        if not canon:
            canon = asset.strip().upper()
        
        # Compute features and signals
        feats = compute_features(px, asset_symbol=asset)
        last_close = _to_float(px.iloc[-1])
        sigs, thresholds = latest_signals(feats, horizons, last_close=last_close, t_map=args.t_map, ci=args.ci)
        
        # Compute diagnostics if requested
        diagnostics = {}
        if args.diagnostics or args.diagnostics_lite or args.pit_calibration or args.model_comparison:
            enable_oos = args.diagnostics
            enable_pit = args.pit_calibration
            enable_model_comp = args.model_comparison
            diagnostics = compute_all_diagnostics(px, feats, enable_oos=enable_oos, enable_pit_calibration=enable_pit, enable_model_comparison=enable_model_comp)
        
        return {
            "status": "success",
            "asset": asset,
            "canon": canon,
            "title": title,
            "px": px,
            "feats": feats,
            "sigs": sigs,
            "thresholds": thresholds,
            "diagnostics": diagnostics,
            "last_close": last_close,
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "asset": asset,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def _process_assets_with_retries(assets: List[str], args: argparse.Namespace, horizons: List[int], max_retries: int = 3):
    """Run asset processing with bounded retries and collect failures.
    Retries only the assets that failed on prior attempts.
    """
    console = Console()
    pending = list(dict.fromkeys(a.strip() for a in assets if a and a.strip()))
    successes: List[Dict] = []
    failures: Dict[str, Dict[str, object]] = {}
    processed_canon = set()

    attempt = 1
    while attempt <= max_retries and pending:
        n_workers = min(cpu_count(), len(pending))
        console.print(f"[cyan]Attempt {attempt}/{max_retries}: processing {len(pending)} assets with {n_workers} workers...[/cyan]")
        work_items = [(asset, args, horizons) for asset in pending]

        with Pool(processes=n_workers) as pool:
            results = pool.map(process_single_asset, work_items)

        next_pending: List[str] = []
        for asset, result in zip(pending, results):
            if not result or result.get("status") != "success":
                err = (result or {}).get("error", "unknown error")
                try:
                    disp = _resolve_display_name(asset.strip().upper())
                except Exception:
                    disp = asset
                entry = failures.get(asset, {"attempts": 0, "last_error": None, "display_name": disp})
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry["last_error"] = err
                entry["display_name"] = entry.get("display_name") or disp
                failures[asset] = entry
                next_pending.append(asset)
                continue

                
            canon = result.get("canon") or asset.strip().upper()
            if canon in processed_canon:
                continue
            processed_canon.add(canon)
            successes.append(result)
            # drop from pending on success; nothing to add to next_pending
            if asset in failures:
                failures.pop(asset, None)

        pending = list(dict.fromkeys(next_pending))
        attempt += 1

    if pending:
        console.print(f"[yellow]Retry budget exhausted; {len(pending)} assets still failing.[/yellow]")
    return successes, failures


def main() -> None:
    args = parse_args()
    horizons = sorted({int(x.strip()) for x in args.horizons.split(",") if x.strip()})

    # Fast path: render from cache only
    if args.from_cache:
        cache_path = args.cache_json or DEFAULT_CACHE_PATH
        if not os.path.exists(cache_path):
            Console().print(f"[red]Cache not found:[/red] {cache_path}")
            return
        with open(cache_path, "r") as f:
            payload = json.load(f)
        horizons_cached = payload.get("horizons") or horizons
        summary_rows_cached = payload.get("summary_rows")
        # Fallback reconstruction if summary_rows missing
        if not summary_rows_cached:
            summary_rows_cached = []
            for asset in payload.get("assets", []):
                sym = asset.get("symbol") or ""
                title = asset.get("title") or sym
                asset_label = build_asset_display_label(sym, title)
                sector = asset.get("sector", "Unspecified")
                horizon_signals = {}
                for sig in asset.get("signals", []):
                    h = sig.get("horizon_days")
                    if h is None:
                        continue
                    horizon_signals[int(h)] = {"label": sig.get("label", "HOLD"), "profit_pln": float(sig.get("profit_pln", 0.0))}
                nearest_label = next(iter(horizon_signals.values()), {}).get("label", "HOLD")
                summary_rows_cached.append({"asset_label": asset_label, "horizon_signals": horizon_signals, "nearest_label": nearest_label, "sector": sector})
        try:
            render_sector_summary_tables(summary_rows_cached, horizons_cached)
        except Exception as e:
            Console().print(f"[yellow]Warning:[/yellow] Could not print summary tables from cache: {e}")
        return

    # Parse assets
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]

    console = Console()
    console.print(f"[cyan]Validating {len(assets)} requested assets against fx_data_utils mappings...[/cyan]")
    for a in assets:
        try:
            _resolve_symbol_candidates(a)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not resolve mapping for {a}: {e}")

    all_blocks = []  # for JSON export
    csv_rows_simple = []  # for CSV simple export
    csv_rows_detailed = []  # for CSV detailed export
    summary_rows = []  # for summary table across assets

    # =========================================================================
    # RETRYING PARALLEL PROCESSING: Compute features/signals with bounded retries
    # =========================================================================
    success_results, failures = _process_assets_with_retries(assets, args, horizons, max_retries=3)
    console.print(f"[green]✓ Parallel computation attempts complete. Now displaying results...[/green]\n")

    # =========================================================================
    # SEQUENTIAL DISPLAY & AGGREGATION: Process results in order with console output
    # =========================================================================
    caption_printed = False
    processed_syms = set()

    for result in success_results:
        # Handle None or error results
        if result is None:
            continue
            
        if result.get("status") == "error":
            asset = result.get("asset", "unknown")
            error = result.get("error", "unknown error")
            console.print(f"[red]Warning:[/red] Failed to process {asset}: {error}")
            if os.getenv('DEBUG'):
                traceback_info = result.get("traceback", "")
                if traceback_info:
                    console.print(f"[dim]{traceback_info}[/dim]")
            continue
        
        if result.get("status") != "success":
            continue
        
        # Extract computed results from worker
        asset = result["asset"]
        canon = result["canon"]
        title = result["title"]
        px = result["px"]
        feats = result["feats"]
        sigs = result["sigs"]
        thresholds = result["thresholds"]
        diagnostics = result["diagnostics"]
        last_close = result["last_close"]
        
        # De-duplicate check
        if canon in processed_syms:
            console.print(f"[yellow]Skipping duplicate:[/yellow] {title} (from input '{asset}')")
            continue
        processed_syms.add(canon)

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
            
            # Pillar 1: Kalman filter drift diagnostics (with refinements)
            if "kalman_log_likelihood" in diagnostics:
                diag_table.add_row("Kalman Filter Log-Likelihood", f"{diagnostics['kalman_log_likelihood']:.2f}")
                if "kalman_process_noise_var" in diagnostics:
                    diag_table.add_row("Kalman Process Noise (q)", f"{diagnostics['kalman_process_noise_var']:.6f}")
                if "kalman_n_obs" in diagnostics:
                    diag_table.add_row("Kalman Observations", f"{diagnostics['kalman_n_obs']}")
                
                # Refinement 1: q optimization results
                if "kalman_q_optimal" in diagnostics and "kalman_q_heuristic" in diagnostics:
                    q_opt = diagnostics["kalman_q_optimal"]
                    q_heur = diagnostics["kalman_q_heuristic"]
                    q_optimized = diagnostics.get("kalman_q_optimization_attempted", False)
                    if np.isfinite(q_opt) and np.isfinite(q_heur):
                        ratio = q_opt / q_heur if q_heur > 0 else 1.0
                        opt_label = "optimized" if q_optimized else "heuristic"
                        diag_table.add_row(f"  q ({opt_label})", f"{q_opt:.6f} ({ratio:.2f}× heuristic)")
                
                # Refinement 2: Kalman gain (situational awareness)
                if "kalman_gain_mean" in diagnostics and "kalman_gain_recent" in diagnostics:
                    gain_mean = diagnostics["kalman_gain_mean"]
                    gain_recent = diagnostics["kalman_gain_recent"]
                    if np.isfinite(gain_mean):
                        # Interpretation: high gain = aggressive learning, low gain = stable drift
                        interpretation = "aggressive" if gain_mean > 0.3 else ("moderate" if gain_mean > 0.1 else "stable")
                        diag_table.add_row(f"  Kalman Gain (mean)", f"{gain_mean:.4f} [{interpretation}]")
                    if np.isfinite(gain_recent):
                        diag_table.add_row(f"  Kalman Gain (recent)", f"{gain_recent:.4f}")
                
                # Refinement 3: Innovation whiteness (model adequacy)
                if "innovation_ljung_box_pvalue" in diagnostics:
                    pvalue = diagnostics["innovation_ljung_box_pvalue"]
                    model_adequate = diagnostics.get("innovation_model_adequate", None)
                    lags = diagnostics.get("innovation_lags_tested", 0)
                    if np.isfinite(pvalue) and model_adequate is not None:
                        color = "green" if model_adequate else "red"
                        status = "PASS" if model_adequate else "FAIL"
                        diag_table.add_row(f"  Innovation Whiteness (Ljung-Box)", f"[{color}]{status}[/{color}] (p={pvalue:.3f}, lags={lags})")
                
                # Level-7 Refinement: Heteroskedastic process noise (q_t = c * σ_t²)
                if "kalman_heteroskedastic_mode" in diagnostics:
                    hetero_mode = diagnostics.get("kalman_heteroskedastic_mode", False)
                    c_opt = diagnostics.get("kalman_c_optimal")
                    if hetero_mode and c_opt is not None and np.isfinite(c_opt):
                        diag_table.add_row(f"  Process Noise Mode", f"[cyan]Heteroskedastic[/cyan] (q_t = c·σ_t²)")
                        diag_table.add_row(f"  Scaling Factor (c)", f"{c_opt:.6f}")
                        # Show q_t statistics if available
                        q_t_mean = diagnostics.get("kalman_q_t_mean")
                        q_t_std = diagnostics.get("kalman_q_t_std")
                        q_t_min = diagnostics.get("kalman_q_t_min")
                        q_t_max = diagnostics.get("kalman_q_t_max")
                        if q_t_mean is not None and np.isfinite(q_t_mean):
                            diag_table.add_row(f"  q_t (mean ± std)", f"{q_t_mean:.6f} ± {q_t_std:.6f}" if q_t_std and np.isfinite(q_t_std) else f"{q_t_mean:.6f}")
                        if q_t_min is not None and q_t_max is not None and np.isfinite(q_t_min) and np.isfinite(q_t_max):
                            diag_table.add_row(f"  q_t range [min, max]", f"[{q_t_min:.6f}, {q_t_max:.6f}]")
                    elif not hetero_mode:
                        diag_table.add_row(f"  Process Noise Mode", f"Homoskedastic (constant q)")
                
                # Level-7+ Refinement: Robust Kalman filtering with Student-t innovations
                if "kalman_robust_t_mode" in diagnostics:
                    robust_t = diagnostics.get("kalman_robust_t_mode", False)
                    nu_robust = diagnostics.get("kalman_nu_robust")
                    if robust_t and nu_robust is not None and np.isfinite(nu_robust):
                        diag_table.add_row(f"  Innovation Distribution", f"[magenta]Student-t[/magenta] (robust filtering)")
                        diag_table.add_row(f"  Innovation ν (degrees of freedom)", f"{nu_robust:.2f}")
                    elif not robust_t:
                        diag_table.add_row(f"  Innovation Distribution", f"Gaussian (standard)")
                
                # Level-7+ Refinement: Regime-dependent drift priors
                if "kalman_regime_prior_used" in diagnostics:
                    regime_prior_used = diagnostics.get("kalman_regime_prior_used", False)
                    if regime_prior_used:
                        regime_current = diagnostics.get("kalman_regime_current", "")
                        drift_prior = diagnostics.get("kalman_regime_drift_prior")
                        if regime_current and drift_prior is not None and np.isfinite(drift_prior):
                            diag_table.add_row(f"  Drift Prior (regime-aware)", f"[yellow]Enabled[/yellow] (regime: {regime_current})")
                            diag_table.add_row(f"  E[μ | Regime={regime_current}]", f"{drift_prior:+.6f}")
                    else:
                        diag_table.add_row(f"  Drift Prior", f"Neutral (μ₀ = 0)")
            
            if "hmm_log_likelihood" in diagnostics:
                diag_table.add_row("HMM Regime Log-Likelihood", f"{diagnostics['hmm_log_likelihood']:.2f}")
                diag_table.add_row("HMM AIC", f"{diagnostics['hmm_aic']:.2f}")
                diag_table.add_row("HMM BIC", f"{diagnostics['hmm_bic']:.2f}")
            
            if "student_t_log_likelihood" in diagnostics:
                diag_table.add_row("Student-t Tail Log-Likelihood", f"{diagnostics['student_t_log_likelihood']:.2f}")
                diag_table.add_row("Student-t Degrees of Freedom (ν)", f"{diagnostics['student_t_nu']:.2f}")
                
                # Tier 2: Display ν standard error (posterior parameter variance)
                if "student_t_se_nu" in diagnostics:
                    se_nu = diagnostics["student_t_se_nu"]
                    if np.isfinite(se_nu) and se_nu > 0:
                        nu_hat = diagnostics.get("student_t_nu", float("nan"))
                        # Coefficient of variation: SE/estimate (relative uncertainty)
                        cv_nu = (se_nu / nu_hat) if np.isfinite(nu_hat) and nu_hat > 0 else float("nan")
                        uncertainty_level = "low" if cv_nu < 0.05 else ("moderate" if cv_nu < 0.10 else "high")
                        diag_table.add_row("  SE(ν) [posterior uncertainty]", f"{se_nu:.3f} ({cv_nu*100:.1f}% CV, {uncertainty_level})")
                    else:
                        diag_table.add_row("  SE(ν) [posterior uncertainty]", f"{se_nu:.3f}")
            
            # Tier 2: Parameter Uncertainty Summary (μ, σ, ν)
            param_unc_env = os.getenv("PARAM_UNC", "sample").strip().lower()
            nu_sample_env = os.getenv("NU_SAMPLE", "true").strip().lower()
            
            param_unc_active = {
                "μ (drift)": "Kalman var_kf → process noise q",
                "σ (volatility)": f"GARCH sampling: {'✓ enabled' if param_unc_env == 'sample' else '✗ disabled'}",
                "ν (tails)": f"Student-t sampling: {'✓ enabled' if nu_sample_env == 'true' else '✗ disabled'}"
            }
            
            diag_table.add_row("", "")  # spacer
            diag_table.add_row("[bold cyan]Tier 2: Posterior Parameter Variance[/bold cyan]", "[bold]Status[/bold]")
            for param, status in param_unc_active.items():
                if "✓" in status:
                    diag_table.add_row(f"  {param}", f"[green]{status}[/green]")
                elif "✗" in status:
                    diag_table.add_row(f"  {param}", f"[yellow]{status}[/yellow]")
                else:
                    diag_table.add_row(f"  {param}", status)
            
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
            
            # Display PIT calibration report if available
            if "pit_calibration" in diagnostics:
                try:
                    from pit_calibration import format_calibration_report
                    calibration_report = format_calibration_report(
                        calibration_results=diagnostics["pit_calibration"],
                        asset_name=asset
                    )
                    console.print(calibration_report)
                except Exception:
                    pass
            
            # Display model comparison results if available
            if "model_comparison" in diagnostics and diagnostics["model_comparison"]:
                from rich.table import Table
                comparison_results = diagnostics["model_comparison"]
                
                # Create comparison table for each category
                for category, result in comparison_results.items():
                    if result is None or not hasattr(result, 'winner_aic'):
                        continue
                    
                    category_title = {
                        'volatility': 'Volatility Models',
                        'tails': 'Tail Distribution Models',
                        'drift': 'Drift Models'
                    }.get(category, category.title())
                    
                    comp_table = Table(title=f"📊 Model Comparison: {category_title} — {asset}")
                    comp_table.add_column("Model", justify="left", style="cyan")
                    comp_table.add_column("Params", justify="right")
                    comp_table.add_column("Log-Lik", justify="right")
                    comp_table.add_column("AIC", justify="right")
                    comp_table.add_column("BIC", justify="right")
                    comp_table.add_column("Δ AIC", justify="right")
                    comp_table.add_column("Δ BIC", justify="right")
                    comp_table.add_column("Akaike Wt", justify="right")
                    
                    for model in result.models:
                        name = model.name
                        
                        # Highlight winners
                        if name == result.winner_aic and name == result.winner_bic:
                            name = f"[bold green]{name}[/bold green] ⭐"
                        elif name == result.winner_aic:
                            name = f"[bold yellow]{name}[/bold yellow] (AIC)"
                        elif name == result.winner_bic:
                            name = f"[bold blue]{name}[/bold blue] (BIC)"
                        
                        delta_aic = result.delta_aic.get(model.name, float('nan'))
                        delta_bic = result.delta_bic.get(model.name, float('nan'))
                        weight = result.akaike_weights.get(model.name, 0.0)
                        
                        # Color code deltas (lower is better)
                        if np.isfinite(delta_aic):
                            if delta_aic < 2.0:
                                delta_aic_str = f"[green]{delta_aic:+.1f}[/green]"
                            elif delta_aic < 7.0:
                                delta_aic_str = f"[yellow]{delta_aic:+.1f}[/yellow]"
                            else:
                                delta_aic_str = f"[red]{delta_aic:+.1f}[/red]"
                        else:
                            delta_aic_str = "—"
                        
                        if np.isfinite(delta_bic):
                            if delta_bic < 2.0:
                                delta_bic_str = f"[green]{delta_bic:+.1f}[/green]"
                            elif delta_bic < 10.0:
                                delta_bic_str = f"[yellow]{delta_bic:+.1f}[/yellow]"
                            else:
                                delta_bic_str = f"[red]{delta_bic:+.1f}[/red]"
                        else:
                            delta_bic_str = "—"
                        
                        comp_table.add_row(
                            name,
                            str(model.n_params),
                            f"{model.log_likelihood:.2f}",
                            f"{model.aic:.2f}",
                            f"{model.bic:.2f}",
                            delta_aic_str,
                            delta_bic_str,
                            f"{weight:.1%}" if np.isfinite(weight) else "—"
                        )
                    
                    # Add recommendation row
                    comp_table.add_row("", "", "", "", "", "", "", "")
                    comp_table.add_row(
                        f"[bold]Recommendation:[/bold]",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        ""
                    )
                    
                    console.print(comp_table)
                    console.print(f"[dim]{result.recommendation}[/dim]\n")
                
                # Summary interpretation
                console.print("[bold cyan]Model Comparison Interpretation:[/bold cyan]")
                console.print("[dim]• Δ AIC/BIC < 2: Substantial support (competitive models)[/dim]")
                console.print("[dim]• Δ AIC/BIC 4-7: Considerably less support[/dim]")
                console.print("[dim]• Δ AIC/BIC > 10: Essentially no support[/dim]")
                console.print("[dim]• Akaike weight: Probability this model is best[/dim]")
                console.print("[dim]• Lower AIC/BIC = better (fit + parsimony tradeoff)[/dim]\n")

        # 🧪 Level-7 Validation Science: Kalman Filter Validation Suite
        if args.validate_kalman:
            try:
                from kalman_validation import (
                    run_full_validation_suite,
                    validate_drift_reasonableness,
                    compare_predictive_likelihood,
                    validate_pit_calibration,
                    analyze_stress_regime_behavior
                )
                from rich.table import Table
                from rich.panel import Panel
                
                console = Console()
                console.print("\n")
                console.print(Panel.fit(
                    f"🧪 [bold cyan]Level-7 Validation Science[/bold cyan] — {asset}\n"
                    "[dim]Does my model behave like reality?[/dim]",
                    border_style="cyan"
                ))
                
                # Extract required series from features
                ret = feats.get("ret")
                mu_kf = feats.get("mu_kf", feats.get("mu"))
                var_kf = feats.get("var_kf", pd.Series(0.0, index=ret.index))
                vol = feats.get("vol")
                
                if mu_kf is not None and ret is not None and vol is not None:
                    # Prepare plot directory if plots requested
                    plot_dir = None
                    if args.validation_plots:
                        plot_dir = "plots/kalman_validation"
                        os.makedirs(plot_dir, exist_ok=True)
                    
                    # 1. Drift Reasonableness Validation
                    console.print("\n[bold yellow]1. Posterior Drift Reasonableness[/bold yellow]")
                    drift_result = validate_drift_reasonableness(
                        px, ret, mu_kf, var_kf, asset_name=asset,
                        plot=args.validation_plots,
                        save_path=f"{plot_dir}/{asset}_drift_validation.png" if plot_dir else None
                    )
                    
                    drift_table = Table(title="Drift Sanity Checks", show_header=True)
                    drift_table.add_column("Metric", style="cyan")
                    drift_table.add_column("Value", justify="right")
                    drift_table.add_column("Status", justify="center")
                    
                    drift_table.add_row(
                        "Observations",
                        str(drift_result.observations),
                        ""
                    )
                    drift_table.add_row(
                        "Drift Smoothness Ratio",
                        f"{drift_result.drift_smoothness_ratio:.4f}",
                        "✅" if drift_result.drift_smoothness_ratio < 0.5 else "⚠️"
                    )
                    drift_table.add_row(
                        "Crisis Uncertainty Spike",
                        f"{drift_result.crisis_uncertainty_spike:.2f}×",
                        "✅" if drift_result.crisis_uncertainty_spike > 1.5 else "⚠️"
                    )
                    drift_table.add_row(
                        "Regime Breaks Detected",
                        "Yes" if drift_result.regime_break_detected else "No",
                        "✅" if drift_result.regime_break_detected else "ℹ️"
                    )
                    drift_table.add_row(
                        "Noise Tracking Score",
                        f"{drift_result.noise_tracking_score:.4f}",
                        "✅" if drift_result.noise_tracking_score < 0.4 else "⚠️"
                    )
                    
                    console.print(drift_table)
                    console.print(f"[dim]{drift_result.diagnostic_message}[/dim]\n")
                    
                    # 2. Predictive Likelihood Improvement
                    console.print("[bold yellow]2. Predictive Likelihood Improvement[/bold yellow]")
                    ll_result = compare_predictive_likelihood(px, asset_name=asset)
                    
                    ll_table = Table(title="Model Comparison (Out-of-Sample)", show_header=True)
                    ll_table.add_column("Model", style="cyan")
                    ll_table.add_column("Log-Likelihood", justify="right")
                    ll_table.add_column("Δ LL", justify="right")
                    
                    ll_table.add_row("Kalman Filter", f"{ll_result.ll_kalman:.2f}", "—")
                    ll_table.add_row(
                        "Zero Drift (μ=0)",
                        f"{ll_result.ll_zero_drift:.2f}",
                        f"[green]{ll_result.delta_ll_vs_zero:+.2f}[/green]" if ll_result.delta_ll_vs_zero > 0 else f"[red]{ll_result.delta_ll_vs_zero:+.2f}[/red]"
                    )
                    ll_table.add_row(
                        "EWMA Drift",
                        f"{ll_result.ll_ewma_drift:.2f}",
                        f"[green]{ll_result.delta_ll_vs_ewma:+.2f}[/green]" if ll_result.delta_ll_vs_ewma > 0 else f"[red]{ll_result.delta_ll_vs_ewma:+.2f}[/red]"
                    )
                    ll_table.add_row(
                        "Constant Drift",
                        f"{ll_result.ll_constant_drift:.2f}",
                        f"[green]{ll_result.delta_ll_vs_constant:+.2f}[/green]" if ll_result.delta_ll_vs_constant > 0 else f"[red]{ll_result.delta_ll_vs_constant:+.2f}[/red]"
                    )
                    
                    console.print(ll_table)
                    console.print(f"[bold]Best Model:[/bold] {ll_result.best_model}")
                    console.print(f"[dim]{ll_result.diagnostic_message}[/dim]\n")
                    
                    # 3. PIT Calibration Check
                    console.print("[bold yellow]3. Probability Integral Transform (PIT) Calibration[/bold yellow]")
                    pit_result = validate_pit_calibration(
                        px, ret, mu_kf, var_kf, vol, asset_name=asset,
                        plot=args.validation_plots,
                        save_path=f"{plot_dir}/{asset}_pit_calibration.png" if plot_dir else None
                    )
                    
                    pit_table = Table(title="Forecast Calibration", show_header=True)
                    pit_table.add_column("Metric", style="cyan")
                    pit_table.add_column("Value", justify="right")
                    pit_table.add_column("Expected", justify="right")
                    pit_table.add_column("Status", justify="center")
                    
                    pit_table.add_row(
                        "Observations",
                        str(pit_result.n_observations),
                        "—",
                        ""
                    )
                    pit_table.add_row(
                        "KS Statistic",
                        f"{pit_result.ks_statistic:.4f}",
                        "—",
                        ""
                    )
                    pit_table.add_row(
                        "KS p-value",
                        f"{pit_result.ks_pvalue:.4f}",
                        "> 0.05",
                        "✅" if pit_result.ks_pvalue >= 0.05 else "⚠️"
                    )
                    pit_table.add_row(
                        "PIT Mean",
                        f"{pit_result.pit_mean:.4f}",
                        "0.5000",
                        "✅" if abs(pit_result.pit_mean - 0.5) < 0.05 else "⚠️"
                    )
                    pit_table.add_row(
                        "PIT Std Dev",
                        f"{pit_result.pit_std:.4f}",
                        f"{expected_std:.4f}",
                        "✅" if abs(pit_result.pit_std - expected_std) < 0.05 else "⚠️"
                    )
                    
                    console.print(pit_table)
                    console.print(f"[dim]{pit_result.diagnostic_message}[/dim]\n")
                    
                    # 4. Stress-Regime Behavior
                    console.print("[bold yellow]4. Stress-Regime Behavior Analysis[/bold yellow]")
                    stress_result = analyze_stress_regime_behavior(
                        px, ret, mu_kf, var_kf, vol, asset_name=asset
                    )
                    
                    stress_table = Table(title="Risk Intelligence", show_header=True)
                    stress_table.add_column("Metric", style="cyan")
                    stress_table.add_column("Normal", justify="right")
                    stress_table.add_column("Stress", justify="right")
                    stress_table.add_column("Ratio", justify="right")
                    
                    stress_table.add_row(
                        "Drift Uncertainty σ(μ̂)",
                        f"{stress_result.avg_uncertainty_normal:.6f}",
                        f"{stress_result.avg_uncertainty_stress:.6f}",
                        f"[green]{stress_result.uncertainty_spike_ratio:.2f}×[/green]" if stress_result.uncertainty_spike_ratio > 1.2 else f"{stress_result.uncertainty_spike_ratio:.2f}×"
                    )
                    stress_table.add_row(
                        "Kelly Half-Fraction",
                        f"{stress_result.avg_kelly_normal:.4f}",
                        f"{stress_result.avg_kelly_stress:.4f}",
                        f"[green]{stress_result.kelly_reduction_ratio:.2f}×[/green]" if stress_result.kelly_reduction_ratio < 0.9 else f"{stress_result.kelly_reduction_ratio:.2f}×"
                    )
                    
                    console.print(stress_table)
                    
                    if stress_result.stress_periods_detected:
                        console.print(f"\n[bold]Stress Periods Detected:[/bold] {len(stress_result.stress_periods_detected)}")
                        for i, (start, end) in enumerate(stress_result.stress_periods_detected[:5], 1):
                            console.print(f"  {i}. {start} → {end}")
                        if len(stress_result.stress_periods_detected) > 5:
                            console.print(f"  ... and {len(stress_result.stress_periods_detected) - 5} more")
                    
                    console.print(f"\n[dim]{stress_result.diagnostic_message}[/dim]\n")
                    
                    # Overall validation summary
                    all_passed = (
                        drift_result.validation_passed and
                        ll_result.improvement_significant and
                        pit_result.calibration_passed and
                        stress_result.system_backed_off
                    )
                    
                    if all_passed:
                        console.print(Panel.fit(
                            "[bold green]✅ ALL VALIDATION CHECKS PASSED[/bold green]\n"
                            "[dim]Model demonstrates structural realism and statistical rigor.[/dim]",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel.fit(
                            "[bold yellow]⚠️ SOME VALIDATION CHECKS FAILED[/bold yellow]\n"
                            "[dim]Review diagnostics above for tuning guidance.[/dim]",
                            border_style="yellow"
                        ))
                else:
                    console.print("[red]⚠️ Kalman filter data not available for validation[/red]")
                    
            except Exception as e:
                console.print(f"[red]⚠️ Validation failed: {e}[/red]")
                import traceback
                console.print(f"[dim]{traceback.format_exc()}[/dim]")

        # Build summary row for this asset
        asset_label = build_asset_display_label(asset, title)
        horizon_signals = {
            int(s.horizon_days): {"label": s.label, "profit_pln": float(s.profit_pln)}
            for s in sigs
        }
        nearest_label = sigs[0].label if sigs else "HOLD"
        summary_rows.append({
            "asset_label": asset_label,
            "horizon_signals": horizon_signals,
            "nearest_label": nearest_label,
            "sector": get_sector(canon),
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
            # stochastic volatility metadata (Level-7: full posterior uncertainty)
            "stochastic_volatility": {
                "enabled": True,
                "method": "bayesian_garch_sampling",
                "parameter_sampling": os.getenv("PARAM_UNC", "sample"),
                "uncertainty_propagated": True,
                "volatility_ci_tracked": True,
                "description": "Volatility treated as latent stochastic process with posterior uncertainty. GARCH parameters sampled from N(theta_hat, Cov) per path. Full h_t trajectories tracked and volatility credible intervals reported per horizon."
            },
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
        # Group summary rows by sector and render one table per sector
        render_sector_summary_tables(summary_rows, horizons)
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not print summary tables: {e}")

    # Build structured failure log for exports
    failure_log = [
        {
            "asset": asset,
            "display_name": info.get("display_name", asset),
            "attempts": info.get("attempts", 0),
            "last_error": info.get("last_error", "")
        }
        for asset, info in failures.items()
    ]

    # Print failure summary table (if any)
    if failures:
        from rich.table import Table
        fail_table = Table(title="Failed Assets After Retries")
        fail_table.add_column("Asset", style="red", justify="left")
        fail_table.add_column("Display Name", justify="left")
        fail_table.add_column("Attempts", justify="right")
        fail_table.add_column("Last Error", justify="left")
        for asset, info in failures.items():
            fail_table.add_row(asset, str(info.get("display_name", asset)), str(info.get("attempts", "")), str(info.get("last_error", "")))
        Console().print(fail_table)

    # Exports
    cache_path = args.cache_json or DEFAULT_CACHE_PATH
    payload = {
        "assets": all_blocks,
        "summary_rows": summary_rows,
        "horizons": horizons,
        "column_descriptions": DETAILED_COLUMN_DESCRIPTIONS,
        "simple_column_descriptions": SIMPLIFIED_COLUMN_DESCRIPTIONS,
        "failed_assets": failure_log,
    }
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not write cache JSON: {e}")
    
    if args.json:
        try:
            with open(args.json, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            Console().print(f"[yellow]Warning:[/yellow] Could not write JSON export: {e}")


if __name__ == "__main__":
    main()
