from __future__ import annotations
"""
Data fetching: price retrieval with PLN conversion, GARCH(1,1) MLE,
and Student-t nu estimation.

Extracted from signals.py (Story 5.4). Contains fetch_px_asset for
multi-asset price data with automatic currency conversion, _garch11_mle
for GARCH(1,1) conditional variance estimation, and _fit_student_nu_mle
for Student-t tail parameter fitting.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

# -- path setup so "from ingestion..." works when run standalone ----------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ingestion.data_utils import (
    _fetch_px_symbol,
    _fetch_with_fallback,
    _ensure_float_series,
    _align_fx_asof,
    _resolve_display_name,
    _resolve_symbol_candidates,
    convert_currency_to_pln,
    convert_price_series_to_pln,
    detect_quote_currency,
    fetch_usd_to_pln_exchange_rate,
)


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
        d2_nll = (nll_plus - 2.0 * nll_0 + nll_minus) / (eps ** 2);
        
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

