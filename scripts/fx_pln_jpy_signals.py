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
from rich.table import Table
import logging
import os

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


# -------------------------
# Utils
# -------------------------

def norm_cdf(x: float) -> float:
    # Numerically stable normal CDF via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _to_float(x) -> float:
    try:
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        if hasattr(x, "item"):
            return float(x.item())
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(()).item())
        return float("nan")
    except Exception:
        return float("nan")


def safe_last(s: pd.Series) -> float:
    try:
        return _to_float(s.iloc[-1])
    except Exception:
        return float("nan")


def winsorize(x, p: float = 0.01):
    """Winsorize a Series or DataFrame column-wise using scalar thresholds.
    - Robust to pandas alignment quirks
    - Avoids deprecated float(Series) paths by using numpy percentiles
    - Gracefully handles empty/singleton inputs by returning them unchanged
    """
    if isinstance(x, pd.DataFrame):
        return x.apply(lambda s: winsorize(s, p))
    if isinstance(x, pd.Series):
        vals = x.to_numpy(dtype=float)
        if vals.size < 3:
            return x  # not enough data to estimate tails
        lo_hi = np.nanpercentile(vals, [100 * p, 100 * (1 - p)])
        lo = float(lo_hi[0])
        hi = float(lo_hi[1])
        clipped = np.clip(vals, lo, hi)
        return pd.Series(clipped, index=x.index, name=getattr(x, "name", None))
    # Fallback: treat as array-like
    arr = np.asarray(x, dtype=float)
    if arr.size < 3:
        return arr
    lo_hi = np.nanpercentile(arr, [100 * p, 100 * (1 - p)])
    lo = float(lo_hi[0])
    hi = float(lo_hi[1])
    return np.clip(arr, lo, hi)


# -------------------------
# Data
# -------------------------

def _download_prices(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Robust Yahoo fetch with multiple strategies.
    Returns a DataFrame with OHLC columns (if available).
    - Tries yf.download first
    - Falls back to Ticker.history
    - Tries again without auto_adjust
    Normalizes DatetimeIndex to tz-naive for stability.
    """
    # Try standard download
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, threads=False)
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
    except Exception:
        pass
    # Try Ticker.history
    try:
        tk = yf.Ticker(symbol)
        df = tk.history(start=start, end=end, auto_adjust=True)
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
    except Exception:
        pass
    # Try without auto_adjust
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
    except Exception:
        pass
    return pd.DataFrame()


# Display-name cache for full asset names (e.g., company longName)
_DISPLAY_NAME_CACHE: Dict[str, str] = {}

def _resolve_display_name(symbol: str) -> str:
    """Return a human-friendly display name for a Yahoo symbol.
    Tries yfinance fast_info/info longName/shortName; falls back to the symbol itself.
    Caches results for repeated use.
    """
    sym = (symbol or "").strip()
    if not sym:
        return symbol
    if sym in _DISPLAY_NAME_CACHE:
        return _DISPLAY_NAME_CACHE[sym]
    name: Optional[str] = None
    try:
        tk = yf.Ticker(sym)
        # Try info first (often has longName)
        try:
            info = tk.info or {}
            name = info.get("longName") or info.get("shortName") or info.get("name")
        except Exception:
            name = None
        if not name:
            # Try fast_info (may have shortName on some tickers)
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    name = fi.get("shortName") or fi.get("longName")
                else:
                    name = getattr(fi, "shortName", None) or getattr(fi, "longName", None)
    except Exception:
        name = None
    disp = str(name).strip() if name else sym
    _DISPLAY_NAME_CACHE[sym] = disp
    return disp


def _fetch_px_symbol(symbol: str, start: Optional[str], end: Optional[str]) -> pd.Series:
    data = _download_prices(symbol, start, end)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {symbol}")
    for col in ("Close", "Adj Close"):
        if isinstance(data, pd.DataFrame) and col in data.columns:
            px = data[col].dropna()
            px.name = "px"
            return px
    if isinstance(data, pd.Series):
        px = data.dropna()
        px.name = "px"
        return px
    raise RuntimeError(f"No price column found for {symbol}")


def _fetch_with_fallback(symbols: List[str], start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    last_err: Optional[Exception] = None
    for sym in symbols:
        try:
            px = _fetch_px_symbol(sym, start, end)
            return px, sym
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"No data for symbols: {symbols}")


def fetch_px(start: Optional[str], end: Optional[str]) -> pd.Series:
    # Backward-compatible: default PAIR (PLNJPY=X)
    return _fetch_px_symbol(PAIR, start, end)


def _fetch_usdpln(start: Optional[str], end: Optional[str]) -> pd.Series:
    """Fetch USD/PLN as a Series. Tries multiple routes:
    1) USDPLN=X directly
    2) Invert PLNUSD=X
    3) Cross via EUR: USDPLN = EURPLN / EURUSD
    """
    # 1) Direct USDPLN
    try:
        s, used = _fetch_with_fallback(["USDPLN=X"], start, end)
        return s
    except Exception:
        pass
    # 2) Invert PLNUSD
    try:
        s, used = _fetch_with_fallback(["PLNUSD=X"], start, end)
        inv = (1.0 / s)
        inv.name = "px"
        return inv
    except Exception:
        pass
    # 3) Cross via EUR
    try:
        eurpln, _ = _fetch_with_fallback(["EURPLN=X"], start, end)
        eurusd, _ = _fetch_with_fallback(["EURUSD=X"], start, end)
        df = pd.concat([eurpln, eurusd], axis=1, join="inner").dropna()
        df.columns = ["eurpln", "eurusd"]
        cross = (df["eurpln"] / df["eurusd"]).rename("px")
        return cross
    except Exception as e:
        raise RuntimeError(f"Unable to get USDPLN via direct, inverse, or EUR cross: {e}")


def _detect_quote_currency(symbol: str) -> str:
    """Try to detect the quote currency for a Yahoo symbol.
    Returns uppercase ISO code like 'USD','EUR','GBP','GBp','JPY', or '' if unknown.
    """
    try:
        tk = yf.Ticker(symbol)
        # fast_info first
        cur = None
        try:
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                cur = fi.get("currency") if isinstance(fi, dict) else getattr(fi, "currency", None)
        except Exception:
            cur = None
        if not cur:
            info = tk.info or {}
            cur = info.get("currency")
        if cur:
            return str(cur).upper()
    except Exception:
        pass
    # Heuristics from suffix
    s = symbol.upper()
    if s.endswith(".DE") or s.endswith(".F") or s.endswith(".BE") or s.endswith(".XETRA"):
        return "EUR"
    if s.endswith(".PA"):
        return "EUR"
    if s.endswith(".L") or s.endswith(".LON"):
        # London often in GBX (pence)
        return "GBX"
    if s.endswith(".VI"):
        return "EUR"
    if s.endswith(".CO"):
        return "DKK"
    if s.endswith(".TO") or s.endswith(".TSX"):
        return "CAD"
    if s.endswith(".SZ") or s.endswith(".SS"):
        return "CNY"
    if s.endswith(".KS") or s.endswith(".KQ"):
        return "KRW"
    # Default to USD
    return "USD"


def _as_series(x) -> pd.Series:
    """Coerce input to a 1-D pandas Series if possible; otherwise return empty Series."""
    if isinstance(x, pd.Series):
        # Squeeze potential 2D values inside the Series
        vals = np.asarray(x.values)
        if vals.ndim == 2 and vals.shape[1] == 1:
            return pd.Series(vals.ravel(), index=x.index, name=getattr(x, "name", None))
        return x
    if isinstance(x, pd.DataFrame):
        # If single column, squeeze; else take the first column
        if x.shape[1] >= 1:
            s = x.iloc[:, 0]
            s.name = getattr(s, "name", x.columns[0])
            return _as_series(s)
        return pd.Series(dtype=float)
    try:
        arr = np.asarray(x)
        if arr.ndim == 0:
            return pd.Series([_to_float(arr)])
        if arr.ndim == 1:
            return pd.Series(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            return pd.Series(arr.ravel())
    except Exception:
        pass
    return pd.Series(dtype=float)


def _ensure_float_series(s: pd.Series) -> pd.Series:
    """Ensure a 1-D float Series free of nested arrays/objects.
    - Coerces to Series via _as_series
    - Converts to numeric dtype; non-convertible entries become NaN
    """
    s = _as_series(s)
    if s.empty:
        return s
    # Try fast astype to float
    try:
        s = s.astype(float)
        return s
    except Exception:
        pass
    # Fallback: to_numeric coercion
    try:
        s = pd.to_numeric(s, errors="coerce")
    except Exception:
        # Last resort: build from numpy values squeezed to 1-D
        vals = np.asarray(s.values)
        if vals.ndim > 1:
            vals = vals.ravel()
        s = pd.Series(vals, index=s.index)
        s = pd.to_numeric(s, errors="coerce")
    return s


def _align_fx_asof(native_px: pd.Series, fx_px: pd.Series, max_gap_days: int = 7) -> pd.Series:
    """Align FX series to native dates using asof within a tolerance window.
    Falls back to forward direction if backward match is missing.
    Returns aligned FX indexed by native dates (NaN where no match within tolerance)."""
    # Ensure inputs are Series
    native_px = _as_series(native_px)
    fx_px = _as_series(fx_px)
    if native_px.empty:
        return pd.Series(index=native_px.index, dtype=float)
    if fx_px.empty:
        return pd.Series(index=native_px.index, dtype=float)
    # Build merge frames
    left = native_px.rename("native").to_frame().reset_index()
    left = left.rename(columns={left.columns[0]: "date"})
    right = fx_px.rename("fx").to_frame().reset_index()
    right = right.rename(columns={right.columns[0]: "date"})
    # Sort and asof merge
    left = left.sort_values("date")
    right = right.sort_values("date")
    tol = pd.Timedelta(days=max_gap_days)
    back = pd.merge_asof(left, right, on="date", direction="backward", tolerance=tol)
    fwd = pd.merge_asof(left, right, on="date", direction="forward", tolerance=tol)
    fx_aligned = back["fx"].fillna(fwd["fx"])  # prefer backward, then forward
    fx_aligned.index = pd.to_datetime(left["date"])  # align index to native dates
    return fx_aligned


def _fx_leg_to_pln(quote_ccy: str, start: Optional[str], end: Optional[str], native_index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
    """Return a Series of FX rate in PLN per 1 unit of quote_ccy.
    Expands the fetch window to cover the native price index +/- 30 days for overlap robustness."""
    q = (quote_ccy or "").upper().strip()
    # Expand window around native index
    s_ext, e_ext = start, end
    if native_index is not None and len(native_index) > 0:
        try:
            s_ext = (pd.to_datetime(native_index.min()) - pd.Timedelta(days=30)).date().isoformat()
            e_ext = (pd.to_datetime(native_index.max()) + pd.Timedelta(days=5)).date().isoformat()
        except Exception:
            pass

    def fetch(sym_list: List[str]) -> pd.Series:
        s, _ = _fetch_with_fallback(sym_list, s_ext, e_ext)
        return s

    if q in ("PLN", "PLN "):
        # Return a flat-1 series over the native index for easy alignment
        return pd.Series(1.0, index=pd.DatetimeIndex(native_index) if native_index is not None else [pd.Timestamp("1970-01-01")])
    if q == "USD":
        return _fetch_usdpln(s_ext, e_ext)
    if q == "EUR":
        try:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            return eurpln
        except Exception:
            # EURPLN via USD: EURPLN = EURUSD * USDPLN
            eurusd, _ = _fetch_with_fallback(["EURUSD=X"], s_ext, e_ext)
            return eurusd * _fetch_usdpln(s_ext, e_ext)
    if q in ("GBP", "GBX", "GBPp", "GBP P", "GBp"):
        gbppln, _ = _fetch_with_fallback(["GBPPLN=X"], s_ext, e_ext)
        return gbppln * (0.01 if q in ("GBX", "GBPp", "GBP P", "GBp") else 1.0)
    if q == "JPY":
        try:
            plnjpy, _ = _fetch_with_fallback(["PLNJPY=X"], s_ext, e_ext)
            return 1.0 / plnjpy
        except Exception:
            jpypln, _ = _fetch_with_fallback(["JPYPLN=X"], s_ext, e_ext)
            return jpypln
    if q == "CAD":
        try:
            cadpln, _ = _fetch_with_fallback(["CADPLN=X"], s_ext, e_ext)
            return cadpln
        except Exception:
            # Try CADUSD cross
            try:
                usdcad, _ = _fetch_with_fallback(["USDCAD=X"], s_ext, e_ext)
                cadusd = 1.0 / usdcad
            except Exception:
                cadusd, _ = _fetch_with_fallback(["CADUSD=X"], s_ext, e_ext)
            return cadusd * _fetch_usdpln(s_ext, e_ext)
    if q == "CHF":
        try:
            chfpln, _ = _fetch_with_fallback(["CHFPLN=X"], s_ext, e_ext)
            return chfpln
        except Exception:
            try:
                usdchf, _ = _fetch_with_fallback(["USDCHF=X"], s_ext, e_ext)
                chfusd = 1.0 / usdchf
            except Exception:
                chfusd, _ = _fetch_with_fallback(["CHFUSD=X"], s_ext, e_ext)
            return chfusd * _fetch_usdpln(s_ext, e_ext)
    if q == "AUD":
        try:
            audpln, _ = _fetch_with_fallback(["AUDPLN=X"], s_ext, e_ext)
            return audpln
        except Exception:
            audusd, _ = _fetch_with_fallback(["AUDUSD=X"], s_ext, e_ext)
            return audusd * _fetch_usdpln(s_ext, e_ext)
    if q == "SEK":
        try:
            sekpln, _ = _fetch_with_fallback(["SEKPLN=X"], s_ext, e_ext)
            return sekpln
        except Exception:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            eurseK, _ = _fetch_with_fallback(["EURSEK=X"], s_ext, e_ext)
            return eurpln / eurseK
    if q == "NOK":
        try:
            nokpln, _ = _fetch_with_fallback(["NOKPLN=X"], s_ext, e_ext)
            return nokpln
        except Exception:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            eurnok, _ = _fetch_with_fallback(["EURNOK=X"], s_ext, e_ext)
            return eurpln / eurnok
    if q == "DKK":
        try:
            dkkpln, _ = _fetch_with_fallback(["DKKPLN=X"], s_ext, e_ext)
            return dkkpln
        except Exception:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            eurdkk, _ = _fetch_with_fallback(["EURDKK=X"], s_ext, e_ext)
            return eurpln / eurdkk
    if q == "HKD":
        try:
            hkdpln, _ = _fetch_with_fallback(["HKDPLN=X"], s_ext, e_ext)
            return hkdpln
        except Exception:
            usdpln = _fetch_usdpln(s_ext, e_ext)
            usdhkd, _ = _fetch_with_fallback(["USDHKD=X"], s_ext, e_ext)
            return usdpln / usdhkd
    if q == "KRW":
        # PLN per KRW = (PLN per USD) / (KRW per USD)
        try:
            usdpln = _fetch_usdpln(s_ext, e_ext)
            usdkrw, _ = _fetch_with_fallback(["USDKRW=X"], s_ext, e_ext)
            # Align by native index if available
            if native_index is not None and len(native_index) > 0:
                usdpln = usdpln.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                usdkrw = usdkrw.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
            return (usdpln / usdkrw).rename("px")
        except Exception:
            # Fallback: try KRWUSD and invert
            try:
                krwusd, _ = _fetch_with_fallback(["KRWUSD=X"], s_ext, e_ext)
                usdpln = _fetch_usdpln(s_ext, e_ext)
                if native_index is not None and len(native_index) > 0:
                    usdpln = usdpln.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                    krwusd = krwusd.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                return (usdpln * krwusd).rename("px")
            except Exception:
                pass
        # As last resort, assume USD (may be wrong for KRW assets but avoids crash)
        return _fetch_usdpln(s_ext, e_ext)
    # Default: assume USD
    return _fetch_usdpln(s_ext, e_ext)


def _convert_to_pln(native_px: pd.Series, quote_ccy: str, start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    """Convert a native price series quoted in quote_ccy into PLN.
    Returns (pln_series, units_suffix).
    """
    sfx = "(PLN)"
    native_px = _ensure_float_series(native_px)
    # Get FX leg over the native range (with padding)
    fx = _fx_leg_to_pln(quote_ccy, start, end, native_index=native_px.index)
    # Try increasingly permissive alignments
    fx_al = _align_fx_asof(native_px, fx, max_gap_days=7)
    if fx_al.isna().all():
        fx_al = _align_fx_asof(native_px, fx, max_gap_days=14)
    if fx_al.isna().all():
        fx_al = _align_fx_asof(native_px, fx, max_gap_days=30)
    # Fallback: strict calendar alignment with ffill/bfill
    if fx_al.isna().all():
        fx_al = fx.reindex(native_px.index).ffill().bfill()
    fx_al = _ensure_float_series(fx_al)
    pln = (native_px * fx_al).dropna()
    pln.name = "px"
    return pln, sfx


def _resolve_symbol_candidates(asset: str) -> List[str]:
    a = asset.strip()
    u = a.upper()
    mapping = {
        # Prefer active, liquid proxies first to avoid Yahoo "possibly delisted" noise
        "GOOO": ["GOOG", "GOOGL", "GOOO"],
        "GLDW": ["GLDM", "GLD", "GLDW"],
        "SGLP": ["SGLP.L", "SGLP", "SGLP.LON"],
        "GLDE": ["GLD", "IAU", "GLDE"],
        "FACC": ["FACC.VI", "FACC"],
        "SLVI": ["SLV", "SLVP", "SLVI"],
        "TKA": ["TKA.DE", "TKA"],
        # Netflix and Novo Nordisk
        "NFLX": ["NFLX"],
        "NOVO": ["NVO", "NOVO-B.CO", "NOVOB.CO", "NOVO-B.CO"],
        "NOVOB": ["NVO", "NOVOB.CO", "NOVO-B.CO"],
        "NVO": ["NVO", "NOVO-B.CO", "NOVOB.CO"],
        # Kratos (alias to KTOS)
        "KRATOS": ["KTOS"],
        # Requested blue chips and defense/aero additions
        "RHEINMETALL": ["RHM.DE", "RHM.F"],
        "RHEINMETAL": ["RHM.DE", "RHM.F"],
        "AIRBUS": ["AIR.PA", "AIR.DE"],
        "RENK": ["R3NK.DE", "RNK.DE"],
        "NORTHROP": ["NOC"],
        "NORTHROP GRUMMAN": ["NOC"],
        "NORTHRUP": ["NOC"],
        "NORTHRUP GRUNMAN": ["NOC"],
        "NVIDIA": ["NVDA"],
        "MICROSOFT": ["MSFT"],
        "APPLE": ["AAPL"],
        "AMD": ["AMD"],
        "UBER": ["UBER"],
        "TESLA": ["TSLA"],
        "VANGUARD SP 500": ["VOO", "VUSA.L"],
        "VANGARD SP 500": ["VOO", "VUSA.L"],
        "VANGUARD S&P 500": ["VOO", "VUSA.L"],
        "THALES": ["HO.PA"],
        "SAMSUNG": ["005930.KS", "005935.KS"],
        "SAMSUNG ELECTRONICS": ["005930.KS", "005935.KS"],
        "TKMS": ["TKA.DE", "TKAMY"],
        "TKMS AG & CO": ["TKA.DE", "TKAMY"],
        # keep identity candidates (with improved MTX mapping to MTU Aero Engines on XETRA)
        "RKLB": ["RKLB"],
        "MTX": ["MTX.DE", "MTX"],
        "IBKR": ["IBKR"],
        "HOOD": ["HOOD"],
    }
    # For known special assets already handled elsewhere, leave as-is
    special = {"PLNJPY=X", "BTC-USD", "BTCUSD=X", "MSTR", "GC=F", "SI=F", "XAU=X", "XAG=X", "XAUUSD=X", "XAGUSD=X"}
    if u in special:
        return [u]
    if u in mapping:
        return mapping[u]
    return [u]


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
        usdpln_px = _fx_leg_to_pln("USD", start, end, native_index=btc_px.index)
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
        usdpln_px = _fx_leg_to_pln("USD", start, end, native_index=mstr_px.index)
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
        usdpln_px = _fetch_usdpln(start, end)
        usdpln_aligned = usdpln_px.reindex(metal_px.index).ffill()
        df = pd.concat([metal_px, usdpln_aligned], axis=1).dropna()
        df.columns = ["metal_usd", "usdpln"]
        px_pln = (df["metal_usd"] * df["usdpln"]).rename("px")
        title = f"{metal_name} in PLN (PLN per troy oz)"
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
    qcy = _detect_quote_currency(used_sym)
    px_pln, _ = _convert_to_pln(px_native, qcy, start, end)
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

    params: Dict[str, float] = {"omega": omega, "alpha": alpha, "beta": beta, "converged": True}
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

    # Trend filter (200D z-distance)
    sma200 = px.rolling(200).mean()
    trend_z = (px - sma200) / px.rolling(200).std()

    # Regime-dependent priors on drift (Bayesian-style shrinkage)
    # Regime thresholds
    vol_shock_thr = 2.5
    vol_calm_thr = 0.85
    trend_thr = 0.7  # |trend_z|>=0.7 considered trending
    prior_scale = 0.25  # prior magnitude scaled to daily vol in trends
    cap_coeff = 0.75    # cap |mu_post| <= cap_coeff * vol

    # Compute regime-dependent weights and prior using numpy for robust alignment
    idx = mu_blend.index
    vr = np.asarray(vol_regime.reindex(idx), dtype=float).ravel()
    tz = np.asarray(trend_z.reindex(idx), dtype=float).ravel()
    vol_arr = np.asarray(vol.reindex(idx), dtype=float).ravel()
    mu_bl_arr = np.asarray(mu_blend.reindex(idx), dtype=float).ravel()

    w_arr = np.full(len(idx), 0.7, dtype=float)
    with np.errstate(invalid='ignore'):
        w_arr[vr <= vol_calm_thr] = 0.5
        w_arr[vr >= vol_shock_thr] = 0.2
    tr_mask_arr = np.isfinite(tz) & (np.abs(tz) >= trend_thr)
    w_arr[tr_mask_arr] = np.maximum(w_arr[tr_mask_arr], 0.85)

    prior_mu_arr = np.zeros(len(idx), dtype=float)
    prior_mu_arr[tr_mask_arr] = np.sign(tz[tr_mask_arr]) * prior_scale * vol_arr[tr_mask_arr]

    mu_post_arr = w_arr * mu_bl_arr + (1.0 - w_arr) * prior_mu_arr
    cap_arr = cap_coeff * vol_arr
    mu_post_arr = np.clip(mu_post_arr, -cap_arr, cap_arr)

    mu_post = pd.Series(mu_post_arr, index=idx)
    # Robust fallback for NaNs
    mu_blend_ser = _ensure_float_series(mu_blend)
    if isinstance(mu_blend_ser, pd.DataFrame):
        mu_blend_ser = mu_blend_ser.iloc[:, 0]
    mu_post = mu_post.combine_first(mu_blend_ser)
    mu_post = mu_post.fillna(0.0)

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

    # Regime label for diagnostics (based on 'now')
    def regime_label(vr: float, tz: float) -> str:
        if np.isfinite(vr) and vr > 1.8:
            if np.isfinite(tz) and tz > 0:
                return "High‑vol uptrend"
            elif np.isfinite(tz) and tz < 0:
                return "High‑vol downtrend"
            return "High volatility"
        if np.isfinite(vr) and vr < 0.85:
            if np.isfinite(tz) and tz > 0:
                return "Calm uptrend"
            elif np.isfinite(tz) and tz < 0:
                return "Calm downtrend"
            return "Calm / range"
        if np.isfinite(tz) and abs(tz) > 0.5:
            return "Trending"
        return "Normal"

    reg = regime_label(vol_reg_now, trend_now)

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

    # Adaptive smoothing alpha from regime (use 'now')
    if np.isfinite(vol_reg_now) and vol_reg_now > 1.5:
        alpha_edge = 0.55
    elif np.isfinite(vol_reg_now) and vol_reg_now < 0.8:
        alpha_edge = 0.30
    else:
        alpha_edge = 0.40
    alpha_p = min(0.75, alpha_edge + 0.10)

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
        # Uncertainty components
        if np.isfinite(vol_reg_now):
            u_vol = float(np.clip(vol_reg_now - 1.0, 0.0, 1.5) / 1.5)
        else:
            u_vol = 0.0
        if np.isfinite(nu_glob):
            nu_eff = float(np.clip(nu_glob, 3.0, 200.0))
            inv_nu = 1.0 / nu_eff
            inv_min, inv_max = 1.0 / 200.0, 1.0 / 3.0
            u_tail = float(np.clip((inv_nu - inv_min) / (inv_max - inv_min), 0.0, 1.0))
        else:
            u_tail = 0.0
        med_sig_H = (med_vol_last * math.sqrt(H)) if (np.isfinite(med_vol_last) and med_vol_last > 0) else sig_H
        ratio = float(sig_H / med_sig_H) if med_sig_H > 0 else 1.0
        u_sig = float(np.clip(ratio - 1.0, 0.0, 1.0))
        # With GARCH modeling volatility, weigh uncertainty mostly by tails and forecast sigma
        U = float(np.clip(0.2 * u_vol + 0.4 * u_tail + 0.4 * u_sig, 0.0, 1.0))
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


# -------------------------
# Output
# -------------------------

def print_table(asset: str, title: str, sigs: List[Signal], px: pd.Series, ci_level: float, used_t_map: bool, show_caption: bool = True) -> None:
    console = Console()
    last_close = _to_float(px.iloc[-1])

    table = Table(title=f"{asset} — {title} — last close {last_close:.4f} on {px.index[-1].date()}")
    # Clearer column headers
    table.add_column("Horizon (trading days)", justify="right")
    table.add_column("Edge z (risk-adjusted)", justify="right")
    table.add_column("Pr[return>0]", justify="right")
    table.add_column("E[log return]", justify="right")
    table.add_column(f"CI ±{int(ci_level*100)}% (log)", justify="right")
    table.add_column("Position strength (0–1)", justify="right")
    table.add_column("Regime", justify="left")
    table.add_column("Profit on 1,000,000 PLN (PLN)", justify="right")
    table.add_column("Signal", justify="center")
    # Add a concise caption describing columns (once, if requested)
    if show_caption:
        cdf_name = "Student-t" if used_t_map else "Normal"
        table.caption = (
            "Edge z = (expected log return / realized vol) scaled to horizon; "
            f"Pr[return>0] mapped from Edge z via {cdf_name} CDF using a single globally fitted Student‑t tail (ν) per asset; "
            f"E[log return] sums daily drift; CI is two-sided {int(ci_level*100)}% band for log return (log domain). "
            "Volatility is modeled via GARCH(1,1) (MLE) with EWMA fallback if unavailable. "
            "Position strength uses a half‑Kelly sizing heuristic (0–1) for real‑world robustness. "
            f"Profit assumes investing 1,000,000 PLN; profit CI is exp-mapped from the log-return CI into PLN. "
            "A minimum edge floor is enforced to reduce churn: if |edge| < EDGE_FLOOR, action is HOLD. BUY = long PLN vs JPY."
            )

    for s in sigs:
        table.add_row(
            str(s.horizon_days),
            f"{s.score:+.2f}",
            f"{100*s.p_up:5.1f}%",
            f"{s.exp_ret:+.4f}",
            f"[{s.ci_low:+.4f}, {s.ci_high:+.4f}]",
            f"{s.position_strength:.2f}",
            s.regime,
            f"{s.profit_pln:,.0f} [ {s.profit_ci_low_pln:,.0f} .. {s.profit_ci_high_pln:,.0f} ]",
            s.label,
        )

    console.print(table)


def timeframe_label(H: int) -> str:
    mapping = {1: "1 day", 2: "2 days", 3: "3 days", 5: "1 week (5d)", 7: "1 week", 10: "2 weeks (10d)", 14: "2 weeks", 21: "1 month", 42: "2 months", 63: "3 months", 84: "4 months", 105: "5 months", 126: "6 months", 189: "9 months", 252: "12 months"}
    return mapping.get(H, f"{H} days")


def _summary_display_label(asset: str, title: str) -> str:
    """Build a concise display like "Ticker (Name)" when available.
    - If title already contains "(SYMBOL)" in its first part, use that part as-is.
    - Else combine the provided asset ticker with the name part from title.
    """
    try:
        name_part = title.split(" — ")[0].strip()
    except Exception:
        name_part = title.strip()
    # If name_part already contains parentheses (e.g., "Company Name (TICKER)"), use it
    if "(" in name_part and ")" in name_part:
        return name_part
    # Otherwise prepend the asset symbol for clarity
    if asset:
        return f"{asset} — {name_part}"
    return name_part


def _format_profit_cell(label: str, profit_pln: float) -> str:
    """Return a rich-formatted cell like "BUY (+12,345)" with color by profit sign.
    Strong labels are bolded. Profit shown in PLN without currency code to keep it compact.
    """
    profit_txt = f"{profit_pln:+,.0f}"
    # Color by profit sign
    if np.isfinite(profit_pln) and profit_pln > 0:
        profit_txt = f"[green]{profit_txt}[/green]"
    elif np.isfinite(profit_pln) and profit_pln < 0:
        profit_txt = f"[red]{profit_txt}[/red]"
    # Emphasize STRONG labels
    lab = label
    if isinstance(label, str) and label.upper().startswith("STRONG "):
        lab = f"[bold]{label}[/bold]"
    return f"{lab} ({profit_txt})"


def _extract_symbol_from_title(title: str) -> str:
    """Extract the canonical symbol from a title like "Company Name (TICKER) — ...".
    Returns empty string if not found.
    Uses the last pair of parentheses to be robust to names containing parentheses.
    """
    try:
        s = str(title)
        # Find last '(' and next ')'
        l = s.rfind('(')
        r = s.find(')', l + 1) if l != -1 else -1
        if l != -1 and r != -1 and r > l + 1:
            token = s[l+1:r].strip()
            # Very basic sanitation
            return token.upper()
    except Exception:
        pass
    return ""


def print_summary_table(summary_rows: List[Dict], horizons: List[int]) -> None:
    """Print a compact summary table across assets.
    Columns: "Ticker (name)", then one column per trading-day horizon (e.g., 1d,3d,7d,...).
    Each cell: "Signal (±Profit)" for a 1,000,000 PLN notional.
    Sorted so that assets whose nearest horizon is SELL come first, then HOLD, then BUY, then STRONG BUY.
    """
    if not summary_rows:
        return
    console = Console()
    table = Table(title="Summary across assets — signals and profit on 1,000,000 PLN")
    table.add_column("Ticker (name)", justify="left", no_wrap=True)
    # Ensure stable order of horizons
    horizons_sorted = list(sorted(horizons))

    # Helper: normalize label to category and priority
    def _label_category_priority(label: str) -> Tuple[str, int]:
        if not isinstance(label, str):
            return ("HOLD", 1)
        u = label.upper().strip()
        # Map STRONG SELL -> SELL bucket
        if "SELL" in u:
            # SELL bucket priority 0 (comes first)
            return ("SELL", 0)
        if u.startswith("STRONG BUY"):
            return ("STRONG BUY", 3)
        if "BUY" in u:
            return ("BUY", 2)
        # Default HOLD bucket
        return ("HOLD", 1)

    # Compute sort keys per asset: (bucket_priority, nearest_horizon_days, display_name)
    def _sort_key(row: Dict) -> Tuple[int, int, str]:
        per_h = row.get("by_horizon", {})
        cat_pri = ("HOLD", 1)
        nearest_H = 10**9
        for H in horizons_sorted:
            if H in per_h:
                label, _profit = per_h[H]
                cat, pri = _label_category_priority(label)
                cat_pri = (cat, pri)
                nearest_H = H
                break
        return (cat_pri[1], nearest_H, str(row.get("display", "")))

    # Sort rows according to the specified order
    summary_rows_sorted = sorted(summary_rows, key=_sort_key)

    for H in horizons_sorted:
        table.add_column(f"{H}d", justify="center")
    for row in summary_rows_sorted:
        cells = [row.get("display", "")] 
        per_h = row.get("by_horizon", {})
        for H in horizons_sorted:
            cell = per_h.get(H)
            if cell is None:
                cells.append("")
            else:
                label, profit = cell
                cells.append(_format_profit_cell(label, profit))
        table.add_row(*cells)
    table.caption = "Profit figures are expected PLN P/L on a 1,000,000 PLN notional for each horizon; signals are model-based."
    console.print(table)


def simple_context(feats: Dict[str, pd.Series]) -> Tuple[str, str, str]:
    trend = safe_last(feats["trend_z"])  # z-distance to 200D SMA
    moms = [safe_last(feats["mom21"]), safe_last(feats["mom63"]), safe_last(feats["mom126"]), safe_last(feats["mom252"])]
    vol_reg = safe_last(feats["vol_regime"])  # relative to 1y median

    # Trend
    if np.isfinite(trend) and trend > 0.5:
        trend_s = "Uptrend"
    elif np.isfinite(trend) and trend < -0.5:
        trend_s = "Downtrend"
    else:
        trend_s = "Sideways"

    # Momentum agreement
    valid_m = [m for m in moms if np.isfinite(m)]
    if valid_m:
        pos = sum(1 for m in valid_m if m > 0)
        if pos >= 3:
            mom_s = "Momentum mostly positive"
        elif pos <= 1:
            mom_s = "Momentum mostly negative"
        else:
            mom_s = "Momentum mixed"
    else:
        mom_s = "Momentum unclear"

    # Volatility regime
    if np.isfinite(vol_reg) and vol_reg > 1.5:
        vol_s = "High volatility (signals weaker)"
    elif np.isfinite(vol_reg) and vol_reg < 0.8:
        vol_s = "Calm volatility (signals stronger)"
    else:
        vol_s = "Normal volatility"

    return trend_s, mom_s, vol_s


def explanation_for_signal(edge: float, p_up: float, trend_s: str, mom_s: str, vol_s: str) -> str:
    # Strength descriptor
    strength = "slight"
    if abs(edge) >= 1.5:
        strength = "strong"
    elif abs(edge) >= 0.7:
        strength = "moderate"
    direction = "rise" if edge >= 0 else "fall"
    return f"{trend_s}. {mom_s}. {vol_s}. About {p_up*100:.0f}% chance of a {direction}. Confidence {strength}."


def print_simple(asset: str, title: str, sigs: List[Signal], px: pd.Series, feats: Dict[str, pd.Series]) -> List[str]:
    console = Console()
    last_close = _to_float(px.iloc[-1])
    trend_s, mom_s, vol_s = simple_context(feats)

    table = Table(title=f"{asset} — {title} — Last price {last_close:.4f} on {px.index[-1].date()}")
    table.add_column("Timeframe", justify="left")
    table.add_column("Chance it goes up", justify="right")
    table.add_column("Recommendation", justify="center")
    table.add_column("Why (plain English)", justify="left")
    table.caption = "Simple view: chance is based on our model today. BUY means we expect the price to rise over the timeframe."

    explanations: List[str] = []
    for s in sigs:
        expl = explanation_for_signal(s.score, s.p_up, trend_s, mom_s, vol_s)
        explanations.append(expl)
        table.add_row(
            timeframe_label(s.horizon_days),
            f"{s.p_up*100:5.1f}%",
            s.label,
            expl,
        )

    console.print(table)
    return explanations


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate signals across multiple horizons for PLN/JPY, Gold (PLN), Silver (PLN), Bitcoin (PLN), and MicroStrategy (PLN).")
    p.add_argument("--start", type=str, default="2005-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--horizons", type=str, default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--assets", type=str, default="PLNJPY=X,GC=F,SI=F,BTC-USD,MSTR,NFLX,NVO,KTOS,RKLB,GOOO,GLDW,SGLP,GLDE,FACC,SLVI,MTX,IBKR,HOOD,RHEINMETALL,AMD,APPLE,AIRBUS,RENK,NORTHROP GRUMMAN,NVIDIA,TKMS AG & CO,MICROSOFT,UBER,VANGUARD S&P 500,THALES,SAMSUNG ELECTRONICS,TESLA", help="Comma-separated Yahoo symbols or friendly names. Metals, FX and USD/EUR/GBP/JPY/CAD/DKK/KRW assets are converted to PLN.")
    p.add_argument("--json", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--simple", action="store_true", help="Print an easy-to-read summary with simple explanations.")
    p.add_argument("--t_map", action="store_true", help="Use Student-t mapping based on realized kurtosis for probabilities (default on).")
    p.add_argument("--no_t_map", dest="t_map", action="store_false", help="Disable Student-t mapping; use Normal CDF.")
    p.add_argument("--ci", type=float, default=0.68, help="Two-sided confidence level for expected move bands (default 0.68 i.e., ~1-sigma).")
    # Caption controls for detailed view
    p.add_argument("--no_caption", action="store_true", help="Suppress the long column explanation caption in detailed tables.")
    p.add_argument("--force_caption", action="store_true", help="Force showing the caption for every detailed table.")
    p.set_defaults(t_map=True)
    return p.parse_args()


# Column descriptions for exports
COLUMN_DESCRIPTIONS = {
    "horizon_trading_days": "Number of trading days in the forecast horizon.",
    "edge_z_risk_adjusted": "Risk-adjusted edge (z-score) combining drift/vol with momentum/trend filters.",
    "prob_up": "Estimated probability the horizon return is positive (Student-t by default).",
    "expected_log_return": "Expected cumulative log return over the horizon from the daily drift estimate.",
    "ci_low_log": "Lower bound of the two-sided confidence interval for log return.",
    "ci_high_log": "Upper bound of the two-sided confidence interval for log return.",
    "profit_pln_on_1m_pln": "Expected profit in Polish zloty (PLN) when investing 1,000,000 PLN.",
    "profit_ci_low_pln": "Lower confidence bound for profit (PLN) on 1,000,000 PLN.",
    "profit_ci_high_pln": "Upper confidence bound for profit (PLN) on 1,000,000 PLN.",
    "signal": "Decision label based on prob_up: BUY (>=58%), HOLD (42–58%), SELL (<=42%).",
}

# Simple-mode column descriptions
SIMPLE_COLUMN_DESCRIPTIONS = {
    "timeframe": "Plain-English period (e.g., 1 day, 1 week, 3 months).",
    "chance_up": "Chance that the price goes up over this period (percent).",
    "recommendation": "BUY (expect price to rise), HOLD (unclear), or SELL (expect price to fall).",
    "why": "Short explanation combining trend, momentum, and volatility context.",
}


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
        canon = _extract_symbol_from_title(title)
        if not canon:
            canon = asset.strip().upper()
        if canon in processed_syms:
            Console().print(f"[yellow]Skipping duplicate:[/yellow] {title} (from input '{asset}')")
            continue
        processed_syms.add(canon)

        feats = compute_features(px)
        last_close = _to_float(px.iloc[-1])
        sigs, thresholds = latest_signals(feats, horizons, last_close=last_close, t_map=args.t_map, ci=args.ci)

        # Print table for this asset
        if args.simple:
            explanations = print_simple(asset, title, sigs, px, feats)
        else:
            # Determine caption policy for detailed view
            if args.force_caption:
                show_caption = True
            elif args.no_caption:
                show_caption = False
            else:
                show_caption = not caption_printed
            print_table(asset, title, sigs, px, ci_level=args.ci, used_t_map=args.t_map, show_caption=show_caption)
            caption_printed = caption_printed or show_caption
            explanations = []

        # Build summary row for this asset
        display = _summary_display_label(asset, title)
        by_h = {int(s.horizon_days): (s.label, float(s.profit_pln)) for s in sigs}
        summary_rows.append({
            "display": display,
            "by_horizon": by_h,
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
        all_blocks.append(block)

        # Prepare CSV rows
        if args.simple:
            for i, s in enumerate(sigs):
                csv_rows_simple.append({
                    "asset": title,
                    "symbol": asset,
                    "timeframe": timeframe_label(s.horizon_days),
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
        print_summary_table(summary_rows, horizons)
    except Exception as e:
        Console().print(f"[yellow]Warning:[/yellow] Could not print summary table: {e}")

    # Exports
    if args.json:
        payload = {
            "assets": all_blocks,
            "column_descriptions": COLUMN_DESCRIPTIONS,
            "simple_column_descriptions": SIMPLE_COLUMN_DESCRIPTIONS,
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
