"""
Data caching and lightweight data access utilities for the options screener/backtester.

This module centralizes:
- Local price history caching (CSV per ticker)
- Cached metadata: earnings dates, option expirations, and option chains (calls)
- Small filesystem helpers

All functions are defensive and safe to use in batch contexts. Network failures
will gracefully fall back to existing caches when possible.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

# Public defaults (overridable from CLI via options.py)
DEFAULT_DATA_DIR = os.environ.get("PRICE_DATA_DIR", "data")
DEFAULT_FORCE_REFRESH = False
REQUIRED_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

# TTLs for metadata caches
_DEF_EXP_TTL_HOURS = int(os.environ.get("EXPIRATIONS_TTL_HOURS", "12"))
_DEF_EARN_TTL_DAYS = int(os.environ.get("EARNINGS_TTL_DAYS", "3"))
_DEF_CHAIN_TTL_MIN = int(os.environ.get("OPTION_CHAIN_TTL_MIN", "60"))

# -------------------- FS helpers --------------------

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _meta_dir(cache_dir: Optional[str] = None) -> str:
    base = cache_dir or DEFAULT_DATA_DIR
    path = os.path.join(base, "meta")
    _ensure_dir(path)
    return path


def _options_dir(ticker: Optional[str] = None, cache_dir: Optional[str] = None) -> str:
    base = cache_dir or DEFAULT_DATA_DIR
    path = os.path.join(base, "options")
    if ticker:
        path = os.path.join(path, ticker.replace("/", "_"))
    _ensure_dir(path)
    return path


def _now_utc() -> datetime:
    return datetime.utcnow()


def _parse_iso(ts: Any) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(ts)
    except Exception:
        return None


def _is_fresh(ts_iso: Any, ttl_seconds: float) -> bool:
    try:
        ts = _parse_iso(ts_iso)
        if ts is None or pd.isna(ts):
            return False
        return (_now_utc() - ts).total_seconds() <= float(ttl_seconds)
    except Exception:
        return False


def _read_meta_json(meta_file: str) -> Dict[str, Any]:
    try:
        import json
        with open(meta_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _write_meta_json(meta_file: str, data: Dict[str, Any]) -> None:
    try:
        import json
        _ensure_dir(os.path.dirname(meta_file))
        with open(meta_file, 'w') as f:
            json.dump(data, f, default=str)
    except Exception:
        pass

# -------------------- Price history cache --------------------

def _price_csv_path(ticker: str, cache_dir: Optional[str] = None) -> str:
    base = cache_dir or DEFAULT_DATA_DIR
    _ensure_dir(base)
    return os.path.join(base, f"{ticker}_1d.csv")


def _sanitize_hist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", *REQUIRED_PRICE_COLS])
    out = df.copy()
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'])
    else:
        out = out.reset_index().rename(columns={'index': 'Date'})
        out['Date'] = pd.to_datetime(out['Date'])
    # Keep only required cols
    cols = ['Date'] + [c for c in REQUIRED_PRICE_COLS if c in out.columns]
    out = out[cols].dropna().sort_values('Date').reset_index(drop=True)
    return out


def load_price_history(ticker: str, years: int = 3, cache_dir: Optional[str] = None, force_refresh: Optional[bool] = None) -> pd.DataFrame:
    """Load daily price history for ticker from local CSV cache or yfinance.
    Will compute realized volatility columns expected by downstream logic.
    """
    import yfinance as yf

    cdir = cache_dir or DEFAULT_DATA_DIR
    frefresh = DEFAULT_FORCE_REFRESH if force_refresh is None else bool(force_refresh)
    csv_path = _price_csv_path(ticker, cdir)

    # Try reading cache first
    if not frefresh and os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df = _sanitize_hist(df)
            if not df.empty:
                # Ensure enough lookback; if too short, try refresh
                if years is None or years <= 0:
                    return df
                cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=365*years)
                if df['Date'].min() <= cutoff - pd.Timedelta(days=5):
                    return df
        except Exception:
            df = pd.DataFrame()
    # Fetch from network
    period = f"{max(1, int(years))}y" if years and years > 0 else "max"
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval='1d', auto_adjust=False, actions=False)
        if hist is None or hist.empty:
            raise RuntimeError("Empty history from yfinance")
        hist = hist.rename(columns={c: c.capitalize() for c in hist.columns})
        hist = hist.reset_index().rename(columns={hist.columns[0]: 'Date'})
        df = _sanitize_hist(hist)
        # Save
        df.to_csv(csv_path, index=False)
    except Exception:
        # On failure, fall back to any existing cache
        if os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df = _sanitize_hist(df)
            except Exception:
                df = pd.DataFrame(columns=['Date', *REQUIRED_PRICE_COLS])
        else:
            df = pd.DataFrame(columns=['Date', *REQUIRED_PRICE_COLS])

    # Add realized volatility features if available
    if not df.empty and 'Close' in df.columns:
        ret = df['Close'].pct_change()
        df['rv21'] = ret.rolling(21).std() * np.sqrt(252)
        df['rv5'] = ret.rolling(5).std() * np.sqrt(252)
        df['rv63'] = ret.rolling(63).std() * np.sqrt(252)
        df['rv21'] = df['rv21'].fillna(method='bfill').fillna(df['rv21'].median())
    return df

# -------------------- Earnings cache --------------------

def get_cached_earnings(ticker: str, cache_dir: Optional[str] = None, force_refresh: Optional[bool] = None) -> List[pd.Timestamp]:
    cdir = cache_dir or DEFAULT_DATA_DIR
    frefresh = DEFAULT_FORCE_REFRESH if force_refresh is None else bool(force_refresh)
    meta_file = os.path.join(_meta_dir(cdir), f"{ticker}_meta.json")
    meta = _read_meta_json(meta_file)
    ts = meta.get('earnings_ts')
    if (not frefresh) and ts and _is_fresh(ts, _DEF_EARN_TTL_DAYS * 86400):
        dates = meta.get('earnings_dates', [])
        return [pd.to_datetime(d) for d in dates]

    # Refresh from yfinance
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        # get_earnings_dates is preferred if available
        try:
            edf = tk.get_earnings_dates(limit=16)
            if edf is None:
                raise RuntimeError("no earnings df")
            if 'Earnings Date' in edf.columns:
                edates = [pd.to_datetime(x).date() for x in edf['Earnings Date'].tolist()]
            elif 'Earnings' in edf.columns:
                edates = [pd.to_datetime(x).date() for x in edf['Earnings'].tolist()]
            else:
                edates = []
        except Exception:
            # fallback: quarterly_financials index sometimes contains dates
            q = tk.quarterly_results
            edates = []
            try:
                if q is not None and not q.empty:
                    edates = [pd.to_datetime(x).date() for x in q.columns]
            except Exception:
                edates = []
        meta['earnings_dates'] = [str(d) for d in edates]
        meta['earnings_ts'] = str(_now_utc())
        _write_meta_json(meta_file, meta)
        return [pd.to_datetime(d) for d in edates]
    except Exception:
        # fallback to stale cache
        dates = meta.get('earnings_dates', [])
        return [pd.to_datetime(d) for d in dates]

# -------------------- Option metadata caches --------------------

def get_cached_option_expirations(ticker: str, cache_dir: Optional[str] = None, force_refresh: Optional[bool] = None) -> List[str]:
    cdir = cache_dir or DEFAULT_DATA_DIR
    frefresh = DEFAULT_FORCE_REFRESH if force_refresh is None else bool(force_refresh)
    meta_file = os.path.join(_meta_dir(cdir), f"{ticker}_meta.json")
    meta = _read_meta_json(meta_file)
    ts = meta.get('expirations_ts')
    if (not frefresh) and ts and _is_fresh(ts, _DEF_EXP_TTL_HOURS * 3600):
        return meta.get('expirations', [])

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        exps = tk.options or []
        meta['expirations'] = exps
        meta['expirations_ts'] = str(_now_utc())
        _write_meta_json(meta_file, meta)
        return exps
    except Exception:
        return meta.get('expirations', [])


def get_cached_option_chain(ticker: str, expiry: str, calls_only: bool = True, cache_dir: Optional[str] = None, force_refresh: Optional[bool] = None) -> pd.DataFrame:
    """Cache the option chain (calls) for a ticker/expiry. Chain is cached to CSV for up to TTL.
    Returns a DataFrame similar to yfinance's calls frame.
    """
    import yfinance as yf

    cdir = cache_dir or DEFAULT_DATA_DIR
    frefresh = DEFAULT_FORCE_REFRESH if force_refresh is None else bool(force_refresh)
    odir = _options_dir(ticker, cdir)
    csv_path = os.path.join(odir, f"{expiry.replace('-', '')}_{'calls' if calls_only else 'full'}.csv")
    meta_file = os.path.join(odir, "_meta.json")
    meta = _read_meta_json(meta_file)
    key = os.path.basename(csv_path)
    ts_key = f"{key}_ts"

    if (not frefresh) and os.path.isfile(csv_path):
        ts = meta.get(ts_key)
        if ts and _is_fresh(ts, _DEF_CHAIN_TTL_MIN * 60):
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass

    # Fetch
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)
        df = chain.calls.copy() if calls_only else pd.concat([chain.calls, chain.puts], ignore_index=True)
        df.to_csv(csv_path, index=False)
        meta[ts_key] = str(_now_utc())
        _write_meta_json(meta_file, meta)
        return df
    except Exception:
        # Fallback to stale cache
        if os.path.isfile(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception:
                pass
        return pd.DataFrame()
