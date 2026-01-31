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
DEFAULT_DATA_DIR = os.environ.get("PRICE_DATA_DIR", "src/data/options")
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
    stock_prices_dir = os.path.join(base, "stock_prices")
    _ensure_dir(stock_prices_dir)
    return os.path.join(stock_prices_dir, f"{ticker}_1d.csv")


def _fill_data_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve real market data without fabricating missing days.
    Only return actual trading data, never interpolate or create fake data points.
    """
    if df.empty or 'Date' not in df.columns:
        return df
    
    # Sort by date to ensure proper ordering and remove any duplicates
    df = df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
    
    # Don't fill gaps - return only real market data
    # Markets are closed on weekends and holidays, gaps are normal and expected
    return df


def _sanitize_hist(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", *REQUIRED_PRICE_COLS])
    out = df.copy()
    if 'Date' in out.columns:
        out['Date'] = pd.to_datetime(out['Date'], utc=True)
    else:
        out = out.reset_index().rename(columns={'index': 'Date'})
        out['Date'] = pd.to_datetime(out['Date'], utc=True)
    # Keep only required cols
    cols = ['Date'] + [c for c in REQUIRED_PRICE_COLS if c in out.columns]
    out = out[cols].dropna().sort_values('Date').reset_index(drop=True)
    
    # Fill any gaps in the data
    out = _fill_data_gaps(out)
    
    return out


def load_price_history(ticker: str, years: int = 3, cache_dir: Optional[str] = None, force_refresh: Optional[bool] = None) -> pd.DataFrame:
    """Load daily price history for ticker from local CSV cache or yfinance.
    Will compute realized volatility columns expected by downstream logic.
    Preserves real market data without fabricating missing dates.
    INCREMENTALLY adds new data to existing cache, never overwrites existing dates.
    PERFORMANCE OPTIMIZED: Reduced redundant operations and memory allocations.
    """
    import yfinance as yf

    cdir = cache_dir or DEFAULT_DATA_DIR
    frefresh = DEFAULT_FORCE_REFRESH if force_refresh is None else bool(force_refresh)
    csv_path = _price_csv_path(ticker, cdir)

    # Always try to load existing cached data first - OPTIMIZED: Use faster CSV parsing
    existing_df = pd.DataFrame()
    if os.path.isfile(csv_path):
        try:
            # PERFORMANCE: Use faster CSV reading with explicit dtypes and parse_dates
            existing_df = pd.read_csv(csv_path, parse_dates=['Date'], dtype={
                'Open': 'float32', 'High': 'float32', 'Low': 'float32', 
                'Close': 'float32', 'Volume': 'int64'
            })
            existing_df = _sanitize_hist(existing_df)
            if not existing_df.empty:
                # OPTIMIZATION: Avoid redundant datetime conversion if already parsed
                if not pd.api.types.is_datetime64_any_dtype(existing_df['Date']):
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'], utc=True)
                elif existing_df['Date'].dt.tz is None:
                    existing_df['Date'] = existing_df['Date'].dt.tz_localize('UTC')
        except Exception:
            existing_df = pd.DataFrame()
    
    # If we have good existing data and force refresh is not requested, check if we need updates
    if not frefresh and not existing_df.empty and len(existing_df) > 100:
        latest_date = existing_df['Date'].max()
        days_old = (pd.Timestamp.now(tz='UTC') - latest_date).days
        
        # Use cache if less than 3 days old to prevent constant re-fetching
        # This preserves original data and prevents fabrication on subsequent runs
        if days_old <= 3:
            # PERFORMANCE: Add volatility features in-place to avoid copying
            if 'Close' in existing_df.columns:
                # OPTIMIZATION: Compute all volatility features at once using vectorized operations
                ret = existing_df['Close'].pct_change()
                sqrt_252 = np.sqrt(252)  # Pre-compute constant
                existing_df['rv21'] = ret.rolling(21).std() * sqrt_252
                existing_df['rv5'] = ret.rolling(5).std() * sqrt_252
                existing_df['rv63'] = ret.rolling(63).std() * sqrt_252
                # OPTIMIZATION: Use faster forward fill then backward fill
                existing_df['rv21'] = existing_df['rv21'].fillna(method='ffill').fillna(method='bfill').fillna(existing_df['rv21'].median())
            return existing_df
    
    # Fetch fresh data from yfinance
    new_df = pd.DataFrame()
    try:
        # Calculate exact start and end dates for the requested period
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=365*years + 30)).strftime('%Y-%m-%d')
        
        tk = yf.Ticker(ticker)
        # Use specific date range instead of period to get more reliable data
        hist = tk.history(start=start_date, end=end_date, interval='1d', auto_adjust=False, actions=False)
        
        if hist is None or hist.empty:
            # Fallback to period-based fetch if date range fails
            period = f"{max(1, int(years))}y" if years and years > 0 else "max"
            hist = tk.history(period=period, interval='1d', auto_adjust=False, actions=False)
        
        if hist is None or hist.empty:
            raise RuntimeError("Empty history from yfinance")
        
        # Filter out future dates from raw yfinance data before any processing
        if not hist.empty:
            hist_index_tz = hist.index.tz if hasattr(hist.index, 'tz') else None
            today = pd.Timestamp.now(tz=hist_index_tz or 'UTC').normalize()
            hist = hist[hist.index <= today].copy()
            
        hist = hist.rename(columns={c: c.capitalize() for c in hist.columns})
        hist = hist.reset_index().rename(columns={hist.columns[0]: 'Date'})
        
        # Filter out future dates - only keep data up to today
        if not hist.empty and 'Date' in hist.columns:
            hist['Date'] = pd.to_datetime(hist['Date'], utc=True)
            today = pd.Timestamp.now(tz='UTC').normalize()
            hist = hist[hist['Date'] <= today].copy()
        
        new_df = _sanitize_hist(hist)
        
    except Exception as e:
        # On failure, fall back to existing cache if available
        if not existing_df.empty:
            new_df = existing_df.copy()
        else:
            new_df = pd.DataFrame(columns=['Date', *REQUIRED_PRICE_COLS])

    # INCREMENTAL MERGE: Combine existing and new data, preserving all existing dates
    # PERFORMANCE OPTIMIZED: Reduced string operations and memory allocations
    if not existing_df.empty and not new_df.empty:
        # OPTIMIZATION: Avoid redundant datetime conversions - should already be done above
        if 'Date' in existing_df.columns and not pd.api.types.is_datetime64_any_dtype(existing_df['Date']):
            existing_df['Date'] = pd.to_datetime(existing_df['Date'], utc=True)
        if 'Date' in new_df.columns and not pd.api.types.is_datetime64_any_dtype(new_df['Date']):
            new_df['Date'] = pd.to_datetime(new_df['Date'], utc=True)
        
        # PERFORMANCE: Use datetime comparison instead of string conversion for faster filtering
        if 'Date' in new_df.columns and 'Date' in existing_df.columns:
            existing_min_date = existing_df['Date'].min()
            existing_max_date = existing_df['Date'].max()
            
            # Filter new data to only dates outside existing range or use set-based filtering for overlaps
            before_existing = new_df[new_df['Date'] < existing_min_date] if not new_df.empty else pd.DataFrame()
            after_existing = new_df[new_df['Date'] > existing_max_date] if not new_df.empty else pd.DataFrame()
            
            # For overlapping dates, use efficient set-based filtering
            overlap_range = new_df[(new_df['Date'] >= existing_min_date) & (new_df['Date'] <= existing_max_date)] if not new_df.empty else pd.DataFrame()
            if not overlap_range.empty:
                # OPTIMIZATION: Use datetime index for faster lookup
                existing_dates_set = set(existing_df['Date'])
                overlap_mask = ~overlap_range['Date'].isin(existing_dates_set)
                overlap_new = overlap_range[overlap_mask] if overlap_mask.any() else pd.DataFrame()
            else:
                overlap_new = pd.DataFrame()
            
            # Combine all truly new data
            truly_new_parts = [df for df in [before_existing, overlap_new, after_existing] if not df.empty]
            truly_new = pd.concat(truly_new_parts, ignore_index=True) if truly_new_parts else pd.DataFrame()
        else:
            truly_new = pd.DataFrame()
        
        # Combine: existing data + only truly new dates
        if not truly_new.empty:
            # PERFORMANCE: Use more efficient concatenation and sorting
            df = pd.concat([existing_df, truly_new], ignore_index=True, sort=False)
            df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)
        else:
            # No new data, keep existing
            df = existing_df
    elif not existing_df.empty:
        # Only existing data available
        df = existing_df
    elif not new_df.empty:
        # Only new data available (first time)
        df = new_df
    else:
        # No data available
        df = pd.DataFrame(columns=['Date', *REQUIRED_PRICE_COLS])

    # Save the merged result (preserves existing + adds new)
    if not df.empty and len(df) > 100:
        df.to_csv(csv_path, index=False)

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
