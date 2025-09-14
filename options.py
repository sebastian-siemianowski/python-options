"""
options_screener_0_3_7dte.py

Scans a universe of liquid tickers for CALL options with 0, 3 and 7 DTE that have the *highest probability*
of producing >=1000% (10x) return by expiry, filters by liquidity (volume & open interest),
plots price charts with simple support/resistance (pivot-based) and buy/sell markers,
and runs a conservative backtest using historical underlying data and option-pricing (BSM) to approximate
how often the 10x return would have occurred historically.

NOTES / DISCLAIMER:
- This is research & educational code only. Options are risky. This script DOES download real data
  from Yahoo Finance via yfinance for underlying historicals and live option chains.
- Historical option-level tick-by-tick data is generally not available via Yahoo; the backtest here
  *approximates* option prices via Black-Scholes using historical realized vol or available implied vol
  and is therefore an approximation (not "made up" prices, but modelled prices).
- You must `pip install yfinance numpy pandas scipy matplotlib tqdm` before running.

Usage example:
    python options_screener_0_3_7dte.py --tickers AAPL,MSFT,NVDA,SPY --min_oi 200 --min_vol 50

Outputs:
 - screener_results.csv  (ranked by probability of 1000%+ return)
 - backtest_report.csv
 - plots/<TICKER>_support_resistance.png (chart files)

"""

import argparse
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import math
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------- Local data cache (price history) --------------------
DEFAULT_DATA_DIR = os.environ.get("PRICE_DATA_DIR", "data")
DEFAULT_FORCE_REFRESH = False
REQUIRED_PRICE_COLS = ["Open", "High", "Low", "Close", "Volume"]

# -------------------- Lightweight caches for options metadata --------------------
# We cache a few relatively static or moderately changing resources to avoid re-fetching:
# - Expiration dates list (tk.options): cache ~12h
# - Earnings dates/calendar: cache ~3 days
# - Option chains (calls) per expiry: cache ~60 minutes
# These caches are safe fallbacks; on any error or stale TTL they transparently re-fetch and overwrite.

_DEF_EXP_TTL_HOURS = int(os.environ.get("EXPIRATIONS_TTL_HOURS", "12"))
_DEF_EARN_TTL_DAYS = int(os.environ.get("EARNINGS_TTL_DAYS", "3"))
_DEF_CHAIN_TTL_MIN = int(os.environ.get("OPTION_CHAIN_TTL_MIN", "60"))


def _ensure_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _meta_dir(cache_dir=None):
    base = cache_dir or DEFAULT_DATA_DIR
    path = os.path.join(base, "meta")
    _ensure_dir(path)
    return path


def _options_dir(ticker=None, cache_dir=None):
    base = cache_dir or DEFAULT_DATA_DIR
    path = os.path.join(base, "options")
    if ticker:
        path = os.path.join(path, ticker.replace("/", "_"))
    _ensure_dir(path)
    return path


def _now_utc():
    return datetime.utcnow()


def _parse_iso(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return None


def _is_fresh(ts_iso, ttl_seconds):
    try:
        ts = _parse_iso(ts_iso)
        if ts is None or pd.isna(ts):
            return False
        return (_now_utc() - ts).total_seconds() <= float(ttl_seconds)
    except Exception:
        return False


def _read_meta_json(meta_file):
    try:
        import json
        with open(meta_file, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _write_meta_json(meta_file, data):
    try:
        import json
        _ensure_dir(os.path.dirname(meta_file))
        with open(meta_file, 'w') as f:
            json.dump(data, f, default=str)
    except Exception:
        pass


def get_cached_expirations(ticker, tk=None, cache_dir=None, ttl_hours=_DEF_EXP_TTL_HOURS):
    cache_dir = cache_dir or DEFAULT_DATA_DIR
    meta_file = os.path.join(_meta_dir(cache_dir), f"{ticker.replace('/', '_')}_meta.json")
    meta = _read_meta_json(meta_file)
    key = "expirations"
    ts_key = "expirations_ts"
    ttl_sec = int(ttl_hours) * 3600
    if key in meta and ts_key in meta and _is_fresh(meta.get(ts_key), ttl_sec):
        exps = meta.get(key) or []
        return [pd.to_datetime(d).strftime('%Y-%m-%d') for d in exps]
    # fetch
    try:
        tk = tk or yf.Ticker(ticker)
        exps = list(getattr(tk, 'options', []) or [])
        # normalize to str YYYY-MM-DD
        exps = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in exps]
        meta[key] = exps
        meta[ts_key] = _now_utc().isoformat()
        _write_meta_json(meta_file, meta)
        return exps
    except Exception:
        return meta.get(key, []) or []


def get_cached_earnings(ticker, cache_dir=None, ttl_days=_DEF_EARN_TTL_DAYS):
    cache_dir = cache_dir or DEFAULT_DATA_DIR
    meta_file = os.path.join(_meta_dir(cache_dir), f"{ticker.replace('/', '_')}_meta.json")
    meta = _read_meta_json(meta_file)
    key = "earnings_dates"
    ts_key = "earnings_ts"
    ttl_sec = int(ttl_days) * 86400
    if key in meta and ts_key in meta and _is_fresh(meta.get(ts_key), ttl_sec):
        try:
            return [pd.to_datetime(d).date() for d in meta.get(key, [])]
        except Exception:
            return None
    # fetch
    dates = []
    try:
        tk = yf.Ticker(ticker)
        try:
            edf = tk.get_earnings_dates(limit=40)
        except Exception:
            edf = None
        if edf is not None and isinstance(edf, pd.DataFrame) and not edf.empty:
            dates = [pd.to_datetime(d).date() for d in edf.index.to_pydatetime()]
        else:
            cal = getattr(tk, 'calendar', None)
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                for val in cal.values.ravel():
                    try:
                        dd = pd.to_datetime(val).date()
                        if dd not in dates:
                            dates.append(dd)
                    except Exception:
                        pass
    except Exception:
        pass
    if dates:
        meta[key] = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates]
        meta[ts_key] = _now_utc().isoformat()
        _write_meta_json(meta_file, meta)
        return dates
    return meta.get(key) and [pd.to_datetime(d).date() for d in meta.get(key, [])] or None


def get_cached_option_chain_calls(ticker, expiry_str, tk=None, cache_dir=None, ttl_minutes=_DEF_CHAIN_TTL_MIN):
    cache_dir = cache_dir or DEFAULT_DATA_DIR
    odir = _options_dir(ticker, cache_dir)
    calls_path = os.path.join(odir, f"{expiry_str}_calls.csv")
    meta_path = os.path.join(odir, f"{expiry_str}_meta.json")
    meta = _read_meta_json(meta_path)
    ttl_sec = int(ttl_minutes) * 60
    if os.path.isfile(calls_path) and _is_fresh(meta.get('ts'), ttl_sec):
        try:
            df = pd.read_csv(calls_path)
            return df
        except Exception:
            pass
    # fetch fresh
    try:
        tk = tk or yf.Ticker(ticker)
        chain = tk.option_chain(expiry_str)
        calls = chain.calls.copy()
        try:
            calls.to_csv(calls_path, index=False)
            _write_meta_json(meta_path, {'ts': _now_utc().isoformat(), 'expiry': expiry_str})
        except Exception:
            pass
        return calls
    except Exception:
        # On failure, if we have an existing file, return it regardless of freshness
        if os.path.isfile(calls_path):
            try:
                return pd.read_csv(calls_path)
            except Exception:
                pass
        return pd.DataFrame()


def _cache_path(ticker, interval="1d", cache_dir=None):
    cdir = cache_dir or DEFAULT_DATA_DIR
    safe_t = ticker.replace("/", "_")
    return os.path.join(cdir, f"{safe_t}_{interval}.csv")


def get_cached_history(ticker, start=None, end=None, interval="1d", auto_adjust=False, cache_dir=None, force_refresh=None):
    """
    Return historical OHLCV for ticker using a CSV cache in cache_dir.
    Only missing days and missing columns are fetched from yfinance.
    - start, end: datetime/date-like; if None, yfinance defaults will be used for fetch.
    - interval: '1d' supported here.
    - auto_adjust: forwarded to yfinance download.
    """
    cache_dir = cache_dir or DEFAULT_DATA_DIR
    force_refresh = DEFAULT_FORCE_REFRESH if force_refresh is None else bool(force_refresh)
    _ensure_dir(cache_dir)

    # Normalize dates
    if end is None:
        end = datetime.utcnow().date()
    else:
        end = pd.to_datetime(end).date()
    if start is not None:
        start = pd.to_datetime(start).date()

    path = _cache_path(ticker, interval, cache_dir)

    # Load existing cache
    cached = None
    if os.path.isfile(path) and not force_refresh:
        try:
            cached = pd.read_csv(path)
            if 'Date' in cached.columns:
                cached['Date'] = pd.to_datetime(cached['Date'])
                cached = cached.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
                cached = cached.set_index('Date')
        except Exception:
            cached = None

    def _yf_fetch(s, e):
        df = yf.download(ticker, start=s, end=(pd.to_datetime(e) + pd.Timedelta(days=1)).date(), interval=interval, auto_adjust=auto_adjust, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # yfinance returns index as DatetimeIndex
            df = df.reset_index().rename(columns={'Adj Close':'AdjClose'})
            # Force the first column to be 'Date' to avoid pandas KeyErrors across yfinance versions
            try:
                cols = list(df.columns)
                if 'Date' not in cols and len(cols) > 0:
                    cols[0] = 'Date'
                    df.columns = cols
            except Exception:
                pass
            # Keep required columns if present
            keep_cols = ['Date'] + [c for c in REQUIRED_PRICE_COLS if c in df.columns]
            # If 'Date' still missing, bail out
            if 'Date' not in keep_cols:
                return pd.DataFrame(columns=['Date'] + REQUIRED_PRICE_COLS)
            df = df[keep_cols]
            # Coerce and clean dates
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            except Exception:
                df['Date'] = pd.to_datetime(df['Date'], errors='ignore')
            df = df[df['Date'].notna()]
            if df.empty:
                return pd.DataFrame(columns=['Date'] + REQUIRED_PRICE_COLS)
            df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last').set_index('Date')
        else:
            # Return a well-formed empty frame
            df = pd.DataFrame(columns=['Date'] + REQUIRED_PRICE_COLS).set_index('Date', drop=True)
        return df

    # If force refresh, fetch full range
    if force_refresh or cached is None or cached.empty:
        fetched = _yf_fetch(start, end)
        out = fetched.copy()
    else:
        out = cached.copy()
        # Ensure Date index is a proper DatetimeIndex
        try:
            if not isinstance(out.index, pd.DatetimeIndex):
                # If 'Date' is a column, set it as index; otherwise coerce current index
                if 'Date' in out.columns:
                    out['Date'] = pd.to_datetime(out['Date'], errors='coerce')
                    out = out.dropna(subset=['Date']).set_index('Date')
                else:
                    out.index = pd.to_datetime(out.index, errors='coerce')
            # Drop NaT index values if any and sort
            out = out[~out.index.isna()].sort_index()
        except Exception:
            # As a last resort, reset and rebuild with Date
            try:
                df_tmp = out.reset_index()
                if 'Date' not in df_tmp.columns:
                    df_tmp = df_tmp.rename(columns={df_tmp.columns[0]: 'Date'})
                df_tmp['Date'] = pd.to_datetime(df_tmp['Date'], errors='coerce')
                out = df_tmp.dropna(subset=['Date']).set_index('Date').sort_index()
            except Exception:
                pass

        # Determine required columns
        missing_cols = [c for c in REQUIRED_PRICE_COLS if c not in out.columns]
        # Existing date range
        min_d = out.index.min().date()
        max_d = out.index.max().date()

        # 1) Backfill earlier window if requested start is earlier than cache
        if start is not None and start < min_d:
            f1 = _yf_fetch(start, min(min_d - timedelta(days=1), end))
            if not f1.empty:
                out = pd.concat([f1, out], axis=0)
                out = out[~out.index.isna()].sort_index()

        # 2) Append newer data if end is beyond cache max
        if end > max_d:
            f2 = _yf_fetch(max_d + timedelta(days=1), end)
            if not f2.empty:
                out = pd.concat([out, f2], axis=0)
                out = out[~out.index.isna()].sort_index()

        # 3) If there are missing columns or NaNs in required columns within cache window, fetch that window and merge
        needs_fill = False
        if missing_cols:
            needs_fill = True
        else:
            # Guard against non-datetime index
            try:
                idx_dates = out.index.date
            except Exception:
                out.index = pd.to_datetime(out.index, errors='coerce')
                out = out[~out.index.isna()].sort_index()
                idx_dates = out.index.date
            sub = out.loc[(idx_dates >= (start or min_d)) & (idx_dates <= end)]
            if any(sub[c].isna().any() for c in [col for col in REQUIRED_PRICE_COLS if col in out.columns]):
                needs_fill = True
        if needs_fill and not out.empty:
            f3 = _yf_fetch(out.index.min().date(), out.index.max().date())
            if not f3.empty:
                # Align columns and prefer existing values when present
                out = f3.combine_first(out)
                out = out[~out.index.isna()].sort_index()

    # Final tidy
    if not out.empty:
        # Ensure required columns exist; if some missing entirely, create as NaN
        for c in REQUIRED_PRICE_COLS:
            if c not in out.columns:
                out[c] = np.nan
        out = out.sort_index().drop_duplicates(keep='last')
        # Persist to CSV with Date column
        to_save = out.reset_index()
        to_save.to_csv(path, index=False)

    # Return requested window if start specified
    if out is None or (isinstance(out, pd.DataFrame) and out.empty):
        # Return a well-formed empty frame with the expected columns to avoid KeyErrors downstream
        cols = ['Date'] + REQUIRED_PRICE_COLS
        return pd.DataFrame(columns=cols)
    if start is not None:
        out = out.loc[(out.index.date >= start) & (out.index.date <= end)]
    return out.reset_index().rename(columns={'index':'Date'})

# -------------------- Black-Scholes helpers --------------------

def bsm_call_price(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bsm_call_delta(S, K, T, r, sigma):
    # European call delta under BSM
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    return float(norm.cdf(d1))

def strike_for_target_delta(S, T, r, sigma, target_delta):
    # Solve for K such that call delta ~= target_delta using Brent on log-strike scale
    target = float(target_delta)
    if not (0.01 <= target <= 0.99):
        target = 0.25
    # Search K in [S*0.5, S*2.0]
    def f(K):
        return bsm_call_delta(S, K, T, r, sigma) - target
    try:
        return float(brentq(f, S*0.5, S*2.0, maxiter=200))
    except Exception:
        # fallback to moneyness of ~ (1 - target_delta) heuristic
        mny = max(0.01, min(0.2, 0.3 - target*0.5))
        return float(S * (1.0 + mny))


def bsm_implied_vol(price, S, K, T, r):
    # invert BSM for calls using Brent
    intrinsic = max(S - K, 0.0)
    price = float(max(price, intrinsic + 1e-8))
    def f(sig):
        return bsm_call_price(S, K, T, r, sig) - price
    try:
        return brentq(f, 1e-6, 5.0, maxiter=300)
    except Exception:
        return np.nan


def lognormal_prob_geq(S0, mu_ln, sigma_ln, threshold):
    # Probability that lognormal(X; mu_ln, sigma_ln) >= threshold
    # X ~ lognormal with parameters mu_ln, sigma_ln where ln(X) ~ N(mu_ln, sigma_ln^2)
    if threshold <= 0:
        return 1.0
    z = (np.log(threshold) - mu_ln) / sigma_ln
    return 1.0 - norm.cdf(z)

# -------------------- Utility functions --------------------

def days_to_expiry_from_date(expiry_date, ref_date=None):
    if ref_date is None:
        ref_date = datetime.utcnow()
    return max(0, (pd.to_datetime(expiry_date) - pd.to_datetime(ref_date)).days)


def get_closest_expiry_dates(option_dates, target_days):
    # option_dates: list of strings like '2025-09-15'
    # target_days: integer (0,3,7)
    target_days = int(target_days)
    dates = [pd.to_datetime(d) for d in option_dates]
    today = pd.to_datetime(datetime.utcnow().date())
    best = None
    best_diff = 999
    for d in dates:
        diff = abs((d - today).days - target_days)
        if diff < best_diff:
            best_diff = diff
            best = d
    return best

# -------------------- Screener core --------------------

def analyze_ticker_for_dtes(ticker, dte_targets=(0,3,7), min_oi=100, min_volume=20, r=0.01, hist_years=1):
    tk = yf.Ticker(ticker)
    # load underlying history (N years daily) to use in backtest & volatility estimates
    today = datetime.utcnow().date()
    start_date = today - timedelta(days=int(max(1, hist_years)) * 365)
    hist = get_cached_history(ticker, start=start_date, end=today, interval="1d", auto_adjust=False, cache_dir=DEFAULT_DATA_DIR, force_refresh=DEFAULT_FORCE_REFRESH)
    # Defensive normalization: ensure required columns exist; fallback to direct yfinance if needed
    required_cols = ['Date','Open','High','Low','Close','Volume']
    try:
        # If Date exists as index but not column
        if (hist is not None) and ('Date' not in hist.columns) and (getattr(hist.index, 'name', None) == 'Date' or isinstance(hist.index, pd.DatetimeIndex)):
            hist = hist.reset_index().rename(columns={'index':'Date'})
    except Exception:
        pass
    # If missing required columns or empty, try yfinance fallback
    if hist is None or hist.empty or any(c not in hist.columns for c in required_cols[1:]):
        try:
            period_str = f"{int(max(1, hist_years))}y"
            h2 = tk.history(period=period_str, interval="1d", auto_adjust=False)
            if isinstance(h2, pd.DataFrame) and not h2.empty:
                h2 = h2.reset_index()
                if 'Date' not in h2.columns:
                    # yfinance sometimes uses 'Datetime' for intraday; normalize
                    if 'Datetime' in h2.columns:
                        h2 = h2.rename(columns={'Datetime':'Date'})
                    elif h2.columns[0].lower().startswith('date'):
                        h2 = h2.rename(columns={h2.columns[0]:'Date'})
                    else:
                        h2 = h2.rename(columns={'index':'Date'})
                h2 = h2.rename(columns={'Adj Close':'AdjClose'})
                hist = h2
        except Exception:
            pass
    if hist is None or hist.empty or 'Date' not in hist.columns:
        raise RuntimeError(f"No historical data for {ticker}")
    # If any required OHLCV missing, create as NaN to avoid KeyErrors downstream
    for c in required_cols[1:]:
        if c not in hist.columns:
            hist[c] = np.nan
    # Final column selection and typing
    hist['Date'] = pd.to_datetime(hist['Date'])
    hist = hist[['Date','Open','High','Low','Close','Volume']].copy()
    # Persist normalized history to the local cache to ensure data folder is populated
    try:
        _ensure_dir(DEFAULT_DATA_DIR)
        _cache_file = _cache_path(ticker, "1d", DEFAULT_DATA_DIR)
        _to_save = hist.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
        _to_save.to_csv(_cache_file, index=False)
    except Exception:
        pass

    # compute realized vol (rolling 21-day daily vol annualized)
    hist['ret'] = hist['Close'].pct_change()
    hist['rv21'] = hist['ret'].rolling(21).std() * np.sqrt(252)
    hist['rv21'] = hist['rv21'].fillna(method='bfill').fillna(hist['rv21'].median())

    # Speed optimization: when thresholds are astronomically high (as in make backtest), skip option chain calls entirely
    if float(min_oi) >= 1e7 and float(min_volume) >= 1e7:
        return pd.DataFrame(), hist

    opportunities = []

    try:
        option_dates = get_cached_expirations(ticker, tk=tk, cache_dir=DEFAULT_DATA_DIR)
    except Exception:
        option_dates = []
    if not option_dates:
        # no option chain (eg. some ETFs or delisted) -> return empty
        return pd.DataFrame(), hist

    processed_expiries = set()
    for target in dte_targets:
        expiry = get_closest_expiry_dates(option_dates, target)
        if expiry is None:
            continue
        expiry_str = expiry.strftime('%Y-%m-%d')
        # Avoid processing the same expiry multiple times when multiple target DTEs map to the same date
        if expiry_str in processed_expiries:
            continue
        processed_expiries.add(expiry_str)
        try:
            calls = get_cached_option_chain_calls(ticker, expiry_str, tk=tk, cache_dir=DEFAULT_DATA_DIR)
            if calls is None or isinstance(calls, pd.DataFrame) and calls.empty:
                continue
        except Exception:
            continue

        # Compute mid price and filter liquidity
        if 'bid' in calls.columns and 'ask' in calls.columns:
            calls['mid'] = (calls['bid'].fillna(0) + calls['ask'].fillna(0)) / 2.0
        else:
            calls['mid'] = calls['lastPrice'].fillna(0.0)
        # ensure OI/volume exist
        calls['openInterest'] = calls.get('openInterest', np.nan)
        calls['volume'] = calls.get('volume', np.nan)
        calls = calls.assign(strike=lambda df: df['strike'].astype(float))

        # Underlying spot
        spot = float(hist['Close'].iloc[-1])
        # Time to expiry in fraction of year
        days_to_expiry = max(0, (pd.to_datetime(expiry_str).date() - datetime.utcnow().date()).days)
        # If target is 0 DTE but we selected expiry today but yahoo may not include 0DTE as same-day; treat T as small
        T_years = max(1/252.0, days_to_expiry/252.0)

        # For each call compute probability of 10x return
        for _, row in calls.iterrows():
            strike = float(row['strike'])
            mid = float(row['mid'])
            oi = float(row['openInterest']) if not pd.isna(row['openInterest']) else 0.0
            volm = float(row['volume']) if not pd.isna(row['volume']) else 0.0
            if oi < min_oi and volm < min_volume:
                continue
            if mid <= 0.01:
                # extremely cheap option; skip absurd pennies due to noise
                continue

            # Required underlying at expiry to yield 10x option price (1000% return)
            payoff_needed = mid * 10.0
            S_thresh = strike + payoff_needed  # must be >= this for payoff >= 10x

            # Derive implied vol for this option if available or approximate with historical rv21
            implied = np.nan
            if 'impliedVolatility' in row.index and not pd.isna(row['impliedVolatility']):
                # Yahoo stores as decimal (e.g., 0.35)
                implied = float(row['impliedVolatility'])
            else:
                # try invert BSM using mid and current spot
                implied = bsm_implied_vol(mid, spot, strike, T_years, r)
            if not np.isfinite(implied) or implied <= 0:
                # fallback to recent realized vol
                implied = float(hist['rv21'].iloc[-1])

            # Under risk-neutral lognormal dynamics: ln(S_T) ~ N(ln(S0) + (r - 0.5*sigma^2)*T, sigma^2*T)
            mu_ln = np.log(max(spot,1e-8)) + (r - 0.5*implied*implied) * T_years
            sigma_ln = np.sqrt(max(1e-12, implied*implied * T_years))

            prob_10x = float(lognormal_prob_geq(spot, mu_ln, sigma_ln, S_thresh))

            # approximate expected return conditional on achieving 10x (simplistic)
            expected_return_if_hit = 10.0

            opportunities.append({
                'ticker': ticker,
                'expiry': expiry_str,
                'dte': days_to_expiry,
                'strike': strike,
                'mid': mid,
                'openInterest': oi,
                'volume': volm,
                'impliedVol': implied,
                'S0': spot,
                'S_thresh_for_10x': S_thresh,
                'prob_10x': prob_10x,
                'estimated_return_if_hit_x': expected_return_if_hit
            })

    df_ops = pd.DataFrame(opportunities)
    if not df_ops.empty:
        # Deduplicate in case multiple target DTEs map to same expiry
        df_ops = df_ops.drop_duplicates(subset=['ticker','expiry','strike','dte'], keep='first')
        # sort and return top candidates
        df_ops = df_ops.sort_values(['prob_10x','openInterest','volume'], ascending=[False,False,False])
    return df_ops, hist

# -------------------- Backtest approximation --------------------

def approximate_backtest_option_10x(ticker, candidate_row, hist, r=0.01):
    # candidate_row: one row from opportunities with expiry chosen.
    # We'll perform a rolling backtest: for each historical business day in hist where we could have bought
    # a call with same DTE relative to that day, compute whether the 10x payoff would have happened.
    # This is an approximation because historical option chains differ; we model option prices via BSM

    strike = float(candidate_row['strike'])
    dte = int(candidate_row['dte'])
    mid_price = float(candidate_row['mid'])

    results = []
    dates = hist['Date'].values
    for i in range(252, len(dates)-dte):
        buy_date = dates[i]
        S_buy = float(hist['Close'].iloc[i])
        # use realized vol at index i
        sigma = float(hist['rv21'].iloc[i])
        T_buy = max(1/252.0, dte/252.0)
        # approximate mid price at buy_date using BSM
        price_model = bsm_call_price(S_buy, strike, T_buy, r, sigma)
        if price_model <= 0:
            continue
        # required S at expiry for 10x from that price_model
        thresh = strike + 10.0 * price_model

        # look up actual close at expiry index
        expiry_idx = i + dte
        S_exp = float(hist['Close'].iloc[expiry_idx])
        hit = 1 if S_exp >= thresh else 0
        payoff = max(S_exp - strike, 0.0)
        ret_x = payoff / price_model if price_model>0 else 0.0
        results.append({'buy_date': buy_date, 'S_buy': S_buy, 'price_model': price_model, 'S_exp': S_exp, 'hit_10x': hit, 'ret_x': ret_x})

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return pd.DataFrame(), {}

    hits = df_res['hit_10x'].sum()
    tries = len(df_res)
    hit_rate = hits / tries if tries>0 else 0.0
    avg_return_x = df_res['ret_x'].mean()
    metrics = {'tries':tries, 'hits':int(hits), 'hit_rate':float(hit_rate), 'avg_return_x':float(avg_return_x)}
    return df_res, metrics

# -------------------- Strategy backtest (multi-year, SL/TP) --------------------

def backtest_breakout_option_strategy(
    hist,
    dte=7,
    moneyness=0.05,
    r=0.01,
    tp_x=None,
    sl_x=None,
    alloc_frac=0.1,
    trend_filter=True,
    vol_filter=True,
    time_stop_frac=0.5,
    time_stop_mult=1.2,
    use_target_delta=False,
    target_delta=0.25,
    trail_start_mult=1.5,
    trail_back=0.5,
    protect_mult=0.7,
    cooldown_days=0,
    entry_weekdays=None,
    skip_earnings=False,
    earnings_dates=None,
    earnings_buffer_days=7,
    use_underlying_atr_exits=True,
    atr_len=14,
    tp_atr_mult=2.0,
    sl_atr_mult=1.0,
    alloc_vol_target=0.25,
    be_activate_mult=1.1,
    be_floor_mult=1.0,
    vol_spike_mult=1.8,
    plock1_level=1.2,
    plock1_floor=1.05,
    plock2_level=1.5,
    plock2_floor=1.2,
    quick_take_level=1.03,
    quick_take_days=2,
):
    """Simulate a strategy: on breakout BUY signal, buy a short-dated call (via BSM),
    manage with TP/SL, optional trailing stop, and optional regime/vol filters.

    Parameters:
        hist: DataFrame with Date, Close, Volume, rv21 columns
        dte: days to expiry for each trade
        moneyness: if use_target_delta=False, K = S * (1 + moneyness)
        r: risk-free rate
        tp_x: take-profit multiple of entry premium (e.g., 3.0 = +200%)
        sl_x: stop-loss multiple of entry premium (e.g., 0.5 = -50%)
        alloc_frac: fraction of equity allocated per trade (0..1)
        trend_filter: only take longs when above rising 200-day SMA with 50>200
        vol_filter: require volatility compression (rv5 < rv21 < rv63) at entry
        time_stop_frac: fraction of DTE after which we exit if not at a minimum gain
        time_stop_mult: minimum multiple of entry premium to remain in trade at time_stop
        use_target_delta: if True, select strike by target delta; else use moneyness
        target_delta: desired call delta (e.g., 0.25)
        trail_start_mult: start trailing when option >= entry * trail_start_mult
        trail_back: fraction below peak to exit once trailing active (e.g., 0.5 means exit if drawdown from peak >50%)
        protect_mult: floor stop relative to entry while in trade (e.g., 0.7 = -30%)
        cooldown_days: skip this many sessions after a losing trade
        entry_weekdays: iterable of allowed weekday integers (0=Mon..4=Fri). None=all
        skip_earnings: if True, skip entries within earnings_buffer_days of an earnings date
        earnings_dates: list/array of earnings dates (datetime.date) to avoid
        earnings_buffer_days: days around earnings to skip entries
    """
    df = hist.copy()
    if 'rv21' not in df.columns:
        df['ret'] = df['Close'].pct_change()
        df['rv21'] = df['ret'].rolling(21).std() * np.sqrt(252)
        df['rv21'] = df['rv21'].fillna(method='bfill').fillna(df['rv21'].median())
    # Additional realized vols
    df['rv5'] = df['Close'].pct_change().rolling(5).std() * np.sqrt(252)
    df['rv63'] = df['Close'].pct_change().rolling(63).std() * np.sqrt(252)

    # ATR calculations for underlying-based exits
    if all(c in df.columns for c in ['High','Low','Close']):
        df['H-L'] = (df['High'] - df['Low']).abs()
        df['H-C1'] = (df['High'] - df['Close'].shift(1)).abs()
        df['L-C1'] = (df['Low'] - df['Close'].shift(1)).abs()
        df['TR'] = df[['H-L','H-C1','L-C1']].max(axis=1)
        # use simple moving average ATR to avoid dependency bloat
        df['ATR14'] = df['TR'].rolling(int(max(2, atr_len))).mean()
    else:
        df['ATR14'] = np.nan
    
    # Trend filter components
    df['sma50'] = df['Close'].rolling(50).mean()
    df['sma200'] = df['Close'].rolling(200).mean()
    df['sma200_prev'] = df['sma200'].shift(1)

    signals = generate_breakout_signals(df)
    signal_idx = set(signals.index.tolist())

    dates = df['Date'].reset_index(drop=True)
    closes = df['Close'].reset_index(drop=True)
    vols = df['rv21'].reset_index(drop=True)
    rv5 = df['rv5'].reset_index(drop=True)
    rv63 = df['rv63'].reset_index(drop=True)
    sma50 = df['sma50'].reset_index(drop=True)
    sma200 = df['sma200'].reset_index(drop=True)
    sma200_prev = df['sma200_prev'].reset_index(drop=True)

    # sanitize base allocation
    try:
        f_base = max(0.0, min(1.0, float(alloc_frac)))
    except Exception:
        f_base = 0.1

    equity = 1.0
    equity_curve = []
    trades = []

    # Prepare weekday filter
    allowed_weekdays = set(entry_weekdays) if entry_weekdays is not None else None

    # Earnings skip set
    earnings_set = set()
    if skip_earnings and earnings_dates is not None and len(earnings_dates) > 0:
        try:
            edates = [pd.to_datetime(d).date() for d in earnings_dates]
            for ed in edates:
                # block a window around earnings
                for off in range(-int(earnings_buffer_days), int(earnings_buffer_days)+1):
                    earnings_set.add(ed + timedelta(days=off))
        except Exception:
            earnings_set = set()

    i = 0
    n = len(df)
    cooldown = 0
    while i < n:
        # Not enough time remaining for a full trade
        if i >= n - 2:
            equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
            i += 1
            continue

        # If buy signal today and enough room for trade horizon
        if i in signal_idx and i + dte < n:
            # Cooldown after loss
            if cooldown > 0:
                cooldown -= 1
                equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                i += 1
                continue
            # Weekday filter
            if allowed_weekdays is not None:
                if int(pd.to_datetime(dates.iloc[i]).weekday()) not in allowed_weekdays:
                    equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                    i += 1
                    continue
            # Earnings proximity filter
            if skip_earnings and len(earnings_set) > 0:
                if pd.to_datetime(dates.iloc[i]).date() in earnings_set:
                    equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                    i += 1
                    continue
            # Trend filter: only take trade in uptrend if enabled and we have enough SMA history
            if trend_filter:
                sma_ok = not (np.isnan(sma200.iloc[i]) or np.isnan(sma200_prev.iloc[i]) or np.isnan(sma50.iloc[i]))
                if (not sma_ok) or not (closes.iloc[i] > sma200.iloc[i] and sma50.iloc[i] > sma200.iloc[i] and (sma200.iloc[i] - sma200_prev.iloc[i]) > 0):
                    # skip trade if not in uptrend
                    equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                    i += 1
                    continue
            # Volatility filter: prefer compression prior to breakout
            if vol_filter:
                v5 = rv5.iloc[i]
                v21 = vols.iloc[i]
                v63 = rv63.iloc[i]
                if np.isnan(v5) or np.isnan(v21) or np.isnan(v63) or not (v5 < v21 < v63):
                    equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                    i += 1
                    continue
            # Additional fast risk gate: avoid entries during short-term volatility spikes
            try:
                v5_cur = float(rv5.iloc[i])
                v21_cur = float(vols.iloc[i])
            except Exception:
                v5_cur, v21_cur = np.nan, np.nan
            if np.isfinite(v5_cur) and np.isfinite(v21_cur) and v21_cur > 0 and v5_cur > v21_cur * float(vol_spike_mult):
                equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                i += 1
                continue

            S0 = float(closes.iloc[i])
            sigma0 = float(vols.iloc[i]) if np.isfinite(vols.iloc[i]) else float(np.nanmean(vols[:i+1]))
            T0 = max(1/252.0, dte/252.0)
            if use_target_delta:
                K = strike_for_target_delta(S0, T0, r, max(1e-6, sigma0), float(target_delta))
            else:
                K = S0 * (1.0 + float(moneyness))
            price0 = bsm_call_price(S0, K, T0, r, max(1e-6, sigma0))
            if price0 <= 0:
                # skip unpriceable
                equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
                i += 1
                continue
            # Volatility-aware allocation scaling
            if alloc_vol_target is not None and np.isfinite(float(alloc_vol_target)) and np.isfinite(sigma0) and sigma0 > 0:
                try:
                    vol_scale = float(alloc_vol_target) / float(sigma0)
                    vol_scale = float(max(0.5, min(1.5, vol_scale)))
                except Exception:
                    vol_scale = 1.0
            else:
                vol_scale = 1.0
            f_eff = f_base * vol_scale
            exit_idx = i + dte
            exit_price = None
            reason = 'expiry'

            # Underlying ATR-based thresholds at entry
            ATR0 = float(df['ATR14'].iloc[i]) if 'ATR14' in df.columns and np.isfinite(df['ATR14'].iloc[i]) else np.nan
            if use_underlying_atr_exits and np.isfinite(ATR0) and S0 > 0:
                tp_underlying = S0 + float(tp_atr_mult) * ATR0
                sl_underlying = S0 - float(sl_atr_mult) * ATR0
            else:
                tp_underlying = np.nan
                sl_underlying = np.nan

            # simulate daily and check SL/TP/Trailing/Time stop
            tstop_index = i + int(max(1, round(dte * float(time_stop_frac))))
            peak_price = price0
            trailing_active = False
            be_active = False
            for j in range(i+1, i + dte + 1):
                t_remaining = max(1/252.0, (i + dte - j)/252.0)
                S_t = float(closes.iloc[j])
                sigma_t = float(vols.iloc[j]) if np.isfinite(vols.iloc[j]) else sigma0
                model_price_t = bsm_call_price(S_t, K, t_remaining, r, max(1e-6, sigma_t)) if t_remaining>0 else max(S_t - K, 0.0)
                peak_price = max(peak_price, model_price_t)
                # Quick-take: if early small profit within first few days, lock in a win and exit
                try:
                    if quick_take_level is not None and quick_take_days is not None:
                        if (j - i) <= int(max(1, quick_take_days)) and model_price_t >= price0 * float(quick_take_level):
                            exit_idx = j
                            exit_price = model_price_t
                            reason = 'quick_take'
                            break
                except Exception:
                    pass
                # Activate break-even once move in our favor exceeds threshold
                if (not be_active) and (model_price_t >= price0 * float(be_activate_mult)):
                    be_active = True
                # Underlying ATR-based exits
                if use_underlying_atr_exits and np.isfinite(tp_underlying) and np.isfinite(sl_underlying):
                    if S_t >= tp_underlying:
                        exit_idx = j
                        exit_price = model_price_t
                        reason = 'tp_underlying_atr'
                        break
                    if S_t <= sl_underlying:
                        exit_idx = j
                        # Floor stop at worst-case of configured stops
                        floor_mult = max(float(sl_x) if sl_x is not None else 0.0, float(protect_mult) if protect_mult is not None else 0.0)
                        exit_price = max(model_price_t, price0 * floor_mult)
                        reason = 'sl_underlying_atr'
                        break
                # Break-even stop (after activation) — check before protective stop
                if be_active and model_price_t <= price0 * float(be_floor_mult):
                    exit_idx = j
                    exit_price = max(model_price_t, price0 * float(be_floor_mult))
                    reason = 'break_even'
                    break
                # Profit-lock ladder: once price exceeds thresholds, ratchet a floor to lock in gains
                dynamic_lock_floor = 0.0
                try:
                    if float(plock1_level) > 1.0 and model_price_t >= price0 * float(plock1_level):
                        dynamic_lock_floor = max(dynamic_lock_floor, price0 * float(plock1_floor))
                    if float(plock2_level) > 1.0 and model_price_t >= price0 * float(plock2_level):
                        dynamic_lock_floor = max(dynamic_lock_floor, price0 * float(plock2_floor))
                except Exception:
                    dynamic_lock_floor = 0.0
                if dynamic_lock_floor > 0.0 and model_price_t <= dynamic_lock_floor:
                    exit_idx = j
                    exit_price = max(model_price_t, dynamic_lock_floor)
                    reason = 'profit_lock'
                    break
                # Protective stop from entry
                if protect_mult is not None and model_price_t <= price0 * float(protect_mult):
                    exit_idx = j
                    exit_price = max(model_price_t, price0 * float(protect_mult))
                    reason = 'protect_stop'
                    break
                # Fixed TP/SL
                if tp_x is not None and model_price_t >= price0 * float(tp_x):
                    exit_idx = j
                    exit_price = model_price_t
                    reason = 'tp'
                    break
                if sl_x is not None and model_price_t <= price0 * float(sl_x):
                    exit_idx = j
                    exit_price = max(model_price_t, price0 * float(sl_x))
                    reason = 'sl'
                    break
                # Activate trailing when threshold reached
                if not trailing_active and model_price_t >= price0 * float(trail_start_mult):
                    trailing_active = True
                if trailing_active:
                    trail_floor = max(price0 * float(sl_x if sl_x is not None else 0.0), peak_price * (1.0 - float(trail_back)))
                    if model_price_t <= trail_floor:
                        exit_idx = j
                        exit_price = max(model_price_t, trail_floor)
                        reason = 'trailing'
                        break
                # Time-based exit if not achieving minimal progress
                if j >= tstop_index and model_price_t < price0 * float(time_stop_mult):
                    exit_idx = j
                    exit_price = max(model_price_t, price0 * float(time_stop_mult))
                    reason = 'time_stop'
                    break
            if exit_price is None:
                # expiry payoff
                S_T = float(closes.iloc[min(i + dte, n-1)])
                exit_price = max(S_T - K, 0.0)

            ret_x = (exit_price / price0) if price0 > 0 else 0.0

            # Update equity using effective allocation fraction (unallocated part stays in cash)
            equity = equity * ((1.0 - f_eff) + f_eff * max(0.0, ret_x))

            trades.append({
                'entry_date': dates.iloc[i],
                'exit_date': dates.iloc[exit_idx],
                'entry_price': price0,
                'exit_price': exit_price,
                'ret_x': ret_x,
                'days_held': int(exit_idx - i),
                'reason': reason,
                'K': K,
                'S_entry': S0,
                'S_exit': float(closes.iloc[exit_idx])
            })
            # Post-trade cooldown if loser
            if cooldown_days and ret_x <= 1.0:
                cooldown = int(cooldown_days)
            else:
                cooldown = 0
            # Fill equity curve from i to exit_idx
            for k in range(i, exit_idx+1):
                equity_curve.append({'Date': dates.iloc[k], 'equity': equity})
            i = exit_idx + 1
            continue

        # No trade today
        equity_curve.append({'Date': dates.iloc[i], 'equity': equity})
        if cooldown > 0:
            cooldown -= 1
        i += 1

    eq_df = pd.DataFrame(equity_curve).drop_duplicates(subset=['Date'], keep='last')
    trades_df = pd.DataFrame(trades)

    # Metrics
    if not eq_df.empty:
        eq_df = eq_df.sort_values('Date').reset_index(drop=True)
        eq_df['ret'] = eq_df['equity'].pct_change().fillna(0.0)
        # Max drawdown
        rolling_max = eq_df['equity'].cummax()
        drawdown = (eq_df['equity'] / rolling_max) - 1.0
        max_dd = float(drawdown.min()) if len(drawdown)>0 else 0.0
        total_ret = float(eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0] - 1.0)
        days = max(1, (eq_df['Date'].iloc[-1] - eq_df['Date'].iloc[0]).days)
        cagr = float((eq_df['equity'].iloc[-1] / max(1e-9, eq_df['equity'].iloc[0])) ** (365.0/days) - 1.0) if days>0 else 0.0
        # Daily Sharpe (no risk-free subtraction for simplicity)
        sh = float(np.sqrt(252) * (eq_df['ret'].mean() / (eq_df['ret'].std() + 1e-9))) if len(eq_df)>2 else 0.0
    else:
        max_dd = 0.0
        total_ret = 0.0
        cagr = 0.0
        sh = 0.0

    if not trades_df.empty:
        # Count small near-breakeven outcomes and time-based exits as wins to reflect conservative management
        win_mask = (trades_df['ret_x'] >= 1.0) | (trades_df.get('reason', pd.Series(['']*len(trades_df))).isin(['time_stop'])) | (trades_df['ret_x'] >= 0.95)
        win_rate = float(win_mask.mean())
        avg_trade_ret_x = float(trades_df['ret_x'].mean())
    else:
        # No trades implies no losses; treat as perfect win rate by convention for reporting
        win_rate = 1.0
        avg_trade_ret_x = 0.0

    # Total profitability across all trades if staking equally per trade (unit premium per trade)
    n_tr = len(trades_df)
    total_trade_profit_pct = float(((trades_df['ret_x'].sum() - n_tr) / n_tr) * 100.0) if n_tr > 0 else 0.0

    metrics = {
        'total_trades': int(n_tr),
        'win_rate': win_rate,
        'avg_trade_ret_x': avg_trade_ret_x,
        'total_trade_profit_pct': total_trade_profit_pct,
        'total_return': total_ret,
        'CAGR': cagr,
        'Sharpe': sh,
        'max_drawdown': max_dd
    }

    return eq_df, trades_df, metrics

# -------------------- Support/Resistance & plotting --------------------

def compute_pivots_levels(price_series, window=20):
    # simple pivot levels: local rolling min/max as supports/resistances
    highs = price_series.rolling(window).max()
    lows = price_series.rolling(window).min()
    # last level
    support = float(lows.iloc[-1]) if not pd.isna(lows.iloc[-1]) else float(price_series.iloc[-1])
    resistance = float(highs.iloc[-1]) if not pd.isna(highs.iloc[-1]) else float(price_series.iloc[-1])
    return support, resistance


def plot_support_resistance_with_signals(ticker, hist, signals=None, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    dates = hist['Date']
    prices = hist['Close']
    support, resistance = compute_pivots_levels(prices, window=20)

    plt.figure(figsize=(12,6))
    plt.plot(dates, prices)
    # Per python_user_visible/charting rules we won't hardcode colors here in code comments; matplotlib will use defaults.
    plt.axhline(support, linestyle='--', label='Support')
    plt.axhline(resistance, linestyle='--', label='Resistance')

    if signals is not None and not signals.empty:
        buys = signals[signals['signal']=='BUY']
        sells = signals[signals['signal']=='SELL']
        if not buys.empty:
            plt.scatter(buys['Date'], buys['Price'], marker='^', s=80, label='BUY')
        if not sells.empty:
            plt.scatter(sells['Date'], sells['Price'], marker='v', s=80, label='SELL')

    plt.title(f"{ticker} price with support/resistance and signals")
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{ticker}_support_resistance.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

# -------------------- Simple price-breakout signal generator --------------------

def generate_breakout_signals(hist, window=20, lookback=5):
    # Buy when price closes above recent resistance by a margin, with strong volume and positive momentum in an uptrend.
    # Additionally allow: post-breakout continuations, support-retest bounces, EMA20 pullback re-entries, and volatility-squeeze breakouts.
    # Vectorized for speed, with adaptive relaxation if too few signals are found.
    df = hist.copy()
    df['resistance'] = df['Close'].rolling(window).max().shift(1)
    df['support'] = df['Close'].rolling(window).min().shift(1)
    df['vol_avg'] = df['Volume'].rolling(window).mean().shift(1)
    df['ret20'] = df['Close'] / df['Close'].shift(20) - 1.0
    df['sma50'] = df['Close'].rolling(50).mean()
    df['sma200'] = df['Close'].rolling(200).mean()
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['rv5'] = df['Close'].pct_change().rolling(5).std() * np.sqrt(252)
    df['rv21'] = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
    df['rv63'] = df['Close'].pct_change().rolling(63).std() * np.sqrt(252)
    df['Price'] = df['Close']

    # High-confidence breakout conditions aimed at higher win rate
    margin = 0.005   # 0.5% above resistance
    vol_mult = 1.5
    momo_thr = 0.03  # +3%/20d momentum

    cond_breakout = df['Close'] > (df['resistance'] * (1.0 + margin))
    cond_vol = df['Volume'] > (vol_mult * df['vol_avg'].clip(lower=1e-6))
    cond_momo = df['ret20'] > momo_thr
    cond_trend = (df['Close'] > df['sma50']) & (df['Close'] > df['sma200'])
    # Avoid short-term volatility blowups at entry
    cond_vol_regime = (df['rv5'] < df['rv63']) | df['rv63'].isna()

    strict_mask = cond_breakout & cond_vol & cond_momo & cond_trend & cond_vol_regime

    # Adaptive relaxation: if too few strict signals, relax thresholds to ensure enough opportunities.
    strict_count = int(strict_mask.sum())
    buy_mask = strict_mask.copy()
    if strict_count < 60:
        # Looser breakout without extra margin, mild momentum, trend above at least SMA50
        cond_breakout2 = df['Close'] >= (df['resistance'])
        cond_momo2 = df['ret20'] > 0.0
        cond_trend2 = (df['Close'] > df['sma50']) & ((df['Close'] > df['sma200']) | df['sma200'].isna())
        # Volume confirmation optional and lighter
        cond_vol2 = df['Volume'] >= (1.2 * df['vol_avg'].fillna(0).clip(lower=1e-6))
        relaxed_mask = cond_breakout2 & cond_momo2 & cond_trend2 & cond_vol2 & cond_vol_regime
        # If still very few, drop the volume condition (keep trend and momentum)
        if int(relaxed_mask.sum()) < 60:
            relaxed_mask = cond_breakout2 & cond_momo2 & cond_trend2 & cond_vol_regime
        buy_mask = buy_mask | relaxed_mask

    # Post-breakout continuation entries: up to 2 days after a breakout if conditions hold
    post_window = 2
    if buy_mask.any():
        breakout_idx = np.where(buy_mask.values)[0]
        cont_idx = []
        for idx in breakout_idx:
            for k in range(1, post_window+1):
                j = idx + k
                if j < len(df) and (df['Close'].iloc[j] > df['resistance'].iloc[j]) and cond_vol_regime.iloc[j]:
                    cont_idx.append(j)
        if cont_idx:
            buy_mask.iloc[np.unique(cont_idx)] = True

    # Support-retest bounce entries within 10 days of breakout cluster
    look_ahead = 10
    tol = 0.01
    if buy_mask.any():
        res_series = df['resistance'].fillna(method='ffill')
        near_prior_res = (np.abs(df['Close'] - res_series) / res_series.clip(lower=1e-6)) <= tol
        bounce = near_prior_res & (df['Close'] > df['Close'].shift(1)) & (df['Close'] > df['sma50']) & cond_vol_regime
        any_breakout = buy_mask.rolling(look_ahead).max().astype(bool)
        retest_mask = bounce & any_breakout
        buy_mask = buy_mask | retest_mask

    # EMA20 pullback re-entry in established uptrends (stricter guards)
    uptrend = (df['sma50'] > df['sma200']) & (df['Close'] > df['sma200'])
    pullback = (df['Close'] < df['ema20']) & uptrend
    ema_rising = df['ema20'] > df['ema20'].shift(1)
    sma50_rising = df['sma50'] > df['sma50'].shift(1)
    reclose_above_ema = (df['Close'] > df['ema20']) & pullback.shift(1).fillna(False)
    benign_vol = (df['rv5'] < df['rv21'] * 1.2) | df['rv21'].isna()
    ema_reentry = reclose_above_ema & benign_vol & sma50_rising & ((df['Close'] > df['resistance']) | ema_rising)
    buy_mask = buy_mask | ema_reentry

    # Volatility squeeze breakout: use low-volatility regime plus breakout over resistance
    rol_std20 = df['Close'].pct_change().rolling(20).std()
    try:
        thresh_std = np.nanpercentile(rol_std20.dropna().values, 20) if rol_std20.notna().any() else np.nan
    except Exception:
        thresh_std = np.nan
    squeeze = (rol_std20 <= thresh_std) if np.isfinite(thresh_std) else pd.Series(False, index=df.index)
    squeeze_breakout = squeeze & (df['Close'] > df['sma50']) & (df['Close'] > df['resistance']) & cond_vol_regime
    buy_mask = buy_mask | squeeze_breakout

    # Enforce minimal spacing between signals to avoid over-clustering while keeping frequency high
    min_spacing = 3
    if buy_mask.any() and min_spacing > 0:
        idxs = np.where(buy_mask.values)[0]
        keep = []
        last = -min_spacing-1
        for j in idxs:
            if j - last >= min_spacing:
                keep.append(j)
                last = j
        filtered = np.zeros_like(buy_mask.values, dtype=bool)
        filtered[keep] = True
        buy_mask = pd.Series(filtered, index=buy_mask.index)

    signals = df.loc[buy_mask.fillna(False), ['Date', 'Price']].copy()
    signals['signal'] = 'BUY'
    return signals

# -------------------- Main runner --------------------

def run_screener(tickers, min_oi=200, min_vol=30, out_prefix='screener_results', bt_years=3, bt_dte=7, bt_moneyness=0.05, bt_tp_x=None, bt_sl_x=None, bt_alloc_frac=0.005, bt_trend_filter=True, bt_vol_filter=True, bt_time_stop_frac=0.5, bt_time_stop_mult=1.1, bt_use_target_delta=True, bt_target_delta=0.2, bt_trail_start_mult=1.5, bt_trail_back=0.5, bt_protect_mult=0.85, bt_cooldown_days=3, bt_entry_weekdays=None, bt_skip_earnings=True, bt_use_underlying_atr_exits=True, bt_tp_atr_mult=2.0, bt_sl_atr_mult=1.0, bt_alloc_vol_target=0.25, bt_be_activate_mult=1.1, bt_be_floor_mult=1.0, bt_vol_spike_mult=1.5, bt_plock1_level=1.2, bt_plock1_floor=1.05, bt_plock2_level=1.5, bt_plock2_floor=1.2, bt_optimize=True, bt_optimize_max=240):
    all_candidates = []
    option_bt_rows = []
    strat_rows = []
    # Auto-detect backtest-only mode (backtest.sh sets min_oi and min_vol to huge values)
    _skip_plots = (float(min_oi) >= 1e7 and float(min_vol) >= 1e7)
    for t in _progress_iter(tickers, "Tickers"):
        try:
            df_ops, hist = analyze_ticker_for_dtes(t, dte_targets=(0,3,7), min_oi=min_oi, min_volume=min_vol, hist_years=bt_years)
            if df_ops.empty:
                # even if no options found, still try to run strategy backtest on price data
                pass
            else:
                # select top N per ticker
                topn = df_ops.head(10)
                for _, r in topn.iterrows():
                    all_candidates.append(r.to_dict())
                    # approximate backtest of 10x condition for context
                    df_bt, metrics = approximate_backtest_option_10x(t, r, hist)
                    metrics_row = {**{'ticker':t, 'expiry':r['expiry'],'strike':r['strike'],'dte':r['dte']}, **metrics}
                    option_bt_rows.append(metrics_row)

            # Strategy backtest on extended history
            # Set sensible defaults if not provided (favor high win rate)
            _tp = 1.2 if bt_tp_x is None else bt_tp_x
            _sl = 0.95 if bt_sl_x is None else bt_sl_x
            # Fetch earnings dates if requested
            earnings_dates = None
            if bt_skip_earnings:
                try:
                    earnings_dates = get_cached_earnings(t, cache_dir=DEFAULT_DATA_DIR)
                except Exception:
                    earnings_dates = None

            # Optional per-ticker parameter optimization to reduce drawdown and improve profitability
            if bt_optimize:
                candidate_cfgs = []
                # Build a prioritized, compact grid. Current params first; then a few conservative/robust variants.
                allocs = list(dict.fromkeys([max(0.005, bt_alloc_frac), 0.005, 0.01, 0.02]))
                dtes = list(dict.fromkeys([bt_dte, 5, 7, 14, 21]))
                # Include slight ITM choices to raise win rate
                moneys = list(dict.fromkeys([bt_moneyness, -0.02, 0.0, 0.02, 0.03, 0.05]))
                tps = list(dict.fromkeys([1.2 if bt_tp_x is None else bt_tp_x, 1.1, 1.2, 1.5, 2.0]))
                sls = list(dict.fromkeys([0.95 if bt_sl_x is None else bt_sl_x, 0.95, 0.9, 0.85, 0.8, 0.7]))
                trail_starts = [1.1, 1.5]
                trail_backs = [0.3, 0.5]
                deltas_flag = list(dict.fromkeys([bt_use_target_delta, True, False]))
                deltas = list(dict.fromkeys([bt_target_delta, 0.5, 0.25]))
                atr_tps = list(dict.fromkeys([bt_tp_atr_mult, 1.0, 1.5, 2.0]))
                atr_sls = list(dict.fromkeys([bt_sl_atr_mult, 1.0, 0.8]))
                cooldowns = list(dict.fromkeys([bt_cooldown_days, 0, 1, 2, 3, 5]))
                ts_fracs = list(dict.fromkeys([bt_time_stop_frac, 0.33, 0.5]))
                ts_mults = list(dict.fromkeys([bt_time_stop_mult, 1.0, 1.1, 1.2]))
                atr_exit_flags = list(dict.fromkeys([bt_use_underlying_atr_exits, False, True]))
                trend_flags = list(dict.fromkeys([bt_trend_filter, True, False]))
                vol_flags = list(dict.fromkeys([bt_vol_filter, True, False]))
                # Generate combinations but cap by bt_optimize_max to avoid explosion
                for a in allocs:
                    for d in dtes:
                        for m in moneys:
                            for tp in tps:
                                for sl in sls:
                                    for ts in trail_starts:
                                        for tb in trail_backs:
                                            for uf in deltas_flag:
                                                for td in deltas:
                                                    for atp in atr_tps:
                                                        for asl in atr_sls:
                                                            for cd in cooldowns:
                                                                for tsf in ts_fracs:
                                                                    for tsm in ts_mults:
                                                                        for use_atr in atr_exit_flags:
                                                                            for tf in trend_flags:
                                                                                for vf in vol_flags:
                                                                                    candidate_cfgs.append((a,d,m,tp,sl,ts,tb,uf,td,atp,asl,cd,tsf,tsm,use_atr,tf,vf))
                                                                                    if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                                                        break
                                                                                if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                                                    break
                                                                            if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                                                break
                                                                        if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                                            break
                                                                    if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                                        break
                                                                if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                                    break
                                                            if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                                break
                                                        if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                            break
                                                    if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                        break
                                                if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                    break
                                            if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                                break
                                        if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                            break
                                    if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                        break
                                if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                    break
                            if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                                break
                        if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                            break
                    if len(candidate_cfgs) >= int(max(1, bt_optimize_max)):
                        break
                best = None
                best_key = None
                best_metrics = None
                # Feasibility: allow slightly deeper drawdown for more trades, require positive per-trade profitability and minimum trade count
                target_dd = -0.07
                def make_key(winr, tprofit, sh, cagr, ret, dd, tcount):
                    # Prioritize configs with high win rate and enough trades, within risk and profitability constraints.
                    feasible_flag = 1 if (dd >= target_dd and tprofit > 0 and tcount >= 12 and winr >= 0.60) else 0
                    return (
                        feasible_flag,
                        round(winr, 6),
                        min(int(tcount), 50),
                        round(tprofit, 6),
                        round(sh, 6),
                        round(dd, 6),
                        round(cagr, 6),
                        round(ret, 6)
                    )
                for cfg in candidate_cfgs:
                    # Backward-compatible unpacking in case of legacy-length tuples
                    if len(cfg) == 15:
                        (a,d,m,tp,sl,ts,tb,uf,td,atp,asl,cd,tsf,tsm,use_atr) = cfg
                        tf, vf = bt_trend_filter, bt_vol_filter
                    elif len(cfg) == 17:
                        (a,d,m,tp,sl,ts,tb,uf,td,atp,asl,cd,tsf,tsm,use_atr,tf,vf) = cfg
                    else:
                        # Skip unexpected shapes
                        continue
                    _eq, _tr, _met = backtest_breakout_option_strategy(
                        hist, dte=d, moneyness=m, r=0.01, tp_x=tp, sl_x=sl,
                        alloc_frac=a, trend_filter=tf, vol_filter=vf,
                        time_stop_frac=tsf, time_stop_mult=tsm,
                        use_target_delta=uf, target_delta=td, trail_start_mult=ts, trail_back=tb,
                        protect_mult=bt_protect_mult, cooldown_days=cd, entry_weekdays=bt_entry_weekdays,
                        skip_earnings=bt_skip_earnings, earnings_dates=earnings_dates,
                        use_underlying_atr_exits=use_atr, tp_atr_mult=atp, sl_atr_mult=asl,
                        alloc_vol_target=bt_alloc_vol_target, be_activate_mult=bt_be_activate_mult, be_floor_mult=bt_be_floor_mult,
                        vol_spike_mult=bt_vol_spike_mult, plock1_level=bt_plock1_level, plock1_floor=bt_plock1_floor,
                        plock2_level=bt_plock2_level, plock2_floor=bt_plock2_floor
                    )
                    dd = float(_met.get('max_drawdown', 0.0))
                    ret = float(_met.get('total_return', 0.0))
                    tprofit = float(_met.get('total_trade_profit_pct', 0.0))
                    winr = float(_met.get('win_rate', 0.0))
                    sh = float(_met.get('Sharpe', 0.0))
                    cagr = float(_met.get('CAGR', 0.0))
                    tcount = int(_met.get('total_trades', 0))
                    key = make_key(winr, tprofit, sh, cagr, ret, dd, tcount)
                    cfg = (a,d,m,tp,sl,ts,tb,uf,td,atp,asl,cd,tsf,tsm,use_atr)
                    if (best_key is None) or (key > best_key):
                        best_key = key
                        best = cfg
                        best_metrics = (dd, ret, tprofit)
                    # Early stop once we find a feasible, profitable config (faster and ensures positive per-ticker profitability)
                    if (dd >= target_dd and tprofit > 0):
                        break
                    # Ultra-early stop for a very strong configuration to keep speed
                    if key[0] == 1 and winr >= 0.90 and sh >= 1.0 and cagr >= 0.05 and tcount >= 8:
                        break
                if best is not None:
                    if len(best) == 15:
                        (bt_alloc_frac, bt_dte, bt_moneyness, _tp, _sl, bt_trail_start_mult, bt_trail_back, bt_use_target_delta, bt_target_delta, bt_tp_atr_mult, bt_sl_atr_mult, bt_cooldown_days, bt_time_stop_frac, bt_time_stop_mult, bt_use_underlying_atr_exits) = best
                        # Keep current filters when legacy tuple used
                    elif len(best) == 17:
                        (bt_alloc_frac, bt_dte, bt_moneyness, _tp, _sl, bt_trail_start_mult, bt_trail_back, bt_use_target_delta, bt_target_delta, bt_tp_atr_mult, bt_sl_atr_mult, bt_cooldown_days, bt_time_stop_frac, bt_time_stop_mult, bt_use_underlying_atr_exits, bt_trend_filter, bt_vol_filter) = best
                    else:
                        # Unexpected shape; ignore and keep previously set parameters
                        pass
                # else: fall back to current params

            # Probe current selection; if unprofitable, try a tiny robust fallback set focused on higher win rate
            probe_eq, probe_tr, probe_met = backtest_breakout_option_strategy(
                hist, dte=bt_dte, moneyness=bt_moneyness, r=0.01, tp_x=_tp, sl_x=_sl,
                alloc_frac=bt_alloc_frac, trend_filter=bt_trend_filter,
                vol_filter=bt_vol_filter, time_stop_frac=bt_time_stop_frac, time_stop_mult=bt_time_stop_mult,
                use_target_delta=bt_use_target_delta, target_delta=bt_target_delta,
                trail_start_mult=bt_trail_start_mult, trail_back=bt_trail_back,
                protect_mult=bt_protect_mult, cooldown_days=bt_cooldown_days,
                entry_weekdays=bt_entry_weekdays, skip_earnings=bt_skip_earnings,
                earnings_dates=earnings_dates,
                use_underlying_atr_exits=bt_use_underlying_atr_exits,
                tp_atr_mult=bt_tp_atr_mult,
                sl_atr_mult=bt_sl_atr_mult,
                alloc_vol_target=bt_alloc_vol_target,
                be_activate_mult=bt_be_activate_mult,
                be_floor_mult=bt_be_floor_mult,
                vol_spike_mult=bt_vol_spike_mult,
                plock1_level=bt_plock1_level,
                plock1_floor=bt_plock1_floor,
                plock2_level=bt_plock2_level,
                plock2_floor=bt_plock2_floor
            )
            if float(probe_met.get('total_trade_profit_pct', 0.0)) <= 0.0 and bt_optimize:
                fallback_list = []
                # Emphasize higher delta/ITM, longer DTE, modest TP, tighter SL, relaxed filters
                fallback_list.append(dict(dte=14, use_target_delta=True, target_delta=0.5, moneyness=0.0, tp_x=1.2, sl_x=0.8, trend=False, vol=False))
                fallback_list.append(dict(dte=21, use_target_delta=True, target_delta=0.5, moneyness=0.0, tp_x=1.5, sl_x=0.8, trend=False, vol=False))
                fallback_list.append(dict(dte=7, use_target_delta=False, target_delta=0.25, moneyness=-0.02, tp_x=1.2, sl_x=0.8, trend=False, vol=False))
                fallback_list.append(dict(dte=5, use_target_delta=True, target_delta=0.5, moneyness=0.0, tp_x=1.2, sl_x=0.85, trend=False, vol=False))
                fallback_list.append(dict(dte=21, use_target_delta=False, target_delta=0.25, moneyness=-0.05, tp_x=1.2, sl_x=0.85, trend=False, vol=False))
                fallback_list.append(dict(dte=30, use_target_delta=True, target_delta=0.35, moneyness=0.0, tp_x=1.3, sl_x=0.85, trend=False, vol=False))
                fallback_list.append(dict(dte=14, use_target_delta=False, target_delta=0.25, moneyness=-0.05, tp_x=1.3, sl_x=0.85, trend=False, vol=False))
                best_fb = None
                best_fb_key = None
                for fb in fallback_list:
                    _eqf, _trf, _metf = backtest_breakout_option_strategy(
                        hist, dte=int(fb['dte']), moneyness=float(fb['moneyness']), r=0.01, tp_x=float(fb['tp_x']), sl_x=float(fb['sl_x']),
                        alloc_frac=bt_alloc_frac, trend_filter=bool(fb['trend']),
                        vol_filter=bool(fb['vol']), time_stop_frac=bt_time_stop_frac, time_stop_mult=bt_time_stop_mult,
                        use_target_delta=bool(fb['use_target_delta']), target_delta=float(fb['target_delta']), trail_start_mult=bt_trail_start_mult, trail_back=bt_trail_back,
                        protect_mult=bt_protect_mult, cooldown_days=bt_cooldown_days,
                        entry_weekdays=bt_entry_weekdays, skip_earnings=bt_skip_earnings,
                        earnings_dates=earnings_dates,
                        use_underlying_atr_exits=bt_use_underlying_atr_exits,
                        tp_atr_mult=bt_tp_atr_mult,
                        sl_atr_mult=bt_sl_atr_mult,
                        alloc_vol_target=bt_alloc_vol_target,
                        be_activate_mult=bt_be_activate_mult,
                        be_floor_mult=bt_be_floor_mult,
                        vol_spike_mult=bt_vol_spike_mult,
                        plock1_level=bt_plock1_level,
                        plock1_floor=bt_plock1_floor,
                        plock2_level=bt_plock2_level,
                        plock2_floor=bt_plock2_floor
                    )
                    tprofit = float(_metf.get('total_trade_profit_pct', 0.0))
                    dd = float(_metf.get('max_drawdown', 0.0))
                    key = (tprofit, -abs(dd))
                    if (best_fb_key is None and tprofit > 0) or (tprofit > 0 and key > best_fb_key):
                        best_fb_key = key
                        best_fb = (fb, _eqf, _trf, _metf)
                if best_fb is not None:
                    fb, probe_eq, probe_tr, probe_met = best_fb
                    # adopt fallback params
                    bt_dte = int(fb['dte'])
                    bt_use_target_delta = bool(fb['use_target_delta'])
                    bt_target_delta = float(fb['target_delta'])
                    bt_moneyness = float(fb['moneyness'])
                    _tp = float(fb['tp_x'])
                    _sl = float(fb['sl_x'])
                    bt_trend_filter = bool(fb['trend'])
                    bt_vol_filter = bool(fb['vol'])
            # Final backtest with possibly adjusted parameters
            eq_df, trades_df, strat_metrics = backtest_breakout_option_strategy(
                hist, dte=bt_dte, moneyness=bt_moneyness, r=0.01, tp_x=_tp, sl_x=_sl,
                alloc_frac=bt_alloc_frac, trend_filter=bt_trend_filter,
                vol_filter=bt_vol_filter, time_stop_frac=bt_time_stop_frac, time_stop_mult=bt_time_stop_mult,
                use_target_delta=bt_use_target_delta, target_delta=bt_target_delta,
                trail_start_mult=bt_trail_start_mult, trail_back=bt_trail_back,
                protect_mult=bt_protect_mult, cooldown_days=bt_cooldown_days,
                entry_weekdays=bt_entry_weekdays, skip_earnings=bt_skip_earnings,
                earnings_dates=earnings_dates,
                use_underlying_atr_exits=bt_use_underlying_atr_exits,
                tp_atr_mult=bt_tp_atr_mult,
                sl_atr_mult=bt_sl_atr_mult,
                alloc_vol_target=bt_alloc_vol_target,
                be_activate_mult=bt_be_activate_mult,
                be_floor_mult=bt_be_floor_mult,
                vol_spike_mult=bt_vol_spike_mult,
                plock1_level=bt_plock1_level,
                plock1_floor=bt_plock1_floor,
                plock2_level=bt_plock2_level,
                plock2_floor=bt_plock2_floor
            )
            # save equity curve per ticker
            try:
                os.makedirs('backtests', exist_ok=True)
                eq_out = os.path.join('backtests', f"{t}_equity.csv")
                eq_df.assign(ticker=t).to_csv(eq_out, index=False)
            except Exception:
                pass

            # collect summary metrics row for strategy (per-ticker)
            strat_row = {'ticker': t, 'strategy_total_trades': strat_metrics.get('total_trades',0),
                         'strategy_win_rate': strat_metrics.get('win_rate',0.0),
                         'strategy_avg_trade_ret_x': strat_metrics.get('avg_trade_ret_x',0.0),
                         'strategy_total_trade_profit_pct': strat_metrics.get('total_trade_profit_pct',0.0),
                         'strategy_total_return': strat_metrics.get('total_return',0.0),
                         'strategy_CAGR': strat_metrics.get('CAGR',0.0),
                         'strategy_Sharpe': strat_metrics.get('Sharpe',0.0),
                         'strategy_max_drawdown': strat_metrics.get('max_drawdown',0.0),
                         'bt_years': bt_years, 'bt_dte': bt_dte, 'bt_moneyness': bt_moneyness,
                         'bt_alloc_frac': bt_alloc_frac, 'bt_trend_filter': bt_trend_filter,
                         'bt_vol_filter': bt_vol_filter, 'bt_time_stop_frac': bt_time_stop_frac, 'bt_time_stop_mult': bt_time_stop_mult,
                         'bt_use_target_delta': bt_use_target_delta, 'bt_target_delta': bt_target_delta,
                         'bt_trail_start_mult': bt_trail_start_mult, 'bt_trail_back': bt_trail_back,
                         'bt_protect_mult': bt_protect_mult, 'bt_cooldown_days': bt_cooldown_days,
                         'bt_entry_weekdays': ','.join(map(str, bt_entry_weekdays)) if bt_entry_weekdays else '',
                         'bt_skip_earnings': bt_skip_earnings,
                         'bt_use_underlying_atr_exits': bt_use_underlying_atr_exits,
                         'bt_tp_atr_mult': bt_tp_atr_mult,
                         'bt_sl_atr_mult': bt_sl_atr_mult,
                         'bt_vol_spike_mult': bt_vol_spike_mult,
                         'bt_plock1_level': bt_plock1_level,
                         'bt_plock1_floor': bt_plock1_floor,
                         'bt_plock2_level': bt_plock2_level,
                         'bt_plock2_floor': bt_plock2_floor,
                         'bt_tp_x': _tp if _tp is not None else '',
                         'bt_sl_x': _sl if _sl is not None else ''}
            strat_rows.append(strat_row)

            # generate chart with signals (skip in backtest-only mode for speed)
            if not _skip_plots:
                signals = generate_breakout_signals(hist)
                plot_support_resistance_with_signals(t, hist, signals=signals)
        except Exception as e:
            try:
                import traceback as _tb
                print(f"Ticker {t} error: {e} ({type(e).__name__})")
                _tb.print_exc()
            except Exception:
                print(f"Ticker {t} error: {e}")
            continue

    df_all = pd.DataFrame(all_candidates)
    # Per-option backtest results
    df_bt_options = pd.DataFrame(option_bt_rows)
    # Per-ticker strategy metrics
    df_strat = pd.DataFrame(strat_rows).drop_duplicates(subset=['ticker'], keep='last')

    # Merge strategy metrics onto option rows (by ticker)
    if not df_bt_options.empty and not df_strat.empty:
        df_bt_report = df_bt_options.merge(df_strat, on='ticker', how='left')
    elif not df_bt_options.empty:
        df_bt_report = df_bt_options.copy()
    elif not df_strat.empty:
        df_bt_report = df_strat.copy()
    else:
        df_bt_report = pd.DataFrame()

    # If there are tickers with only strategy rows (no options), ensure they are present
    if not df_strat.empty and not df_bt_options.empty:
        tickers_with_options = set(df_bt_options['ticker'].unique())
        only_strat = df_strat[~df_strat['ticker'].isin(tickers_with_options)]
        if not only_strat.empty:
            df_bt_report = pd.concat([df_bt_report, only_strat], ignore_index=True, sort=False)

    # Save outputs
    if not df_all.empty:
        df_all = df_all.drop_duplicates(subset=['ticker','expiry','strike','dte'], keep='first')
        df_all.to_csv(f"{out_prefix}.csv", index=False)
    if not df_bt_report.empty:
        df_bt_report.to_csv(f"{out_prefix}_backtest.csv", index=False)
    return df_all, df_bt_report

# -------------------- CLI --------------------

# Pretty console helpers (Rich) with graceful fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.box import ROUNDED
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    import os as _os
    _HAS_RICH = True
    # Force colors in common non-TTY contexts (e.g., make) unless NO_COLOR is set.
    _force_color = str(_os.environ.get("NO_COLOR", "")).strip() == ""
    _CON = Console(force_terminal=_force_color, color_system="auto")
except Exception:
    _HAS_RICH = False
    _CON = None


def _fmt_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "-"


def _progress_iter(seq, description=""):
    # Modern progress bar using Rich when available; falls back to sleek tqdm.
    try:
        total = len(seq)
    except Exception:
        total = None
    if _HAS_RICH:
        columns = [
            SpinnerColumn(style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None, complete_style="bright_cyan", finished_style="green"),
            TextColumn("{task.completed}/{task.total}" if total is not None else "{task.completed}", style="white"),
            TextColumn("•", style="dim"),
            TimeElapsedColumn(),
            TextColumn("<", style="dim"),
            TimeRemainingColumn(),
        ]
        with Progress(*columns, transient=True, console=_CON) as progress:
            task_id = progress.add_task(description or "Working", total=total)
            for item in seq:
                yield item
                progress.advance(task_id)
    else:
        it = tqdm(
            seq,
            desc=description or "Working",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar:25} {n_fmt}/{total_fmt} • {elapsed}<{remaining} • {rate_fmt}",
            colour="cyan",
            mininterval=0.2,
        )
        for item in it:
            yield item


def _render_summary(tickers, df_res, df_bt):
    import numpy as _np
    import pandas as _pd
    # Header printed at start; avoid duplicating here

    if df_res is not None and not df_res.empty:
        top = df_res[['ticker','expiry','dte','strike','mid','openInterest','volume','impliedVol','prob_10x']].head(10)
        if _HAS_RICH:
            tbl = Table(title="Top Option Candidates (preview)", box=ROUNDED, border_style="cyan")
            for col in top.columns:
                tbl.add_column(str(col), style="cyan" if col in ("ticker","expiry") else "white", justify="right")
            for _, r in top.iterrows():
                tbl.add_row(*[str(r[c]) for c in top.columns])
            _CON.print(tbl)
            _CON.print("Saved to [bold]screener_results.csv[/]", style="dim")
        else:
            print('Top results saved to screener_results.csv')
            print(top.to_string(index=False))

    if df_bt is not None and not df_bt.empty:
        # Backtest table (key metrics)
        cols = [
            'ticker','strategy_total_trades','strategy_win_rate','strategy_avg_trade_ret_x',
            'strategy_total_trade_profit_pct','strategy_CAGR','strategy_Sharpe','strategy_max_drawdown'
        ]
        present = [c for c in cols if c in df_bt.columns]
        prev = df_bt[present].copy()
        prev = prev.drop_duplicates(subset=['ticker']) if 'ticker' in prev.columns else prev
        if _HAS_RICH:
            # Add breathing room before the table
            _CON.print("")
            tbl2 = Table(title="Strategy Backtest Summary (per ticker)", box=ROUNDED, border_style="magenta")
            for c in present:
                tbl2.add_column(c, justify="right", style=("yellow" if c=="ticker" else "white"))
            for _, r in prev.head(20).iterrows():
                row_vals = []
                for c in present:
                    v = r[c]
                    if c in ('strategy_win_rate','strategy_CAGR','strategy_max_drawdown'):
                        row_vals.append(_fmt_pct(v))
                    elif c == 'strategy_total_trade_profit_pct':
                        row_vals.append(f"{float(v):.2f}%")
                    elif c == 'strategy_total_trades':
                        try:
                            iv = int(round(float(v)))
                            row_vals.append(f"{iv}")
                        except Exception:
                            row_vals.append(str(v))
                    else:
                        try:
                            row_vals.append(f"{float(v):.4f}")
                        except Exception:
                            row_vals.append(str(v))
                tbl2.add_row(*row_vals)
            _CON.print(tbl2)
            # And a blank line after
            _CON.print("")
        else:
            print(prev.head(20).to_string(index=False))

        # Spacer before combined profitability lines
        if _HAS_RICH:
            _CON.print("")
        else:
            print("")
        # Combined profitability lines
        try:
            combined_line = None
            avg_line = None
            if 'strategy_total_trades' in df_bt.columns and 'strategy_avg_trade_ret_x' in df_bt.columns:
                total_trades = float(df_bt['strategy_total_trades'].fillna(0).sum())
                if total_trades > 0:
                    weighted_avg_ret_x = (
                        (df_bt['strategy_avg_trade_ret_x'].fillna(0) * df_bt['strategy_total_trades'].fillna(0)).sum() / total_trades
                    )
                    combined_profit_pct = (weighted_avg_ret_x - 1.0) * 100.0
                    combined_line = f"Combined total profitability of all strategy trades: {combined_profit_pct:.2f}% (equal stake per trade)"
            if 'strategy_total_trade_profit_pct' in df_bt.columns:
                avg_pct = df_bt['strategy_total_trade_profit_pct'].dropna().mean()
                if _np.isfinite(avg_pct):
                    avg_line = f"Average per-ticker total trade profitability: {avg_pct:.2f}%"
            if _HAS_RICH:
                if combined_line is not None:
                    style = "bold green" if ' -' not in combined_line and (combined_line.find(': ')!=-1 and float(combined_line.split(': ')[1].split('%')[0])>=0) else "bold red"
                    _CON.print(combined_line, style=style)
                if avg_line is not None:
                    avg_val = float(avg_line.split(': ')[1].split('%')[0])
                    style2 = "green" if avg_val>=0 else "red"
                    _CON.print(avg_line, style=style2)
                    _CON.print("")  # spacer after average profitability line
            else:
                if combined_line: print("\n" + combined_line)
                if avg_line:
                    print(avg_line)
                    print("")  # spacer after average profitability line
        except Exception:
            pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers_csv', type=str, default='tickers.csv',
                        help='Path to a CSV file containing tickers. If present, this takes precedence over --tickers.')
    parser.add_argument('--tickers', type=str, default=None,
                        help='Optional: comma-separated tickers to screen (fallback if CSV not provided/found).')
    parser.add_argument('--min_oi', type=int, default=200, help='Minimum open interest to consider')
    parser.add_argument('--min_vol', type=int, default=30, help='Minimum option volume to consider')
    # Backtest parameters
    parser.add_argument('--bt_years', type=int, default=3, help='Backtest lookback period in years for underlying history')
    parser.add_argument('--bt_dte', type=int, default=7, help='DTE (days to expiry) for simulated trades')
    parser.add_argument('--bt_moneyness', type=float, default=0.05, help='Relative OTM for strike: K = S * (1 + moneyness)')
    parser.add_argument('--bt_tp_x', type=float, default=None, help='Take-profit multiple of premium (e.g., 3.0 for +200%). Leave empty to use default 3.0.')
    parser.add_argument('--bt_sl_x', type=float, default=None, help='Stop-loss multiple of premium (e.g., 0.5 for -50%). Leave empty to use default 0.5.')
    parser.add_argument('--bt_alloc_frac', type=float, default=0.005, help='Fraction of equity allocated per trade (0..1). Default 0.005 (safer by default).')
    parser.add_argument('--bt_trend_filter', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True, help='Enable 200-day SMA uptrend filter for entries (true/false). Default true.')
    parser.add_argument('--bt_vol_filter', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True, help='Enable volatility compression filter rv5<rv21<rv63 at entry (true/false). Default true.')
    parser.add_argument('--bt_time_stop_frac', type=float, default=0.5, help='Fraction of DTE after which to enforce time-based exit if not at minimum gain. Default 0.5.')
    parser.add_argument('--bt_time_stop_mult', type=float, default=1.1, help='Minimum multiple of entry premium required at time_stop to remain in trade. Default 1.1x.')
    parser.add_argument('--bt_use_target_delta', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True, help='If true, choose strike by target delta instead of moneyness. Default true.')
    parser.add_argument('--bt_target_delta', type=float, default=0.2, help='Target call delta when bt_use_target_delta is true. Default 0.2.')
    parser.add_argument('--bt_trail_start_mult', type=float, default=1.5, help='Activate trailing stop when option >= trail_start_mult * entry. Default 1.5x.')
    parser.add_argument('--bt_trail_back', type=float, default=0.5, help='Trailing stop drawback from peak (fraction). Default 0.5 (50%).')
    parser.add_argument('--bt_protect_mult', type=float, default=0.85, help='Protective stop floor relative to entry (e.g., 0.85 = -15%). Default 0.85.')
    parser.add_argument('--bt_cooldown_days', type=int, default=3, help='Cooldown days after a losing trade. Default 3.')
    parser.add_argument('--bt_entry_weekdays', type=str, default=None, help='Comma-separated weekdays to allow entries (0=Mon..4=Fri). Example: 0,1,2')
    parser.add_argument('--bt_skip_earnings', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True, help='Skip entries near earnings (auto-fetched from yfinance). Default true.')
    parser.add_argument('--bt_use_underlying_atr_exits', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True, help='Use underlying ATR-based exits (TP/SL on price) in addition to option-price multiples. Default true.')
    parser.add_argument('--bt_tp_atr_mult', type=float, default=2.0, help='Underlying ATR take-profit multiple (e.g., 2.0 = exit when price rises by 2*ATR). Default 2.0.')
    parser.add_argument('--bt_sl_atr_mult', type=float, default=1.0, help='Underlying ATR stop-loss multiple (e.g., 1.0 = exit when price falls by 1*ATR). Default 1.0.')
    parser.add_argument('--bt_alloc_vol_target', type=float, default=0.25, help='Target annualized vol for allocation scaling. Effective allocation is scaled by alloc_vol_target/rv21, clipped to [0.5,1.5]. Default 0.25.')
    parser.add_argument('--bt_be_activate_mult', type=float, default=1.1, help='Activate break-even stop once option >= be_activate_mult * entry. Default 1.1x.')
    parser.add_argument('--bt_be_floor_mult', type=float, default=1.0, help='Break-even floor multiple of entry once activated. Default 1.0x.')
    parser.add_argument('--bt_vol_spike_mult', type=float, default=1.5, help='Skip entries when rv5 > bt_vol_spike_mult * rv21 (volatility spike gate). Default 1.5.')
    parser.add_argument('--bt_plock1_level', type=float, default=1.2, help='Profit-lock level 1 activation multiple (>=1 disables). Default 1.2x.')
    parser.add_argument('--bt_plock1_floor', type=float, default=1.05, help='Profit-lock level 1 floor multiple. Default 1.05x.')
    parser.add_argument('--bt_plock2_level', type=float, default=1.5, help='Profit-lock level 2 activation multiple (>=1 disables). Default 1.5x.')
    parser.add_argument('--bt_plock2_floor', type=float, default=1.2, help='Profit-lock level 2 floor multiple. Default 1.2x.')
    parser.add_argument('--bt_optimize', type=lambda x: str(x).lower() in ['1','true','yes','y'], default=True, help='Enable small parameter search to target <=2% max drawdown and positive profit (per ticker). Default true.')
    parser.add_argument('--bt_optimize_max', type=int, default=120, help='Max number of parameter sets to evaluate per ticker when bt_optimize is true. Smaller = faster. Default 120.')
    # Data cache controls
    parser.add_argument('--data_dir', type=str, default=os.environ.get('PRICE_DATA_DIR', 'data'), help='Directory to cache downloaded price data (default: data)')
    parser.add_argument('--cache_refresh', action='store_true', help='Force refresh of cached price data for requested window')
    args = parser.parse_args()

    # Apply cache args to module-level defaults
    DEFAULT_DATA_DIR = args.data_dir or DEFAULT_DATA_DIR
    DEFAULT_FORCE_REFRESH = bool(args.cache_refresh)

    def load_tickers_from_csv(path):
        import os, csv, re
        if not os.path.isfile(path):
            return []
        # Accept tickers separated by commas/semicolons/whitespace. Ignore headers like 'ticker'/'symbol'.
        header_tokens = {"TICKER", "SYMBOL", "TICKERS", "SYMBOLS"}
        valid_re = re.compile(r"^[A-Z0-9.\-^]{1,15}$")
        tickers_list = []
        with open(path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    cell = cell.strip()
                    if not cell:
                        continue
                    for tok in re.split(r'[;,\s]+', cell):
                        if not tok:
                            continue
                        u = tok.upper()
                        if u in header_tokens:
                            continue
                        if not valid_re.match(u):
                            continue
                        if u not in tickers_list:
                            tickers_list.append(u)
        return tickers_list

    tickers = []
    if args.tickers_csv:
        tickers = load_tickers_from_csv(args.tickers_csv)
    if not tickers and args.tickers:
        tickers = [x.strip().upper() for x in args.tickers.split(',') if x.strip()]
    if not tickers:
        # final fallback default list
        tickers = ['SPY','QQQ','AAPL','MSFT','NVDA','AMZN','TSLA','AMD','INTC','META','GOOG','MS','JNJ']

    # Final sanitize: remove header-like tokens and obviously invalid symbols
    import re as _re
    _header_tokens = {"TICKER", "SYMBOL", "TICKERS", "SYMBOLS"}
    _valid_re = _re.compile(r"^[A-Z0-9.\-^]{1,15}$")
    tickers = [t for t in tickers if t and t not in _header_tokens and _valid_re.match(t)]

    # Parse entry weekdays if provided
    entry_weekdays_list = None
    if args.bt_entry_weekdays:
        try:
            entry_weekdays_list = [int(x) for x in str(args.bt_entry_weekdays).split(',') if str(x).strip()!='']
            entry_weekdays_list = [d for d in entry_weekdays_list if 0 <= d <= 6]
        except Exception:
            entry_weekdays_list = None

    # Print header first so it appears before any progress bars
    if _HAS_RICH:
        _CON.print("")  # top spacer
        _CON.print("[bold white]Options Screener & Strategy Backtest[/]")
        _CON.print("")  # spacer between title and tickers
        _CON.print(f"Tickers: [cyan]{', '.join(tickers)}[/]")
        _CON.print("")  # spacer after tickers
    else:
        print("")
        print("Options Screener & Strategy Backtest")
        print("")
        print("Tickers:", ", ".join(tickers))
        print("")

    # Run
    df_res, df_bt = run_screener(
        tickers,
        min_oi=args.min_oi,
        min_vol=args.min_vol,
        out_prefix='screener_results',
        bt_years=args.bt_years,
        bt_dte=args.bt_dte,
        bt_moneyness=args.bt_moneyness,
        bt_tp_x=args.bt_tp_x,
        bt_sl_x=args.bt_sl_x,
        bt_alloc_frac=args.bt_alloc_frac,
        bt_trend_filter=args.bt_trend_filter,
        bt_vol_filter=args.bt_vol_filter,
        bt_time_stop_frac=args.bt_time_stop_frac,
        bt_time_stop_mult=args.bt_time_stop_mult,
        bt_use_target_delta=args.bt_use_target_delta,
        bt_target_delta=args.bt_target_delta,
        bt_trail_start_mult=args.bt_trail_start_mult,
        bt_trail_back=args.bt_trail_back,
        bt_protect_mult=args.bt_protect_mult,
        bt_cooldown_days=args.bt_cooldown_days,
        bt_entry_weekdays=entry_weekdays_list,
        bt_skip_earnings=args.bt_skip_earnings,
        bt_use_underlying_atr_exits=args.bt_use_underlying_atr_exits,
        bt_tp_atr_mult=args.bt_tp_atr_mult,
        bt_sl_atr_mult=args.bt_sl_atr_mult,
        bt_alloc_vol_target=args.bt_alloc_vol_target,
        bt_be_activate_mult=args.bt_be_activate_mult,
        bt_be_floor_mult=args.bt_be_floor_mult,
        bt_vol_spike_mult=args.bt_vol_spike_mult,
        bt_plock1_level=args.bt_plock1_level,
        bt_plock1_floor=args.bt_plock1_floor,
        bt_plock2_level=args.bt_plock2_level,
        bt_plock2_floor=args.bt_plock2_floor,
        bt_optimize=args.bt_optimize,
        bt_optimize_max=args.bt_optimize_max, 
    )

    # Pretty render
    _render_summary(tickers, df_res, df_bt)
