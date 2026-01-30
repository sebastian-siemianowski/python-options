import argparse
import math
import os
import sys
import time
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# Reuse existing project caches for price history
try:
    from data_cache import load_price_history
except Exception:
    load_price_history = None

# Pretty console output with Rich
try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, TaskProgressColumn
    from rich.text import Text
    from rich.style import Style
    from rich import box
    from rich.columns import Columns
    from rich.rule import Rule
    from rich.align import Align
    from rich.padding import Padding
    from rich.layout import Layout
    from datetime import datetime
    RICH_AVAILABLE = True
except Exception:  # pragma: no cover
    Console = None
    Table = None
    Panel = None
    box = None
    RICH_AVAILABLE = False

# Defaults (can be overridden via CLI)
DEFAULT_UNIVERSE = 'csv'  # 'csv' or 'russell_2500'
DEFAULT_CSV_PATH = 'data/universes/russell5000_tickers.csv'  # Use 5000 ticker universe

# Market cap bounds (USD)
# Widen defaults to reduce over-filtering of small caps in Russell-like universes.
MIN_MKT_CAP = 10e6       # 10 million (was 300 million)
MAX_MKT_CAP = 100e9      # 100 billion (was 10 billion)

# How many to output
TOP_N = 100

# Universe sanity check (abort if universe too small)
MIN_UNIVERSE_SIZE = 4500  # Updated for russell5000 universe

SLEEP_SECONDS = 0.05  # rate-limiting between requests

# ========================== Helpers =====================================

# Simple meta cache for Ticker.info with TTL under data/meta/
META_DIR = os.path.join('data', 'meta')
INFO_TTL_DAYS_DEFAULT = 1
SECTOR_EVS: Dict[str, List[float]] = {}

# Lightweight market cap cache (leverages builder cache when available)
CAPS_CACHE_FILE = os.path.join(META_DIR, 'caps_cache.csv')  # from builder

_CAPS_CACHE_MAP: Dict[str, float] = {}
_CAPS_CACHE_LOADED = False

def _load_caps_cache_once() -> None:
    global _CAPS_CACHE_LOADED, _CAPS_CACHE_MAP
    if _CAPS_CACHE_LOADED:
        return
    _CAPS_CACHE_LOADED = True
    _CAPS_CACHE_MAP = {}
    try:
        if os.path.exists(CAPS_CACHE_FILE):
            df = pd.read_csv(CAPS_CACHE_FILE)
            for _, row in df.iterrows():
                sym = str(row.get('ticker', '')).strip().upper()
                val = row.get('marketCap', None)
                try:
                    cap = float(val) if pd.notna(val) else float('nan')
                except Exception:
                    cap = float('nan')
                if sym:
                    _CAPS_CACHE_MAP[sym] = cap
    except Exception:
        _CAPS_CACHE_MAP = {}


def _meta_path_for(ticker: str) -> str:
    os.makedirs(META_DIR, exist_ok=True)
    return os.path.join(META_DIR, f"{ticker.upper()}_meta.json")


def _now_ts_iso() -> str:
    try:
        return pd.Timestamp.utcnow().isoformat()
    except Exception:
        import datetime
        return datetime.datetime.utcnow().isoformat()


def _is_fresh_iso(ts_iso: str, ttl_days: float) -> bool:
    try:
        t = pd.to_datetime(ts_iso)
        return (pd.Timestamp.utcnow() - t).total_seconds() <= ttl_days * 86400.0
    except Exception:
        return False


def get_info_cached(tk: yf.Ticker, ttl_days: Optional[float] = None) -> Dict:
    ttl = INFO_TTL_DAYS_DEFAULT if ttl_days is None else float(ttl_days)
    try:
        sym = tk.ticker if hasattr(tk, 'ticker') else None
        if not sym:
            return tk.info or {}
        path = _meta_path_for(sym)
        data = {}
        if os.path.exists(path):
            try:
                import json
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if data and _is_fresh_iso(data.get('info_ts', ''), ttl) and isinstance(data.get('info'), dict):
            return data.get('info', {})
        # refresh
        info_live = tk.info or {}
        try:
            import json
            with open(path, 'w') as f:
                json.dump({'info': info_live, 'info_ts': _now_ts_iso()}, f)
        except Exception:
            pass
        return info_live
    except Exception:
        try:
            return tk.info or {}
        except Exception:
            return {}

def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        b = float(b)
        if b == 0:
            return None
        return float(a) / b
    except Exception:
        return None


def clamp(x: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
    if x is None:
        return None
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return None


def logistic(x: float, k: float = 3.0, x0: float = 0.5) -> float:
    # 0..1 S-shaped mapping centered at x0
    try:
        import math as _m
        return 1.0 / (1.0 + _m.exp(-k * (float(x) - x0)))
    except Exception:
        return 0.5


def human_money(x: float) -> str:
    try:
        n = float(x)
    except Exception:
        return "-"
    sign = "-" if n < 0 else ""
    n = abs(n)
    units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
    for thresh, suffix in units:
        if n >= thresh:
            val = n / thresh
            return f"{sign}{val:,.1f}{suffix}"
    return f"{sign}{n:,.0f}"


def fmt_pct(p: float, decimals: int = 2) -> str:
    try:
        return f"{float(p):.{decimals}f}%"
    except Exception:
        return "-"

def fetch_russell_2500_tickers() -> List[str]:
    """Best-effort fetch of a small-cap universe via Wikipedia.
    Note: pandas.read_html requires 'lxml'. We keep this as an optional path only.
    """
    url = "https://en.wikipedia.org/wiki/Russell_2000"
    try:
        tables = pd.read_html(url)
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any('ticker' in c or 'symbol' in c for c in cols):
                possible = t[[c for c in t.columns if 'ticker' in c.lower() or 'symbol' in c.lower()]].copy()
                tickers = (
                    possible.iloc[:, 0]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\W+$", "", regex=True)
                    .tolist()
                )
                return sorted(set([x for x in tickers if x and x.upper() != 'N/A']))
    except Exception as e:
        print("Warning: couldn't fetch Russell tickers from Wikipedia (ok):", e)
    # Fallback minimal list — recommend CSV
    return [
        "AAPL", "MSFT"
    ]


def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    s = s.replace(' ', '')
    # Map share-class dots to Yahoo dashes (e.g., BRK.B -> BRK-B)
    if '.' in s:
        s = s.replace('.', '-')
    return s


def load_tickers_from_csv(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tickers CSV not found at {path}. Provide --csv PATH or place a Russell 2500 CSV at {path} with a 'ticker' column (one symbol per row)."
        )
    try:
        # Try flexible parsing: column named 'ticker' or 'symbol'; else first column; else parse first row comma-separated
        df = pd.read_csv(path)
        cols_lower = [c.lower() for c in df.columns]
        candidates = [i for i, c in enumerate(cols_lower) if c in ("ticker", "tickers", "symbol", "symbols")] 
        raw: List[str]
        if candidates:
            raw = df.iloc[:, candidates[0]].astype(str).tolist()
        elif df.shape[1] >= 1:
            series = df.iloc[:, 0].astype(str)
            # Handle potential single-cell comma list rows
            raw = []
            for cell in series:
                parts = [p for p in str(cell).replace(";", ",").split(",") if p.strip()]
                raw.extend(parts)
        else:
            # as a last resort, read raw file and split by commas/whitespace
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            raw = [p for p in raw_text.replace(";", ",").replace("\n", ",").split(",") if p.strip()]
        # Normalize and clean
        mapped = [normalize_symbol(t) for t in raw if t and str(t).strip().upper() != 'N/A']
        tickers = sorted(set(mapped))
        if not tickers:
            raise ValueError("No tickers parsed from CSV after normalization.")
        return tickers
    except Exception as e:
        raise RuntimeError(f"Failed to parse tickers from {path}: {e}")


def cagr(begin: float, end: float, years: int = 3) -> Optional[float]:
    try:
        if begin <= 0 or end <= 0:
            return None
        return (end / begin) ** (1.0 / years) - 1.0
    except Exception:
        return None


def get_market_cap(tk: yf.Ticker, retries: int = 3) -> Optional[float]:
    """Robust market cap fetch with multiple fallbacks and shared cache.
    Order of attempts:
    0) Shared caps_cache.csv if present (builder output)
    1) fast_info.market_cap (object or dict)
    2) info['marketCap']
    3) EV inversion when enterpriseValue, totalDebt, totalCash available
    4) sharesOutstanding/impliedSharesOutstanding/floatShares * price
       where price is from fast_info/info/history or cached price history.
    """
    # Load shared cache once and try it first
    try:
        _load_caps_cache_once()
        sym = tk.ticker if hasattr(tk, 'ticker') else None
        if sym:
            key = str(sym).strip().upper()
            cap0 = _CAPS_CACHE_MAP.get(key, None)
            if cap0 is not None and pd.notna(cap0) and float(cap0) > 0:
                return float(cap0)
    except Exception:
        pass

    def _get_fast_info_cap(fi_obj) -> Optional[float]:
        try:
            if fi_obj is None:
                return None
            # fast_info may be an object with attributes or a dict
            if isinstance(fi_obj, dict):
                v = fi_obj.get('market_cap') or fi_obj.get('marketCap')
                return float(v) if v is not None else None
            v = getattr(fi_obj, 'market_cap', None)
            if v is None:
                v = getattr(fi_obj, 'marketCap', None)
            return float(v) if v is not None else None
        except Exception:
            return None

    def _get_price_snapshot(tk_: yf.Ticker) -> Optional[float]:
        # Try fast_info.last_price, then info current price, then last close from cache/history
        try:
            fi = getattr(tk_, 'fast_info', None)
            if fi is not None:
                if isinstance(fi, dict):
                    p = fi.get('last_price') or fi.get('lastPrice') or fi.get('lastClose') or fi.get('regular_market_price')
                    if p is not None:
                        return float(p)
                else:
                    for attr in ('last_price','lastPrice','last_close','regular_market_price','previous_close'):
                        val = getattr(fi, attr, None)
                        if val is not None:
                            return float(val)
        except Exception:
            pass
        try:
            info = tk_.info or {}
            p = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if p is not None:
                return float(p)
        except Exception:
            pass
        # price cache
        try:
            if load_price_history is not None:
                df = load_price_history(tk_.ticker if hasattr(tk_, 'ticker') else None, years=1)
                if df is not None and not df.empty and 'Close' in df.columns:
                    return float(pd.Series(df['Close']).dropna().iloc[-1])
        except Exception:
            pass
        try:
            hist = tk_.history(period='5d', interval='1d', auto_adjust=True)
            if hist is not None and not hist.empty and 'Close' in hist.columns:
                return float(hist['Close'].dropna().iloc[-1])
        except Exception:
            pass
        return None

    def _get_shares_outstanding(tk_: yf.Ticker) -> Optional[float]:
        # Try info field, then balance sheet rows, then implied/float-based approximations
        try:
            info = tk_.info or {}
            for key in ('sharesOutstanding','shares_outstanding','impliedSharesOutstanding','implied_shares_outstanding'):
                so = info.get(key)
                if so is not None and float(so) > 0:
                    return float(so)
            # As a last resort, approximate from floatShares by scaling to total (heuristic factor 1.3)
            fs = info.get('floatShares') or info.get('float_shares')
            if fs is not None and float(fs) > 0:
                approx = float(fs) * 1.3
                if approx > 0:
                    return approx
        except Exception:
            pass
        try:
            bs = getattr(tk_, 'balance_sheet', None)
            if isinstance(bs, pd.DataFrame) and not bs.empty:
                vals = _get_series_values(bs, ['ordinary shares number', 'common stock shares outstanding', 'shareissued', 'common shares', 'weighted average shares outstanding'], take=1)
                if vals and len(vals) >= 1:
                    so2 = float(vals[0])
                    if so2 > 0:
                        return so2
        except Exception:
            pass
        try:
            bsq = getattr(tk_, 'quarterly_balance_sheet', None)
            if isinstance(bsq, pd.DataFrame) and not bsq.empty:
                vals = _get_series_values(bsq, ['ordinary shares number', 'common stock shares outstanding', 'shareissued', 'common shares', 'weighted average shares outstanding'], take=1)
                if vals and len(vals) >= 1:
                    so3 = float(vals[0])
                    if so3 > 0:
                        return so3
        except Exception:
            pass
        return None

    def _ev_inversion_cap(info_dict: Dict) -> Optional[float]:
        try:
            ev = info_dict.get('enterpriseValue') or info_dict.get('enterprise_value')
            td = info_dict.get('totalDebt') or info_dict.get('total_debt')
            cash = info_dict.get('totalCash') or info_dict.get('total_cash') or info_dict.get('cash')
            if ev is None or (td is None and cash is None):
                return None
            td = float(td) if td is not None else 0.0
            cash = float(cash) if cash is not None else 0.0
            ev = float(ev)
            cap = ev - (td - cash)
            if cap > 0:
                return cap
        except Exception:
            pass
        return None

    cap: Optional[float] = None
    for attempt in range(max(1, int(retries) + 1)):
        # 1) fast_info
        try:
            fi = getattr(tk, 'fast_info', None)
            cap = _get_fast_info_cap(fi)
            if cap is not None and cap > 0:
                break
        except Exception:
            pass
        # 2) info marketCap or EV inversion
        info = {}
        try:
            info = tk.info or {}
            v = info.get('marketCap') or info.get('market_cap')
            if v is not None and float(v) > 0:
                cap = float(v)
                break
            # EV inversion
            cap_ev = _ev_inversion_cap(info)
            if cap_ev is not None and cap_ev > 0:
                cap = float(cap_ev)
                break
        except Exception:
            info = {}
        # 3) shares * price
        try:
            price = _get_price_snapshot(tk)
            shares = _get_shares_outstanding(tk)
            if price is not None and shares is not None and price > 0 and shares > 0:
                cap = float(price) * float(shares)
                if cap > 0:
                    break
        except Exception:
            pass
        time.sleep(0.12 * (1 + attempt))

    # Persist to in-memory cache for this run; file persisted later
    try:
        if cap is not None and cap > 0:
            _load_caps_cache_once()
            if sym:
                _CAPS_CACHE_MAP[str(sym).strip().upper()] = float(cap)
            return float(cap)
    except Exception:
        pass
    return None


def _sorted_columns(df: pd.DataFrame) -> List:
    cols = list(df.columns)
    # Try convert to datetime for sorting descending
    try:
        cols_sorted = sorted(cols, key=lambda c: pd.to_datetime(str(c), errors='coerce'), reverse=True)
        return cols_sorted
    except Exception:
        return cols


def get_revenue_values(tk: yf.Ticker, max_points: int = 6) -> Optional[List[float]]:
    # Try annual income statement first
    try:
        # yfinance >=0.2: income_stmt property
        income = getattr(tk, 'income_stmt', None)
        if isinstance(income, pd.DataFrame) and not income.empty:
            df = income.copy()
            idx = [str(i).lower() for i in df.index]
            candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue' or 'net sales' in name or 'sales' in name or 'total operating revenue' in name or 'operating revenue' in name]
            if candidates:
                row = df.iloc[candidates[0]]
                vals = []
                for c in _sorted_columns(df):
                    val = row.get(c)
                    try:
                        if pd.notna(val) and float(val) != 0:
                            vals.append(float(val))
                    except Exception:
                        pass
                if len(vals) >= 4:
                    return vals[:4]  # most recent first
                if len(vals) >= 2:
                    return vals[:min(len(vals), 6)]
    except Exception:
        pass

    # Fallback: legacy financials table
    try:
        fin = tk.financials
        if isinstance(fin, pd.DataFrame) and not fin.empty:
            df = fin.copy()
            idx = [str(i).lower() for i in df.index]
            candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue' or 'net sales' in name or 'sales' in name or 'total operating revenue' in name or 'operating revenue' in name]
            if candidates:
                row = df.iloc[candidates[0]]
                vals = []
                for c in _sorted_columns(df):
                    val = row.get(c)
                    try:
                        if pd.notna(val) and float(val) != 0:
                            vals.append(float(val))
                    except Exception:
                        pass
                if len(vals) >= 4:
                    return vals[:4]
                if len(vals) >= 2:
                    return vals[:min(len(vals), 6)]
    except Exception:
        pass

    # Fallback: quarterly income statement -> construct rolling 4-quarter TTM sums
    try:
        q = getattr(tk, 'quarterly_income_stmt', None)
        if isinstance(q, pd.DataFrame) and not q.empty:
            df = q.copy()
            idx = [str(i).lower() for i in df.index]
            candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue' or 'net sales' in name or 'sales' in name or 'total operating revenue' in name or 'operating revenue' in name]
            if candidates:
                row = df.iloc[candidates[0]]
                # Sort columns descending by quarter date
                cols = _sorted_columns(df)
                series = []
                for c in cols:
                    try:
                        v = row.get(c)
                        if pd.isna(v):
                            v = None
                        else:
                            v = float(v)
                    except Exception:
                        v = None
                    series.append(v)
                # Build rolling 4-quarter TTM sums shifted by 0, 4, 8, 12 quarters
                def sum_window(start: int) -> Optional[float]:
                    window = series[start:start+4]
                    if len(window) < 4:
                        return None
                    nums = [x for x in window if x is not None]
                    # tolerate one missing quarter in the TTM window
                    if len(nums) < 3:
                        return None
                    return float(sum(nums))
                vals = []
                for offset in (0, 4, 8, 12):
                    s = sum_window(offset)
                    if s is not None:
                        vals.append(s)
                if len(vals) >= 4:
                    return vals[:4]
                if len(vals) >= 2:
                    return vals[:len(vals)]
    except Exception:
        pass

    return None


# =================== 100x Bagger Scoring Utilities =====================

def _get_stmt(tk: yf.Ticker, attr: str) -> Optional[pd.DataFrame]:
    try:
        df = getattr(tk, attr, None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.copy()
    except Exception:
        return None
    return None


def _get_last_row_value(df: pd.DataFrame, row_names: List[str]) -> Optional[float]:
    if df is None or df.empty:
        return None
    idx = [str(i).strip().lower() for i in df.index]
    for nm in row_names:
        nm_l = nm.lower()
        candidates = [i for i, name in enumerate(idx) if nm_l == name or nm_l in name]
        if candidates:
            row = df.iloc[candidates[0]]
            # sort columns by date desc
            cols = _sorted_columns(df)
            for c in cols:
                try:
                    val = row.get(c)
                    if pd.notna(val):
                        v = float(val)
                        if v == 0:
                            continue
                        return v
                except Exception:
                    continue
    return None


def _get_series_values(df: pd.DataFrame, row_names: List[str], take: int = 4) -> Optional[List[float]]:
    if df is None or df.empty:
        return None
    idx = [str(i).strip().lower() for i in df.index]
    for nm in row_names:
        nm_l = nm.lower()
        candidates = [i for i, name in enumerate(idx) if nm_l == name or nm_l in name]
        if candidates:
            row = df.iloc[candidates[0]]
            cols = _sorted_columns(df)
            vals: List[float] = []
            for c in cols:
                try:
                    v = row.get(c)
                    if pd.notna(v) and float(v) != 0:
                        vals.append(float(v))
                except Exception:
                    pass
            if vals:
                return vals[:take]
    return None


# ---- Quarterly-to-TTM helpers for better coverage ----

def _ttm_from_quarterly(df: Optional[pd.DataFrame], row_names: List[str], quarters: int = 4) -> Optional[float]:
    """Sum the most recent `quarters` quarterly values for a given row.
    Returns None if insufficient data.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    idx = [str(i).strip().lower() for i in df.index]
    for nm in row_names:
        nm_l = nm.lower()
        candidates = [i for i, name in enumerate(idx) if nm_l == name or nm_l in name]
        if not candidates:
            continue
        row = df.iloc[candidates[0]]
        cols = _sorted_columns(df)
        vals: List[float] = []
        for c in cols:
            try:
                v = row.get(c)
                if pd.notna(v):
                    vv = float(v)
                    # yfinance often reports CapEx as negative; treat numerically
                    vals.append(vv)
            except Exception:
                continue
        if len(vals) >= quarters:
            return float(sum(vals[:quarters]))
    return None


def compute_ltm_momentum_pair(tk: yf.Ticker) -> Tuple[Optional[float], Optional[float]]:
    """Return (current_ltm_momentum, previous_ltm_momentum).
    Current: last 4q vs prior 4q; Previous: the 4q window before that vs its prior 4q.
    Requires at least 12 quarters of data for both; with 8 quarters, only current is computed.
    """
    try:
        q = getattr(tk, 'quarterly_income_stmt', None)
        if not isinstance(q, pd.DataFrame) or q.empty:
            return None, None
        idx = [str(i).strip().lower() for i in q.index]
        candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue' or 'net sales' in name or 'sales' in name]
        if not candidates:
            return None, None
        row = q.iloc[candidates[0]]
        cols = _sorted_columns(q)
        vals: List[Optional[float]] = []
        for c in cols:
            try:
                v = row.get(c)
                vals.append(float(v) if pd.notna(v) else None)
            except Exception:
                vals.append(None)
        numeric = [v for v in vals if v is not None]
        if len(numeric) < 8:
            return None, None
        seq: List[float] = []
        for v in vals:
            if v is not None:
                seq.append(float(v))
            if len(seq) >= 20:
                break
        # compute helper
        def mom_from(seq_slice: List[float]) -> Optional[float]:
            if len(seq_slice) < 8:
                return None
            s_recent = sum(seq_slice[:4])
            s_prior = sum(seq_slice[4:8])
            if s_prior <= 0 or s_recent <= 0:
                return None
            return (s_recent / s_prior) - 1.0
        current = mom_from(seq)
        previous = None
        if len(seq) >= 12:
            previous = mom_from(seq[4:12])
        return current, previous
    except Exception:
        return None, None


def compute_ltm_momentum(tk: yf.Ticker) -> Optional[float]:
    cur, _ = compute_ltm_momentum_pair(tk)
    return cur


def compute_margins(tk: yf.Ticker) -> Tuple[Optional[float], Optional[float]]:
    """Return (gross_margin, operating_margin) in fraction form using a simple, robust source.

    Rationale:
    - yfinance statement schemas vary a lot across tickers/sectors and often cause gaps.
    - For UX reliability, we now derive margins primarily from Ticker.info where Yahoo already
      computes trailing margins. If unavailable, we default to neutral 0.0 values.

    This change removes fragile multi-path statement parsing to eliminate persistent NaNs in gm/om.
    """
    gm = 0.0
    om = 0.0
    try:
        info = tk.info or {}
        gm_info = info.get('grossMargins') or info.get('gross_margins')
        if gm_info is not None:
            try:
                gm = float(gm_info)
            except Exception:
                gm = 0.0
        om_info = info.get('operatingMargins') or info.get('operating_margins')
        if om_info is None:
            # fallback to profit margin as weaker proxy for operating performance
            om_info = info.get('profitMargins') or info.get('profit_margins')
        if om_info is not None:
            try:
                om = float(om_info)
            except Exception:
                om = 0.0
    except Exception:
        # keep neutral defaults
        gm, om = 0.0, 0.0

    # Clip to sane bounds
    try:
        gm = max(-1.0, min(1.0, float(gm)))
    except Exception:
        gm = 0.0
    try:
        om = max(-1.0, min(1.0, float(om)))
    except Exception:
        om = 0.0

    return gm, om


def compute_fcf_margin(tk: yf.Ticker) -> Optional[float]:
    """Compute Free Cash Flow margin = (CFO - Capex) / Revenue.
    Prefers annual statements with a quarterly TTM fallback for robustness.
    As a final fallback, uses Ticker.info fields if available.
    """
    try:
        # Revenue from income (annual), fallback to TTM from quarterly
        inc_a = _get_stmt(tk, 'income_stmt') or _get_stmt(tk, 'financials')
        revenue = _get_last_row_value(inc_a, ['total revenue', 'totalrevenue', 'revenue', 'revenues', 'net sales', 'sales', 'total operating revenue'])
        if revenue is None or revenue == 0:
            inc_q = _get_stmt(tk, 'quarterly_income_stmt')
            revenue = _ttm_from_quarterly(inc_q, ['total revenue', 'totalrevenue', 'revenue', 'revenues', 'net sales', 'sales', 'total operating revenue'])
        # Cash flow statements
        cf_a = _get_stmt(tk, 'cashflow') or _get_stmt(tk, 'cash_flow')
        cfo = _get_last_row_value(cf_a, [
            'total cash from operating activities', 'operating cash flow',
            'totalcashfromoperatingactivities', 'net cash provided by operating activities',
            'cashflowfromoperatingactivities', 'net cash provided by (used in) operating activities',
            'cash provided by operating activities'
        ])
        capex = _get_last_row_value(cf_a, [
            'capital expenditures', 'capital expenditure', 'capitalexpenditures',
            'purchase of property and equipment', 'purchases of property and equipment',
            'payments for property, plant and equipment', 'acquisition of property and equipment'
        ])
        if cfo is None or capex is None:
            # Try quarterly TTM
            cf_q = _get_stmt(tk, 'quarterly_cashflow') or _get_stmt(tk, 'quarterly_cash_flow')
            if cfo is None:
                cfo = _ttm_from_quarterly(cf_q, [
                    'total cash from operating activities', 'operating cash flow',
                    'totalcashfromoperatingactivities', 'net cash provided by operating activities',
                    'cashflowfromoperatingactivities', 'net cash provided by (used in) operating activities',
                    'cash provided by operating activities'
                ])
            if capex is None:
                capex = _ttm_from_quarterly(cf_q, [
                    'capital expenditures', 'capital expenditure', 'capitalexpenditures',
                    'purchase of property and equipment', 'purchases of property and equipment',
                    'payments for property, plant and equipment', 'acquisition of property and equipment'
                ])
        # FINAL FALLBACK using Ticker.info if any of the three components are missing
        if (revenue is None or revenue == 0) or cfo is None or capex is None:
            try:
                info = tk.info or {}
                if revenue in (None, 0):
                    rev_info = info.get('totalRevenue') or info.get('total_revenue') or info.get('netRevenue')
                    if rev_info is not None:
                        revenue = float(rev_info)
                if cfo is None:
                    cfo_info = info.get('operatingCashflow') or info.get('operating_cashflow') or info.get('totalCashFromOperatingActivities')
                    if cfo_info is not None:
                        cfo = float(cfo_info)
                if capex is None:
                    capex_info = info.get('capitalExpenditures') or info.get('capital_expenditures')
                    if capex_info is not None:
                        capex = float(capex_info)
            except Exception:
                pass
        if revenue is None or revenue == 0 or cfo is None or capex is None:
            # Final neutral fallback to avoid NaNs downstream
            return 0.0
        # Capex is often negative in cash flow statements (cash outflow).
        # FCF = CFO - Capex (so if Capex is negative, subtracting adds magnitude).
        fcf = float(cfo) - float(capex)
        return clamp(safe_div(fcf, revenue), -1.0, 1.0)
    except Exception:
        # Final neutral fallback
        return 0.0


def compute_ps_ratio(market_cap: Optional[float], tk: yf.Ticker) -> Optional[float]:
    # Price-to-Sales using most recent annual revenue proxy
    if market_cap is None or market_cap <= 0:
        return None
    inc_a = _get_stmt(tk, 'income_stmt') or _get_stmt(tk, 'financials')
    rev_vals = _get_series_values(inc_a, ['total revenue', 'totalrevenue', 'revenue', 'revenues', 'net sales', 'sales'], take=1)
    rev = rev_vals[0] if rev_vals else None
    return safe_div(market_cap, rev)


def compute_net_debt_ev(market_cap: Optional[float], tk: yf.Ticker) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # Returns (net_debt, ev, net_debt_to_ev)
    if market_cap is None or market_cap <= 0:
        return None, None, None
    bs = _get_stmt(tk, 'balance_sheet')
    debt = _get_last_row_value(bs, ['total debt', 'short long term debt total', 'long term debt', 'totaldebt', 'debt'])
    cash = _get_last_row_value(bs, ['cash', 'cash and cash equivalents', 'cashandcashequivalentsatcarryingvalue'])
    # Fallback to quarterly balance sheet if needed
    if (debt is None or cash is None):
        bs_q = _get_stmt(tk, 'quarterly_balance_sheet')
        if debt is None:
            debt = _get_last_row_value(bs_q, ['total debt', 'short long term debt total', 'long term debt', 'totaldebt', 'debt'])
        if cash is None:
            cash = _get_last_row_value(bs_q, ['cash', 'cash and cash equivalents', 'cashandcashequivalentsatcarryingvalue'])
    net_debt = None
    if debt is not None and cash is not None:
        net_debt = float(debt) - float(cash)
    elif debt is not None:
        net_debt = float(debt)
    elif cash is not None:
        net_debt = -float(cash)
    if net_debt is None:
        return None, None, None
    ev = float(market_cap) + float(net_debt)
    nd_ev = safe_div(net_debt, ev) if ev is not None and ev != 0 else None
    return net_debt, ev, nd_ev


def compute_dilution_rate(tk: yf.Ticker) -> Optional[float]:
    """Approximate shares CAGR over ~3y using balance sheet share count rows when available.
    Fallbacks:
    - Try quarterly balance sheet if annual missing.
    - As a last resort, if we only have a single point (e.g., info['sharesOutstanding']), return None (avoid false signal).
    """
    bs = _get_stmt(tk, 'balance_sheet')
    vals = _get_series_values(bs, ['ordinary shares number', 'common stock shares outstanding', 'shareissued', 'common shares', 'weighted average shares outstanding'])
    # Fallback: quarterly balance sheet
    if (not vals or len(vals) < 2):
        bs_q = _get_stmt(tk, 'quarterly_balance_sheet')
        q_vals = _get_series_values(bs_q, ['ordinary shares number', 'common stock shares outstanding', 'shareissued', 'common shares', 'weighted average shares outstanding'])
        if q_vals and len(q_vals) >= 2:
            vals = q_vals
    # If still not enough points, try info snapshot just to check non-zero; cannot compute rate though
    if not vals or len(vals) < 2:
        try:
            info = tk.info or {}
            so = info.get('sharesOutstanding') or info.get('shares_outstanding')
            _ = float(so) if so is not None else None
        except Exception:
            _ = None
        return None
    # vals are most-recent first; pick far and near
    try:
        recent = float(vals[0])
        past = float(vals[min(3, len(vals)-1)])
        if recent <= 0 or past <= 0:
            return None
        years = 3.0
        rate = (recent / past) ** (1.0 / years) - 1.0
        return rate
    except Exception:
        return None


def price_vol_and_drawdown(tk: yf.Ticker, fast: bool = False) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    # Returns (ann_vol, max_drawdown, sma200_regime_above)
    try:
        # Prefer cached price history if available
        df = None
        if load_price_history is not None:
            years = 1 if fast else 3
            try:
                df = load_price_history(tk.ticker if hasattr(tk, 'ticker') else None, years=years)
            except Exception:
                df = None
        if df is None or df.empty or 'Close' not in df.columns:
            period = '1y' if fast else '3y'
            hist = tk.history(period=period, interval='1d', auto_adjust=True)
            if hist is None or hist.empty or 'Close' not in hist:
                return None, None, None
            df = hist.reset_index().rename(columns={'index':'Date'})
        close = pd.Series(df['Close']).dropna()
        if len(close) < 60:
            return None, None, None
        # Robust volatility: winsorize returns at 5th/95th percentiles
        rets = close.pct_change().dropna()
        if len(rets) < 20:
            return None, None, None
        q_lo, q_hi = np.nanpercentile(rets.values, [5, 95])
        rets_w = rets.clip(lower=q_lo, upper=q_hi)
        ann_vol = float(np.nanstd(rets_w)) * (252.0 ** 0.5)
        # Max drawdown
        vals = close.values
        roll_max = np.maximum.accumulate(vals)
        dd = (vals - roll_max) / roll_max
        max_dd = float(np.nanmin(dd)) if len(dd) else None
        if max_dd is not None:
            max_dd = abs(max_dd)
        # 200d SMA regime
        sma_window = 200 if not fast else 100
        sma = close.rolling(sma_window).mean()
        regime = None
        try:
            regime = 1 if float(close.iloc[-1]) >= float(sma.iloc[-1]) else 0
        except Exception:
            regime = None
        return ann_vol, max_dd, regime
    except Exception:
        return None, None, None


def compute_breakout_score(tk: yf.Ticker, fast: bool = False) -> Optional[float]:
    """Return a conservative upside breakout score in [0,1].
    Ingredients:
      - 55-day Donchian upper-band breakout using High (or Close fallback), prior-high (shifted)
      - 20-day return filter (recent momentum)
      - 100/200-day SMA alignment (trend filter)
      - Volume confirmation via 20-day z-score if Volume available
    Returns None if insufficient data or features cannot be computed.
    """
    try:
        # Prefer cache if available
        df = None
        if load_price_history is not None:
            years = 1 if fast else 2
            try:
                df = load_price_history(tk.ticker if hasattr(tk, 'ticker') else None, years=years)
            except Exception:
                df = None
        if df is None or df.empty or 'Close' not in df.columns:
            period = '1y' if fast else '2y'
            hist = tk.history(period=period, interval='1d', auto_adjust=True)
            if hist is None or hist.empty or 'Close' not in hist:
                return None
            df = hist.reset_index().rename(columns={'index':'Date'})
        close = pd.Series(df['Close']).dropna()
        if len(close) < 60:
            return None
        high = pd.Series(df['High']) if 'High' in df.columns else close
        vol = pd.Series(df['Volume']) if 'Volume' in df.columns else None
        # Donchian 55-day prior high
        win = 55 if not fast else 34
        upper = high.rolling(win, min_periods=win).max().shift(1)
        if upper.isna().all() or pd.isna(upper.iloc[-1]):
            return None
        last = float(close.iloc[-1])
        up = float(upper.iloc[-1])
        if not np.isfinite(last) or not np.isfinite(up) or up <= 0:
            return None
        # Breakout distance
        dist = (last - up) / up
        # 20-day momentum
        ret20 = None
        if len(close) >= 22 and close.iloc[-21] > 0:
            ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
        # SMA alignment
        sma100 = close.rolling(100 if not fast else 60).mean()
        sma200 = close.rolling(200 if not fast else 120).mean()
        align = None
        try:
            align = 1 if (float(last) > float(sma100.iloc[-1]) and float(sma100.iloc[-1]) >= float(sma200.iloc[-1])) else 0
        except Exception:
            align = None
        # Volume z-score
        vol_z = None
        if vol is not None and isinstance(vol, pd.Series):
            vwin = 20
            if len(vol.dropna()) >= vwin + 1:
                vmean = vol.rolling(vwin).mean()
                vstd = vol.rolling(vwin).std()
                try:
                    denom = vstd.iloc[-1]
                    vol_z = float((vol.iloc[-1] - vmean.iloc[-1]) / denom) if denom and denom != 0 else None
                except Exception:
                    vol_z = None
        # Map to base score
        base = 0.0
        if dist > 0:
            s_dist = _piecewise_score(dist, [(0.0, 0.55), (0.02, 0.7), (0.05, 0.85), (0.10, 0.95), (0.20, 1.0)])
            base += s_dist if s_dist is not None else 0.6
        else:
            near = _piecewise_score(dist, [(-0.03, 0.0), (-0.01, 0.1), (0.0, 0.2)])
            base += near if near is not None else 0.0
        if ret20 is not None:
            mom = _piecewise_score(ret20, [(-0.05, 0.0), (0.0, 0.2), (0.05, 0.6), (0.10, 0.85), (0.20, 1.0)])
            if mom is not None:
                base = 0.7 * base + 0.3 * mom
        if align is not None:
            if align == 1:
                base = clamp((base or 0.0) + 0.1, 0.0, 1.0) or base
            else:
                base = clamp((base or 0.0) * 0.85, 0.0, 1.0) or base
        if vol_z is not None and np.isfinite(vol_z):
            mult = _piecewise_score(vol_z, [(-1.0, 0.9), (0.0, 1.0), (1.0, 1.05), (2.0, 1.12), (3.0, 1.18)])
            if mult is not None:
                base = clamp(base * float(mult), 0.0, 1.0) or base
        score = 0.5 * base + 0.5 * logistic(base, k=3.5, x0=0.55)
        # Additional short-horizon drawdown safety
        try:
            w = 63
            if len(close) > w:
                window = close.iloc[-w:]
                maxp = float(np.nanmax(window.values))
                if maxp > 0:
                    dd3m = 1.0 - (float(window.iloc[-1]) / maxp)
                    if dd3m > 0.4:
                        score *= 0.8
        except Exception:
            pass
        return float(clamp(score, 0.0, 1.0))
    except Exception:
        return None

def _piecewise_score(x: Optional[float], points: List[Tuple[float, float]]) -> Optional[float]:
    """Map x to [0,1] by linear interpolation between sorted (x, y) points.
    points: list of (x_value, y_score) tuples sorted by x.
    Returns None if x is None.
    """
    if x is None:
        return None
    try:
        x = float(x)
        pts = sorted(points, key=lambda p: p[0])
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
                return y0 + t * (y1 - y0)
        return pts[-1][1]
    except Exception:
        return None


def compute_bagger_subscores(
    tk: yf.Ticker,
    market_cap: Optional[float],
    rev_cagr_3y: Optional[float],
    rev_recent: Optional[float],
    rev_series: Optional[List[float]],
    args
) -> Dict[str, Optional[float]]:
    """Compute sub-scores for the 100× bagger model.
    This function is intentionally defensive: it never raises and always returns
    a dict with the expected keys, using None for missing components.
    """
    # Initialize defaults so we can always return a dict even on partial failures
    gm = om = fcf_margin = None
    ps = evs = ann_vol = max_dd = dil_rate = None
    growth_raw = quality_raw = discipline_raw = val_raw = risk_raw = None
    r2_stability = None
    size_bonus = None

    try:
        # Horizon-sensitive growth target
        T = int(getattr(args, 'bagger_horizon', 20))
        r_star = (100.0 ** (1.0 / T)) - 1.0  # required CAGR

        # Base-revenue floor for guarding tiny denominators
        base_rev_floor = float(getattr(args, 'bagger_base_rev_floor', 5e7))  # default $50M
        rev_base = rev_series[3] if rev_series and len(rev_series) >= 4 else None

        # Growth: map c/r* ratio to 0..1 via piecewise and logistic; blend with LTM momentum and 5Y CAGR; apply base-revenue scaling
        if rev_cagr_3y is not None:
            try:
                ratio = rev_cagr_3y / max(1e-9, r_star)
                cagr3_score = _piecewise_score(ratio, [(0.0,0.0),(0.5,0.3),(1.0,0.7),(1.5,0.9),(2.0,1.0)])
                if cagr3_score is not None:
                    cagr3_score = 0.5 * cagr3_score + 0.5 * logistic(cagr3_score, k=4.0, x0=0.6)
                # 5Y CAGR if available
                cagr5_score = None
                try:
                    if rev_series and len(rev_series) >= 6 and rev_series[5] > 0 and rev_series[0] > 0:
                        c5 = (rev_series[0] / rev_series[5]) ** (1.0/5.0) - 1.0
                        ratio5 = c5 / max(1e-9, r_star)
                        cagr5_score = _piecewise_score(ratio5, [(0.0,0.0),(0.5,0.25),(1.0,0.6),(1.5,0.85),(2.0,0.97)])
                        if cagr5_score is not None:
                            cagr5_score = 0.5 * cagr5_score + 0.5 * logistic(cagr5_score, k=3.5, x0=0.55)
                except Exception:
                    cagr5_score = None
                # LTM momentum and trend
                ltm_cur, ltm_prev = compute_ltm_momentum_pair(tk)
                ltm_mom = ltm_cur
                ltm_trend_bonus = None
                # Map LTM momentum relative to a softer target (half of r_star) to 0..1
                ltm_score = None
                if ltm_mom is not None:
                    mom_ratio = ltm_mom / max(1e-9, (0.5 * r_star))
                    ltm_score = _piecewise_score(mom_ratio, [(0.0,0.0),(0.5,0.35),(1.0,0.7),(1.5,0.9),(2.0,1.0)])
                    if ltm_score is not None:
                        ltm_score = 0.4 * ltm_score + 0.6 * logistic(ltm_score, k=4.0, x0=0.6)
                if ltm_cur is not None and ltm_prev is not None:
                    # bonus 0..0.1 when momentum improving
                    delta = ltm_cur - ltm_prev
                    ltm_trend_bonus = _piecewise_score(delta, [(-0.5,0.0), (0.0,0.0), (0.2,0.05), (0.5,0.1)])
                # Blend components
                parts = []
                if cagr3_score is not None:
                    parts.append((cagr3_score, 0.5))
                if cagr5_score is not None:
                    parts.append((cagr5_score, 0.2))
                if ltm_score is not None:
                    parts.append((ltm_score, 0.3))
                if parts:
                    num = sum(v*w for v, w in parts)
                    den = sum(w for _, w in parts)
                    growth_raw = num / den if den > 0 else None
                    if ltm_trend_bonus is not None and growth_raw is not None:
                        growth_raw = clamp(growth_raw + 0.1 * float(ltm_trend_bonus), 0.0, 1.0)
                # Apply base revenue cap to mitigate tiny denominators
                if growth_raw is not None and rev_base is not None:
                    growth_cap = _piecewise_score(rev_base, [(0.0,0.4),(2e7,0.6),(5e7,0.8),(1e8,1.0)])
                    if growth_cap is not None:
                        growth_raw = growth_raw * float(growth_cap)
                # Save ltm_momentum into locals for export later
                ltm_momentum_local = ltm_mom
            except Exception:
                growth_raw = None
                ltm_momentum_local = None
        else:
            ltm_momentum_local = None

        # Growth stability (R^2 on log revenue vs time using available points)
        try:
            series = rev_series if rev_series else None
            if series and len(series) >= 4 and all(v is not None and v > 0 for v in series[:4]):
                y = np.log(series[:4])  # most-recent first
                x = np.arange(len(y), dtype=float)
                x = (x - np.mean(x)) / (np.std(x) + 1e-9)
                y_hat = np.poly1d(np.polyfit(x, y, 1))(x)
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-9
                r2 = 1.0 - ss_res / ss_tot
                r2_stability = max(0.0, min(1.0, r2))
        except Exception:
            r2_stability = None

        # Quality (margins) and FCF margin
        try:
            gm, om = compute_margins(tk)
        except Exception:
            gm, om = None, None
        try:
            gm_score = _piecewise_score(gm, [(-0.2,0.0),(0.0,0.2),(0.3,0.6),(0.5,0.85),(0.7,1.0)]) if gm is not None else None
            om_score = _piecewise_score(om, [(-0.5,0.0),(0.0,0.3),(0.1,0.6),(0.2,0.8),(0.3,0.95)]) if om is not None else None
        except Exception:
            gm_score = om_score = None
        try:
            fcf_margin = compute_fcf_margin(tk)
            fcf_score = _piecewise_score(fcf_margin, [(-0.5,0.0),(0.0,0.4),(0.1,0.7),(0.2,0.9),(0.3,1.0)]) if fcf_margin is not None else None
        except Exception:
            fcf_margin = None
            fcf_score = None
        try:
            comps = []
            if gm_score is not None:
                comps.append((gm_score, 0.4))
            if fcf_score is not None:
                comps.append((fcf_score, 0.4))
            if om_score is not None:
                comps.append((om_score, 0.2))
            if comps:
                num = sum(v*w for v, w in comps)
                den = sum(w for _, w in comps)
                quality_raw = num / den if den > 0 else None
            else:
                # Neutral fallback to avoid NaN quality when data is sparse
                quality_raw = 0.5
        except Exception:
            quality_raw = 0.5

        # Sector (for valuation normalization)
        try:
            info = get_info_cached(tk, ttl_days=getattr(args, 'info_ttl_days', 1.0))
            sector = str(info.get('sector', 'UNKNOWN')).strip().upper() if isinstance(info, dict) else 'UNKNOWN'
            if not sector:
                sector = 'UNKNOWN'
        except Exception:
            sector = 'UNKNOWN'

        # Discipline (net debt/EV, dilution)
        ev = None
        try:
            _, ev, nd_ev = compute_net_debt_ev(market_cap, tk)
            nd_score = None
            if nd_ev is not None:
                nd_score = _piecewise_score(nd_ev, [(-0.5,1.0),(0.0,0.9),(0.2,0.7),(0.4,0.4),(0.6,0.2),(0.8,0.0)])
        except Exception:
            nd_score = None
        try:
            dil_rate = compute_dilution_rate(tk)
            dil_score = None
            if dil_rate is not None:
                dil_score = _piecewise_score(dil_rate, [(-0.1,1.0),(0.0,0.9),(0.05,0.7),(0.1,0.4),(0.2,0.2),(0.3,0.0)])
        except Exception:
            dil_rate = None
            dil_score = None
        try:
            if nd_score is not None or dil_score is not None:
                nd = nd_score if nd_score is not None else 0.6
                dl = dil_score if dil_score is not None else 0.6
                discipline_raw = 0.6 * nd + 0.4 * dl
            else:
                # Neutral fallback to avoid NaN discipline when data is sparse
                discipline_raw = 0.5
        except Exception:
            discipline_raw = 0.5

        # Valuation (prefer EV/S when EV available; else P/S)
        try:
            ps = compute_ps_ratio(market_cap, tk)
        except Exception:
            ps = None
        # Fallback P/S from recent revenue we already computed
        try:
            if (ps is None or not np.isfinite(ps)) and market_cap is not None and rev_recent is not None and rev_recent > 0:
                ps = safe_div(market_cap, rev_recent)
        except Exception:
            pass
        try:
            if ev is not None and rev_recent is not None and rev_recent > 0:
                evs = safe_div(ev, rev_recent)
        except Exception:
            evs = None
        # Clamp extreme multiples to reduce outlier impact
        try:
            if evs is not None and np.isfinite(evs):
                evs = max(0.01, min(100.0, float(evs)))
            if ps is not None and np.isfinite(ps):
                ps = max(0.01, min(100.0, float(ps)))
        except Exception:
            pass
        try:
            if rev_cagr_3y is not None:
                target_ps = min(25.0, 1.0 + 40.0 * max(0.0, min(0.5, rev_cagr_3y)))
                metric = evs if evs is not None else ps
                if metric is not None:
                    # Growth-adjusted multiple: penalize high EV/S more when growth is low
                    gadj = float(metric) / (1.0 + max(0.0, float(rev_cagr_3y)))
                    gadj = max(0.01, min(100.0, gadj))
                    val_raw = _piecewise_score(gadj, [
                        (0.5,1.0), (target_ps,0.8), (target_ps*1.5,0.5), (target_ps*2.0,0.2), (target_ps*3.0,0.0)
                    ])
        except Exception:
            val_raw = None

        # Risk (volatility and drawdown)
        try:
            fast_flag = getattr(args, 'fast_bagger', False)
            ann_vol, max_dd, regime = price_vol_and_drawdown(tk, fast=fast_flag)
            vol_score = _piecewise_score(ann_vol, [(0.1,1.0),(0.2,0.9),(0.4,0.6),(0.6,0.3),(0.8,0.1),(1.0,0.0)]) if ann_vol is not None else None
            dd_score = _piecewise_score(max_dd, [(0.1,1.0),(0.2,0.8),(0.4,0.5),(0.6,0.2),(0.8,0.05),(0.9,0.0)]) if max_dd is not None else None
            if vol_score is not None or dd_score is not None:
                v = vol_score if vol_score is not None else 0.6
                d = dd_score if dd_score is not None else 0.6
                risk_raw = 0.5 * v + 0.5 * d
                # Regime penalty if below 200d SMA
                if regime is not None and regime == 0:
                    risk_raw = clamp((risk_raw or 0.5) * 0.9, 0.0, 1.0)
        except Exception:
            ann_vol, max_dd, regime = None, None, None
            risk_raw = None

        # Technical breakout (optional)
        breakout_score = None
        try:
            if not getattr(args, 'no_breakout', False):
                breakout_score = compute_breakout_score(tk, fast=getattr(args, 'fast_bagger', False))
                # If risk is very poor, dampen breakout to avoid chasing noisy spikes
                if breakout_score is not None and (max_dd is not None and max_dd > 0.7):
                    breakout_score = float(clamp(breakout_score * 0.8, 0.0, 1.0))
        except Exception:
            breakout_score = None

        # Size prior bonus
        try:
            if market_cap is not None:
                size_bonus = _piecewise_score(market_cap, [
                    (1e7, 0.08), (5e7, 0.06), (1e8, 0.04), (3e8, 0.02), (1e9, 0.01), (3e9, 0.0), (1e10, -0.01)
                ])
        except Exception:
            size_bonus = None

    except Exception:
        # If anything above unexpectedly raised, fall back to minimal context
        T = int(getattr(args, 'bagger_horizon', 20))
        r_star = (100.0 ** (1.0 / T)) - 1.0
        base_rev_floor = float(getattr(args, 'bagger_base_rev_floor', 5e7))
        rev_base = rev_series[3] if rev_series and len(rev_series) >= 4 else None

    return {
        'growth': growth_raw,
        'quality': quality_raw,
        'discipline': discipline_raw,
        'valuation': val_raw,
        'risk': risk_raw,
        'gm': gm,
        'om': om,
        'ps': ps,
        'evs': evs,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'dilution_rate': dil_rate,
        'r_star': r_star,
        'r2_stability': r2_stability,
        'fcf_margin': fcf_margin,
        'size_bonus': size_bonus,
        'rev_base': rev_base,
        'base_rev_floor': base_rev_floor,
        'market_cap': market_cap,
        'ltm_momentum': ltm_momentum_local,
        'regime': regime,
        'breakout': breakout_score,
    }


def compute_bagger_score(sub: Dict[str, Optional[float]], args=None) -> Tuple[Optional[int], Optional[float]]:
    # Combine subscores with weights and guardrails; return (score0_100, composite_S)
    if not sub:
        return None, None
    # Red flags
    gm = sub.get('gm')
    max_dd = sub.get('max_dd')
    rev_cagr = sub.get('rev_cagr_3y')
    rev_base = sub.get('rev_base')
    base_rev_floor = float(getattr(args, 'bagger_base_rev_floor', 5e7)) if args is not None else 5e7

    hard_cap = None
    if gm is not None and gm < 0.0:
        hard_cap = 20
    if max_dd is not None and max_dd > 0.85:
        hard_cap = 10 if hard_cap is None else min(hard_cap, 10)
    if rev_cagr is not None and rev_cagr < 0.0:
        hard_cap = 25 if hard_cap is None else min(hard_cap, 25)
    if rev_base is not None and rev_base < base_rev_floor and rev_cagr is not None and rev_cagr > 0.8:
        # Tiny starting revenue with extreme CAGR: cap aggressively
        hard_cap = 30 if hard_cap is None else min(hard_cap, 30)

    # Adjust growth by stability if available
    growth = sub.get('growth')
    r2 = sub.get('r2_stability')
    if growth is not None and r2 is not None:
        growth = 0.8 * float(growth) + 0.2 * float(r2)
    # Compose discipline etc.
    components = {
        'growth': growth if growth is not None else sub.get('growth'),
        'quality': sub.get('quality'),
        'discipline': sub.get('discipline'),
        'valuation': sub.get('valuation'),
        'risk': sub.get('risk'),
    }
    weights = {
        'growth': 0.35,
        'quality': 0.20,
        'discipline': 0.15,
        'valuation': 0.15,
        'risk': 0.15,
    }
    S = 0.0
    wsum = 0.0
    for k, w in weights.items():
        v = components.get(k)
        if v is not None:
            S += w * float(v)
            wsum += w
    if wsum == 0.0:
        return None, None
    S = S / wsum

    # Integrate breakout as a modest, gated boost
    try:
        bw = float(getattr(args, 'bagger_breakout_weight', 0.1)) if args is not None else 0.1
        breakout = sub.get('breakout')
        if bw > 0.0 and breakout is not None:
            gate = 1.0
            regime = sub.get('regime')
            if regime is not None and int(regime) == 0:
                gate *= 0.5
            risk = sub.get('risk')
            if risk is not None:
                gate *= (0.5 + 0.5 * float(clamp(risk, 0.0, 1.0)))
            growth_for_gate = growth if growth is not None else sub.get('growth')
            if growth_for_gate is not None:
                gate *= (0.3 + 0.7 * float(clamp(growth_for_gate, 0.0, 1.0)))
            S = float(clamp(S + bw * float(breakout) * gate, 0.0, 1.0))
    except Exception:
        pass

    # Size prior bonus (very small caps slightly favored), very small effect
    size_bonus = sub.get('size_bonus')
    if size_bonus is not None:
        S = float(clamp(S + 0.1 * float(size_bonus), 0.0, 1.0))

    # Probability-like mapping with conservative prior blending
    k = 3.0
    x0 = 0.55  # slightly right-shifted to be conservative
    P = logistic(S, k=k, x0=x0)
    # Blend with Bayesian prior (basis points)
    prior_bp = float(getattr(args, 'bagger_prior_bp', 1.0)) if args is not None else 1.0
    p0 = max(0.0, min(1.0, prior_bp / 10000.0))  # bp -> fraction
    alpha = 0.9  # model confidence weight
    P_final = alpha * P + (1 - alpha) * p0

    score = int(round(100.0 * P_final))
    if hard_cap is not None:
        score = min(score, hard_cap)
    return score, S


# ═══════════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKER - Module-level function for Pool
# ═══════════════════════════════════════════════════════════════════════════════

def _process_ticker_worker(args_tuple: Tuple) -> Tuple:
    """Worker function for multiprocessing Pool. Processes a single ticker.
    
    Args:
        args_tuple: (ticker_sym, min_mkt_cap, max_mkt_cap, no_bagger, bagger_horizon, 
                     no_breakout, fast_bagger, bagger_breakout_weight, bagger_prior_bp,
                     bagger_base_rev_floor, info_ttl_days, export_subscores)
    
    Returns:
        Tuple of ('ok'|'fail'|'skip_cap_low'|'skip_cap_high', data, debug_info)
    """
    (ticker_sym, min_mkt_cap, max_mkt_cap, no_bagger, bagger_horizon,
     no_breakout, fast_bagger, bagger_breakout_weight, bagger_prior_bp,
     bagger_base_rev_floor, info_ttl_days, export_subscores) = args_tuple
    
    # Create a simple args-like object for compatibility
    class ArgsProxy:
        pass
    args = ArgsProxy()
    args.min_mkt_cap = min_mkt_cap
    args.max_mkt_cap = max_mkt_cap
    args.no_bagger = no_bagger
    args.bagger_horizon = bagger_horizon
    args.no_breakout = no_breakout
    args.fast_bagger = fast_bagger
    args.bagger_breakout_weight = bagger_breakout_weight
    args.bagger_prior_bp = bagger_prior_bp
    args.bagger_base_rev_floor = bagger_base_rev_floor
    args.info_ttl_days = info_ttl_days
    args.export_subscores = export_subscores
    
    try:
        tk = yf.Ticker(ticker_sym)
        mktcap = get_market_cap(tk)
        if mktcap is None:
            return ('fail', ticker_sym, "no market cap")
        if mktcap < min_mkt_cap:
            return ('skip_cap_low', ticker_sym, None)
        if mktcap > max_mkt_cap:
            return ('skip_cap_high', ticker_sym, None)

        rev_vals = get_revenue_values(tk)
        if not rev_vals or len(rev_vals) < 2:
            return ('fail', ticker_sym, "insufficient revenue history")

        past_index = min(3, len(rev_vals) - 1)
        recent_rev = rev_vals[0]
        past_rev = rev_vals[past_index]
        effective_years = max(1, past_index)
        c = cagr(past_rev, recent_rev, years=effective_years)
        if c is None or (isinstance(c, float) and math.isnan(c)):
            return ('fail', ticker_sym, "cagr failed")

        bagger_score = None
        bagger_S = None
        subs = None
        if not no_bagger:
            try:
                subs = compute_bagger_subscores(tk, mktcap, c, recent_rev, rev_vals, args)
                if subs is not None:
                    subs['rev_cagr_3y'] = c
                bagger_score, bagger_S = compute_bagger_score(subs, args)
            except Exception:
                subs = None
                bagger_score = None
                bagger_S = None
            if bagger_score is None:
                try:
                    T = int(bagger_horizon)
                    r_star = (100.0 ** (1.0 / T)) - 1.0
                    ratio = c / max(1e-9, r_star) if c is not None else None
                    if ratio is not None:
                        base = _piecewise_score(ratio, [(0.0,0.0),(0.5,0.25),(1.0,0.6),(1.5,0.85),(2.0,0.97),(3.0,0.995)])
                        if base is not None:
                            p = 0.5 * base + 0.5 * logistic(base, k=3.0, x0=0.55)
                            bagger_score = int(round(100.0 * clamp(p, 0.0, 0.999)))
                            bagger_S = float(base)
                except Exception:
                    pass

        row = {
            "ticker": ticker_sym,
            "marketCap": mktcap,
            "rev_3yr_ago": past_rev,
            "rev_recent": recent_rev,
            "rev_3yr_cagr": c,
            "rev_cagr_years": effective_years,
            "tradingview": f"https://www.tradingview.com/chart/?symbol={ticker_sym}",
            "bagger_horizon": bagger_horizon if not no_bagger else None,
            "bagger_score": bagger_score,
            "bagger_S": bagger_S,
        }
        
        if subs is None:
            subs = {}
        row.update({
            "_sub_growth": subs.get("growth"),
            "_sub_quality": subs.get("quality"),
            "_sub_discipline": subs.get("discipline"),
            "_sub_valuation": subs.get("valuation"),
            "_sub_risk": subs.get("risk"),
            "ps": subs.get("ps"),
            "evs": subs.get("evs"),
            "gm": subs.get("gm"),
            "om": subs.get("om"),
            "ann_vol": subs.get("ann_vol"),
            "max_dd": subs.get("max_dd"),
            "dilution_rate": subs.get("dilution_rate"),
            "r_star": subs.get("r_star"),
            "r2_stability": subs.get("r2_stability"),
            "fcf_margin": subs.get("fcf_margin"),
            "size_bonus": subs.get("size_bonus"),
            "rev_base": subs.get("rev_base"),
            "ltm_momentum": subs.get("ltm_momentum"),
            "breakout": subs.get("breakout"),
            "regime": subs.get("regime"),
        })
        
        try:
            info = get_info_cached(tk, ttl_days=info_ttl_days)
            row["sector"] = str(info.get('sector', 'UNKNOWN')).strip().upper()
        except Exception:
            row["sector"] = 'UNKNOWN'

        if export_subscores:
            row.update({
                "sub_growth": row.get("_sub_growth"),
                "sub_quality": row.get("_sub_quality"),
                "sub_discipline": row.get("_sub_discipline"),
                "sub_valuation": row.get("_sub_valuation"),
                "sub_risk": row.get("_sub_risk"),
            })

        return ('ok', row, {})
    except Exception as e:
        return ('fail', ticker_sym, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Top 100 Stock Screener — Find high-growth small/mid caps by revenue CAGR with optional 100x bagger scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  make top100                           # Run with defaults (top 100 by revenue CAGR)
  make top100 ARGS="--sort_by bagger"   # Sort by 100× Bagger Score instead
  make top100 ARGS="--top_n 50"         # Output only top 50
  make top100 ARGS="--plain"            # Plain text output (no colors)
        """
    )
    parser.add_argument('--universe', choices=['csv', 'russell_2500'], default=DEFAULT_UNIVERSE,
                        help="Universe source. 'csv' reads from --csv file; 'russell_2500' tries Wikipedia (needs lxml).")
    parser.add_argument(
        '--csv',
        dest='csv_path',
        default=DEFAULT_CSV_PATH,
        help=f"Path to tickers CSV (default: {DEFAULT_CSV_PATH})"
    )
    parser.add_argument('--min_mkt_cap', type=float, default=MIN_MKT_CAP, help="Min market cap in USD")
    parser.add_argument('--max_mkt_cap', type=float, default=MAX_MKT_CAP, help="Max market cap in USD")
    parser.add_argument('--top_n', type=int, default=TOP_N, help="Number of rows to output")
    parser.add_argument('--min_universe_size', type=int, default=MIN_UNIVERSE_SIZE,
                        help="Minimum number of tickers required in the universe before running (default 2400)")
    parser.add_argument('--plain', action='store_true', help='Plain text output (disable rich tables)')
    parser.add_argument('--show_failures', action='store_true', help='Show sample list of failed/skipped tickers')
    # 100x bagger scoring options
    parser.add_argument('--no_bagger', action='store_true', help='Disable 100x bagger scoring for speed')
    parser.add_argument('--bagger_horizon', type=int, choices=[5,10,15,20,25,30], default=5,
                        help='Horizon in years for the 100x target (affects growth thresholds). Default: 5')
    parser.add_argument('--bagger_verbose', action='store_true', help='Show sub-score breakdown in output table')
    parser.add_argument('--no_breakout', action='store_true', help='Disable breakout-aware component in bagger scoring')
    parser.add_argument('--bagger_breakout_weight', type=float, default=0.1, help='Weight of breakout component added to composite S (default 0.1)')
    parser.add_argument('--fast_bagger', action='store_true', help='Skip heavy computations (drawdown/EVS) for speed')
    parser.add_argument('--bagger_prior_bp', type=float, default=1.0,
                        help='Bayesian prior for 100x probability in basis points (default: 1 bp = 0.01%)')
    parser.add_argument('--bagger_base_rev_floor', type=float, default=5e7,
                        help='Base revenue floor used to downweight extreme growth from tiny base (default: 50,000,000)')
    parser.add_argument('--export_subscores', action='store_true',
                        help='Include sub-score and diagnostic columns in CSV output')
    parser.add_argument('--sort_by', choices=['cagr','bagger'], default='cagr',
                        help='Sort output by rev CAGR (cagr) or 100x bagger score (bagger). Default: cagr')
    parser.add_argument('--debug_bagger', type=int, default=0,
                        help='Debug mode: process only N tickers and print factor/score distributions and a small per-ticker breakdown')
    parser.add_argument('--no_sector_norm', action='store_true', help='Disable sector-aware valuation normalization')
    parser.add_argument('--info_ttl_days', type=float, default=1.0, help='TTL in days for cached Ticker.info snapshots (default 1)')
    # Parallelism controls
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel worker threads to use for per-ticker processing (default: auto; 0=auto).')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing and run single-threaded.')
    parser.add_argument('--rate_per_sec', type=float, default=0.0,
                        help='Optional global soft rate limit (requests/sec) for yfinance calls. 0 disables.')
    args = parser.parse_args()

    # Ensure parent directory exists for the CSV path so users can drop the file in place
    if args.universe == 'csv':
        csv_dir = os.path.dirname(args.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

    # Resolve universe
    if args.universe == 'csv':
        try:
            tickers = load_tickers_from_csv(args.csv_path)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        tickers = fetch_russell_2500_tickers()
        if len(tickers) <= 2:
            print("Notice: Russell fetch returned a tiny list. Use --universe csv --csv tickers.csv for a real universe.")

    # ═══════════════════════════════════════════════════════════════════════════════
    # STARTUP UX - Beautiful introduction
    # ═══════════════════════════════════════════════════════════════════════════════
    use_rich_startup = RICH_AVAILABLE and not args.plain
    
    if use_rich_startup:
        console = Console()
        console.clear()
        console.print()
        
        # Hero header
        header = Text()
        header.append("\n")
        header.append("T O P   1 0 0", style="bold bright_white")
        header.append("\n")
        header.append("Revenue Growth Screener", style="dim italic")
        header.append("\n")
        
        console.print(Panel(
            Align.center(header),
            border_style="bright_white",
            box=box.DOUBLE,
            padding=(1, 6),
        ))
        console.print()
        
        # Config bar
        config = Table.grid(padding=(0, 4))
        config.add_column(justify="center")
        config.add_column(justify="center")
        config.add_column(justify="center")
        config.add_column(justify="center")
        config.add_row(
            Text.assemble(("◉ ", "cyan"), (f"{len(tickers):,}", "bold white"), (" universe", "dim")),
            Text.assemble(("◎ ", "green"), (f"{human_money(args.min_mkt_cap)}–{human_money(args.max_mkt_cap)}", "bold white"), (" cap", "dim")),
            Text.assemble(("◉ ", "yellow"), (f"{args.top_n}", "bold white"), (" output", "dim")),
            Text.assemble(("◎ ", "magenta"), ("24h", "bold white"), (" cache", "dim")),
        )
        console.print(Align.center(config))
        console.print()
        console.print()
        
        # Data phase header
        phase = Text()
        phase.append("  DATA  ", style="bold black on white")
        phase.append("  Loading market data with smart caching", style="dim")
        console.print(phase)
        console.print()
    else:
        print(f"Got {len(tickers)} tickers in universe.")

    start_ts = time.time()
    rows: List[Dict[str, float]] = []
    failed: List[Tuple[str, str]] = []
    skipped_low = 0
    skipped_high = 0

    processed = 0
    debug_limit = int(getattr(args, 'debug_bagger', 0) or 0)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # BATCH PROCESSING WITH CACHING - Much faster than individual requests
    # ═══════════════════════════════════════════════════════════════════════════════
    
    CACHE_DIR = "src/quant/cache/top100"
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def get_cache_path(ticker: str) -> str:
        return os.path.join(CACHE_DIR, f"{ticker.upper()}.json")
    
    def load_from_cache(ticker: str, max_age_hours: float = 24.0) -> Optional[Dict]:
        """Load ticker data from cache if fresh enough."""
        path = get_cache_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            cached_ts = data.get('_cached_at', 0)
            age_hours = (time.time() - cached_ts) / 3600.0
            if age_hours > max_age_hours:
                return None
            return data
        except Exception:
            return None
    
    def save_to_cache(ticker: str, data: Dict) -> None:
        """Save ticker data to cache."""
        try:
            data['_cached_at'] = time.time()
            path = get_cache_path(ticker)
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception:
            pass
    
    use_rich_progress = RICH_AVAILABLE and not args.plain
    tick_list = tickers[:debug_limit] if debug_limit > 0 else tickers
    
    # Phase 1: Check cache
    cached_data: Dict[str, Dict] = {}
    tickers_to_fetch: List[str] = []
    
    if use_rich_progress:
        console_p = Console()
        console_p.print("    [dim]Checking cache...[/dim]")
    
    for t in tick_list:
        cached = load_from_cache(t, max_age_hours=24.0)
        if cached:
            cached_data[t] = cached
        else:
            tickers_to_fetch.append(t)
    
    if use_rich_progress:
        console_p.print(f"    [green]●[/green] [bold]{len(cached_data):,}[/bold] [dim]cached[/dim]  [yellow]●[/yellow] [bold]{len(tickers_to_fetch):,}[/bold] [dim]to fetch[/dim]")
        console_p.print()
    
    # Phase 2: Batch fetch uncached tickers
    if tickers_to_fetch:
        if use_rich_progress:
            console_p.print("    [dim]Fetching market data...[/dim]")
        
        BATCH_SIZE = 50
        fetched_info: Dict[str, Dict] = {}
        
        def fetch_ticker_info(ticker: str) -> Tuple[str, Optional[Dict]]:
            try:
                tk = yf.Ticker(ticker)
                info = tk.info or {}
                inc = None
                try:
                    inc_df = getattr(tk, 'income_stmt', None) or getattr(tk, 'financials', None)
                    if isinstance(inc_df, pd.DataFrame) and not inc_df.empty:
                        inc = inc_df.to_dict()
                except:
                    pass
                return (ticker, {'info': info, 'income': inc})
            except Exception:
                return (ticker, None)
        
        if use_rich_progress:
            from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
            
            with Progress(
                TextColumn("    "),
                SpinnerColumn(spinner_name="dots", style="white"),
                TextColumn("[dim]Fetching[/dim]"),
                BarColumn(bar_width=30, style="dim white", complete_style="white", finished_style="bright_white"),
                TaskProgressColumn(),
                TextColumn("[dim]•[/dim]"),
                TimeElapsedColumn(),
                console=Console(),
                transient=False,
            ) as progress:
                task = progress.add_task("Fetching", total=len(tickers_to_fetch))
                
                for i in range(0, len(tickers_to_fetch), BATCH_SIZE):
                    batch = tickers_to_fetch[i:i+BATCH_SIZE]
                    
                    with ThreadPoolExecutor(max_workers=8) as executor:
                        futures = {executor.submit(fetch_ticker_info, t): t for t in batch}
                        for future in as_completed(futures):
                            ticker, data = future.result()
                            if data:
                                fetched_info[ticker] = data
                                save_to_cache(ticker, data)
                            progress.advance(task)
                    
                    time.sleep(0.3)
        else:
            for i in range(0, len(tickers_to_fetch), BATCH_SIZE):
                batch = tickers_to_fetch[i:i+BATCH_SIZE]
                with ThreadPoolExecutor(max_workers=8) as executor:
                    for ticker, data in executor.map(lambda t: fetch_ticker_info(t), batch):
                        if data:
                            fetched_info[ticker] = data
                            save_to_cache(ticker, data)
                print(f"  Batch {i//BATCH_SIZE + 1}/{(len(tickers_to_fetch) + BATCH_SIZE - 1)//BATCH_SIZE}")
                time.sleep(0.3)
        
        cached_data.update(fetched_info)
    
    if use_rich_progress:
        console_p.print()
        console_p.print(f"    [green]●[/green] [bold]{len(cached_data):,}[/bold] [dim]total with data[/dim]")
        console_p.print()
        phase2 = Text()
        phase2.append("  SCORING  ", style="bold black on white")
        phase2.append("  Computing growth metrics", style="dim")
        console_p.print(phase2)
        console_p.print()
    
    # Phase 3: Process all tickers
    def process_from_cache(ticker: str, data: Dict) -> Tuple:
        try:
            info = data.get('info', {})
            income_data = data.get('income')
            
            mktcap = info.get('marketCap') or info.get('market_cap')
            if mktcap is None:
                shares = info.get('sharesOutstanding')
                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
                if shares and price:
                    mktcap = float(shares) * float(price)
            
            if mktcap is None:
                return ('fail', ticker, "no market cap")
            
            mktcap = float(mktcap)
            if mktcap < args.min_mkt_cap:
                return ('skip_cap_low', ticker, None)
            if mktcap > args.max_mkt_cap:
                return ('skip_cap_high', ticker, None)
            
            # Get revenue
            rev_vals = None
            if income_data:
                try:
                    inc_df = pd.DataFrame(income_data)
                    idx = [str(i).lower() for i in inc_df.index]
                    candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue']
                    if candidates:
                        row = inc_df.iloc[candidates[0]]
                        vals = []
                        for c in sorted(inc_df.columns, key=lambda x: str(x), reverse=True):
                            try:
                                val = row.get(c)
                                if pd.notna(val) and float(val) != 0:
                                    vals.append(float(val))
                            except:
                                pass
                        if len(vals) >= 2:
                            rev_vals = vals[:4]
                except:
                    pass
            
            # Fallback to info totalRevenue - use it as single point if we have nothing
            if not rev_vals:
                total_rev = info.get('totalRevenue') or info.get('revenue')
                if total_rev and float(total_rev) > 0:
                    # Try to get previous year revenue from revenueGrowth
                    rev_growth = info.get('revenueGrowth')
                    if rev_growth is not None and float(total_rev) > 0:
                        try:
                            # revenueGrowth is YoY growth rate, so prev = current / (1 + growth)
                            prev_rev = float(total_rev) / (1.0 + float(rev_growth))
                            rev_vals = [float(total_rev), prev_rev]
                        except:
                            pass
                    if not rev_vals:
                        # Just use total revenue - will fail CAGR but better than nothing
                        rev_vals = [float(total_rev)]
            
            if not rev_vals or len(rev_vals) < 2:
                return ('fail', ticker, "insufficient revenue")
            
            past_index = min(3, len(rev_vals) - 1)
            recent_rev = rev_vals[0]
            past_rev = rev_vals[past_index]
            effective_years = max(1, past_index)
            c = cagr(past_rev, recent_rev, years=effective_years)
            
            if c is None or (isinstance(c, float) and math.isnan(c)):
                return ('fail', ticker, "cagr failed")
            
            bagger_score = None
            bagger_S = None
            
            if not args.no_bagger:
                try:
                    T = int(args.bagger_horizon)
                    r_star = (100.0 ** (1.0 / T)) - 1.0
                    ratio = c / max(1e-9, r_star)
                    base = _piecewise_score(ratio, [(0.0,0.0),(0.5,0.25),(1.0,0.6),(1.5,0.85),(2.0,0.97),(3.0,0.995)])
                    if base is not None:
                        gm = info.get('grossMargins') or 0.0
                        om = info.get('operatingMargins') or 0.0
                        gm_score = _piecewise_score(gm, [(-0.2,0.0),(0.0,0.2),(0.3,0.6),(0.5,0.85),(0.7,1.0)]) or 0.5
                        om_score = _piecewise_score(om, [(-0.5,0.0),(0.0,0.3),(0.1,0.6),(0.2,0.8),(0.3,0.95)]) or 0.5
                        quality = 0.6 * gm_score + 0.4 * om_score
                        S = 0.5 * base + 0.3 * quality + 0.2 * 0.5
                        p = 0.5 * S + 0.5 * logistic(S, k=3.0, x0=0.55)
                        bagger_score = int(round(100.0 * clamp(p, 0.0, 0.999)))
                        bagger_S = float(S)
                except:
                    pass
            
            row = {
                "ticker": ticker,
                "marketCap": mktcap,
                "rev_3yr_ago": past_rev,
                "rev_recent": recent_rev,
                "rev_3yr_cagr": c,
                "rev_cagr_years": effective_years,
                "tradingview": f"https://www.tradingview.com/chart/?symbol={ticker}",
                "bagger_horizon": args.bagger_horizon if not args.no_bagger else None,
                "bagger_score": bagger_score,
                "bagger_S": bagger_S,
                "sector": str(info.get('sector', 'UNKNOWN')).strip().upper(),
            }
            return ('ok', row, {})
        except Exception as e:
            return ('fail', ticker, str(e))
    
    if use_rich_progress:
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn
        
        with Progress(
            TextColumn("    "),
            SpinnerColumn(spinner_name="dots", style="white"),
            TextColumn("[dim]Scoring[/dim]"),
            BarColumn(bar_width=30, style="dim white", complete_style="white", finished_style="bright_white"),
            TaskProgressColumn(),
            TextColumn("[dim]•[/dim]"),
            TimeElapsedColumn(),
            console=Console(),
            transient=False,
        ) as progress:
            task = progress.add_task("Processing", total=len(tick_list))
            
            for ticker in tick_list:
                data = cached_data.get(ticker, {})
                status = process_from_cache(ticker, data)
                
                if status[0] == 'ok':
                    _, row, _ = status
                    rows.append(row)
                    processed += 1
                elif status[0] == 'skip_cap_low':
                    skipped_low += 1
                elif status[0] == 'skip_cap_high':
                    skipped_high += 1
                else:
                    _, sym, reason = status
                    failed.append((sym, reason))
                progress.advance(task)
    else:
        for ticker in tqdm(tick_list, desc="Processing"):
            data = cached_data.get(ticker, {})
            status = process_from_cache(ticker, data)
            
            if status[0] == 'ok':
                _, row, _ = status
                rows.append(row)
                processed += 1
            elif status[0] == 'skip_cap_low':
                skipped_low += 1
            elif status[0] == 'skip_cap_high':
                skipped_high += 1
            else:
                _, sym, reason = status
                failed.append((sym, reason))
                        
    elapsed = time.time() - start_ts
    total = len(tickers)
    kept = len(rows)
    skipped = total - kept

    # Persist updated market cap cache for future runs (merge rows into cache file)
    try:
        _load_caps_cache_once()
        # Merge current successful caps into cache map
        for r in rows:
            try:
                sym = str(r.get('ticker','')).strip().upper()
                capv = r.get('marketCap', None)
                if sym and capv is not None and float(capv) > 0:
                    _CAPS_CACHE_MAP[sym] = float(capv)
            except Exception:
                pass
        # Write cache file
        os.makedirs(META_DIR, exist_ok=True)
        caps_rows = [(k, v) for k, v in _CAPS_CACHE_MAP.items() if v is not None and pd.notna(v) and float(v) > 0]
        if caps_rows:
            pd.DataFrame(caps_rows, columns=['ticker','marketCap']).to_csv(CAPS_CACHE_FILE, index=False)
    except Exception:
        pass

    # Aggregate failure reasons
    reason_counts: Dict[str, int] = {}
    for _, reason in failed:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if skipped_low:
        reason_counts["filtered by market cap (below min)"] = reason_counts.get("filtered by market cap (below min)", 0) + skipped_low
    if skipped_high:
        reason_counts["filtered by market cap (above max)"] = reason_counts.get("filtered by market cap (above max)", 0) + skipped_high

    df = pd.DataFrame(rows)

    # Sector-aware valuation normalization post-pass
    if not args.no_bagger and not args.no_sector_norm and not df.empty:
        try:
            # Use EV/S if available else P/S, per sector percentile (lower multiple → better score)
            metric = df["evs"].where(df["evs"].notna(), df["ps"])
            df["val_metric"] = metric
            # Compute per-sector percentiles robustly
            def sector_pct(g: pd.DataFrame) -> pd.Series:
                m = g["val_metric"].astype(float)
                # rank with pct; smaller multiples better (low rank → low pct). Convert to (1-pct)
                pct = m.rank(pct=True, method='average')
                return 1.0 - pct
            pct_series = df.groupby(df["sector"].fillna("UNKNOWN"), group_keys=False).apply(sector_pct)
            df["sector_val_pct"] = pct_series.values
            # Map percentile to a [0,1] valuation score; blend with existing _sub_valuation when present
            # Higher percentile (cheaper vs sector) → higher score
            val_sec_score = df["sector_val_pct"].clip(0,1).apply(lambda p: _piecewise_score(p, [(0.0,0.0),(0.25,0.4),(0.5,0.7),(0.75,0.9),(0.9,0.97),(1.0,1.0)]))
            df["_sub_valuation_sec"] = val_sec_score
            # Blend: 60% sector-relative, 40% global valuation when both exist; else whichever exists
            def blend_val(row):
                a = row.get("_sub_valuation_sec")
                b = row.get("_sub_valuation")
                try:
                    if pd.notna(a) and pd.notna(b):
                        return 0.6*float(a) + 0.4*float(b)
                    if pd.notna(a):
                        return float(a)
                    if pd.notna(b):
                        return float(b)
                except Exception:
                    pass
                return np.nan
            df["_sub_valuation_adj"] = df.apply(blend_val, axis=1)
            # Recompute bagger_S and bagger_score when we have an adjusted valuation component
            mask = df["_sub_valuation_adj"].notna()
            if mask.any():
                # Rebuild composite S partially: replace valuation component and re-map to score
                def recompute_score(row):
                    try:
                        components = {
                            'growth': row.get('_sub_growth', np.nan),
                            'quality': row.get('_sub_quality', np.nan),
                            'discipline': row.get('_sub_discipline', np.nan),
                            'valuation': row.get('_sub_valuation_adj', np.nan),
                            'risk': row.get('_sub_risk', np.nan),
                        }
                        weights = {'growth':0.35,'quality':0.20,'discipline':0.15,'valuation':0.15,'risk':0.15}
                        S = 0.0; wsum = 0.0
                        for k,w in weights.items():
                            v = components.get(k)
                            if pd.notna(v):
                                S += w*float(v); wsum += w
                        if wsum == 0:
                            return row.get('bagger_score'), row.get('bagger_S')
                        S = S/wsum
                        # size bonus
                        try:
                            size_bonus = row.get('size_bonus')
                            if pd.notna(size_bonus):
                                S = float(clamp(S + 0.1*float(size_bonus), 0.0, 1.0))
                        except Exception:
                            pass
                        P = logistic(S, k=3.0, x0=0.55)
                        prior_bp = float(getattr(args, 'bagger_prior_bp', 1.0))
                        p0 = max(0.0, min(1.0, prior_bp/10000.0))
                        P_final = 0.9*P + 0.1*p0
                        return int(round(100.0*P_final)), S
                    except Exception:
                        return row.get('bagger_score'), row.get('bagger_S')
                recomputed = df.apply(recompute_score, axis=1)
                df['bagger_score'] = [a for a,_ in recomputed]
                df['bagger_S'] = [b for _,b in recomputed]
        except Exception as _e:
            # Non-fatal; continue without sector normalization
            pass

    if df.empty:
        print("No rows collected. Tips: ensure your CSV has many small/mid tickers; consider widening cap bounds; check network access.")
        if args.show_failures and failed:
            print("Sample failures (up to 20):")
            for f in failed[:20]:
                print(f)
        sys.exit(0)

    df["rev_3yr_cagr_pct"] = df["rev_3yr_cagr"] * 100.0
    # Choose sort key
    sort_cols = ["rev_3yr_cagr_pct"]
    ascending = [False]
    title_metric = "3Y Revenue CAGR"
    if not args.no_bagger and "bagger_score" in df.columns and args.sort_by == 'bagger':
        sort_cols = ["bagger_score", "rev_3yr_cagr_pct"]
        ascending = [False, False]
        title_metric = "100× Bagger Score"
    df_sorted = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    top = df_sorted.head(args.top_n)

    out_csv = "top100_screener_results.csv"
    top.to_csv(out_csv, index=False)

    # ═══════════════════════════════════════════════════════════════════════════════════
    #  APPLE-QUALITY UX OUTPUT - Clean, Minimal, Powerful
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    use_rich = RICH_AVAILABLE and not args.plain

    if use_rich:
        console = Console()
        console.print()
        
        # ─────────────────────────────────────────────────────────────────────────────
        # HEADER - Clean Apple style
        # ─────────────────────────────────────────────────────────────────────────────
        console.print(Panel(
            Align.center(Text("TOP 100 SCREENER", style="bold white")),
            border_style="bright_white",
            padding=(0, 4),
            subtitle="[dim]Revenue Growth Analysis[/dim]"
        ))
        console.print()
        
        # ─────────────────────────────────────────────────────────────────────────────
        # KEY METRICS - Minimal data density, maximum clarity
        # ─────────────────────────────────────────────────────────────────────────────
        metrics_grid = Table.grid(padding=(0, 4))
        metrics_grid.add_column(justify="center")
        metrics_grid.add_column(justify="center")
        metrics_grid.add_column(justify="center")
        metrics_grid.add_column(justify="center")
        
        metrics_grid.add_row(
            f"[dim]Scanned[/dim]\n[bold bright_white]{total:,}[/bold bright_white]",
            f"[dim]Matched[/dim]\n[bold bright_green]{kept:,}[/bold bright_green]",
            f"[dim]Duration[/dim]\n[bold bright_cyan]{elapsed:.0f}s[/bold bright_cyan]",
            f"[dim]Output[/dim]\n[bold bright_yellow]{len(top)}[/bold bright_yellow]",
        )
        console.print(Align.center(metrics_grid))
        console.print()
        
        # ─────────────────────────────────────────────────────────────────────────────
        # TOP 3 WINNERS - Hero section
        # ─────────────────────────────────────────────────────────────────────────────
        if len(top) >= 1:
            console.print(Align.center(Text("── TOP PERFORMERS ──", style="dim")))
            console.print()
            
            winner_panels = []
            medals = ["🥇", "🥈", "🥉"]
            medal_colors = ["bold bright_yellow", "white", "rgb(205,127,50)"]
            
            for i in range(min(3, len(top))):
                row_data = top.iloc[i]
                ticker = row_data["ticker"]
                cagr_val = row_data["rev_3yr_cagr_pct"]
                cap = human_money(row_data["marketCap"])
                score = row_data.get("bagger_score", None)
                
                content = f"{medals[i]} [{medal_colors[i]}]{ticker}[/{medal_colors[i]}]\n"
                content += f"[bright_green]+{cagr_val:.0f}%[/bright_green] CAGR\n"
                content += f"[dim]{cap}[/dim]"
                if score and not pd.isna(score):
                    content += f"\n[cyan]Score: {int(score)}[/cyan]"
                
                winner_panels.append(Panel(
                    Align.center(content),
                    border_style="dim" if i > 0 else "bright_yellow",
                    width=20,
                    padding=(0, 1)
                ))
            
            console.print(Align.center(Columns(winner_panels, padding=(0, 2))))
            console.print()
        
        # ─────────────────────────────────────────────────────────────────────────────
        # MAIN TABLE - Clean, scannable, no noise
        # ─────────────────────────────────────────────────────────────────────────────
        console.print(Rule(f"[bold]Top {len(top)} by {title_metric}[/bold]", style="dim"))
        console.print()
        
        tbl = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
            padding=(0, 2),
            collapse_padding=True,
            show_edge=False,
        )
        
        tbl.add_column("#", justify="right", style="dim", width=4)
        tbl.add_column("TICKER", style="bold bright_white", width=8)
        tbl.add_column("MCAP", justify="right", style="dim", width=8)
        if not args.no_bagger:
            tbl.add_column("SCORE", justify="center", width=7)
        tbl.add_column("REVENUE", justify="right", width=9)
        tbl.add_column("CAGR", justify="right", width=12)
        tbl.add_column("SECTOR", style="dim", width=12)
        
        def format_score(s):
            if s is None or pd.isna(s):
                return "[dim]—[/dim]"
            s = int(s)
            if s >= 80:
                return f"[bold bright_green]{s}[/bold bright_green]"
            elif s >= 60:
                return f"[green]{s}[/green]"
            elif s >= 40:
                return f"[yellow]{s}[/yellow]"
            else:
                return f"[dim]{s}[/dim]"
        
        def format_cagr(c):
            try:
                c = float(c)
                if c >= 100:
                    return f"[bold bright_green]+{c:.0f}%[/bold bright_green]"
                elif c >= 50:
                    return f"[bright_green]+{c:.0f}%[/bright_green]"
                elif c >= 25:
                    return f"[green]+{c:.0f}%[/green]"
                elif c >= 0:
                    return f"[white]+{c:.0f}%[/white]"
                else:
                    return f"[red]{c:.0f}%[/red]"
            except:
                return "[dim]—[/dim]"
        
        def short_sector(s):
            if not s or s == "UNKNOWN":
                return "[dim]—[/dim]"
            mapping = {
                'TECHNOLOGY': 'Tech',
                'HEALTHCARE': 'Health',
                'FINANCIAL SERVICES': 'Finance',
                'CONSUMER CYCLICAL': 'Consumer',
                'COMMUNICATION SERVICES': 'Comms',
                'INDUSTRIALS': 'Industrial',
                'CONSUMER DEFENSIVE': 'Staples',
                'BASIC MATERIALS': 'Materials',
                'REAL ESTATE': 'Real Est',
                'ENERGY': 'Energy',
                'UTILITIES': 'Utilities',
            }
            return mapping.get(str(s).upper(), str(s)[:8])
        
        for idx, row_data in top.iterrows():
            rank = idx + 1
            rank_str = f"[bold bright_yellow]{rank}[/bold bright_yellow]" if rank <= 3 else f"[dim]{rank}[/dim]"
            
            cells = [
                rank_str,
                row_data["ticker"],
                human_money(row_data["marketCap"]),
            ]
            if not args.no_bagger:
                cells.append(format_score(row_data.get("bagger_score")))
            cells.extend([
                human_money(row_data["rev_recent"]),
                format_cagr(row_data["rev_3yr_cagr_pct"]),
                short_sector(row_data.get("sector", "")),
            ])
            tbl.add_row(*cells)
        
        console.print(tbl)
        console.print()
        
        # ─────────────────────────────────────────────────────────────────────────────
        # FOOTER - Minimal, actionable
        # ─────────────────────────────────────────────────────────────────────────────
        console.print(Rule(style="dim"))
        footer = Table.grid(padding=(0, 3))
        footer.add_column(justify="left")
        footer.add_column(justify="right")
        footer.add_row(
            f"[green]✓[/green] [white]Saved to[/white] [bold]{out_csv}[/bold]",
            f"[dim]Cap: {human_money(args.min_mkt_cap)}–{human_money(args.max_mkt_cap)}[/dim]"
        )
        console.print(footer)
        console.print()
        
    else:
        # ─────────────────────────────────────────────────────────────────────────────
        # PLAIN TEXT FALLBACK
        # ─────────────────────────────────────────────────────────────────────────────
        print()
        print("═" * 60)
        print(f"  TOP {len(top)} SCREENER RESULTS")
        print("═" * 60)
        print(f"  Scanned: {total:,}  |  Matched: {kept:,}  |  Time: {elapsed:.0f}s")
        print("─" * 60)
        
        cols = ["ticker", "marketCap", "rev_recent", "rev_3yr_cagr_pct"]
        if not args.no_bagger and "bagger_score" in top.columns:
            cols.insert(2, "bagger_score")
        print(top[cols].to_string(index=False))
        
        print("─" * 60)
        print(f"  ✓ Saved to: {out_csv}")
        print()


if __name__ == '__main__':
    main()
