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

# Pretty console output
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
except Exception:  # pragma: no cover
    Console = None
    Table = None
    box = None

# Defaults (can be overridden via CLI)
DEFAULT_UNIVERSE = 'csv'  # 'csv' or 'russell_2500'
DEFAULT_CSV_PATH = 'data/universes/russell2500_tickers.csv'

# Market cap bounds (USD)
MIN_MKT_CAP = 300e6      # 300 million
MAX_MKT_CAP = 10e9       # 10 billion

# How many to output
TOP_N = 50

# Universe sanity check (abort if universe too small)
MIN_UNIVERSE_SIZE = 2400

SLEEP_SECONDS = 0.05  # rate-limiting between requests

# ========================== Helpers =====================================

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


def get_market_cap(tk: yf.Ticker) -> Optional[float]:
    # Prefer fast_info when available
    mkt = None
    try:
        fi = getattr(tk, 'fast_info', None)
        if fi and isinstance(fi, dict):
            mkt = fi.get('market_cap') or fi.get('marketCap')
    except Exception:
        pass
    if mkt is None:
        try:
            info = tk.info or {}
            mkt = info.get('marketCap') or info.get('market_cap')
        except Exception:
            mkt = None
    return float(mkt) if mkt is not None else None


def _sorted_columns(df: pd.DataFrame) -> List:
    cols = list(df.columns)
    # Try convert to datetime for sorting descending
    try:
        cols_sorted = sorted(cols, key=lambda c: pd.to_datetime(str(c), errors='coerce'), reverse=True)
        return cols_sorted
    except Exception:
        return cols


def get_revenue_values(tk: yf.Ticker) -> Optional[List[float]]:
    # Try annual income statement first
    try:
        # yfinance >=0.2: income_stmt property
        income = getattr(tk, 'income_stmt', None)
        if isinstance(income, pd.DataFrame) and not income.empty:
            df = income.copy()
            idx = [str(i).lower() for i in df.index]
            candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue' or 'net sales' in name or 'sales' in name]
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
    except Exception:
        pass

    # Fallback: legacy financials table
    try:
        fin = tk.financials
        if isinstance(fin, pd.DataFrame) and not fin.empty:
            df = fin.copy()
            idx = [str(i).lower() for i in df.index]
            candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue' or 'net sales' in name or 'sales' in name]
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
    except Exception:
        pass

    # Fallback: quarterly income statement -> construct rolling 12-quarter sums for 4 points
    try:
        q = getattr(tk, 'quarterly_income_stmt', None)
        if isinstance(q, pd.DataFrame) and not q.empty:
            df = q.copy()
            idx = [str(i).lower() for i in df.index]
            candidates = [i for i, name in enumerate(idx) if 'total revenue' in name or 'totalrevenue' in name or name.strip() == 'revenue' or 'net sales' in name or 'sales' in name]
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
                # Build rolling 12-quarter sums shifted by 0, 4, 8, 12 quarters
                def sum_window(start: int) -> Optional[float]:
                    window = series[start:start+12]
                    if len(window) < 12:
                        return None
                    nums = [x for x in window if x is not None]
                    if len(nums) < 10:  # tolerate some missing but require most
                        return None
                    return float(sum(nums))
                vals = []
                for offset in (0, 4, 8, 12):
                    s = sum_window(offset)
                    if s is not None:
                        vals.append(s)
                if len(vals) >= 4:
                    return vals[:4]
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


def price_vol_and_drawdown(tk: yf.Ticker, fast: bool = False) -> Tuple[Optional[float], Optional[float]]:
    # Returns (ann_vol, max_drawdown) using 1y (fast) or 3y window
    try:
        period = '1y' if fast else '3y'
        hist = tk.history(period=period, interval='1d', auto_adjust=True)
        if hist is None or hist.empty or 'Close' not in hist:
            return None, None
        close = hist['Close'].dropna()
        if len(close) < 60:
            return None, None
        rets = close.pct_change().dropna()
        ann_vol = float(np.std(rets)) * (252.0 ** 0.5)
        # Max drawdown
        roll_max = np.maximum.accumulate(close.values)
        dd = (close.values - roll_max) / roll_max
        max_dd = float(np.min(dd)) if len(dd) else None
        if max_dd is not None:
            max_dd = abs(max_dd)
        return ann_vol, max_dd
    except Exception:
        return None, None


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

        # Growth: map c/r* ratio to 0..1 via piecewise and logistic; apply base-revenue scaling
        if rev_cagr_3y is not None:
            try:
                ratio = rev_cagr_3y / max(1e-9, r_star)
                growth_raw = _piecewise_score(ratio, [(0.0,0.0),(0.5,0.3),(1.0,0.7),(1.5,0.9),(2.0,1.0)])
                if growth_raw is not None:
                    growth_raw = 0.5 * growth_raw + 0.5 * logistic(growth_raw, k=4.0, x0=0.6)
                    if rev_base is not None:
                        growth_cap = _piecewise_score(rev_base, [(0.0,0.4),(2e7,0.6),(5e7,0.8),(1e8,1.0)])
                        if growth_cap is not None:
                            growth_raw = growth_raw * float(growth_cap)
            except Exception:
                growth_raw = None

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
            ann_vol, max_dd = price_vol_and_drawdown(tk, fast=getattr(args, 'fast_bagger', False))
            vol_score = _piecewise_score(ann_vol, [(0.1,1.0),(0.2,0.9),(0.4,0.6),(0.6,0.3),(0.8,0.1),(1.0,0.0)]) if ann_vol is not None else None
            dd_score = _piecewise_score(max_dd, [(0.1,1.0),(0.2,0.8),(0.4,0.5),(0.6,0.2),(0.8,0.05),(0.9,0.0)]) if max_dd is not None else None
            if vol_score is not None or dd_score is not None:
                v = vol_score if vol_score is not None else 0.6
                d = dd_score if dd_score is not None else 0.6
                risk_raw = 0.5 * v + 0.5 * d
        except Exception:
            ann_vol, max_dd, risk_raw = None, None, None

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


def main():
    parser = argparse.ArgumentParser(description="Top N small/mid caps by 3-year revenue CAGR (yfinance) with optional 100x bagger scoring")
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
    parser.add_argument('--bagger_horizon', type=int, choices=[10,15,20,25,30], default=20,
                        help='Horizon in years for the 100x target (affects growth thresholds). Default: 20')
    parser.add_argument('--bagger_verbose', action='store_true', help='Show sub-score breakdown in output table')
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

    print(f"Got {len(tickers)} tickers in universe.")

    start_ts = time.time()
    rows: List[Dict[str, float]] = []
    failed: List[Tuple[str, str]] = []
    skipped_cap = 0

    # Debug collections for distribution snapshots
    debug_vals = {
        'bagger_score': [], 'rev_cagr_3y': [], 'growth': [], 'quality': [], 'discipline': [], 'valuation': [], 'risk': [],
        'gm': [], 'om': [], 'fcf_margin': [], 'ps': [], 'evs': [], 'ann_vol': [], 'max_dd': [], 'dilution_rate': []
    }
    # Coverage counters for debug mode
    debug_cov = {
        'subs_returns': 0,
        'growth_ok': 0,
        'quality_ok': 0,
        'discipline_ok': 0,
        'valuation_ok': 0,
        'risk_ok': 0,
        'fallback_growth_only': 0
    }

    processed = 0
    debug_limit = int(getattr(args, 'debug_bagger', 0) or 0)

    for t in tqdm(tickers, desc="Tickers"):
        # If in debug mode and reached the limit, stop early
        if debug_limit > 0 and processed >= debug_limit:
            break
        try:
            tk = yf.Ticker(t)
            mktcap = get_market_cap(tk)
            if mktcap is None:
                failed.append((t, "no market cap"))
                time.sleep(SLEEP_SECONDS)
                continue
            if not (args.min_mkt_cap <= mktcap <= args.max_mkt_cap):
                skipped_cap += 1
                time.sleep(SLEEP_SECONDS)
                continue

            rev_vals = get_revenue_values(tk)
            if not rev_vals or len(rev_vals) < 4:
                failed.append((t, "insufficient revenue history"))
                time.sleep(SLEEP_SECONDS)
                continue

            recent_rev = rev_vals[0]
            rev_3yr_ago = rev_vals[3]

            c = cagr(rev_3yr_ago, recent_rev, years=3)
            if c is None or (isinstance(c, float) and math.isnan(c)):
                failed.append((t, "cagr failed"))
                time.sleep(SLEEP_SECONDS)
                continue

            # 100x bagger scoring (optional)
            bagger_score = None
            bagger_S = None
            subs = None
            used_fallback = False
            if not args.no_bagger:
                try:
                    subs = compute_bagger_subscores(tk, mktcap, c, recent_rev, rev_vals, args)
                    # add rev_cagr to subs for downstream guardrails
                    if subs is not None:
                        subs['rev_cagr_3y'] = c
                    bagger_score, bagger_S = compute_bagger_score(subs, args)
                except Exception:
                    # Non-fatal; proceed to fallback below
                    bagger_score = None
                    bagger_S = None
                    subs = None
                # Fallback: if advanced scoring failed or yielded None, compute a growth-only score
                if bagger_score is None:
                    try:
                        T = int(getattr(args, 'bagger_horizon', 20))
                        r_star = (100.0 ** (1.0 / T)) - 1.0
                        # Map ratio of observed CAGR to required CAGR to 0..100
                        ratio = c / max(1e-9, r_star) if c is not None else None
                        if ratio is not None:
                            # Piecewise to [0,1], then logistic for probability-like shape
                            base = _piecewise_score(ratio, [(0.0,0.0),(0.5,0.25),(1.0,0.6),(1.5,0.85),(2.0,0.97),(3.0,0.995)])
                            if base is not None:
                                p = 0.5 * base + 0.5 * logistic(base, k=3.0, x0=0.55)
                                bagger_score = int(round(100.0 * clamp(p, 0.0, 0.999)))
                                bagger_S = float(base)
                                used_fallback = True
                    except Exception:
                        pass

            row = {
                "ticker": t,
                "marketCap": mktcap,
                "rev_3yr_ago": rev_3yr_ago,
                "rev_recent": recent_rev,
                "rev_3yr_cagr": c,
            }
            # Attach bagger fields
            row["bagger_horizon"] = args.bagger_horizon if not args.no_bagger else None
            row["bagger_score"] = bagger_score
            row["bagger_S"] = bagger_S
            if subs and getattr(args, 'export_subscores', False):
                row.update({
                    "sub_growth": subs.get("growth"),
                    "sub_quality": subs.get("quality"),
                    "sub_discipline": subs.get("discipline"),
                    "sub_valuation": subs.get("valuation"),
                    "sub_risk": subs.get("risk"),
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
                })

            # Collect debug metrics (append for every processed ticker to keep array lengths equal)
            if debug_limit > 0:
                debug_vals['bagger_score'].append(bagger_score if bagger_score is not None else np.nan)
                debug_vals['rev_cagr_3y'].append(c if c is not None else np.nan)
                s = subs or {}
                debug_vals['growth'].append(s.get('growth', np.nan))
                debug_vals['quality'].append(s.get('quality', np.nan))
                debug_vals['discipline'].append(s.get('discipline', np.nan))
                debug_vals['valuation'].append(s.get('valuation', np.nan))
                debug_vals['risk'].append(s.get('risk', np.nan))
                debug_vals['gm'].append(s.get('gm', np.nan))
                debug_vals['om'].append(s.get('om', np.nan))
                debug_vals['fcf_margin'].append(s.get('fcf_margin', np.nan))
                debug_vals['ps'].append(s.get('ps', np.nan))
                debug_vals['evs'].append(s.get('evs', np.nan))
                debug_vals['ann_vol'].append(s.get('ann_vol', np.nan))
                debug_vals['max_dd'].append(s.get('max_dd', np.nan))
                debug_vals['dilution_rate'].append(s.get('dilution_rate', np.nan))

            rows.append(row)
            processed += 1

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            failed.append((t, str(e)))
            time.sleep(SLEEP_SECONDS)
            continue

    elapsed = time.time() - start_ts
    total = len(tickers)
    kept = len(rows)
    skipped = total - kept

    # Aggregate failure reasons
    reason_counts: Dict[str, int] = {}
    for _, reason in failed:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if skipped_cap:
        reason_counts["filtered by market cap"] = reason_counts.get("filtered by market cap", 0) + skipped_cap

    df = pd.DataFrame(rows)
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

    out_csv = "top50_small_mid_revenue_cagr.csv"
    top.to_csv(out_csv, index=False)

    # Debug: print factor/score distribution snapshots
    if debug_limit > 0 and len(debug_vals.get('rev_cagr_3y', [])) > 0:
        try:
            debug_df = pd.DataFrame(debug_vals)
            # Keep only numeric columns
            for col in list(debug_df.columns):
                debug_df[col] = pd.to_numeric(debug_df[col], errors='coerce')
            percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
            desc = debug_df.describe(percentiles=percentiles).T
            desc = desc[['count','mean','std','5%','25%','50%','75%','95%']].rename(columns={'50%':'median'})
            if Console is not None and Table is not None and not args.plain:
                console = Console()
                console.print("\n[bold blue]Debug distributions (processed first {} tickers)[/bold blue]".format(processed))
                tbl = Table(box=box.SIMPLE_HEAVY if box else None, show_lines=False)
                tbl.add_column("Metric")
                for c in ['count','mean','std','5%','25%','median','75%','95%']:
                    tbl.add_column(c, justify='right')
                for metric, rowv in desc.iterrows():
                    vals = [metric]
                    for c in ['count','mean','std','5%','25%','median','75%','95%']:
                        val = rowv.get(c)
                        if pd.isna(val):
                            vals.append("-")
                        else:
                            if metric in ("bagger_score",):
                                vals.append(f"{val:,.1f}")
                            elif metric.endswith("_cagr_3y") or metric in ("ann_vol","max_dd","fcf_margin","gm","om"):
                                vals.append(f"{val:,.4f}")
                            else:
                                vals.append(f"{val:,.3f}")
                    tbl.add_row(*vals)
                console.print(tbl)
            else:
                print(f"\nDebug distributions (processed first {processed} tickers)")
                print(desc.to_string())
        except Exception as _e:
            print(f"[debug] failed to build distribution snapshot: {_e}")

    use_rich = (Console is not None and Table is not None and not args.plain)

    if use_rich:
        console = Console()
        horizon_note = f" | 100× horizon: {args.bagger_horizon}y" if not args.no_bagger else ""
        console.print(f"[bold green]Top {len(top)} by {title_metric}[/bold green]  "
                      f"[dim](Universe: {total:,} | Kept: {kept:,} | Skipped: {skipped:,} | "
                      f"Cap range: {human_money(args.min_mkt_cap)}–{human_money(args.max_mkt_cap)} | "
                      f"Elapsed: {elapsed/60:.1f} min{horizon_note})[/dim]")
        tbl = Table(box=box.SIMPLE_HEAVY if box else None, show_lines=False)
        tbl.add_column("#", justify="right", style="bold")
        tbl.add_column("Ticker", style="cyan", no_wrap=True)
        tbl.add_column("Mkt Cap", justify="right")
        if not args.no_bagger:
            tbl.add_column("100× Score", justify="right")
        tbl.add_column("Rev CAGR 3Y", justify="right")
        tbl.add_column("Rev 3Y Ago", justify="right")
        tbl.add_column("Rev Recent", justify="right")
        for idx, row in top.iterrows():
            cells = [
                f"{idx+1}",
                str(row["ticker"]),
                human_money(row["marketCap"]),
            ]
            if not args.no_bagger:
                cells.append("-" if pd.isna(row.get("bagger_score")) else f"{int(row.get('bagger_score'))}")
            cells.extend([
                fmt_pct(row["rev_3yr_cagr_pct"]),
                human_money(row["rev_3yr_ago"]),
                human_money(row["rev_recent"]) 
            ])
            tbl.add_row(*cells)
        console.print(tbl)
        console.print(f"Saved CSV → [bold]{out_csv}[/bold]")

        if reason_counts:
            console.print("\n[dim]Skip summary:[/dim]")
            # Show top 5 reasons
            items = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            for reason, cnt in items:
                console.print(f"  • {reason}: {cnt}")
        if args.show_failures and failed:
            console.print("\n[dim]Sample failures (up to 20):[/dim]")
            for f in failed[:20]:
                console.print(f"  • {f[0]} — {f[1]}")
    else:
        print(f"Saved top {len(top)} to {out_csv}")
        # Select columns for plain output
        cols = ["ticker", "marketCap"]
        if not args.no_bagger and "bagger_score" in top.columns:
            cols.append("bagger_score")
        cols += ["rev_3yr_cagr_pct", "rev_3yr_ago", "rev_recent"]
        print(top[cols].to_string(index=False))
        if reason_counts:
            print("\nSkip summary:")
            items = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
            for reason, cnt in items:
                print(f"  - {reason}: {cnt}")
        if args.show_failures and failed:
            print("\nSome tickers failed or were skipped (sample):")
            for f in failed[:20]:
                print(f)


if __name__ == '__main__':
    main()
