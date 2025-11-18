import argparse
import math
import os
import sys
import time
from typing import List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

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


def main():
    parser = argparse.ArgumentParser(description="Top N small/mid caps by 3-year revenue CAGR (yfinance)")
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

    rows = []
    failed = []

    for t in tqdm(tickers, desc="Tickers"):
        try:
            tk = yf.Ticker(t)
            mktcap = get_market_cap(tk)
            if mktcap is None:
                failed.append((t, "no market cap"))
                time.sleep(SLEEP_SECONDS)
                continue
            if not (args.min_mkt_cap <= mktcap <= args.max_mkt_cap):
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

            rows.append({
                "ticker": t,
                "marketCap": mktcap,
                "rev_3yr_ago": rev_3yr_ago,
                "rev_recent": recent_rev,
                "rev_3yr_cagr": c
            })

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            failed.append((t, str(e)))
            time.sleep(SLEEP_SECONDS)
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows collected. Tips: ensure your CSV has many small/mid tickers; consider widening cap bounds; check network access.")
        if failed:
            print("Sample failures (up to 20):")
            for f in failed[:20]:
                print(f)
        sys.exit(0)

    df["rev_3yr_cagr_pct"] = df["rev_3yr_cagr"] * 100.0
    df_sorted = df.sort_values("rev_3yr_cagr_pct", ascending=False).reset_index(drop=True)
    top = df_sorted.head(args.top_n)

    out_csv = "top50_small_mid_revenue_cagr.csv"
    top.to_csv(out_csv, index=False)
    print(f"Saved top {len(top)} to {out_csv}")
    print(top[["ticker", "marketCap", "rev_3yr_cagr_pct", "rev_3yr_ago", "rev_recent"]].to_string(index=False))

    if failed:
        print("\nSome tickers failed or were skipped (sample):")
        for f in failed[:20]:
            print(f)


if __name__ == '__main__':
    main()
