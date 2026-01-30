#!/usr/bin/env python3
"""
Builds a Russell 2500 tickers CSV at data/russell2500_tickers.csv by scraping public sources
and ranking the Russell 3000 universe by current market cap (via yfinance) to select
the smallest 2,500 names.

Notes:
- This is a best-effort public-source builder; official FTSE Russell files may differ slightly.
- Requires: pandas, lxml, yfinance, tqdm
- Runtime: fetching market caps for ~3000 tickers can take several minutes.

Usage:
  python src/russell2500.py [--out data/russell2500_tickers.csv]

"""
import argparse
import os
import sys
import time
from typing import List, Set, Tuple

import pandas as pd
import yfinance as yf
from tqdm import tqdm

DEFAULT_OUT = "data/universes/russell2500_tickers.csv"
DATA_DIR = "data"

# NASDAQ Trader symbol directories (pipe-delimited)
NASDAQ_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

SLEEP_BETWEEN_REQUESTS = 0.02  # be polite to Yahoo
target_note = ""
TARGET_SIZE = 2500
# Sanity threshold: combined primary listings should exceed this count
MIN_EXPECTED_CANDIDATES = 4000

# Cache locations
META_DIR = os.path.join(DATA_DIR, "meta")
CAPS_CACHE = os.path.join(META_DIR, "caps_cache.csv")


def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    s = s.replace(" ", "")
    if "." in s:
        s = s.replace(".", "-")
    return s


def _read_nasdaq_table(url: str, symbol_col: str) -> pd.DataFrame:
    # These files are pipe-delimited and end with a footer line starting with "File Creation Time".
    try:
        df = pd.read_csv(url, sep='|', engine='python', dtype=str)
        # Drop footer rows without the symbol column present
        df = df[df[symbol_col].notna()]
        return df
    except Exception as e:
        print(f"Warning: failed to read {url}: {e}")
        return pd.DataFrame()


def fetch_us_primary_listings_from_nasdaq_trader() -> List[str]:
    """Fetch US primary listings from NASDAQ Trader directories and filter to common stocks.
    Excludes ETFs and test issues. Heuristically excludes preferreds, warrants, rights, and units.
    """
    dfs = []
    nasdaq = _read_nasdaq_table(NASDAQ_LISTED_URL, symbol_col='Symbol')
    if not nasdaq.empty:
        nasdaq = nasdaq.rename(columns={"Symbol": "SYMBOL", "Security Name": "SECURITY_NAME", "ETF": "ETF", "Test Issue": "TEST_ISSUE"})
        dfs.append(nasdaq[["SYMBOL", "SECURITY_NAME", "ETF", "TEST_ISSUE"]])
    other = _read_nasdaq_table(OTHER_LISTED_URL, symbol_col='ACT Symbol')
    if not other.empty:
        other = other.rename(columns={"ACT Symbol": "SYMBOL", "Security Name": "SECURITY_NAME", "ETF": "ETF", "Test Issue": "TEST_ISSUE"})
        dfs.append(other[["SYMBOL", "SECURITY_NAME", "ETF", "TEST_ISSUE"]])

    if not dfs:
        return []

    all_df = pd.concat(dfs, ignore_index=True)
    # Basic cleaning
    all_df["SYMBOL"] = all_df["SYMBOL"].astype(str).str.strip()
    all_df["SECURITY_NAME"] = all_df.get("SECURITY_NAME", "").astype(str)
    all_df["ETF"] = all_df.get("ETF", "N").astype(str).str.upper().fillna("N")
    all_df["TEST_ISSUE"] = all_df.get("TEST_ISSUE", "N").astype(str).str.upper().fillna("N")

    # Exclusions
    mask_keep = (
        (all_df["ETF"] != "Y") &
        (all_df["TEST_ISSUE"] != "Y")
    )

    # Heuristic exclusions based on name keywords
    bad_keywords = [
        "PREFERRED", "PFD", "WARRANT", "RIGHTS", "UNIT", "UNITS",
        "NOTE", "BOND", "DEPOSITARY", "ADR", "FUND", "TRUST"
    ]
    upper_names = all_df["SECURITY_NAME"].str.upper().fillna("")
    for kw in bad_keywords:
        mask_keep &= ~upper_names.str.contains(rf"\b{kw}\b", regex=True)

    filtered = all_df[mask_keep]

    # Normalize to Yahoo symbols and dedupe
    syms = [normalize_symbol(s) for s in filtered["SYMBOL"].tolist()]
    # Drop anything obviously not a stock ticker
    valid_syms = []
    for s in syms:
        if not s:
            continue
        if any(ch in s for ch in [" ", "/", "^"]):
            continue
        valid_syms.append(s)

    uniq = sorted(set(valid_syms))
    return uniq


def _load_caps_cache() -> dict:
    if not os.path.exists(CAPS_CACHE):
        return {}
    try:
        df = pd.read_csv(CAPS_CACHE)
        out = {}
        for _, row in df.iterrows():
            sym = str(row.get("ticker", "")).strip().upper()
            val = row.get("marketCap", None)
            try:
                cap = float(val) if pd.notna(val) else float("nan")
            except Exception:
                cap = float("nan")
            if sym:
                out[sym] = cap
        return out
    except Exception:
        return {}


def _save_caps_cache(cache: dict) -> None:
    os.makedirs(META_DIR, exist_ok=True)
    rows = [(k, v) for k, v in cache.items()]
    df = pd.DataFrame(rows, columns=["ticker", "marketCap"])
    df.to_csv(CAPS_CACHE, index=False)


def _fetch_cap_once(ticker: str) -> float:
    """Robust market cap fetch using multiple fallbacks.
    Tries fast_info, info['marketCap'], then sharesOutstanding * price.
    Returns NaN on failure.
    """
    try:
        tk = yf.Ticker(ticker)
        # 1) fast_info (object or dict)
        try:
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    m = fi.get("market_cap") or fi.get("marketCap")
                    if m is not None and float(m) > 0:
                        return float(m)
                else:
                    v = getattr(fi, 'market_cap', None)
                    if v is None:
                        v = getattr(fi, 'marketCap', None)
                    if v is not None and float(v) > 0:
                        return float(v)
        except Exception:
            pass
        # 2) info['marketCap']
        try:
            info = tk.info or {}
            m = info.get("marketCap") or info.get("market_cap")
            if m is not None and float(m) > 0:
                return float(m)
        except Exception:
            info = {}
        # 3) sharesOutstanding * price
        price = None
        try:
            # fast_info last/regular price
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                if isinstance(fi, dict):
                    price = fi.get('last_price') or fi.get('regular_market_price') or fi.get('lastClose') or fi.get('previous_close')
                else:
                    for attr in ("last_price","regular_market_price","last_close","previous_close"):
                        val = getattr(fi, attr, None)
                        if val is not None:
                            price = val
                            break
            if price is None and info:
                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if price is None:
                # last close from short history
                hist = tk.history(period='5d', interval='1d', auto_adjust=True)
                if hist is not None and not hist.empty and 'Close' in hist.columns:
                    price = float(hist['Close'].dropna().iloc[-1])
            price = float(price) if price is not None else None
        except Exception:
            price = None
        shares = None
        try:
            so = info.get('sharesOutstanding') if info else None
            if so is not None:
                shares = float(so)
            else:
                # balance sheet fallback
                bs = getattr(tk, 'balance_sheet', None)
                if isinstance(bs, pd.DataFrame) and not bs.empty and 'Common Stock' in bs.index:
                    # too schema-dependent; skip heavy parsing in builder for speed
                    shares = None
        except Exception:
            shares = None
        if price is not None and price > 0 and shares is not None and shares > 0:
            return float(price) * float(shares)
        return float('nan')
    except Exception:
        return float('nan')


def get_market_cap_with_cache(ticker: str, cache: dict, retries: int = 2, delay: float = 0.2) -> float:
    key = ticker.upper()
    if key in cache and pd.notna(cache[key]):
        return float(cache[key])
    cap = float("nan")
    for attempt in range(retries + 1):
        cap = _fetch_cap_once(ticker)
        if pd.notna(cap) and cap > 0:
            break
        time.sleep(delay * (1 + attempt))
    cache[key] = cap
    return cap


def rank_smallest_by_market_cap(universe: List[str], workers: int = 0) -> List[Tuple[str, float]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = _load_caps_cache()
    rows: List[Tuple[str, float]] = []

    # Decide worker count (0 -> auto)
    if workers is None or workers <= 0:
        try:
            cpu = os.cpu_count() or 4
            workers = min(16, max(4, cpu * 2))
        except Exception:
            workers = 8

    def _fetch(sym: str) -> Tuple[str, float]:
        # polite pacing per thread
        m = get_market_cap_with_cache(sym, cache)
        return (sym, m)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, sym): sym for sym in universe}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Fetching market caps"):
            try:
                sym, m = fut.result()
                rows.append((sym, m))
            except Exception:
                # mark as NaN on error
                rows.append((futs[fut], float('nan')))

    # Persist cache after pass
    _save_caps_cache(cache)

    df = pd.DataFrame(rows, columns=["ticker", "marketCap"]) 
    # Drop missing market caps
    df = df.dropna(subset=["marketCap"]) 
    # Keep positive caps
    df = df[df["marketCap"] > 0]
    # Sort ascending (smallest first)
    df = df.sort_values("marketCap", ascending=True).reset_index(drop=True)
    return list(df.itertuples(index=False, name=None))  # list of (ticker, cap)


def main():
    ap = argparse.ArgumentParser(description="Build Russell 2500 tickers CSV from public sources")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"Output CSV path (default: {DEFAULT_OUT})")
    ap.add_argument("--min_size", type=int, default=TARGET_SIZE, help="Target list size (default 2500)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    print("Gathering US primary listings from NASDAQ Trader symbol directories…")
    candidates = fetch_us_primary_listings_from_nasdaq_trader()
    print(f"Collected {len(candidates)} unique candidate symbols before cap ranking.")

    if len(candidates) < MIN_EXPECTED_CANDIDATES:
        print(
            f"Error: Only {len(candidates)} candidates found from NASDAQ Trader directories; this is too few to approximate R2500."
        )
        sys.exit(2)

    ranked = rank_smallest_by_market_cap(candidates)
    if len(ranked) < args.min_size:
        print(
            f"Error: Only {len(ranked)} symbols have valid market caps; cannot build Russell 2500 approximation."
        )
        sys.exit(3)

    smallest = ranked[: args.min_size]
    tickers_out = [t for t, _ in smallest]

    # Write CSV
    out_path = args.out
    # Ensure parent directory exists for the output path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out = pd.DataFrame({"ticker": tickers_out})
    df_out.to_csv(out_path, index=False)

    print(f"Wrote {len(tickers_out)} tickers to {out_path}")


if __name__ == "__main__":
    main()
