#!/usr/bin/env python3
"""
fx_data_utils.py

Data fetching and utility functions for FX signals.
Separates data acquisition and currency conversion logic from signal computation.
"""

from __future__ import annotations

import math
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import pathlib
import functools
from threading import Lock
import requests

# Rich for beautiful table output
try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    Console = None
    Table = None

# Configure yfinance with longer timeout (default is 10 seconds, which is too short for bulk downloads)
YFINANCE_TIMEOUT = 60  # seconds

# Apply timeout to yfinance session
try:
    yf.utils.requests_session = requests.Session()
    yf.utils.requests_session.request = functools.partial(
        yf.utils.requests_session.request, timeout=YFINANCE_TIMEOUT
    )
except Exception:
    pass  # Fallback: yfinance will use its defaults

PRICE_CACHE_DIR = os.path.join("scripts", "quant", "cache", "prices")
PRICE_CACHE_DIR_PATH = pathlib.Path(PRICE_CACHE_DIR)
PRICE_CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)

FAILED_CACHE_DIR = os.path.join("scripts", "quant", "cache", "failed")
FAILED_CACHE_DIR_PATH = pathlib.Path(FAILED_CACHE_DIR)
FAILED_CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
FAILED_ASSETS_FILE = FAILED_CACHE_DIR_PATH / "failed_assets.json"

# ============================================================================
# OFFLINE MODE - Use cached data only, don't try to download from Yahoo Finance
# ============================================================================
# Set via environment variable: OFFLINE_MODE=1
# When enabled:
# - Uses only cached price data from scripts/quant/cache/prices
# - Skips all Yahoo Finance API calls
# - Allows system to run without network connectivity
# ============================================================================
OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "").lower() in ("1", "true", "yes", "on")

# Symbols known to be problematic (delisted, frequently failing, etc.)
# These will be skipped during bulk downloads to avoid noise
KNOWN_PROBLEMATIC_SYMBOLS = {
#     "PACB",   # Pacific Biosciences - frequently delisted warnings
}

_price_file_lock = Lock()

# Deterministic overrides for Yahoo tickers that routinely fail or change format
YAHOO_TICKER_OVERRIDES = {
    "XAGUSD=X": "SI=F",
    "VOYG.TO": "VOYG",
    "MTXDE": "MTX.DE",
    "GEV.OL": "GEV",
    "KOZ1-DE": "KOG.OL",
    "KOZ1DE": "KOG.OL",
    "MTX-DE": "MTX.DE",
    "BAYN-DE": "BAYN.DE",
    "VOW3DE": "VOW3.DE",
    'VOW3-DE': "VOW3.DE",
    "R3NK-DE": "R3NK.DE",
    "RHMDE": "RHM.DE",
    "HEIA-AS": "HEIA.AS",
    "TKA-DE": "TKA.DE",
    "THEON": "THEON.AS",
    "THEON-AS": "THEON.AS",
    "EXA": "EXA.PA",
}

# Precious metal FX tickers mapped to liquid COMEX futures because Yahoo spot tickers are unreliable
PRECIOUS_METAL_FX_TO_FUTURES = {
    "XAUUSD=X": "GC=F",
    "XAGUSD=X": "SI=F",
    "XPTUSD=X": "PL=F",
    "XPDUSD=X": "PA=F",
}

# Yahoo punctuation/country suffix normalization hotfixes
PUNCTUATION_OVERRIDES = {
    "BRK.B": "BRK-B",
    # Moog class A is dot on Yahoo
    "MOG-A": "MOG.A",
    "MOG.A": "MOG.A",
    "BTC.USD": "BTC-USD",
    "HO-PA": "HO.PA",
    "HOPA": "HO.PA",
    "AIRPA": "AIR.PA",
    "SAF-PA": "SAF.PA",
    "TKA-DE": "TKA.DE",
    "RHM-DE": "RHM.DE",
    "BMW3-DE": "BMW3.DE",
    "BAYN-DE": "BAYN.DE",
    "KOZ1-DE": "KOG.OL",
    "HEIAAS": "HEIA.AS",
    "SAFPA": "SAF.PA",
    "AIR-PA": "AIR.PA",
    "BA-L": "BA.L",
    "005930-KS": "005930.KS",
    "FACCVI": "FACC.VI",
    "HAGDE": "HAG.DE",
    "HO-PA": "HO.PA",
    "BTCUSD": "BTC-USD",
    "BTCUSD=X": "BTC-USD",
}

COUNTRY_SUFFIX_COMPLETIONS = {
    "BAYN": "BAYN.DE",
    "BMW3": "BMW3.DE",
    "VOW3": "VOW3.DE",
    "RHM": "RHM.DE",
    "TKA": "TKA.DE",
    "R3NK": "R3NK.DE",
    "KOZ1.DE": "KOG.OL",
    "HEIA": "HEIA.AS",
    "SAF": "SAF.PA",
    "SGLP": "SGLP.L",
    "MAGD": "MAGD.L",
}

# Proxy overrides for synthetic/alias tickers to liquid underlyings
PROXY_OVERRIDES = {
    "TSLD": "TSLA",
    "QQQO": "QQQ",
    "METI": "META",
    "MSFI": "MSFT",
    "MAGD": "QQQ",
    "AVGI": "AVGO",
    "BABI": "BABA",
    "BABY": "BABA",
    "AMDI": "AMD",
    "DFNG": "ITA",
    "GFA": "ANGL",
    "TRET": "VNQ",
    "GLDE": "GLD",
    "GOOO": "GOOG",
    "SGLP": "SGLP.L",
    "HAG": "HAG.DE",
    "HO": "HO.PA",
    "BAYN": "BAYN.DE",
    "BMW3": "BMW3.DE",
    "VOW3": "VOW3.DE",
    "RHM": "RHM.DE",
    "R3NK": "R3NK.DE",
    "KOZ1": "KOG.OL",
    "XAGUSD": "SI=F",
    "XAGUSD=X": "SI=F",
    "MANT": "CACI",
    "SAF": "SAF.PA",
    "AIR": "AIR.PA",
    "BTCUSD": "BTC-USD",
    "MAMET": "META",
    "SIFSKYH": "SIF",
    "GLDW": "GLD",
    "FACC": "FACC.VI",
    "EXA": "EXA.PA",
    "THEON": "THEON.AS",
}


# Display-name cache for full asset names (e.g., company longName)
_DISPLAY_NAME_CACHE: Dict[str, str] = {}

# Pretty-print headers only once per run
_SYMBOL_MAP_HEADER_PRINTED = False
_SYMBOL_FAIL_HEADER_PRINTED = False

# Optional ANSI color (disable with FX_LOG_COLOR=0)
_USE_COLOR = os.getenv("FX_LOG_COLOR", "1") != "0"
_COLORS = {
    "green": "\033[92m" if _USE_COLOR else "",
    "yellow": "\033[93m" if _USE_COLOR else "",
    "red": "\033[91m" if _USE_COLOR else "",
    "cyan": "\033[96m" if _USE_COLOR else "",
    "reset": "\033[0m" if _USE_COLOR else "",
}

# Column widths for pretty tables
_COL_W_ORIG = 14
_COL_W_NORM = 14
_COL_W_ACT = 27
_COL_W_WHY = 35


def _c(text: str, color: str) -> str:
    return f"{_COLORS.get(color, '')}{text}{_COLORS['reset']}" if text else text


def _cap_date_to_today(date_str: Optional[str]) -> Optional[str]:
    """Cap a date string to today's date. Never return a future date.
    This prevents attempts to download data for dates that don't exist yet.
    """
    if date_str is None:
        return None
    try:
        dt = pd.to_datetime(date_str).date()
        today = datetime.now().date()
        if dt > today:
            return today.isoformat()
        return date_str
    except Exception:
        return date_str


def _human_action(meta: Dict) -> str:
    t = (meta or {}).get("type")
    if t == "proxy_override":
        return "Proxied"
    if t == "suffix_completion":
        return "Exchange suffix added"
    if t == "punctuation_fix":
        return "Punctuation fixed"
    if t == "commodity_proxy":
        return "Commodity proxy"
    if t == "override":
        return "Override"
    return "Adjusted"


def _human_reason(meta: Dict) -> str:
    r = (meta or {}).get("reason") or ""
    mapping = {
        "mapped_to_liquid_underlying": "Use more liquid listing",
        "country_exchange_suffix": "Add correct exchange suffix",
        "Yahoo punctuation grammar": "Normalize Yahoo punctuation",
        "Yahoo spot unavailable": "Spot unsupported; use futures",
        "deterministic_override": "Explicit override",
    }
    return mapping.get(r, r or "Normalized")


# Rich-based symbol mapping table (world-class UX)
_SYMBOL_MAP_TABLE = None
_SYMBOL_FAIL_TABLE = None
_RICH_CONSOLE = None
# Track logged symbols to prevent duplicates
_LOGGED_SYMBOL_MAPPINGS: set = set()
_LOGGED_SYMBOL_FAILURES: set = set()


def _get_rich_console():
    global _RICH_CONSOLE
    if _RICH_CONSOLE is None and _HAS_RICH:
        _RICH_CONSOLE = Console(force_terminal=True)
    return _RICH_CONSOLE


def _init_symbol_map_table():
    """Create a beautiful Rich table for symbol mappings."""
    table = Table(
        title="[bold cyan]Symbol Mappings[/bold cyan]",
        show_header=True,
        header_style="bold white",
        border_style="cyan",
        title_style="bold cyan",
        padding=(0, 1),
        collapse_padding=True,
    )
    table.add_column("Original", style="cyan", justify="left", width=14)
    table.add_column("Normalized", style="white", justify="left", width=14)
    table.add_column("Action", style="green", justify="left", width=24)
    table.add_column("Reason", style="white", justify="left", width=30)
    return table


def _init_symbol_fail_table():
    """Create a beautiful Rich table for symbol failures."""
    table = Table(
        title="[bold red]Symbol Failures[/bold red]",
        show_header=True,
        header_style="bold white",
        border_style="red",
        title_style="bold red",
        padding=(0, 1),
        collapse_padding=True,
    )
    table.add_column("Original", style="yellow", justify="left", width=14)
    table.add_column("Status", style="red", justify="left", width=24)
    table.add_column("Reason", style="white", justify="left", width=35)
    return table


def _map_border() -> str:
    return f"+{'-'*(_COL_W_ORIG+2)}+{'-'*(_COL_W_NORM+2)}+{'-'*(_COL_W_ACT+2)}+{'-'*(_COL_W_WHY+2)}+"


def _fail_border() -> str:
    return f"+{'-'*(_COL_W_ORIG+2)}+{'-'*(_COL_W_ACT+2)}+{'-'*(_COL_W_WHY+2)}+"


def _print_symbol_map_header():
    global _SYMBOL_MAP_HEADER_PRINTED, _SYMBOL_MAP_TABLE
    if _SYMBOL_MAP_HEADER_PRINTED:
        return
    if _HAS_RICH:
        _SYMBOL_MAP_TABLE = _init_symbol_map_table()
    else:
        print("\n" + _c("Symbol mappings", "cyan"))
        print(_map_border())
        print(f"| {'Original':<{_COL_W_ORIG}} | {'Normalized':<{_COL_W_NORM}} | {'Action':<{_COL_W_ACT}} | {'Why':<{_COL_W_WHY}} |")
        print(_map_border())
    _SYMBOL_MAP_HEADER_PRINTED = True


def _print_symbol_fail_header():
    global _SYMBOL_FAIL_HEADER_PRINTED, _SYMBOL_FAIL_TABLE
    if _SYMBOL_FAIL_HEADER_PRINTED:
        return
    if _HAS_RICH:
        _SYMBOL_FAIL_TABLE = _init_symbol_fail_table()
    else:
        print("\n" + _c("Symbol failures", "red"))
        print(_fail_border())
        print(f"| {'Original':<{_COL_W_ORIG}} | {'Status':<{_COL_W_ACT}} | {'Reason':<{_COL_W_WHY}} |")
        print(_fail_border())
    _SYMBOL_FAIL_HEADER_PRINTED = True


def _log_symbol_map(original: str, normalized: str, meta: Dict) -> None:
    global _LOGGED_SYMBOL_MAPPINGS
    # Deduplicate - only log each (original, normalized) pair once
    key = (original, normalized)
    if key in _LOGGED_SYMBOL_MAPPINGS:
        return
    _LOGGED_SYMBOL_MAPPINGS.add(key)
    
    _print_symbol_map_header()
    action = _human_action(meta)
    why = _human_reason(meta)
    
    if _HAS_RICH and _SYMBOL_MAP_TABLE is not None:
        _SYMBOL_MAP_TABLE.add_row(
            original,
            f"[bold]{normalized}[/bold]",
            f"[green]✔ {action}[/green]",
            why
        )
    else:
        icon = "✔" if _USE_COLOR else "*"
        orig_disp = _c(original, "cyan")
        action_disp = _c(icon + " " + action, "green")
        print(f"| {orig_disp:<{_COL_W_ORIG}} | {normalized:<{_COL_W_NORM}} | {action_disp:<{_COL_W_ACT}} | {why:<{_COL_W_WHY}} |")


def _log_symbol_fail(original: str, status: str, reason: Optional[str] = None) -> None:
    _print_symbol_fail_header()
    r = reason or ""
    
    if _HAS_RICH and _SYMBOL_FAIL_TABLE is not None:
        _SYMBOL_FAIL_TABLE.add_row(
            original,
            f"[red]✖ {status}[/red]",
            r
        )
    else:
        icon = "✖" if _USE_COLOR else "!"
        status_disp = _c(icon + " " + status, "red")
        print(f"| {original:<{_COL_W_ORIG}} | {status_disp:<{_COL_W_ACT}} | {r:<{_COL_W_WHY}} |")


def print_symbol_tables() -> None:
    """Print the accumulated symbol mapping and failure tables.
    Call this after all symbols have been processed."""
    global _SYMBOL_MAP_TABLE, _SYMBOL_FAIL_TABLE, _SYMBOL_MAP_HEADER_PRINTED, _SYMBOL_FAIL_HEADER_PRINTED
    
    if _HAS_RICH:
        console = _get_rich_console()
        if _SYMBOL_MAP_TABLE is not None and _SYMBOL_MAP_TABLE.row_count > 0:
            console.print()
            console.print(_SYMBOL_MAP_TABLE)
        if _SYMBOL_FAIL_TABLE is not None and _SYMBOL_FAIL_TABLE.row_count > 0:
            console.print()
            console.print(_SYMBOL_FAIL_TABLE)
    else:
        # For non-Rich output, tables were printed incrementally
        if _SYMBOL_MAP_HEADER_PRINTED:
            print(_map_border())
        if _SYMBOL_FAIL_HEADER_PRINTED:
            print(_fail_border())


def reset_symbol_tables() -> None:
    """Reset the symbol tables for a fresh run."""
    global _SYMBOL_MAP_TABLE, _SYMBOL_FAIL_TABLE, _SYMBOL_MAP_HEADER_PRINTED, _SYMBOL_FAIL_HEADER_PRINTED
    _SYMBOL_MAP_TABLE = None
    _SYMBOL_FAIL_TABLE = None
    _SYMBOL_MAP_HEADER_PRINTED = False
    _SYMBOL_FAIL_HEADER_PRINTED = False


def _safe_lookup(container, key: str) -> Optional[str]:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    if hasattr(container, key):
        return getattr(container, key)
    return None


def _detect_delisted_or_unsupported(symbol: str, ticker: yf.Ticker) -> Tuple[bool, str]:
    timezone = None
    exchange = None
    history_empty = False

    try:
        timezone = _safe_lookup(getattr(ticker, "fast_info", None), "timezone")
        if timezone is None:
            timezone = _safe_lookup(getattr(ticker, "info", None), "timezone")
    except Exception:
        timezone = None

    try:
        exchange = _safe_lookup(getattr(ticker, "fast_info", None), "exchange")
        if exchange is None:
            exchange = _safe_lookup(getattr(ticker, "info", None), "exchange")
    except Exception:
        exchange = None

    try:
        hist = ticker.history(period="1mo", auto_adjust=False)
        if hist is None or hist.empty:
            history_empty = True
    except Exception:
        history_empty = True

    if timezone in (None, ""):
        return True, "missing_timezone"
    if exchange in (None, ""):
        return True, "missing_exchange"
    if history_empty:
        return True, "empty_history"
    return False, ""


def normalize_yahoo_ticker(symbol: str, perform_lookup: bool = True) -> Tuple[str, Dict]:
    meta: Dict[str, object] = {"original": symbol}

    if symbol is None:
        meta["status"] = "invalid"
        _log_symbol_fail("None", "invalid", "empty symbol")
        return "", meta

    raw = str(symbol).strip()
    if raw == "":
        meta["status"] = "invalid"
        _log_symbol_fail(symbol, "invalid", "empty symbol")
        return raw, meta

    normalized = raw
    upper = raw.upper()

    # Punctuation and suffix grammar fixes
    if upper in PUNCTUATION_OVERRIDES:
        normalized = PUNCTUATION_OVERRIDES[upper]
        meta["type"] = "punctuation_fix"
        meta["reason"] = "Yahoo punctuation grammar"
        _log_symbol_map(raw, normalized, meta)
        upper = normalized.upper()

    if upper in COUNTRY_SUFFIX_COMPLETIONS:
        normalized = COUNTRY_SUFFIX_COMPLETIONS[upper]
        meta["type"] = meta.get("type", "suffix_completion")
        meta["reason"] = meta.get("reason", "country_exchange_suffix")
        _log_symbol_map(raw, normalized, meta)
        upper = normalized.upper()

    mapped_or_fixed = meta.get("type") is not None

    if upper in PROXY_OVERRIDES:
        normalized = PROXY_OVERRIDES[upper]
        meta["type"] = "proxy_override"
        meta["reason"] = "mapped_to_liquid_underlying"
        _log_symbol_map(raw, normalized, meta)
        mapped_or_fixed = True
        upper = normalized.upper()

    if upper in YAHOO_TICKER_OVERRIDES:
        normalized = YAHOO_TICKER_OVERRIDES[upper]
        meta["type"] = "override"
        meta["reason"] = "deterministic_override"
        _log_symbol_map(raw, normalized, meta)
        mapped_or_fixed = True
    elif upper in PRECIOUS_METAL_FX_TO_FUTURES:
        normalized = PRECIOUS_METAL_FX_TO_FUTURES[upper]
        meta["type"] = "commodity_proxy"
        meta["reason"] = "Yahoo spot unavailable"
        _log_symbol_map(raw, normalized, meta)
        mapped_or_fixed = True

    if perform_lookup and not mapped_or_fixed:
        try:
            tk = yf.Ticker(normalized)
            flagged, reason = _detect_delisted_or_unsupported(normalized, tk)
        except Exception:
            flagged = True
            reason = "ticker_init_failed"
        if flagged:
            meta["status"] = "delisted_or_unsupported"
            meta["reason"] = reason
            _log_symbol_fail(raw, "delisted_or_unsupported", reason)
            return normalized, meta

    meta["status"] = "ok"
    return normalized, meta


# -------------------------
# Asset Universe Configuration
# -------------------------

# -------------------------
# Company/Asset Display Names Dictionary
# -------------------------
# Internal dictionary to avoid external API calls for company names.
# This is used by _resolve_display_name() before falling back to Yahoo Finance API.
COMPANY_NAMES: Dict[str, str] = {
    # FX Pairs
    "PLNJPY=X": "Polish Zloty / Japanese Yen",
    "USDPLN=X": "US Dollar / Polish Zloty",
    "EURPLN=X": "Euro / Polish Zloty",
    "GBPPLN=X": "British Pound / Polish Zloty",
    "CHFPLN=X": "Swiss Franc / Polish Zloty",
    "JPYPLN=X": "Japanese Yen / Polish Zloty",
    "EURUSD=X": "Euro / US Dollar",
    "GBPUSD=X": "British Pound / US Dollar",
    "USDJPY=X": "US Dollar / Japanese Yen",
    
    # Commodities & Futures
    "GC=F": "Gold Futures (COMEX)",
    "SI=F": "Silver Futures (COMEX)",
    "PL=F": "Platinum Futures (NYMEX)",
    "PA=F": "Palladium Futures (NYMEX)",
    "CL=F": "Crude Oil Futures (WTI)",
    "NG=F": "Natural Gas Futures",
    "XAUUSD=X": "Gold Spot (XAU/USD)",
    "XAGUSD=X": "Silver Spot (XAG/USD)",
    
    # Cryptocurrency
    "BTC-USD": "Bitcoin (BTC/USD)",
    "ETH-USD": "Ethereum (ETH/USD)",
    "SOL-USD": "Solana (SOL/USD)",
    
    # Major ETFs
    "SPY": "SPDR S&P 500 ETF Trust",
    "VOO": "Vanguard S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust (Nasdaq-100)",
    "IWM": "iShares Russell 2000 ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    "GLD": "SPDR Gold Shares ETF",
    "SLV": "iShares Silver Trust ETF",
    "SMH": "VanEck Semiconductor ETF",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    
    # VanEck ETFs
    "AFK": "VanEck Africa Index ETF",
    "ANGL": "VanEck Fallen Angel High Yield Bond ETF",
    "CNXT": "VanEck ChiNext ETF",
    "DURA": "VanEck Morningstar Durable Dividend ETF",
    "EGPT": "VanEck Egypt Index ETF",
    "FLTR": "VanEck Investment Grade Floating Rate ETF",
    "GDX": "VanEck Gold Miners ETF",
    "GDXJ": "VanEck Junior Gold Miners ETF",
    "GLIN": "VanEck India Growth Leaders ETF",
    "MOTG": "VanEck Morningstar Global Wide Moat ETF",
    "IDX": "VanEck Indonesia Index ETF",
    "DFNG": "VanEck Defense ETF",
    
    # Information Technology
    "AAPL": "Apple Inc.",
    "ACN": "Accenture plc",
    "ADBE": "Adobe Inc.",
    "AMD": "Advanced Micro Devices, Inc.",
    "AVGO": "Broadcom Inc.",
    "CRM": "Salesforce, Inc.",
    "CSCO": "Cisco Systems, Inc.",
    "IBM": "International Business Machines Corporation",
    "INTC": "Intel Corporation",
    "INTU": "Intuit Inc.",
    "MSFT": "Microsoft Corporation",
    "NOW": "ServiceNow, Inc.",
    "NVDA": "NVIDIA Corporation",
    "ORCL": "Oracle Corporation",
    "PLTR": "Palantir Technologies Inc.",
    "QCOM": "QUALCOMM Incorporated",
    "TXN": "Texas Instruments Incorporated",
    "GOOG": "Alphabet Inc. (Class C)",
    "GOOGL": "Alphabet Inc. (Class A)",
    "META": "Meta Platforms, Inc.",
    "NFLX": "Netflix, Inc.",
    "MSTR": "MicroStrategy Incorporated",
    "IONQ": "IonQ, Inc.",
    "IOT": "Samsara Inc.",
    "SOUN": "SoundHound AI, Inc.",
    "SYM": "Symbotic Inc.",
    
    # Health Care
    "ABBV": "AbbVie Inc.",
    "ABT": "Abbott Laboratories",
    "AMGN": "Amgen Inc.",
    "BMY": "Bristol-Myers Squibb Company",
    "CVS": "CVS Health Corporation",
    "DHR": "Danaher Corporation",
    "GILD": "Gilead Sciences, Inc.",
    "ISRG": "Intuitive Surgical, Inc.",
    "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly and Company",
    "MDT": "Medtronic plc",
    "MRK": "Merck & Co., Inc.",
    "NVO": "Novo Nordisk A/S",
    "PFE": "Pfizer Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "REGN": "Regeneron Pharmaceuticals, Inc.",
    "BEAM": "Beam Therapeutics Inc.",
    "CRSP": "CRISPR Therapeutics AG",
    "CELH": "Celsius Holdings, Inc.",
    
    # Financials
    "AIG": "American International Group, Inc.",
    "AXP": "American Express Company",
    "BAC": "Bank of America Corporation",
    "BK": "The Bank of New York Mellon Corporation",
    "BLK": "BlackRock, Inc.",
    "BRK-B": "Berkshire Hathaway Inc. (Class B)",
    "BRK.B": "Berkshire Hathaway Inc. (Class B)",
    "C": "Citigroup Inc.",
    "COF": "Capital One Financial Corporation",
    "GS": "The Goldman Sachs Group, Inc.",
    "IBKR": "Interactive Brokers Group, Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "MA": "Mastercard Incorporated",
    "MET": "MetLife, Inc.",
    "MS": "Morgan Stanley",
    "PYPL": "PayPal Holdings, Inc.",
    "SCHW": "The Charles Schwab Corporation",
    "USB": "U.S. Bancorp",
    "V": "Visa Inc.",
    "WFC": "Wells Fargo & Company",
    "HOOD": "Robinhood Markets, Inc.",
    "PNC": "PNC Financial Services Group, Inc.",
    "TFC": "Truist Financial Corporation",
    "NU": "Nu Holdings Ltd.",
    
    # Consumer Discretionary
    "AMZN": "Amazon.com, Inc.",
    "BKNG": "Booking Holdings Inc.",
    "GM": "General Motors Company",
    "HD": "The Home Depot, Inc.",
    "LOW": "Lowe's Companies, Inc.",
    "MCD": "McDonald's Corporation",
    "NKE": "NIKE, Inc.",
    "SBUX": "Starbucks Corporation",
    "TGT": "Target Corporation",
    "TSLA": "Tesla, Inc.",
    "GRND": "Grindr Inc.",
    
    # Industrials
    "CAT": "Caterpillar Inc.",
    "DE": "Deere & Company",
    "EMR": "Emerson Electric Co.",
    "FDX": "FedEx Corporation",
    "MMM": "3M Company",
    "UBER": "Uber Technologies, Inc.",
    "UNP": "Union Pacific Corporation",
    "UPS": "United Parcel Service, Inc.",
    "HON": "Honeywell International Inc.",
    
    # Defense & Aerospace (US)
    "ACHR": "Archer Aviation Inc.",
    "AIR": "AAR Corp.",
    "AIRI": "Air Industries Group",
    "AIRO": "AIRO Group Holdings Inc.",
    "AOUT": "American Outdoor Brands, Inc.",
    "ASTC": "Astrotech Corporation",
    "ATI": "ATI Inc.",
    "ATRO": "Astronics Corporation",
    "AVAV": "AeroVironment, Inc.",
    "AXON": "Axon Enterprise, Inc.",
    "AZ": "A2Z Cust2Mate Solutions Corp.",
    "BA": "The Boeing Company",
    "BAH": "Booz Allen Hamilton Holding Corporation",
    "BETA": "Beta Technologies Inc.",
    "BWXT": "BWX Technologies, Inc.",
    "BYRN": "Byrna Technologies Inc.",
    "CACI": "CACI International Inc.",
    "CAE": "CAE Inc.",
    "CDRE": "Cadre Holdings, Inc.",
    "CODA": "Coda Octopus Group, Inc.",
    "CVU": "CPI Aerostructures, Inc.",
    "CW": "Curtiss-Wright Corporation",
    "DCO": "Ducommun Incorporated",
    "DFSC": "Defsec Technologies Inc.",
    "DPRO": "Draganfly Inc.",
    "DRS": "Leonardo DRS, Inc.",
    "EH": "EHang Holdings Limited",
    "EMBJ": "Embraer S.A.",
    "ESLT": "Elbit Systems Ltd.",
    "EVEX": "Eve Holding, Inc.",
    "EVTL": "Vertical Aerospace Ltd.",
    "FJET": "Starfighters Space Inc.",
    "FLY": "Firefly Aerospace Inc.",
    "FTAI": "FTAI Aviation Ltd.",
    "GD": "General Dynamics Corporation",
    "GE": "GE Aerospace",
    "GPUS": "Hyperscale Data Inc.",
    "HEI": "HEICO Corporation",
    "HII": "Huntington Ingalls Industries, Inc.",
    "HOVR": "New Horizon Aircraft Ltd.",
    "HWM": "Howmet Aerospace Inc.",
    "HXL": "Hexcel Corporation",
    "ISSC": "Innovative Solutions & Support, Inc.",
    "JOBY": "Joby Aviation, Inc.",
    "KITT": "Nauticus Robotics, Inc.",
    "KRMN": "Karman Holdings Inc.",
    "KTOS": "Kratos Defense & Security Solutions, Inc.",
    "LDOS": "Leidos Holdings, Inc.",
    "LHX": "L3Harris Technologies, Inc.",
    "LMT": "Lockheed Martin Corporation",
    "LOAR": "Loar Holdings Inc.",
    "LUNR": "Intuitive Machines, Inc.",
    "MANT": "ManTech International Corporation",
    "MNTS": "Momentus Inc.",
    "MOG.A": "Moog Inc. (Class A)",
    "MOG-A": "Moog Inc. (Class A)",
    "MRCY": "Mercury Systems, Inc.",
    "MSA": "MSA Safety Incorporated",
    "NOC": "Northrop Grumman Corporation",
    "NPK": "National Presto Industries, Inc.",
    "OPXS": "Optex Systems Holdings, Inc.",
    "OSK": "Oshkosh Corporation",
    "PEW": "Grabagun Digital Holdings Inc.",
    "PKE": "Park Aerospace Corp.",
    "PL": "Planet Labs PBC",
    "POWW": "AMMO, Inc.",
    "PRZO": "ParaZero Technologies Ltd.",
    "RCAT": "Red Cat Holdings, Inc.",
    "RDW": "Redwire Corporation",
    "RGR": "Sturm, Ruger & Company, Inc.",
    "RKLB": "Rocket Lab USA, Inc.",
    "RTX": "RTX Corporation",
    "SAIC": "Science Applications International Corporation",
    "SARO": "StandardAero, Inc.",
    "SATL": "Satellogic Inc.",
    "SIDU": "Sidus Space, Inc.",
    "SIF": "SIFCO Industries, Inc.",
    "SKYH": "Sky Harbour Group Corporation",
    "SPAI": "Safe Pro Group Inc.",
    "SPCE": "Virgin Galactic Holdings, Inc.",
    "SPR": "Spirit AeroSystems Holdings, Inc.",
    "SWBI": "Smith & Wesson Brands, Inc.",
    "TATT": "TAT Technologies Ltd.",
    "TDG": "TransDigm Group Incorporated",
    "TDY": "Teledyne Technologies Incorporated",
    "TXT": "Textron Inc.",
    "VSAT": "ViaSat, Inc.",
    "VSEC": "VSE Corporation",
    "VTSI": "VirTra, Inc.",
    "VVX": "V2X, Inc.",
    "VWAV": "Visionwave Holdings Inc.",
    "VOYG": "Voyager Technologies Inc.",
    "WWD": "Woodward, Inc.",
    "ASTS": "AST SpaceMobile, Inc.",
    "BKSY": "BlackSky Technology Inc.",
    
    # Defense & Aerospace (European)
    "RHM.DE": "Rheinmetall AG",
    "RHM": "Rheinmetall AG",
    "AIR.PA": "Airbus SE",
    "HO.PA": "Thales S.A.",
    "HO": "Thales S.A.",
    "HAG.DE": "Hensoldt AG",
    "HAG": "Hensoldt AG",
    "BA.L": "BAE Systems plc",
    "FACC.VI": "FACC AG",
    "FACC": "FACC AG",
    "MTX.DE": "MTU Aero Engines AG",
    "SAF.PA": "Safran SA",
    "SAF": "Safran SA",
    "HEIA.AS": "HEICO Corporation (Class A)",
    "THEON.AS": "Theon International PLC",
    "THEON": "Theon International PLC",
    "KOG.OL": "Kongsberg Gruppen ASA",
    "R3NK.DE": "Renk Group AG",
    "AM": "Dassault Aviation SA",
    "EXA.PA": "Exail Technologies",
    "EXA": "Exail Technologies",
    "FINMY": "Leonardo S.p.A. (ADR)",
    "SAABY": "Saab AB (ADR)",
    
    # Communication Services
    "CMCSA": "Comcast Corporation",
    "DIS": "The Walt Disney Company",
    "T": "AT&T Inc.",
    "TMUS": "T-Mobile US, Inc.",
    "VZ": "Verizon Communications Inc.",
    
    # Consumer Staples
    "CL": "Colgate-Palmolive Company",
    "COST": "Costco Wholesale Corporation",
    "KO": "The Coca-Cola Company",
    "MDLZ": "Mondelēz International, Inc.",
    "MO": "Altria Group, Inc.",
    "PEP": "PepsiCo, Inc.",
    "PG": "The Procter & Gamble Company",
    "PM": "Philip Morris International Inc.",
    "WMT": "Walmart Inc.",
    
    # Energy
    "COP": "ConocoPhillips",
    "CVX": "Chevron Corporation",
    "XOM": "Exxon Mobil Corporation",
    
    # Utilities
    "NEE": "NextEra Energy, Inc.",
    "SO": "The Southern Company",
    "DUK": "Duke Energy Corporation",
    
    # Real Estate
    "SPG": "Simon Property Group, Inc.",
    "AMT": "American Tower Corporation",
    
    # Materials
    "LIN": "Linde plc",
    "NEM": "Newmont Corporation",
    "AEM": "Agnico Eagle Mines Limited",
    "B": "Barnes Group Inc.",
    
    # Materials — Copper
    "FCX": "Freeport-McMoRan Inc.",
    "SCCO": "Southern Copper Corporation",
    "TECK": "Teck Resources Limited",
    "HBM": "Hudbay Minerals Inc.",
    "IVN.TO": "Ivanhoe Mines Ltd.",
    
    # Materials — Rare Earths & Strategic Minerals
    "MP": "MP Materials Corp.",
    "LYSCF": "Lynas Rare Earths Limited",
    "UUUU": "Energy Fuels Inc.",
    "ILKAF": "Iluka Resources Limited",
    
    # Materials — Lithium
    "ALB": "Albemarle Corporation",
    "SQM": "Sociedad Química y Minera de Chile S.A.",
    
    # Materials — Specialty metals
    "ATI": "ATI Inc.",
    "MTRN": "Materion Corporation",
    
    # Materials — Uranium & nuclear
    "CCJ": "Cameco Corporation",
    "DNN": "Denison Mines Corp.",
    
    # Materials — Mining infrastructure
    "ERMAY": "Eramet S.A.",
    "GLNCY": "Glencore plc",
    "RIO": "Rio Tinto Group",
    
    # Materials — Major Gold Producers
    "NEM": "Newmont Corporation",
    "GOLD": "Barrick Gold Corporation",
    "AEM": "Agnico Eagle Mines Limited",
    "KGC": "Kinross Gold Corporation",
    "CDE": "Coeur Mining, Inc.",
    "HL": "Hecla Mining Company",
    "IAUX": "i-80 Gold Corp.",
    "HYMC": "Hycroft Mining Holding Corporation",
    
    # Materials — Silver & Dual Gold/Silver Producers
    "PAAS": "Pan American Silver Corp.",
    "GORO": "Gold Resource Corporation",
    "USAS": "Americas Gold and Silver Corporation",
    
    # Materials — Streaming & Royalty Companies
    "WPM": "Wheaton Precious Metals Corp.",
    "RGLD": "Royal Gold, Inc.",
    "GROY": "Gold Royalty Corp.",
    
    # Materials — Juniors & Exploration
    "SLVR": "Silver Tiger Metals Inc.",
    "EXK": "Endeavour Silver Corp.",
    "SVM": "Silvercorp Metals Inc.",
    "AG": "First Majestic Silver Corp.",
    
    # Asian Tech & Manufacturing
    "005930.KS": "Samsung Electronics Co., Ltd.",
    
    # German Companies
    "TKA.DE": "thyssenkrupp AG",
    "TKA": "thyssenkrupp AG",
    "VOW3.DE": "Volkswagen AG (Preferred)",
    "VOW3": "Volkswagen AG (Preferred)",
    "BMW3.DE": "Bayerische Motoren Werke AG (Preferred)",
    "BAYN.DE": "Bayer AG",
    "BAYN": "Bayer AG",
    
    # Polish Companies
    "ACP": "Asseco Poland S.A.",
    "SNT": "Synektik S.A.",
    
    # Options/Structured Products
    "TSLD": "iShares Tesla Options ETF",
    "PLTI": "iShares Palantir Options ETF",
    "QQQO": "iShares Nasdaq 100 Options ETF",
    "METI": "iShares Meta Options ETF",
    "MSFI": "iShares Microsoft Options ETF",
    "MSTP": "YieldMax MSTR Option Income Strategy ETF",
    "MAGD": "iShares Magnificent 7 Options ETF",
    "GOOO": "iShares Alphabet Options ETF",
    "AMDI": "iShares AMD Options ETF",
    "AMZD": "iShares Amazon Options ETF",
    "AVGI": "iShares Broadcom Options ETF",
    "BABI": "iShares Alibaba Options ETF",
    "BABY": "iShares Alibaba Options ETF",
    "AAPI": "iShares Apple Options ETF",
    "GLDE": "iShares Gold Yield Options ETF",
    "SLVI": "iShares Silver Yield Options ETF",
    
    # Gold/Precious Metals
    "SGLP": "Invesco Physical Gold ETC",
    "GLDW": "WisdomTree Core Physical Gold",
    
    # E-commerce & Fintech
    "GLBE": "Global-E Online Ltd.",
    "AFRM": "Affirm Holdings, Inc.",
    "COIN": "Coinbase Global, Inc.",
    "HUBS": "HubSpot, Inc.",
    
    # VanEck ETFs (additional)
    "MLN": "VanEck Long Muni ETF",
    "MOAT": "VanEck Morningstar Wide Moat ETF",
    "MOO": "VanEck Agribusiness ETF",
    "MOTI": "VanEck Morningstar International Moat ETF",
    "NLR": "VanEck Uranium+Nuclear Energy ETF",
    "OIH": "VanEck Oil Services ETF",
    "PPH": "VanEck Pharmaceutical ETF",
    "REMX": "VanEck Rare Earth/Strategic Metals ETF",
    "ESPO": "VanEck Video Gaming and eSports ETF",
    "GFA": "VanEck Global Fallen Angel High Yield Bond ETF",
    "TRET": "VanEck Global Real Estate ETF",
    
    # Nuclear & Energy
    "OKLO": "Oklo Inc.",
    "CCJ": "Cameco Corporation",
    "UUUU": "Energy Fuels Inc.",
    "NXE": "NexGen Energy Ltd.",
    "DNN": "Denison Mines Corp.",
    "UEC": "Uranium Energy Corp.",
    "GEV": "GE Vernova Inc.",
    "LEU": "Centrus Energy Corp.",
    "SMR": "NuScale Power Corporation",
    
    # Critical Materials & Mining
    "MP": "MP Materials Corp.",
    "CRML": "Critical Metals PLC",
    "IDR": "Idaho Strategic Resources, Inc.",
    "FCX": "Freeport-McMoRan Inc.",
    "UAMY": "United States Antimony Corporation",
    
    # Drones & Robotics
    "ONDS": "Ondas Holdings Inc.",
    "UMAC": "Unusual Machines, Inc.",
    
    # AI Infrastructure & Data Centers
    "IREN": "Iris Energy Limited",
    "NBIS": "Nebius Group N.V.",
    "CIFR": "Cipher Mining Inc.",
    "CRWV": "CoreWeave, Inc.",
    "GLXY": "Galaxy Digital Holdings Ltd.",
    
    # Growth Screen Stocks
    "NUTX": "Nutex Health, Inc.",
    "MU": "Micron Technology, Inc.",
    "SANM": "Sanmina Corporation",
    "SEZL": "Sezzle Inc.",
    "AMCR": "Amcor plc",
    "PSIX": "Power Solutions International, Inc.",
    "DLO": "DLocal Limited",
    "COMM": "CommScope Holding Company, Inc.",
    "PGY": "Pagaya Technologies Ltd.",
    "FOUR": "Shift4 Payments, Inc.",
    
    # AI Infrastructure & Semiconductor Equipment
    "ASML": "ASML Holding N.V.",
    "LRCX": "Lam Research Corporation",
    "AMAT": "Applied Materials, Inc.",
    "TSM": "Taiwan Semiconductor Manufacturing Company Limited",
    "ARM": "Arm Holdings plc",
    "SMCI": "Super Micro Computer, Inc.",
    "ANET": "Arista Networks, Inc.",
    "SNPS": "Synopsys, Inc.",
    "CDNS": "Cadence Design Systems, Inc.",
    
    # Cloud & Cybersecurity Infrastructure
    "CRWD": "CrowdStrike Holdings, Inc.",
    "ZS": "Zscaler, Inc.",
    "DDOG": "Datadog, Inc.",
    "SNOW": "Snowflake Inc.",
    "MDB": "MongoDB, Inc.",
    
    # AI Power & Semiconductor Choke Points
    "NVTS": "Navitas Semiconductor Corporation",
    "WOLF": "Wolfspeed, Inc.",
    "AEHR": "Aehr Test Systems",
    "ALAB": "Astera Labs, Inc.",
    "CRDO": "Credo Technology Group Holding Ltd.",
    
    # Batteries & Energy Tech
    "ENVX": "Enovix Corporation",
    "QS": "QuantumScape Corporation",
    
    # TechBio / AI-Drug Discovery
    "RXRX": "Recursion Pharmaceuticals, Inc.",
    "SDGR": "Schrödinger, Inc.",
    "ABCL": "AbCellera Biologics Inc.",
    "NTLA": "Intellia Therapeutics, Inc.",
    "TEM": "Tempus AI, Inc.",
    
    # Quantum Computing
    "QBTS": "D-Wave Quantum Inc.",
    "ARQQ": "Arqit Quantum Inc.",
    "RGTI": "Rigetti Computing, Inc.",
    "QUBT": "Quantum Computing Inc.",
    
    # Space & Satellite
    "SPIR": "Spire Global, Inc.",
    "GSAT": "Globalstar, Inc.",
    "IRDM": "Iridium Communications Inc.",
    "MDA.TO": "MDA Space Ltd.",
    "MDALF": "MDA Space Ltd. (ADR)",
    
    # AI Software & Data
    "CFLT": "Confluent, Inc.",
    "ESTC": "Elastic N.V.",
    "PATH": "UiPath Inc.",
    
    # AI Hardware & Edge Compute
    "MRVL": "Marvell Technology, Inc.",
    "NXPI": "NXP Semiconductors N.V.",
    "ADI": "Analog Devices, Inc.",
    "ON": "ON Semiconductor Corporation",
    "ARM": "Arm Holdings plc",
    "000660.KS": "SK Hynix Inc.",
    "8035.T": "Tokyo Electron Limited",
    "6723.T": "Renesas Electronics Corporation",
    
    # AI Power Semiconductors
    "IFX.DE": "Infineon Technologies AG",
    "STM": "STMicroelectronics N.V.",
    "MPWR": "Monolithic Power Systems, Inc.",
    "AOSL": "Alpha and Omega Semiconductor Limited",
    "POWI": "Power Integrations, Inc.",
    "VSH": "Vishay Intertechnology, Inc.",
    "2308.TW": "Delta Electronics, Inc.",
    "ETN": "Eaton Corporation plc",
    
    # Industrial & Infrastructure
    "PWR": "Quanta Services, Inc.",
    "VRT": "Vertiv Holdings Co.",
    "JCI": "Johnson Controls International plc",
    
    # Advanced Materials & Manufacturing
    "CRS": "Carpenter Technology Corporation",
    "KMT": "Kennametal Inc.",
    
    # Biotech Platforms
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "ILMN": "Illumina, Inc.",
    "PACB": "Pacific Biosciences of California, Inc.",
    
    # Energy Transition & Storage
    "FLNC": "Fluence Energy, Inc.",
    "ENPH": "Enphase Energy, Inc.",
    "BE": "Bloom Energy Corporation",
    
    # Additional XAGUSD variant
    "XAGUSD": "Silver Spot (XAG/USD)",
}


def get_company_name(symbol: str) -> Optional[str]:
    """
    Get the display name for a symbol from the internal dictionary.
    
    This function provides a fast, offline lookup for company/asset names
    without making external API calls.
    
    Args:
        symbol: The ticker symbol (e.g., "AAPL", "GC=F")
        
    Returns:
        The company/asset display name if found, None otherwise.
        
    Example:
        >>> get_company_name("AAPL")
        'Apple Inc.'
        >>> get_company_name("UNKNOWN")
        None
    """
    if not symbol:
        return None
    sym = symbol.strip().upper()
    
    # Direct lookup
    if sym in COMPANY_NAMES:
        return COMPANY_NAMES[sym]
    
    # Try without exchange suffix
    base_sym = sym.split(".")[0].split("=")[0].split("-")[0]
    if base_sym in COMPANY_NAMES:
        return COMPANY_NAMES[base_sym]
    
    return None


# Default asset universe: comprehensive list of FX pairs, commodities, stocks, and ETFs
# This centralized constant is used by fx_pln_jpy_signals.py and tuning/tune_q_mle.py
# Individual scripts can override via command-line arguments
DEFAULT_ASSET_UNIVERSE = [
    # FX pairs
    "PLNJPY=X",
    # Commodities
    "GC=F",  # Gold futures
    "SI=F",  # Silver futures
    # Cryptocurrency
    "BTC-USD",
    "MSTR",  # MicroStrategy

    # -------------------------
    # Major Indices and Broad Market ETFs
    # -------------------------
    "SPY",   # SPDR S&P 500 ETF
    "VOO",   # Vanguard S&P 500 ETF
    "GLD",   # Gold ETF
    "SLV",   # Silver ETF

    # -------------------------
    # S&P 100 Companies by Sector
    # -------------------------

    # Information Technology
    "AAPL",   # Apple Inc.
    "ACN",    # Accenture
    "ADBE",   # Adobe Inc.
    "AMD",    # Advanced Micro Devices
    "AVGO",   # Broadcom
    "CRM",    # Salesforce
    "CSCO",   # Cisco
    "IBM",    # IBM
    "INTC",   # Intel
    "INTU",   # Intuit
    "MSFT",   # Microsoft
    "NOW",    # ServiceNow
    "NVDA",   # Nvidia
    "ORCL",   # Oracle Corporation
    "PLTR",   # Palantir Technologies
    "QCOM",   # Qualcomm
    "SMH",    # VanEck Semiconductor ETF
    "TXN",    # Texas Instruments
    "GOOG",   # Alphabet Inc. (Class C)
    "GOOGL",  # Alphabet Inc. (Class A)
    "META",   # Meta Platforms
    "NFLX",   # Netflix, Inc.

    # Health Care
    "ABBV",   # AbbVie
    "ABT",    # Abbott Laboratories
    "AMGN",   # Amgen
    "BMY",    # Bristol Myers Squibb
    "CVS",    # CVS Health
    "DHR",    # Danaher Corporation
    "GILD",   # Gilead Sciences
    "ISRG",   # Intuitive Surgical
    "JNJ",    # Johnson & Johnson
    "LLY",    # Eli Lilly and Company
    "MDT",    # Medtronic
    "MRK",    # Merck & Co.
    "NVO",    # Novo Nordisk
    "PFE",    # Pfizer
    "TMO",    # Thermo Fisher Scientific
    "UNH",    # UnitedHealth Group

    # Financials
    "AIG",    # American International Group
    "AXP",    # American Express
    "BAC",    # Bank of America
    "BK",     # BNY Mellon
    "BLK",    # BlackRock
    "BRK-B",  # Berkshire Hathaway (Class B)
    "C",      # Citigroup
    "COF",    # Capital One
    "GS",     # Goldman Sachs
    "IBKR",   # Interactive Brokers
    "JPM",    # JPMorgan Chase
    "MA",     # Mastercard
    "MET",    # MetLife
    "MS",     # Morgan Stanley
    "PYPL",   # PayPal
    "SCHW",   # Charles Schwab Corporation
    "USB",    # U.S. Bancorp
    "V",      # Visa Inc.
    "WFC",    # Wells Fargo
    "HOOD",   # Robinhood

    # Consumer Discretionary
    "AMZN",   # Amazon
    "BKNG",   # Booking Holdings
    "GM",     # General Motors
    "HD",     # Home Depot
    "LOW",    # Lowe's
    "MCD",    # McDonald's
    "NKE",    # Nike, Inc.
    "SBUX",   # Starbucks
    "TGT",    # Target Corporation
    "TSLA",   # Tesla, Inc.

    # Industrials
    "CAT",    # Caterpillar Inc.
    "DE",     # Deere & Company
    "EMR",    # Emerson Electric
    "FDX",    # FedEx
    "MMM",    # 3M
    "UBER",   # Uber
    "UNP",    # Union Pacific Corporation
    "UPS",    # United Parcel Service

    # Military
    "ACHR",   # Archer Aviation Inc
    "AIR",    # AAR Corp
    "AIRI",   # Air Industries Group
    "AIRO",   # AIRO Group Holdings Inc
    "AOUT",   # American Outdoor Brands Inc
    "ASTC",   # Astrotech Corp
    "ATI",    # ATI Inc
    "ATRO",   # Astronics Corporation
    "AVAV",   # AeroVironment, Inc.
    "AXON",   # Axon Enterprise, Inc.
    "AZ"     # A2Z Cust2Mate Solutions Corp
    "BA",     # Boeing Co
    "BAH",    # Booz Allen Hamilton Holding Corp
    "BETA",   # Beta Technologies Inc
    "BWXT",   # BWX Technologies, Inc.
    "BYRN",   # Byrna Technologies Inc
    "CACI",   # CACI International Inc
    "CAE",    # CAE Inc
    "CDRE",   # Cadre Holdings Inc
    "CODA",   # Coda Octopus Group Inc
    "CVU",    # CPI Aerostructures Inc
    "CW",     # Curtiss-Wright Corporation
    "DCO",    # Ducommun Incorporated
    "DFSC",   # Defsec Technologies Inc
    "DPRO",   # Draganfly Inc
    "DRS",    # Leonardo DRS Inc
    "EH",     # Ehang Holdings Ltd
    "EMBJ",   # Embraer SA
    "ESLT",   # Elbit Systems Ltd
    "EVEX",   # Eve Holding Inc
    "EVTL",   # Vertical Aerospace Ltd
    "FJET",   # Starfighters Space Inc
    "FLY",    # Firefly Aerospace Inc
    "FTAI",   # FTAI Aviation
    "GD",     # General Dynamics Corp
    "GE",     # GE Aerospace
    "GPUS",   # Hyperscale Data Inc
    "HEI",    # HEICO Corporation
    "HEIA.AS",   # HEICO Corp. (Class A)
    "HII",    # Huntington Ingalls Industries, Inc.
    "HOVR",   # New Horizon Aircraft Ltd
    "HWM",    # Howmet Aerospace Inc.
    "HXL",    # Hexcel Corporation
    "HON",    # Honeywell International Inc
    "ISSC",   # Innovative Solutions & Support Inc
    "JOBY",   # Joby Aviation
    "KITT",   # Nauticus Robotics Inc
    "KRMN",   # Karman Holdings Inc
    "KTOS",   # Kratos Defense & Security Solutions, Inc.
    "LDOS",   # Leidos Holdings, Inc.
    "LHX",    # L3Harris Technologies Inc
    "LMT",    # Lockheed Martin Corp
    "LOAR",   # Loar Holdings Inc
    "LUNR",   # Intuitive Machines Inc
    "MANT",   # ManTech International Corporation
    "MNTS",   # Momentus Inc
    "MOG.A",  # Moog Inc
    "MRCY",   # Mercury Systems, Inc.
    "MSA",    # MSA Safety Incorporated
    "NOC",    # Northrop Grumman Corp
    "NPK",    # National Presto Industries Inc
    "OPXS",   # Optex Systems Holdings Inc
    "OSK",    # Oshkosh Corporation
    "PEW",    # Grabagun Digital Holdings Inc
    "PKE",    # Park Aerospace Corp
    "PL",     # Planet Labs PBC
    "POWW",   # Outdoor Holding Co
    "PRZO",   # Parazero Technologies Ltd
    "RCAT",   # Red Cat Holdings, Inc.
    "RDW",    # Redwire Corp
    "RGR",    # Sturm Ruger & Co Inc
    "RKLB",   # Rocket Lab Corp
    "RTX",    # RTX Corp
    "SAIC",   # Science Applications International Corp
    "SARO",   # StandardAero Inc
    "SATL",   # Satellogic Inc
    "SIDU",   # Sidus Space Inc
    "SIF",    # SIFCO Industries Inc
    "SKYH",   # Sky Harbour Group Corp
    "SPAI",   # Safe Pro Group Inc
    "SPCE",   # Virgin Galactic Holdings Inc
    "SPR",    # Spirit AeroSystems Holdings, Inc.
    "SWBI",   # Smith & Wesson Brands Inc
    "TATT",   # TAT Technologies Ltd
    "TDG",    # TransDigm Group Inc
    "TDY",    # Teledyne Technologies Incorporated
    "TXT",    # Textron Inc.
    "VSAT",   # ViaSat Inc
    "VSEC",   # VSE Corporation
    "VTSI",   # Virtra Inc
    "VVX",    # V2X, Inc.
    "VWAV",   # Visionwave Holdings Inc
    "VOYG",   # Voyager Technologies Inc
    "WWD",    # Woodward, Inc.
    "RHM.DE", # Rheinmetall (German defense)
    "AIR.PA", # Airbus (European aerospace)
    "HO.PA",  # Thales (French defense electronics)
    "HAG.DE", # Hensoldt (German defense electronics)
    "BA.L",   # BAE Systems (British defense)
    "FACC.VI",# FACC AG (Austrian aerospace components)
    "MTX.DE", # MTU Aero Engines (German aerospace)

    # Communication Services
    "CMCSA",  # Comcast
    "DIS",    # Walt Disney Company (The)
    "T",      # AT&T
    "TMUS",   # T-Mobile US
    "VZ",     # Verizon

    # Consumer Staples
    "CL",     # Colgate-Palmolive
    "COST",   # Costco
    "KO",     # Coca-Cola Company (The)
    "MDLZ",   # Mondelēz International
    "MO",     # Altria
    "PEP",    # PepsiCo
    "PG",     # Procter & Gamble
    "PM",     # Philip Morris International
    "WMT",    # Walmart

    # Energy
    "COP",    # ConocoPhillips
    "CVX",    # Chevron Corporation
    "XOM",    # ExxonMobil

    # Utilities
#     "DUK",    # Duke Energy
    "NEE",    # NextEra Energy
    "SO",     # Southern Company

    # Real Estate
#     "AMT",    # American Tower
    "SPG",    # Simon Property Group

    # Materials
    "LIN",    # Linde plc
    "NEM",    # Newmont Mining

    # -------------------------
    # Materials — Copper (AI + grids + EVs bottleneck)
    # -------------------------
    "FCX",    # Freeport-McMoRan — Largest copper torque in public markets
    "SCCO",   # Southern Copper — Ultra-low cost curve, insane margins
    "TECK",   # Teck Resources — Copper + metallurgical coal, becoming copper pure-play
    "HBM",    # Hudbay Minerals — Mid-cap copper with asymmetric upside
    "IVN.TO", # Ivanhoe Mines — Kamoa-Kakula, best copper discovery in decades (Toronto)

    # -------------------------
    # Materials — Rare Earths & Strategic Minerals (non-China exposure)
    # -------------------------
    "MP",     # MP Materials — Only meaningful US rare-earth supply chain
    "LYSCF",  # Lynas Rare Earths — Cleanest ex-China REE operator
    "UUUU",   # Energy Fuels — Rare earth processing + uranium = strategic leverage
    "ILKAF",  # Iluka Resources — Critical mineral separation expertise

    # -------------------------
    # Materials — Lithium (survivors)
    # -------------------------
    "ALB",    # Albemarle — The Linde-like compounder of lithium
    "SQM",    # SQM — Lowest cost lithium brines
    # Note: ALTM (Arcadium Lithium) removed - ticker issues post Livent/Allkem merger

    # -------------------------
    # Materials — Specialty metals (defense, aerospace, AI hardware)
    # -------------------------
    "ATI",    # ATI Inc — Titanium + aerospace + defense
    "MTRN",   # Materion — Beryllium for satellites, defense, AI hardware
    # Note: TMST (TimkenSteel) removed - company went private in 2023

    # -------------------------
    # Materials — Uranium & nuclear materials (energy security)
    # -------------------------
    "CCJ",    # Cameco — The NVIDIA of uranium
    "DNN",    # Denison Mines — In-situ recovery tech, asymmetric upside

    # -------------------------
    # Materials — Mining infrastructure & processing leverage
    # -------------------------
    "ERMAY",  # Eramet — Nickel + manganese for batteries
    "GLNCY",  # Glencore — Physical control of supply chains
    "RIO",    # Rio Tinto — Copper + aluminum + iron + future lithium

    # -------------------------
    # Materials — Major Gold Producers (Large Caps & Core Drivers)
    # -------------------------
    "NEM",    # Newmont Corporation — World's largest gold producer
    "GOLD",   # Barrick Gold Corporation — Global mining, low cost curves
    "AEM",    # Agnico Eagle Mines — Diversified, politically stable regions
    "KGC",    # Kinross Gold Corporation — Mid-tier with exploration upside
    "CDE",    # Coeur Mining — Gold + silver, Nevada/Alaska assets
    "HL",     # Hecla Mining Company — Major silver with gold by-product
    "IAUX",   # i-80 Gold Corp — U.S. focused gold + silver producer
    "HYMC",   # Hycroft Mining Holding — Nevada project, high optionality

    # -------------------------
    # Materials — Silver & Dual Gold/Silver Producers
    # -------------------------
    "PAAS",   # Pan American Silver — Largest silver producer with gold by-product
    "GORO",   # Gold Resource Corporation — Mixed gold + silver portfolio
    "USAS",   # Americas Gold and Silver Corporation — Growth stage precious metals

    # -------------------------
    # Materials — Streaming & Royalty Companies (Higher Margin, Lower Capex Risk)
    # -------------------------
    "WPM",    # Wheaton Precious Metals — Streaming, gold + silver exposure
    "RGLD",   # Royal Gold — High-quality precious metals royalty
    "GROY",   # Gold Royalty Corp — Precious metals royalty with growth optionality

    # -------------------------
    # Materials — Juniors & Exploration / Optionality Plays
    # -------------------------
    "SLVR",   # Silver Tiger Metals — High-grade silver + gold exploration
    "EXK",    # Endeavour Silver Corp — Producing silver miner, expansion catalysts
    "SVM",    # Silvercorp Metals — China & North America silver/gold diversified
    "AG",     # First Majestic Silver Corp — Silver producer, Mexico portfolio

    # Asian Tech & Manufacturing
    "005930.KS", # Samsung Electronics (Korean)

    # Banks (US top 10)
    "JPM",   # JPMorgan Chase
    "BAC",   # Bank of America
    "WFC",   # Wells Fargo
    "C",     # Citigroup
    "USB",   # U.S. Bancorp
    "PNC",   # PNC Financial Services
    "TFC",   # Truist Financial
    "GS",    # Goldman Sachs
    "MS",    # Morgan Stanley
    "BK",    # Bank of New York Mellon

    # Additional requested tickers
    "SOUN",  # SoundHound AI Inc-A
    "SYM",   # Symbotic Inc
    "THEON.AS", # Theon International
    "TKA",   # Thyssen-Krupp AG
    "TSLD",  # IS Tesla Options
    "VOW3",  # Volkswagen AG-Pref
    "XAGUSD",# London Silver Commodity
    "PLTI",  # IS Palantir Options
    "QQQO",  # IS Nasdaq 100 Options
    "R3NK.DE",  # Renk Group AG
    "REGN",  # Regeneron Pharmaceuticals
    "RHM.DE",   # Rheinmetall AG
    "RKLB",  # Rocket Lab Corp
    "SAABY", # Saab AB Unsponsored ADR
    "SAF",   # Safran SA
    "SGLP",  # Invesco Physical Gold
    "SLVI",  # IS Silver Yield Options
    "SNT",   # Synektik SA
    "METI",  # IS Meta Options
    "MRCY",  # Mercury Systems Inc
    "MSFI",  # IS Microsoft Options
    "MSTP",  # YieldMax MSTR Option Income
    "NU",    # Nu Holdings Ltd
    "IONQ",  # IonQ Inc
    "IOT",   # Samsara Inc
    "KOG.OL",  # Kongsberg Gruppen ASA
    "MAGD",  # IS Magnificent 7 Options
    "FACC",  # FACC AG
    "FINMY", # Leonardo SPA ADR
    "GLBE",  # Global-E Online Ltd
    "GLDE",  # IS Gold Yield Options
    "GLDW",  # WisdomTree Core Physical Gold
    "GOOO",  # IS Alphabet Options
    "GRND",  # Grindr Inc
    "HAG",   # Hensoldt AG
    "HO.PA",    # Thales SA - !!!!Temp change
    "BAYN",  # Bayer AG
    "BEAM",  # Beam Therapeutics Inc
    "BKSY",  # BlackSky Technology
    "BMW3.DE",  # Bayerische Motoren Werke AG
    "CELH",  # Celsius Holdings Inc
    "CRSP",  # CRISPR Therapeutics AG
    "CW",    # Curtiss-Wright Corp
    "DFNG",  # VanEck Defense ETF
    "DPRO",  # Draganfly Inc
    "ESLT",  # Elbit Systems Ltd
    "EXA",   # Exail Technologies
    "ASTS",  # AST SpaceMobile Inc
    "AVAV",  # AeroVironment Inc
    "AVGI",  # IS Broadcom Options
    "AVGO",  # Broadcom Inc
    "B",     # Barrick Mining Corp
    "BABI",  # IS Alibaba Options
    "BABY",  # IS Alibaba Options (alt)
    "AAPI",  # IS Apple Options
    "ACP",   # Asseco Poland SA
    "AEM",   # Agnico Eagle Mines Ltd
    "AIR",   # Airbus SE
    "AIRI",  # Air Industries Group
    "AM",    # Dassault Aviation SA
    "AMD",   # Advanced Micro Devices Inc
    "AMDI",  # IS AMD Options
    "AMZD",  # IS Amazon Options
    "AMZN",  # Amazon.com Inc
    "FTAI",  # FTAI Aviation Ltd
    "MSTR",  # MicroStrategy Inc

    # -------------------------
    # VanEck ETFs
    # -------------------------
    "AFK",    # VanEck Africa Index ETF
    "ANGL",   # VanEck Fallen Angel High Yield Bond ETF
#     "BRF",    # VanEck Brazil Small-Cap ETF
    "CNXT",   # VanEck ChiNext ETF
    "DURA",   # VanEck Morningstar Durable Dividend ETF
    "EGPT",   # VanEck Egypt Index ETF
#     "EMLC",   # VanEck J.P. Morgan EM Local Currency Bond ETF
    "FLTR",   # VanEck Investment Grade Floating Rate ETF
    "GDX",    # VanEck Gold Miners ETF
    "GDXJ",   # VanEck Junior Gold Miners ETF
    "GLIN",   # VanEck India Growth Leaders ETF
    "MOTG",   # VanEck Morningstar Global Wide Moat ETF
#     "GRNB",   # VanEck Green Bond ETF
#     "HYEM",   # VanEck Emerging Markets High Yield Bond ETF
    "IDX",    # VanEck Indonesia Index ETF
#     "ITM",    # VanEck Intermediate Muni ETF
    "MLN",    # VanEck Long Muni ETF
    "MOAT",   # VanEck Morningstar Wide Moat ETF
    "MOO",    # VanEck Agribusiness ETF
    "MOTI",   # VanEck Morningstar International Moat ETF
    "NLR",    # VanEck Uranium+Nuclear Energy ETF
    "OIH",    # VanEck Oil Services ETF
    "PPH",    # VanEck Pharmaceutical ETF
    "REMX",   # VanEck Rare Earth/Strategic Metals ETF
#     "RSX",    # VanEck Russia ETF
#     "RSXJ",   # VanEck Russia Small-Cap ETF
#     "RTH",    # VanEck Retail ETF
#     "SLX",    # VanEck Steel ETF
#     "SMOG",   # VanEck Low Carbon Energy ETF
#     "VNM",    # VanEck Vietnam ETF
    "ESPO",   # VanEck Video Gaming and eSports UCITS ETF
    "GFA",    # VanEck Global Fallen Angel High Yield Bond UCITS ETF
#     "HDRO",   # VanEck Hydrogen Economy UCITS ETF
#     "TCBT",   # VanEck iBoxx EUR Corporates UCITS ETF
#     "TDIV",   # VanEck Morningstar Developed Markets Dividend Leaders UCITS ETF
#     "TEET",   # VanEck Sustainable European Equal Weight UCITS ETF
#     "TGBT",   # VanEck iBoxx EUR Sovereign Diversified 1-10 UCITS ETF
    "TRET",   # VanEck Global Real Estate UCITS ETF
#     "TSWE",   # VanEck Sustainable World Equal Weight UCITS ETF
#     "TAT",    # VanEck iBoxx EUR Sovereign Capped AAA-AA 1-5 UCITS ETF

    # -------------------------
    # Thematic additions
    # -------------------------
    # Nuclear / Uranium
    "OKLO",   # Oklo Inc (nuclear reactors)
    "CCJ",    # Cameco Corporation (uranium mining)
    "UUUU",   # Energy Fuels Inc (uranium/rare earth)
    "NXE",    # NexGen Energy Ltd (uranium exploration)
    "DNN",    # Denison Mines Corp (uranium mining)
    "UEC",    # Uranium Energy Corp (uranium mining)
    "GEV",    # GE Vernova (power generation)
    "LEU",    # Centrus Energy Corp (uranium enrichment)
    # Critical materials
    "MP",
    "CRML",
    "IDR",
    "FCX",
    "UAMY",
    # Space
    "RKLB",
    "ASTS",
    "PL",
    "BKSY",
    "LUNR",
    # Drones
    "ONDS",
    "UMAC",
    "AVAV",
    "KTOS",
    "DPRO",
    # AI utility / infrastructure
    "IREN",
    "NBIS",
    "CIFR",
    "CRWV",
    "GLXY",
    # Growth Screen (Michael Kao list)
    "NUTX",
    "RCAT",
    "MU",
    "SANM",
    "SEZL",
    "AMCR",
    "PSIX",
    "DLO",
    "COMM",
    "PGY",
    "FOUR",

    # -------------------------
    # AI Infrastructure & Semiconductor Equipment
    # -------------------------
    "ASML",   # ASML Holding — Monopoly in EUV lithography, gatekeeper of advanced chips
    "LRCX",   # Lam Research — Critical wafer fabrication equipment, AI fabs leverage
    "AMAT",   # Applied Materials — Broadest semiconductor tooling exposure
    "TSM",    # TSMC — Foundry backbone of AI civilization
    "ARM",    # Arm Holdings — Architecture layer for future compute
    "SMCI",   # Super Micro Computer — AI server systems integrator
    "ANET",   # Arista Networks — AI data-center networking spine
    "SNPS",   # Synopsys — Chip design software monopoly layer
    "CDNS",   # Cadence Design — Digital silicon design infrastructure
    "000660.KS", # SK Hynix — Dominant in HBM3/HBM4, NVIDIA's memory supplier
    "8035.T",    # Tokyo Electron — Japanese precision, leading-edge process control

    # -------------------------
    # AI Hardware / Edge Compute
    # -------------------------
    "MRVL",   # Marvell Technology — Networking + storage + custom silicon
    "NXPI",   # NXP Semiconductors — Edge AI + automotive brains
    "ADI",    # Analog Devices — Sensor-to-AI interface layer
    "ON",     # ON Semiconductor — EV + power + edge AI
    "6723.T", # Renesas Electronics — Embedded + power control, industrial AI

    # -------------------------
    # AI Power Semiconductors
    # -------------------------
    "IFX.DE", # Infineon Technologies — World leader in power electronics, SiC
    "STM",    # STMicroelectronics — Strong in SiC and industrial power
    "TXN",    # Texas Instruments — Analog power, decades-long compounder
    "MPWR",   # Monolithic Power Systems — Elite margins, AI server power
    "AOSL",   # Alpha and Omega Semiconductor — High-voltage MOSFETs
    "POWI",   # Power Integrations — Power efficiency specialists
    "VSH",    # Vishay Intertechnology — Essential components
    "2308.TW", # Delta Electronics — Power supplies for hyperscale data centers
    "ETN",    # Eaton Corporation — Grid, UPS, power conditioning

    # -------------------------
    # Industrial / Infrastructure Transition
    # -------------------------
    "CRWD",   # CrowdStrike — AI-native cybersecurity platform
    "ZS",     # Zscaler — Zero-trust security backbone
    "DDOG",   # Datadog — Cloud observability nervous system
    "SNOW",   # Snowflake — Data-cloud substrate for AI
    "MDB",    # MongoDB — Modern data infrastructure layer

    # -------------------------
    # Fintech & Digital Finance
    # -------------------------
    "HUBS",   # HubSpot — SMB operating system
    # "SQ",   # Block — Fintech + merchant + crypto convergence (disabled: Yahoo Finance YFTzMissingError)
    "AFRM",   # Affirm — Consumer credit restructuring layer
    "COIN",   # Coinbase — Regulated crypto financial infrastructure
    # Note: NU (Nubank) already included above

    # -------------------------
    # AI Power & Semiconductor Choke Points
    # -------------------------
    "NVTS",   # Navitas Semiconductor — GaN/SiC power for AI data centers; long power-cycle optionality
    "WOLF",   # Wolfspeed — SiC substrate scale; post-restructuring rebound optionality
    "AEHR",   # Aehr Test Systems — SiC test bottleneck; picks-and-shovels
    "ALAB",   # Astera Labs — AI interconnect/PCIe/CXL infrastructure optionality
    "CRDO",   # Credo Technology — High-speed connectivity for AI racks

    # -------------------------
    # Batteries & Energy Tech Moonshots
    # -------------------------
    "ENVX",   # Enovix — 3D battery architecture; nonlinear upside if it scales
    "QS",     # QuantumScape — Solid-state battery; classic "works or doesn't" asymmetry
    "SMR",    # NuScale Power — Small modular nuclear optionality if deployment inflects

    # -------------------------
    # TechBio / AI-Drug Discovery Platforms
    # -------------------------
    "RXRX",   # Recursion — AI+biotech platform bet; NVIDIA collaboration tailwind
    "SDGR",   # Schrödinger — Computational chemistry stack
    "ABCL",   # AbCellera — Antibody discovery platform economics
    "NTLA",   # Intellia — In-vivo gene editing optionality
    "TEM",    # Tempus AI — Clinical data + AI diagnostics flywheel optionality

    # -------------------------
    # Quantum / Security Optionality
    # -------------------------
    "QBTS",   # D-Wave Quantum — Quantum systems & services; asymmetric if adoption accelerates
    "ARQQ",   # Arqit Quantum — Post-quantum security; very high dispersion
    "RGTI",   # Rigetti Computing — Full-stack quantum computing
    "QUBT",   # Quantum Computing Inc. — Quantum solutions

    # -------------------------
    # Space / Spectrum / Network Rails
    # -------------------------
    "SPIR",   # Spire Global — Space data constellation economics
    "GSAT",   # Globalstar — Spectrum asset optionality
    "IRDM",   # Iridium — Satcom network rail with upside skew
    "MDA.TO", # MDA Space — Satellite systems & robotics backbone
    "MDALF",  # MDA Space (US ADR)

    # -------------------------
    # AI Software / Data Gravity
    # -------------------------
    "CFLT",   # Confluent — Kafka data spine of real-time AI systems
    "ESTC",   # Elastic — Search + observability + AI substrate
    "PATH",   # UiPath — AI automation layer

    # -------------------------
    # Industrial / Infrastructure Transition
    # -------------------------
    "PWR",    # Quanta Services — Grid + energy infrastructure rebuild
    "VRT",    # Vertiv — Data-center power & cooling backbone
    "JCI",    # Johnson Controls — Smart buildings, energy efficiency OS

    # -------------------------
    # Advanced Materials / Manufacturing
    # -------------------------
    "CRS",    # Carpenter Technology — Specialty alloys for aerospace/nuclear/hypersonic
    "KMT",    # Kennametal — Cutting tools for advanced manufacturing

    # -------------------------
    # Biotech Platforms (Genomics / Sequencing)
    # -------------------------
    "VRTX",   # Vertex Pharma — Genetic disease cash-flow engine
    "ILMN",   # Illumina — Genomics sequencing infrastructure
    "PACB",   # PacBio — Long-read sequencing optionality

    # -------------------------
    # Energy Transition / Storage / Grid
    # -------------------------
    "FLNC",   # Fluence Energy — Grid-scale battery systems
    "ENPH",   # Enphase Energy — Solar + storage intelligence
    "BE",     # Bloom Energy — Solid oxide fuel cells

    # =========================================================================
    # SMALL-MID CAP SCREENER PICKS (Top 100 Revenue Growth Screener)
    # =========================================================================

    # -------------------------
    # Small-Mid Cap — Finance (🟢 Institutional-Grade ≥70)
    # -------------------------
    "AGNC",   # AGNC Investment Corp. — Mortgage REIT, institutional dividend
    "ABTC",   # AMERIBANCORP, Inc. — Regional banking, high growth
    "AXG",    # Axos Financial, Inc. — Digital banking platform
    "ASB",    # Associated Banc-Corp — Regional bank, Midwest footprint
    "ANGX",   # Angel Oak Financial Strategies — Specialty finance

    # -------------------------
    # Small-Mid Cap — Biotech/Healthcare (🟢 Institutional-Grade ≥70)
    # -------------------------
    "ANNA",   # Annovis Bio, Inc. — Neurodegenerative disease therapeutics
    "BBIO",   # BridgeBio Pharma, Inc. — Genetic disease platform
    "ATAI",   # atai Life Sciences N.V. — Mental health biotech
    "APLM",   # Apollomics, Inc. — Oncology-focused biotech

    # -------------------------
    # Small-Mid Cap — Technology/Space (🟢 Institutional-Grade ≥70)
    # -------------------------
    # "ASTS",  # AST SpaceMobile, Inc. — Already in universe (Space section)
    "BZAI",   # Banzai International, Inc. — AI-powered engagement platform

    # -------------------------
    # Small-Mid Cap — Real Estate (🟢 Institutional-Grade ≥70)
    # -------------------------
    "AIRE",   # reAlpha Tech Corp. — AI-driven real estate platform

    # -------------------------
    # Small-Mid Cap — Finance (🟡 Tradeable 60-69)
    # -------------------------
    "BMHL",   # BM Technologies, Inc. — Digital banking fintech
    "BNZI",   # Banzai International, Inc. — Engagement tech (alt ticker)
    "BNKK",   # BNK Financial Group, Inc. — Korean financial services
    "BTCS",   # BTCS Inc. — Blockchain infrastructure
    "BCAL",   # California BanCorp — Regional California bank
    "ARR",    # ARMOUR Residential REIT, Inc. — Mortgage REIT

    # -------------------------
    # Small-Mid Cap — Biotech/Healthcare (🟡 Tradeable 60-69)
    # -------------------------
    "ALNY",   # Alnylam Pharmaceuticals, Inc. — RNAi therapeutics leader
    "APLS",   # Apellis Pharmaceuticals, Inc. — Complement pathway therapies
    "APLT",   # Applied Therapeutics, Inc. — Rare disease treatments
    "BETA",   # Beta Bionics, Inc. — Diabetes management tech

    # -------------------------
    # Small-Mid Cap — Technology/Industrial (🟡 Tradeable 60-69)
    # -------------------------
    "ASPI",   # ASP Isotopes Inc. — Isotope enrichment technology
    "ABAT",   # American Battery Technology Company — Battery recycling
    "ADUR",   # Aduro Clean Technologies — Chemical recycling tech
    "APLD",   # Applied Digital Corporation — AI data center infrastructure
    "ALMU",   # Aeluma, Inc. — Compound semiconductor materials
    "AMZE",   # Amaze Holdings, Inc. — Technology solutions
    "AIFF",   # Firefly Neuroscience, Inc. — AI neuroscience platform
]

MAPPING = {
    # Prefer active, liquid proxies first to avoid Yahoo "possibly delisted" noise
    "GOOO": ["GOOG", "GOOGL", "GOOO"],
    "GLDW": ["GLD", "GLDM", "GLDW"],
    "SGLP": ["SGLP.L", "SGLP", "SGLP.LON"],
    "GLDE": ["GLD", "IAU", "GLDE"],
    "FACC": ["FACC.VI", "FACC"],
    "SLVI": ["SLV", "SLVP", "SLVI"],
    "TKA": ["TKA.DE", "TKA"],

    # Silver spot and commodity proxies
    "XAGUSD": ["SI=F", "SLV"],
    "XAGUSD=X": ["SI=F", "SLV"],

    # German/European tickers that need suffix completion
    "VOW3": ["VOW3.DE", "VOW3"],
    "BMW3": ["BMW3.DE", "BMW3"],
    "BAYN": ["BAYN.DE", "BAYN"],
    "HAG": ["HAG.DE", "HAG"],
    "R3NK": ["R3NK.DE", "R3NK"],
    "KOZ1": ["KOG.OL", "KOZ1"],
    "EXA": ["EXA.PA", "EXA"],
    "THEON": ["THEON.AS", "THEON"],

    # Canadian tickers
    "MDA": ["MDA.TO", "MDALF"],
    "MDA.TO": ["MDA.TO", "MDALF"],
    "MDALF": ["MDALF", "MDA.TO"],
    "IVN": ["IVN.TO"],  # Ivanhoe Mines - Canadian
    "IVN.TO": ["IVN.TO"],

    # YieldMax and structured product proxies (route to underlying)
    "QQQO": ["QQQ"],
    "MAGD": ["QQQ"],
    "TSLD": ["TSLA"],
    "METI": ["META"],
    "MSFI": ["MSFT"],
    "AVGI": ["AVGO"],
    "BABI": ["BABA"],
    "BABY": ["BABA"],
    "AMDI": ["AMD"],
    "DFNG": ["ITA"],
    "GFA": ["ANGL"],
    "TRET": ["VNQ"],

    # Share class and punctuation variants (explicit mappings)
    "BRK.B": ["BRK-B"],
    "HEIA": ["HEI-A", "HEI.A", "HEI"],
    "MOG.A": ["MOG-A"],
    "MANT": ["CACI"],

    # Netflix and Novo Nordisk
    "NFLX": ["NFLX"],
    "NOVO": ["NVO", "NOVO-B.CO", "NOVOB.CO", "NOVO-B.CO"],

    # Kratos alias
    "KRATOS": ["KTOS"],

    # Requested blue chips and defense/aero additions
    "RHEINMETALL": ["RHM.DE", "RHM.F", "RHM"],
    "AIRBUS": ["AIR.PA", "AIR.DE"],
    "RENK": ["R3NK.DE", "RNK.DE", "R3NK"],
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
    "THALES": ["HO.PA", "HO"],
    "HENSOLDT": ["HAG.DE", "HAG"],
    "SAMSUNG": ["005930.KS", "005935.KS"],
    "TKMS AG & CO": ["TKA.DE", "TKAMY"],
    "BAE SYSTEMS": ["BA.L"],
    "BAE": ["BA.L"],
    "NEWMONT": ["NEM"],
    "NEWMONT CORP": ["NEM"],
    "HOWMET": ["HWM"],
    "HOWMET AEROSPACE": ["HWM"],
    "BROADCOM": ["AVGO"],

    # SPY, S&P 500, Magnificent 7, and Semiconductor ETFs
    "SPY": ["SPY"],
    "SP500": ["^GSPC", "SPY"],
    "S&P500": ["^GSPC", "SPY"],
    "S&P 500": ["^GSPC", "SPY"],
    "MAGNIFICENT 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "MAGNIFICENT SEVEN": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "MAG7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "GOOGLE": ["GOOGL", "GOOG"],
    "ALPHABET": ["GOOGL", "GOOG"],
    "AMAZON": ["AMZN"],
    "META": ["META"],
    "FACEBOOK": ["META"],
    "SEMICONDUCTOR ETF": ["SMH", "SOXX"],
    "SMH": ["SMH"],
    "SOXX": ["SOXX"],

    # Identity / alias candidates
    "RKLB": ["RKLB"],
    "MTX.DE": ["MTX.DE"],
    "IBKR": ["IBKR"],
    "HOOD": ["HOOD"],

    # Thematic additions
    "OKLO": ["OKLO"],
    "OKLO INC": ["OKLO"],
    "CAMECO": ["CCJ"],
    "CAMECO CORPORATION": ["CCJ"],
    "ENERGY FUELS": ["UUUU"],
    "GE VERNOVA": ["GEV"],
    "CENTRUS ENERGY": ["LEU"],
    "MP MATERIALS": ["MP"],
    "CRITICAL METALS": ["CRML"],
    "IDAHO STRATEGIC RESOURCES": ["IDR"],
    "FREEPORT": ["FCX"],
    "FREEPORT-MCMORAN": ["FCX"],
    "UNITED STATES ANTIMONY": ["UAMY"],
    "U.S. ANTIMONY": ["UAMY"],
    "ONDAS": ["ONDS"],
    "UNUSUAL MACHINES": ["UMAC"],
    "AEROVIRONMENT": ["AVAV"],
    "KRATOS": ["KTOS"],
    "DRAGANFLY": ["DPRO"],
    "IRIS ENERGY": ["IREN"],
    "NEBIUS": ["NBIS"],
    "CIPHER MINING": ["CIFR"],
    "COREWEAVE": ["CRWV"],
    "GALAXY DIGITAL": ["GLXY"],
    "AST SPACEMOBILE": ["ASTS"],
    "BLACKSKY": ["BKSY"],
    "INTUITIVE MACHINES": ["LUNR"],
    "NUTEX": ["NUTX"],
    "RED CAT": ["RCAT"],
    "MICRON": ["MU"],
    "SEI INVESTMENTS": ["SEI", "SEIC"],
    "SANMINA": ["SANM"],
    "SEZZLE": ["SEZL"],
    "AMCOR": ["AMCR"],
    "POWER SOLUTIONS": ["PSIX"],
    "DLOCAL": ["DLO"],
    "COMMSCOPE": ["COMM"],
    "PAGAYA": ["PGY"],
    "SHIFT4": ["FOUR"],

    # S&P 100 additions by sector
    # Information Technology
    "ACCENTURE": ["ACN"],
    "ADOBE": ["ADBE"],
    "SALESFORCE": ["CRM"],
    "CISCO": ["CSCO"],
    "INTEL": ["INTC"],
    "ORACLE": ["ORCL"],
    "PALANTIR": ["PLTR"],
    "QUALCOMM": ["QCOM"],
    "TEXAS INSTRUMENTS": ["TXN"],

    # Health Care
    "ABBVIE": ["ABBV"],
    "ABBOTT LABS": ["ABT"],
    "AMGEN": ["AMGN"],
    "BRISTOL MYERS SQUIBB": ["BMY"],
    "CVS HEALTH": ["CVS"],
    "DANAHER": ["DHR"],
    "GILEAD SCIENCES": ["GILD"],
    "INTUITIVE SURGICAL": ["ISRG"],
    "JOHNSON & JOHNSON": ["JNJ"],
    "ELI LILLY": ["LLY"],
    "MEDTRONIC": ["MDT"],
    "MERCK": ["MRK"],
    "THERMO FISHER SCIENTIFIC": ["TMO"],
    "UNITEDHEALTH": ["UNH"],

    # Financials
    "AMERICAN INTERNATIONAL GROUP": ["AIG"],
    "AMERICAN EXPRESS": ["AXP"],
    "BANK OF AMERICA": ["BAC"],
    "BNY MELLON": ["BK"],
    "BLACKROCK": ["BLK"],
    "BERKSHIRE HATHAWAY": ["BRK-B"],
    "CITIGROUP": ["C"],
    "CAPITAL ONE": ["COF"],
    "GOLDMAN SACHS": ["GS"],
    "JPMORGAN CHASE": ["JPM"],
    "MASTERCARD": ["MA"],
    "METLIFE": ["MET"],
    "MORGAN STANLEY": ["MS"],
    "PAYPAL": ["PYPL"],
    "CHARLES SCHWAB": ["SCHW"],
    "US BANCORP": ["USB"],
    "VISA": ["V"],
    "WELLS FARGO": ["WFC"],

    # Consumer Discretionary
    "BOOKING HOLDINGS": ["BKNG"],
    "GENERAL MOTORS": ["GM"],
    "HOME DEPOT": ["HD"],
    "LOWES": ["LOW"],
    "MCDONALDS": ["MCD"],
    "NIKE": ["NKE"],
    "STARBUCKS": ["SBUX"],
    "TARGET": ["TGT"],

    # Industrials
    "CATERPILLAR": ["CAT"],
    "DEERE & COMPANY": ["DE"],
    "EMERSON ELECTRIC": ["EMR"],
    "FEDEX": ["FDX"],
    "3M": ["MMM"],
    "UNION PACIFIC": ["UNP"],
    "UPS": ["UPS"],

    # Military / Defense & Aerospace
    "BOEING": ["BA"],
    "ARCHER AVIATION": ["ACHR"],
    "AAR CORP": ["AIR"],
    "AIR INDUSTRIES GROUP": ["AIRI"],
    "AIRO GROUP HOLDINGS": ["AIRO"],
    "AMERICAN OUTDOOR BRANDS": ["AOUT"],
    "ASTROTECH": ["ASTC"],
    "ATI": ["ATI"],
    "ASTRONICS": ["ATRO"],
    "AEROVIRONMENT": ["AVAV"],
    "AXON ENTERPRISE": ["AXON"],
    "A2Z CUST2MATE SOLUTIONS": ["AZ"],
    "BOOZ ALLEN HAMILTON": ["BAH"],
    "BETA TECHNOLOGIES": ["BETA"],
    "BWX TECHNOLOGIES": ["BWXT"],
    "BYRNA TECHNOLOGIES": ["BYRN"],
    "CACI INTERNATIONAL": ["CACI"],
    "CAE": ["CAE"],
    "CADRE HOLDINGS": ["CDRE"],
    "CODA OCTOPUS GROUP": ["CODA"],
    "CPI AEROSTRUCTURES": ["CVU"],
    "CURTISS-WRIGHT": ["CW"],
    "DUCOMMUN": ["DCO"],
    "DEFSEC TECHNOLOGIES": ["DFSC"],
    "DRAGANFLY": ["DPRO"],
    "LEONARDO DRS": ["DRS"],
    "EHANG HOLDINGS": ["EH"],
    "EMBRAER": ["EMBJ"],
    "ELBIT SYSTEMS": ["ESLT"],
    "EVE HOLDING": ["EVEX"],
    "VERTICAL AEROSPACE": ["EVTL"],
    "STARFIGHTERS SPACE": ["FJET"],
    "FIREFLY AEROSPACE": ["FLY"],
    "FTAI AVIATION": ["FTAI"],
    "GENERAL DYNAMICS": ["GD"],
    "GE AEROSPACE": ["GE"],
    "HYPERSCALE DATA": ["GPUS"],
    "HEICO": ["HEI", "HEIA"],
    "HUNTINGTON INGALLS INDUSTRIES": ["HII"],
    "NEW HORIZON AIRCRAFT": ["HOVR"],
    "HOWMET AEROSPACE": ["HWM"],
    "HEXCEL": ["HXL"],
    "HONEYWELL INTERNATIONAL": ["HON"],
    "INNOVATIVE SOLUTIONS & SUPPORT": ["ISSC"],
    "JOBY AVIATION": ["JOBY"],
    "NAUTICUS ROBOTICS": ["KITT"],
    "KARMAN HOLDINGS": ["KRMN"],
    "KRATOS DEFENSE & SECURITY SOLUTIONS": ["KTOS"],
    "LEIDOS HOLDINGS": ["LDOS"],
    "L3HARRIS TECHNOLOGIES": ["LHX"],
    "LOCKHEED MARTIN": ["LMT"],
    "LOAR HOLDINGS": ["LOAR"],
    "INTUITIVE MACHINES": ["LUNR"],
    "MANTECH INTERNATIONAL": ["MANT"],
    "MOMENTUS": ["MNTS"],
    "MOOG": ["MOG.A"],
    "MERCURY SYSTEMS": ["MRCY"],
    "MSA SAFETY": ["MSA"],
    "NORTHROP GRUMMAN": ["NOC"],
    "NATIONAL PRESTO INDUSTRIES": ["NPK"],
    "OPTEX SYSTEMS HOLDINGS": ["OPXS"],
    "OSHKOSH": ["OSK"],
    "GRABAGUN DIGITAL HOLDINGS": ["PEW"],
    "PARK AEROSPACE": ["PKE"],
    "PLANET LABS": ["PL"],
    "OUTDOOR HOLDING": ["POWW"],
    "PARAZERO TECHNOLOGIES": ["PRZO"],
    "RED CAT HOLDINGS": ["RCAT"],
    "REDWIRE": ["RDW"],
    "STURM RUGER & CO": ["RGR"],
    "ROCKET LAB": ["RKLB"],
    "RTX": ["RTX"],
    "SCIENCE APPLICATIONS INTERNATIONAL": ["SAIC"],
    "STANDARDAERO": ["SARO"],
    "SATELLOGIC": ["SIDU"],
    "SIFCO INDUSTRIES": ["SIF"],
    "SKY HARBOUR GROUP": ["SKYH"],
    "SAFE PRO GROUP": ["SPAI"],
    "VIRGIN GALACTIC HOLDINGS": ["SPCE"],
    "SPIRIT AEROSYSTEMS HOLDINGS": ["SPR"],
    "SMITH & WESSON BRANDS": ["SWBI"],
    "TAT TECHNOLOGIES": ["TATT"],
    "TRANSDIGM GROUP": ["TDG"],
    "TELEDYNE TECHNOLOGIES": ["TDY"],
    "TRIUMPH GROUP": ["TGI"],
    "TEXTRON": ["TXT"],
    "VIASAT": ["VSAT"],
    "VSE": ["VSEC"],
    "VIRTRA": ["VTSI"],
    "V2X": ["VVX"],
    "VISIONWAVE HOLDINGS": ["VWAV"],
    "VOYAGER TECHNOLOGIES": ["VOYG"],
    "WOODWARD": ["WWD"],

    # Communication Services
    "COMCAST": ["CMCSA"],
    "DISNEY": ["DIS"],
    "AT&T": ["T"],
    "T-MOBILE": ["TMUS"],
    "VERIZON": ["VZ"],

    # Consumer Staples
    "COLGATE-PALMOLIVE": ["CL"],
    "COSTCO": ["COST"],
    "COCA-COLA": ["KO"],
    "MONDELEZ": ["MDLZ"],
    "ALTRIA": ["MO"],
    "PEPSICO": ["PEP"],
    "PROCTER & GAMBLE": ["PG"],
    "PHILIP MORRIS": ["PM"],
    "WALMART": ["WMT"],

    # Energy
    "CONOCOPHILLIPS": ["COP"],
    "CHEVRON": ["CVX"],
    "EXXONMOBIL": ["XOM"],

    # Utilities (commented keep for reference)
    # "DUKE ENERGY": ["DUK"],
    "NEXTERA ENERGY": ["NEE"],
    "SOUTHERN COMPANY": ["SO"],

    # Real Estate (commented keep for reference)
    # "AMERICAN TOWER": ["AMT"],
    # "SIMON PROPERTY GROUP": ["SPG"],

    # Materials
    "LINDE": ["LIN"],

    # VanEck ETFs and related
    "VANECK SEMICONDUCTOR": ["SMH"],
    "VANECK GOLD MINERS": ["GDX"],
    "VANECK JUNIOR GOLD MINERS": ["GDXJ"],
    "VANECK OIL SERVICES": ["OIH"],
    "VANECK RETAIL": ["RTH"],
    "VANECK AGRIBUSINESS": ["MOO"],
    "VANECK GAMING ETF": ["ESPO"],
    # "VANECK AFRICA INDEX": ["AFK"],
    "VANECK FALLEN ANGEL HIGH YIELD BOND": ["ANGL"],
    # "VANECK BRAZIL SMALL-CAP": ["BRF"],
    "VANECK CHINEXT": ["CNXT"],
    "VANECK MORNINGSTAR DURABLE DIVIDEND": ["DURA"],
    "VANECK EGYPT INDEX": ["EGPT"],
    # "VANECK JP MORGAN EM LOCAL CURRENCY BOND": ["EMLC"],
    "VANECK INVESTMENT GRADE FLOATING RATE": ["FLTR"],
    "VANECK INDIA GROWTH LEADERS": ["GLIN"],
    "VANECK MORNINGSTAR GLOBAL WIDE MOAT": ["MOTG"],
    # "VANECK GREEN BOND": ["GRNB"],
    # "VANECK EMERGING MARKETS HIGH YIELD BOND": ["HYEM"],
    # "VANECK INDONESIA INDEX": ["IDX"],
    "VANECK INTERMEDIATE MUNI": ["ITM"],
    # "VANECK LONG MUNI": ["MLN"],
    # "VANECK MORNINGSTAR WIDE MOAT": ["MOAT"],
    "VANECK MORNINGSTAR INTERNATIONAL MOAT": ["MOTI"],
    "VANECK URANIUM+NUCLEAR ENERGY": ["NLR"],
    "VANECK PHARMACEUTICAL": ["PPH"],
    "VANECK RARE EARTH/STRATEGIC METALS": ["REMX"],
    # "VANECK RUSSIA": ["RSX"],
    # "VANECK RUSSIA SMALL-CAP": ["RSXJ"],
    # "VANECK STEEL": ["SLX"],
    # "VANECK LOW CARBON ENERGY": ["SMOG"],
    # "VANECK VIETNAM": ["VNM"],
    # "VANECK GLOBAL FALLEN ANGEL HIGH YIELD BOND UCITS": ["GFA"],
    # "VANECK HYDROGEN ECONOMY UCITS": ["HDRO"],
    # "VANECK IBOXX EUR CORPORATES UCITS": ["TCBT"],
    # "VANECK MORNINGSTAR DEVELOPED MARKETS DIVIDEND LEADERS UCITS": ["TDIV"],
    # "VANECK SUSTAINABLE EUROPEAN EQUAL WEIGHT UCITS": ["TEET"],
    # "VANECK IBOXX EUR SOVEREIGN DIVERSIFIED 1-10 UCITS": ["TGBT"],
    "VANECK GLOBAL REAL ESTATE UCITS": ["TRET"],
    # "VANECK SUSTAINABLE WORLD EQUAL WEIGHT UCITS": ["TSWE"],
    # "VANECK IBOXX EUR SOVEREIGN CAPPED AAA-AA 1-5 UCITS": ["TAT"],

    "AZBA": ["AZ", "AZBA"],
    "MAMET": ["META", "MAMET"],
    "SIFSKYH": ["SIF"],
}

SECTOR_MAP = {
    "FX / Commodities / Crypto": {
        "PLNJPY=X", "GC=F", "SI=F", "BTC-USD", "BTCUSD=X", "MSTR", "XAGUSD", "SGLP", "SGLP.L", "GLDE", "GLDW", "GLDM", "SLVI",
        "BAL"
    },
    "Indices / Broad ETFs": {
        "SPY", "VOO", "GLD", "SLV", "SMH", "SOXX", "QQQ", "MOAT", "MOO", "MOTI", "ITA"
    },
    "Information Technology": {
        "AAPL", "ACN", "ADBE", "AMD", "AVGO", "CRM", "CSCO", "IBM", "INTC", "INTU", "MSFT", "NOW", "NVDA", "ORCL", "PLTR", "QCOM", "TXN", "GOOG", "GOOGL", "META", "NFLX", "AMZN",
        "SOUN", "SYM", "IONQ", "IOT", "GLBE", "GRND", "BABA", "ESPO"
    },
    "Health Care": {
        "ABBV", "ABT", "AMGN", "BMY", "CVS", "DHR", "GILD", "ISRG", "JNJ", "LLY", "MDT", "MRK", "NVO", "PFE", "TMO", "UNH", "REGN", "CRSP", "BEAM", "BAYN", "PPH"
    },
    "Financials": {
        "AIG", "AXP", "BLK", "BRK.B", "BRK-B", "COF", "IBKR", "MA", "MET", "PYPL", "SCHW", "V", "HOOD", "NU", "ACP"
    },
    "Banks": {
        "JPM", "BAC", "WFC", "C", "USB", "PNC", "TFC", "GS", "MS", "BK"
    },
    "Consumer Discretionary": {
        "BKNG", "GM", "HD", "LOW", "MCD", "NKE", "SBUX", "TGT", "TSLA", "VOW3", "VOW3.DE", "BMW", "BMW.DE", "BMW3.DE", "CELH"
    },
    "Industrials": {
        "CAT", "DE", "EMR", "FDX", "MMM", "UBER", "UNP", "UPS", "TKA", "TKA.DE", "MTX.DE"
    },
    "Defense & Aerospace": {
        "ACHR", "AIR", "AIRI", "AIRO", "AOUT", "ASTC", "ATI", "ATRO", "AVAV", "AXON", "AZ", "BA", "BAH", "BETA", "BWXT", "BYRN", "CACI", "CAE", "CDRE", "CODA", "CVU", "CW", "DCO", "DFSC", "DPRO", "DRS", "EH", "EMBJ", "ESLT", "EVEX", "EVTL", "FJET", "FLY", "FTAI", "GD", "GE", "GPUS", "HEI", "HEIA", "HEI.A", "HEI-A", "HII", "HOVR", "HWM", "HXL", "HON", "ISSC", "JOBY", "KITT", "KRMN", "KTOS", "LDOS", "LHX", "LMT", "LOAR", "LUNR", "MANT", "MNTS", "MOG.A", "MOG-A", "MRCY", "MSA", "NOC", "NPK", "OPXS", "OSK", "PEW", "PKE", "PL", "POWW", "PRZO", "RCAT", "RDW", "RGR", "RKLB", "RTX", "SAIC", "SARO", "SATL", "SIDU", "SIF", "SKYH", "SPAI", "SPCE", "SPR", "SWBI", "TATT", "TDG", "TDY", "TXT", "VSAT", "VSEC", "VTSI", "VVX", "VWAV", "VOYG", "WWD", "RHM.DE", "AIR.PA", "HO.PA", "HAG.DE", "BA.L", "FACC.VI", "MTX.DE", "R3NK", "R3NK.DE", "KOG.OL", "SAABY", "SAF", "FINMY", "EXA", "EXA.PA", "BKSY", "ASTS", "THEON", "THEON.AS", "KOG", "KOG.OL",
        "SNT"
    },
    "Communication Services": {"CMCSA", "DIS", "T", "TMUS", "VZ"},
    "Consumer Staples": {"CL", "COST", "KO", "MDLZ", "MO", "PEP", "PG", "PM", "WMT"},
    "Energy": {"COP", "CVX", "XOM", "AM", "OIH"},
    "Utilities": {"NEE", "SO"},
    "Real Estate": {"SPG", "VNQ"},
    "Materials": {
        "LIN", "NEM", "B", "AEM", "GDX", "GDXJ", "REMX", "MOO", "CRS", "KMT",
        # Copper
        "FCX", "SCCO", "TECK", "HBM", "IVN.TO",
        # Rare Earths & Strategic Minerals
        "MP", "LYSCF", "UUUU", "ILKAF",
        # Lithium
        "ALB", "SQM",
        # Specialty metals
        "ATI", "MTRN",
        # Uranium & nuclear materials
        "CCJ", "DNN",
        # Mining infrastructure & processing
        "ERMAY", "GLNCY", "RIO",
        # Major Gold Producers
        "GOLD", "KGC", "CDE", "HL", "IAUX", "HYMC",
        # Silver & Dual Gold/Silver Producers
        "PAAS", "GORO", "USAS",
        # Streaming & Royalty Companies
        "WPM", "RGLD", "GROY",
        # Juniors & Exploration
        "SLVR", "EXK", "SVM", "AG"
    },
    "Asian Tech & Manufacturing": {"005930.KS"},
    "VanEck ETFs": {"AFK", "ANGL", "CNXT", "EGPT", "FLTR", "GLIN", "MOTG", "IDX", "MLN", "NLR", "DURA"},
    "Options / Structured Products": {
        "TSLD", "PLTI", "QQQO", "METI", "MSFI", "MAGD", "AMDI", "AMZD", "GOOO", "BABI", "BABY", "AAPI", "AVGI", "MSTP"
    },
    "Nuclear": {"OKLO", "CCJ", "UUUU", "GEV", "LEU", "SMR", "NXE", "UEC"},
    "Critical Materials": {"MP", "CRML", "IDR", "FCX", "UAMY"},
    "Space": {"RKLB", "ASTS", "PL", "BKSY", "LUNR", "SPIR", "GSAT", "IRDM", "ASTR", "MDA.TO", "MDALF"},
    "Drones": {"ONDS", "UMAC", "AVAV", "KTOS", "DPRO"},
    "AI Utility / Infrastructure": {"IREN", "NBIS", "CIFR", "CRWV", "GLXY", "SMCI", "ANET", "VRT", "2308.TW", "ETN"},
    "AI Software / Data Platforms": {"CFLT", "ESTC", "PATH", "DDOG", "SNOW", "MDB"},
    "Semiconductor Equipment": {"ASML", "LRCX", "AMAT", "TSM", "ARM", "SNPS", "CDNS", "8035.T"},
    "AI Power Semiconductors": {"NVTS", "WOLF", "AEHR", "ALAB", "CRDO", "IFX.DE", "STM", "MPWR", "AOSL", "POWI", "VSH"},
    "AI Hardware / Edge Compute": {"MRVL", "NXPI", "ADI", "ON", "000660.KS", "6723.T"},
    "Cloud & Cybersecurity": {"CRWD", "ZS"},
    "Fintech": {"HUBS", "AFRM", "NU", "COIN", "PYPL"},
    "Batteries & Energy Tech": {"ENVX", "QS", "FLNC", "ENPH", "BE"},
    "TechBio / AI Drug Discovery": {"RXRX", "SDGR", "ABCL", "NTLA", "TEM"},
    "Biotech Platforms / Genomics": {"VRTX", "ILMN", "PACB", "EXAI"},
    "Quantum Computing": {"IONQ", "QBTS", "ARQQ", "RGTI", "QUBT"},
    "Industrial Infrastructure": {"PWR", "JCI", "ETN", "2308.TW"},
    "Growth Screen (Michael Kao List)": {"ASTS", "NUTX", "RCAT", "MU", "SANM", "SEZL", "AMCR", "PSIX", "DLO", "COMM", "PGY", "FOUR"}
}

def get_sector(symbol: str) -> str:
    s = symbol.upper().strip()
    candidates = {s}
    if "-" in s:
        candidates.add(s.replace("-", "."))
    if "." in s:
        base, suffix = s.rsplit(".", 1)
        if len(suffix) <= 4:
            candidates.add(base)
    for cand in candidates:
        for sector, tickers in SECTOR_MAP.items():
            if cand in tickers:
                return sector
    return "Unspecified"

# FX rate cache (JSON on disk) to avoid repeated network calls
FX_RATE_CACHE_PATH = os.path.join("scripts", "quant", "cache", "fx_rates.json")
FX_RATE_CACHE_MAX_AGE_DAYS = 3  # Cache FX rates for 3 days to reduce rate limiting
_FX_RATE_CACHE: Optional[Dict[str, dict]] = None

# Static mapping of known symbols to their quote currencies
# This avoids API calls for commonly used assets
KNOWN_SYMBOL_CURRENCIES: Dict[str, str] = {
    # =========================================================================
    # US Stocks (USD) - Major companies
    # =========================================================================
    "AAPL": "USD", "MSFT": "USD", "GOOGL": "USD", "GOOG": "USD", "AMZN": "USD",
    "NVDA": "USD", "META": "USD", "TSLA": "USD", "BRK-B": "USD", "BRK.B": "USD",
    "JPM": "USD", "V": "USD", "MA": "USD", "UNH": "USD", "JNJ": "USD",
    "HD": "USD", "PG": "USD", "XOM": "USD", "CVX": "USD", "BAC": "USD",
    "ABBV": "USD", "MRK": "USD", "PFE": "USD", "COST": "USD", "KO": "USD",
    "PEP": "USD", "AVGO": "USD", "LLY": "USD", "WMT": "USD", "MCD": "USD",
    "CSCO": "USD", "ABT": "USD", "TMO": "USD", "DHR": "USD", "ACN": "USD",
    "ADBE": "USD", "AMD": "USD", "CRM": "USD", "INTC": "USD", "INTU": "USD",
    "NFLX": "USD", "ORCL": "USD", "QCOM": "USD", "TXN": "USD", "NOW": "USD",
    "IBM": "USD", "PLTR": "USD", "MSTR": "USD", "COIN": "USD", "HOOD": "USD",
    "PYPL": "USD", "GS": "USD", "MS": "USD", "C": "USD", "WFC": "USD",
    "USB": "USD", "PNC": "USD", "TFC": "USD", "BK": "USD", "SCHW": "USD",
    "BLK": "USD", "AXP": "USD", "COF": "USD", "MET": "USD", "AIG": "USD",
    "IBKR": "USD", "NU": "USD", "AFRM": "USD", "HUBS": "USD",
    
    # Health Care (USD)
    "AMGN": "USD", "BMY": "USD", "CVS": "USD", "GILD": "USD", "ISRG": "USD",
    "MDT": "USD", "NVO": "USD", "REGN": "USD", "BEAM": "USD", "CRSP": "USD",
    "CELH": "USD", "VRTX": "USD", "ILMN": "USD", "PACB": "USD", "RXRX": "USD",
    "SDGR": "USD", "ABCL": "USD", "NTLA": "USD", "TEM": "USD",
    
    # Consumer (USD)
    "GM": "USD", "LOW": "USD", "NKE": "USD", "SBUX": "USD", "TGT": "USD",
    "BKNG": "USD", "GRND": "USD",
    
    # Industrial (USD)
    "CAT": "USD", "DE": "USD", "EMR": "USD", "FDX": "USD", "MMM": "USD",
    "UBER": "USD", "UNP": "USD", "UPS": "USD", "HON": "USD",
    
    # Communication (USD)
    "CMCSA": "USD", "DIS": "USD", "T": "USD", "TMUS": "USD", "VZ": "USD",
    
    # Consumer Staples (USD)
    "CL": "USD", "MDLZ": "USD", "MO": "USD", "PM": "USD",
    
    # Energy (USD)
    "COP": "USD",
    
    # Utilities (USD)
    "NEE": "USD", "SO": "USD", "DUK": "USD",
    
    # Real Estate (USD)
    "SPG": "USD", "AMT": "USD",
    
    # Materials (USD)
    "LIN": "USD", "NEM": "USD", "AEM": "USD", "B": "USD", "FCX": "USD",
    # Materials — Copper
    "SCCO": "USD", "TECK": "USD", "HBM": "USD",
    "IVN.TO": "CAD",  # Ivanhoe Mines - Toronto Stock Exchange
    # Materials — Rare Earths & Strategic Minerals
    "MP": "USD", "LYSCF": "USD", "UUUU": "USD", "ILKAF": "USD",
    # Materials — Lithium
    "ALB": "USD", "SQM": "USD",
    # Materials — Specialty metals
    "ATI": "USD", "MTRN": "USD",
    # Materials — Uranium
    "CCJ": "USD", "DNN": "USD",
    # Materials — Mining infrastructure (ADRs - USD)
    "ERMAY": "USD", "GLNCY": "USD", "RIO": "USD",
    # Materials — Major Gold Producers
    "GOLD": "USD", "KGC": "USD", "CDE": "USD", "HL": "USD", "IAUX": "USD", "HYMC": "USD",
    # Materials — Silver & Dual Gold/Silver Producers
    "PAAS": "USD", "GORO": "USD", "USAS": "USD",
    # Materials — Streaming & Royalty Companies
    "WPM": "USD", "RGLD": "USD", "GROY": "USD",
    # Materials — Juniors & Exploration
    "SLVR": "USD", "EXK": "USD", "SVM": "USD", "AG": "USD",
    
    # =========================================================================
    # Defense & Aerospace (US - USD)
    # =========================================================================
    "BA": "USD", "LMT": "USD", "RTX": "USD", "NOC": "USD", "GD": "USD",
    "GE": "USD", "HII": "USD", "LHX": "USD", "LDOS": "USD", "KTOS": "USD",
    "AVAV": "USD", "RKLB": "USD", "JOBY": "USD", "AXON": "USD",
    "ACHR": "USD", "AIR": "USD", "AIRI": "USD", "AIRO": "USD", "AOUT": "USD",
    "ASTC": "USD", "ATI": "USD", "ATRO": "USD", "AZ": "USD", "BAH": "USD",
    "BETA": "USD", "BWXT": "USD", "BYRN": "USD", "CACI": "USD", "CAE": "USD",
    "CDRE": "USD", "CODA": "USD", "CVU": "USD", "CW": "USD", "DCO": "USD",
    "DFSC": "USD", "DPRO": "USD", "DRS": "USD", "EH": "USD", "EMBJ": "USD",
    "EVEX": "USD", "EVTL": "USD", "FJET": "USD", "FLY": "USD", "FTAI": "USD",
    "GPUS": "USD", "HEI": "USD", "HOVR": "USD", "HWM": "USD", "HXL": "USD",
    "ISSC": "USD", "KITT": "USD", "KRMN": "USD", "LOAR": "USD", "LUNR": "USD",
    "MANT": "USD", "MNTS": "USD", "MOG.A": "USD", "MOG-A": "USD", "MRCY": "USD",
    "MSA": "USD", "NPK": "USD", "OPXS": "USD", "OSK": "USD", "PEW": "USD",
    "PKE": "USD", "PL": "USD", "POWW": "USD", "PRZO": "USD", "RCAT": "USD",
    "RDW": "USD", "RGR": "USD", "SAIC": "USD", "SARO": "USD", "SATL": "USD",
    "SIDU": "USD", "SIF": "USD", "SKYH": "USD", "SPAI": "USD", "SPCE": "USD",
    "SPR": "USD", "SWBI": "USD", "TATT": "USD", "TDG": "USD", "TDY": "USD",
    "TXT": "USD", "VSAT": "USD", "VSEC": "USD", "VTSI": "USD", "VVX": "USD",
    "VWAV": "USD", "VOYG": "USD", "WWD": "USD", "ASTS": "USD", "BKSY": "USD",
    
    # =========================================================================
    # AI & Semiconductors (USD)
    # =========================================================================
    "ASML": "USD", "LRCX": "USD", "AMAT": "USD", "TSM": "USD", "ARM": "USD",
    "SMCI": "USD", "ANET": "USD", "SNPS": "USD", "CDNS": "USD", "MRVL": "USD",
    "NXPI": "USD", "ADI": "USD", "ON": "USD", "NVTS": "USD", "WOLF": "USD",
    "AEHR": "USD", "ALAB": "USD", "CRDO": "USD", "MU": "USD",
    # AI Power Semiconductors (USD)
    "STM": "USD", "MPWR": "USD", "AOSL": "USD", "POWI": "USD", "VSH": "USD", "ETN": "USD",
    
    # AI & Semiconductors (International)
    "000660.KS": "KRW",  # SK Hynix (Korean Won)
    "8035.T": "JPY",     # Tokyo Electron (Japanese Yen)
    "6723.T": "JPY",     # Renesas Electronics (Japanese Yen)
    "IFX.DE": "EUR",     # Infineon (Euro)
    "2308.TW": "TWD",    # Delta Electronics (Taiwan Dollar)
    
    # Cloud & Cybersecurity (USD)
    "CRWD": "USD", "ZS": "USD", "DDOG": "USD", "SNOW": "USD", "MDB": "USD",
    "CFLT": "USD", "ESTC": "USD", "PATH": "USD",
    
    # Quantum (USD)
    "IONQ": "USD", "QBTS": "USD", "ARQQ": "USD", "RGTI": "USD", "QUBT": "USD",
    
    # Space (USD)
    "SPIR": "USD", "GSAT": "USD", "IRDM": "USD",
    
    # Fintech (USD)
    "SOUN": "USD", "SYM": "USD", "IOT": "USD", "GLBE": "USD",
    
    # Nuclear & Energy (USD)
    "OKLO": "USD", "CCJ": "USD", "UUUU": "USD", "GEV": "USD", "LEU": "USD",
    "SMR": "USD", "ENVX": "USD", "QS": "USD", "FLNC": "USD", "ENPH": "USD", "BE": "USD",
    
    # Infrastructure (USD)
    "PWR": "USD", "VRT": "USD", "JCI": "USD", "CRS": "USD", "KMT": "USD",
    
    # Growth Screen (USD)
    "NUTX": "USD", "SANM": "USD", "SEZL": "USD", "AMCR": "USD", "PSIX": "USD",
    "DLO": "USD", "COMM": "USD", "PGY": "USD", "FOUR": "USD",
    
    # Drones (USD)
    "ONDS": "USD", "UMAC": "USD",
    
    # AI Infrastructure (USD)
    "IREN": "USD", "NBIS": "USD", "CIFR": "USD", "CRWV": "USD", "GLXY": "USD",
    
    # Critical Materials (USD)
    "MP": "USD", "CRML": "USD", "IDR": "USD", "UAMY": "USD",
    
    # =========================================================================
    # ETFs (USD)
    # =========================================================================
    "SPY": "USD", "VOO": "USD", "QQQ": "USD", "IWM": "USD", "DIA": "USD",
    "GLD": "USD", "SLV": "USD", "SMH": "USD", "XLF": "USD", "XLK": "USD",
    "XLE": "USD", "XLV": "USD", "XLI": "USD", "XLP": "USD", "XLY": "USD",
    "XLU": "USD", "XLB": "USD", "XLRE": "USD",
    "GDX": "USD", "GDXJ": "USD", "ARKK": "USD", "ARKG": "USD",
    # VanEck ETFs
    "AFK": "USD", "ANGL": "USD", "CNXT": "USD", "DURA": "USD", "EGPT": "USD",
    "FLTR": "USD", "GLIN": "USD", "MOTG": "USD", "IDX": "USD", "DFNG": "USD",
    "MLN": "USD", "MOAT": "USD", "MOO": "USD", "MOTI": "USD", "NLR": "USD",
    "OIH": "USD", "PPH": "USD", "REMX": "USD", "ESPO": "USD", "GFA": "USD", "TRET": "USD",
    # Options ETFs
    "TSLD": "USD", "PLTI": "USD", "QQQO": "USD", "METI": "USD", "MSFI": "USD",
    "MSTP": "USD", "MAGD": "USD", "GOOO": "USD", "AMDI": "USD", "AMZD": "USD",
    "AVGI": "USD", "BABI": "USD", "BABY": "USD", "AAPI": "USD", "GLDE": "USD",
    "SLVI": "USD", "SGLP": "USD", "GLDW": "USD",
    
    # =========================================================================
    # Commodities/Futures (USD)
    # =========================================================================
    "GC=F": "USD", "SI=F": "USD", "CL=F": "USD", "NG=F": "USD",
    "PL=F": "USD", "PA=F": "USD",
    
    # =========================================================================
    # Crypto (USD)
    # =========================================================================
    "BTC-USD": "USD", "ETH-USD": "USD", "SOL-USD": "USD",
    
    # =========================================================================
    # European Stocks
    # =========================================================================
    # German (EUR)
    "RHM.DE": "EUR", "MTX.DE": "EUR", "HAG.DE": "EUR", "TKA.DE": "EUR",
    "VOW3.DE": "EUR", "BMW3.DE": "EUR", "BAYN.DE": "EUR", "R3NK.DE": "EUR",
    "RHM": "EUR", "HAG": "EUR", "TKA": "EUR", "VOW3": "EUR", "BAYN": "EUR",
    # French (EUR)
    "AIR.PA": "EUR", "SAF.PA": "EUR", "HO.PA": "EUR", "EXA.PA": "EUR",
    "SAF": "EUR", "HO": "EUR", "EXA": "EUR", "AM": "EUR",
    # Dutch (EUR)
    "ASML": "EUR", "HEIA.AS": "EUR", "THEON.AS": "EUR", "THEON": "EUR",
    # Austrian (EUR)
    "FACC.VI": "EUR", "FACC": "EUR",
    # UK (GBX = pence)
    "BA.L": "GBX",
    # Norwegian (NOK)
    "KOG.OL": "NOK",
    # Israeli (USD - listed on US exchanges)
    "ESLT": "USD",
    # ADRs (USD)
    "FINMY": "USD", "SAABY": "USD", "MDALF": "USD",
    
    # =========================================================================
    # Asian Stocks
    # =========================================================================
    # Korean (KRW)
    "005930.KS": "KRW",
    # Canadian (CAD)
    "MDA.TO": "CAD",
    
    # =========================================================================
    # Polish Stocks (PLN)
    # =========================================================================
    "ACP": "PLN", "SNT": "PLN",
    
    # =========================================================================
    # FX Pairs
    # =========================================================================
    "PLNJPY=X": "JPY", "USDPLN=X": "PLN", "EURPLN=X": "PLN",
    "GBPPLN=X": "PLN", "EURUSD=X": "USD", "USDJPY=X": "JPY",
    "EURJPY=X": "JPY", "GBPUSD=X": "USD", "USDCHF=X": "CHF",
    "AUDUSD=X": "USD", "USDCAD=X": "CAD", "NZDUSD=X": "USD",
    "CHFPLN=X": "PLN", "CADPLN=X": "PLN", "AUDPLN=X": "PLN",
    "JPYPLN=X": "PLN", "SEKPLN=X": "PLN", "NOKPLN=X": "PLN",
    "DKKPLN=X": "PLN", "HKDPLN=X": "PLN",
    "EURSEK=X": "SEK", "EURNOK=X": "NOK", "EURDKK=X": "DKK",
    "USDHKD=X": "HKD", "USDKRW=X": "KRW",
}

# Cache for detected currencies to avoid repeated API calls
_CURRENCY_DETECTION_CACHE: Dict[str, str] = {}


def get_default_asset_universe() -> List[str]:
    """
    Get the default asset universe.

    Returns a copy of the centralized asset list to prevent external modification.
    This function provides a clean API for retrieving the asset universe.

    Returns:
        List of asset symbols/names
    """
    return DEFAULT_ASSET_UNIVERSE.copy()


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

_PX_CACHE: Dict[Tuple[str, Optional[str], Optional[str]], pd.DataFrame] = {}


def _px_cache_key(symbol: str, start: Optional[str], end: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    return (symbol.upper().strip(), start, end)


def _store_cached_prices(symbol: str, start: Optional[str], end: Optional[str], df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    _PX_CACHE[_px_cache_key(symbol, start, end)] = df


def _price_cache_path(symbol: str) -> pathlib.Path:
    safe = symbol.replace("/", "_").replace("=", "_").replace(":", "_")
    return PRICE_CACHE_DIR_PATH / f"{safe.upper()}.csv"


# Standard column names for price data
STANDARD_PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close"]  # Columns that must have data


def _normalize_price_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize a price DataFrame to have standard columns.
    
    Standard format:
    - Index: Date (datetime, timezone-naive)
    - Columns: Open, High, Low, Close, Adj Close, Volume, Ticker
    
    Removes rows where all price columns are NaN (e.g., dates before company existed).
    
    Args:
        df: Input DataFrame (may have various column formats from Yahoo Finance)
        symbol: Ticker symbol to add to Ticker column
        
    Returns:
        Normalized DataFrame with standard columns, empty rows removed
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_PRICE_COLUMNS)
    
    result = pd.DataFrame(index=df.index)
    
    # Handle MultiIndex columns (from yf.download with single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex - take first level values
        flat_df = pd.DataFrame(index=df.index)
        for col_name in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            try:
                if col_name in df.columns.get_level_values(0):
                    col_data = df[col_name]
                    if isinstance(col_data, pd.DataFrame):
                        col_data = col_data.iloc[:, 0]
                    flat_df[col_name] = col_data
            except Exception:
                pass
        df = flat_df
    
    # Map various column name formats to standard names
    column_mappings = {
        "Open": ["Open", "open", "OPEN"],
        "High": ["High", "high", "HIGH"],
        "Low": ["Low", "low", "LOW"],
        "Close": ["Close", "close", "CLOSE", "Price", "price"],
        "Adj Close": ["Adj Close", "Adj_Close", "adj close", "Adjusted Close", "adjusted_close", "AdjClose"],
        "Volume": ["Volume", "volume", "VOLUME", "Vol", "vol"],
    }
    
    for standard_name, possible_names in column_mappings.items():
        for name in possible_names:
            if name in df.columns:
                col_data = df[name]
                # Ensure it's a Series
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                result[standard_name] = col_data
                break
        else:
            # Column not found - use NaN
            result[standard_name] = np.nan
    
    # If Close is NaN but Adj Close exists, copy Adj Close to Close
    if result["Close"].isna().all() and not result["Adj Close"].isna().all():
        result["Close"] = result["Adj Close"]
    
    # If Adj Close is NaN but Close exists, copy Close to Adj Close
    if result["Adj Close"].isna().all() and not result["Close"].isna().all():
        result["Adj Close"] = result["Close"]
    
    # Ensure numeric types for price/volume columns
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    
    # Remove rows where ALL price columns are NaN (dates before company existed)
    # Keep rows where at least one price column has data
    price_cols = [c for c in PRICE_COLUMNS if c in result.columns]
    if price_cols:
        has_any_price = result[price_cols].notna().any(axis=1)
        result = result[has_any_price]
    
    # Add Ticker column (after filtering to avoid issues)
    result["Ticker"] = symbol.upper()
    
    # Normalize index
    result.index.name = "Date"
    
    return result


def _load_disk_prices(symbol: str) -> Optional[pd.DataFrame]:
    """Load price data from disk cache.
    
    Returns DataFrame with standard columns: Open, High, Low, Close, Adj Close, Volume, Ticker
    Filters out rows where all price columns are NaN.
    """
    path = _price_cache_path(symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        
        # Normalize index to timezone-naive datetimes, drop duplicates, and sort
        df.index = pd.to_datetime(df.index, format="ISO8601", errors="coerce")
        df = df[~df.index.isna()]
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.index.name = "Date"
        
        # Check if Ticker column exists; if not, normalize the dataframe
        if "Ticker" not in df.columns:
            df = _normalize_price_dataframe(df, symbol)
        else:
            # Filter out rows where all price columns are NaN (cleanup existing caches)
            price_cols = [c for c in PRICE_COLUMNS if c in df.columns]
            if price_cols:
                has_any_price = df[price_cols].notna().any(axis=1)
                df = df[has_any_price]
        
        return df
    except Exception:
        return None


def get_price_series(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """Extract a price series from a standardized price DataFrame.
    
    Args:
        df: DataFrame with standard columns (Open, High, Low, Close, Adj Close, Volume, Ticker)
        price_col: Column to extract ("Close", "Adj Close", "Open", "High", "Low")
        
    Returns:
        Series with price data, or empty Series if column not found
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    
    # Try requested column first
    if price_col in df.columns:
        px = df[price_col]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        return pd.to_numeric(px, errors="coerce").dropna()
    
    # Fallback to Close or Adj Close
    for col in ["Close", "Adj Close"]:
        if col in df.columns:
            px = df[col]
            if isinstance(px, pd.DataFrame):
                px = px.iloc[:, 0]
            return pd.to_numeric(px, errors="coerce").dropna()
    
    return pd.Series(dtype=float)


def _store_disk_prices(symbol: str, df: pd.DataFrame) -> None:
    """Store price data to disk cache with standard format.
    
    Standard format:
    - Index: Date
    - Columns: Open, High, Low, Close, Adj Close, Volume, Ticker
    """
    if df is None or df.empty:
        return
    
    # Normalize to standard format before saving
    normalized = _normalize_price_dataframe(df, symbol)
    if normalized.empty:
        return
    
    path = _price_cache_path(symbol)
    tmp = path.with_suffix(".tmp")
    with _price_file_lock:
        try:
            normalized.to_csv(tmp)
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


def _merge_and_store(symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new price data with existing cached data and store.
    
    Both existing and new data are normalized to standard format before merging.
    """
    # Normalize new data
    new_normalized = _normalize_price_dataframe(new_df, symbol)
    
    existing = _load_disk_prices(symbol)
    if existing is not None and not existing.empty:
        # Existing data is already normalized by _load_disk_prices
        combined = pd.concat([existing, new_normalized], axis=0)
        # Remove Ticker column temporarily for deduplication, then add back
        ticker_val = symbol.upper()
        if "Ticker" in combined.columns:
            combined = combined.drop(columns=["Ticker"])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        combined["Ticker"] = ticker_val
    else:
        combined = new_normalized
    
    _store_disk_prices(symbol, combined)
    return combined


def clean_price_cache(verbose: bool = True) -> Dict[str, int]:
    """Clean all cached price files by removing empty rows.
    
    Removes rows where all price columns (Open, High, Low, Close, Adj Close) are NaN.
    This cleans up data for companies that didn't exist in the requested date range.
    
    Args:
        verbose: If True, print progress information
        
    Returns:
        Dict mapping symbol to number of rows removed
    """
    results = {}
    cache_files = list(PRICE_CACHE_DIR_PATH.glob("*.csv"))
    
    if verbose:
        print(f"Cleaning {len(cache_files)} cached price files...")
    
    for path in cache_files:
        symbol = path.stem
        try:
            # Read raw data
            df = pd.read_csv(path, index_col=0)
            original_rows = len(df)
            
            if original_rows == 0:
                continue
            
            # Normalize index
            df.index = pd.to_datetime(df.index, format="ISO8601", errors="coerce")
            df = df[~df.index.isna()]
            
            # Filter out rows where all price columns are NaN
            price_cols = [c for c in PRICE_COLUMNS if c in df.columns]
            if price_cols:
                has_any_price = df[price_cols].notna().any(axis=1)
                df_clean = df[has_any_price]
            else:
                df_clean = df
            
            rows_removed = original_rows - len(df_clean)
            
            if rows_removed > 0:
                # Save cleaned data
                df_clean = df_clean[~df_clean.index.duplicated(keep="last")].sort_index()
                df_clean.index.name = "Date"
                
                # Ensure Ticker column exists
                if "Ticker" not in df_clean.columns:
                    df_clean["Ticker"] = symbol.upper()
                
                tmp = path.with_suffix(".tmp")
                df_clean.to_csv(tmp)
                os.replace(tmp, path)
                
                results[symbol] = rows_removed
                if verbose:
                    print(f"  ✓ {symbol}: removed {rows_removed} empty rows ({len(df_clean)} remaining)")
            
        except Exception as e:
            if verbose:
                print(f"  ✗ {symbol}: error - {e}")
    
    total_removed = sum(results.values())
    if verbose:
        print(f"\nCleaned {len(results)} files, removed {total_removed} total empty rows")
    
    return results


# ============================================================================
# Failed Assets Cache Management
# ============================================================================

def save_failed_assets(failures: Dict[str, Dict], append: bool = True) -> str:
    """Save failed assets to cache/failed/failed_assets.json.
    
    Args:
        failures: Dict mapping asset symbol to failure info dict containing:
            - display_name: Human-readable name
            - attempts: Number of retry attempts
            - last_error: Last error message
            - traceback: Optional full traceback
        append: If True, merge with existing failures; if False, overwrite
    
    Returns:
        Path to the saved file
    """
    if not failures:
        return str(FAILED_ASSETS_FILE)
    
    FAILED_CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    existing = {}
    if append and FAILED_ASSETS_FILE.exists():
        try:
            with open(FAILED_ASSETS_FILE, "r") as f:
                data = json.load(f)
                existing = data.get("failures", {})
        except Exception:
            existing = {}
    
    # Merge: update existing with new failures
    for asset, info in failures.items():
        existing[asset] = {
            "display_name": info.get("display_name", asset),
            "attempts": info.get("attempts", 1),
            "last_error": str(info.get("last_error", "")),
            "traceback": str(info.get("traceback", "")) if info.get("traceback") else None,
            "timestamp": datetime.now().isoformat(),
        }
    
    payload = {
        "updated": datetime.now().isoformat(),
        "count": len(existing),
        "failures": existing,
    }
    
    try:
        tmp = FAILED_ASSETS_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, FAILED_ASSETS_FILE)
    except Exception as e:
        print(f"Warning: Could not save failed assets: {e}")
    
    return str(FAILED_ASSETS_FILE)


def load_failed_assets() -> Dict[str, Dict]:
    """Load failed assets from scripts/quant/cache/failed/failed_assets.json.
    
    Returns:
        Dict mapping asset symbol to failure info
    """
    if not FAILED_ASSETS_FILE.exists():
        return {}
    try:
        with open(FAILED_ASSETS_FILE, "r") as f:
            data = json.load(f)
            return data.get("failures", {})
    except Exception:
        return {}


def get_failed_asset_symbols() -> List[str]:
    """Get list of failed asset symbols.
    
    Returns:
        List of asset symbols that have failed
    """
    failures = load_failed_assets()
    return list(failures.keys())


def purge_failed_assets_from_cache(verbose: bool = True) -> Dict[str, bool]:
    """Remove cached price data for all failed assets.
    
    This cleans up potentially corrupted or problematic cache files
    for assets that have consistently failed to process.
    
    Args:
        verbose: If True, print progress information
    
    Returns:
        Dict mapping asset symbol to whether it was successfully purged
    """
    failures = load_failed_assets()
    if not failures:
        if verbose:
            print("No failed assets found in scripts/quant/cache/failed/failed_assets.json")
        return {}
    
    results = {}
    purged_count = 0
    
    for asset in failures.keys():
        # Normalize symbol for cache lookup
        safe = asset.strip().upper().replace("=", "_").replace("/", "_").replace(":", "_")
        cache_path = PRICE_CACHE_DIR_PATH / f"{safe}.csv"
        
        try:
            if cache_path.exists():
                cache_path.unlink()
                results[asset] = True
                purged_count += 1
                if verbose:
                    print(f"  ✓ Purged: {asset} ({cache_path.name})")
            else:
                results[asset] = False
                if verbose:
                    print(f"  - Not cached: {asset}")
        except Exception as e:
            results[asset] = False
            if verbose:
                print(f"  ✗ Failed to purge {asset}: {e}")
    
    if verbose:
        print(f"\nPurged {purged_count}/{len(failures)} cached files")
    
    return results


def clear_failed_assets_list(verbose: bool = True) -> bool:
    """Clear the failed assets list (but don't purge cache files).
    
    Returns:
        True if successfully cleared, False otherwise
    """
    try:
        if FAILED_ASSETS_FILE.exists():
            FAILED_ASSETS_FILE.unlink()
            if verbose:
                print(f"Cleared failed assets list: {FAILED_ASSETS_FILE}")
            return True
        else:
            if verbose:
                print("No failed assets file to clear")
            return True
    except Exception as e:
        if verbose:
            print(f"Failed to clear: {e}")
        return False


def _get_cached_prices(symbol: str, start: Optional[str], end: Optional[str]) -> Optional[pd.DataFrame]:
    """Get cached prices, but only if they cover the requested date range.
    Returns None if cache is stale or doesn't cover the range, forcing a refresh.
    """
    hit = _PX_CACHE.get(_px_cache_key(symbol, start, end))
    if hit is not None:
        return hit
    disk = _load_disk_prices(symbol)
    if disk is None or disk.empty:
        return None
    # Index already normalized in _load_disk_prices

    # Check if cache extends to the requested end date (or today if end is None)
    cache_max_date = pd.to_datetime(disk.index.max()).date()
    if end:
        requested_end = pd.to_datetime(end).date()
    else:
        requested_end = datetime.now().date()

    # If cache is more than 1 day behind the requested end, return None to force refresh
    # (Allow 1 day buffer for weekends/holidays)
    if (requested_end - cache_max_date).days > 1:
        return None

    if start:
        disk = disk[disk.index >= pd.to_datetime(start)]
    if end:
        disk = disk[disk.index <= pd.to_datetime(end)]
    if disk.empty:
        return None
    _PX_CACHE[_px_cache_key(symbol, start, end)] = disk
    return disk


def _download_prices(symbol: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    """Robust Yahoo fetch with multiple strategies.
    Returns a DataFrame with OHLC columns (if available).
    - Checks if cached data is up-to-date; if not, fetches incremental updates
    - Tries yf.download first
    - Falls back to Ticker.history
    - Tries again without auto_adjust
    - Falls back to period='max' pulls to dodge timezone/metadata issues
    Normalizes DatetimeIndex to tz-naive for stability.
    """
    import time
    
    # Cap end date to today - never attempt to fetch future data
    end = _cap_date_to_today(end)
    
    cached = _get_cached_prices(symbol, start, end)
    if cached is not None and not cached.empty:
        return cached
    
    # Load disk cache first - needed for both offline mode and incremental fetching
    disk_df = _load_disk_prices(symbol)
    
    # =========================================================================
    # OFFLINE MODE: Use cached data only, skip all Yahoo Finance API calls
    # =========================================================================
    if OFFLINE_MODE:
        if disk_df is not None and not disk_df.empty:
            _PX_CACHE[_px_cache_key(symbol, start, end)] = disk_df
            return disk_df
        # No cached data available in offline mode
        return pd.DataFrame()
    
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        if df is not None and not df.empty:
            if hasattr(df.index, "tz") and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        return df if df is not None else pd.DataFrame()

    # Determine incremental fetch window based on disk cache
    fetch_start = start

    # Determine the target end date
    if end:
        target_end = pd.to_datetime(end).date()
    else:
        target_end = datetime.now().date()

    if disk_df is not None and not disk_df.empty:
        last_dt = pd.to_datetime(disk_df.index.max()).date()
        # If disk cache already covers up to target end, use it
        if last_dt >= target_end:
            _PX_CACHE[_px_cache_key(symbol, start, end)] = disk_df
            return disk_df
        # Otherwise, fetch incrementally from day after last cached date
        fetch_start = (last_dt + timedelta(days=1)).isoformat()

    # Helper to attempt fetch with retries for transient errors
    def _try_fetch_with_retry(fetch_func, max_retries=2, delay=0.5):
        last_exc = None
        for attempt in range(max_retries):
            try:
                result = fetch_func()
                if result is not None and not result.empty:
                    return result
            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                # Retry on transient timezone/metadata errors
                if "timezone" in err_str or "tz" in err_str or "metadata" in err_str:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                break
        return None

    # Try standard download
    df = _try_fetch_with_retry(
        lambda: _normalize(yf.download(symbol, start=fetch_start, end=end, auto_adjust=True, progress=False, threads=False, timeout=YFINANCE_TIMEOUT))
    )
    if df is not None and not df.empty:
        if disk_df is not None and not disk_df.empty:
            df = _merge_and_store(symbol, pd.concat([disk_df, df]))
        else:
            _store_disk_prices(symbol, df)
        _store_cached_prices(symbol, start, end, df)
        return df

    # Try Ticker.history
    df = _try_fetch_with_retry(
        lambda: _normalize(yf.Ticker(symbol).history(start=fetch_start, end=end, auto_adjust=True, timeout=YFINANCE_TIMEOUT))
    )
    if df is not None and not df.empty:
        if disk_df is not None and not disk_df.empty:
            df = _merge_and_store(symbol, pd.concat([disk_df, df]))
        else:
            _store_disk_prices(symbol, df)
        _store_cached_prices(symbol, start, end, df)
        return df

    # Try without auto_adjust
    df = _try_fetch_with_retry(
        lambda: _normalize(yf.download(symbol, start=fetch_start, end=end, auto_adjust=False, progress=False, threads=False, timeout=YFINANCE_TIMEOUT))
    )
    if df is not None and not df.empty:
        if disk_df is not None and not disk_df.empty:
            df = _merge_and_store(symbol, pd.concat([disk_df, df]))
        else:
            _store_disk_prices(symbol, df)
        _store_cached_prices(symbol, start, end, df)
        return df

    # Period max fallback to dodge tz/metadata issues for dash/dot tickers
    df = _try_fetch_with_retry(
        lambda: _normalize(yf.Ticker(symbol).history(period="max", auto_adjust=True, timeout=YFINANCE_TIMEOUT)),
        max_retries=3,
        delay=1.0
    )
    if df is not None and not df.empty:
        if disk_df is not None and not disk_df.empty:
            df = _merge_and_store(symbol, pd.concat([disk_df, df]))
        else:
            _store_disk_prices(symbol, df)
        _store_cached_prices(symbol, start, end, df)
        return df

    # Final fallback with yf.download period=max
    df = _try_fetch_with_retry(
        lambda: _normalize(yf.download(symbol, period="max", auto_adjust=True, progress=False, threads=False, timeout=YFINANCE_TIMEOUT)),
        max_retries=2,
        delay=1.0
    )
    if df is not None and not df.empty:
        if disk_df is not None and not disk_df.empty:
            df = _merge_and_store(symbol, pd.concat([disk_df, df]))
        else:
            _store_disk_prices(symbol, df)
        _store_cached_prices(symbol, start, end, df)
        return df

    # =========================================================================
    # GRACEFUL FALLBACK: Return cached disk data when Yahoo Finance unavailable
    # =========================================================================
    # If all Yahoo Finance attempts failed but we have valid cached data on disk,
    # return the cached data rather than failing. This allows the system to
    # continue operating when network is unavailable or symbols are delisted.
    # =========================================================================
    if disk_df is not None and not disk_df.empty:
        _PX_CACHE[_px_cache_key(symbol, start, end)] = disk_df
        return disk_df

    return pd.DataFrame()


def download_prices_bulk(symbols: List[str], start: Optional[str], end: Optional[str], chunk_size: int = 10, progress: bool = True, log_fn=None, skip_individual_fallback: bool = False, max_workers: int = 12, force_online: bool = False, show_symbol_tables: bool = True) -> Dict[str, pd.Series]:
    """Download multiple symbols in chunks to reduce rate limiting.
    Uses yf.download with list input; falls back to per-symbol for failures.
    Populates the local price cache so subsequent single fetches reuse data.
    Returns mapping symbol -> close price Series. Progress logging is on by default.
    
    Args:
        symbols: List of ticker symbols to download
        start: Start date for data range
        end: End date for data range
        chunk_size: Number of symbols to download per chunk
        progress: Whether to show progress output
        log_fn: Optional logging function
        skip_individual_fallback: If True, skip individual download fallback (for multi-pass bulk downloads)
        max_workers: Maximum number of parallel download workers
        force_online: If True, ignore OFFLINE_MODE and always attempt downloads (for make data)
        show_symbol_tables: If True, print symbol mapping/failure tables at the end
    """
    # Determine if we should skip downloads (respect OFFLINE_MODE unless force_online)
    skip_downloads = OFFLINE_MODE and not force_online
    
    log = log_fn if log_fn is not None else (print if progress else None)
    cleaned = [s.strip() for s in symbols if s and s.strip()]
    dedup: List[str] = []
    seen = set()
    for s in cleaned:
        u = s.upper()
        if u not in seen:
            seen.add(u)
            dedup.append(u)

    # Normalize symbols and keep lineage for deterministic mapping
    normalized_entries: List[Tuple[str, str, Dict]] = []
    for sym in dedup:
        norm, meta = normalize_yahoo_ticker(sym, perform_lookup=False)
        normalized_entries.append((sym, norm, meta))

    result: Dict[str, pd.Series] = {}
    cached_hits = 0

    # Build primary fetch targets with resolution fallbacks; skip explicit delisted/invalid
    primary_map: Dict[str, str] = {}
    primary_to_originals: Dict[str, List[str]] = {}
    for orig, norm, meta in normalized_entries:
        status = meta.get("status")
        if status in {"invalid", "delisted_or_unsupported"}:
            # Preserve lineage with an empty series to avoid silent drops
            result[orig] = pd.Series(dtype=float)
            continue
        try:
            cands = _resolve_symbol_candidates(norm)
            primary = cands[0] if cands else norm
        except Exception:
            primary = norm
        primary_map[orig] = primary
        primary_to_originals.setdefault(primary, []).append(orig)

    # Try cached first for all primaries
    # Check if cached data covers the requested date range (especially the end date)
    satisfied_primaries: Set[str] = set()
    
    # Determine the target end date for staleness check
    if end:
        target_end = pd.to_datetime(end).date()
    else:
        target_end = datetime.now().date()
    
    for primary, orig_list in list(primary_to_originals.items()):
        disk = _load_disk_prices(primary)
        if disk is not None and not disk.empty:
            # Check if cache is stale (doesn't extend close enough to the target end date)
            cache_max_date = pd.to_datetime(disk.index.max()).date()
            # Allow 3 days buffer for weekends/holidays, but require recent data
            if (target_end - cache_max_date).days > 3:
                # Cache is stale - need to re-download
                continue
            
            # Filter to requested date range
            if start:
                disk = disk[disk.index >= pd.to_datetime(start)]
            if end:
                disk = disk[disk.index <= pd.to_datetime(end)]
            if not disk.empty:
                # Use standardized price extraction
                ser = get_price_series(disk, "Close")
                if not ser.empty:
                    ser.name = "px"
                    for orig in orig_list:
                        result[orig] = ser
                    cached_hits += 1
                    satisfied_primaries.add(primary)

    remaining_primaries = [p for p in primary_to_originals if p not in satisfied_primaries]

    # =========================================================================
    # OFFLINE MODE: Skip all downloads, return only cached data
    # (unless force_online=True, which overrides for make data)
    # =========================================================================
    if skip_downloads:
        if log:
            log(f"OFFLINE MODE: Using {cached_hits} cached symbols, skipping {len(remaining_primaries)} unavailable symbols.")
        return result

    total_need = len(remaining_primaries)
    if log and (cached_hits or total_need):
        total_chunks = (total_need + chunk_size - 1) // max(1, chunk_size)
        log(f"Bulk download: {total_need} uncached, {cached_hits} from cache, {total_chunks} chunk(s), chunk size ≤ {chunk_size}.")

    fetched_count = 0

    def _download_chunk(chunk_syms: List[str]) -> Dict[str, pd.DataFrame]:
        """Download a chunk of symbols and return dict of symbol -> DataFrame (OHLCV data)."""
        out: Dict[str, pd.DataFrame] = {}
        fetch_syms = chunk_syms
        try:
            df = yf.download(fetch_syms, start=start, end=end, auto_adjust=True, group_by="ticker", progress=False, threads=False, timeout=YFINANCE_TIMEOUT)
        except Exception:
            df = None
        if df is None or df.empty:
            return out
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        is_multi = isinstance(df.columns, pd.MultiIndex)
        
        for primary in chunk_syms:
            try:
                if is_multi and len(chunk_syms) > 1:
                    # Extract all OHLCV columns for this symbol
                    sym_df = pd.DataFrame(index=df.index)
                    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                        if (primary, col) in df.columns:
                            sym_df[col] = df[(primary, col)]
                    if not sym_df.empty and ("Close" in sym_df.columns or "Adj Close" in sym_df.columns):
                        out[primary] = sym_df.dropna(how="all")
                else:
                    # Single symbol or non-MultiIndex
                    sym_df = df.copy() if not is_multi else df
                    if not sym_df.empty and ("Close" in sym_df.columns or "Adj Close" in sym_df.columns):
                        out[primary] = sym_df.dropna(how="all")
            except Exception:
                continue
        return out

    chunks = [remaining_primaries[i:i + chunk_size] for i in range(0, len(remaining_primaries), chunk_size) if remaining_primaries[i:i + chunk_size]]
    if log and chunks:
        log(f"  Launching {len(chunks)} chunk(s) with {min(max_workers, len(chunks))} workers…")

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(chunks)))) as ex:
        future_map = {ex.submit(_download_chunk, c): c for c in chunks}
        for fut in as_completed(future_map):
            chunk_syms = future_map[fut]
            try:
                out = fut.result()
            except Exception:
                out = {}
            for primary, sym_df in out.items():
                # Store full OHLCV data using standardized format
                _store_disk_prices(primary, sym_df)
                
                # Extract Close series for return value
                ser = get_price_series(sym_df, "Close")
                if not ser.empty:
                    ser.name = "px"
                    for orig in primary_to_originals.get(primary, []):
                        result[orig] = ser
                    fetched_count += 1
                    satisfied_primaries.add(primary)
            if log:
                done = fetched_count
                pct = (done / max(1, total_need)) * 100.0
                log(f"    ✓ Cached {done}/{total_need} ({pct:.1f}%) so far")

    missing_after_bulk = [p for p in remaining_primaries if p not in satisfied_primaries]
    
    # Skip individual fallback if requested (for multi-pass bulk downloads)
    if skip_individual_fallback:
        if log and missing_after_bulk:
            log(f"  Skipping individual fallback for {len(missing_after_bulk)} symbol(s) (bulk-only mode)")
        return result
    
    if log and missing_after_bulk:
        log(f"  Falling back to individual fetch for {len(missing_after_bulk)} symbol(s)…")

    def _fetch_single(primary: str) -> Optional[Tuple[str, pd.Series]]:
        try:
            df = _download_prices(primary, start, end)
            if df is not None and not df.empty:
                ser = get_price_series(df, "Close")
                if not ser.empty:
                    ser.name = "px"
                    return primary, ser
        except Exception:
            return None
        return None

    if missing_after_bulk:
        with ThreadPoolExecutor(max_workers=min(16, len(missing_after_bulk))) as ex:
            futures = {ex.submit(_fetch_single, p) for p in missing_after_bulk}
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                primary, ser = res
                for orig in primary_to_originals.get(primary, []):
                    result[orig] = ser
                # Data already stored by _download_prices via _store_disk_prices
                satisfied_primaries.add(primary)
    
    # Print accumulated symbol mapping tables (world-class Rich output) if enabled
    if show_symbol_tables:
        print_symbol_tables()
    
    return result


# Display-name cache for full asset names (e.g., company longName)
_DISPLAY_NAME_CACHE: Dict[str, str] = {}

def _resolve_display_name(symbol: str, use_api_fallback: bool = True) -> str:
    """Return a human-friendly display name for a Yahoo symbol.
    
    First checks the internal COMPANY_NAMES dictionary for fast, offline lookup.
    If not found and use_api_fallback is True, tries yfinance fast_info/info 
    longName/shortName; falls back to the symbol itself.
    
    Caches results for repeated use.
    
    Args:
        symbol: The ticker symbol (e.g., "AAPL", "GC=F")
        use_api_fallback: If True, will query Yahoo Finance API when not found
                         in internal dictionary. Set to False for offline-only mode.
    
    Returns:
        Human-friendly display name for the symbol.
        
    Example:
        >>> _resolve_display_name("AAPL")
        'Apple Inc.'
        >>> _resolve_display_name("UNKNOWN_TICKER", use_api_fallback=False)
        'UNKNOWN_TICKER'
    """
    sym = (symbol or "").strip()
    if not sym:
        return symbol
    
    # Check cache first
    if sym in _DISPLAY_NAME_CACHE:
        return _DISPLAY_NAME_CACHE[sym]
    
    # Check internal dictionary (fast, no API call)
    internal_name = get_company_name(sym)
    if internal_name:
        _DISPLAY_NAME_CACHE[sym] = internal_name
        return internal_name
    
    # Also try uppercase version
    sym_upper = sym.upper()
    if sym_upper != sym:
        internal_name = get_company_name(sym_upper)
        if internal_name:
            _DISPLAY_NAME_CACHE[sym] = internal_name
            return internal_name
    
    # If API fallback is disabled, return the symbol itself
    if not use_api_fallback:
        _DISPLAY_NAME_CACHE[sym] = sym
        return sym
    
    # Fallback to Yahoo Finance API (slower, requires network)
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
    """Fetch price series for a symbol.
    
    Uses _download_prices which returns standardized DataFrame format.
    Falls back to various column extraction methods for compatibility.
    
    Returns:
        Series with Close prices, named "px"
    """
    data = _download_prices(symbol, start, end)
    if data is None or data.empty:
        raise RuntimeError(f"No data for {symbol}")
    
    # First try using the standardized get_price_series helper
    px = get_price_series(data, "Close")
    if not px.empty:
        px.name = "px"
        return px
    
    # Fallback: Handle MultiIndex columns (e.g., from yf.download with single ticker)
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        for col_name in ("Close", "Adj Close"):
            try:
                if col_name in data.columns.get_level_values(0):
                    px = data[col_name]
                    if isinstance(px, pd.DataFrame):
                        px = px.iloc[:, 0]
                    px = pd.to_numeric(px, errors="coerce").dropna()
                    if not px.empty:
                        px.name = "px"
                        return px
            except Exception:
                continue
        # Try (col_name, symbol) tuple
        for col_name in ("Close", "Adj Close"):
            try:
                for sym_variant in [symbol, symbol.upper()]:
                    if (col_name, sym_variant) in data.columns:
                        px = pd.to_numeric(data[(col_name, sym_variant)], errors="coerce").dropna()
                        if not px.empty:
                            px.name = "px"
                            return px
            except Exception:
                continue
    
    # Fallback: Handle regular single-level columns
    for col in ("Close", "Adj Close", "close", "Price"):
        if isinstance(data, pd.DataFrame) and col in data.columns:
            col_data = data[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            px = pd.to_numeric(col_data, errors="coerce").dropna()
            if not px.empty:
                px.name = "px"
                return px
    
    # Fallback: If data is a Series
    if isinstance(data, pd.Series):
        px = pd.to_numeric(data, errors="coerce").dropna()
        if not px.empty:
            px.name = "px"
            return px
    
    raise RuntimeError(f"No price column found for {symbol}")


def _load_fx_cache() -> Dict[str, dict]:
    global _FX_RATE_CACHE
    if _FX_RATE_CACHE is not None:
        return _FX_RATE_CACHE
    try:
        with open(FX_RATE_CACHE_PATH, "r") as f:
            _FX_RATE_CACHE = json.load(f)
    except Exception:
        _FX_RATE_CACHE = {}
    return _FX_RATE_CACHE


def _store_fx_cache(cache: Dict[str, dict]) -> None:
    try:
        os.makedirs(os.path.dirname(FX_RATE_CACHE_PATH), exist_ok=True)
        tmp = FX_RATE_CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, FX_RATE_CACHE_PATH)
    except Exception:
        pass


def _maybe_load_cached_series(symbol: str, start: Optional[str], end: Optional[str], max_age_days: int = FX_RATE_CACHE_MAX_AGE_DAYS) -> Optional[pd.Series]:
    cache = _load_fx_cache()
    entry = cache.get(symbol)
    if not entry:
        return None
    try:
        ts = datetime.fromisoformat(entry.get("timestamp"))
        if datetime.utcnow() - ts > timedelta(days=max_age_days):
            return None
        cached_start = entry.get("start")
        cached_end = entry.get("end")
        # Require coverage of requested window when provided
        if start and cached_start and cached_start > start:
            return None
        if end and cached_end and cached_end < end:
            return None
        data = entry.get("data", [])
        if not data:
            return None
        idx = [pd.to_datetime(d[0]) for d in data]
        vals = [float(d[1]) for d in data]
        s = pd.Series(vals, index=idx, name="px").dropna()
        return s
    except Exception:
        return None


def _maybe_store_cached_series(symbol: str, series: pd.Series) -> None:
    if series is None or series.empty:
        return
    cache = _load_fx_cache()
    try:
        s_clean = _ensure_float_series(series).dropna()
        if s_clean.empty:
            return
        data = [(pd.to_datetime(i).date().isoformat(), float(v)) for i, v in s_clean.items()]
        cache[symbol] = {
            "start": s_clean.index.min().date().isoformat(),
            "end": s_clean.index.max().date().isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        _store_fx_cache(cache)
    except Exception:
        pass


def _fetch_with_fallback(symbols: List[str], start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    # Cap end date to today - never attempt to fetch future data
    end = _cap_date_to_today(end)
    
    last_err: Optional[Exception] = None
    for sym in symbols:
        # Try cached FX rates first to avoid extra network calls
        cached = _maybe_load_cached_series(sym, start, end)
        if cached is not None:
            # Ensure cached data is numeric
            cached = pd.to_numeric(cached, errors="coerce").dropna()
            if not cached.empty:
                return cached, sym
        try:
            px = _fetch_px_symbol(sym, start, end)
            # Ensure numeric dtype
            px = pd.to_numeric(px, errors="coerce").dropna()
            if px.empty:
                raise RuntimeError(f"No numeric data for {sym}")
            _maybe_store_cached_series(sym, px)
            return px, sym
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError(f"No data for symbols: {symbols}")


def fetch_px(pair: str, start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    """Fetch price series with mapping/dot-dash fallbacks.
    Returns (series, display_name).
    """
    candidates = _resolve_symbol_candidates(pair)
    last_err: Optional[Exception] = None
    for sym in candidates:
        try:
            px = _fetch_px_symbol(sym, start, end)
            # Ensure numeric dtype to prevent downstream multiplication errors
            px = pd.to_numeric(px, errors="coerce").dropna()
            if px.empty:
                raise RuntimeError(f"No numeric data for {sym}")
            px.name = "px"
            title = _resolve_display_name(sym)
            return px, title
        except Exception as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError(f"No data for {pair}")


def fetch_usd_to_pln_exchange_rate(start: Optional[str], end: Optional[str]) -> pd.Series:
    """Fetch USD/PLN exchange rate as a Series. Tries multiple routes:
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


def detect_quote_currency(symbol: str) -> str:
    """Try to detect the quote currency for a Yahoo symbol.
    Returns uppercase ISO code like 'USD','EUR','GBP','GBp','JPY', or '' if unknown.
    
    Uses a three-tier approach to minimize API calls:
    1. Check static KNOWN_SYMBOL_CURRENCIES dictionary
    2. Check in-memory cache of previously detected currencies
    3. Use suffix-based heuristics (no API call)
    4. Only call Yahoo Finance API as last resort
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return "USD"
    
    # Tier 1: Check static dictionary (instant, no API)
    if sym in KNOWN_SYMBOL_CURRENCIES:
        return KNOWN_SYMBOL_CURRENCIES[sym]
    
    # Also check without suffix for common patterns
    base_sym = sym.split(".")[0].split("=")[0].split("-")[0]
    if base_sym in KNOWN_SYMBOL_CURRENCIES:
        return KNOWN_SYMBOL_CURRENCIES[base_sym]
    
    # Tier 2: Check in-memory cache
    if sym in _CURRENCY_DETECTION_CACHE:
        return _CURRENCY_DETECTION_CACHE[sym]
    
    # Tier 3: Suffix-based heuristics (no API call)
    s = sym.upper()
    suffix_currency = None
    if s.endswith(".DE") or s.endswith(".F") or s.endswith(".BE") or s.endswith(".XETRA"):
        suffix_currency = "EUR"
    elif s.endswith(".PA"):
        suffix_currency = "EUR"
    elif s.endswith(".L") or s.endswith(".LON"):
        suffix_currency = "GBX"  # London often in pence
    elif s.endswith(".VI"):
        suffix_currency = "EUR"
    elif s.endswith(".AS"):  # Amsterdam
        suffix_currency = "EUR"
    elif s.endswith(".CO"):
        suffix_currency = "DKK"
    elif s.endswith(".OL"):
        suffix_currency = "NOK"
    elif s.endswith(".ST"):
        suffix_currency = "SEK"
    elif s.endswith(".TO") or s.endswith(".TSX"):
        suffix_currency = "CAD"
    elif s.endswith(".SZ") or s.endswith(".SS"):
        suffix_currency = "CNY"
    elif s.endswith(".KS") or s.endswith(".KQ"):
        suffix_currency = "KRW"
    elif s.endswith(".HK"):
        suffix_currency = "HKD"
    elif s.endswith(".T"):  # Tokyo
        suffix_currency = "JPY"
    elif "=X" in s:
        # FX pair - currency is complex, default to USD
        suffix_currency = "USD"
    elif "=F" in s:
        # Futures - typically USD
        suffix_currency = "USD"
    elif "-USD" in s:
        # Crypto pairs
        suffix_currency = "USD"
    
    if suffix_currency:
        _CURRENCY_DETECTION_CACHE[sym] = suffix_currency
        return suffix_currency
    
    # Tier 4: Yahoo Finance API (last resort, may cause rate limiting)
    # Only call API if we truly don't know the currency
    try:
        tk = yf.Ticker(symbol)
        cur = None
        # Try fast_info first (less likely to rate limit)
        try:
            fi = getattr(tk, "fast_info", None)
            if fi is not None:
                cur = fi.get("currency") if isinstance(fi, dict) else getattr(fi, "currency", None)
        except Exception:
            cur = None
        if not cur:
            # Try info (more likely to rate limit)
            try:
                info = tk.info or {}
                cur = info.get("currency")
            except Exception:
                cur = None
        if cur:
            detected = str(cur).upper()
            _CURRENCY_DETECTION_CACHE[sym] = detected
            return detected
    except Exception:
        pass
    
    # Default to USD for US-like symbols
    _CURRENCY_DETECTION_CACHE[sym] = "USD"
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
    # Always use pd.to_numeric first - this handles strings and mixed types robustly
    try:
        s = pd.to_numeric(s, errors="coerce")
    except Exception:
        pass
    # Then try astype to float for final conversion
    try:
        s = s.astype(float)
        return s
    except Exception:
        pass
    # Last resort: build from numpy values squeezed to 1-D
    try:
        vals = np.asarray(s.values)
        if vals.ndim > 1:
            vals = vals.ravel()
        s = pd.Series(vals, index=s.index)
        s = pd.to_numeric(s, errors="coerce")
    except Exception:
        pass
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


def convert_currency_to_pln(quote_ccy: str, start: Optional[str], end: Optional[str], native_index: Optional[pd.DatetimeIndex] = None) -> pd.Series:
    """Return a Series of FX rate in PLN per 1 unit of quote_ccy.
    Expands the fetch window to cover the native price index +/- 30 days for overlap robustness."""
    q = (quote_ccy or "").upper().strip()
    # Expand window around native index
    s_ext, e_ext = start, end
    if native_index is not None and len(native_index) > 0:
        try:
            s_ext = (pd.to_datetime(native_index.min()) - pd.Timedelta(days=30)).date().isoformat()
            e_ext = (pd.to_datetime(native_index.max()) + pd.Timedelta(days=5)).date().isoformat()
            # Cap end date to today - never fetch future data
            e_ext = _cap_date_to_today(e_ext)
        except Exception:
            pass

    def fetch(sym_list: List[str]) -> pd.Series:
        s, _ = _fetch_with_fallback(sym_list, s_ext, e_ext)
        return s

    if q in ("PLN", "PLN "):
        # Return a flat-1 series over the native index for easy alignment
        return pd.Series(1.0, index=pd.DatetimeIndex(native_index) if native_index is not None else [pd.Timestamp("1970-01-01")])
    if q == "USD":
        return fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    if q == "EUR":
        try:
            eurpln, _ = _fetch_with_fallback(["EURPLN=X"], s_ext, e_ext)
            return eurpln
        except Exception:
            # EURPLN via USD: EURPLN = EURUSD * USDPLN
            eurusd, _ = _fetch_with_fallback(["EURUSD=X"], s_ext, e_ext)
            return eurusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
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
            return cadusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
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
            return chfusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    if q == "AUD":
        try:
            audpln, _ = _fetch_with_fallback(["AUDPLN=X"], s_ext, e_ext)
            return audpln
        except Exception:
            audusd, _ = _fetch_with_fallback(["AUDUSD=X"], s_ext, e_ext)
            return audusd * fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
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
            usdpln = fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
            usdhkd, _ = _fetch_with_fallback(["USDHKD=X"], s_ext, e_ext)
            return usdpln / usdhkd
    if q == "KRW":
        # PLN per KRW = (PLN per USD) / (KRW per USD)
        try:
            usdpln = fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
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
                usdpln = fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
                if native_index is not None and len(native_index) > 0:
                    usdpln = usdpln.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                    krwusd = krwusd.reindex(pd.DatetimeIndex(native_index)).ffill().bfill()
                return (usdpln * krwusd).rename("px")
            except Exception:
                pass
        # As last resort, assume USD (may be wrong for KRW assets but avoids crash)
        return fetch_usd_to_pln_exchange_rate(s_ext, e_ext)
    # Default: assume USD
    return fetch_usd_to_pln_exchange_rate(s_ext, e_ext)


def convert_price_series_to_pln(native_px: pd.Series, quote_ccy: str, start: Optional[str], end: Optional[str]) -> Tuple[pd.Series, str]:
    """Convert a native price series quoted in quote_ccy into PLN.
    Returns (pln_series, units_suffix).
    """
    sfx = "(PLN)"
    # Ensure native prices are numeric
    native_px = pd.to_numeric(native_px, errors="coerce").dropna()
    if native_px.empty:
        raise RuntimeError("No numeric native price data for PLN conversion")
    native_px = _ensure_float_series(native_px)
    # Get FX leg over the native range (with padding)
    fx = convert_currency_to_pln(quote_ccy, start, end, native_index=native_px.index)
    # Ensure FX series is numeric
    fx = pd.to_numeric(fx, errors="coerce").dropna()
    if fx.empty:
        raise RuntimeError(f"No numeric FX data for {quote_ccy} to PLN conversion")
    # Try increasingly permissive alignments
    fx_al = _align_fx_asof(native_px, fx, max_gap_days=7)
    if fx_al.isna().all():
        fx_al = _align_fx_asof(native_px, fx, max_gap_days=14)
    if fx_al.isna().all():
        fx_al = _align_fx_asof(native_px, fx, max_gap_days=30)
    # Fallback: strict calendar alignment with ffill/bfill
    if fx_al.isna().all():
        fx_al = fx.reindex(native_px.index).ffill().bfill()
    fx_al = pd.to_numeric(fx_al, errors="coerce")
    fx_al = _ensure_float_series(fx_al)
    if fx_al.isna().all() or fx_al.empty:
        raise RuntimeError(f"No overlapping FX data for {quote_ccy} to PLN conversion")
    pln = (native_px * fx_al).dropna()
    pln.name = "px"
    return pln, sfx


def _resolve_symbol_candidates(asset: str) -> List[str]:
    """Resolve symbol candidates using explicit mappings only.
    No automatic dot/dash variant generation to avoid hard-to-trace bugs.
    """
    a = asset.strip()
    u = a.upper()

    # Use global mapping for explicit symbol resolution
    mapping = dict(MAPPING)

    # Get explicit mappings only - no automatic variant generation
    mapped = mapping.get(u, [])

    # Build candidate list: mapped first, then original symbol
    candidates: List[str] = list(mapped)
    if u not in candidates:
        candidates.append(u)
    if a not in candidates:
        candidates.append(a)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for c in candidates:
        if c not in seen:
            deduped.append(c)
            seen.add(c)
    return deduped


def drop_first_k_from_kalman_cache(k: int = 4, cache_path: str = 'cache/kalman_q_cache.json') -> List[str]:
    """
    Remove the first k tickers from the Kalman q cache JSON (in file order).
    Returns the list of removed tickers (empty if none removed).
    Safe against missing/invalid cache and writes atomically like other helpers.
    """
    path = pathlib.Path(cache_path)
    if not path.exists():
        return []
    try:
        raw = path.read_text()
        data = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Failed to load cache {cache_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError('Cache JSON is not an object mapping tickers to entries')
    keys = list(data.keys())
    removed = keys[: max(0, int(k))]
    if not removed:
        return []
    for key in removed:
        data.pop(key, None)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return removed
