"""
Shared asset classification for Student-t calibration profiles.

The retune workflow sources its ticker universe from ingestion.data_utils.
These helpers keep model-side priors aligned with that internal universe while
still falling back gracefully for ad hoc symbols.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Optional

import numpy as np


_INDEX_ALIAS_MAP = {
    "_GSPC": "^GSPC",
    "_IXIC": "^IXIC",
    "_DJI": "^DJI",
    "_RUT": "^RUT",
    "_NDX": "^NDX",
    "_VIX": "^VIX",
}

_YAHOO_ALIAS_MAP = {
    "ACP": "ACP.WA",
    "AM": "AM.PA",
    "AIR": "AIR.PA",
    "BAYN": "BAYN.DE",
    "BMW3": "BMW3.DE",
    "FACC": "FACC.VI",
    "HAG": "HAG.DE",
    "HO": "HO.PA",
    "KOG": "KOG.OL",
    "MAGD": "MAGD.L",
    "MDA": "MDA.TO",
    "R3NK": "R3NK.DE",
    "RHM": "RHM.DE",
    "SAF": "SAF.PA",
    "SNT": "SNT.WA",
    "TKA": "TKA.DE",
    "THEON": "THEON.AS",
    "VOW3": "VOW3.DE",
    "XAGUSD": "XAGUSD=X",
    "XAUUSD": "XAUUSD=X",
}


def normalize_symbol(symbol: object) -> str:
    """Normalize Yahoo symbols and cache-safe variants for classification."""
    if symbol is None:
        return ""
    sym = str(symbol).strip().upper()
    if not sym:
        return ""
    sym = _INDEX_ALIAS_MAP.get(sym, sym)
    sym = _YAHOO_ALIAS_MAP.get(sym, sym)
    if sym.endswith("_X") and len(sym) >= 8:
        sym = f"{sym[:-2]}=X"
    elif sym.endswith("_F") and len(sym) >= 5:
        sym = f"{sym[:-2]}=F"
    elif sym.endswith("_USD") and "-" not in sym:
        sym = f"{sym[:-4]}-USD"
    if sym == "BRK.B":
        sym = "BRK-B"
    return sym


METALS_GOLD_SYMBOLS = frozenset({
    "GC=F", "XAUUSD", "XAUUSD=X", "GLD", "GLDM", "IAU", "SGOL", "SGLP",
    "SGLP.L", "GLDE", "GLDW", "GDX", "GDXJ",
    "NEM", "GOLD", "B", "AEM", "KGC", "CDE", "IAUX", "HYMC", "GROY",
    "WPM", "RGLD",
})

METALS_SILVER_SYMBOLS = frozenset({
    "SI=F", "XAGUSD", "XAGUSD=X", "SLV", "SIVR", "SLVI", "SLVR",
    "PAAS", "HL", "EXK", "SVM", "AG", "GORO", "USAS",
})

METALS_OTHER_SYMBOLS = frozenset({
    "HG=F", "PL=F", "PA=F", "COPX", "PPLT", "FCX", "SCCO", "TECK",
    "HBM", "IVN.TO", "MP", "LYSCF", "UUUU", "ILKAF", "ALB", "SQM",
    "ATI", "MTRN", "CCJ", "DNN", "NXE", "UEC", "LEU", "REMX", "NLR",
    "CRML", "IDR", "UAMY", "ABAT", "ASPI",
})

CRYPTO_SYMBOLS = frozenset({
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", "DOGE-USD",
    "AVAX-USD", "MATIC-USD", "LINK-USD", "XRP-USD", "BNB-USD",
    "SHIB-USD", "LTC-USD", "BCH-USD",
})

INDEX_SYMBOLS = frozenset({
    "^GSPC", "^IXIC", "^DJI", "^RUT", "^NDX", "^VIX", "SPX", "NDX",
    "DJIA", "VIX",
    "SPY", "VOO", "VTI", "QQQ", "IWM", "OEF", "DIA", "RSP", "SMH",
    "SOXX", "DFNG", "ITA", "AFK", "ANGL", "CNXT", "DURA", "GLIN",
    "MOTG", "IDX", "MLN", "MOAT", "MOO", "MOTI", "OIH", "PPH",
    "ESPO", "GFA", "TRET", "VNQ", "QQQO", "MAGD", "MAGD.L",
    "XLE", "XLK", "XLC", "XLB", "XLP", "XLU", "XLI", "XLF", "XLV",
    "XLRE", "XLY",
})

LARGE_CAP_SYMBOLS = frozenset({
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "BRK-B", "JPM", "JNJ", "UNH", "V", "MA", "PG", "HD", "BAC",
    "XOM", "CVX", "ABBV", "KO", "PEP", "COST", "MRK", "AVGO", "LLY",
    "TMO", "ORCL", "ACN", "CRM", "ADBE", "NFLX", "AMD", "INTC",
    "QCOM", "TXN", "GS", "MS", "SCHW", "BLK", "AXP", "WFC", "C",
    "BK", "USB", "PYPL", "AIG", "COF", "MET", "CAT", "DE", "GE",
    "UPS", "UNP", "FDX", "EMR", "MMM", "BA", "HON", "LMT", "RTX",
    "NOC", "GD", "LHX", "TDG", "HWM", "TXT", "LDOS", "CW", "LIN",
    "PFE", "DIS", "NKE", "SBUX", "MCD", "LOW", "TGT", "BKNG", "GM",
    "WMT", "CL", "MDLZ", "MO", "PM", "ABT", "AMGN", "BMY", "CVS",
    "DHR", "GILD", "ISRG", "MDT", "NVO", "REGN", "COP", "NEE",
    "SO", "SPG", "IBM", "CSCO", "INTU", "NOW", "ASML", "TSM",
    "ARM", "AMAT", "LRCX", "ADI", "NXPI", "MPWR", "ETN", "ANET",
    "SNPS", "CDNS", "MRVL", "MU", "VRTX", "ALNY",
})

HIGH_VOL_EQUITY_SYMBOLS = frozenset({
    "MSTR", "AMZE", "RCAT", "SMCI", "RGTI", "QBTS", "BKSY", "SPCE",
    "ABTC", "BZAI", "BNZI", "AIRI", "ESLT", "QS", "QUBT", "PACB",
    "APLM", "NVTS", "ACHR", "GORO", "USAS", "ONDS", "GPUS",
    "SOUN", "SYM", "LUNR", "RKLB", "ASTS", "PL", "DPRO",
    "OKLO", "CRML", "UAMY", "IREN", "NBIS", "CIFR", "CRWV", "GLXY",
    "NUTX", "SEZL", "PGY", "ALAB", "CRDO", "ENVX", "SMR", "RXRX",
    "SDGR", "ABCL", "NTLA", "TEM", "ARQQ", "SPIR", "GSAT", "CFLT",
    "ESTC", "PATH", "FLNC", "ENPH", "BE", "AGNC", "ANNA", "ATAI",
    "AIRE", "BMHL", "BTCS", "ANGX", "ASPI", "ABAT", "ADUR", "APLD",
    "ALMU", "AIFF", "AOUT", "ASTC", "AIRO", "EVEX", "EVTL", "FJET",
    "HOVR", "KITT", "MNTS", "OPXS", "POWW", "PRZO", "SATL", "SIDU",
    "SKYH", "SPAI", "VTSI", "VWAV",
})

SMALL_CAP_SYMBOLS = frozenset({
    "UPST", "AFRM", "IONQ", "SNAP", "DKNG", "PLTR", "SOFI", "RIVN",
    "LCID", "RIOT", "MARA", "NET", "RBLX", "HOOD", "COIN", "CRWD",
    "DDOG", "ZS", "SNOW", "MDB", "HUBS", "IOT", "GLBE", "GRND", "DUOL",
    "CELH", "CRSP", "BEAM", "ASTS", "AVAV",
    "KTOS", "FTAI", "MRCY", "NU", "IOT", "FCX", "HBM", "MP", "UUUU",
    "DNN", "UEC", "GEV", "LEU", "NXE", "WOLF", "AEHR", "AOSL",
    "POWI", "VSH", "VRT", "JCI", "PWR", "CRS", "KMT", "ILMN",
    "BBIO", "APLS", "BETA", "BCAL", "ARR",
}) | HIGH_VOL_EQUITY_SYMBOLS


@lru_cache(maxsize=1)
def get_internal_universe_symbols() -> frozenset[str]:
    """Return normalized symbols used by the retune/default asset universe."""
    symbols = set()
    for module_name in ("ingestion.data_utils", "src.ingestion.data_utils"):
        try:
            module = __import__(module_name, fromlist=["DEFAULT_ASSET_UNIVERSE"])
            universe = getattr(module, "DEFAULT_ASSET_UNIVERSE", [])
            symbols.update(normalize_symbol(sym) for sym in universe)
            break
        except Exception:
            continue
    return frozenset(sym for sym in symbols if sym)


@lru_cache(maxsize=1)
def get_market_analysis_symbols() -> frozenset[str]:
    """Return normalized market index, broad ETF, and sector ETF symbols."""
    symbols = set()
    for module_name in ("ingestion.data_utils", "src.ingestion.data_utils"):
        try:
            module = __import__(module_name, fromlist=["MARKET_ANALYSIS_ETFS"])
            symbols.update(normalize_symbol(sym) for sym in getattr(module, "MARKET_ANALYSIS_ETFS", []))
            break
        except Exception:
            continue
    return frozenset(sym for sym in symbols if sym)


def is_internal_universe_symbol(symbol: object) -> bool:
    return normalize_symbol(symbol) in get_internal_universe_symbols()


def _annualized_volatility(returns: Iterable[float]) -> Optional[float]:
    try:
        arr = np.asarray(returns, dtype=float).ravel()
    except Exception:
        return None
    valid = arr[np.isfinite(arr)]
    if valid.size < 30:
        return None
    return float(np.std(valid) * np.sqrt(252.0))


def _class_from_returns(returns: Iterable[float]) -> Optional[str]:
    ann_vol = _annualized_volatility(returns)
    if ann_vol is None:
        return None
    if ann_vol >= 0.55:
        return "high_vol_equity"
    if ann_vol >= 0.35:
        return "small_cap"
    return "large_cap"


def is_forex_symbol(symbol: object) -> bool:
    sym = normalize_symbol(symbol)
    if sym in {"XAUUSD=X", "XAGUSD=X"}:
        return False
    return sym.endswith("=X")


def is_crypto_symbol(symbol: object) -> bool:
    sym = normalize_symbol(symbol)
    if sym in CRYPTO_SYMBOLS:
        return True
    base = sym.split("-", 1)[0]
    return sym.endswith("-USD") and base in {
        "BTC", "ETH", "SOL", "ADA", "DOT", "DOGE", "AVAX", "MATIC",
        "LINK", "XRP", "BNB", "SHIB", "LTC", "BCH",
    }


def is_index_symbol(symbol: object) -> bool:
    sym = normalize_symbol(symbol)
    return sym.startswith("^") or sym in INDEX_SYMBOLS or sym in get_market_analysis_symbols()


def detect_asset_class(asset_symbol: object, returns: Iterable[float] | None = None) -> Optional[str]:
    """
    Detect the model calibration class for a symbol.

    Returns detailed classes used by MS-q/profile logic, for example
    metals_gold, high_vol_equity, crypto, index, forex, large_cap, small_cap.
    """
    sym = normalize_symbol(asset_symbol)
    if not sym:
        return None

    if sym in METALS_GOLD_SYMBOLS:
        return "metals_gold"
    if sym in METALS_SILVER_SYMBOLS:
        return "metals_silver"
    if sym in METALS_OTHER_SYMBOLS or (sym.endswith("=F") and sym not in {"GC=F", "SI=F"}):
        return "metals_other"
    if sym == "GC=F":
        return "metals_gold"
    if sym == "SI=F":
        return "metals_silver"
    if is_forex_symbol(sym):
        return "forex"
    if is_crypto_symbol(sym):
        return "crypto"
    if is_index_symbol(sym):
        return "index"

    returns_class = _class_from_returns(returns) if returns is not None else None
    if returns_class == "high_vol_equity" or sym in HIGH_VOL_EQUITY_SYMBOLS:
        return "high_vol_equity"
    if returns_class in {"large_cap", "small_cap"}:
        return returns_class
    if sym in LARGE_CAP_SYMBOLS:
        return "large_cap"
    if sym in SMALL_CAP_SYMBOLS:
        return "small_cap"
    if sym in get_internal_universe_symbols():
        return "small_cap"
    return None


def classify_asset_for_phi(asset_symbol: object, returns: Iterable[float] | None = None) -> str:
    """
    Classify for phi-prior selection.

    Returns one of: index, large_cap, small_cap, crypto, metals, forex,
    high_vol, default.
    """
    asset_class = detect_asset_class(asset_symbol, returns=returns)
    if asset_class in {"metals_gold", "metals_silver", "metals_other"}:
        return "metals"
    if asset_class == "high_vol_equity":
        return "high_vol"
    if asset_class in {"index", "large_cap", "small_cap", "crypto", "forex"}:
        return asset_class
    return "default"
