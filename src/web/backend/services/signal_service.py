"""
Signal service — reads cached signal data and high conviction signals.

Includes in-memory caching to avoid re-reading JSON on every request.
"""

import json
import math
import os
import glob
import time
from typing import Any, Dict, List, Optional

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
DATA_DIR = os.path.join(SRC_DIR, "data")

DEFAULT_CACHE_PATH = os.path.join(DATA_DIR, "currencies", "fx_plnjpy.json")
HIGH_CONVICTION_DIR = os.path.join(DATA_DIR, "high_conviction")

# ── In-memory cache ─────────────────────────────────────────────────
_signal_cache: Dict[str, Any] = {}
_signal_cache_mtime: float = 0.0


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None for JSON compliance."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _invalidate_signal_cache() -> None:
    """Force reload on next access."""
    global _signal_cache, _signal_cache_mtime
    _signal_cache = {}
    _signal_cache_mtime = 0.0


def get_cached_signals(cache_path: str = DEFAULT_CACHE_PATH) -> Optional[Dict[str, Any]]:
    """Load signals from the JSON cache, with in-memory caching by mtime."""
    global _signal_cache, _signal_cache_mtime
    if not os.path.isfile(cache_path):
        return None
    try:
        mtime = os.path.getmtime(cache_path)
        if _signal_cache and mtime == _signal_cache_mtime:
            return _signal_cache
        with open(cache_path, "r") as f:
            data = json.load(f)
        _signal_cache = data
        _signal_cache_mtime = mtime
        return data
    except (json.JSONDecodeError, IOError):
        return None


def get_summary_rows(cache_path: str = DEFAULT_CACHE_PATH) -> List[Dict[str, Any]]:
    """Return the summary_rows array from the signal cache."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return [_decorate_summary_row(row) for row in data.get("summary_rows", [])]


def get_asset_blocks(cache_path: str = DEFAULT_CACHE_PATH) -> List[Dict[str, Any]]:
    """Return the full asset blocks with all signal fields."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return data.get("assets", [])


def get_failed_assets(cache_path: str = DEFAULT_CACHE_PATH) -> List[str]:
    """Return list of assets that failed during signal computation."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return data.get("failed_assets", [])


def get_horizons(cache_path: str = DEFAULT_CACHE_PATH) -> List[int]:
    """Return the horizons used in the signal computation."""
    data = get_cached_signals(cache_path)
    if data is None:
        return []
    return data.get("horizons", [])


def get_high_conviction_signals(signal_type: str = "buy") -> List[Dict[str, Any]]:
    """
    Load high conviction signals from the buy/ or sell/ directory.
    
    Args:
        signal_type: 'buy' or 'sell'
    
    Returns:
        List of signal dictionaries
    """
    dir_path = os.path.join(HIGH_CONVICTION_DIR, signal_type)
    if not os.path.isdir(dir_path):
        return []

    signals = []
    for filepath in sorted(glob.glob(os.path.join(dir_path, "*.json"))):
        # Skip manifest/metadata files
        if os.path.basename(filepath) == "manifest.json":
            continue
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    signals.extend(entry for entry in data if "ticker" in entry)
                elif "ticker" in data:
                    signals.append(data)
        except (json.JSONDecodeError, IOError):
            continue
    return _sanitize_for_json(signals)


def get_cache_age_seconds(cache_path: str = DEFAULT_CACHE_PATH) -> Optional[float]:
    """Return the age of the signal cache in seconds, or None if missing."""
    if not os.path.isfile(cache_path):
        return None
    import time
    return time.time() - os.path.getmtime(cache_path)


# Sector consolidation mapping — mirrors signals_ux.py.
#
# Signals page display is intentionally a little more opinionated than the raw
# model cache: currencies get their own panel, while the key market indices sit
# with commodities/crypto so the macro pulse is visible near the top.
SECTOR_CONSOLIDATION = {
    "Indices / Broad ETFs": "Indices & ETFs",
    "Indices": "Indices & ETFs",
    "Nuclear": "Critical Materials & Nuclear",
    "Critical Materials": "Critical Materials & Nuclear",
    "AI Utility / Infrastructure": "AI & Infrastructure",
    "AI Software / Data Platforms": "AI & Infrastructure",
    "AI Hardware / Edge Compute": "AI & Infrastructure",
    "AI Power Semiconductors": "AI & Infrastructure",
    "Semiconductor Equipment": "Semiconductors",
    "FX / Commodities / Crypto": "Core Markets, Commodities & Crypto",
}

KEY_MARKET_SYMBOLS = {"^GSPC", "^NDX", "SPY", "QQQ"}
ISO_CURRENCY_CODES = {
    "AUD", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD",
    "HUF", "INR", "JPY", "KRW", "MXN", "NOK", "NZD", "PLN", "SEK", "SGD",
    "TRY", "USD", "ZAR",
}


def _extract_symbol_from_label(label: str) -> str:
    """Extract `SYM` from labels like `Company Name (SYM)`."""
    if not label:
        return ""
    if "(" in label and ")" in label:
        return label.rsplit("(", 1)[-1].split(")", 1)[0].strip()
    return label.strip()


def _row_symbol(row: Dict[str, Any]) -> str:
    return str(row.get("symbol") or _extract_symbol_from_label(row.get("asset_label", ""))).strip().upper()


def _is_currency_symbol(symbol: str) -> bool:
    """Return true for fiat FX pairs such as PLNJPY=X, but not XAGUSD=X."""
    if not symbol.endswith("=X"):
        return False
    pair = symbol[:-2].replace("/", "").replace("-", "").upper()
    if len(pair) != 6:
        return False
    base, quote = pair[:3], pair[3:]
    return base in ISO_CURRENCY_CODES and quote in ISO_CURRENCY_CODES


def _display_sector_for_row(row: Dict[str, Any]) -> str:
    symbol = _row_symbol(row)
    if _is_currency_symbol(symbol):
        return "Currencies"
    if symbol in KEY_MARKET_SYMBOLS:
        return "Core Markets, Commodities & Crypto"
    return _consolidate_sector(row.get("sector", ""))


def _decorate_summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    raw_sector = row.get("sector", "")
    out["raw_sector"] = raw_sector
    out["sector"] = _display_sector_for_row(row)
    return out


def _consolidate_sector(sector: str) -> str:
    """Map raw sector to consolidated display sector."""
    return SECTOR_CONSOLIDATION.get(sector, sector) if sector else "Other"


def _classify_label(label: str) -> str:
    """Normalise signal label to one of 6 categories."""
    label = (label or "HOLD").upper().strip()
    if label == "STRONG BUY":
        return "strong_buy"
    if label == "STRONG SELL":
        return "strong_sell"
    if "BUY" in label:
        return "buy"
    if "SELL" in label:
        return "sell"
    if label == "EXIT":
        return "exit"
    return "hold"


def get_signal_stats(cache_path: str = DEFAULT_CACHE_PATH) -> Dict[str, Any]:
    """Return summary statistics about the current signal cache.

    Counts are per-asset (using nearest_label), NOT per-horizon.
    """
    data = get_cached_signals(cache_path)
    if data is None:
        return {"cached": False, "total_assets": 0, "failed": 0}

    summary_rows = data.get("summary_rows", [])
    failed = data.get("failed_assets", [])
    counts = {"strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0, "exit": 0}

    for row in summary_rows:
        # Use nearest_label for per-asset classification (not horizon_signals)
        nearest = row.get("nearest_label", "HOLD")
        cat = _classify_label(nearest)
        counts[cat] = counts.get(cat, 0) + 1

    age = get_cache_age_seconds(cache_path)
    return {
        "cached": True,
        "total_assets": len(summary_rows),
        "failed": len(failed),
        "strong_buy_signals": counts["strong_buy"],
        "buy_signals": counts["buy"],
        "sell_signals": counts["sell"],
        "hold_signals": counts["hold"],
        "strong_sell_signals": counts["strong_sell"],
        "exit_signals": counts["exit"],
        "cache_age_seconds": round(age, 1) if age is not None else None,
    }


def get_signals_by_sector(cache_path: str = DEFAULT_CACHE_PATH) -> List[Dict[str, Any]]:
    """Group summary rows by consolidated sector with aggregate stats."""
    rows = get_summary_rows(cache_path)
    sectors: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        sector = _consolidate_sector(row.get("sector", ""))
        if sector not in sectors:
            sectors[sector] = {
                "name": sector,
                "assets": [],
                "strong_buy": 0,
                "buy": 0,
                "hold": 0,
                "sell": 0,
                "strong_sell": 0,
                "exit": 0,
                "momentum_scores": [],
                "crash_risk_scores": [],
            }
        s = sectors[sector]
        s["assets"].append(row)
        m = row.get("momentum_score")
        if m is not None:
            s["momentum_scores"].append(m)
        cr = row.get("crash_risk_score")
        if cr is not None:
            s["crash_risk_scores"].append(cr)
        for _hz, sig in row.get("horizon_signals", {}).items():
            cat = _classify_label(sig.get("label"))
            s[cat] = s.get(cat, 0) + 1

    result = []
    for sec in sorted(sectors.values(), key=lambda x: (-len(x["assets"]), x["name"])):
        ms = sec.pop("momentum_scores")
        crs = sec.pop("crash_risk_scores")
        sec["asset_count"] = len(sec["assets"])
        sec["avg_momentum"] = round(sum(ms) / len(ms), 2) if ms else 0
        sec["avg_crash_risk"] = round(sum(crs) / len(crs), 3) if crs else 0
        result.append(sec)
    return result


def get_strong_signal_symbols(cache_path: str = DEFAULT_CACHE_PATH) -> Dict[str, List[Dict[str, Any]]]:
    """Return symbols that have STRONG BUY or STRONG SELL labels, grouped."""
    rows = get_summary_rows(cache_path)
    strong_buy = []
    strong_sell = []

    for row in rows:
        for _hz, sig in row.get("horizon_signals", {}).items():
            label = (sig.get("label") or "").upper().strip()
            if label == "STRONG BUY":
                # Extract symbol from asset_label "Company Name (SYM)" → SYM
                asset_label = row.get("asset_label", "")
                sym = asset_label.split("(")[-1].rstrip(")").strip() if "(" in asset_label else asset_label
                strong_buy.append({
                    "symbol": sym,
                    "asset_label": asset_label,
                    "sector": _consolidate_sector(row.get("sector", "")),
                    "horizon": _hz,
                    "p_up": sig.get("p_up", 0.5),
                    "exp_ret": sig.get("exp_ret", 0),
                    "momentum": row.get("momentum_score", 0),
                })
                break  # one entry per asset
            elif label == "STRONG SELL":
                asset_label = row.get("asset_label", "")
                sym = asset_label.split("(")[-1].rstrip(")").strip() if "(" in asset_label else asset_label
                strong_sell.append({
                    "symbol": sym,
                    "asset_label": asset_label,
                    "sector": _consolidate_sector(row.get("sector", "")),
                    "horizon": _hz,
                    "p_up": sig.get("p_up", 0.5),
                    "exp_ret": sig.get("exp_ret", 0),
                    "momentum": row.get("momentum_score", 0),
                })
                break

    return {"strong_buy": strong_buy, "strong_sell": strong_sell}
