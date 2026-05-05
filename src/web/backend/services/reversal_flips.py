"""
Recent price-reversal flips used by the Signals and Watchlist quick filters.

The detector mirrors the SignalDetailPanel reversal overlay: ATR-smoothed bands,
a 2.4x threshold, and BUY/SELL labels only when the trend state flips.
"""

import math
import csv
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from web.backend.services.data_service import PRICES_DIR
from web.backend.services.signal_service import get_summary_rows


ATR_PERIOD = 14
REVERSAL_MULTIPLIER = 2.4
_CACHE_TTL_SECONDS = 60.0
_cache_key: Optional[Tuple[int, int]] = None
_cache_built_at: float = 0.0
_cache_payload: Dict[str, Any] = {}


def _extract_symbol_from_label(label: str) -> str:
    if not label:
        return ""
    if "(" in label and ")" in label:
        return label.rsplit("(", 1)[-1].split(")", 1)[0].strip()
    return label.strip()


def _row_symbol(row: Dict[str, Any]) -> str:
    return str(row.get("symbol") or _extract_symbol_from_label(row.get("asset_label", ""))).strip()


def _finite_number(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _price_file_candidates(symbol: str) -> List[str]:
    sanitized = symbol.replace("=", "_").replace("-", "_")
    return [
        f"{symbol}_1d.csv",
        f"{symbol.upper()}_1d.csv",
        f"{symbol}.csv",
        f"{symbol.upper()}.csv",
        f"{symbol.replace('-', '_')}_1d.csv",
        f"{symbol.replace('-', '_')}.csv",
        f"{sanitized}_1d.csv",
        f"{sanitized.upper()}_1d.csv",
        f"{sanitized}.csv",
        f"{sanitized.upper()}.csv",
    ]


def _load_price_bars(symbol: str, tail: int) -> List[Dict[str, Any]]:
    filepath = None
    for pattern in _price_file_candidates(symbol):
        candidate = os.path.join(PRICES_DIR, pattern)
        if os.path.isfile(candidate):
            filepath = candidate
            break
    if not filepath:
        return []

    bars: List[Dict[str, Any]] = []
    window: deque[Dict[str, Any]] = deque(maxlen=tail)
    try:
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                open_ = _finite_number(row.get("Open") or row.get("open"))
                high = _finite_number(row.get("High") or row.get("high"))
                low = _finite_number(row.get("Low") or row.get("low"))
                close = _finite_number(row.get("Close") or row.get("close"))
                date = row.get("Date") or row.get("date") or row.get("time")
                if not date or open_ is None or high is None or low is None or close is None:
                    continue
                window.append({
                    "date": str(date),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                })
    except OSError:
        return []
    bars.extend(window)
    return bars


def _compute_atr(bars: List[Dict[str, Any]], period: int = ATR_PERIOD) -> List[float]:
    true_ranges: List[float] = []
    for i, bar in enumerate(bars):
        if i == 0:
            tr = max(0.0, bar["high"] - bar["low"])
        else:
            prev_close = bars[i - 1]["close"]
            tr = max(
                max(0.0, bar["high"] - bar["low"]),
                abs(bar["high"] - prev_close),
                abs(bar["low"] - prev_close),
            )
        true_ranges.append(tr)

    atr: List[float] = []
    for i, tr in enumerate(true_ranges):
        start = max(0, i - period + 1)
        window = true_ranges[start:i + 1]
        avg = sum(window) / max(1, len(window))
        fallback = max(1e-9, bars[i]["high"] - bars[i]["low"])
        atr.append(avg or fallback or tr or 1e-9)
    return atr


def _latest_recent_flip(symbol: str, bars: List[Dict[str, Any]], recent_days: int) -> Dict[str, Any]:
    empty = {
        "symbol": symbol,
        "signal": None,
        "signal_date": None,
        "bars_ago": None,
        "price": None,
    }
    if len(bars) < 2:
        return empty

    atr = _compute_atr(bars)
    trend = 1 if bars[0]["close"] >= bars[0]["open"] else -1
    mid0 = (bars[0]["high"] + bars[0]["low"]) / 2.0
    final_upper = mid0 + REVERSAL_MULTIPLIER * atr[0]
    final_lower = mid0 - REVERSAL_MULTIPLIER * atr[0]
    latest_flip: Optional[Dict[str, Any]] = None

    for i, bar in enumerate(bars):
        prev_bar = bars[max(0, i - 1)]
        midpoint = (bar["high"] + bar["low"]) / 2.0
        basic_upper = midpoint + REVERSAL_MULTIPLIER * atr[i]
        basic_lower = midpoint - REVERSAL_MULTIPLIER * atr[i]
        prev_trend = trend

        if i > 0:
            final_upper = basic_upper if basic_upper < final_upper or prev_bar["close"] > final_upper else final_upper
            final_lower = basic_lower if basic_lower > final_lower or prev_bar["close"] < final_lower else final_lower

            if prev_trend == 1 and bar["close"] < final_lower:
                trend = -1
            elif prev_trend == -1 and bar["close"] > final_upper:
                trend = 1

        if i > 0 and trend != prev_trend:
            latest_flip = {
                "symbol": symbol,
                "signal": "buy" if trend == 1 else "sell",
                "signal_date": bar["date"],
                "bars_ago": len(bars) - 1 - i,
                "price": round(bar["close"], 4),
            }

    if not latest_flip:
        return empty
    if latest_flip["bars_ago"] is None or latest_flip["bars_ago"] >= recent_days:
        return empty
    return latest_flip


def get_recent_reversal_flips(recent_days: int = 4, tail: int = 365) -> Dict[str, Any]:
    """
    Return recent BUY/SELL reversal flips for every symbol in the signal summary.

    `recent_days` is interpreted as recent trading bars, inclusive of the latest
    bar. A value of 4 therefore covers today plus the previous three bars.
    """
    global _cache_key, _cache_built_at, _cache_payload

    safe_recent_days = max(1, min(int(recent_days or 4), 30))
    safe_tail = max(40, min(int(tail or 365), 2000))
    key = (safe_recent_days, safe_tail)
    now = time.time()
    if _cache_key == key and _cache_payload and now - _cache_built_at < _CACHE_TTL_SECONDS:
        return _cache_payload

    rows = get_summary_rows()
    symbols: List[str] = []
    seen = set()
    for row in rows:
        symbol = _row_symbol(row)
        if not symbol:
            continue
        upper = symbol.upper()
        if upper in seen:
            continue
        seen.add(upper)
        symbols.append(symbol)

    signals: Dict[str, Dict[str, Any]] = {}
    counts = {"buy": 0, "sell": 0}
    latest_date: Optional[str] = None

    for symbol in symbols:
        bars = _load_price_bars(symbol, safe_tail)
        if bars:
            latest_date = max(latest_date or bars[-1]["date"], bars[-1]["date"])
        entry = _latest_recent_flip(symbol, bars, safe_recent_days)
        signals[symbol] = entry
        signal = entry.get("signal")
        if signal in counts:
            counts[signal] += 1

    payload = {
        "signals": signals,
        "counts": counts,
        "recent_days": safe_recent_days,
        "tail": safe_tail,
        "total": len(symbols),
        "latest_date": latest_date,
        "built_at": now,
    }
    _cache_key = key
    _cache_built_at = now
    _cache_payload = payload
    return payload
