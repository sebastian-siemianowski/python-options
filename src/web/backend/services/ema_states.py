"""
EMA states service — computes per-symbol EMA snapshots (9, 50, 600) so the
front-end can offer "below EMA" filters on the high-conviction lists.

Cached in-memory and invalidated when any underlying CSV mtime changes.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any, Dict, Optional

from web.backend.services.data_service import PRICES_DIR

_EMA_PERIODS = (9, 50, 600)
_CACHE_TTL_SEC = 300  # 5 min hard ceiling regardless of mtimes

_cache: Dict[str, Any] = {
    "built_at": 0.0,
    "mtime_signature": None,
    "states": {},  # symbol -> dict
}


def _signature() -> Optional[float]:
    """Cheap mtime sum over price CSVs — fingerprint for cache validity."""
    if not os.path.isdir(PRICES_DIR):
        return None
    sig = 0.0
    try:
        for fp in glob.iglob(os.path.join(PRICES_DIR, "*.csv")):
            try:
                sig += os.path.getmtime(fp)
            except OSError:
                continue
    except OSError:
        return None
    return sig


def _safe_float(val: Any) -> Optional[float]:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _compute_one(filepath: str) -> Optional[Dict[str, Any]]:
    """Read one CSV, return {price, ema9, ema50, ema600, below_*}."""
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        # Only need Close. Fast path: read just that column + tail.
        df = pd.read_csv(filepath, usecols=["Close"])
    except Exception:
        return None
    if df.empty:
        return None
    close = df["Close"].astype(float).dropna()
    if close.empty:
        return None

    price = _safe_float(close.iloc[-1])
    if price is None:
        return None

    out: Dict[str, Any] = {"price": price}
    n = len(close)
    for period in _EMA_PERIODS:
        # Need a reasonable warm-up — at least the period itself.
        if n < period:
            out[f"ema{period}"] = None
            out[f"below_{period}"] = None
            continue
        ema_val = _safe_float(close.ewm(span=period, adjust=False).mean().iloc[-1])
        out[f"ema{period}"] = ema_val
        out[f"below_{period}"] = (
            bool(price < ema_val) if ema_val is not None else None
        )
    return out


def _symbol_for(filepath: str) -> str:
    fname = os.path.basename(filepath)
    if fname.endswith("_1d.csv"):
        return fname[:-7]
    if fname.endswith(".csv"):
        return fname[:-4]
    return fname


def _build() -> Dict[str, Dict[str, Any]]:
    if not os.path.isdir(PRICES_DIR):
        return {}
    states: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(glob.glob(os.path.join(PRICES_DIR, "*.csv"))):
        snap = _compute_one(fp)
        if snap is None:
            continue
        sym = _symbol_for(fp)
        states[sym] = snap
    return states


def get_all_ema_states(force: bool = False) -> Dict[str, Any]:
    """
    Return EMA snapshot map keyed by symbol. Cached.

    Shape:
      {
        "states": { "AAPL": { "price": 224.1, "ema9": ..., "ema50": ...,
                              "ema600": ..., "below_9": true, ... }, ... },
        "count": 412,
        "periods": [9, 50, 600],
        "built_at": 1730000000.0
      }
    """
    now = time.time()
    sig = _signature()
    fresh = (
        not force
        and _cache["states"]
        and _cache["mtime_signature"] == sig
        and (now - _cache["built_at"]) < _CACHE_TTL_SEC
    )
    if not fresh:
        _cache["states"] = _build()
        _cache["mtime_signature"] = sig
        _cache["built_at"] = now
    return {
        "states": _cache["states"],
        "count": len(_cache["states"]),
        "periods": list(_EMA_PERIODS),
        "built_at": _cache["built_at"],
    }
