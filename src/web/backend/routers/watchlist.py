"""
Watchlist router — user-curated list of ticker symbols.

Persists to ``src/data/watchlist.json`` so the list survives server restarts.
The watchlist stores only symbol strings; the frontend filters the already-
loaded signals payload by these symbols and reuses the main AllAssetsTable
component so visual parity with the default view is automatic.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .ticker_aliases import TICKER_ALIASES, canonicalize

router = APIRouter()

# ── Persistence ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir))
_SRC_DIR = os.path.abspath(os.path.join(_BACKEND_DIR, os.pardir, os.pardir))
_DATA_DIR = os.path.join(_SRC_DIR, "data")
_WATCHLIST_PATH = os.path.join(_DATA_DIR, "watchlist.json")

# Allow letters, digits, dot, dash, caret, equals (covers FX, futures, indices).
_SYMBOL_RX = re.compile(r"^[A-Z0-9][A-Z0-9._=^\-]{0,19}$")
_MAX_SYMBOLS = 200

_lock = threading.Lock()


def _load() -> List[str]:
    """Read stored symbols from disk, canonicalizing and deduping on the fly.

    Older entries may have been written before the alias map existed (e.g. a
    raw ``"BAYN"`` instead of ``"BAYN.DE"``). We transparently heal them on
    read so the watchlist UI and the signals engine agree on symbol form.
    """
    if not os.path.isfile(_WATCHLIST_PATH):
        return []
    try:
        with open(_WATCHLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    raw: List[str]
    if isinstance(data, list):
        raw = [s for s in data if isinstance(s, str)]
    elif isinstance(data, dict) and isinstance(data.get("symbols"), list):
        raw = [s for s in data["symbols"] if isinstance(s, str)]
    else:
        return []

    # Canonicalize + dedupe while preserving insertion order.
    seen: set[str] = set()
    healed: List[str] = []
    for sym in raw:
        canon = canonicalize(sym)
        if not canon or not _SYMBOL_RX.match(canon):
            continue
        if canon in seen:
            continue
        seen.add(canon)
        healed.append(canon)

    # Persist if anything changed so the on-disk file self-heals over time.
    if healed != raw:
        try:
            _save(healed)
        except OSError:
            pass
    return healed


def _save(symbols: List[str]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp_path = _WATCHLIST_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"symbols": symbols}, f, indent=2)
    os.replace(tmp_path, _WATCHLIST_PATH)


def _normalize(symbol: str) -> str:
    # Apply the alias map FIRST so user-friendly inputs like "BAYN" or "HO"
    # resolve to their Yahoo form ("BAYN.DE", "HO.PA") before validation.
    # Unknown symbols fall through as upper-cased pass-through.
    sym = canonicalize(symbol)
    if not _SYMBOL_RX.match(sym):
        raise HTTPException(
            status_code=400,
            detail=(
                "Symbol must start with a letter or digit and contain only "
                "A-Z, 0-9, '.', '-', '=', or '^' (max 20 chars)."
            ),
        )
    return sym


# ── Schemas ──────────────────────────────────────────────────────────────────


class WatchlistResponse(BaseModel):
    symbols: List[str]


class WatchlistAddRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("", response_model=WatchlistResponse)
@router.get("/", response_model=WatchlistResponse)
def get_watchlist() -> WatchlistResponse:
    with _lock:
        return WatchlistResponse(symbols=_load())


@router.post("", response_model=WatchlistResponse, status_code=201)
@router.post("/", response_model=WatchlistResponse, status_code=201)
def add_symbol(req: WatchlistAddRequest) -> WatchlistResponse:
    sym = _normalize(req.symbol)
    with _lock:
        symbols = _load()
        if sym in symbols:
            # Idempotent add — return existing list without error.
            return WatchlistResponse(symbols=symbols)
        if len(symbols) >= _MAX_SYMBOLS:
            raise HTTPException(
                status_code=400,
                detail=f"Watchlist limit reached ({_MAX_SYMBOLS}).",
            )
        symbols.append(sym)
        _save(symbols)
        return WatchlistResponse(symbols=symbols)


@router.delete("/{symbol}", response_model=WatchlistResponse)
def remove_symbol(symbol: str) -> WatchlistResponse:
    sym = _normalize(symbol)
    with _lock:
        symbols = _load()
        if sym not in symbols:
            raise HTTPException(
                status_code=404,
                detail=f"{sym} is not in the watchlist.",
            )
        symbols = [s for s in symbols if s != sym]
        _save(symbols)
        return WatchlistResponse(symbols=symbols)


class ProxyMapResponse(BaseModel):
    # Map of user-facing symbol -> primary symbol actually tuned / present in
    # signals output.  Needed so the watchlist UI can match chips like
    # "DFNG" (proxies to "ITA") or "XAGUSD=X" (proxies to "SI=F") against
    # the signals set and avoid showing false "not matched" warnings.
    proxies: dict[str, str]


@router.get("/proxy-map", response_model=ProxyMapResponse)
def get_proxy_map() -> ProxyMapResponse:
    proxies: dict[str, str] = {}
    try:
        # Import lazily so this module has no hard dependency on ingestion at
        # import time (the ingestion module pulls heavy numeric deps).
        from ingestion.data_utils import DEFAULT_ASSET_UNIVERSE, PROXY_OVERRIDES, normalize_yahoo_ticker  # type: ignore
    except Exception:
        DEFAULT_ASSET_UNIVERSE = []
        PROXY_OVERRIDES = {}
        normalize_yahoo_ticker = None

    # Defensive copy as plain dict of str->str. Include both explicit proxy
    # overrides and deterministic Yahoo normalisations from the internal
    # universe so persisted raw watchlist symbols can match tuned signal rows.
    for key, value in PROXY_OVERRIDES.items():
        proxies[str(key).upper()] = str(value).upper()

    for key, value in TICKER_ALIASES.items():
        proxies[str(key).upper()] = str(value).upper()

    if normalize_yahoo_ticker is not None:
        for symbol in DEFAULT_ASSET_UNIVERSE:
            try:
                normalized, meta = normalize_yahoo_ticker(str(symbol), perform_lookup=False)
            except Exception:
                continue
            if normalized and str(normalized).upper() != str(symbol).upper():
                proxies[str(symbol).upper()] = str(normalized).upper()
                if isinstance(meta, dict):
                    original = meta.get("original")
                    if original:
                        proxies[str(original).upper()] = str(normalized).upper()

    return ProxyMapResponse(proxies=proxies)
