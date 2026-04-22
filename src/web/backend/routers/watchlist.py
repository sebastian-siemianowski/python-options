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
    if not os.path.isfile(_WATCHLIST_PATH):
        return []
    try:
        with open(_WATCHLIST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [s for s in data if isinstance(s, str)]
    if isinstance(data, dict) and isinstance(data.get("symbols"), list):
        return [s for s in data["symbols"] if isinstance(s, str)]
    return []


def _save(symbols: List[str]) -> None:
    os.makedirs(_DATA_DIR, exist_ok=True)
    tmp_path = _WATCHLIST_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"symbols": symbols}, f, indent=2)
    os.replace(tmp_path, _WATCHLIST_PATH)


def _normalize(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
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
