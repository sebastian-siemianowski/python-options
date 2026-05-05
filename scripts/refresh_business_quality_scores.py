#!/usr/bin/env python3
"""
Refresh business quality scores from the configured asset universe.

The UI consumes ``src/web/backend/services/quality_scores.py``.  That module
keeps a human-curated fallback table, while this script creates a dated JSON
overlay with per-symbol component scores and rationale.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
QUALITY_SERVICE = REPO_ROOT / "src" / "web" / "backend" / "services" / "quality_scores.py"
UNIVERSE_FILE = REPO_ROOT / "src" / "ingestion" / "data_utils.py"
ALIASES_FILE = REPO_ROOT / "src" / "web" / "backend" / "routers" / "ticker_aliases.py"
PRICE_DIR = REPO_ROOT / "src" / "data" / "prices"
OUTPUT_PATH = REPO_ROOT / "src" / "data" / "quality" / "business_quality_scores.json"


SNAPSHOT_FIELDS = [
    "quoteType",
    "shortName",
    "longName",
    "sector",
    "industry",
    "marketCap",
    "enterpriseValue",
    "totalRevenue",
    "revenueGrowth",
    "earningsGrowth",
    "grossMargins",
    "operatingMargins",
    "profitMargins",
    "returnOnEquity",
    "freeCashflow",
    "operatingCashflow",
    "totalCash",
    "totalDebt",
    "currentRatio",
    "debtToEquity",
    "beta",
    "overallRisk",
    "auditRisk",
    "boardRisk",
    "compensationRisk",
    "shareHolderRightsRisk",
    "priceToBook",
    "trailingPE",
    "forwardPE",
    "currency",
]


ISO_CCY = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "CAD",
    "NZD",
    "SGD",
    "HKD",
    "ZAR",
    "MXN",
    "TRY",
    "SEK",
    "NOK",
    "DKK",
    "CNY",
    "PLN",
    "KRW",
}


BROAD_ETFS = {
    "SPY",
    "VOO",
    "QQQ",
    "IWM",
    "OEF",
    "DIA",
    "XLK",
    "XLC",
    "XLF",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLE",
    "XLU",
    "XLRE",
    "XLB",
    "SMH",
    "SOXX",
    "MOAT",
    "MOTG",
    "MOTI",
}


STRUCTURED_PRODUCT_HINTS = (
    "OPTION",
    "OPTIONS",
    "YIELD",
    "INCOME",
    "COVERED CALL",
    "BUFFER",
    "DAILY TARGET",
    "2X",
    "3X",
    "LEVERAGED",
    "SHORT",
)


def load_literal_assignment(path: Path, name: str, default: Any) -> Any:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == name:
                    return ast.literal_eval(node.value)
    return default


def load_symbol_comments() -> dict[str, str]:
    labels: dict[str, str] = {}
    line_re = re.compile(r'"([^"]+)"\s*,\s*#\s*(.+)$')
    for line in UNIVERSE_FILE.read_text().splitlines():
        match = line_re.search(line)
        if not match:
            continue
        symbol, comment = match.groups()
        labels.setdefault(symbol.upper(), comment.strip())

    alias_re = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"\s*,\s*#\s*(.+)$')
    if ALIASES_FILE.exists():
        for line in ALIASES_FILE.read_text().splitlines():
            match = alias_re.search(line)
            if not match:
                continue
            raw, canonical, comment = match.groups()
            labels.setdefault(raw.upper(), comment.strip())
            labels.setdefault(canonical.upper(), comment.strip())
    return labels


def normalize_cached_stem(stem: str) -> str:
    if stem.endswith("_X"):
        return f"{stem[:-2]}=X"
    if stem.endswith("_F"):
        return f"{stem[:-2]}=F"
    return stem


def symbol_variants(symbol: str) -> list[str]:
    upper = symbol.strip().upper()
    variants = [
        upper,
        upper.replace("_X", "=X") if upper.endswith("_X") else upper,
        upper.replace("_F", "=F") if upper.endswith("_F") else upper,
        upper.replace("=", "_"),
        upper.replace("_", "="),
        upper.replace(".", "-"),
        upper.replace("-", "."),
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for item in variants:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def linear_score(value: Any, low: float, high: float, floor: float = 5.0, ceiling: float = 95.0) -> float | None:
    v = num(value)
    if v is None:
        return None
    if high == low:
        return None
    return clamp(floor + (v - low) * (ceiling - floor) / (high - low), floor, ceiling)


def average(values: list[float | None], default: float | None = None) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return default
    return sum(clean) / len(clean)


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def history_years(symbol: str) -> float | None:
    for variant in symbol_variants(symbol):
        safe = variant.replace("=", "_").replace("/", "_").replace(":", "_")
        path = PRICE_DIR / f"{safe}.csv"
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        if len(lines) <= 2:
            return 0.0
        first = lines[1].split(",", 1)[0]
        last = lines[-1].split(",", 1)[0]
        try:
            start = datetime.fromisoformat(first[:10])
            end = datetime.fromisoformat(last[:10])
        except ValueError:
            return len(lines) / 252.0
        return max(0.0, (end - start).days / 365.25)
    return None


def price_profile(symbol: str) -> dict[str, Any]:
    for variant in symbol_variants(symbol):
        safe = variant.replace("=", "_").replace("/", "_").replace(":", "_")
        path = PRICE_DIR / f"{safe}.csv"
        if not path.exists():
            continue
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        if len(lines) <= 1:
            return {}
        last = lines[-1].split(",")
        if len(last) < 7:
            return {}
        try:
            last_date = datetime.fromisoformat(last[0][:10]).date()
        except ValueError:
            last_date = None
        close_value = num(last[4])
        volume_value = num(last[6])
        stale_days = None
        if last_date is not None:
            stale_days = (datetime.now(timezone.utc).date() - last_date).days
        return {
            "last_date": last_date.isoformat() if last_date else None,
            "last_close": close_value,
            "last_volume": volume_value,
            "stale_days": stale_days,
        }
    return {}


def classify_asset(symbol: str, info: dict[str, Any]) -> str:
    upper = symbol.upper()
    quote_type = str(info.get("quoteType") or "").upper()
    name = str(info.get("shortName") or info.get("longName") or "").upper()

    if upper.startswith("^") or quote_type == "INDEX":
        return "index"
    if upper.endswith("=X"):
        pair = upper[:-2].replace("/", "").replace("-", "").replace("_", "")
        if pair.startswith(("XAU", "XAG")) or pair.endswith(("XAU", "XAG")):
            return "commodity"
        if len(pair) == 6 and pair[:3] in ISO_CCY and pair[3:] in ISO_CCY:
            return "currency"
    if upper.endswith("=F"):
        return "commodity"
    if upper.endswith("-USD") or quote_type == "CRYPTOCURRENCY":
        return "crypto"
    if quote_type in {"ETF", "MUTUALFUND"}:
        return "etf"
    if any(token in name for token in (" ETF", " FUND", " TRUST", " UCITS", " ETN")):
        return "etf"
    if quote_type in {"EQUITY", "ADR"}:
        return "equity"
    return "equity"


def score_growth(info: dict[str, Any], prior: float) -> float:
    revenue = linear_score(info.get("revenueGrowth"), -0.25, 0.35, 10, 95)
    earnings = linear_score(info.get("earningsGrowth"), -0.35, 0.45, 10, 95)
    total_revenue = num(info.get("totalRevenue"))
    if revenue is None and total_revenue is not None:
        revenue = 52.0 if total_revenue > 0 else 20.0
    return average([revenue, earnings], default=0.55 * prior + 22.5) or 50.0


def score_profitability(info: dict[str, Any], sector: str, prior: float) -> float:
    financial = "FINANCIAL" in sector.upper()
    if financial:
        return average(
            [
                linear_score(info.get("returnOnEquity"), -0.05, 0.22, 20, 90),
                linear_score(info.get("profitMargins"), -0.05, 0.35, 20, 90),
            ],
            default=0.55 * prior + 20,
        ) or 50.0
    fcf = num(info.get("freeCashflow"))
    operating_cf = num(info.get("operatingCashflow"))
    fcf_score = None
    if fcf is not None or operating_cf is not None:
        fcf_score = 75.0 if (fcf or operating_cf or 0) > 0 else 25.0
    return average(
        [
            linear_score(info.get("grossMargins"), 0.05, 0.75, 15, 90),
            linear_score(info.get("operatingMargins"), -0.20, 0.35, 10, 95),
            linear_score(info.get("profitMargins"), -0.25, 0.30, 10, 90),
            linear_score(info.get("returnOnEquity"), -0.15, 0.35, 10, 90),
            fcf_score,
        ],
        default=0.50 * prior + 20,
    ) or 50.0


def market_cap_score(market_cap: Any) -> float | None:
    cap = num(market_cap)
    if cap is None or cap <= 0:
        return None
    if cap >= 1_000_000_000_000:
        return 95.0
    if cap >= 250_000_000_000:
        return 88.0
    if cap >= 75_000_000_000:
        return 80.0
    if cap >= 20_000_000_000:
        return 70.0
    if cap >= 5_000_000_000:
        return 60.0
    if cap >= 1_000_000_000:
        return 50.0
    if cap >= 250_000_000:
        return 38.0
    return 24.0


def score_market_position(info: dict[str, Any], prior: float) -> float:
    cap_score = market_cap_score(info.get("marketCap"))
    gross_score = linear_score(info.get("grossMargins"), 0.15, 0.75, 35, 85)
    if prior >= 85:
        prior_moat = prior
    elif prior >= 70:
        prior_moat = prior - 2
    else:
        prior_moat = prior
    if cap_score is None and gross_score is None:
        return prior
    cap = cap_score if cap_score is not None else prior
    gross = gross_score if gross_score is not None else prior
    return clamp(cap * 0.45 + prior_moat * 0.35 + gross * 0.20)


def score_financial_health(info: dict[str, Any], sector: str, prior: float) -> float:
    if "FINANCIAL" in sector.upper():
        return average(
            [
                linear_score(info.get("returnOnEquity"), 0.00, 0.18, 35, 85),
                market_cap_score(info.get("marketCap")),
                linear_score(info.get("beta"), 2.0, 0.6, 25, 80),
            ],
            default=0.55 * prior + 18,
        ) or 50.0

    cash = num(info.get("totalCash"))
    debt = num(info.get("totalDebt"))
    fcf = num(info.get("freeCashflow"))
    cash_debt = None
    if cash is not None and debt is not None:
        if debt <= 0:
            cash_debt = 90.0
        else:
            cash_debt = linear_score(cash / debt, 0.0, 2.5, 20, 90)
            if fcf is not None and fcf > 0:
                cash_debt = max(cash_debt or 0.0, linear_score(fcf / debt, 0.0, 0.75, 35, 90) or 0.0)

    fcf_health = None
    if fcf is not None:
        if fcf > 10_000_000_000:
            fcf_health = 92.0
        elif fcf > 1_000_000_000:
            fcf_health = 84.0
        elif fcf > 0:
            fcf_health = 76.0
        else:
            fcf_health = 24.0

    debt_equity = num(info.get("debtToEquity"))
    debt_equity_score = None
    if debt_equity is not None:
        debt_equity_score = linear_score(debt_equity, 250.0, 10.0, 20, 85)

    return average(
        [
            cash_debt,
            debt_equity_score,
            linear_score(info.get("currentRatio"), 0.55, 3.0, 25, 85),
            fcf_health,
            market_cap_score(info.get("marketCap")),
        ],
        default=0.55 * prior + 18,
    ) or 50.0


def score_survivability(info: dict[str, Any], symbol: str, prior: float) -> float:
    years = history_years(symbol)
    years_score = None
    if years is not None:
        years_score = linear_score(years, 0.5, 12.0, 25, 90)
    risk_scores = [
        linear_score(info.get("overallRisk"), 10.0, 1.0, 20, 80),
        linear_score(info.get("auditRisk"), 10.0, 1.0, 20, 80),
        linear_score(info.get("boardRisk"), 10.0, 1.0, 20, 80),
    ]
    scale_record = average([years_score, market_cap_score(info.get("marketCap")), prior], default=prior) or prior
    governance = average(risk_scores, default=scale_record) or scale_record
    return clamp(scale_record * 0.85 + governance * 0.15)


def score_moat(info: dict[str, Any], market_position: float, prior: float) -> float:
    gross_score = linear_score(info.get("grossMargins"), 0.20, 0.80, 30, 90)
    margin_score = linear_score(info.get("operatingMargins"), 0.00, 0.35, 35, 90)
    qualitative_anchor = 0.72 * prior + 0.28 * market_position
    return average([qualitative_anchor, gross_score, margin_score], default=prior) or prior


def non_company_score(symbol: str, asset_type: str, info: dict[str, Any], prior: float | None) -> int:
    upper = symbol.upper()
    name = str(info.get("shortName") or info.get("longName") or "").upper()
    if prior is not None:
        base = float(prior)
    elif asset_type == "currency":
        pair = upper[:-2].replace("/", "").replace("-", "").replace("_", "")
        base = 62.0 if pair[:3] in {"USD", "EUR", "CHF", "JPY"} or pair[3:] in {"USD", "EUR", "CHF", "JPY"} else 52.0
    elif asset_type == "commodity":
        base = 55.0 if upper.startswith(("GC", "XAU")) else 48.0
    elif asset_type == "crypto":
        base = 42.0 if upper.startswith("BTC") else 34.0
    elif asset_type == "index":
        base = 70.0 if upper in {"^GSPC", "^NDX", "^IXIC"} else 60.0
    else:
        base = 58.0

    if asset_type == "etf":
        if upper in BROAD_ETFS:
            base = max(base, 68.0)
        if any(hint in name for hint in STRUCTURED_PRODUCT_HINTS):
            base = min(base, 55.0)
        if any(token in name for token in ("GOLD", "SILVER", "COMMODITY")):
            base = min(max(base, 48.0), 58.0)
    return int(round(clamp(base)))


def component_score(symbol: str, info: dict[str, Any], prior: int | None, labels: dict[str, str]) -> dict[str, Any]:
    asset_type = classify_asset(symbol, info)
    prior_score = float(prior if prior is not None else 50)
    name = info.get("shortName") or info.get("longName") or labels.get(symbol.upper()) or symbol

    if asset_type != "equity":
        score = non_company_score(symbol, asset_type, info, prior)
        return {
            "score": score,
            "asset_type": asset_type,
            "name": name,
            "sector": info.get("sector"),
            "components": {},
            "confidence": 0.65 if prior is not None else 0.45,
            "prior": prior,
            "rationale": f"{asset_type} instrument scored from structure/liquidity rather than operating-company fundamentals.",
        }

    if not info and prior is None:
        profile = price_profile(symbol)
        stale_days = num(profile.get("stale_days"))
        last_close = num(profile.get("last_close"))
        last_volume = num(profile.get("last_volume"))
        if last_close is not None and last_close < 1.0:
            fallback_score = 22
            reason = "sub-dollar cached price and no current fundamentals"
        elif stale_days is not None and stale_days > 120:
            fallback_score = 38
            reason = "stale/delisted profile with no current fundamentals"
        elif last_volume == 0:
            fallback_score = 35
            reason = "zero-volume latest bar and no current fundamentals"
        else:
            fallback_score = 45
            reason = "no current fundamentals; conservative local-price fallback"
        return {
            "score": fallback_score,
            "asset_type": asset_type,
            "name": name,
            "sector": None,
            "industry": None,
            "components": {},
            "confidence": 0.25,
            "prior": prior,
            "price_profile": profile,
            "rationale": f"Fallback score: {reason}.",
        }

    sector = str(info.get("sector") or "")
    growth = score_growth(info, prior_score)
    profitability = score_profitability(info, sector, prior_score)
    market_position = score_market_position(info, prior_score)
    financial_health = score_financial_health(info, sector, prior_score)
    survivability = score_survivability(info, symbol, prior_score)
    moat = score_moat(info, market_position, prior_score)

    computed = (
        growth * 0.20
        + profitability * 0.20
        + market_position * 0.20
        + financial_health * 0.15
        + survivability * 0.15
        + moat * 0.10
    )

    observed = sum(1 for field in SNAPSHOT_FIELDS if info.get(field) not in (None, ""))
    confidence = clamp(observed / 16.0, 0.35, 0.95)
    if prior is not None:
        final = computed * 0.70 + prior_score * 0.30
        if prior_score >= 80 and profitability >= 58 and market_position >= 68 and financial_health >= 45:
            final = max(final, prior_score - 6)
        elif prior_score >= 70 and profitability >= 55 and market_position >= 60 and financial_health >= 45:
            final = max(final, prior_score - 5)
    else:
        final = computed * confidence + 50.0 * (1.0 - confidence)

    score = int(round(clamp(final)))
    drivers = []
    if growth >= 70:
        drivers.append("growth")
    if profitability >= 70:
        drivers.append("profitability")
    if market_position >= 75:
        drivers.append("scale/moat")
    if financial_health < 45:
        drivers.append("balance-sheet risk")
    if survivability < 45:
        drivers.append("short/fragile operating record")
    if not drivers:
        drivers.append("mixed fundamentals")

    return {
        "score": score,
        "asset_type": asset_type,
        "name": name,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "components": {
            "revenue_growth": round(growth, 1),
            "profitability": round(profitability, 1),
            "market_position": round(market_position, 1),
            "financial_health": round(financial_health, 1),
            "survivability": round(survivability, 1),
            "innovation_moat": round(moat, 1),
        },
        "confidence": round(confidence, 2),
        "prior": prior,
        "rationale": f"Operating-company score driven by {', '.join(drivers)}.",
    }


def build_universe() -> tuple[list[str], dict[str, int], dict[str, list[str]], dict[str, str]]:
    scores = load_literal_assignment(QUALITY_SERVICE, "QUALITY_SCORES", {})
    configured = load_literal_assignment(UNIVERSE_FILE, "DEFAULT_ASSET_UNIVERSE", [])
    mapping = load_literal_assignment(UNIVERSE_FILE, "MAPPING", {})
    labels = load_symbol_comments()

    price_symbols = []
    if PRICE_DIR.exists():
        price_symbols = [normalize_cached_stem(path.stem) for path in PRICE_DIR.glob("*.csv")]

    universe: list[str] = []
    seen: set[str] = set()
    for source in (configured, price_symbols, scores.keys()):
        for raw in source:
            symbol = str(raw).strip().upper()
            if not symbol or symbol in seen:
                continue
            universe.append(symbol)
            seen.add(symbol)

    return sorted(universe), {str(k).upper(): int(v) for k, v in scores.items()}, mapping, labels


def fetch_info(symbol: str, mapping: dict[str, list[str]], retries: int = 2) -> tuple[str, dict[str, Any], str | None]:
    try:
        import yfinance as yf
    except ImportError as exc:
        return symbol, {}, f"yfinance unavailable: {exc}"

    candidates: list[str] = []
    for candidate in [symbol, *mapping.get(symbol, [])]:
        for variant in symbol_variants(candidate):
            if variant not in candidates:
                candidates.append(variant)

    last_error: str | None = None
    for candidate in candidates[:8]:
        for attempt in range(retries):
            try:
                info = yf.Ticker(candidate).get_info()
                if isinstance(info, dict) and (info.get("quoteType") or info.get("shortName") or info.get("marketCap")):
                    snapshot = {field: safe_json(info.get(field)) for field in SNAPSHOT_FIELDS if field in info}
                    return candidate, snapshot, None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:140]}"
                if attempt + 1 < retries:
                    time.sleep(0.35)
    return symbol, {}, last_error or "no usable Yahoo Finance info"


def refresh(limit: int | None, workers: int, force: bool) -> dict[str, Any]:
    universe, priors, mapping, labels = build_universe()
    if limit:
        universe = universe[:limit]

    existing_details: dict[str, Any] = {}
    if OUTPUT_PATH.exists() and not force:
        try:
            existing_payload = json.loads(OUTPUT_PATH.read_text())
            existing_details = existing_payload.get("details", {})
        except (OSError, json.JSONDecodeError):
            existing_details = {}

    details: dict[str, Any] = {}
    raw_info: dict[str, dict[str, Any]] = {}
    fetch_symbols: dict[str, str] = {}
    errors: dict[str, str] = {}

    to_fetch = []
    for symbol in universe:
        cached = existing_details.get(symbol)
        if cached and cached.get("source_snapshot") and not force:
            raw_info[symbol] = cached["source_snapshot"]
            fetch_symbols[symbol] = cached.get("fetch_symbol", symbol)
        else:
            to_fetch.append(symbol)

    completed = 0
    total = len(to_fetch)
    if total:
        print(f"Fetching Yahoo Finance fundamentals for {total} symbols with {workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fetch_info, symbol, mapping): symbol for symbol in to_fetch}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            completed += 1
            try:
                fetch_symbol, info, error = future.result()
            except Exception as exc:
                fetch_symbol, info, error = symbol, {}, f"{type(exc).__name__}: {str(exc)[:140]}"
            raw_info[symbol] = info
            if error:
                errors[symbol] = error
            scored = component_score(symbol, info, priors.get(symbol), labels)
            scored["fetch_symbol"] = fetch_symbol
            scored["source_snapshot"] = info
            details[symbol] = scored
            fetch_symbols[symbol] = fetch_symbol
            if completed % 20 == 0 or completed == total:
                print(f"  {completed:>4}/{total:<4} fetched; latest {symbol}: {scored['score']}")

    for symbol in universe:
        if symbol in details:
            continue
        info = raw_info.get(symbol, {})
        scored = component_score(symbol, info, priors.get(symbol), labels)
        scored["fetch_symbol"] = fetch_symbols.get(symbol, symbol)
        scored["source_snapshot"] = info
        details[symbol] = scored

    scores = {symbol: int(details[symbol]["score"]) for symbol in sorted(details)}
    changed = {}
    for symbol, score in scores.items():
        prior = priors.get(symbol)
        if prior is not None and prior != score:
            changed[symbol] = {"from": prior, "to": score, "delta": score - prior}

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance fundamentals via yfinance plus local price-history survivability and existing qualitative prior for moat/market-position context.",
        "method_version": "business-quality-v2.0",
        "formula": {
            "revenue_growth": 0.20,
            "profitability": 0.20,
            "market_position": 0.20,
            "financial_health": 0.15,
            "survivability": 0.15,
            "innovation_moat": 0.10,
        },
        "coverage": {
            "symbols_scored": len(scores),
            "configured_or_cached_universe": len(universe),
            "prior_scores": len(priors),
            "changed_prior_scores": len(changed),
            "fetch_errors": len(errors),
        },
        "scores": scores,
        "details": {symbol: details[symbol] for symbol in sorted(details)},
        "changed": changed,
        "errors": errors,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT_PATH}")
    print(json.dumps(payload["coverage"], indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh generated business quality score overlay.")
    parser.add_argument("--limit", type=int, default=None, help="Limit symbols for a smoke run.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent Yahoo Finance requests.")
    parser.add_argument("--force", action="store_true", help="Ignore existing generated snapshots.")
    args = parser.parse_args()

    refresh(limit=args.limit, workers=args.workers, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
