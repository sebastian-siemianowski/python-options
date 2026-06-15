"""Post-disclosure success ranking for politician trade filers."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any


BUY_TRANSACTION_TYPES = {"purchase", "received"}
SELL_TRANSACTION_TYPES = {"sale", "sale_partial"}
SCORABLE_TRANSACTION_TYPES = BUY_TRANSACTION_TYPES | SELL_TRANSACTION_TYPES
DEFAULT_REQUIRED_ALIASES = ("nancy pelosi",)


@dataclass
class PricePoint:
    day: date
    close: float


@dataclass
class PriceSeries:
    symbol: str
    points: list[PricePoint]

    @property
    def latest_date(self) -> date | None:
        return self.points[-1].day if self.points else None

    def first_on_or_after(self, target: date, *, as_of: date) -> PricePoint | None:
        for point in self.points:
            if target <= point.day <= as_of:
                return point
        return None

    def last_on_or_before(self, target: date, *, after: date, as_of: date) -> PricePoint | None:
        bounded_target = min(target, as_of)
        candidate = None
        for point in self.points:
            if point.day <= after:
                continue
            if point.day > bounded_target:
                break
            candidate = point
        return candidate


@dataclass
class TradeEvaluation:
    filer_key: str
    filer_name: str
    ticker: str
    transaction_type: str
    disclosure_date: str
    entry_date: str
    exit_date: str
    holding_days: int
    raw_return: float
    signed_return: float
    amount_mid_usd: float


@dataclass
class TraderBucket:
    filer_key: str
    filer_name: str
    chamber: str | None = None
    party: str | None = None
    state: str | None = None
    total_trades: int = 0
    scored_trades: int = 0
    wins: int = 0
    amount_mid_usd: float = 0.0
    weighted_return_sum: float = 0.0
    weight_sum: float = 0.0
    returns: list[float] = field(default_factory=list)
    tickers: set[str] = field(default_factory=set)
    included_by_requested_profile: bool = False
    overall_rank: int | None = None
    filter_rank: int | None = None

    def add_total(self, row: dict[str, Any]) -> None:
        self.total_trades += 1
        self.chamber = self.chamber or _clean_optional(row.get("chamber"))
        self.party = self.party or _clean_optional(row.get("party"))
        self.state = self.state or _clean_optional(row.get("state"))

    def add_evaluation(self, evaluation: TradeEvaluation) -> None:
        self.scored_trades += 1
        self.amount_mid_usd += evaluation.amount_mid_usd
        weight = math.sqrt(max(evaluation.amount_mid_usd, 1.0))
        self.weighted_return_sum += evaluation.signed_return * weight
        self.weight_sum += weight
        self.returns.append(evaluation.signed_return)
        self.tickers.add(evaluation.ticker)
        if evaluation.signed_return > 0:
            self.wins += 1

    @property
    def coverage(self) -> float:
        if not self.total_trades:
            return 0.0
        return round(self.scored_trades / self.total_trades, 4)

    @property
    def average_signed_return(self) -> float:
        if not self.weight_sum:
            return 0.0
        return self.weighted_return_sum / self.weight_sum

    @property
    def median_signed_return(self) -> float:
        if not self.returns:
            return 0.0
        ordered = sorted(self.returns)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2

    @property
    def win_rate(self) -> float:
        if not self.scored_trades:
            return 0.0
        return self.wins / self.scored_trades

    @property
    def success_score(self) -> float:
        reliability = min(1.0, math.sqrt(self.scored_trades / 5))
        return (self.average_signed_return * 100.0) * reliability

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.filter_rank,
            "overall_rank": self.overall_rank,
            "filer_key": self.filer_key,
            "filer_name": self.filer_name,
            "chamber": self.chamber,
            "party": self.party,
            "state": self.state,
            "total_trades": self.total_trades,
            "scored_trades": self.scored_trades,
            "coverage": self.coverage,
            "win_rate": round(self.win_rate, 4),
            "average_signed_return_pct": round(self.average_signed_return * 100.0, 2),
            "median_signed_return_pct": round(self.median_signed_return * 100.0, 2),
            "success_score": round(self.success_score, 2),
            "amount_mid_usd": round(self.amount_mid_usd, 2),
            "top_tickers": sorted(self.tickers)[:8],
            "included_by_requested_profile": self.included_by_requested_profile,
        }


def build_successful_trader_leaderboard(
    rows: list[dict[str, Any]],
    *,
    prices_dir: str | Path,
    limit: int = 10,
    horizon_days: int = 90,
    min_holding_days: int = 5,
    as_of_date: str | None = None,
    required_aliases: tuple[str, ...] = DEFAULT_REQUIRED_ALIASES,
) -> dict[str, Any]:
    """Rank filers by post-disclosure signed returns using local price history."""
    as_of = _parse_iso_date(as_of_date) or date.today()
    prices_root = Path(prices_dir)
    cache: dict[str, PriceSeries | None] = {}
    buckets: dict[str, TraderBucket] = {}
    scored_trade_count = 0
    unscored_trade_count = 0

    for row in rows:
        filer_name = str(row.get("filer_name") or "").strip()
        if not filer_name:
            continue
        filer_key = canonical_filer_key(filer_name)
        bucket = buckets.setdefault(filer_key, TraderBucket(filer_key=filer_key, filer_name=filer_name))
        bucket.add_total(row)
        evaluation = evaluate_trade_success(
            row,
            prices_dir=prices_root,
            price_cache=cache,
            horizon_days=horizon_days,
            min_holding_days=min_holding_days,
            as_of=as_of,
        )
        if evaluation is None:
            unscored_trade_count += 1
            continue
        bucket.add_evaluation(evaluation)
        scored_trade_count += 1

    ranked = sorted(
        (bucket for bucket in buckets.values() if bucket.scored_trades > 0),
        key=lambda bucket: (
            bucket.success_score,
            bucket.average_signed_return,
            bucket.win_rate,
            bucket.scored_trades,
            bucket.amount_mid_usd,
            bucket.filer_name,
        ),
        reverse=True,
    )
    for idx, bucket in enumerate(ranked, start=1):
        bucket.overall_rank = idx

    selected = _select_leaderboard(ranked, limit=limit, required_aliases=required_aliases)
    for idx, bucket in enumerate(selected, start=1):
        bucket.filter_rank = idx

    return {
        "methodology": {
            "label": f"{horizon_days}D post-disclosure signed return",
            "description": (
                "Ranks parsed congressional filers by signed market movement after the public disclosure date. "
                "Purchases score when the asset rises; sales score when the asset falls. "
                "Rows without a mapped public ticker or enough price history are excluded from scoring, not hidden from the feed."
            ),
            "horizon_days": horizon_days,
            "min_holding_days": min_holding_days,
            "as_of_date": as_of.isoformat(),
            "required_profiles": list(required_aliases),
        },
        "limit": limit,
        "scored_trade_count": scored_trade_count,
        "unscored_trade_count": unscored_trade_count,
        "eligible_trader_count": len(ranked),
        "leaderboard": [bucket.to_dict() for bucket in selected],
    }


def evaluate_trade_success(
    row: dict[str, Any],
    *,
    prices_dir: Path,
    price_cache: dict[str, PriceSeries | None],
    horizon_days: int,
    min_holding_days: int,
    as_of: date,
) -> TradeEvaluation | None:
    ticker = str(row.get("ticker") or "").strip().upper()
    transaction_type = str(row.get("transaction_type") or "").strip().lower()
    disclosure = _parse_iso_date(row.get("disclosure_date"))
    if not ticker or transaction_type not in SCORABLE_TRANSACTION_TYPES or disclosure is None:
        return None
    if disclosure > as_of:
        return None
    series = price_cache.get(ticker)
    if ticker not in price_cache:
        series = load_price_series(ticker, prices_dir=prices_dir)
        price_cache[ticker] = series
    if series is None or not series.points:
        return None
    entry = series.first_on_or_after(disclosure, as_of=as_of)
    if entry is None:
        return None
    target = disclosure + timedelta(days=max(1, horizon_days))
    exit_point = series.last_on_or_before(target, after=entry.day, as_of=as_of)
    if exit_point is None:
        return None
    holding_days = (exit_point.day - entry.day).days
    if holding_days < min_holding_days or entry.close <= 0:
        return None
    raw_return = (exit_point.close - entry.close) / entry.close
    direction = -1.0 if transaction_type in SELL_TRANSACTION_TYPES else 1.0
    amount_mid = _number_or_zero(row.get("amount_mid_usd")) or 1.0
    return TradeEvaluation(
        filer_key=canonical_filer_key(row.get("filer_name")),
        filer_name=str(row.get("filer_name") or ""),
        ticker=ticker,
        transaction_type=transaction_type,
        disclosure_date=disclosure.isoformat(),
        entry_date=entry.day.isoformat(),
        exit_date=exit_point.day.isoformat(),
        holding_days=holding_days,
        raw_return=raw_return,
        signed_return=raw_return * direction,
        amount_mid_usd=amount_mid,
    )


def load_price_series(symbol: str, *, prices_dir: str | Path) -> PriceSeries | None:
    root = Path(prices_dir)
    candidates = (root / f"{symbol}.csv", root / f"{symbol}_1d.csv")
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return None
    points: list[PricePoint] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                day = _parse_iso_date(row.get("Date"))
                close = _number_or_zero(row.get("Adj Close")) or _number_or_zero(row.get("Close"))
                if day is None or close <= 0:
                    continue
                points.append(PricePoint(day=day, close=close))
    except Exception:
        return None
    points.sort(key=lambda point: point.day)
    return PriceSeries(symbol=symbol, points=points)


def canonical_filer_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "," in text:
        parts = [part.strip() for part in text.split(",", 1)]
        if all(parts):
            text = f"{parts[1]} {parts[0]}"
    text = re.sub(r"\b(hon|honorable|rep|representative|sen|senator|mr|mrs|ms|dr)\b\.?", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def selected_filer_keys(leaderboard: dict[str, Any]) -> set[str]:
    return {
        str(row.get("filer_key") or "")
        for row in leaderboard.get("leaderboard", [])
        if row.get("filer_key")
    }


def _select_leaderboard(
    ranked: list[TraderBucket],
    *,
    limit: int,
    required_aliases: tuple[str, ...],
) -> list[TraderBucket]:
    safe_limit = max(1, limit)
    selected = list(ranked[:safe_limit])
    selected_keys = {bucket.filer_key for bucket in selected}
    required_keys = {canonical_filer_key(alias) for alias in required_aliases}
    for required_key in required_keys:
        if required_key in selected_keys:
            continue
        candidate = next((bucket for bucket in ranked if bucket.filer_key == required_key), None)
        if candidate is None:
            continue
        candidate.included_by_requested_profile = True
        if len(selected) < safe_limit:
            selected.append(candidate)
        else:
            selected[-1] = candidate
        selected_keys = {bucket.filer_key for bucket in selected}
    return selected[:safe_limit]


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _number_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _clean_optional(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
