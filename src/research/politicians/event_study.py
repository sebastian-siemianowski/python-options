"""Disclosure-date event windows for politician trade research."""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path
from typing import Any

from ingestion.politicians.quality import confidence_bucket


DEFAULT_PRIOR_TRADING_DAYS = 5
DEFAULT_FORWARD_TRADING_DAYS = (1, 7, 30, 90)
ENRICHMENT_SOURCE_LABEL = "enrichment_not_source_disclosure"
RETROSPECTIVE_ONLY_NOTE = (
    "Transaction-date analysis is RETROSPECTIVE_ONLY. Production research "
    "windows must use disclosure_date to avoid lookahead leakage."
)


def build_disclosure_event_window(
    trade: dict[str, Any],
    *,
    prices_dir: str | Path,
    prior_days: int = DEFAULT_PRIOR_TRADING_DAYS,
    forward_days: tuple[int, ...] = DEFAULT_FORWARD_TRADING_DAYS,
) -> dict[str, Any]:
    """Return an event window keyed by disclosure_date, not transaction_date."""
    return _build_event_window(
        trade,
        prices_dir=prices_dir,
        anchor_field="disclosure_date",
        prior_days=prior_days,
        forward_days=forward_days,
    )


def build_transaction_retrospective_event_window(
    trade: dict[str, Any],
    *,
    prices_dir: str | Path,
    prior_days: int = DEFAULT_PRIOR_TRADING_DAYS,
    forward_days: tuple[int, ...] = DEFAULT_FORWARD_TRADING_DAYS,
) -> dict[str, Any]:
    """Return a transaction-date event window for RETROSPECTIVE_ONLY research."""
    window = _build_event_window(
        trade,
        prices_dir=prices_dir,
        anchor_field="transaction_date",
        prior_days=prior_days,
        forward_days=forward_days,
    )
    window["report_label"] = "RETROSPECTIVE_ONLY"
    window["production_usage"] = "forbidden_for_signal_generation"
    return window


def _build_event_window(
    trade: dict[str, Any],
    *,
    prices_dir: str | Path,
    anchor_field: str,
    prior_days: int,
    forward_days: tuple[int, ...],
) -> dict[str, Any]:
    symbol = str(trade.get("ticker") or "").upper()
    anchor_date = trade.get(anchor_field)
    warnings: list[str] = []
    if not symbol:
        warnings.append("missing_ticker")
    if not anchor_date:
        warnings.append(f"missing_{anchor_field}")
    if warnings:
        return _warning_result(trade, warnings, event_time_anchor=anchor_field)

    prices_path = Path(prices_dir) / f"{symbol}_1d.csv"
    if not prices_path.exists():
        return _warning_result(trade, [f"missing_price_data:{symbol}"], event_time_anchor=anchor_field)

    prices = _load_prices(prices_path)
    if not prices:
        return _warning_result(trade, [f"empty_price_data:{symbol}"], event_time_anchor=anchor_field)

    anchor_idx = _first_index_on_or_after(prices, str(anchor_date))
    if anchor_idx is None:
        return _warning_result(trade, [f"{anchor_field}_after_price_history:{anchor_date}"], event_time_anchor=anchor_field)

    anchor = prices[anchor_idx]
    prior = prices[max(0, anchor_idx - prior_days):anchor_idx]
    forward: dict[str, dict[str, Any] | None] = {}
    for days in forward_days:
        idx = anchor_idx + days
        if idx < len(prices):
            target = prices[idx]
            forward[str(days)] = {
                "date": target["date"],
                "close": target["close"],
                "return_pct": ((target["close"] / anchor["close"]) - 1.0) * 100 if anchor["close"] else None,
            }
        else:
            forward[str(days)] = None
            warnings.append(f"missing_forward_{days}d")

    return {
        "status": "ok" if not warnings else "valid_with_warnings",
        "ticker": symbol,
        "event_time_anchor": anchor_field,
        "disclosure_date": trade.get("disclosure_date"),
        "transaction_date": trade.get("transaction_date"),
        "retrospective_only_note": RETROSPECTIVE_ONLY_NOTE,
        "anchor": anchor,
        "prior": prior,
        "forward": forward,
        "warnings": warnings,
    }


def compute_disclosure_event_study(
    trades: list[dict[str, Any]],
    *,
    prices_dir: str | Path,
    forward_days: tuple[int, ...] = DEFAULT_FORWARD_TRADING_DAYS,
    min_sample_size: int = 5,
) -> dict[str, Any]:
    """Aggregate disclosure-date event returns by disclosure and trade attributes."""
    groups: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    included_count = 0

    for trade in trades:
        window = build_disclosure_event_window(
            trade,
            prices_dir=prices_dir,
            forward_days=forward_days,
        )
        if window["status"] == "warning" or any(window["forward"].get(str(day)) is None for day in forward_days):
            excluded.append({
                "trade_id": trade.get("trade_id"),
                "ticker": trade.get("ticker"),
                "warnings": window.get("warnings", []),
            })
            continue
        included_count += 1
        key = _group_key(trade)
        group = groups.setdefault(key, {
            "group": {
                "transaction_type": key[0],
                "chamber": key[1],
                "asset_type": key[2],
                "amount_bucket": key[3],
                "parser_confidence_bucket": key[4],
            },
            "sample_count": 0,
            "horizons": {str(day): [] for day in forward_days},
        })
        group["sample_count"] += 1
        for day in forward_days:
            forward = window["forward"][str(day)]
            group["horizons"][str(day)].append(float(forward["return_pct"]))

    finalized_groups = []
    warnings = []
    for group in groups.values():
        horizon_stats = {
            horizon: _return_stats(values)
            for horizon, values in group["horizons"].items()
        }
        finalized = {
            "group": group["group"],
            "sample_count": group["sample_count"],
            "horizons": horizon_stats,
        }
        if group["sample_count"] < min_sample_size:
            warning = {
                "type": "sample_size_too_small",
                "sample_count": group["sample_count"],
                "min_sample_size": min_sample_size,
                "group": group["group"],
            }
            finalized.setdefault("warnings", []).append(warning)
            warnings.append(warning)
        finalized_groups.append(finalized)

    finalized_groups.sort(key=lambda item: (
        item["group"]["transaction_type"],
        item["group"]["chamber"],
        item["group"]["asset_type"],
        item["group"]["amount_bucket"],
        item["group"]["parser_confidence_bucket"],
    ))
    return {
        "event_time_anchor": "disclosure_date",
        "forward_trading_days": list(forward_days),
        "input_count": len(trades),
        "included_count": included_count,
        "excluded_count": len(excluded),
        "excluded_events": excluded,
        "groups": finalized_groups,
        "warnings": warnings,
    }


def compute_transaction_date_retrospective_analysis(
    trades: list[dict[str, Any]],
    *,
    prices_dir: str | Path,
    forward_days: tuple[int, ...] = DEFAULT_FORWARD_TRADING_DAYS,
    min_sample_size: int = 5,
) -> dict[str, Any]:
    """
    Compare transaction-date returns with disclosure-date returns.

    This report is deliberately labelled RETROSPECTIVE_ONLY because
    transaction_date was not knowable by the system until disclosure.
    """
    transaction_returns: dict[str, list[float]] = {str(day): [] for day in forward_days}
    disclosure_returns: dict[str, list[float]] = {str(day): [] for day in forward_days}
    delay_groups: dict[tuple[str, str], list[int]] = {}
    excluded: list[dict[str, Any]] = []
    paired_count = 0

    for trade in trades:
        delay = _filing_delay_days(trade)
        if delay is not None:
            delay_key = (
                str(trade.get("chamber") or "unknown").lower(),
                str(trade.get("transaction_type") or "unknown").lower(),
            )
            delay_groups.setdefault(delay_key, []).append(delay)

        transaction_window = build_transaction_retrospective_event_window(
            trade,
            prices_dir=prices_dir,
            forward_days=forward_days,
        )
        disclosure_window = build_disclosure_event_window(
            trade,
            prices_dir=prices_dir,
            forward_days=forward_days,
        )
        warnings = _window_exclusion_warnings("transaction_date", transaction_window, forward_days)
        warnings.extend(_window_exclusion_warnings("disclosure_date", disclosure_window, forward_days))
        direction = _transaction_direction(trade.get("transaction_type"))
        if direction == 0.0:
            warnings.append("unsupported_direction_for_edge_analysis")
        if warnings:
            excluded.append({
                "trade_id": trade.get("trade_id"),
                "ticker": trade.get("ticker"),
                "warnings": warnings,
            })
            continue

        paired_count += 1
        for day in forward_days:
            key = str(day)
            transaction_forward = transaction_window["forward"][key]
            disclosure_forward = disclosure_window["forward"][key]
            transaction_returns[key].append(direction * float(transaction_forward["return_pct"]))
            disclosure_returns[key].append(direction * float(disclosure_forward["return_pct"]))

    delay_summary = [
        {
            "chamber": key[0],
            "transaction_type": key[1],
            "sample_count": len(values),
            "median_delay_days": _median(values),
        }
        for key, values in delay_groups.items()
    ]
    delay_summary.sort(key=lambda item: (item["chamber"], item["transaction_type"]))
    edge_disappearance = {
        key: _edge_disappearance_stats(transaction_returns[key], disclosure_returns[key])
        for key in transaction_returns
    }
    warnings = []
    if paired_count < min_sample_size:
        warnings.append({
            "type": "sample_size_too_small",
            "sample_count": paired_count,
            "min_sample_size": min_sample_size,
        })

    return {
        "report_label": "RETROSPECTIVE_ONLY",
        "event_time_anchor": "transaction_date",
        "comparison_anchor": "disclosure_date",
        "retrospective_only_note": RETROSPECTIVE_ONLY_NOTE,
        "production_usage": "forbidden_for_signal_generation",
        "forward_trading_days": list(forward_days),
        "input_count": len(trades),
        "paired_event_count": paired_count,
        "excluded_count": len(excluded),
        "excluded_events": excluded,
        "median_filing_delay_days": delay_summary,
        "edge_disappearance": edge_disappearance,
        "warnings": warnings,
    }


def compute_committee_sector_clustering_report(
    trades: list[dict[str, Any]],
    *,
    members: dict[str, Any] | list[dict[str, Any]] | None = None,
    sector_lookup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a committee-sector clustering heatmap from enrichment metadata."""
    member_lookup = _build_member_lookup(members)
    heatmap: dict[tuple[str, str], dict[str, Any]] = {}
    source_counts = {
        "committee": {},
        "sector": {},
    }
    unknown_committee_count = 0
    unknown_sector_count = 0

    for trade in trades:
        committees, committee_source = _resolve_committees(trade, member_lookup)
        sector, sector_source = _resolve_sector(trade, sector_lookup)
        _increment_source(source_counts["committee"], committee_source)
        _increment_source(source_counts["sector"], sector_source)
        if committees == ["unknown"]:
            unknown_committee_count += 1
        if sector == "unknown":
            unknown_sector_count += 1
        amount_mid = _amount_mid_estimate(trade)
        amount_min = _number_or_zero(trade.get("amount_min_usd"))
        amount_max = _number_or_zero(trade.get("amount_max_usd"))
        for committee in committees:
            key = (committee, sector)
            bucket = heatmap.setdefault(key, {
                "committee": committee,
                "sector": sector,
                "trade_count": 0,
                "amount_mid_usd": 0.0,
                "amount_min_usd": 0.0,
                "amount_max_usd": 0.0,
                "committee_data_source": committee_source,
                "sector_data_source": sector_source,
                "data_classification": ENRICHMENT_SOURCE_LABEL,
            })
            bucket["trade_count"] += 1
            bucket["amount_mid_usd"] += amount_mid
            bucket["amount_min_usd"] += amount_min
            bucket["amount_max_usd"] += amount_max
            bucket["committee_data_source"] = _merge_source(bucket["committee_data_source"], committee_source)
            bucket["sector_data_source"] = _merge_source(bucket["sector_data_source"], sector_source)

    rows = sorted(
        heatmap.values(),
        key=lambda item: (item["trade_count"], item["amount_mid_usd"], item["committee"], item["sector"]),
        reverse=True,
    )
    return {
        "report_type": "committee_sector_clustering",
        "data_classification": ENRICHMENT_SOURCE_LABEL,
        "committee_data_label": "Committee enrichment from official member metadata when available; not a disclosure source field.",
        "sector_data_label": "Sector enrichment from existing signal-engine sector mappings where available.",
        "input_count": len(trades),
        "heatmap": rows,
        "unknown_committee_count": unknown_committee_count,
        "unknown_sector_count": unknown_sector_count,
        "source_counts": source_counts,
        "amount_estimate_label": "Estimated midpoint from public disclosure amount bucket",
    }


def _warning_result(
    trade: dict[str, Any],
    warnings: list[str],
    *,
    event_time_anchor: str = "disclosure_date",
) -> dict[str, Any]:
    return {
        "status": "warning",
        "ticker": trade.get("ticker"),
        "event_time_anchor": event_time_anchor,
        "disclosure_date": trade.get("disclosure_date"),
        "transaction_date": trade.get("transaction_date"),
        "retrospective_only_note": RETROSPECTIVE_ONLY_NOTE,
        "anchor": None,
        "prior": [],
        "forward": {},
        "warnings": warnings,
    }


def _build_member_lookup(members: dict[str, Any] | list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not members:
        return lookup
    entries: list[dict[str, Any]]
    if isinstance(members, dict):
        raw_entries = members.get("members") if isinstance(members.get("members"), list) else None
        if raw_entries is not None:
            entries = [entry for entry in raw_entries if isinstance(entry, dict)]
        else:
            entries = []
            for key, value in members.items():
                if isinstance(value, dict):
                    entries.append({"filer_id": key, **value})
    else:
        entries = [entry for entry in members if isinstance(entry, dict)]

    for entry in entries:
        for candidate in (
            entry.get("filer_id"),
            entry.get("member_id"),
            entry.get("bioguide_id"),
            entry.get("filer_name"),
            entry.get("name"),
        ):
            normalized = _normalize_lookup_key(candidate)
            if normalized:
                lookup[normalized] = entry
    return lookup


def _resolve_committees(
    trade: dict[str, Any],
    member_lookup: dict[str, dict[str, Any]],
) -> tuple[list[str], str]:
    row_committees = _extract_committees(trade)
    if row_committees:
        return row_committees, "trade_row_enrichment"
    for candidate in (trade.get("filer_id"), trade.get("filer_name")):
        metadata = member_lookup.get(_normalize_lookup_key(candidate))
        if not metadata:
            continue
        committees = _extract_committees(metadata)
        if committees:
            return committees, "official_member_metadata"
    return ["unknown"], "unknown"


def _extract_committees(row: dict[str, Any]) -> list[str]:
    committees: set[str] = set()
    raw = row.get("committees")
    if isinstance(raw, list):
        committees.update(str(item).strip() for item in raw if str(item).strip())
    elif isinstance(raw, str) and raw.strip():
        committees.add(raw.strip())
    committee = row.get("committee")
    if committee:
        committees.add(str(committee).strip())
    return sorted(committees)


def _resolve_sector(
    trade: dict[str, Any],
    sector_lookup: dict[str, Any] | None,
) -> tuple[str, str]:
    for field in ("sector", "asset_sector"):
        if trade.get(field):
            return str(trade[field]).strip() or "unknown", "trade_row_enrichment"
    ticker = str(trade.get("ticker") or "").upper().strip()
    if not ticker:
        return "unknown", "unknown"
    mapped = _sector_from_lookup(ticker, sector_lookup)
    if mapped:
        return mapped, "existing_sector_mapping"
    mapped = _sector_from_existing_mapping(ticker)
    if mapped:
        return mapped, "existing_sector_mapping"
    return "unknown", "unknown"


def _sector_from_lookup(ticker: str, sector_lookup: dict[str, Any] | None) -> str | None:
    if not sector_lookup:
        return None
    direct = sector_lookup.get(ticker)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for sector, symbols in sector_lookup.items():
        if isinstance(symbols, (list, set, tuple)) and ticker in {str(symbol).upper() for symbol in symbols}:
            return str(sector)
    return None


def _sector_from_existing_mapping(ticker: str) -> str | None:
    try:
        from ingestion.data_utils import get_sector
    except Exception:
        return None
    sector = get_sector(ticker)
    if not sector or sector == "Unspecified":
        return None
    return sector


def _normalize_lookup_key(value: Any) -> str:
    return str(value or "").strip().lower()


def _amount_mid_estimate(trade: dict[str, Any]) -> float:
    if trade.get("amount_mid_usd") not in (None, ""):
        return _number_or_zero(trade.get("amount_mid_usd"))
    minimum = trade.get("amount_min_usd")
    maximum = trade.get("amount_max_usd")
    if minimum not in (None, "") and maximum not in (None, ""):
        return (_number_or_zero(minimum) + _number_or_zero(maximum)) / 2.0
    return _number_or_zero(minimum)


def _number_or_zero(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _increment_source(counts: dict[str, int], source: str) -> None:
    counts[source] = counts.get(source, 0) + 1


def _merge_source(current: str, incoming: str) -> str:
    if current == incoming:
        return current
    if current == "mixed" or incoming == "mixed":
        return "mixed"
    return "mixed"


def _group_key(trade: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(trade.get("transaction_type") or "unknown").lower(),
        str(trade.get("chamber") or "unknown").lower(),
        str(trade.get("asset_type") or "unknown").lower(),
        _amount_bucket(trade),
        confidence_bucket(trade.get("parser_confidence")),
    )


def _amount_bucket(trade: dict[str, Any]) -> str:
    if trade.get("amount_bucket"):
        return str(trade["amount_bucket"])
    minimum = trade.get("amount_min_usd")
    maximum = trade.get("amount_max_usd")
    if minimum is not None and maximum is not None:
        return f"{int(float(minimum))}-{int(float(maximum))}"
    if trade.get("amount_mid_usd") is not None:
        return "midpoint_only"
    return "unknown"


def _return_stats(values: list[float]) -> dict[str, Any]:
    sample_count = len(values)
    if sample_count == 0:
        return {"sample_count": 0, "mean_return_pct": None, "ci95_low": None, "ci95_high": None}
    mean = sum(values) / sample_count
    if sample_count == 1:
        return {
            "sample_count": sample_count,
            "mean_return_pct": round(mean, 6),
            "ci95_low": round(mean, 6),
            "ci95_high": round(mean, 6),
        }
    variance = sum((value - mean) ** 2 for value in values) / (sample_count - 1)
    stderr = math.sqrt(variance) / math.sqrt(sample_count)
    margin = 1.96 * stderr
    return {
        "sample_count": sample_count,
        "mean_return_pct": round(mean, 6),
        "ci95_low": round(mean - margin, 6),
        "ci95_high": round(mean + margin, 6),
    }


def _window_exclusion_warnings(
    anchor: str,
    window: dict[str, Any],
    forward_days: tuple[int, ...],
) -> list[str]:
    warnings = [f"{anchor}:{warning}" for warning in window.get("warnings", [])]
    if window.get("status") == "warning":
        return warnings or [f"{anchor}:window_unavailable"]
    for day in forward_days:
        if window.get("forward", {}).get(str(day)) is None:
            warnings.append(f"{anchor}:missing_forward_{day}d")
    return warnings


def _transaction_direction(value: Any) -> float:
    normalized = str(value or "").lower()
    if normalized in {"purchase", "received"}:
        return 1.0
    if normalized in {"sale", "sale_partial"}:
        return -1.0
    return 0.0


def _filing_delay_days(trade: dict[str, Any]) -> int | None:
    delay = trade.get("delay_days")
    if isinstance(delay, int):
        return delay
    if isinstance(delay, float):
        return int(delay)
    transaction_date = trade.get("transaction_date")
    disclosure_date = trade.get("disclosure_date")
    if not transaction_date or not disclosure_date:
        return None
    try:
        return (
            date.fromisoformat(str(disclosure_date)) - date.fromisoformat(str(transaction_date))
        ).days
    except ValueError:
        return None


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[midpoint])
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0


def _edge_disappearance_stats(
    transaction_values: list[float],
    disclosure_values: list[float],
) -> dict[str, Any]:
    sample_count = min(len(transaction_values), len(disclosure_values))
    if sample_count == 0:
        return {
            "sample_count": 0,
            "transaction_anchor_mean_signed_return_pct": None,
            "disclosure_anchor_mean_signed_return_pct": None,
            "apparent_edge_delta_pct": None,
            "apparent_edge_disappeared_pct": None,
        }
    transaction_mean = sum(transaction_values[:sample_count]) / sample_count
    disclosure_mean = sum(disclosure_values[:sample_count]) / sample_count
    delta = transaction_mean - disclosure_mean
    disappeared = None if abs(transaction_mean) < 1e-12 else (delta / abs(transaction_mean)) * 100
    return {
        "sample_count": sample_count,
        "transaction_anchor_mean_signed_return_pct": round(transaction_mean, 6),
        "disclosure_anchor_mean_signed_return_pct": round(disclosure_mean, 6),
        "apparent_edge_delta_pct": round(delta, 6),
        "apparent_edge_disappeared_pct": None if disappeared is None else round(disappeared, 6),
    }


def _load_prices(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            date = row.get("Date") or row.get("date")
            close = row.get("Close") or row.get("close")
            if not date or close in (None, ""):
                continue
            try:
                rows.append({"date": date, "close": float(close)})
            except ValueError:
                continue
    rows.sort(key=lambda row: row["date"])
    return rows


def _first_index_on_or_after(prices: list[dict[str, Any]], target_date: str) -> int | None:
    for idx, row in enumerate(prices):
        if row["date"] >= target_date:
            return idx
    return None
