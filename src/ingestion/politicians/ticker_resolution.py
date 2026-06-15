"""Conservative asset-name to ticker resolution."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ingestion.politicians.paths import get_politicians_data_dir


UNMAPPED_BY_ASSET_TYPE = {
    "private_asset": "private_asset",
    "fund": "fund_unmapped",
    "bond": "bond_unmapped",
}


def resolve_ticker(
    *,
    asset_name: str | None,
    explicit_ticker: str | None = None,
    asset_type: str | None = None,
    aliases: dict[str, Any] | None = None,
    alias_path: str | Path | None = None,
    candidate_lookup: Callable[[str], list[str]] | None = None,
) -> dict[str, Any]:
    """Resolve ticker conservatively from explicit source value, alias, or candidates."""
    if explicit_ticker and explicit_ticker.strip():
        return {
            "ticker": explicit_ticker.strip().upper(),
            "ticker_resolution_status": "explicit",
            "ticker_resolution_reason": "source_ticker",
            "ticker_resolution_confidence": 1.0,
        }

    normalized_asset_type = (asset_type or "unknown").strip().lower()
    if normalized_asset_type in UNMAPPED_BY_ASSET_TYPE:
        return _unresolved(UNMAPPED_BY_ASSET_TYPE[normalized_asset_type])

    alias_map = aliases if aliases is not None else load_issuer_aliases(alias_path)
    alias = _find_alias(asset_name, alias_map)
    if alias:
        validate_manual_alias(alias)
        return {
            "ticker": alias["ticker"].upper(),
            "ticker_resolution_status": "alias",
            "ticker_resolution_reason": alias["reason"],
            "ticker_resolution_confidence": 0.95,
            "ticker_alias_metadata": {
                key: alias.get(key)
                for key in ("added_by", "added_at", "reason", "source_note")
                if alias.get(key) is not None
            },
        }

    if candidate_lookup and asset_name:
        candidates = sorted(set(candidate_lookup(_normalize_asset_name(asset_name))))
        if len(candidates) == 1:
            return {
                "ticker": candidates[0].upper(),
                "ticker_resolution_status": "candidate",
                "ticker_resolution_reason": "single_candidate",
                "ticker_resolution_confidence": 0.70,
            }
        if len(candidates) > 1:
            return {
                "ticker": None,
                "ticker_resolution_status": "ambiguous",
                "ticker_resolution_reason": "ambiguous",
                "ticker_resolution_confidence": 0.0,
                "ticker_candidates": candidates,
            }

    return _unresolved("unknown")


def load_issuer_aliases(alias_path: str | Path | None = None) -> dict[str, Any]:
    """Load issuer aliases from JSON, returning empty map when absent."""
    path = Path(alias_path) if alias_path else get_politicians_data_dir() / "issuer_aliases.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def make_manual_alias(
    *,
    ticker: str,
    added_by: str,
    reason: str,
    source_note: str | None = None,
) -> dict[str, str]:
    """Create a manual alias payload with required audit metadata."""
    payload = {
        "ticker": ticker.upper(),
        "added_by": added_by,
        "added_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "reason": reason,
    }
    if source_note:
        payload["source_note"] = source_note
    return payload


def validate_manual_alias(alias: dict[str, Any]) -> None:
    """Require audit metadata for manual aliases."""
    for field in ("ticker", "added_by", "added_at", "reason"):
        if not alias.get(field):
            raise ValueError(f"Manual ticker alias missing required field: {field}")


def _find_alias(asset_name: str | None, aliases: dict[str, Any]) -> dict[str, Any] | None:
    if not asset_name:
        return None
    normalized = _normalize_asset_name(asset_name)
    normalized_aliases = {
        _normalize_asset_name(key): value
        for key, value in aliases.items()
    }
    return normalized_aliases.get(normalized)


def _normalize_asset_name(value: str) -> str:
    text = value.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\b(class|cl)\s+([a-z])\b", r"class \2", text)
    text = re.sub(r"\b(incorporated|inc|corp|corporation|co|company|ltd|plc|trust|common stock|ordinary shares|adr|sponsored adr)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _unresolved(reason: str) -> dict[str, Any]:
    return {
        "ticker": None,
        "ticker_resolution_status": reason,
        "ticker_resolution_reason": reason,
        "ticker_resolution_confidence": 0.0,
    }
