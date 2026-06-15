"""
Compliance and data-use guardrails for politician disclosure monitoring.

Congressional financial disclosures are public records, but they are delayed,
range-based, and subject to official source terms. Keep this module small and
dependency-free so ingestion, API, and tests can import it safely.
"""

from __future__ import annotations

import os
from typing import Final, TypedDict


class DataUsePolicy(TypedDict):
    """Machine-readable policy copy for API and UI surfaces."""

    title: str
    summary: str
    bullets: list[str]
    prohibited_labels: list[str]
    official_sources: list[str]
    reviewed_at: str


DATA_USE_NOTICE: Final[str] = (
    "Politician trade records are delayed, range-based public disclosures from "
    "official sources. They are research context only, not real-time trade "
    "confirmations, investment advice, copy-trade instructions, credit-rating "
    "inputs, solicitation material, or an insider-trading signal."
)

DATA_USE_POLICY: Final[DataUsePolicy] = {
    "title": "Politician Disclosure Data Use Policy",
    "summary": DATA_USE_NOTICE,
    "bullets": [
        (
            "Public filings are delayed disclosures, not real-time trade "
            "confirmations or execution feeds."
        ),
        (
            "Amounts are reported in broad ranges and must be displayed as "
            "estimates or buckets, never as exact position sizes."
        ),
        (
            "The data must not be used for credit rating, unlawful purposes, "
            "solicitation, or any deployment prohibited by official source terms."
        ),
        (
            "The product must not label these records as insider-trading signals "
            "or invite users to copy a politician's trades."
        ),
        (
            "Research and backtests must use disclosure_date or filed_date as "
            "the knowable-time anchor to avoid lookahead leakage."
        ),
        (
            "Every displayed record must preserve source attribution, official "
            "document links, parser confidence, and delayed-data context."
        ),
    ],
    "prohibited_labels": [
        "copy trade",
        "follow this trade",
        "guaranteed edge",
        "insider-trading signal",
        "real-time politician trades",
    ],
    "official_sources": [
        "House Clerk Financial Disclosure Reports",
        "House Clerk Financial Disclosure Search",
        "Senate eFD public search",
        "Senate Select Committee on Ethics public disclosure materials",
    ],
    "reviewed_at": "2026-05-29",
}

VALID_COMPLIANCE_MODES: Final[tuple[str, ...]] = ("research_only", "internal", "public")
DEFAULT_COMPLIANCE_MODE: Final[str] = "research_only"


def is_politicians_enabled(raw_value: str | None = None) -> bool:
    """Return whether politician monitoring is enabled by environment flag."""
    value = os.getenv("POLITICIANS_ENABLED", "1") if raw_value is None else raw_value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_compliance_mode(raw_value: str | None = None) -> dict[str, object]:
    """Return the normalized compliance mode plus validation metadata."""
    requested = (
        os.getenv("POLITICIANS_COMPLIANCE_MODE", DEFAULT_COMPLIANCE_MODE)
        if raw_value is None
        else raw_value
    )
    normalized = str(requested).strip().lower()
    if normalized in VALID_COMPLIANCE_MODES:
        return {
            "compliance_mode": normalized,
            "requested_compliance_mode": normalized,
            "compliance_mode_valid": True,
            "valid_compliance_modes": list(VALID_COMPLIANCE_MODES),
        }
    return {
        "compliance_mode": DEFAULT_COMPLIANCE_MODE,
        "requested_compliance_mode": normalized,
        "compliance_mode_valid": False,
        "valid_compliance_modes": list(VALID_COMPLIANCE_MODES),
        "warning": (
            f"Unknown POLITICIANS_COMPLIANCE_MODE={requested!r}; "
            f"falling back to {DEFAULT_COMPLIANCE_MODE!r}."
        ),
    }


def get_feature_availability() -> dict[str, object]:
    """Return feature availability and compliance-mode metadata."""
    mode = get_compliance_mode()
    enabled = is_politicians_enabled()
    return {
        "enabled": enabled,
        "status": "available" if enabled else "disabled",
        "disabled_reason": None if enabled else "POLITICIANS_ENABLED=0",
        **mode,
    }


def get_data_use_notice() -> dict[str, object]:
    """Return a JSON-serializable policy payload for API responses."""
    return {
        "title": DATA_USE_POLICY["title"],
        "summary": DATA_USE_POLICY["summary"],
        "bullets": list(DATA_USE_POLICY["bullets"]),
        "official_sources": list(DATA_USE_POLICY["official_sources"]),
        "reviewed_at": DATA_USE_POLICY["reviewed_at"],
    }
