"""Tests for conservative politician ticker resolution."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.politicians.ticker_resolution import (
    load_issuer_aliases,
    make_manual_alias,
    resolve_ticker,
    validate_manual_alias,
)


def test_resolver_trusts_explicit_source_ticker_first():
    result = resolve_ticker(asset_name="Some Ambiguous Name", explicit_ticker="nvda", asset_type="stock")

    assert result["ticker"] == "NVDA"
    assert result["ticker_resolution_status"] == "explicit"
    assert result["ticker_resolution_confidence"] == 1.0


def test_resolver_uses_manual_aliases_with_audit_metadata(tmp_path):
    alias = make_manual_alias(ticker="NVDA", added_by="test", reason="official issuer alias", source_note="fixture")
    alias_path = tmp_path / "issuer_aliases.json"
    alias_path.write_text(json.dumps({"NVIDIA Corporation": alias}), encoding="utf-8")

    result = resolve_ticker(asset_name="NVIDIA Corp. Common Stock", asset_type="stock", alias_path=alias_path)

    assert load_issuer_aliases(alias_path)["NVIDIA Corporation"]["ticker"] == "NVDA"
    assert result["ticker"] == "NVDA"
    assert result["ticker_resolution_status"] == "alias"
    assert result["ticker_alias_metadata"]["added_by"] == "test"
    assert result["ticker_alias_metadata"]["source_note"] == "fixture"


def test_manual_alias_requires_metadata():
    with pytest.raises(ValueError, match="added_by"):
        validate_manual_alias({"ticker": "NVDA", "added_at": "2026-01-01", "reason": "missing"})


def test_private_fund_and_bond_assets_are_marked_unmapped():
    assert resolve_ticker(asset_name="Private LLC", asset_type="private_asset")["ticker_resolution_reason"] == "private_asset"
    assert resolve_ticker(asset_name="Mutual Fund", asset_type="fund")["ticker_resolution_reason"] == "fund_unmapped"
    assert resolve_ticker(asset_name="Treasury Bond", asset_type="bond")["ticker_resolution_reason"] == "bond_unmapped"


def test_resolver_never_guesses_when_multiple_public_candidates_match():
    result = resolve_ticker(
        asset_name="Acme Corp",
        asset_type="stock",
        candidate_lookup=lambda name: ["ACME", "ACMEB"],
    )

    assert result["ticker"] is None
    assert result["ticker_resolution_status"] == "ambiguous"
    assert result["ticker_candidates"] == ["ACME", "ACMEB"]


def test_suffixes_share_classes_etfs_and_adrs_resolve_by_alias():
    aliases = {
        "Berkshire Hathaway Class B": make_manual_alias(ticker="BRK.B", added_by="test", reason="share class"),
        "SPDR S&P 500 ETF": make_manual_alias(ticker="SPY", added_by="test", reason="etf alias"),
        "Alibaba Group Holding": make_manual_alias(ticker="BABA", added_by="test", reason="adr alias"),
    }

    brk = resolve_ticker(asset_name="Berkshire Hathaway Inc. Class B Common Stock", asset_type="stock", aliases=aliases)
    spy = resolve_ticker(asset_name="SPDR S&P 500 ETF Trust", asset_type="etf", aliases=aliases)
    baba = resolve_ticker(asset_name="Alibaba Group Holding Sponsored ADR", asset_type="stock", aliases=aliases)

    assert brk["ticker"] == "BRK.B"
    assert spy["ticker"] == "SPY"
    assert baba["ticker"] == "BABA"
