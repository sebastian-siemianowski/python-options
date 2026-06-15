"""Tests for disclosure-date politician event windows."""

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.politicians.event_study import RETROSPECTIVE_ONLY_NOTE, build_disclosure_event_window


def _write_prices(path, symbol="NVDA", start=date(2026, 1, 1), days=140):
    prices = path / f"{symbol}_1d.csv"
    with prices.open("w", encoding="utf-8") as handle:
        handle.write("Date,Close\n")
        for idx in range(days):
            current = start + timedelta(days=idx)
            handle.write(f"{current.isoformat()},{100 + idx}\n")
    return prices


def test_event_window_is_keyed_by_disclosure_date_not_transaction_date(tmp_path):
    _write_prices(tmp_path)
    trade = {"ticker": "NVDA", "transaction_date": "2026-01-10", "disclosure_date": "2026-01-20"}

    window = build_disclosure_event_window(trade, prices_dir=tmp_path)

    assert window["event_time_anchor"] == "disclosure_date"
    assert window["anchor"]["date"] == "2026-01-20"
    assert window["transaction_date"] == "2026-01-10"


def test_event_window_includes_prior_5_and_forward_1_7_30_90_trading_days(tmp_path):
    _write_prices(tmp_path)
    trade = {"ticker": "NVDA", "disclosure_date": "2026-01-20"}

    window = build_disclosure_event_window(trade, prices_dir=tmp_path)

    assert len(window["prior"]) == 5
    assert set(window["forward"].keys()) == {"1", "7", "30", "90"}
    assert window["forward"]["1"]["date"] == "2026-01-21"
    assert window["forward"]["90"]["date"] == "2026-04-20"


def test_event_window_missing_price_data_returns_structured_warning(tmp_path):
    trade = {"ticker": "NVDA", "disclosure_date": "2026-01-20"}

    window = build_disclosure_event_window(trade, prices_dir=tmp_path)

    assert window["status"] == "warning"
    assert window["warnings"] == ["missing_price_data:NVDA"]
    assert window["anchor"] is None


def test_transaction_date_analysis_is_labelled_retrospective_only(tmp_path):
    _write_prices(tmp_path)
    trade = {"ticker": "NVDA", "transaction_date": "2026-01-10", "disclosure_date": "2026-01-20"}

    window = build_disclosure_event_window(trade, prices_dir=tmp_path)

    assert "RETROSPECTIVE_ONLY" in window["retrospective_only_note"]
    assert window["retrospective_only_note"] == RETROSPECTIVE_ONLY_NOTE
