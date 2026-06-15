"""Disclosure-date event-study report tests."""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.politicians.event_study import (
    build_disclosure_event_window,
    compute_disclosure_event_study,
    compute_transaction_date_retrospective_analysis,
)
from research.politicians.report import event_study_cli_main


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_event_study_supports_standard_forward_horizons(tmp_path):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_prices(prices_dir / "NVDA_1d.csv", "2026-01-01", 120)

    report = compute_disclosure_event_study([_trade("NVDA")], prices_dir=prices_dir, min_sample_size=1)

    assert report["event_time_anchor"] == "disclosure_date"
    assert report["forward_trading_days"] == [1, 7, 30, 90]
    horizons = report["groups"][0]["horizons"]
    assert set(horizons) == {"1", "7", "30", "90"}
    assert all(horizons[key]["sample_count"] == 1 for key in horizons)


def test_event_study_groups_by_trade_attributes_and_confidence_bucket(tmp_path):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_prices(prices_dir / "NVDA_1d.csv", "2026-01-01", 120)

    report = compute_disclosure_event_study([
        _trade("NVDA", transaction_type="purchase", chamber="house", asset_type="stock", amount_min=1001, amount_max=15000, confidence=0.96),
        _trade("NVDA", transaction_type="purchase", chamber="house", asset_type="stock", amount_min=1001, amount_max=15000, confidence=0.96),
    ], prices_dir=prices_dir, min_sample_size=1)

    group = report["groups"][0]
    assert group["group"] == {
        "transaction_type": "purchase",
        "chamber": "house",
        "asset_type": "stock",
        "amount_bucket": "1001-15000",
        "parser_confidence_bucket": "high",
    }
    assert group["sample_count"] == 2
    assert group["horizons"]["1"]["ci95_low"] <= group["horizons"]["1"]["mean_return_pct"] <= group["horizons"]["1"]["ci95_high"]


def test_event_study_excludes_missing_or_stale_price_data(tmp_path):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_prices(prices_dir / "NVDA_1d.csv", "2026-01-01", 120)
    _write_prices(prices_dir / "AAPL_1d.csv", "2026-01-01", 20)

    report = compute_disclosure_event_study([
        _trade("NVDA"),
        _trade("AAPL"),
        _trade("MSFT"),
    ], prices_dir=prices_dir, min_sample_size=1)

    assert report["included_count"] == 1
    assert report["excluded_count"] == 2
    warnings = [warning for event in report["excluded_events"] for warning in event["warnings"]]
    assert "missing_forward_90d" in warnings
    assert "missing_price_data:MSFT" in warnings


def test_event_study_command_prints_small_sample_warning(tmp_path, capsys):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_prices(prices_dir / "NVDA_1d.csv", "2026-01-01", 120)
    trades_path = tmp_path / "trades.jsonl"
    trades_path.write_text(json.dumps(_trade("NVDA")) + "\n", encoding="utf-8")

    exit_code = event_study_cli_main([
        "--trades", str(trades_path),
        "--prices-dir", str(prices_dir),
        "--min-sample-size", "5",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "WARNING sample_size_too_small" in output
    assert '"event_time_anchor": "disclosure_date"' in output


def test_transaction_date_retrospective_report_is_labeled_and_quantifies_edge(tmp_path):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_prices(prices_dir / "NVDA_1d.csv", "2026-01-01", 180)
    trades = [
        _trade("NVDA", transaction_date="2026-01-10", disclosure_date="2026-01-20"),
        _trade("NVDA", transaction_date="2026-01-12", disclosure_date="2026-01-24"),
    ]

    report = compute_transaction_date_retrospective_analysis(
        trades,
        prices_dir=prices_dir,
        min_sample_size=1,
    )

    assert report["report_label"] == "RETROSPECTIVE_ONLY"
    assert report["event_time_anchor"] == "transaction_date"
    assert report["comparison_anchor"] == "disclosure_date"
    delays = {
        (row["chamber"], row["transaction_type"]): row
        for row in report["median_filing_delay_days"]
    }
    assert delays[("house", "purchase")]["median_delay_days"] == 11.0
    edge = report["edge_disappearance"]["7"]
    assert edge["sample_count"] == 2
    assert edge["transaction_anchor_mean_signed_return_pct"] > edge["disclosure_anchor_mean_signed_return_pct"]
    assert edge["apparent_edge_disappeared_pct"] > 0


def test_retrospective_command_prints_clear_research_only_label(tmp_path, capsys):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_prices(prices_dir / "NVDA_1d.csv", "2026-01-01", 180)
    trades_path = tmp_path / "trades.jsonl"
    trades_path.write_text(json.dumps(_trade("NVDA")) + "\n", encoding="utf-8")

    exit_code = event_study_cli_main([
        "--trades", str(trades_path),
        "--prices-dir", str(prices_dir),
        "--min-sample-size", "1",
        "--retrospective-transaction-date",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"report_label": "RETROSPECTIVE_ONLY"' in output
    assert '"edge_disappearance"' in output


def test_production_event_study_helpers_default_to_disclosure_date(tmp_path):
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    _write_prices(prices_dir / "NVDA_1d.csv", "2026-01-01", 180)
    trade = _trade("NVDA", transaction_date="2026-01-05", disclosure_date="2026-01-20")

    window = build_disclosure_event_window(trade, prices_dir=prices_dir)
    report = compute_disclosure_event_study([trade], prices_dir=prices_dir, min_sample_size=1)

    assert window["event_time_anchor"] == "disclosure_date"
    assert window["anchor"]["date"] == "2026-01-20"
    assert report["event_time_anchor"] == "disclosure_date"


def test_transaction_date_return_helpers_are_not_imported_by_production_paths():
    forbidden = (
        "compute_transaction_date_retrospective_analysis",
        "build_transaction_retrospective_event_window",
        "transaction_anchor_mean_signed_return_pct",
    )
    production_roots = (
        REPO_ROOT / "src" / "decision",
        REPO_ROOT / "src" / "web" / "backend",
        REPO_ROOT / "src" / "web" / "frontend" / "src",
    )
    offenders = []
    for root in production_roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".ts", ".tsx"}:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                if marker in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{marker}")

    assert offenders == []


def _trade(
    ticker,
    *,
    transaction_type="purchase",
    chamber="house",
    asset_type="stock",
    amount_min=1001,
    amount_max=15000,
    confidence=0.96,
    transaction_date="2026-01-01",
    disclosure_date="2026-01-10",
):
    return {
        "trade_id": f"{ticker}-{transaction_type}-{transaction_date}",
        "ticker": ticker,
        "transaction_type": transaction_type,
        "chamber": chamber,
        "asset_type": asset_type,
        "amount_min_usd": amount_min,
        "amount_max_usd": amount_max,
        "parser_confidence": confidence,
        "disclosure_date": disclosure_date,
        "transaction_date": transaction_date,
    }


def _write_prices(path, start, count):
    start_date = date.fromisoformat(start)
    rows = ["Date,Close\n"]
    for idx in range(count):
        rows.append(f"{(start_date + timedelta(days=idx)).isoformat()},{100 + idx}\n")
    path.write_text("".join(rows), encoding="utf-8")
