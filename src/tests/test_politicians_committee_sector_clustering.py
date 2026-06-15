"""Tests for committee-sector clustering enrichment reports."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.politicians.event_study import compute_committee_sector_clustering_report
from research.politicians.report import event_study_cli_main


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_clustering_maps_tickers_to_sectors_and_members_to_committees():
    trades = [
        _trade("jane-doe", "Jane Doe", "NVDA", amount_mid=10_000),
        _trade("jane-doe", "Jane Doe", "MSFT", amount_mid=5_000),
    ]
    members = {
        "jane-doe": {
            "filer_name": "Jane Doe",
            "committees": ["Financial Services"],
        },
    }
    sector_lookup = {
        "NVDA": "Technology",
        "MSFT": "Technology",
    }

    report = compute_committee_sector_clustering_report(
        trades,
        members=members,
        sector_lookup=sector_lookup,
    )

    assert report["data_classification"] == "enrichment_not_source_disclosure"
    assert "not a disclosure source field" in report["committee_data_label"]
    assert "existing signal-engine sector mappings" in report["sector_data_label"]
    assert report["unknown_committee_count"] == 0
    assert report["unknown_sector_count"] == 0
    assert report["source_counts"]["committee"] == {"official_member_metadata": 2}
    assert report["source_counts"]["sector"] == {"existing_sector_mapping": 2}
    assert report["heatmap"][0] == {
        "committee": "Financial Services",
        "sector": "Technology",
        "trade_count": 2,
        "amount_mid_usd": 15_000.0,
        "amount_min_usd": 0.0,
        "amount_max_usd": 0.0,
        "committee_data_source": "official_member_metadata",
        "sector_data_source": "existing_sector_mapping",
        "data_classification": "enrichment_not_source_disclosure",
    }


def test_clustering_keeps_missing_committee_and_sector_unknown_not_inferred():
    report = compute_committee_sector_clustering_report([
        _trade("unknown-filer", "Unknown Filer", "NOTREAL", amount_mid=1_000),
    ], sector_lookup={})

    assert report["unknown_committee_count"] == 1
    assert report["unknown_sector_count"] == 1
    assert report["heatmap"][0]["committee"] == "unknown"
    assert report["heatmap"][0]["sector"] == "unknown"
    assert report["heatmap"][0]["committee_data_source"] == "unknown"
    assert report["heatmap"][0]["sector_data_source"] == "unknown"


def test_clustering_cli_loads_member_metadata_and_prints_heatmap(tmp_path, capsys):
    trades_path = tmp_path / "trades.jsonl"
    members_path = tmp_path / "members.json"
    trades_path.write_text(json.dumps(_trade("jane-doe", "Jane Doe", "NVDA", amount_mid=8_000)) + "\n", encoding="utf-8")
    members_path.write_text(json.dumps({"jane-doe": {"committees": ["Armed Services"]}}), encoding="utf-8")

    exit_code = event_study_cli_main([
        "--trades", str(trades_path),
        "--members", str(members_path),
        "--committee-sector-clustering",
    ])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert '"report_type": "committee_sector_clustering"' in output
    assert '"committee": "Armed Services"' in output


def test_ui_labels_committee_data_as_enrichment_not_source_field():
    drawer = REPO_ROOT / "src" / "web" / "frontend" / "src" / "features" / "politicians" / "components" / "FilerDrawer.tsx"
    text = drawer.read_text(encoding="utf-8")

    assert "Committee Enrichment" in text
    assert "not a disclosure source field" in text
    assert "Top Sectors (Enrichment)" in text


def _trade(filer_id, filer_name, ticker, *, amount_mid):
    return {
        "trade_id": f"{filer_id}-{ticker}",
        "filer_id": filer_id,
        "filer_name": filer_name,
        "ticker": ticker,
        "transaction_type": "purchase",
        "chamber": "house",
        "amount_mid_usd": amount_mid,
    }
