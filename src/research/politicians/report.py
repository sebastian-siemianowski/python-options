"""Research report commands for politician disclosure analytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.politicians.event_study import (
    DEFAULT_FORWARD_TRADING_DAYS,
    compute_committee_sector_clustering_report,
    compute_disclosure_event_study,
    compute_transaction_date_retrospective_analysis,
)


def run_disclosure_event_study_report(
    *,
    trades_path: str | Path,
    prices_dir: str | Path,
    min_sample_size: int = 5,
) -> dict[str, Any]:
    """Load normalized trades and return a disclosure-date event-study report."""
    trades = _read_jsonl(Path(trades_path))
    return compute_disclosure_event_study(
        trades,
        prices_dir=prices_dir,
        forward_days=DEFAULT_FORWARD_TRADING_DAYS,
        min_sample_size=min_sample_size,
    )


def run_transaction_date_retrospective_report(
    *,
    trades_path: str | Path,
    prices_dir: str | Path,
    min_sample_size: int = 5,
) -> dict[str, Any]:
    """Load normalized trades and return a RETROSPECTIVE_ONLY transaction-date report."""
    trades = _read_jsonl(Path(trades_path))
    return compute_transaction_date_retrospective_analysis(
        trades,
        prices_dir=prices_dir,
        forward_days=DEFAULT_FORWARD_TRADING_DAYS,
        min_sample_size=min_sample_size,
    )


def run_committee_sector_clustering_report(
    *,
    trades_path: str | Path,
    members_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load normalized trades and return a committee-sector enrichment heatmap."""
    trades = _read_jsonl(Path(trades_path))
    members = _read_json_or_jsonl(Path(members_path)) if members_path else None
    return compute_committee_sector_clustering_report(trades, members=members)


def print_disclosure_event_study_report(report: dict[str, Any]) -> None:
    """Print report JSON plus clear sample-size warnings."""
    print(json.dumps(report, indent=2, sort_keys=True))
    for warning in report.get("warnings", []):
        group = warning.get("group", {})
        print(
            "WARNING sample_size_too_small "
            f"n={warning.get('sample_count')} min={warning.get('min_sample_size')} "
            f"group={group}"
        )


def event_study_cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Disclosure-date politician event study")
    parser.add_argument("--trades", required=True, help="Path to normalized trades.jsonl")
    parser.add_argument("--prices-dir", help="Directory containing SYMBOL_1d.csv files")
    parser.add_argument("--min-sample-size", type=int, default=5)
    parser.add_argument("--members", help="Optional members.json or JSONL metadata for committee enrichment")
    parser.add_argument(
        "--retrospective-transaction-date",
        action="store_true",
        help="Run RETROSPECTIVE_ONLY transaction-date comparison against disclosure-date returns",
    )
    parser.add_argument(
        "--committee-sector-clustering",
        action="store_true",
        help="Run committee-sector clustering enrichment report",
    )
    args = parser.parse_args(argv)
    if args.committee_sector_clustering:
        report = run_committee_sector_clustering_report(
            trades_path=args.trades,
            members_path=args.members,
        )
    elif args.retrospective_transaction_date:
        if not args.prices_dir:
            parser.error("--prices-dir is required unless --committee-sector-clustering is used")
        report = run_transaction_date_retrospective_report(
            trades_path=args.trades,
            prices_dir=args.prices_dir,
            min_sample_size=args.min_sample_size,
        )
    else:
        if not args.prices_dir:
            parser.error("--prices-dir is required unless --committee-sector-clustering is used")
        report = run_disclosure_event_study_report(
            trades_path=args.trades,
            prices_dir=args.prices_dir,
            min_sample_size=args.min_sample_size,
        )
    print_disclosure_event_study_report(report)
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json_or_jsonl(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    raise SystemExit(event_study_cli_main())
