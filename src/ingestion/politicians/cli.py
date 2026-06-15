"""CLI entrypoint for politician disclosure ingestion."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from ingestion.politicians.sources.house import download_house_ptr_pdfs, download_house_yearly_archive
from ingestion.politicians.sources.senate import search_senate_ptr_filings
from ingestion.politicians.daily import run_daily_politicians_sync
from ingestion.politicians.sync import run_incremental_sync
from ingestion.politicians.parse import parse_local_raw_artifacts
from ingestion.politicians.status import get_politicians_status, render_politicians_status


def build_parser() -> argparse.ArgumentParser:
    """Build the politician ingestion CLI parser."""
    parser = argparse.ArgumentParser(prog="python -m ingestion.politicians.cli")
    subparsers = parser.add_subparsers(dest="source", required=True)

    house = subparsers.add_parser("house", help="House Clerk disclosure source")
    house_sub = house.add_subparsers(dest="command", required=True)
    backfill = house_sub.add_parser("backfill", help="Download House yearly archive")
    backfill.add_argument("--year", type=int, required=True)
    backfill.add_argument("--archive-url", action="append", default=None)
    backfill.add_argument("--index-url", default=None)
    fetch_pdfs = house_sub.add_parser("fetch-pdfs", help="Download House PTR PDFs from manifest rows")
    fetch_pdfs.add_argument("--year", type=int, required=True)

    senate = subparsers.add_parser("senate", help="Senate eFD disclosure source")
    senate_sub = senate.add_subparsers(dest="command", required=True)
    senate_search = senate_sub.add_parser("search", help="Search Senate PTR filings by filing date window")
    senate_search.add_argument("--from", dest="date_from", required=True)
    senate_search.add_argument("--to", dest="date_to", required=True)

    sync = subparsers.add_parser("sync", help="Run politician disclosure sync")
    sync_sub = sync.add_subparsers(dest="command", required=True)
    incremental = sync_sub.add_parser("incremental", help="Run default incremental sync")
    incremental.add_argument("--from", dest="date_from", default=None)
    incremental.add_argument("--to", dest="date_to", default=None)
    daily = sync_sub.add_parser("daily", help="Run daily sync, parse, validation, and cache refresh task")
    daily.add_argument("--from", dest="date_from", default=None)
    daily.add_argument("--to", dest="date_to", default=None)

    parse = subparsers.add_parser("parse", help="Parse local politician raw artifacts")
    parse_sub = parse.add_subparsers(dest="command", required=True)
    parse_sub.add_parser("local", help="Inventory local raw artifacts into JSONL")

    subparsers.add_parser("status", help="Show politician disclosure ingestion status")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.source == "house" and args.command == "backfill":
        kwargs = {"archive_urls": args.archive_url}
        if args.index_url:
            kwargs["index_url"] = args.index_url
        manifest = download_house_yearly_archive(args.year, **kwargs)
        print(json.dumps({
            "source": manifest["source"],
            "year": manifest["year"],
            "status": manifest["status"],
            "artifact_count": manifest["artifact_count"],
            "errors": manifest["errors"],
        }, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "ok" else 1

    if args.source == "house" and args.command == "fetch-pdfs":
        manifest = download_house_ptr_pdfs(args.year)
        print(json.dumps({
            "source": manifest["source"],
            "year": manifest["year"],
            "status": manifest["status"],
            "filing_count": manifest["filing_count"],
            "pdf_count": manifest["pdf_count"],
            "missing_artifact_count": manifest["missing_artifact_count"],
            "errors": manifest["errors"],
        }, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "ok" else 1

    if args.source == "senate" and args.command == "search":
        manifest = search_senate_ptr_filings(date_from=args.date_from, date_to=args.date_to)
        print(json.dumps({
            "source": manifest["source"],
            "year": manifest["year"],
            "status": manifest["status"],
            "filing_count": manifest["filing_count"],
            "errors": manifest["errors"],
        }, indent=2, sort_keys=True))
        return 0 if manifest["status"] == "ok" else 1

    if args.source == "sync" and args.command == "incremental":
        result = run_incremental_sync(date_from=args.date_from, date_to=args.date_to)
        statuses = [row["status"] for row in result["results"]]
        print(json.dumps({
            "date_window": result["date_window"],
            "status": "ok" if all(status == "ok" for status in statuses) else "degraded",
            "results": result["results"],
        }, indent=2, sort_keys=True))
        return 0 if all(status == "ok" for status in statuses) else 1

    if args.source == "sync" and args.command == "daily":
        result = run_daily_politicians_sync(date_from=args.date_from, date_to=args.date_to)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ok" else 1

    if args.source == "parse" and args.command == "local":
        result = parse_local_raw_artifacts()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "ok" else 1

    if args.source == "status":
        status = get_politicians_status()
        render_politicians_status(status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    parser.error("Unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
