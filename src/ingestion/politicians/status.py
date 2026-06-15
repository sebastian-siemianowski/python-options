"""Status reporting for politician disclosure ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ingestion.politicians.paths import ensure_politicians_data_dirs
from ingestion.politicians.source_health import read_source_health
from ingestion.politicians.sync import read_sync_state


def get_politicians_status(data_root: str | Path | None = None) -> dict[str, Any]:
    """Return source health, counts, parse errors, and newest disclosure."""
    root = ensure_politicians_data_dirs(data_root)
    filings = _read_jsonl(root / "filings.jsonl")
    trades = _read_jsonl(root / "trades.jsonl")
    parse_errors = _read_jsonl(root / "parse_errors.jsonl")
    newest = _newest_date(row.get("filed_date") for row in filings)
    return {
        "status": "ok",
        "source_health": read_source_health(root),
        "sync_state": read_sync_state(root),
        "filing_count": len(filings),
        "trade_count": len(trades),
        "parse_error_count": len(parse_errors),
        "validation_summary": _validation_summary(parse_errors),
        "newest_disclosure": newest,
    }


def render_politicians_status(status: dict[str, Any], *, console=None) -> None:
    """Render a Rich status table."""
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    out = console or Console()
    table = Table(title="Politician Disclosure Status")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Filings", str(status["filing_count"]))
    table.add_row("Trades", str(status["trade_count"]))
    table.add_row("Parse errors", str(status["parse_error_count"]))
    for error, count in status.get("validation_summary", {}).items():
        table.add_row(f"Validation {error}", str(count))
    table.add_row("Newest disclosure", str(status["newest_disclosure"] or "-"))
    for source, entry in status.get("source_health", {}).get("sources", {}).items():
        table.add_row(f"{source} health", str(entry.get("status", "unknown")))
    out.print(table)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"parse_error": line.strip()})
    return rows


def _newest_date(values) -> str | None:
    dates = sorted(str(value) for value in values if value)
    return dates[-1] if dates else None


def _validation_summary(parse_errors: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in parse_errors:
        for error in row.get("errors", []):
            summary[error] = summary.get(error, 0) + 1
    return dict(sorted(summary.items()))
